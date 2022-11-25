import os
import itertools
import json

from typing import *

from tqdm import tqdm

import rasterio as rio
import numpy as np

from torch.utils.data.dataset import Dataset

class DatasetScanner():
    """
    Legge dalla master folder le cartelle all'interno delle quali ci sono pre, post e mask.
    Successivamente genera un csv con tutte le informazioni relative alla singola attivazione.
    In questo modo evitiamo di mantenere in memoria a runtime 520 giga di informazioni sul path assoluto
    E processiamo riga per riga il csv risultate. (generato da log_to_file()).
    """
    def __init__(self, master_folder_path=None, log_file_path=None, validation_file_path=None, valid_list=None, ignore_list=None, verbose=0):
        self.master_folder_path=master_folder_path
        self.log_file_path=log_file_path
        self.validation_file_path=validation_file_path
        self.verbose=verbose
        self.valid_list=valid_list
        self.ignore_list=ignore_list

    def scan_master_folder(self) -> List[str]:
        return os.listdir(self.master_folder_path)

    def trim_master_folder(self) -> Tuple(List[str], List[str]):
        try:
            with open(f"{self.validation_file_path}") as val_file:
                self.valid_list = json.load(val_file)
                for path in self.valid_list:
                    if os.isdir(path):
                        pass
                    else:
                        self.ignore_list.append(path)
        except:
            print(f"Error reading validation file: {self.validation_file_path}")
        finally:
            return self.valid_list, self.ignore_list

    def log_header_to_file(self, header="act_id") -> bool:
        """
        csv_header = act_id
        """
        try:
            with open(f"{self.log_file_path}", "w") as log_file:
                if self.verbose==1:
                    print(f"Writing header into log file: {self.log_file_path}")
                log_file.write(header + "\n")
        except:
            print(f"Error writing header into log file: {self.log_file_path}")
        finally:
            return True

    def log_to_file(self) -> bool:
        try:
            if not self.valid_list:
                with open(f"{self.log_file_path}", "a") as log_file:
                    for path in self.scan_master_folder().filter(lambda folder : os.isdir(folder)):
                        if self.verbose==1:
                            print(f"Writing {path} record into log file: {self.log_file_path}")
                    log_file.write(f"{path}")
            else:
                with open(f"{self.log_file_path}", "a") as log_file:
                    for path in self.valid_list:
                        if self.verbose==1:
                            print(f"Writing {path} record into log file: {self.log_file_path}")
                    log_file.write(f"{path}")
        except:
            print(f"Error writing act_id info into log file: {self.log_file_path}")
        finally:
            return True

class DatasetFormatter():
    """
    Questa classe effettua un check sulla dimensione di ogni immagine in input (presa da validated_list di DataScanner). 
        - Se la dimensione è giusta -> tiling
        - Se la dimensione è sbagliata -> padding/overlapping -> tiling
        - Ogni immagine viene poi salvata in una directory con la seguente struttura:
            - act_id
                - pre:
                    - act_tiles
                - post:
                    - act_tiles
                - mask:
                    - mask_tiles
    """
    def __init__(self, master_folder_path=None, log_file_path=None, log_filename=None, master_dict_path=None, valid_list=None, tile_height=512, tile_width=512, thr_pixels = 112, use_pre=False, verbose=0):
        self.master_folder_path=master_folder_path
        self.log_file_path=log_file_path
        self.log_filename=log_filename
        self.master_dict_path=master_dict_path
        self.valid_list=valid_list
        self.tile_height=tile_height
        self.tile_width=tile_width
        self.thr_pixels = thr_pixels
        self.use_pre=use_pre
        self.verbose=verbose

        self.master_dict_info = {"processing_info": []}

    def log_dict_info(self,
    act_id : str,
    idx : int,
    tile_info : List[str],
    subitem : str
     ) -> Dict:
        logging_info = {
            f"{act_id}_{idx}" : {
                f"tile_info_{subitem}" : tile_info
            }
        }
        self.master_dict_info["processing_info"].append(logging_info)
        return self.master_dict_info

    ## Context:  tutto questo viene fatto mentre fai il tiling
    # controlla il numero di dimensioni delle immagini
    # se non ce ne sono abbastanza aumenta gli assi
    # controlla i valori delle dimensioni,
    # int(ratio dell'altezza per 512) = numero di tile che ci vengono interi
    # ratio della larghezza per 512 = numero di tile che ci vengono interi
    # dato un threshold,
    # se il la tile ha larghezza o lunghezza minore del threshold
    # discarda la tile 
    # altrimenti
    # calcola quanti pixel mancano dall'avere l'immagine delle dimensioni desiderate (overlapping)
    # e per capire da che pixel partire per completare l'ultima tile 
    # fine dell'ultima tile intera - i pixel di overlapping. 

    
    def offset_tiles(self, img: np.ndarrary, tile_dim= [512, 512]) -> Tuple[List[float], List[float]]:
        """
        Suppose to have arrays formatted as channel first
        Questa funzione controlla le dimensioni dell'immagine data in ingresso, nel caso in cui il numero della dimensione
        non fosse quello desiderarto aggiungine una (questo è per le maschere che saranno in bianco e nero, un canale). 
        Una volta aggiustato il numero di dimensioni, fai il check sui valori delle dimensioni e nel caso in cui non 
        siano divisibili per 512 fai l'overlapping dell'ultima tile che rimane con dimensioni non giuste con la penultima tile
        img: ndarray
        thr: minimum pixels are needed to create another tile
        """
        if len(img.shape) == 2:
            axis = 0
            img = np.expand_dims(img, axis = axis)
            # da aggiungere se vogliamo i channel alla fine.
            #img = np.moveaxis(img, 0, -1)
        # cambiare questo se non è channel first
        _, heigth, width = img.shape
        tile_h, tile_w = tile_dim

        # check for the heigth
        # dato che abbiamo immagini minimo di 512 x 512 non ci preoccupiamo di fare un upsampling
        if heigth % tile_h == 0:
            num_tile_h = int(heigth/tile_h)
            offsets_h = [i*tile_h for i in range(num_tile_h)]
        elif heigth % tile_h != 0:
            # controlla prima il resto quant'è remaining = resto
            remaining_h = heigth % tile_h
            if remaining_h > self.thr_pixels:
                num_tile_h = int(np.floor(heigth/tile_h)) + 1 
            else:
                num_tile_h = int(np.floor(heigth/tile_h))
            offsets_h = [i*tile_h for i in range(num_tile_h-1)]
            offsets_h.append(offsets_h[-1] - (tile_h - remaining_h))
        
        # check for the width
        # dato che abbiamo immagini minimo di 512 x 512 non ci preoccupiamo di fare un upsampling
        if width % tile_w == 0:
            num_tile_w = int(width/tile_w)
            offsets_w = [i*tile_w for i in range(num_tile_w)]
        elif width % tile_w != 0:
            # controlla prima il resto quant'è remaining = resto
            remaining_w = width % tile_w
            if remaining_w > self.thr_pixels:
                num_tile_w = int(np.floor(width/tile_w)) + 1
            else:
                num_tile_w = int(np.floor(width/tile_w))
            offsets_w = [i*tile_w for i in range(num_tile_w-1)]
            offsets_w.append(offsets_w[-1] - (tile_w - remaining_w))

        return offsets_w, offsets_h

    def get_tiles(self, ds):
        # shape = (c, h, w) questa è l'immagine intera
        img_as_np = ds.read()
        ncols, nrows = ds.meta['width'], ds.meta['height']
        #offsets = itertools.product(range(0, ncols, self.tile_width), range(0, nrows, self.tile_height))
        offsets_w, offsets_h = self.offset_tiles(img_as_np)
        offsets = itertools.product(offsets_w, offsets_h)
        big_window = rio.windows.Window(col_off=0,
                                    row_off=0,
                                    width=ncols,
                                    height=nrows)

        for col_off, row_off in offsets:
            window = rio.windows.Window(col_off=col_off, row_off=row_off, width=self.tile_width, height=self.tile_height).intersection(big_window)
            transform = rio.windows.transform(window, ds.transform)
            yield window, transform

    def tiling(self) -> List[str]:
        #input_path = os.join.path(self.master_folder_path, act_path)
        master_index = 0
        input_paths = list()

        if self.use_pre:
            input_paths_subitems = ["pre/", "post/", "mask/"]
        else:
            input_paths_subitems = ["post/", "mask/"]

        try:
            with open(f"{self.log_file_path}/{self.log_filename}", "r") as act_paths_file:
                for act_path in act_paths_file:
                    input_paths.append(act_path)
        except:
            print(f"Error building useful activations list.")
        finally:
            if self.verbose==1:
                print("Finished building input paths list. Proceeding.")

        for idx in tqdm(range(len(input_paths))):
            tile_info = list()
            current_processing_folder=f"processed_{input_paths[idx]}"
            os.mkdir(f"processed_{input_paths[idx]}")
            if self.verbose==1:
                print(f"Working with: {input_paths[idx]}")
                print(f"Creating processing folder: processed_{input_paths[idx]}")
            for subitem in input_paths_subitems:
                with rio.open(os.path.join(input_paths[idx], subitem)) as inds:

                    #tile_width, tile_height = self.tile_width, self.tile_height

                    meta = inds.meta.copy()

                    #qui creo la cartella di output delle tiles
                    os.mkdir(f"{current_processing_folder}/{subitem}")
                    output_path = f"{current_processing_folder}/{subitem}"
                    output_filename = 'tile_{}_{}.tif'

                for window, transform in self.get_tiles(inds):
                    master_index += 1
                    print(window)
                    meta['transform'] = transform
                    meta['width'], meta['height'] = window.width, window.height
                    outpath = os.path.join(output_path,output_filename.format(int(window.col_off), int(window.row_off)))
                    with rio.open(outpath, 'w', **meta) as outds:
                        outds.write(inds.read(window=window))
                        tile_info.append(inds.read(window=window))


                    self.log_dict_info(
                        act_id=input_paths[idx],
                        idx = master_index,
                        tiles_info=tile_info,
                        subitem=subitem
                    )

                    #qui cancello la cartella originale dopo aver fatto la lavorazione
                os.rmdir(input_paths[idx])

        json.dumps(self.log_dict_info, self.master_dict_path)
            
class ImageDataset(Dataset):
    def __init__(self, processed_master_folder_path=None, master_dict_path=None, tile_width=512, tile_height=512, transform=None, use_pre=False, verbose=0):
        self.processed_master_folder_path=processed_master_folder_path
        self.master_dict_path=master_dict_path
        self.tile_width=tile_width
        self.tile_heigth=tile_height
        self.transform=transform
        self.use_pre=use_pre
        self.verbose=verbose

        self.master_dict = json.load(self.master_dict_path)

    def __len__(self) -> int:
        try:
            with open(f"{self.master_dict_path}", "r") as jsonfp:
                info_dict = json.load(jsonfp)
        except:
            print(f"Error dealing with log json file at: {self.master_dict_path}")
        finally:
            return len(info_dict)

    def __getitem__(self, idx):
        """
        Questa funzione prende in input un indice relativo all'immagine e alla maschera che vogliamo caricare sull'algoritmo da trainare
        e, tramite l'indice, carica la relativa immagine e maschera e le ritorna in un dizionario.
        """
        local_dict = dict()

        local_dict['post_image_tile'] = self.master_dict[f"act_id_{idx}"]["tile_info_post"]
        local_dict['mask_tile'] = self.master_dict[f"act_id_{idx}"]["tile_info_mask"]
        if self.use_pre:
            local_dict['pre_image_tile'] = self.master_dict[f"act_id_{idx}"]["tile_info_pre"]

        return local_dict
    

    



