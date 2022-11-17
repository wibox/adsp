import os
import itertools
import json

from typing import *

from tqdm import tqdm

import rasterio as rio

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
    def __init__(self, master_folder_path=None, log_file_path=None, log_filename=None, master_dict_path=None, valid_list=None, tile_height=512, tile_width=512, use_pre=False, verbose=0):
        self.master_folder_path=master_folder_path
        self.log_file_path=log_file_path
        self.log_filename=log_filename
        self.master_dict_path=master_dict_path
        self.valid_list=valid_list
        self.tile_height=tile_height
        self.tile_width=tile_width
        self.use_pre=use_pre
        self.verbose=verbose

        self.master_dict_info = {"processing_info": []}

    def log_dict_info(self,
    act_id : str,
    tiles_info : List[str],
    subitem : str
     ) -> Dict:
        logging_info = {
            f"{act_id}" : {
                f"tiles_info_{subitem}" : tiles_info
            }
        }
        self.master_dict_info["processing_info"].append(logging_info)
        return self.master_dict_info

    def check_dimensions(self) -> bool:
        """
        Implementare questa funzione non ha senso se rasterio effettua overlapping in automatico. Da verificare.
        """
        pass

    def get_tiles(self, ds):
        ncols, nrows = ds.meta['width'], ds.meta['height']
        offsets = itertools.product(range(0, ncols, self.tile_width), range(0, nrows, self.tile_height))
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
        input_paths = list()

        #log_info
        # per act_paths usiamo direttamente input_paths
        tiles_info_pre_list = list()
        tiles_info_post_list = list()
        tiles_info_mask_list = list()

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
            tiles_info = list()
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
                    print(window)
                    meta['transform'] = transform
                    meta['width'], meta['height'] = window.width, window.height
                    outpath = os.path.join(output_path,output_filename.format(int(window.col_off), int(window.row_off)))
                    with rio.open(outpath, 'w', **meta) as outds:
                        outds.write(inds.read(window=window))
                        tiles_info.append(inds.read(window=window))


                self.log_dict_info(
                    act_id=input_paths[idx],
                    tiles_info=tiles_info,
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
    
    def __len__(self) -> int:
        try:
            with open(f"{self.master_dict_path}", "r") as jsonfp:
                info_dict = json.load(jsonfp)
        except:
            print(f"Error dealing with log json file at: {self.master_dict_path}")
        finally:
            return len(info_dict)

    def __getitem__(self):
        pass