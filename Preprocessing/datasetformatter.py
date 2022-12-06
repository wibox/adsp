import os
import itertools

from typing import *

from tqdm import tqdm

import rasterio as rio
import numpy as np

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
    def __init__(self, master_folder_path=None, log_file_path=None, log_filename=None, master_dict_path=None, valid_list=None, tile_height=512, tile_width=512, thr_pixels = 112, use_pre=False, dataset=None, verbose=0):
        self.master_folder_path=master_folder_path
        self.log_file_path=log_file_path
        self.log_filename=log_filename
        self.master_dict_path=master_dict_path
        self.valid_list=valid_list
        self.tile_height=tile_height
        self.tile_width=tile_width
        self.thr_pixels = thr_pixels
        self.use_pre=use_pre
        self.dataset=dataset
        self.verbose=verbose

        self.master_dict_info = {"processing_info": []}

        if self.use_pre:
            self.input_paths_subitems = ["pre", "post", "mask"]
        else:
            self.input_paths_subitems = ["post", "mask"]

    def build_input_paths(self) -> Tuple[Dict, List[str]]:
        input_paths = dict()
        activations = list()
        try:
            with open(f"{self.log_file_path}/{self.log_filename}", "r") as act_paths_file:
                next(act_paths_file)
                for act_path in act_paths_file:
                    act_id = act_path.split(",")[0]
                    post = act_path.split(",")[2]
                    mask = act_path.split(",")[3]
                    if self.use_pre:
                        pre = act_path.split(",")[1]
                        input_paths[act_id] = {
                            "pre" : pre,
                            "post" : post,
                            "mask" : mask
                        }
                    input_paths[act_id] = {
                            "post" : post,
                            "mask" : mask
                        }
                    activations.append(act_id)
            return input_paths, activations
        except:
            print(f"Error building useful activations list.")
        finally:
            if self.verbose==1:
                print("Finished building input paths list. Proceeding.")

    def log_dict_info(self, act_id : str, idx : int, tile_info : List[str], subitem : str) -> bool:
        logging_info = {
            f"{act_id}_{idx}" : {
                f"tile_info_{subitem}" : tile_info
            }
        }
        try:
            self.master_dict_info["processing_info"].append(logging_info)
            return True
        except:
            raise Exception("Could not log info into master_dict.")
    
    def offset_tiles(self, img: np.ndarray, tile_dim = [512, 512]) -> Tuple[List[float], List[float]]:
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
            offsets_h.append(heigth - tile_h)
        
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
            offsets_w.append(width - tile_w)

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
        master_index = 0

        input_paths, activations = self.build_input_paths()

        for idx in tqdm(range(len(activations))):

            tile_info = list()
            current_processing_folder=f"processed_{activations[idx]}"
            os.mkdir(f"{self.master_folder_path}/processed_{activations[idx]}")

            if self.verbose==1:
                print(f"Working with: {activations[idx]}")
                print(f"Creating processing folder: processed_{activations[idx]}")

            for subitem in self.input_paths_subitems:
                # print(input_paths[activations[idx]][subitem])
                if os.path.isfile(input_paths[activations[idx]][subitem]):
                    with rio.open(input_paths[activations[idx]][subitem]) as inds:
                        meta = inds.meta.copy()
                        #qui creo la cartella di output delle tiles
                        os.mkdir(f"{self.master_folder_path}/{current_processing_folder}/{subitem}")
                        output_path = f"{self.master_folder_path}/{current_processing_folder}/{subitem}"
                        output_filename = 'tile_{}_{}.tif'

                        for window, transform in self.get_tiles(inds):
                            master_index += 1
                            meta['transform'] = transform
                            meta['width'], meta['height'] = window.width, window.height
                            outpath = os.path.join(output_path,output_filename.format(int(window.col_off), int(window.row_off)))
                            with rio.open(outpath, 'w', **meta) as outds:
                                outds.write(inds.read(window=window))
                                tile_info.append(inds.read(window=window))

                            self.log_dict_info(
                                act_id=activations[idx],
                                idx = master_index,
                                tile_info=tile_info,
                                subitem=subitem
                            )