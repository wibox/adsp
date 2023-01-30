import json
import os

from typing import *

import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import rasterio as rio

class ColombaDataset(Dataset):
    def __init__(
        self,
        model_type : str,
        formatted_folder_path : str = None,
        log_folder : str = None,
        master_dict=None,
        transformations : List[Any] = None,
        use_pre : bool = False,
        verbose : int = 0,
        specific_indeces : List[int] = 0,
        return_path : bool = False
    ):

        self.model_type=model_type
        self.formatted_folder_path=formatted_folder_path
        self.log_folder=log_folder
        self.master_dict=master_dict
        self.transformations=transformations
        self.use_pre=use_pre
        self.verbose=verbose
        self.specific_indeces = specific_indeces
        self.return_path = return_path # if True returns tile path, il False return np.ndarray of tiles

        self.post_tiles = list()
        self.mask_tiles = list()
        self.activations = os.listdir(self.formatted_folder_path)
        self.bands = {
                "B01": 0,
                "B02": 1,
                "B03": 2,
                "B04": 3,
                "B05": 4,
                "B06": 5,
                "B07": 6,
                "B08": 7,
                "B8A": 8,
                "B09": 9,
                # "B10": 10,
                "B11": 11,
                "B12": 12,
        }
        if self.model_type == "vanilla" or self.model_type == "ben":
            self.bands_name = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
        else:
            self.bands_name = ["B04", "B03", "B02", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
        self.bands_idx = [self.bands[x]+1 for x in self.bands_name]

        self.BAND_STATS_S2 = {
            "mean": {
                "B01": 340.76769064,
                "B02": 429.9430203,
                "B03": 614.21682446,
                "B04": 590.23569706,
                "B05": 950.68368468,
                "B06": 1792.46290469,
                "B07": 2075.46795189,
                "B08": 2218.94553375,
                "B8A": 2266.46036911,
                "B09": 2246.0605464,
                "B11": 1594.42694882,
                "B12": 1009.32729131,
            },
            "std": {
                "B01": 554.81258967,
                "B02": 572.41639287,
                "B03": 582.87945694,
                "B04": 675.88746967,
                "B05": 729.89827633,
                "B06": 1096.01480586,
                "B07": 1273.45393088,
                "B08": 1365.45589904,
                "B8A": 1356.13789355,
                "B09": 1302.3292881,
                "B11": 1079.19066363,
                "B12": 818.86747235,
            },
        }
        self.SHUB_MEAN = [ x for x in self.BAND_STATS_S2["mean"].values()]
        self.SHUB_STD = [ x for x in self.BAND_STATS_S2["std"].values()]

        try:
            loading_completed = False
            with open(os.path.join(self.log_folder, self.master_dict), "r") as md: #master_dict
                self.loaded_tile_data = json.load(md) # questo è il dizionario completo
            loading_completed = True
        except OSError as ose:
            print(ose)
        finally:
            return None

    def _load_tiles(self) -> bool:
        """
        Questa funzione deve caricare nell'istanza della classe le liste di tutti i path di post e mask per ogni attivazione.
        self.post_tiles = [[tiles_post_act_1] + [tiles_post_act_2] + ...]
        self.mask_tiles = [[tiles_mask_act_1] + [tiles_mask_act_2] + ...]
        """
        completed = False
        try:
            for activation_idx in range(len(self.loaded_tile_data['processing_info'])):
                current_key = list(self.loaded_tile_data['processing_info'][activation_idx].keys())[0]
                self.post_tiles.extend(self.loaded_tile_data['processing_info'][activation_idx][current_key]["tile_info_post"][0])
                self.mask_tiles.extend(self.loaded_tile_data['processing_info'][activation_idx][current_key]["tile_info_mask"][0])
            completed = True
            print("Tiles loaded successfully!")

            # Ritorna uno specifico subset di post_tiles e mask_tiles (da estendere per pre)
            # se specificati in self.specific_indeces
            if self.specific_indeces is not None:
                self.post_tiles =  np.array(self.post_tiles)
                self.post_tiles = self.post_tiles[self.specific_indeces].to_list()
                self.mask_tiles = np.array(self.mask_tiles)
                self.mask_tiles = self.mask_tiles[self.specific_indeces].to_list()

            if len(self.post_tiles) != len(self.mask_tiles):
                raise Exception("Incoherent number of tiles for post and mask!")
            else:
                self.num_mask_tiles = len(self.mask_tiles)
                self.num_post_tiles = len(self.post_tiles)
        except Exception as e:
            print(e.format_exc())
        finally:
            return completed

    def _read_tile_image(self, is_tile : bool, tile_path : str = None) -> Union[np.ndarray, None]:
        current_image = None
        try:
            if tile_path is not None:
                with rio.open(tile_path) as input_tile_path:
                    if is_tile:
                        current_image = input_tile_path.read(self.bands_idx)
                    else:
                        current_image = input_tile_path.read()
            else:
                raise Exception("Provided empty tile!")
        except Exception as e:
            print(e.format_exc())
        finally:
            return current_image

    def _make_channels_first(self, mask : np.ndarray) -> np.ndarray:
        return np.moveaxis(mask, -1, 0)

    def _make_channels_last(self, image : np.ndarray, mask : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        _swap_image = np.moveaxis(image, 0, -1)
        _swap_mask = np.moveaxis(mask, 0, -1)
        return _swap_image, _swap_mask

    def perform_min_max(self, input_img : np.ndarray, dataset_type:str):
        mean = np.array(self.SHUB_MEAN)
        std = np.array(self.SHUB_STD)
        mins = (mean - 2 * std)[:, None, None].astype(np.float32)
        maxs = (mean + 2 * std)[:, None, None].astype(np.float32)
        output_img = (input_img - mins) / (maxs-mins)
        return output_img

    def _format_image(self, img : np.ndarray = None) -> Union[None, np.ndarray]:
        return np.clip(img, 0, 1)

    def _format_mask(self, mask : np.ndarray = None) -> Union[None, np.ndarray]:
        return np.clip(mask, 0, 1)

    def __len__(self) -> int:
        if len(self.post_tiles) == len(self.mask_tiles):
            return len(self.post_tiles)
        else:
            raise Exception("Number of tiles different from number of masks.")

    def __getitem__(self, idx) -> Union[Dict[str, str], Dict[str, np.ndarray]]:
        """
        Questa funzione prende in input un indice relativo all'immagine e alla maschera che vogliamo caricare sull'algoritmo da trainare
        e, tramite l'indice, carica la relativa immagine e maschera e le ritorna in un dizionario se self.return_path è False, altrimenti
        ritorna il path di post_tile e mask_tile.
        """
        if self.return_path:

            item_dict = dict()
            my_image = None
            my_mask = None

            if self.transformations is not None:
                my_image = self.transformations(self.post_tiles[idx])
                my_mask = self.transformations(self.mask_tiles[idx])
            
            if my_image is not None or my_mask is not None:
                item_dict["image"] = my_image
                item_dict["mask"] = my_mask
                return item_dict
            else:
                raise Exception("Error when loading mask or image.")

        else:

            my_image = self._read_tile_image(tile_path=self.post_tiles[idx], is_tile=True)
            my_mask = self._read_tile_image(tile_path=self.mask_tiles[idx], is_tile=False)

            if self.transformations is not None:
                # my_image = self._format_image(img=my_image)
                my_image, my_mask = self._make_channels_last(image=my_image, mask=my_mask)
                applied_transform = self.transformations(image=my_image, mask=my_mask)
                my_image = applied_transform['image'].numpy()
                # my_image = self.perform_min_max(input_img=my_image, dataset_type=self.dataset_type)
                my_mask = applied_transform['mask'].numpy()
                my_mask = self._make_channels_first(mask=my_mask)
                # my_mask = self._format_mask(mask=my_mask)
                # my_image = self._format_image(img=my_image)
                my_mask = (my_mask>0).astype(np.uint8)
                
            return my_image, my_mask