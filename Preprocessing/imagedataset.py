import json
import os

from typing import *

import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import rasterio as rio

class ImageDataset(Dataset):
    def __init__(
        self,
        formatted_folder_path : str = None,
        log_folder : str = None,
        master_dict=None,
        transformations : List[Any] = None,
        use_pre : bool = False,
        verbose : int = 0,
        specific_indeces : List[int] = 0,
        return_path : bool = False
    ):

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

    def _read_tile_image(self, tile_path : str = None) -> Union[np.ndarray, None]:
        current_image = None
        try:
            if tile_path is not None:
                with rio.open(tile_path) as input_tile_path:
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

    def _format_image(self, img : np.ndarray = None) -> Union[None, np.ndarray]:
        _formatted_image = list()
        _formatted_image.append(img[3, :, :])
        _formatted_image.append(img[2, :, :])
        _formatted_image.append(img[1, :, :])
        _formatted_image.append(img[4, :, :])
        _formatted_image.append(img[5, :, :])
        _formatted_image.append(img[6, :, :])
        _formatted_image.append(img[7, :, :])
        _formatted_image.append(img[8, :, :])
        _formatted_image.append(img[10, :, :])
        _formatted_image.append(img[11, :, :])
        _formatted_image = np.array(_formatted_image)
        return np.clip(_formatted_image, 0, 1)

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

            my_image = self._read_tile_image(tile_path=self.post_tiles[idx])
            my_mask = self._read_tile_image(tile_path=self.mask_tiles[idx])

            if self.transformations is not None:
                my_image = self._format_image(img=my_image)
                my_image, my_mask = self._make_channels_last(image=my_image, mask=my_mask)
                applied_transform = self.transformations(image=my_image, mask=my_mask)
                my_image = applied_transform['image'].numpy()
                my_mask = applied_transform['mask'].numpy()
                my_mask = self._make_channels_first(mask=my_mask)
                my_mask = self._format_mask(mask=my_mask)
                my_mask = (my_mask>0).astype(np.uint8)
                
            return my_image, my_mask