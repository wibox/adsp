from torch.utils.data.dataset import Dataset

import json
import os
import traceback
from typing import *

import numpy as np
import rasterio as rio

from tqdm import tqdm

class EffisDataset(Dataset):
    def __init__(
        self,
        log_folder : str,
        master_dict : str,
        transformations : List[Any]
    ):
        self.log_folder = log_folder
        self.master_dict = master_dict
        self.transformations = transformations

        self.num_tiles = 0
        self.post_tiles : List[str] = list()
        self.mask_tiles : List[str] = list()

    def _load_tiles(self) -> Tuple[bool, Dict[Any, Any]]:
        completed = False
        self.loaded_tile_data = {}
        try:
            with open(os.path.join(self.log_folder, self.master_dict), "r") as md:
                self.loaded_tile_data = json.load(md)
            print("Loading tiles paths in dataset...")
            for act in tqdm(self.loaded_tile_data.keys()):
                self.num_tiles += len(self.loaded_tile_data[act]["tile_info_post"])
                self.post_tiles.extend(self.loaded_tile_data[act]["tile_info_post"])
                self.mask_tiles.extend(self.loaded_tile_data[act]["tile_info_mask"])
        except Exception as e:
            print(traceback.format_exc())
        finally:
            return completed

    def _read_tile_image(self, is_tile : bool, tile_path : str) -> Union[np.ndarray, None]:
        loaded_img = None
        try:
            if tile_path is not None:
                with rio.open(tile_path) as input_tile_path:
                    if is_tile:
                        #loaded_img = input_tile_path.read(self.bands_idx)
                        loaded_img = input_tile_path.read()
                    else:
                        loaded_img = input_tile_path.read()
            else:
                raise Exception("Provided empty tile!")
        except Exception as e:
            print(e.format_exc())
        finally:
            return loaded_img

    def _make_channels_first(self, mask : np.ndarray) -> np.ndarray:
        return np.moveaxis(mask, -1, 0)

    def _make_channels_last(self, image : np.ndarray, mask : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        _swap_image = np.moveaxis(image, 0, -1)
        _swap_mask = np.moveaxis(mask, 0, -1)
        return _swap_image, _swap_mask

    def _format_image(self, img : np.ndarray = None) -> Union[None, np.ndarray]:
        return np.clip(img, 0, 1)

    def _format_mask(self, mask : np.ndarray = None) -> Union[None, np.ndarray]:
        return np.clip(mask, 0, 1)

    def __len__(self) -> int:
        if self.num_tiles == 0:
            raise Exception("Number of provided tiles is 0.")
        return self.num_tiles

    def __getitem__(self, index : int) -> Tuple[Union[np.ndarray, None], Union[np.ndarray, None]]:
        my_image = self._read_tile_image(is_tile=True, tile_path = self.post_tiles[index])
        my_mask = self._read_tile_image(is_tile=False, tile_path = self.mask_tiles[index])
        
        if self.transformations is not None:
            my_image, my_mask = self._make_channels_last(image=my_image, mask=my_mask)
            applied_transform = self.transformations(image=my_image, mask=my_mask)
            my_image = applied_transform['image'].numpy()
            my_mask = applied_transform['mask'].numpy()
            my_mask = self._make_channels_first(mask=my_mask)
            # my_mask = self._format_mask(mask=my_mask)
            # my_image = self._format_image(img=my_image)
            my_mask = (my_mask>0).astype(np.uint8)
        
        return my_image, my_mask