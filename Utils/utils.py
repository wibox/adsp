import os
import sys
import json
import traceback

from typing import *

import torch
import numpy as np
import random
import rasterio as rio
from termcolor import colored

def read_tile(bands : List[int], tile_path : str = None) -> Union[np.ndarray, None]:
    current_image = None
    try:
        if tile_path is not None:
            with rio.open(tile_path) as input_tile_path:
                current_image = input_tile_path.read(bands)
        else:
            raise Exception("Provided empty tile!")
    except Exception as e:
        print(e.format_exc())
    finally:
        return current_image

def format_image(img : np.ndarray = None) -> Union[None, np.ndarray]:
    return np.clip(img, 0, 1)

def format_mask(mask : np.ndarray = None) -> Union[None, np.ndarray]:
    return np.clip(mask, 0, 1)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class Configurator():
    def __init__(self, filepath : str = None, filename : str = None):
        self.filepath = filepath
        self.filename = filename
        self.config = self._load_config()

    def _load_config(self) -> Tuple[bool, Dict[str, str]]:
        completed = False
        try:
            with open(os.path.join(self.filepath, self.filename), "r") as f:
                config = json.load(f)
            print(colored("Configurations correctly loaded.", "green"))
        except Exception as e:
            print(colored("Coulnd't load configurations properly.", "red"))
            print(traceback.format_exc())
            sys.exit()
        finally:
            return config

    def get_config(self) -> Dict[str, str]:
        return self.config

def freeze_encoder(model):
    for child in model.encoder.children():
        for param in child.parameters():
            param.requires_grad = False
    return None

def unfreeze(model):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = True
    return None