import os
import sys
import glob
import json
import traceback

from typing import *

import torch
import numpy as np
import random
import rasterio as rio
from rasterio.merge import merge

def read_tile(tile_path : str = None) -> Union[np.ndarray, None]:
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
                self.config = json.load(f)
        except Exception as e:
            print(e.format_exc())
            sys.exit()
        finally:
            return completed, self.config

    def get_config(self) -> dict:
        return self.config