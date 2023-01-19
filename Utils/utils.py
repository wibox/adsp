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
from Utils.smooth_segmenter import predict_img_with_smooth_windowing

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

def merge_tiles(activations_path : str) -> bool:
    # output_fp_gt = "gt_final.tif"
    # output_fp_pred = "pred_final.tif"
    # for activation in os.listdir(activations_path):

    #     if not os.path.exists(os.path.join(activations_path, activation, "output_gt")):
    #         os.makedirs(f"{activations_path}/{activation}/output_gt")
    #     if not os.path.exists(os.path.join(activations_path, activation, "output_pred")):
    #         os.makedirs(f"{activations_path}/{activation}/output_pred")
    #     tiles = glob.glob(f"{activations_path}/{activation}/*.tif")
    #     gt_tiles = [gt_tile for gt_tile in tiles if gt_tile.split("/")[-1].split("_")[1]=="gt"]
    #     pred_tiles = [pred_tile for pred_tile in tiles if pred_tile.split("/")[-1].split("_")[1]=="pred"]
    #     gt_mosaic = list()
    #     pred_mosaic = list()
    #     for gt_tile, pred_tile in zip(gt_tiles, pred_tiles):
    #         gt_src = rio.open(gt_tile)
    #         pred_src = rio.open(pred_tile)
    #         gt_mosaic.append(gt_src)
    #         pred_mosaic.append(pred_src)
    #     print(len(gt_mosaic))
    #     complete_gt_mosaic, gt_out_trans = merge(gt_mosaic)
    #     complete_pred_mosaic, pred_out_trans = merge(pred_mosaic)

    #     with rio.open(os.path.join(activations_path, activation, "output_gt", output_fp_gt), "w", driver="GTiff", height=complete_gt_mosaic.shape[1], width=complete_gt_mosaic.shape[2], count=1, transform=gt_out_trans) as output:
    #         output.write(complete_gt_mosaic)
    #     with rio.open(os.path.join(activations_path, activation, "output_pred", output_fp_pred), "w", driver="GTiff", height=complete_pred_mosaic.shape[1], width=complete_pred_mosaic.shape[2], count=1, transform=pred_out_trans) as output:
    #         output.write(complete_pred_mosaic)


def format_image(img : np.ndarray = None) -> Union[None, np.ndarray]:
    _formatted_image = list()
    #_formatted_image.append(img[0, :, :])
    _formatted_image.append(img[1, :, :])
    _formatted_image.append(img[2, :, :])
    _formatted_image.append(img[3, :, :])
    _formatted_image.append(img[4, :, :])
    _formatted_image.append(img[5, :, :])
    _formatted_image.append(img[6, :, :])
    _formatted_image.append(img[7, :, :])
    _formatted_image.append(img[8, :, :])
    #_formatted_image.append(img[9, :, :])
    _formatted_image.append(img[10, :, :])
    _formatted_image.append(img[11, :, :])
    _formatted_image = np.array(_formatted_image)
    return np.clip(_formatted_image, 0, 1)

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
                self.config = json.load(f)
        except Exception as e:
            print(e.format_exc())
            sys.exit()
        finally:
            return completed, self.config

    def get_config(self) -> dict:
        return self.config