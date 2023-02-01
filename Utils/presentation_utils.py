from typing import *
import numpy as np
from .errors import *
import itertools
import rasterio as rio
from rasterio.plot import show
import os
import glob
import torch
from tqdm import tqdm
import traceback

def offset_tiles(img : Union[np.ndarray, None] = None, tile_dim : Tuple[int, int] = (512, 512)) -> Tuple[Union[List[float], Any], Union[List[float], Any]]:
    """
    This function performs the tiling of an input image according to specific criteria.
    If image shape is not (<num_channels>, <height>, <width>) then expand one dimension.
    If image.height % <tile_size> |= 0 or image.width % <tile_size> != 0 perform overlapping with threshold.
    """
    try:
        if img is None:
            raise NotAnImage(path=img)
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
            if remaining_h > 112:
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
            if remaining_w > 112:
                num_tile_w = int(np.floor(width/tile_w)) + 1
            else:
                num_tile_w = int(np.floor(width/tile_w))
            offsets_w = [i*tile_w for i in range(num_tile_w-1)]
            offsets_w.append(width - tile_w)

    except NotAnImage as nai:
        print(nai)
    finally:
        return offsets_w, offsets_h

def get_tiles(ds):
    """
    Generator used to yield tiles one at a time. Employed in self.tiling()
    """
    # shape = (c, h, w) questa è l'immagine intera
    img_as_np = ds.read()
    ncols, nrows = ds.meta['width'], ds.meta['height']
    #offsets = itertools.product(range(0, ncols, self.tile_width), range(0, nrows, self.tile_height))
    offsets_w, offsets_h = offset_tiles(img_as_np)

    if offsets_w is None or offsets_h is None:
        raise NoneTypeEncountered(correct_type="List[float]")
    
    offsets = itertools.product(offsets_w, offsets_h)
    big_window = rio.windows.Window(col_off=0,
                                row_off=0,
                                width=ncols,
                                height=nrows)

    for col_off, row_off in offsets:
        window = rio.windows.Window(col_off=col_off, row_off=row_off, width=512, height=512).intersection(big_window)
        transform = rio.windows.transform(window, ds.transform)
        yield window, transform

def tiling(initial_img_path : str, output_path_str : str = "tmp/tiles"):
    try:
        if not os.path.exists("tmp/tiles/"):
            print("Creating log folders...")
            os.makedirs("tmp/tiles/")
        with rio.open(initial_img_path, "r") as inds:
            print("Loading initial image into memory...")
            print(f"Test image shape: {inds.read().shape}")

            meta = inds.meta.copy()
            output_filename = 'tile_{}_{}.tif'
            output_path = output_path_str

            print("Computing tiles...")
            for window, transform in get_tiles(inds): # retrieving tiles and log paths
                meta['transform'] = transform
                meta['width'], meta['height'] = window.width, window.height
                outpath = os.path.join(output_path,output_filename.format(int(window.col_off), int(window.row_off)))
                with rio.open(outpath, 'w', **meta) as outds:
                    outds.write(inds.read(window=window))
    except Exception as e:
        print(traceback.format_exc())
    finally:
        return True

def merge_tiles(test_img_path : str) -> bool:
    starting_img = rio.open(test_img_path, "r").read()
    empty_canvas = np.zeros(shape=(1, starting_img.shape[1], starting_img.shape[2]))
    print("Empty canvas shape: ", empty_canvas.shape)

    mosaic = glob.glob("tmp/predictions/*.tif")
    col_idx = set()
    row_idx = set()
    for tile_info in mosaic:
        col, row = int(tile_info.split("_")[1]), int(tile_info.split("_")[2].split(".")[0])
        col_idx.add(col)
        row_idx.add(row)

    columns = list()
    for col_id in sorted(col_idx):
        initial_canvas = np.empty(shape=(1, 512, 512))
        for row_id in sorted(row_idx):
            new_patch = rio.open(f"tmp/predictions/pred_{col_id}_{row_id}.tif", "r").read()
            # offset = 512-(row_id-512) if row_id%512 != 0 else 0
            offset = 512-(row_id%512) if row_id%512 != 0 else 0
            initial_canvas = np.concatenate((initial_canvas, new_patch[:, offset:, :]), axis=1)
        columns.append(initial_canvas[:, 512:, :])
    empty_final_canvas = np.empty(shape=(1, starting_img.shape[1], 512))
    for column in columns:
        print(column.shape)
        empty_final_canvas = np.concatenate((empty_final_canvas, column[:, :, :]), axis=2)
    return empty_final_canvas, starting_img.shape[2]

def make_predictions(model : Any, model_type : str, tiles_path : str = "tmp/tiles") -> bool:
    bands = {
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
            "B11": 11,
            "B12": 12,
    }
    if model_type == "vanilla" or model_type == "ben" or model_type=="effis":
        bands_name = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
    else:
        bands_name = ["B04", "B03", "B02", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
    bands_idx = [bands[x]+1 for x in bands_name]

    if not os.path.exists("tmp/predictions/"):
        os.makedirs("tmp/predictions/")
    
    for tile_info in tqdm(glob.glob(f"{tiles_path}/*.tif")):
        bound1, bound2 = tile_info.split("_")[1], tile_info.split("_")[2].split(".")[0]
        current_tile = format_image(img=rio.open(tile_info, "r").read(bands_idx))
        current_tile = np.expand_dims(current_tile, axis=0)
        prediction = model(torch.tensor(current_tile))
        prediction = torch.sigmoid(prediction.detach())
        prediction = prediction.numpy()
        prediction = np.where(prediction[0] > 0.5, 1, 0).astype(np.float32)
        # show(prediction, cmap="gray")
        with rio.open(f"tmp/predictions/pred_{bound1}_{bound2}.tif", "w", driver="GTiff", height=512, width=512, count=1, dtype=str(prediction.dtype)) as outds:
            outds.write(prediction)

def format_image(img : np.ndarray = None) -> Union[None, np.ndarray]:
    return np.clip(img, 0, 1)

def format_mask(mask : np.ndarray = None) -> Union[None, np.ndarray]:
    return np.clip(mask, 0, 1)

def write_results(final_img : np.ndarray) -> bool:
    height = final_img.shape[1]
    width = final_img.shape[2]
    with rio.open("tmp/example.tif", "w", driver="GTiff", height=height, width=width, count=1, dtype=str(final_img.dtype)) as outds:
        outds.write(final_img)