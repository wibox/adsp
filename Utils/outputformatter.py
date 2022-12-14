from Preprocessing import imagedataset

import torch
import onnx
import onnxruntime as ort
import numpy as np
import rasterio as rio

from typing import *

import os
import traceback

class OutputFormatter():
    """
    Questa classe prende in input una tupla di (<immagine originale>, <maschera originale>, <maschera predetta>)
    Fa l'hstack e le salva in un certo path per ogni ingresso dato in input e fa la stessa cosa per il testing -> op
    """
    def __init__(
        self,
        device : str,
        filepath : str,
        test_ds : imagedataset.ImageDataset,
        best_model_path : str,
        bands : int = 1,
        verbose : int = 1
    ):
        self.device = device
        self.filepath = filepath
        self.test_ds = test_ds
        self.best_model_path = best_model_path
        self.bands = bands
        self.verbose = verbose

        self.best_model = None
        self.output_path = None

    def _initialize(self):
        if os.path.exists(self.best_model_path):
            self.best_model = onnx.load(self.best_model_path)
            onnx.checker.check_model(self.best_model)
            self.ort_session = ort.InferenceSession(self.best_model_path)
        else:
            raise Exception("Best model not found.")
        
        if not os.path.exists(os.path.join(self.filepath, "test_output_colombaset")):
            print(f"Creating test's output folder: {os.path.join(self.filepath, 'test_output_colombaset')}")
            os.makedirs(os.path.join(f"{self.filepath}"), "test_output_colombaset")
        self.output_path = os.path.join(self.filepath, 'test_output_colombaset')

    def _save_output(self, cat : str, _input : np.ndarray = None, idx : int = 0) -> bool:
        completed = False
        height = 512
        width = 512
        try:
            with rio.open(os.path.join(f"{self.output_path}", f"{cat}_{idx}.tif"), "w", driver="GTiff", height=height, width=width, count=1, dtype=str(_input.dtype)) as outds:
                outds.write(np.squeeze(_input, axis=0), indexes=self.bands)
        except Exception as e:
            print(traceback.format_exc())
        finally:
            return completed

    def _compute_output(self):
        for idx in range(len(self.test_ds)):
            # prendi l'immagine
            current_image, current_gt_mask = self.test_ds[idx]
            # la converti in tensore
            #current_tensor = torch.from_numpy(current_image).to(self.device).unsqueeze(0)
            # le fai fare la prediction
            current_image = np.expand_dims(current_image, axis=0)
            predicted_mask = self.ort_session.run(None, {self.ort_session.get_inputs()[0].name : current_image})
            predicted_mask = np.squeeze(np.array(predicted_mask[0]), axis=0)
            predicted_mask = np.where(predicted_mask > .5, 1, 0)
            # ti ricostruisci la maschera
            # predicted_mask = predicted_mask.detach().cpu().squeeze().numpy()
            # faccio hstack
            # formatted_output = np.hstack((current_gt_mask, np.squeeze(np.array(predicted_mask[0]), axis=0)))
            # salvo nel path
            self._save_output(cat="gt", _input=current_gt_mask, idx=idx)
            self._save_output(cat="pred", _input=predicted_mask, idx=idx)