from Preprocessing import imagedataset

import torch
import onnx
import onnxruntime as ort
import numpy as np
import rasterio as rio

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
        bands : int = 3,
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
        
        if not os.path.exists(self.filepath):
            print(f"Creating test's output folder: {os.path.join(self.filepath, 'test_output_colombaset')}")
            os.makedirs(os.path.join(f"{self.filepath}"), "test_output_colombaset")
            self.output_path = os.path.join(self.filepath, 'test_output_colombaset')

    def _save_output(self, _input : np.ndarray = None, idx : int = 0) -> bool:
        completed = False
        height = 512
        width = 512
        try:
            with rio.open(os.path.join(f"{self.output_path}", f"{idx}.tif"), "w", driver="GTiff", height=height, width=3*width, count=1, dtype=str(_input.dtype)) as outds:
                outds.write(_input, indexes=self.bands)
        except Exception as e:
            print(traceback.format_exc())
        finally:
            return completed

    def _compute_output(self):
        for idx in range(len(self.test_ds)):
            # prendi l'immagine
            current_image, current_gt_mask = self.test_ds[idx]
            # la converti in tensore
            current_tensor = torch.from_numpy(current_image).to(self.device).unsqueeze(0)
            # le fai fare la prediction
            predicted_mask = self.ort_session.run(None, {self.ort_session.get_inputs()[0].name : current_tensor})
            # ti ricostruisci la maschera
            # predicted_mask = predicted_mask.detach().cpu().squeeze().numpy()
            # faccio hstack
            formatted_output = np.hstack([current_gt_mask, predicted_mask[0]])
            # salvo nel path
            self._save_output(_input=formatted_output, idx=idx)