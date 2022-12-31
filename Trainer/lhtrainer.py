from Preprocessing import imagedataset

from typing import *

import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader

class Trainer():
    def __init__(
        self,
        model : Any,
        training_dataset : imagedataset.ImageDataset,
        validation_dataset : imagedataset.ImageDataset,
        epochs : int,
        device : str,
        loss : Any,
        optimizer : Any,
        batch_size : int,
        num_workers : int,
        shuffle : bool,
        metrics : Any
    ):
        self.model = model
        self.train_ds = training_dataset
        self.val_ds = validation_dataset
        self.epochs = epochs
        self.device = device
        self.loss = loss
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.metrics = metrics
        self.export_format = torch.rand(1, 1, 512, 512, requires_grad=True, device='cuda')

    def _initialize(self):
        self.train_ds._load_tiles()
        self.train_dl = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)
        self.val_ds._load_tiles()
        self.val_dl = DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)

        self.train_epoch = smp.utils.train.TrainEpoch(
            model=self.model,
            loss=self.loss,
            metrics=self.metrics,
            optimizer=self.optimizer,
            device=self.device,
            verbose=True
        )
        self.valid_epoch = smp.utils.train.ValidEpoch(
            model=self.model,
            loss=self.loss,
            metrics=self.metrics,
            device=self.device,
            verbose=True
        )

    def _train(self):
        
        best_iou_score = 0.0
        train_logs_list, valid_logs_list = list(), list()

        for epoch_idx in range(self.epochs):

            print(f"\nEpoch: {epoch_idx}")
            train_logs = self.train_epoch.run(self.train_dl)
            valid_logs = self.valid_epoch.run(self.val_dl)
            train_logs_list.append(train_logs)
            valid_logs_list.append(valid_logs)

        if best_iou_score < valid_logs['iou_score']:
            best_iou_score = valid_logs['iou_score']
            torch.onnx.export(
                self.model,
                args=self.export_format,
                f="/mnt/data1/adsp_data/best_model.onnx",
                export_params=True
                )
            print("Saving model...")

