import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer

from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex

class UNetModule(pl.LightningModule):
    """PytorchLightning trainer built upon conventions to train and log information relative to the model

    Args:
        - model : Any -> model instance from smp
        - criterion : Any -> loss function
        - lr : float -> learning rate
        - plot_every : int -> frequency at which images should be logger in self.Logger

    """

    def __init__(self, model: nn.Module, criterion: nn.Module, learning_rate: float = 1e-3, plot_every: int = 10):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.lr = learning_rate
        self.plot_every = plot_every

        self.iou = BinaryJaccardIndex()
        self.f1 = BinaryF1Score()

    def plot_images(self, x: torch.Tensor, y: torch.Tensor, p: torch.Tensor, mode: str = "train", step: int = 0):
        image = x[0][:3].cpu().numpy()
        label = y[0].cpu().numpy() * 255
        pred = p[0].cpu().numpy()
        self.logger.experiment.add_image(f"{mode}_image", image, self.global_step)
        self.logger.experiment.add_image(f"{mode}_label", label, self.global_step)
        self.logger.experiment.add_image(f"{mode}_pred", pred, self.global_step)

    def training_step(self, batch: tuple, batch_idx: int):
        x, y = batch
        #y = y.unsqueeze(1)
        logits = self.model(x)
        # tensor shapes [batch, 1, h, w]
        loss = self.criterion(logits, y.float())
        preds = torch.sigmoid(logits.detach())
        self.iou(preds, y)
        self.f1(preds, y)
        self.log("performace", {"score":self.iou, "loss":loss}, on_step=True, on_epoch=True, logger=True)
        # self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        # self.log("iou score", self.iou, on_step=True, on_epoch=True, logger=True)
        # debug images
        if self.global_step % self.plot_every == 0:
            self.plot_images(x, y, preds)
        return loss

    def training_epoch_end(self, outputs: list):
        self.log("train_iou_epoch", self.iou)
        self.log("train_f1_epoch", self.f1)

    def validation_step(self, batch : tuple, batch_idx : int):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y.float())
        preds = torch.sigmoid(logits.detach())
        self.iou(preds, y)
        self.f1(preds, y)
        self.log("performace", {"score":self.iou, "loss":loss}, on_step=True, on_epoch=True, logger=True)
        return loss

    def test_step(self, batch: tuple, batch_idx: int):
        x, y = batch
        #y = y.unsqueeze(1)
        logits = self.model(x)
        loss = self.criterion(logits, y.float())
        self.iou(logits, y)
        self.f1(logits, y)
        self.log("test_loss", loss)
        return loss

    def test_epoch_end(self, outputs: list):
        self.log("test_iou_epoch", self.iou)
        self.log("test_f1_epoch", self.f1)

    def configure_optimizers(self) -> Optimizer:
        optimizer = AdamW(self.parameters(), lr=self.lr)
        return optimizer