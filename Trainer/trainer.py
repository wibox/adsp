from typing import *
from ..Utils.errors import *
from ..Utils.transforms import *
from tqdm import tqdm
import rasterio as rio
import torch
from torch.utils.data.dataloader import DataLoader
from torch import nn, optim

class Trainer():
    def __init__(self, device : str = None, net : Any = None, net_args : Dict[str, str] = None, batch_size : int = None, dataset = None, epoch : int = None, act_function = None, lr : float = None, dropout_prob : float = None, loss_function = None, squeeze_mask : bool = True, transforms=None, verbose : int = 1):
        self.device = device
        self.net = net
        self.net_args = net_args
        self.batch_size = batch_size
        self.dataset = dataset
        self.epoch = epoch
        self.act_function = act_function
        self.lr = lr
        self.dropout_prob = dropout_prob
        self.optimizer = optim.Adam(lr=self.lr)
        self.loss_function = loss_function
        self.squeeze_mask = squeeze_mask
        self.transforms = transforms
        self.verbose = verbose

    def _load_tiles(self, _post_tile_path : str = None, _mask_tile_path : str = None) -> Tuple[bool, Union[rio.DatasetReader, None], Union[rio.DatasetReader, None]]:
        completed = False
        _post_dr = None
        _mask_dr = None
        try:
            _post_dr = rio.open(_post_tile_path, mode="r")
            _mask_dr = rio.open(_mask_tile_path, mode="r")
            completed = True
        except Exception as e:
            print(e.format_exc())
        finally:
            return completed, _post_dr, _mask_dr

    def _instantiate(self) -> Any:
        return self.net(**self.net_args)

    def _train(self) -> bool:
        completed = False
        for epoch in tqdm(range(self.epoch)):
            running_loss = 0.0
            epoch_loss = 0.0
            # instantiate dataloader
            my_dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)
            datatype = torch.long
            
            self.net.train()

            for data_idx, batch in enumerate(my_dataloader): # sistemare la __get_item__() -> FATTOOOOOOOOO
                current_image_path, current_mask_path = batch[0], batch[1]
                post_tile_dr, mask_tile_dr = self._load_tiles(_post_tile_path=current_image_path, _mask_tile_path=current_mask_path)
                # apply rasterization and transforms

                if self.transforms is not None:
                    post_tile_dr = self.transforms(post_tile_dr)
                    mask_tile_dr = self.transforms(mask_tile_dr)

                post_tile_dr = post_tile_dr.to(self.device)
                mask_tile_dr = mask_tile_dr.to(self.device, dtype=datatype)

                if self.squeeze_mask:
                    current_mask = current_mask.squeeze(dim=1)

                self.optimizer.zero_grad()
                outputs = self.net(post_tile_dr)
                loss = self.loss_function(outputs, current_mask)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                epoch_loss += loss.item()

            # here validation should be perfomed if cross validation is desired.

        completed = True
        return completed

    def _fect_results(self):
        pass