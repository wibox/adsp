from typing import *
from Utils.errors import *
from Utils.transforms import *
from tqdm import tqdm
import rasterio as rio
import torch
from torch.utils.data.dataloader import DataLoader
from torch import nn, optim

class Trainer():
    def __init__(self, device : str = None, net : Any = None, net_args : Dict[str, str] = None, batch_size : int = None, dataset = None, epoch : int = None, act_function = None, lr : float = None, dropout_prob : float = None, loss_function = None, squeeze_mask : bool = True, transformations=None, verbose : int = 1):
        self.device = device
        self.verbose = verbose
        if self.device == 'cuda':
            if torch.cuda.is_available():
                self.device = device
                if self.verbose > 0:
                    print(f"Found CUDA: using {self.device} for training.")
            else:
                self.device = 'cpu'
        elif device == 'cpu':
            self.device = device
            if self.verbose > 0:
                print(f"CUDA not found: using {self.device} for training.")
        else:
            raise Exception("Invalid device selected.")
            
        self.net = net
        self.net_args = net_args
        self.batch_size = batch_size
        self.dataset = dataset
        self.epoch = epoch
        self.act_function = act_function
        self.lr = lr
        self.dropout_prob = dropout_prob
        self.optimizer = optim.Adam(lr=self.lr, params=self.net.parameters())
        self.loss_function = loss_function
        self.squeeze_mask = squeeze_mask
        self.transformations = transformations

    def _instantiate(self) -> Any:
        return self.net(**self.net_args)

    def _load_tiles(self, _post_tile_path : str = None, _mask_tile_path : str = None) -> Tuple[bool, Union[rio.DatasetReader, None], Union[rio.DatasetReader, None]]:
        completed = False
        _post_dr = None
        # _post_dr_2 = None
        _mask_dr = None
        # print(type(_post_tile_path))
        # print(type(_mask_tile_path))
        try:
            _post_dr_1 = rio.open(_post_tile_path[0], mode="r")
            # _post_dr_2 = rio.open(_post_tile_path_2[0], mode="r")
            _mask_dr = rio.open(_mask_tile_path[0], mode="r")
            completed = True
        except Exception as e:
            print(e.format_exc())
        finally:
            return completed, _post_dr, _mask_dr

    def _train(self) -> bool:
        completed = False
        for epoch in tqdm(range(self.epoch)):
            running_loss = 0.0
            epoch_loss = 0.0
            # instantiate dataloader
            my_dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)
            datatype = torch.long
            
            self.net.train()

            for data_idx, batch in enumerate(my_dataloader):
                current_image_path, current_mask_path = batch[0], batch[1]
                loading_complete, post_tile_dr, mask_tile_dr = self._load_tiles(_post_tile_path=current_image_path, _mask_tile_path=current_mask_path)
                if loading_complete:
                    if self.transformations is not None:
                        post_tile_dr, mask_tile_dr = self.transformations((post_tile_dr, mask_tile_dr))
                    else:
                        raise Exception("Please provide useful transformations (ToTensor() needed)")

                    post_tile_dr = post_tile_dr.to(self.device)
                    mask_tile_dr = mask_tile_dr.to(self.device, dtype=datatype)

                    if self.squeeze_mask:
                        current_mask = mask_tile_dr.squeeze(dim=1)

                    self.optimizer.zero_grad()
                    outputs = self.net(post_tile_dr)
                    loss = self.loss_function(outputs, current_mask)
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item()
                    epoch_loss += loss.item()
                else:
                    raise Exception("Error during tiles loading")

        completed = True
        return completed

    def _fect_results(self):
        pass