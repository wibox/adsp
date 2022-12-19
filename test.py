from Preprocessing import datasetformatter as df
from Preprocessing import datasetscanner as ds
from Preprocessing import imagedataset as image_dataset

from Trainer import lhtrainer
from Nnet.lhNET import lhnet
from Utils.albu_transformer import OptimusPrime

from torch import nn
from catalyst.contrib.nn import DiceLoss, IoULoss

from copy import deepcopy
import os
import sys
import inspect
import argparse

from sklearn.model_selection import train_test_split
import numpy as np

from Nnet.argoNET.our_datamodule import ArgonetDataModule
from Nnet.argoNET.moco2_module import MocoV2
from Nnet.argoNET.ssl_finetuner import SSLFineTuner
import torch
from pytorch_lightning import Trainer

parser = argparse.ArgumentParser()
parser = Trainer.add_argparse_args(parser)
parser = SSLFineTuner.add_model_specific_args(parser)
parser = argparse.ArgumentParser(parents=[parser], conflict_handler='resolve', add_help=False)
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--data_dir', type=str)
parser.add_argument('--lmdb', action='store_true')
parser.add_argument('--backbone_type', type=str, default='imagenet')
parser.add_argument('--base_encoder', type=str, default='resnet18')
parser.add_argument('--ckpt_path', type=str, default=None)
parser.add_argument('--max_epochs', type=int, default=100)
parser.add_argument('--train_frac', type=float, default=1)
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, currentdir)


INITIAL_DATASET_PATH = "/home/francesco/Desktop/colomba_dataset"
FORMATTED_DATASET_PATH = "/home/francesco/Desktop/formatted_colombaset"

datascanner = ds.DatasetScanner(
        master_folder_path=INITIAL_DATASET_PATH,
        log_file_path="Log/master_folder_log_colombaset.csv",
        validation_file_path=None,
        dataset="colombaset",
        verbose=1
    )

datascanner.scan_master_folder()
datascanner.log_header_to_file(header="act_id,pre,post,mask")
datascanner.log_to_file()

dataformatter = df.DatasetFormatter(
        master_folder_path=FORMATTED_DATASET_PATH,
        log_file_path="Log/",
        log_filename="master_folder_log_colombaset.csv",
        master_dict_path="/home/francesco/Desktop/adsp/Log/",
        master_dict_filename="master_dict.json",
        tile_height=512,
        tile_width=512,
        thr_pixels=112,
        use_pre=True,
        dataset="colombaset",
        verbose=1
)

dataformatter.tiling()

ds = image_dataset.ImageDataset(
        formatted_folder_path="/home/francesco/Desktop/formatted_colombaset",
        log_folder="Log",
        master_dict="master_dict.json",
        transformations=None,
        use_pre=False,
        verbose=1,
        specific_indeces=None,
        return_path=False
)

ds._load_tiles()
##### FIN QUI TUTTO OK, ADESSO TOCCA ALLA RETE #####

# nnet = lhnet.LHNet(
#         encoder="resnet50",
#         encoder_weights="imagenet",
#         in_channels=3,
#         classes=2
# )
# model, preprocess_input = nnet._get_model_and_input_preprocessing()

# criterion = {
#     "dice": DiceLoss(),
#     "iou": IoULoss(),
#     "bce": nn.BCEWithLogitsLoss()
# }

# num_epochs = 15
# batch_size = 10
# num_workers = 1
# validation_set_size = 0.2
# random_state = 777
# train_transform_f = None
# valid_transform_f = None
# learning_rate = 1e-3
# encoder_learning_rate = 5e-4
# layerwise_params_weight_decay = 3e-5
# adam_weight_decay = 3e-4
# fp16 = None
# filepath = "Log"

# shifu = lhtrainer.Trainer(
#         model = model,
#         transformer = OptimusPrime(tile_dimension=512),
#         criterion=criterion,
#         num_epochs=num_epochs,
#         batch_size=batch_size,
#         num_workers=num_workers,
#         validation_set_size=validation_set_size,
#         random_state=random_state,
#         train_transform_f=train_transform_f,
#         valid_transform_f=valid_transform_f,
#         learning_rate=learning_rate,
#         encoder_learning_rate=encoder_learning_rate,
#         layerwise_params_weight_decay=layerwise_params_weight_decay,
#         adam_weight_decay=adam_weight_decay,
#         fp16=fp16,
#         filepath=filepath
# )

# loaders = shifu._get_loaders(num_tiles=len(ds.post_tiles))

# shifu._train(loaders=loaders)

range_indices = len(ds.post_tiles)
indices = np.arange(range_indices)
train_indices, valid_indices = train_test_split(indices, train_size=.8, test_size=.2, random_state=777, shuffle=True)

datamodule = ArgonetDataModule(
        train_indices=train_indices,
        valid_indices=valid_indices,
        data_dir=None,
        lmdb=False,
        batch_size=10,
        num_workers=1,
        train_frac=80
)
model = MocoV2.load_from_checkpoint("/home/francesco/Desktop/seco_resnet50_100k.ckpt")
emb_dim = model.mlp_dim
backbone = deepcopy(model.encoder_q)

model = SSLFineTuner(
backbone=backbone,
in_features=emb_dim,
num_classes=datamodule.num_classes,
hidden_dim=None)

model.example_input_array = torch.zeros((1, 3, 128, 128))

trainer = Trainer.from_argparse_args(
args,
logger=None,
checkpoint_callback=None,
weights_summary='full',
check_val_every_n_epoch=10
)
trainer.fit(model, datamodule=datamodule)

