from Preprocessing import datasetformatter as df
from Preprocessing import datasetscanner as ds
from Preprocessing import imagedataset as image_dataset

from Trainer import trainer

from Nnet.gafesNET import unet, encoder, decoder

from torch import nn

from copy import deepcopy
import os
import sys
import inspect

from torchvision.transforms import transforms

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, currentdir)

from Utils.transforms import *

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

# adesso provo se funziona l'imagedatset_class
train_ds = image_dataset.ImageDataset(
        formatted_folder_path="/home/francesco/Desktop/formatted_colombaset",
        log_folder="Log",
        master_dict="master_dict.json",
        transform=None,
        use_pre=False,
        verbose=1
)

train_ds._load_tiles()

# model = moco.MocoV2.load_from_checkpoint("/home/francesco/Desktop/seco_resnet50_1m.ckpt")
# backbone = deepcopy(model.encoder_q)
# net = argoNET.get_segmentation_model(
#         backbone=nn.Sequential(*list(backbone.children())[:-1], nn.Flatten()),
#         feature_indices=(4, 3, 2),
#         feature_channels=(64, 64, 128, 256, 512)
# )

net = unet.Unet()
net_args = []

transformations = transforms.Compose([
        ToTensor()
])

myTrainer = trainer.Trainer(
        device='cuda',
        net=net,
        net_args=net_args,
        batch_size=20,
        dataset=train_ds,
        lr=1e-5,
        epoch=10,
        act_function=nn.ReLU,
        transformations=transformations
)

myTrainer._train()

