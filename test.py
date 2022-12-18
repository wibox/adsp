from Preprocessing import datasetformatter as df
from Preprocessing import datasetscanner as ds
from Preprocessing import imagedataset as image_dataset

from Trainer import trainer
from Nnet.lhNET import lhnet
from Utils.albu_transformer import OptimusPrime

from torch import nn
from catalyst.contrib.nn import DiceLoss, IoULoss

from copy import deepcopy
import os
import sys
import inspect


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
        verbose=1
)

ds._load_tiles()
print("Total number of tiles: ", ds.post_tiles)
##### FIN QUI TUTTO OK, ADESSO TOCCA ALLA RETE! #####
model, preprocess_input = lhnet._get_model_and_input_preprocessing()

criterion = {
    "dice": DiceLoss(),
    "iou": IoULoss(),
    "bce": nn.BCEWithLogitsLoss()
}

num_epochs = 15
batch_size = 30
num_workers = 4
validation_set_size = 0.2
random_state = 777
train_transform_f = None
valid_transform_f = None
learning_rate = 1e-3
encoder_learning_rate = 5e-4
layerwise_params_weight_decay = 3e-5
adam_weight_decay = 3e-4
fp16 = None
filepath = "Log"

shifu = trainer.Trainer(
        model = model,
        transformer = OptimusPrime(tile_dimension=512),
        criterion=criterion,
        num_epochs=num_epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        validation_set_size=validation_set_size,
        random_state=random_state,
        train_transform_f=train_transform_f,
        valid_transform_f=valid_transform_f,
        learning_rate=learning_rate,
        encoder_learning_rate=encoder_learning_rate,
        layerwise_params_weight_decay=layerwise_params_weight_decay,
        adam_weight_decay=adam_weight_decay,
        fp16=fp16,
        filepath=filepath
)

loaders = shifu._get_loaders(num_tiles=len(ds.post_tiles), post_tiles=ds.post_tiles, mask_tiles=ds.mask_tiles)

shifu._train(loaders=loaders)

