from Preprocessing import datasetformatter as df
from Preprocessing import datasetscanner as ds
from Preprocessing import imagedataset as image_dataset

from Trainer import lhtrainer
from Nnet.lhNET import lhnet

from Utils import lhtransformer, outputformatter

from sklearn.model_selection import train_test_split
import numpy as np

from segmentation_models_pytorch import utils
import torch
from torch.utils.data import DataLoader

import os


INITIAL_DATASET_PATH = "/mnt/data1/adsp_data/colomba_dataset"
FORMATTED_DATASET_PATH = "/mnt/data1/adsp_data/formatted_colombaset"

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
        master_dict_path="Log/",
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
        formatted_folder_path=FORMATTED_DATASET_PATH,
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

nnet = lhnet.LHNet(
        encoder="resnet50",
        in_channels=10,
        classes=2
)
_, model = nnet._create_model()

indices = np.arange(len(ds.post_tiles))
train_indices, val_indices = train_test_split(indices, test_size=0.8, train_size=0.2, shuffle=True, random_state=777)

mean = [
    0.12375696117681859,
    0.1092774636368323,
    0.1010855203267882,
    0.1142398616114001,
    0.1592656692023089,
    0.18147236008771792,
    0.1745740312291377,
    0.19501607349635292,
    0.15428468872076637,
    0.10905050699570007,
]

std = [
    0.03958795985905458,
    0.047778262752410296,
    0.06636616706371974,
    0.06358874912497474,
    0.07744387147984592,
    0.09101635085921553,
    0.09218466562387101,
    0.10164581233948201,
    0.09991773043519253,
    0.08780632509122865
]

my_transformer = lhtransformer.OptimusPrime(
        mean=mean,
        std=std
)

train_transforms = my_transformer.compose([
    my_transformer.flip(),
    my_transformer.rotate(),
#     my_transformer.color_jitter(brigthness=.2, contrast=.2, saturation=.2, hue=.2),
    my_transformer.channel_shuffle(),
    my_transformer.gauss_noise(var_limit=(1, 10), mean=0),
    my_transformer.post_transforms()
])
valid_transforms = my_transformer.compose([my_transformer.post_transforms()])

train_ds = image_dataset.ImageDataset(
        formatted_folder_path=FORMATTED_DATASET_PATH,
        log_folder="Log",
        master_dict="master_dict.json",
        use_pre=False,
        verbose=1,
        specific_indeces=train_indices,
        return_path=False,
        transformations=train_transforms
)

val_ds = image_dataset.ImageDataset(
        formatted_folder_path=FORMATTED_DATASET_PATH,
        log_folder="Log",
        master_dict="master_dict.json",
        use_pre=False,
        verbose=1,
        specific_indeces=val_indices,
        return_path=False,
        transformations=valid_transforms
)

epochs = 5
device = 'cuda'
loss = utils.losses.DiceLoss()
optimizer = torch.optim.Adam(
        [dict(params=model.parameters(), lr=1e-4)]
)
batch_size = 10
num_workers = 1
metrics = [
        utils.metrics.IoU(threshold=.5)
]

shifu = lhtrainer.Trainer(
        model=model,
        training_dataset=train_ds,
        validation_dataset=val_ds,
        epochs=epochs,
        device=device,
        loss=loss,
        optimizer=optimizer,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        metrics=metrics
)

shifu._initialize()
shifu._train()

# load best model found
# if os.path.exists("./best_model.pth"):
#         best_model = torch.load("./best_model.pth", map_location=device)

# creating test dataset
test_indices = [1, 2, 3]
test_ds = image_dataset.ImageDataset(
        formatted_folder_path=FORMATTED_DATASET_PATH,
        log_folder="Log",
        master_dict="master_dict.json",
        transformations=None,
        use_pre=False,
        verbose=1,
        specific_indeces=test_indices,
        return_path=False
)
test_ds._load_tiles()

output_formatter = outputformatter.OutputFormatter(
        device=device,
        filepath="/mnt/data1/adsp_data",
        test_ds=test_ds,
        best_model_path="/mnt/data1/adsp_data/best_model.pth",
        verbose=1
)

output_formatter._initialize()
output_formatter._compute_output()