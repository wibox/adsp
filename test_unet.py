from Preprocessing import datasetformatter as dataset_formatter
from Preprocessing import datasetscanner as dataset_scanner
from Preprocessing import imagedataset as image_dataset

from Utils import lhtransformer

from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import DataLoader

INITIAL_DATASET_PATH = "/mnt/data1/adsp_data/colomba_dataset"
FORMATTED_DATASET_PATH = "/mnt/data1/adsp_data/formatted_colombaset"
TEST_DATASET_PATH = "/mnt/data1/adsp_data/test_colombaset"
FORMATTED_TEST_DATASET_PATH = "/mnt/data1/adsp_data/formatted_test_colombaset"

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

datascanner = dataset_scanner.DatasetScanner(
        master_folder_path=INITIAL_DATASET_PATH,
        log_file_path="Log/master_folder_log_colombaset.csv",
        validation_file_path=None,
        dataset="colombaset",
        verbose=1
    )

datascanner.scan_master_folder()
datascanner.log_header_to_file(header="act_id,pre,post,mask")
datascanner.log_to_file()

dataformatter = dataset_formatter.DatasetFormatter(
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

test_transforms = my_transformer.compose([
    my_transformer.post_transforms()
])

test_datascanner = dataset_scanner.DatasetScanner(
        master_folder_path=TEST_DATASET_PATH,
        log_file_path="Log/test_master_folder_log_colombaset.csv",
        validation_file_path=None,
        dataset="colombaset",
        verbose=1
    )

test_datascanner.scan_master_folder()
test_datascanner.log_header_to_file(header="act_id,pre,post,mask")
test_datascanner.log_to_file()

test_dataformatter = dataset_formatter.DatasetFormatter(
        master_folder_path=FORMATTED_TEST_DATASET_PATH,
        log_file_path="Log/",
        log_filename="test_master_folder_log_colombaset.csv",
        master_dict_path="Log/",
        master_dict_filename="test_master_dict.json",
        tile_height=512,
        tile_width=512,
        thr_pixels=112,
        use_pre=True,
        dataset="colombaset",
        verbose=1
)

test_dataformatter.tiling()


indices = np.arange(len(ds.post_tiles))
train_indices, val_indices = train_test_split(indices, test_size=0.8, train_size=0.2, shuffle=True, random_state=777)

train_transforms = my_transformer.compose([
    my_transformer.flip(),
    my_transformer.fixed_rotate(),
    my_transformer.random_brightness_contrast(),
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
train_ds._load_tiles()

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
val_ds._load_tiles()

test_ds = image_dataset.ImageDataset(
        formatted_folder_path=FORMATTED_TEST_DATASET_PATH,
        log_folder="Log",
        master_dict="test_master_dict.json",
        transformations=test_transforms,
        use_pre=False,
        verbose=1,
        specific_indeces=None,
        return_path=False
)
test_ds._load_tiles()

train_loader = DataLoader(train_ds, batch_size=10, shuffle=True, num_workers=1)
val_loader = DataLoader(val_ds, batch_size=10, shuffle=True, num_workers=1)

from Nnet.unet.unet import UNET
from Trainer.unet_trainer import UnetTrainer
import torch

model = UNET()

shifu = UnetTrainer(
    model = model,
    optimizer = torch.optim.AdamW([dict(params=model.parameters(), lr=1e-4)]),
    loss = torch.nn.BCEWithLogitsLoss(),
    scaler = torch.cuda.amp.GradScaler(),
    epochs=15,
    batch_size=10,
    train_loader=train_loader,
    val_loader=val_loader,
    device='cuda',
    checkpoint_path="/mnt/data1/adsp_data/checkpoints",
    checkpoint_filename="best_model.pth.tar"
)

shifu.train()
