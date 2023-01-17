from Preprocessing import datasetformatter as dataset_formatter
from Preprocessing import datasetscanner as dataset_scanner
from Preprocessing import imagedataset as image_dataset

from Utils import lhtransformer

from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything

from Utils.light_module import UNetModule
import torch
from pytorch_lightning import Trainer
from Utils.light_module import UNetModule
from pytorch_lightning.loggers import TensorBoardLogger
# from Nnet.unet.unet import UNET
import segmentation_models_pytorch as smp

INITIAL_DATASET_PATH = "/mnt/data1/adsp_data/colomba_dataset"
FORMATTED_DATASET_PATH = "/mnt/data1/adsp_data/formatted_colombaset"
TEST_DATASET_PATH = "/mnt/data1/adsp_data/test_colombaset"
FORMATTED_TEST_DATASET_PATH = "/mnt/data1/adsp_data/formatted_test_colombaset"

if __name__ == "__main__":
    seed_everything(51996)

    my_transformer = lhtransformer.OptimusPrime()
    train_transforms = my_transformer.compose([
    my_transformer.gauss_noise(var_limit=(250, 1250), mean=0),
    my_transformer.random_brightness_contrast(),
    my_transformer.fixed_rotate(),
    my_transformer.post_transforms_vanilla()
    ])
    test_transforms = my_transformer.compose([
    my_transformer.post_transforms_vanilla()
    ])

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
    thr_pixels=0,
    use_pre=True,
    dataset="colombaset",
    verbose=1
    )

    dataformatter.tiling()

    ds = image_dataset.ImageDataset(
    formatted_folder_path=FORMATTED_DATASET_PATH,
    log_folder="Log",
    master_dict="master_dict.json",
    transformations=train_transforms,
    use_pre=False,
    verbose=1,
    specific_indeces=None,
    return_path=False
    )

    ds._load_tiles()

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
    thr_pixels=0,
    use_pre=True,
    dataset="colombaset",
    verbose=1
    )

    test_dataformatter.tiling()

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

    train_loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=4, persistent_workers=True)
    model = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=12, classes=1, encoder_depth=4)
    tb_logger = TensorBoardLogger(save_dir="logs/")
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.0))
    module = UNetModule(model=model, criterion=criterion, learning_rate=1e-4)
    trainer = Trainer(max_epochs=5, accelerator="gpu", devices=1, num_nodes=1)
    trainer.fit(model=module, train_dataloaders=train_loader)
    trainer.test(model=module, dataloaders=test_loader)