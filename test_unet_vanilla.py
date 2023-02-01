from Preprocessing import datasetformatter as dataset_formatter
from Preprocessing import datasetscanner as dataset_scanner
from Preprocessing import colombadataset as image_dataset
from Utils import lhtransformer
from Utils.light_module import UNetModule
from Utils.utils import seed_worker, freeze_encoder, Configurator

from torch.utils.data import DataLoader
import torch
import segmentation_models_pytorch as smp
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import random
import numpy as np
from termcolor import colored

INITIAL_DATASET_PATH = "/mnt/data1/adsp_data/colomba_dataset"
FORMATTED_DATASET_PATH = "/mnt/data1/adsp_data/formatted_colombaset"
TEST_DATASET_PATH = "/mnt/data1/adsp_data/test_colombaset"
FORMATTED_TEST_DATASET_PATH = "/mnt/data1/adsp_data/formatted_test_colombaset"

if __name__ == "__main__":
    print(colored("Training and testing a UNet from scratch.", "green"))
    configurator = Configurator(filepath="config", filename="config.json")
    config = configurator.get_config()
    INITIAL_DATASET_PATH = config["EMS_DATASET_PATH"]
    FORMATTED_DATASET_PATH = config["FORMATTED_EMS_PATH"]
    TEST_DATASET_PATH = config["TEST_EMS_PATH"]
    FORMATTED_TEST_DATASET_PATH = config["FORMATTED_TEST_EMS_PATH"]
    random.seed(51996)
    np.random.seed(51996)
    torch.manual_seed(51996)
    g = torch.Generator()
    g.manual_seed(51996)

    my_transformer = lhtransformer.OptimusPrime()
    train_transforms = my_transformer.compose([
        my_transformer.shift_scale_rotate(),
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
        thr_pixels=112,
        use_pre=True,
        dataset="colombaset",
        verbose=1
    )

    dataformatter.tiling()

    ds = image_dataset.ColombaDataset(
        model_type="vanilla",
        formatted_folder_path=FORMATTED_DATASET_PATH,
        log_folder="Log",
        master_dict="master_dict.json",
        transformations=train_transforms,
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
        thr_pixels=112,
        use_pre=True,
        dataset="colombaset",
        verbose=1
    )

    test_dataformatter.tiling()

    test_ds = image_dataset.ColombaDataset(
        model_type="vanilla",
        formatted_folder_path=FORMATTED_TEST_DATASET_PATH,
        log_folder="Log",
        master_dict="test_master_dict.json",
        transformations=test_transforms,
        specific_indeces=None,
        return_path=False
    )
    test_ds._load_tiles()

    train_loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=10, worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=10, persistent_workers=True, worker_init_fn=seed_worker)

    model = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=12, classes=1)
    tb_logger = TensorBoardLogger(save_dir="logs/")
    criterion = torch.nn.BCEWithLogitsLoss()
    module = UNetModule(model=model, criterion=criterion, learning_rate=1e-4)
    logger = TensorBoardLogger("tb_logs", name="vanilla_net")
    trainer = Trainer(max_epochs=3, accelerator="gpu", devices=1, num_nodes=1, logger=logger)
    trainer.fit(model=module, train_dataloaders=train_loader)
    trainer.test(model=module, dataloaders=test_loader)
    print("Saving model...")
    torch.save(model.state_dict(), "models/trained_models/vanilla.pth")
