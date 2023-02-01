from Preprocessing import datasetformatter as dataset_formatter
from Preprocessing import datasetscanner as dataset_scanner
from Preprocessing import colombadataset, effisdataset
from Utils import lhtransformer
from Utils.light_module import UNetModule
from Utils.utils import seed_worker, Configurator

from torch.utils.data import DataLoader
import torch
import segmentation_models_pytorch as smp
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import random
import numpy as np
from termcolor import colored

INITIAL_DATASET_PATH = "/mnt/data1/adsp_data/effis"

if __name__ == "__main__":
    print(colored("Training a UNet over Effis and testing it on EMS.", "green"))
    configurator = Configurator(filepath="config", filename="config.json")
    config = configurator.get_config()
    EFFIS_PATH = config["EFFIS_PATH"]
    COMPLETE_EMS_PATH = config["COMPLETE_EMS_PATH"]
    FORMATTED_COMPLETE_EMS_PATH = config["FORMATTED_COMPLETE_EMS_PATH"]
    random.seed(51996)
    np.random.seed(51996)
    torch.manual_seed(51996)
    g = torch.Generator()
    g.manual_seed(51996)
    
    my_transformer = lhtransformer.OptimusPrime()
    train_transforms = my_transformer.compose([
        my_transformer.shift_scale_rotate(),
        my_transformer.post_transforms_bigearthnet()
    ])
    test_transforms = my_transformer.compose([
        my_transformer.post_transforms_bigearthnet()
    ])

    datascanner = dataset_scanner.DatasetScanner(
    master_folder_path=EFFIS_PATH,
    log_file_path="",
    validation_file_path=None,
    dataset="sub-effis",
    verbose=1
    )
    datascanner.log_to_file()

    effis_train = effisdataset.EffisDataset(
        log_folder="Log/",
        master_dict = "master_dict_train_effis.json",
        transformations = None
    )
    effis_train._load_tiles()

    effis_val = effisdataset.EffisDataset(
        log_folder="Log/",
        master_dict = "master_dict_val_effis.json",
        transformations = None
    )
    effis_val._load_tiles()

    #### COLOMBASET ###
    ds_colomba = dataset_scanner.DatasetScanner(
        master_folder_path=COMPLETE_EMS_PATH,
        log_file_path="Log/master_folder_log_complete_colombaset.csv",
        validation_file_path=None,
        dataset="colombaset",
        verbose=1
    )

    ds_colomba.scan_master_folder()
    ds_colomba.log_header_to_file(header="act_id,pre,post,mask")
    ds_colomba.log_to_file()

    dataformatter = dataset_formatter.DatasetFormatter(
        master_folder_path=FORMATTED_COMPLETE_EMS_PATH,
        log_file_path="Log/",
        log_filename="master_folder_log_complete_colombaset.csv",
        master_dict_path="Log/",
        master_dict_filename="master_dict_complete_colombaset.json",
        tile_height=512,
        tile_width=512,
        thr_pixels=112,
        use_pre=True,
        dataset="colombaset",
        verbose=1
    )

    dataformatter.tiling()

    ds = colombadataset.ColombaDataset(
        model_type="ben",
        formatted_folder_path=FORMATTED_COMPLETE_EMS_PATH,
        log_folder="Log",
        master_dict="master_dict.json",
        transformations=test_transforms,
        specific_indeces=None,
        return_path=False
    )

    ds._load_tiles()

    train_loader = DataLoader(effis_train, batch_size=4, shuffle=True, num_workers=10, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(effis_val, batch_size=4, shuffle=False, num_workers=10, worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=10, persistent_workers=True, worker_init_fn=seed_worker, generator=g)

    tb_logger = TensorBoardLogger(save_dir="logs/")
    model = smp.Unet(encoder_name="resnet50", in_channels=12, encoder_weights=None)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3.0))
    module = UNetModule(model=model, criterion=criterion, learning_rate=1e-4)
    logger = TensorBoardLogger("tb_logs", name="effis_vanilla_net")
    trainer = Trainer(max_epochs=5, accelerator="gpu", devices=1, num_nodes=1, logger=logger)
    trainer.fit(model=module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model=module, dataloaders=test_loader)
    print("Saving model...")
    torch.save(model.state_dict(), "models/trained_models/effis.pth")