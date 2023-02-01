from Preprocessing import datasetformatter as dataset_formatter
from Preprocessing import datasetscanner as dataset_scanner
from Preprocessing import colombadataset, effisdataset
from Utils.custom_parser import custom_parser
from Utils.lhtransformer import OptimusPrime
from Utils.errors import *
from Utils.utils import Configurator, freeze_encoder, seed_worker
from Utils.light_module import UNetModule

import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from termcolor import colored

import random
import numpy as np

if __name__ == "__main__":
    ### GENERAL INITIALIZATIONS
    args = custom_parser()
    configurator = Configurator(filepath="config", filename="config.json")
    config = configurator.get_config()
    SUB_EMS_PATH = config["EMS_DATASET_PATH"]
    FORMATTED_SUB_EMS_PATH = config["FORMATTED_EMS_PATH"]
    TEST_SUB_EMS_PATH = config["TEST_EMS_PATH"]
    FORMATTED_TEST_SUB_EMS_PATH = config["FORMATTED_TEST_EMS_PATH"]
    EFFIS_PATH = config["EFFIS_PATH"]
    COMPLETE_EMS_PATH = config["COMPLETE_EMS_PATH"]
    FORMATTED_COMPLETE_EMS_PATH = config["FORMATTED_COMPLETE_EMS_PATH"]
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)

    my_transformer = OptimusPrime()
    train_transforms = my_transformer.compose([
        my_transformer.shift_scale_rotate(),
        my_transformer.post_transforms_bigearthnet()
    ])
    test_transforms = my_transformer.compose([
        my_transformer.post_transforms_bigearthnet()
    ])

    if args.dataset == "ben" or args.dataset == "vanilla" or args.dataset == "effis":
        channels=12
    else:
        channels=10

    ###
    if args.dataset == "ben"  or args.dataset == "imagenet" or  args.dataset == "vanilla":
        datascanner = dataset_scanner.DatasetScanner(
            master_folder_path=SUB_EMS_PATH,
            log_file_path="Log/master_folder_log_colombaset.csv",
            validation_file_path=None,
            dataset="colombaset",
            verbose=1
        )

        datascanner.scan_master_folder()
        datascanner.log_header_to_file(header="act_id,pre,post,mask")
        datascanner.log_to_file()

        dataformatter = dataset_formatter.DatasetFormatter(
            master_folder_path=FORMATTED_SUB_EMS_PATH,
            log_file_path="Log/",
            log_filename="master_folder_log_colombaset.csv",
            master_dict_path="Log/",
            master_dict_filename="master_dict.json",
            tile_height=args.tile_size,
            tile_width=args.tile_size,
            thr_pixels=args.threshold,
            use_pre=args.use_pre,
            dataset="colombaset",
            verbose=1
        )

        dataformatter.tiling()

        ds = colombadataset.ColombaDataset(
            model_type="ben",
            formatted_folder_path=FORMATTED_SUB_EMS_PATH,
            log_folder="Log",
            master_dict="master_dict.json",
            transformations=train_transforms,
            specific_indeces=None,
            return_path=False
        )

        ds._load_tiles()

        test_datascanner = dataset_scanner.DatasetScanner(
            master_folder_path=TEST_SUB_EMS_PATH,
            log_file_path="Log/test_master_folder_log_colombaset.csv",
            validation_file_path=None,
            dataset="colombaset",
            verbose=1
        )

        test_datascanner.scan_master_folder()
        test_datascanner.log_header_to_file(header="act_id,pre,post,mask")
        test_datascanner.log_to_file()

        test_dataformatter = dataset_formatter.DatasetFormatter(
            master_folder_path=FORMATTED_TEST_SUB_EMS_PATH,
            log_file_path="Log/",
            log_filename="test_master_folder_log_colombaset.csv",
            master_dict_path="Log/",
            master_dict_filename="test_master_dict.json",
            tile_height=args.tile_size,
            tile_width=args.tile_size,
            thr_pixels=args.threshold,
            use_pre=args.use_pre,
            dataset="colombaset",
            verbose=1
        )

        test_dataformatter.tiling()

        test_ds = colombadataset.ColombaDataset(
            model_type="ben",
            formatted_folder_path=FORMATTED_TEST_SUB_EMS_PATH,
            log_folder="Log",
            master_dict="test_master_dict.json",
            transformations=test_transforms,
            specific_indeces=None,
            return_path=False
        )
        test_ds._load_tiles()

        train_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, persistent_workers=True, worker_init_fn=seed_worker, generator=g)

        tb_logger = TensorBoardLogger(save_dir="logs/")
        model = smp.Unet(encoder_name=args.encoder_name, in_channels=channels, encoder_weights=None, classes=1)
        if args.dataset == "ben":
            print(colored("Loading BigEarthNet checkpoint", "green"))
            model.encoder.load_state_dict(torch.load("models/checkpoints/checkpoint-30.pth.tar"), strict=False)
        elif args.dataset == "imagenet":
            print(colored("Loading ImageNet checkpoint", "green"))
            model.encoder.load_state_dict(torch.load("models/checkpoints/10bandsimagenet.pth"), strict=False)
        elif args.dataset == "vanilla":
            print(colored("Training from scratch selected. No checkpoints are going to be loaded.", "red"))
        if args.freeze_encoder:
            freeze_encoder(model=model)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3.0))
        module = UNetModule(model=model, criterion=criterion, learning_rate=1e-4)
        logger = TensorBoardLogger("tb_logs", name="ben_net")
        trainer = Trainer(max_epochs=args.epochs, accelerator="gpu", devices=1, num_nodes=1, logger=logger)
        trainer.fit(model=module, train_dataloaders=train_loader)
        trainer.test(model=module, dataloaders=test_loader)
        if args.save:
            print("Saving model...")
            torch.save(model.state_dict(), f"models/trained_models/{args.dataset}.pth")
    elif args.datset == "effis":
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
            tile_height=args.pixel_size,
            tile_width=args.pixel_size,
            thr_pixels=args.threshold,
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

        train_loader = DataLoader(effis_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g)
        val_loader = DataLoader(effis_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g)
        test_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, persistent_workers=True, worker_init_fn=seed_worker, generator=g)

        tb_logger = TensorBoardLogger(save_dir="logs/")
        model = smp.Unet(encoder_name=args.encoder_name, in_channels=channels, encoder_weights=None, classes=1)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3.0))
        module = UNetModule(model=model, criterion=criterion, learning_rate=1e-4)
        logger = TensorBoardLogger("tb_logs", name="effis_vanilla_net")
        trainer = Trainer(max_epochs=args.epochs, accelerator="gpu", devices=1, num_nodes=1, logger=logger)
        trainer.fit(model=module, train_dataloaders=train_loader, val_dataloaders=val_loader)
        trainer.test(model=module, dataloaders=test_loader)
        if args.save:
            print("Saving model...")
            torch.save(model.state_dict(), f"models/trained_models/{args.dataset}.pth")