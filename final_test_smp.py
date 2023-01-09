from Preprocessing import datasetformatter as dataset_formatter
from Preprocessing import datasetscanner as dataset_scanner
from Preprocessing import imagedataset as image_dataset

from Utils import lhtransformer

from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything

INITIAL_DATASET_PATH = "/mnt/data1/adsp_data/colomba_dataset"
FORMATTED_DATASET_PATH = "/mnt/data1/adsp_data/formatted_colombaset"
TEST_DATASET_PATH = "/mnt/data1/adsp_data/test_colombaset"
FORMATTED_TEST_DATASET_PATH = "/mnt/data1/adsp_data/formatted_test_colombaset"

if __name__ == "__main__":
    seed_everything(777)
    my_transformer = lhtransformer.OptimusPrime(
            mean=None,
            std=None
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
            transformations=None,
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

    test_transforms = my_transformer.compose([
        my_transformer.post_transforms()
    ])

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

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4)
    #val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=4, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=4, persistent_workers=True)

    # from Nnet.unet.unet import UNET
    from Utils.light_module import UNetModule
    import torch
    import segmentation_models_pytorch as smp

    # model = UNET()

    from pytorch_lightning import Trainer
    from Utils.light_module import UNetModule
    from pytorch_lightning.loggers import TensorBoardLogger
    
    tb_logger = TensorBoardLogger(save_dir="logs/")
    model = smp.Unet(encoder_name="resnet50", in_channels=10, encoder_weights=None)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(5.0))
    module = UNetModule(model=model, criterion=criterion, learning_rate=1e-4)
    trainer = Trainer(max_epochs=15, accelerator="gpu", devices=1, num_nodes=1)
    trainer.fit(model=module, train_dataloaders=train_loader)
    trainer.test(model=module, dataloaders=test_loader)