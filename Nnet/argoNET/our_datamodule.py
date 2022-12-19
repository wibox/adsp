import os

from PIL import Image
from torchvision import transforms
from pytorch_lightning import LightningDataModule

from Preprocessing import imagedataset
from .data import random_subset, LMDBDataset, InfiniteDataLoader


class ArgonetDataModule(LightningDataModule):

    def __init__(self, train_indices, valid_indices, data_dir, bands=None, train_frac=None, val_frac=None, lmdb=False, batch_size=32, num_workers=16, seed=42):
        super().__init__()
        self.train_indices = train_indices
        self.valid_indices = valid_indices
        self.data_dir = data_dir
        self.bands = bands
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.lmdb = lmdb
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.train_dataset = None
        self.val_dataset = None

    @property
    def num_classes(self):
        return 19

    def setup(self, stage=None):
        train_transforms = self.train_transform() if self.train_transforms is None else self.train_transforms
        if self.lmdb:
            self.train_dataset = LMDBDataset(
                lmdb_file=os.path.join(self.data_dir, 'train.lmdb'),
                transform=train_transforms
            )
        else:
            train_dataset = imagedataset.ImageDataset(
                    formatted_folder_path="/home/francesco/Desktop/formatted_colombaset",
                    log_folder="Log",
                    master_dict="master_dict.json",
                    transformations=None,
                    use_pre=False,
                    verbose=1,
                    specific_indeces=self.train_indices,
                    return_path=False
            )

            train_dataset._load_tiles()
        if self.train_frac is not None and self.train_frac < 1:
            self.train_dataset = random_subset(self.train_dataset, self.train_frac, self.seed)

        val_transforms = self.val_transform() if self.val_transforms is None else self.val_transforms
        if self.lmdb:
            self.val_dataset = LMDBDataset(
                lmdb_file=os.path.join(self.data_dir, 'val.lmdb'),
                transform=val_transforms
            )
        else:
            self.val_dataset = imagedataset.ImageDataset(
                    formatted_folder_path="/home/francesco/Desktop/formatted_colombaset",
                    log_folder="Log",
                    master_dict="master_dict.json",
                    transformations=None,
                    use_pre=False,
                    verbose=1,
                    specific_indeces=self.valid_indices,
                    return_path=False
            )

            self.val_dataset._load_tiles()
        if self.val_frac is not None and self.val_frac < 1:
            self.val_dataset = random_subset(self.val_dataset, self.val_frac, self.seed)

    @staticmethod
    def train_transform():
        return transforms.Compose([
            transforms.Resize((128, 128), interpolation=Image.BICUBIC),
            transforms.ToTensor()
        ])

    @staticmethod
    def val_transform():
        return transforms.Compose([
            transforms.Resize((128, 128), interpolation=Image.BICUBIC),
            transforms.ToTensor()
        ])

    def train_dataloader(self):
        return InfiniteDataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self):
        return InfiniteDataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )