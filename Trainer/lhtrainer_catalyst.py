# OUR STUFF
from Utils.albu_transformer import OptimusPrime
from Preprocessing.imagedataset import ImageDataset
# MAIN MODULES
from catalyst import utils
from catalyst.contrib.nn import RAdam, Lookahead
from catalyst.dl import SupervisedRunner, DiceCallback, IouCallback, CriterionCallback, MetricAggregationCallback
from catalyst.contrib.callbacks import DrawMasksCallback
from torch.utils.data import DataLoader
from torch import optim
# ACCESSORIES MODULES
import numpy as np
from sklearn.model_selection import train_test_split
# NATIVE MODULES
from typing import *
import collections


class Trainer():
    def __init__(self,
                 model: Any = None,
                 transformer: OptimusPrime = None,
                 criterion: Dict[str, Any] = None,
                 num_epochs: int = 15,
                 batch_size: int = 30,
                 num_workers: int = 4,
                 validation_set_size: float = 0.2,
                 random_state: int = 777,  # for train test split
                 train_transform_f: Any = None,
                 valid_transform_f: Any = None,
                 learning_rate: float = 1e-3,
                 encoder_learning_rate: float = 5e-4,
                 layerwise_params_weight_decay: float = 3e-5,
                 adam_weight_decay: float = 3e-4,
                 fp16: float = None,
                 filepath: str = "Log"
                 ):
        self.model = model
        self.transformer = transformer
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.validation_set_size = validation_set_size
        self.random_state = random_state
        self.train_transform_f = train_transform_f
        self.valid_transform_f = valid_transform_f
        self.learning_rate = learning_rate,
        self.encoder_learning_rate = encoder_learning_rate
        self.layerwise_params_weight_decay = layerwise_params_weight_decay
        self.adam_weight_decay = adam_weight_decay
        self.fp16 = fp16
        self.filepath = filepath
        self.logdir = self.filepath
        self.device = utils.get_device()
        # training-useful objects
        self.layerwise_params = {
            "encoder*": dict(lr=self.encoder_learning_rate, weight_decay=self.layerwise_params_weight_decay)}
        self.model_params = utils.process_model_params(
            self.model, layerwise_params=self.layerwise_params)
        self.base_optimizer = RAdam(
            self.model_params, lr=self.learning_rate, weight_decay=self.adam_weight_decay)
        self.optimizer = Lookahead(self.base_optimizer)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=0.25, patience=2)
        # Tile Augmentation transformations
        self.train_transforms = self.transformer.compose([
            self.transformer.resize_transforms(),
            self.transformer.hard_transforms(),
            self.transformer.post_transforms()
        ])
        self.valid_transforms = self.transformer.compose([
            self.transformer.pre_transforms(),
            self.transformer.post_transforms()
        ])

    def _get_loaders(self, num_tiles: int = 0) -> Dict[str, DataLoader]:

        indices = np.arange(num_tiles)

        # Let's divide the data set into train and valid parts.
        train_indices, valid_indices = train_test_split(
            indices, train_size=1-self.validation_set_size, test_size=self.validation_set_size, random_state=self.random_state, shuffle=True)

        train_dataset = ImageDataset(
            formatted_folder_path="/home/francesco/Desktop/formatted_colombaset",
            log_folder="Log",
            master_dict="master_dict.json",
            transformations=None,
            use_pre=False,
            verbose=1,
            specific_indeces=train_indices,
            return_path=False
        )
        train_dataset._load_tiles()
        valid_dataset = ImageDataset(
            formatted_folder_path="/home/francesco/Desktop/formatted_colombaset",
            log_folder="Log",
            master_dict="master_dict.json",
            transformations=None,
            use_pre=False,
            verbose=1,
            specific_indeces=valid_indices,
            return_path=False
        )
        valid_dataset._load_tiles()

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
        )

        # And excpect to get an OrderedDict of loaders
        loaders = collections.OrderedDict()
        loaders["train"] = train_loader
        loaders["valid"] = valid_loader

        return loaders

    def _train(self, loaders: Dict[str, DataLoader] = None) -> None:
        runner = SupervisedRunner(
            device=self.device, input_key="image", input_target_key="mask")

        callbacks = [
            # Each criterion is calculated separately.
            CriterionCallback(
                input_key="mask",
                prefix="loss_dice",
                criterion_key="dice"
            ),
            CriterionCallback(
                input_key="mask",
                prefix="loss_iou",
                criterion_key="iou"
            ),
            CriterionCallback(
                input_key="mask",
                prefix="loss_bce",
                criterion_key="bce"
            ),

            # And only then we aggregate everything into one loss.
            MetricAggregationCallback(
                prefix="loss",
                mode="weighted_sum",  # can be "sum", "weighted_sum" or "mean"
                # because we want weighted sum, we need to add scale for each loss
                metrics={"loss_dice": 1.0, "loss_iou": 1.0, "loss_bce": 0.8},
            ),

            # metrics
            DiceCallback(input_key="mask"),
            IouCallback(input_key="mask"),
            # visualization
            DrawMasksCallback(output_key='logits',
                              input_image_key='image',
                              input_mask_key='mask',
                              summary_step=50
                              )
        ]

        runner.train(
            model=self.model,
            criterion=self.criterion,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            # our dataloaders
            loaders=loaders,
            # We can specify the callbacks list for the experiment;
            callbacks=callbacks,
            # path to save logs
            logdir=self.logdir,
            num_epochs=self.num_epochs,
            # save our best checkpoint by IoU metric
            main_metric="iou",
            # IoU needs to be maximized.
            minimize_metric=False,
            # for FP16. It uses the variable from the very first cell
            fp16=self.fp16,
            # prints train logs
            verbose=True,
        )
