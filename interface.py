from Preprocessing import dataloader as dl
from nnet.argoNet import argoNET
from nnet.doveNet import unet

import argparse

if __name__ == "__main__":
    # Parser configuration
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--net-type",
        type=str,
        default="argonet",
        help="Type of neural network to be used: argonet/dovenet."
        )
    parser.add_argument(
        "--dataset",
        type=str,
        defualt="colombaset",
        help="Dataset to be used either for training, testing or validation. Use colombaset/full-effis/sub-effis"
    )
    parser.add_argument(
        "--master-folder",
        type=str,
        default=None,
        help="Path to data folder, to be specified according to position of <dataset>."
    )
    parser.add_argument(
        "--validation-file",
        type=str,
        default=None,
        help="Validation.json used to filter useful activations (those which contain pre, post and mask)."
    )
    parser.add_argument(
        "--log-path", 
        type=str, 
        default="log/", 
        help="Location of general log files."
        )
    parser.add_argument(
        "--tile-size", 
        type=int, 
        default=512, 
        help="Size of tiles to be created."
        )
    parser.add_argument(
        "--threshold",
        type=int,
        default=112,
        help="Specific threshold used to perform overlapping while tiling."
    )
    parser.add_argument(
        "--use-pre", 
        type=bool, 
        default=False, 
        help="Either to use or not use the pre-fire of each activation in the provided dataset: True/False."
        )
    parser.add_argument(
        "--op",
        type=str,
        default="train",
        help="Train, test or validate a specific dataset: use train/test/val."
    )
    parser.add_argument(
        "--activation-function",
        type=str,
        default="ReLU",
        help="Activation function to be used through torch utilities: ReLU, Tanh, LeakyReLU"
    )
    args = parser.parse_args()

    # Inserire la logica di inizializzazione degli oggetti
        # nn logic
    if args.net_type == "argo":
        #use argonet
        net = argoNET.get_segmentation_model(
            backbone="path",
            feature_indices=None,
            feature_channels=12
        )
    elif args.net_type == "dove":
        #use colomba nn
        if args.activation_function.lower() == "relu":
            act=None
        elif args.activation_function.lower() == "tanh":
            act=None
        elif args.activation_function.lower() == "leakyrelu":
            act=None
        else:
            raise Exception("Invalid activation function selected, please select one among: ReLU, LeakyReLU, Tanh")

        net = unet.UNet(
            n_channels=12,
            n_classes=2,
            act=act,
            first_ch_out=64,
            alpha=1.0,
            dropout=True,
            gradcam=False
        )
    else:
        raise Exception("Invalid Neural Network model selected, please use python3 interface.py --help")
    ### QUI INIZIA IL MAIN ###
        # General structures initialization
    ds = dl.DatasetScanner(
        master_folder_path=args.master_folder,
        log_file_path="log/",
        validation_file_path=args.validation_file,
        dataset=args.dataset,
        verbose=1
    )

    df = dl.DatasetFormatter(
        master_folder_path=args.master_folder,
        log_file_path="log/",
        log_filename="activations.csv",
        master_dict_path="log/",
        tile_height=args.tile_size,
        tile_width=args.tile_size,
        thr_pixels=args.threshold,
        use_pre=args.use_pre,
        dataset=args.dataset,
        verbose=1
    )

    id = dl.ImageDataset(
        processed_master_folder_path=args.master_folder,
        master_dict_path="log/",
        tile_width=args.tile_size,
        tile_height=args.tile_size,
        transform=None,
        use_pre=args.use_pre,
        verbose=1
    )

    # train_dataset = SatelliteDataset(self.master_folder, self.mask_intervals, self.mask_one_hot, self.height, self.width, self.product_list, self.mode, self.filter_validity_mask, self.train_transforms, self.process_dict, self.csv_path, train_set, self.ignore_list, self.mask_filtering, self.only_burnt, mask_postfix=mask_postfix)
        # here we decide the logic for the selected op mode.
    if args.op == "train":
        # Inizializzare i vari dataset con i vari split (per il dataset che abbiamo scelto dal parser)
        pass
    elif args.op == "test":
        # Inizializzare i vari dataset con i vari split (per il dataset che abbiamo scelto dal parser)
        pass
    elif args.op == "val":
        # Inizializzare i vari dataset con i vari split (per il dataset che abbiamo scelto dal parser)
        pass
    # Computare le metriche in output