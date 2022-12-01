from Preprocessing import datasetformatter as df
from Preprocessing import datasetscanner as ds
from Preprocessing import imagedataset as id
from Nnet.argoNet import argoNET, moco
from Nnet.doveNet import unet

import argparse
from copy import deepcopy

import torch.nn as nn

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
        default="colombaset",
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
    
        # nn logic
    if args.net_type == "argonet":
        #use argonet
        model = moco.MocoV2.load_from_checkpoint(args.ckpt_path)
        backbone = deepcopy(model.encoder_q)
        net = argoNET.get_segmentation_model(
            backbone=nn.Sequential(*list(backbone.children())[:-1], nn.Flatten()),
            feature_indices=(4, 3, 2),
            feature_channels=(64, 64, 128, 256, 512)
        )
    elif args.net_type == "dovenet":
        #use colomba nn
        if args.activation_function.lower() == "relu":
            act=nn.ReLU
        elif args.activation_function.lower() == "tanh":
            act=nn.Tanh
        elif args.activation_function.lower() == "leakyrelu":
            act=nn.LeakyReLU
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

        # General structures initialization
    datascanner = ds.DatasetScanner(
        master_folder_path=args.master_folder,
        log_file_path=f"log/master_folder_log_{args.dataset}.csv",
        validation_file_path=args.validation_file,
        dataset=args.dataset,
        verbose=1
    )

    dataformatter = df.DatasetFormatter(
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

    imagedataset = id.ImageDataset(
        processed_master_folder_path=args.master_folder,
        master_dict_path="log/",
        tile_width=args.tile_size,
        tile_height=args.tile_size,
        transform=None,
        use_pre=args.use_pre,
        verbose=1
    )
        # actions to be performed by up-initialized classes for each dataset
    
    if args.dataset == "colombaset":
        # logging actions
        datascanner.scan_master_folder()
        datascanner.log_header_to_file(header="act_id,pre,post,mask")
        datascanner.log_to_file()
        # formatting actions

        # dataset initialisation
    elif args.dataset == "full-effis":
        # logging actions

        # formatting actions

        # dataset initialisation
        pass
    elif args.dataset == "sub-effis":
        # logging actions

        # formatting actions

        # dataset initialisation
        pass
    else:
        raise Exception("Dataset not found! Please use python3 interface.py --help")

    if args.op == "train":
        pass
    elif args.op == "test":
        pass
    elif args.op == "val":
        pass
    # Computare le metriche in output