from Preprocessing import datasetformatter as df
from Preprocessing import datasetscanner as ds
from Preprocessing import imagedataset as id
from Nnet.argoNet import argoNET, moco
from Nnet.doveNet import unet
from Utils.custom_parser import custom_parser
from Utils.errors import *


import argparse
from copy import deepcopy

import torch.nn as nn

if __name__ == "__main__":
    # Parser configuration
    args = custom_parser()
    
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
            raise WrongArgument(argument="--activation-function")

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
        raise WrongArgument(argument="--net-type")

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
        dataformatter.tiling()
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
        raise WrongArgument(argument="--dataset")

    if args.op == "train":
        pass
    elif args.op == "test":
        pass
    elif args.op == "val":
        pass
    else:
        raise WrongArgument(argument="--op")
    # Computare le metriche in output