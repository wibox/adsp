from Preprocessing import dataloader as dl
from nnet.argoNet import argoNET
from nnet.doveNet import unet

import argparse
import os
import sys

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
        help="Dataset to be used either for training, testing or validation. Use colombaset/..."
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
    args = parser.parse_args()

    # Inserire la logica di inizializzazione degli oggetti
        # nn logic
    if args.net_type == "argo":
        #use argonet
        net = argoNET.UNet()
        seg_enc = argoNET.SegmentationEncoder()
    elif args.net_type == "dove":
        #use colomba nn
        net = unet.UNet()
    else:
        raise Exception("Invalid Neural Network model selected, please use python3 interface.py --help")
    ### QUI INIZIA IL MAIN ###
        # General structures initialization
    ds = dl.DatasetScanner(
        master_folder_path="/hdd/effis-wildfire",
        log_file_path=os.join.path("/logs", "act_id.csv")
    )

    df = dl.DatasetFormatter(

    )

    # train_dataset = SatelliteDataset(self.master_folder, self.mask_intervals, self.mask_one_hot, self.height, self.width, self.product_list, self.mode, self.filter_validity_mask, self.train_transforms, self.process_dict, self.csv_path, train_set, self.ignore_list, self.mask_filtering, self.only_burnt, mask_postfix=mask_postfix)
        # here we decide the logic for the selected op mode.
    if args.op == "train":
        pass
    elif args.op == "test":
        pass
    elif args.op == "val":
        pass
    # Inizializzare i vari dataset con i vari split (per il dataset che abbiamo scelto dal parser)
    # Inizializzare la rete scelta (dal parser)
    # Computare le metriche in output