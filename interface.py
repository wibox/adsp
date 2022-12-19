from Preprocessing import datasetformatter as df
from Preprocessing import datasetscanner as ds
from Preprocessing import imagedataset as id
from Utils.custom_parser import custom_parser
from Utils.errors import *

if __name__ == "__main__":
    # Parser configuration
    args = custom_parser()
    
    # nn logic
    if args.net_type == "argonet":
        pass
    elif args.net_type == "dovenet":
        pass
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
        verbose=1,
        specific_indeces=None,
        return_path=False
    )
    
    # actions to be performed by up-initialized classes for each dataset
    if args.dataset == "colombaset":
        # logging actions
        datascanner.scan_master_folder()
        datascanner.log_header_to_file(header="act_id,pre,post,mask")
        datascanner.log_to_file()
        # formatting actions
        dataformatter.tiling()
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
        train_dataset = ...
        # training routines
        # outputs routines
    elif args.op == "test":
        pass
    elif args.op == "val":
        pass
    else:
        raise WrongArgument(argument="--op")
    # Computare le metriche in output