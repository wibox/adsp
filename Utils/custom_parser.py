import argparse
def custom_parser():
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
    return parser.parse_args()