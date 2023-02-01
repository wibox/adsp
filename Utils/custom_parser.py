import argparse
import multiprocessing
def custom_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=51996,
        help="Seed value for reproducibility."
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
        "--dataset",
        type=str,
        choices=["vanilla", "ben", "effis", "imagenet"],
        default="ben",
        help="Decides the dataset to be used with underlying UNET."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs used to fine-tune or train from scratch."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Batch size used for dataloaders."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of workers used in dataloaders."
    )
    parser.add_argument(
        "--freeze-encoder",
        type=bool,
        default=False,
        help="Wheather to freeze or not the current encoder's weights."
    )
    parser.add_argument(
        "--encoder-name",
        type=str,
        default="resnet50",
        help="Encoder to be used in Segmentation Models Pytorch implementation of Unet."
    )
    parser.add_argument(
        "--save",
        type=bool,
        default=True,
        help="Wheather to save or not the trained model in models/trained_models/"
    )
    return parser.parse_args()