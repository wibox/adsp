from Utils.outputformatter import OutputFormatter
from Utils.lhtransformer import OptimusPrime
from Preprocessing.imagedataset import ImageDataset as image_dataset
import segmentation_models_pytorch as smp
import torch

models = ["vanilla", "imagenet", "bigearthnet"]

my_transformer = OptimusPrime()
test_transforms = my_transformer.compose([
    my_transformer.post_transforms_vanilla()
])

test_ds = image_dataset(
    formatted_folder_path="/mnt/data1/adsp_data/test_colombaset",
    log_folder="Log",
    master_dict="test_master_dict.json",
    transformations=test_transforms,
    use_pre=False,
    verbose=1,
    specific_indeces=None,
    return_path=False
)
test_ds._load_tiles()

for model in models:
    if model == "vanilla":
        print("Building vanilla network with finetuned model")
        net = smp.Unet(encoder_name="resnet50", in_channels=10, encoder_weights=None)
        output_formatter = OutputFormatter(
            model=net,
            filepath="/mnt/data1/adsp_data",
            test_ds=test_ds,
            best_model_path="models/vanilla.pth",
            test_output_path="vanilla_test_output_colombaset"
        )
        print("Computing output...")
        output_formatter.compute_output()
    elif model == "imagenet":
        print("Building network with finetuned model over imagenet pretrain")
        net = smp.Unet(encoder_name="resnet50", in_channels=10, encoder_weights="imagenet")
        net.encoder.load_state_dict(torch.load("models/10bandsimagenet.pth"))
        output_formatter = OutputFormatter(
            model=net,
            filepath="/mnt/data1/adsp_data",
            test_ds=test_ds,
            best_model_path="models/imagenet.pth",
            test_output_path="imagenet_test_output_colombaset"
        )
        print("Computing output...")
        output_formatter.compute_output()
    elif model == "bigearthnet":
        print("Building network with finetuned model over bigearthnet pretrain")
        net = smp.Unet(encoder_name="resnet50", in_channels=10, encoder_weights=None)
        net.encoder.load_state_dict(torch.load("models/checkpoint-30.pth.tar"), strict=False)
        output_formatter = OutputFormatter(
            model=net,
            filepath="/mnt/data1/adsp_data",
            test_ds=test_ds,
            best_model_path="models/ben.pth",
            test_output_path="ben_test_output_colombaset"
        )
        print("Computing output...")
        output_formatter.compute_output()