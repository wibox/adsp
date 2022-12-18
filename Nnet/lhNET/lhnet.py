import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn

def _get_model_and_input_preprocessing(encoder_name : str = "resnet50", encoder_weights : str = "imagenet", in_channels : int = 13, classes : int = 2):
    
    model = smp.Unet(
    encoder_name=encoder_name,
    encoder_weights=encoder_weights,
    in_channels=in_channels,
    classes=classes,
    )

    preprocess_input = get_preprocessing_fn('resnet50', pretrained='imagenet')

    return model, preprocess_input