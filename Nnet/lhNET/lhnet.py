import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn

class LHNet():
    def __init__(
        self,
        encoder : str = "resnet50",
        encoder_weights : str = "imagenet",
        in_channels : int = 3,
        classes : int = 2
    ):
        self.encoder = encoder
        self.encoder_weights = encoder_weights
        self.in_channels = in_channels
        self.classes = classes

    def _get_model_and_input_preprocessing(self):
        
        self.model = smp.Unet(
        encoder_name=self.encoder_name,
        encoder_weights=self.encoder_weights,
        in_channels=self.in_channels,
        classes=self.classes,
        )

        self.preprocess_input = get_preprocessing_fn('resnet50', pretrained='imagenet')

        return self.model, self.preprocess_input