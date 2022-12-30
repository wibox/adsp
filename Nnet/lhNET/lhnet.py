from torchgeo.models import resnet50
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch import Unet

from typing import *
import traceback

class LHNet():
    def __init__(
        self,
        encoder : str = "resnet50",
        in_channels : int = 10,
        classes : int = 2
    ):
        self.encoder = encoder
        self.in_channels = in_channels
        self.classes = classes

    def _instantiate_nets(self) -> Tuple[dict, dict]:
        m1 = resnet50("sentinel2", "all", pretrained=True)
        m2 = get_encoder(self.encoder, in_channels=self.in_channels)
        # solo per confronto
        ckpt1 = m1.state_dict()
        ckpt2 = m2.state_dict()

        return ckpt1, ckpt2

    def _check_weights_compatibility(self, ckpt1 : dict, ckpt2 : dict) -> bool:
        # di nuovo, questo puo' essere fatto anche solo una volta
        compatible = dict()
        for key in ckpt1.keys():
            if key not in ckpt2:
                print(f"{key} not in timm resnet50")
                continue
            t1 = ckpt1[key]
            t2 = ckpt2[key]
            if t1.shape != t2.shape:
                print(f"{key} tensors are different: {t1.shape} - {t2.shape}")
            compatible[key] = t1

        assert len(compatible) == len(ckpt2)

        return compatible

    def _create_model(self) -> Union[Tuple[bool, Any], Tuple[bool, None]]:

        completed = False
        segmenter = None

        ckpt1, ckpt2 = self._instantiate_nets()
        new_weights_dict = self._check_weights_compatibility(ckpt1=ckpt1, ckpt2=ckpt2)
        # e con questo carica una UNet con 10 canali e lo state_dict della resnet50 di torchgeo
        segmenter = Unet(encoder_name=self.encoder, encoder_weights=None, in_channels=self.in_channels)
        try:
            segmenter.encoder.load_state_dict(new_weights_dict)
            completed = True
        except Exception as e:
            print(traceback.format_exc())
        finally:
            return completed, segmenter
