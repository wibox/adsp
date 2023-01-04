from typing import *

import albumentations as albu
from albumentations.pytorch import ToTensorV2

class OptimusPrime():
    def __init__(self,
                mean : List[float],
                std : List[float],
                tile_dimension : int = 512,
                ):
        self.mean = mean
        self.std = std
        self.tile_dimension = tile_dimension
        
    def compose(self, transforms_to_compose : List[Any] = None) -> albu.Compose:
        # combine all augmentations into single pipeline
        if transforms_to_compose:
            result = albu.Compose(
              [item for sublist in transforms_to_compose for item in sublist],
              additional_targets =
              {
                  "mask" : "mask"
              }
            )
            return result
        else:
            raise Exception("Empty set of transformations to compose passed.")
    
    # def pre_transforms(self) -> List[Any]:
    #     return [albu.Resize(self.tile_dimension, self.tile_dimension, p=1)]
    
    def post_transforms(self) -> List[Any]:
    # using ImageNet image normalization
        return [albu.Normalize(mean=self.mean, std=self.std), ToTensorV2()]

    def channel_shuffle(self, p : float = .5) -> List[Any]:
        return [albu.ChannelShuffle(p=p)]

    def random_brightness_contrast(self, brightness_limit : float = .2, contrast_limit : float = .2, brightness_by_max : bool = True, p : float = .5) -> List[Any]:
        return [albu.RandomBrightnessContrast(brightness_limit=brightness_limit, contrast_limit=contrast_limit, brightness_by_max=brightness_by_max, p=p)]

    def hue_saturation(self, hue_shift_limit : int = 20, val_shift_limit : int = 20, sat_shift_limit : int = 30, p : float = .5) -> List[Any]:
        return [albu.HueSaturationValue(hue_shift_limit=hue_shift_limit, val_shift_limit=val_shift_limit, sat_shift_limit=sat_shift_limit, p=p)]

    def gauss_noise(self, var_limit : Tuple[float, float], mean : float, per_channel : bool = True, p : float = .5) -> List[Any]:
        return [albu.GaussNoise(var_limit=var_limit, mean=mean, per_channel=per_channel, p=p)]

    def flip(self, p : float = .5) -> List[Any]:
        return [albu.Flip(p=p)]

    def rotate(self, limit : int = 360, p : float = .5) -> List[Any]:
        return [albu.Rotate(limit=limit, p=p)]

    def fixed_rotate(self, p : float = .5) -> List[Any]:
        return [albu.RandomRotate90(p=p)]