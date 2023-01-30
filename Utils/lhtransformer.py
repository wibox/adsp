from typing import *

import albumentations as albu
from albumentations.pytorch import ToTensorV2

class OptimusPrime():
    def __init__(self,
                tile_dimension : int = 512,
                ):
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

    def post_transforms_vanilla(self) -> List[Any]:
        return [ToTensorV2()]

    def post_transforms_imagenet(self) -> List[Any]:
        mean = [0.485, 0.456, 0.406, 0.485, 0.456, 0.406, 0.485, 0.456, 0.406, 0.485]#, 0.456, 0.406]
        std = [0.229, 0.224, 0.225, 0.229, 0.224, 0.225, 0.229, 0.224, 0.225, 0.229]#, 0.224, 0.225]
        # return [albu.Normalize(mean=mean, std=std), ToTensorV2()]
        return [ToTensorV2()]

    def post_transforms_bigearthnet(self) -> List[Any]:
        BAND_STATS_S2 = {
            "mean": {
                "B01": 340.76769064,
                "B02": 429.9430203,
                "B03": 614.21682446,
                "B04": 590.23569706,
                "B05": 950.68368468,
                "B06": 1792.46290469,
                "B07": 2075.46795189,
                "B08": 2218.94553375,
                "B8A": 2266.46036911,
                "B09": 2246.0605464,
                "B11": 1594.42694882,
                "B12": 1009.32729131,
            },
            "std": {
                "B01": 554.81258967,
                "B02": 572.41639287,
                "B03": 582.87945694,
                "B04": 675.88746967,
                "B05": 729.89827633,
                "B06": 1096.01480586,
                "B07": 1273.45393088,
                "B08": 1365.45589904,
                "B8A": 1356.13789355,
                "B09": 1302.3292881,
                "B11": 1079.19066363,
                "B12": 818.86747235,
            },
        }
        # SHUB_MEAN = [ x / 10000 for x in BAND_STATS_S2["mean"].values()]
        # SHUB_STD = [ x / 10000 for x in BAND_STATS_S2["std"].values()]
        SHUB_MEAN = [ x / 10000 for x in BAND_STATS_S2["mean"].values()]
        SHUB_STD = [ x / 10000 for x in BAND_STATS_S2["std"].values()]
        # return [albu.Normalize(mean=SHUB_MEAN, std=SHUB_STD), ToTensorV2()]
        return [ToTensorV2()]

    # def channel_shuffle(self, p : float = .5) -> List[Any]:
    #     return [albu.ChannelShuffle(p=p)]
    ###
    def random_brightness_contrast(self, brightness_limit : float = .2, contrast_limit : float = .2, brightness_by_max : bool = True, p : float = .5) -> List[Any]:
        return [albu.RandomBrightnessContrast(brightness_limit=brightness_limit, contrast_limit=contrast_limit, brightness_by_max=brightness_by_max, p=p)]

    # def hue_saturation(self, hue_shift_limit : int = 20, val_shift_limit : int = 20, sat_shift_limit : int = 30, p : float = .5) -> List[Any]:
    #     return [albu.HueSaturationValue(hue_shift_limit=hue_shift_limit, val_shift_limit=val_shift_limit, sat_shift_limit=sat_shift_limit, p=p)]
    ### mean 0 var 250 - 1250
    def gauss_noise(self, var_limit : Tuple[float, float], mean : float, per_channel : bool = True, p : float = .5) -> List[Any]:
        return [albu.GaussNoise(var_limit=var_limit, mean=mean, per_channel=per_channel, p=p)]

    # def flip(self, p : float = .5) -> List[Any]:
    #     return [albu.Flip(p=p)]

    # def rotate(self, limit : int = 360, p : float = .5) -> List[Any]:
    #     return [albu.Rotate(limit=limit, p=p)]
    ###
    def fixed_rotate(self, p : float = .5) -> List[Any]:
        return [albu.RandomRotate90(p=p)]

    def shift_scale_rotate(self, p : float = .5) -> List[Any]:
        return [albu.ShiftScaleRotate(p=p)]