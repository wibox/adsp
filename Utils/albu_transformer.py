import albumentations as albu
from albumentations.pytorch import ToTensorV2
from typing import *

class OptimusPrime():
  def __init__(self, tile_dimension : int = 512):
    self.tile_dimension = tile_dimension

  def pre_transforms(self) -> List[Any]:
      return [albu.Resize(self.tile_dimension, self.tile_dimension, p=1)]

  def hard_transforms(self):
      result = [
        albu.RandomRotate90(),
        albu.Cutout(),
        albu.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.3
        ),
        albu.GridDistortion(p=0.3),
        albu.HueSaturationValue(p=0.3)
      ]
      return result
  
  def resize_transforms(self, image_size : int = 512) -> List[albu.OneOf]:
      BORDER_CONSTANT = 0
      # pre_size = int(image_size * 1.5)
      pre_size = int(image_size)
      random_crop = albu.Compose([
        albu.SmallestMaxSize(pre_size, p=1),
        albu.RandomCrop(
            image_size, image_size, p=1
        )
      ])
      rescale = albu.Compose([albu.Resize(image_size, image_size, p=1)])
      random_crop_big = albu.Compose([
        albu.LongestMaxSize(pre_size, p=1),
        albu.RandomCrop(
            image_size, image_size, p=1
        )

      ])
      # Converts the image to a square of size image_size x image_size
      result = [
        albu.OneOf([
            random_crop,
            rescale,
            random_crop_big
        ], p=1)
      ]
      return result
  
  def post_transforms(self) -> List[Any]:
      # using ImageNet image normalization
      return [albu.Normalize(), ToTensorV2()]
  
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