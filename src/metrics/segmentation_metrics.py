# -*- Extract from resources/unetr_btcv_segmentation_3d.ipynb -*-
from monai.metrics import DiceMetric

def get_dice_metric():
    return DiceMetric(include_background=True, reduction="mean")
