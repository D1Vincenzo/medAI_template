# -*- Extract from resources/unetr_btcv_segmentation_3d.ipynb -*-
from monai.losses import DiceCELoss

def get_loss():
    return DiceCELoss(to_onehot_y=True, softmax=True)
