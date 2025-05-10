# -*- Extract from resources/unetr_btcv_segmentation_3d.ipynb -*-
from monai.networks.nets import UNETR
import torch.nn as nn

class UNETRModel(nn.Module):
    def __init__(self, in_channels=1, out_channels=14, img_size=(96, 96, 96), feature_size=16):
        super().__init__()
        self.model = UNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            feature_size=feature_size,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            # pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0,
        )

    def forward(self, x):
        return self.model(x)
