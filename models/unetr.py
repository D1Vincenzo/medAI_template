import torch
import torch.nn as nn
from monai.networks.nets import UNETR
from configs import unetr_cfg as cfg
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class UNETRModel(nn.Module):
    def __init__(self, in_channels=1, out_channels=14, img_size=(96, 96, 96), feature_size=16):
        super(UNETRModel, self).__init__()
        self.model = UNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            feature_size=feature_size,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            proj_type="perceptron",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0,
        )

    def forward(self, x):
        return self.model(x)
