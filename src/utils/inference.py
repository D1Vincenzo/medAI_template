# -*- Extract from resources/unetr_btcv_segmentation_3d.ipynb -*-
from monai.inferers import sliding_window_inference

def infer_sliding_window(inputs, model, roi_size=(96, 96, 96), sw_batch_size=4, overlap=0.8):
    return sliding_window_inference(inputs, roi_size, sw_batch_size, model, overlap=overlap)
