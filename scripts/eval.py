# scripts/eval.py

import torch
from configs import unetr_cfg as cfg
from models.unetr import UNETRModel
import os
from data.unetr_data_loader import DataModule
from monai.inferers import sliding_window_inference
import matplotlib.pyplot as plt


def main():
    print("Starting evaluation...")

    case_num = [0, 1, 2]
    device = cfg.DEVICE
    model = UNETRModel().to(device)
    
    dm = DataModule()
    test_loader = dm.get_test_loader()
    
    
    model.load_state_dict(torch.load(os.path.join(cfg.ROOT_DIR, "best_metric_model.pth")))
    model.eval()
    

    for i, batch in enumerate(test_loader):
        if i not in case_num:
            continue
        
        with torch.no_grad():
            img = batch["image"][0]   # (1, C, D, H, W) → (C, D, H, W)
            label = batch["label"][0]
            slice_idx = 200

            val_inputs = img.unsqueeze(0).to(device)   # (1, C, D, H, W)
            val_labels = label.unsqueeze(0).to(device)

            val_outputs = sliding_window_inference(val_inputs, cfg.PATCH_SIZE, 4, model, overlap=0.8)

            plt.figure("check", (18, 6))

            plt.subplot(1, 3, 1)
            plt.title("image")
            plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, slice_idx], cmap="gray")

            plt.subplot(1, 3, 2)
            plt.title("label")
            plt.imshow(val_labels.cpu().numpy()[0, 0, :, :, slice_idx])

            plt.subplot(1, 3, 3)
            plt.title("output")
            plt.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, slice_idx])

            plt.show()
        

if __name__ == "__main__":
    main()
