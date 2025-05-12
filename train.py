# scripts/train.py

import torch
import random
import numpy as np

from configs import unetr_cfg as cfg
import os
from tqdm import tqdm
from data.unetr_data_loader import DataModule
from engine.trainer import Trainer
from models.unetr import UNETRModel

dm = DataModule()

def set_seed(seed=cfg.SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    print("Starting training...")
    set_seed(cfg.SEED)

    progress_bar = tqdm(total=cfg.MAX_EPOCHS, desc="Training Progress", dynamic_ncols=True)
    dm.setup()
    train_loader, val_loader = dm.get_dataloaders()

    model = UNETRModel()
    trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader, device=cfg.DEVICE)

    for epoch in range(cfg.MAX_EPOCHS):
        trainer.train(do_validation=True)
        progress_bar.update(1)

    progress_bar.close()


if __name__ == "__main__":
    main()
