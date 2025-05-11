# data/data_module.py

from monai.data import DataLoader, CacheDataset, load_decathlon_datalist
from data.transforms import get_train_transforms, get_val_transforms
from configs import unetr_cfg as cfg


class DataModule:
    def __init__(self):
        self.data_dir = cfg.DATA_ROOT
        self.json_path = self.data_dir + '/' + cfg.DATASET_JSON

        # Load data split paths from json (but not actual data yet)
        self.train_list = load_decathlon_datalist(self.json_path, data_list_key="training")
        self.val_list = load_decathlon_datalist(self.json_path, data_list_key="validation")[0:1]
        self.test_list = load_decathlon_datalist(self.json_path, data_list_key="test")[0:1]

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def setup(self, test_mode=False):
        if self.train_ds is None and not test_mode:
            self.train_ds = CacheDataset(
                data=self.train_list,
                transform=get_train_transforms(),
                cache_num=24,
                cache_rate=1.0,
                num_workers=8,
            )
        if self.val_ds is None and not test_mode:
            self.val_ds = CacheDataset(
                data=self.val_list,
                transform=get_val_transforms(),
                cache_num=6,
                cache_rate=1.0,
                num_workers=4,
            )
        if self.test_ds is None and test_mode:
            self.test_ds = CacheDataset(
                data=self.test_list,
                transform=get_val_transforms(),
                cache_num=6,
                cache_rate=1.0,
                num_workers=4,
            )

    def get_dataloaders(self):
        self.setup()
        train_loader = DataLoader(
            self.train_ds,
            batch_size=cfg.BATCH_SIZE,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )
        
        val_loader = DataLoader(
            self.val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        return train_loader, val_loader

    def get_test_loader(self):
        self.setup(test_mode=True)
        test_loader = DataLoader(
            self.test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        return test_loader
