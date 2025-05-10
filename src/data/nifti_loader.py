from lightning import LightningDataModule
from monai.data import CacheDataset, DataLoader, load_decathlon_datalist
from src.utils.transforms import get_train_transforms, get_val_transforms

class NiftiDataModule(LightningDataModule):
    def __init__(self, data_dir, json_list, batch_size=2, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.json_list = json_list
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def setup(self, stage=None):
        train_transforms = get_train_transforms()
        val_transforms = get_val_transforms()

        train_list = load_decathlon_datalist(
            data_list_file_path=self.json_list,
            is_segmentation=True,
            data_list_key="training",
            base_dir=self.data_dir
        )[0:1]

        val_list = load_decathlon_datalist(
            data_list_file_path=self.json_list,
            is_segmentation=True,
            data_list_key="validation",
            base_dir=self.data_dir
        )[0:1]  # Only one validation sample

        # test_list = load_decathlon_datalist(
        #     data_list_file_path=self.json_list,
        #     is_segmentation=True,
        #     data_list_key="test",
        #     base_dir=self.data_dir
        # )

        if len(train_list) > 0:
            self.train_ds = CacheDataset(data=train_list, transform=train_transforms, cache_rate=1.0)
        if len(val_list) > 0:
            self.val_ds = CacheDataset(data=val_list, transform=val_transforms, cache_rate=1.0)
        # if len(test_list) > 0:
        #     self.test_ds = CacheDataset(data=test_list, transform=val_transforms, cache_rate=1.0)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=1, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=1, num_workers=self.num_workers)
