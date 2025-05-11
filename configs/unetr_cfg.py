# configs/default.py

import os
from pathlib import Path
import torch

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent

# 数据路径
DATASET_JSON = "dataset_0.json"  
DATA_ROOT = os.path.join(ROOT_DIR, "datasets", "syn3193805")

# 输出模型保存目录
OUTPUT_DIR = os.path.join(ROOT_DIR, "unetr_btcv_output")

# 模型参数
MODEL_NAME = "unetr"
NUM_CLASSES = 14
INPUT_CHANNELS = 1
IMAGE_SIZE = (96, 96, 96)

# 训练参数
SEED = 42
BATCH_SIZE = 2
MAX_EPOCHS = 500
EVAL_EPOCH = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EVAL_NUM = 500
PATCH_SIZE = (96, 96, 96)
NUM_SAMPLES = 1

# 网络推理参数
INFERER = {
    "mode": "sliding_window",
    "roi_size": (96, 96, 96),
    "sw_batch_size": 1,
    "overlap": 0.5,
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"