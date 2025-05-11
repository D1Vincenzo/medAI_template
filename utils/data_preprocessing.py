import os
import json
from pathlib import Path
import shutil
from tqdm import tqdm

# 原始路径
raw_img_dir = Path("raw_datasets/RawData/Training/img")
raw_label_dir = Path("raw_datasets/RawData/Training/label")

# 输出路径（符合 Decathlon 格式）
output_dir = Path("datasets/syn3193805")
imagesTr_dir = output_dir / "imagesTr"
labelsTr_dir = output_dir / "labelsTr"
output_json = output_dir / "dataset_0.json"

# 创建目录
imagesTr_dir.mkdir(parents=True, exist_ok=True)
labelsTr_dir.mkdir(parents=True, exist_ok=True)

# 查找所有图像文件
img_files = sorted(raw_img_dir.glob("img*.nii.gz"))
label_files = sorted(raw_label_dir.glob("label*.nii.gz"))

assert len(img_files) == len(label_files), "图像与标签数量不匹配！"

# 划分比例
num_total = len(img_files)
num_train = int(num_total * 0.8)
num_val = int(num_total * 0.1)
num_test = num_total - num_train - num_val

# 初始化 JSON 字段
json_dict = {
    "name": "BTCV",
    "description": "btcv preprocessed from RawData",
    "tensorImageSize": "3D",
    "modality": {"0": "CT"},
    "labels": {
        "0": "background",
        "1": "spleen",
        "2": "rkid",
        "3": "lkid",
        "4": "gall",
        "5": "eso",
        "6": "liver",
        "7": "sto",
        "8": "aorta",
        "9": "IVC",
        "10": "veins",
        "11": "pancreas",
        "12": "rad",
        "13": "lad"
    },
    "numTraining": num_train,
    "numTest": num_test,
    "training": [],
    "validation": [],
    "test": []
}

# 复制并生成 JSON 条目
for idx, (img_path, label_path) in enumerate(tqdm(zip(img_files, label_files), total=num_total)):
    new_img_name = f"img{idx+1:04d}.nii.gz"
    new_label_name = f"label{idx+1:04d}.nii.gz"

    shutil.copy(img_path, imagesTr_dir / new_img_name)
    shutil.copy(label_path, labelsTr_dir / new_label_name)

    entry = {
        "image": f"imagesTr/{new_img_name}",
        "label": f"labelsTr/{new_label_name}"
    }

    if idx < num_train:
        json_dict["training"].append(entry)
    elif idx < num_train + num_val:
        json_dict["validation"].append(entry)
    else:
        json_dict["test"].append(entry)


# 写入 JSON 文件
with open(output_json, "w") as f:
    json.dump(json_dict, f, indent=4)

print(f"✅ 数据预处理完成！共 {num_total} 个样本")
print(f"📄 JSON 文件保存于: {output_json.resolve()}")
