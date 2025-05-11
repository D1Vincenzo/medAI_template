# engine/trainer.py

import os
import torch
from torch.optim import AdamW
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from configs import unetr_cfg as cfg

import gc


class Trainer:
    def __init__(self, model, train_loader, val_loader, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=cfg.LEARNING_RATE,
            weight_decay=cfg.WEIGHT_DECAY,
        )

        self.dice_metric = DiceMetric(include_background=False, reduction="none", get_not_nans=False)
        self.max_epochs = cfg.MAX_EPOCHS
        self.eval_epoch = cfg.EVAL_EPOCH
        self.eval_num = cfg.EVAL_NUM

        self.post_label = AsDiscrete(to_onehot=cfg.NUM_CLASSES)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=cfg.NUM_CLASSES)

        # 用 self.xxx 保留状态变量
        self.global_step = 0
        self.global_step_best = 0
        self.dice_val_best = 0.0

        self.epoch_loss_values = []
        self.metric_values = []
        self.iteration_list = []
        self.current_iteration = 0

    def validation(self):
        self.model.eval()
        with torch.no_grad():
            for batch in self.val_loader:
                val_inputs, val_labels = (batch["image"].to(self.device), batch["label"].to(self.device))
                val_outputs = sliding_window_inference(val_inputs, cfg.PATCH_SIZE, 4, self.model)
                val_labels_list = decollate_batch(val_labels)
                val_labels_convert = [self.post_label(t) for t in val_labels_list]
                val_outputs_list = decollate_batch(val_outputs)
                val_output_convert = [self.post_pred(t) for t in val_outputs_list]
                self.dice_metric(y_pred=val_output_convert, y=val_labels_convert)

        # mean_dice_val = self.dice_metric.aggregate().item()
        
        dice_array = self.dice_metric.aggregate(reduction=None).cpu().numpy()  # shape = (batch, classes)
        self.dice_metric.reset()

        # ✅ 取 batch 维度平均（常见做法），得到每类 Dice 值
        if dice_array.ndim == 2:
            dice_per_class = dice_array.mean(axis=0)  # 平均每类 Dice
        else:
            dice_per_class = dice_array  # fallback：已经是 1D

        # 构造完整类列表（13 前景类）
        full_dice = [0.0] * (cfg.NUM_CLASSES - 1)
        for i in range(min(len(full_dice), len(dice_per_class))):
            full_dice[i] = float(dice_per_class[i])  # ✅ 强制转换为 float

        print("\n🔎 Per-Class Dice Scores:")
        for i, score in enumerate(full_dice):
            print(f"  Class {i + 1}: Dice = {score:.4f}")


        # ✅ 平均 Dice
        mean_dice_val = sum(full_dice) / len(full_dice)
        print(f"\n✅ Mean Dice (excluding background): {mean_dice_val:.4f}")
        
        class_6_dice = full_dice[5] 
        print(f"Class 6 Dice: {class_6_dice:.4f}")
        
        # return mean_dice_val
        return class_6_dice

    # ==== 训练函数 ====
    def train(self, do_validation=False):

        self.model.train()
        epoch_loss = 0

        for step, batch in enumerate(self.train_loader):
            x, y = batch["image"].to(self.device), batch["label"].to(self.device)
            logit_map = self.model(x)
            loss = self.loss_function(logit_map, y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            epoch_loss += loss.item()
            self.global_step += 1

        avg_loss = epoch_loss / (step + 1)
        self.epoch_loss_values.append(avg_loss)

        # print(f"Epoch {epoch + 1}/{self.max_epochs} - Loss: {avg_loss:.4f}")

        if do_validation:
            dice_val = self.validation()
            self.metric_values.append(dice_val)
            self.current_iteration += self.eval_num
            self.iteration_list.append(self.current_iteration)

            if dice_val > self.dice_val_best:
                self.dice_val_best = dice_val
                self.global_step_best = self.global_step
                torch.save(self.model.state_dict(), os.path.join(cfg.ROOT_DIR, "best_metric_model.pth"))
                print(f"✅ Model Saved! Best Dice: {self.dice_val_best:.4f}")
            else:
                print(f"Model Not Saved. Best Dice: {self.dice_val_best:.4f}, Current: {dice_val:.4f}")

        torch.cuda.empty_cache()
        gc.collect()
