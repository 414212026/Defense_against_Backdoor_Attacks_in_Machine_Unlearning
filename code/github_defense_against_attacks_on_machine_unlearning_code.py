

# 原作者的 github：https://secartifacts.github.io/usenixsec2024/appendix-files/sec24winterae-final97.pdf
# 原作者的pre-training dataset and config: https://drive.google.com/drive/u/0/folders/1NMrjpS7TqVHEBtKKF6HXZJSQ8BU8zMpU

# Below is an sample code for cifar10 badnet PARN-18 machine unlearning attack and defense

####################### forget_cover_clean_ratio=1 ############################

import torch
# ✅ 添加（或确认你已添加）seed函数
def set_all_seeds(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)  # 🔧 MODIFIED: ensure deterministic
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_all_seeds(42)
forget_cover_clean_ratio = 1
import sys
import os
# Windows/Linux：Ctrl + Alt + S ->Build, Execution, Deployment → Console → Python Console ->Environment variables
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
import numpy as np
import random
from torch.utils.data import Subset, DataLoader
import argparse

# ✅ 设置项目路径
project_root = r"C:\pycharm\pycharm_project_folder\20250609_1\uba-Inf"
sys.path.append(project_root)

# ✅ 设置保存路径和模型文件路径
save_path = os.path.join(project_root, "record", "perturb_badnet_preactresnet")
checkpoint_path = os.path.join(save_path, "perturb_result.pt")
index_path = os.path.join(save_path, "cv_bd_index.pt")

# ✅ 加载模型
from utils.aggregate_block.model_trainer_generate import generate_cls_model

model = generate_cls_model(model_name="preactresnet18", num_classes=10, image_size=32)

checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint["model"])
init_state = checkpoint["model"]
device = torch.device("cuda")
model.to(device)
model.eval()
BASE_SEED=42
#
# # ✅ 加载数据集：恢复 bd_train, clean_test, bd_test
from uba.uba_utils.basic_utils import get_perturbed_datasets


def rebuild_bd_train_from_saved_indices(args, index_path):
    index_dict = torch.load(index_path)
    bd_index = index_dict["bd_index"]
    cv_index = index_dict["cv_index"]

    # 获取原始的 four datasets
    clean_train, clean_test, bd_train, bd_test = get_perturbed_datasets(args)

    # 获取干净样本的索引（poison_indicator == 0）
    train_indicator = bd_train.poison_indicator
    clean_index = np.where(train_indicator == 0)[0]

    # 构建完整子集索引：bd_index + cv_index + clean_index
    subset_index = list(bd_index) + list(cv_index) + list(clean_index)

    # 重建 bd_train
    bd_train.subset(subset_index)

    return bd_train, clean_test, bd_test


# ✅ 调用函数，重建数据集
args = argparse.Namespace(
    dataset_folder=save_path,
    dataset_name="perturb_result.pt",
    add_cover=True
)
bd_train, clean_test, bd_test = rebuild_bd_train_from_saved_indices(args, index_path)

import torch
import random
from torch.utils.data import Dataset, Subset, DataLoader, ConcatDataset
import numpy as np


class FixLabelTensor(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        x, y = self.dataset[idx][:2]  # 只取前两个元素
        return x, torch.tensor(y).long()

    def __len__(self):
        return len(self.dataset)


#
def create_dataloader(dataset, start, end, shuffle=False):
    return DataLoader(
        Subset(dataset, list(range(start, end))),
        batch_size=128,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True
    )


# ✅ 加载索引
index_dict = torch.load(index_path)
bd_index = index_dict["bd_index"]
cv_index = index_dict["cv_index"]
train_indicator = bd_train.poison_indicator
clean_index = np.where(train_indicator == 0)[0]

# ✅ 划分数量
n_bd = len(bd_index)
n_cv = len(cv_index)
n_cl = len(clean_index)

# ✅ 创建基础 dataloader
dl_poison_train = create_dataloader(bd_train, 0, n_bd)
dl_cover_train = create_dataloader(bd_train, n_bd, n_bd + n_cv)
dl_pure_clean_train = create_dataloader(bd_train, n_bd + n_cv, n_bd + n_cv + n_cl)
dl_pure_clean_test = DataLoader(clean_test, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
dl_poison_test = DataLoader(bd_test, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)


def get_shuffled_indices(dataset, seed):
    np.random.seed(seed)
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    return indices


retain_dataset = ConcatDataset([
    FixLabelTensor(dl_poison_train.dataset),
    FixLabelTensor(dl_pure_clean_train.dataset)
])
forget_dataset = ConcatDataset([
    FixLabelTensor(dl_cover_train.dataset)
])

retain_indices = get_shuffled_indices(retain_dataset, seed=42)
forget_indices = get_shuffled_indices(forget_dataset, seed=42)

save_dict = {
    # 原始数据
    "bd_train": bd_train,
    "clean_test": clean_test,
    "bd_test": bd_test,

    # 用于训练评估的 dataset
    "retain_dataset": retain_dataset,
    "forget_dataset": forget_dataset,
    "poison_dataset": FixLabelTensor(dl_poison_train.dataset),
    "test_poison_dataset": FixLabelTensor(dl_poison_test.dataset),
    "test_clean_dataset": FixLabelTensor(dl_pure_clean_test.dataset),

    # index
    "retain_indices": retain_indices,
    "forget_indices": forget_indices,
}

torch.save(save_dict, os.path.join(save_path, "final_all_datasets.pt"))
print("[✅] 已保存 final_all_datasets.pt，包含 bd_train、测试集、训练集各类拼接数据")

data = torch.load(os.path.join(save_path, "final_all_datasets.pt"))

# 基础数据集（如果你之后需要重新生成 dataloader）
bd_train = data["bd_train"]
clean_test = data["clean_test"]
bd_test = data["bd_test"]

# 构建 DataLoader
train_retain_dl = DataLoader(
    Subset(data["retain_dataset"], data["retain_indices"]),
    batch_size=128, shuffle=False, num_workers=0, pin_memory=True
)

train_forget_dl = DataLoader(
    Subset(data["forget_dataset"], data["forget_indices"]),
    batch_size=128, shuffle=False, num_workers=0, pin_memory=True
)

dl_poison_train = DataLoader(
    data["poison_dataset"],
    batch_size=128, shuffle=False, num_workers=0, pin_memory=True
)

test_poison_dl = DataLoader(
    data["test_poison_dataset"],
    batch_size=128, shuffle=False, num_workers=0, pin_memory=True
)

test_clean_dl = DataLoader(
    data["test_clean_dataset"],
    batch_size=128, shuffle=False, num_workers=0, pin_memory=True
)

import torch
import numpy as np
import random

#
# ✅ 添加导入
import torch.backends.cudnn  # 🔧 MODIFIED: ensure deterministic behavior




def evaluate_acc(model, loader, device):
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            x, y, *_ = batch
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

    acc = correct / total if total > 0 else 0
    acc_str = f"Accuracy: {correct}/{total} = {acc:.4f}"
    acc_num = round(acc, 4)

    return acc_str, acc_num


import os
import time
import csv
from tqdm import tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd

'''basic func'''


def evaluate_target_confidence(model, dataloader, target_label, device):
    confidences = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch[0].to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            target_probs = probs[:, target_label]
            confidences.extend(target_probs.cpu().numpy())
    return float(np.mean(confidences))


def plot_target_confidence_distribution(
        model, dataloader, device, epoch,
        save_dir, dataset_name,
        label_mode='target',  # 'target' or 'gt'
        target_label=6,
        bins=20,
        threshold=0.9, y_limit=3000, coefficient_loss_forget_ce=0, coefficient_loss_forget_data_kl=0,
        coefficient_loss_forget_label_kl=0
):
    """
    返回：
        avg_conf  -> 平均confidence
        high_count -> >= threshold的样本数
    """

    os.makedirs(save_dir, exist_ok=True)
    confs = []
    other_label_high_count = {}  # 记录其他label的高置信度数量

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            if label_mode == 'target':
                assert target_label is not None, "label_mode='target' 需要指定 target_label"
                conf_batch = probs[:, target_label]

                # 统计除 gt 和 target 外的 label
                for i in range(probs.size(0)):
                    for cls in range(probs.size(1)):
                        if cls != target_label and cls != labels[i].item():
                            if probs[i, cls].item() > 0.5:
                                other_label_high_count[cls] = other_label_high_count.get(cls, 0) + 1

            elif label_mode == 'gt':
                conf_batch = probs[torch.arange(labels.size(0), device=device), labels]
            else:
                raise ValueError("label_mode 必须是 'target' 或 'gt'")

            confs.extend(conf_batch.cpu().numpy())

    # 打印其他label的高置信度统计
    if label_mode == 'target':
        print(f"[Epoch {epoch}] Other-label high conf counts: {other_label_high_count}")

    confs = np.array(confs)
    avg_conf = float(confs.mean())
    high_count = int((confs >= threshold).sum())  # >= threshold 的样本数

    # 画直方图
    plt.figure(figsize=(6, 4))
    plt.hist(confs, bins=bins, range=(0, 1), edgecolor="black", alpha=0.75)
    plt.ylim(0, y_limit)  # 限制纵轴
    title = f"{dataset_name} - {label_mode.upper()} confidence (epoch {epoch})"
    plt.title(title)
    plt.xlabel("Probability")
    plt.ylabel("Count")
    fname = os.path.join(save_dir,
                         f"{dataset_name}_{coefficient_loss_forget_ce}_{coefficient_loss_forget_data_kl}_{coefficient_loss_forget_label_kl}_{label_mode}_epoch{epoch}.png")
    # if os.path.exists(fname):
    #     os.remove(fname)
    #     time.sleep(1.1)
    #     print("exist file and remove and sleep 1.1 s")

    root, ext = os.path.splitext(fname)
    tmp = root + ".tmp" + ext

    # 先写到临时文件
    plt.savefig(tmp)
    plt.close()

    # 强制把数据写到磁盘
    with open(tmp, "rb+") as f:
        os.fsync(f.fileno())

    # 原子替换为正式文件（mtime 会以新文件为准）
    os.replace(tmp, fname)

    # 可选：再强制设置一次修改时间为“现在”
    # now = time.time()
    # os.utime(fname, (now, now))
    #
    # # 打印实际修改时间，验证
    # from datetime import datetime
    # print("saved:", fname, "mtime:", datetime.fromtimestamp(os.path.getmtime(fname)))

    return avg_conf, high_count







import csv, os
import numpy as np
import torch

@torch.no_grad()
def compute_pred_class_hist(model, dataloader, device, num_classes=10, normalize=True):
    """统计 argmax 预测类别分布"""
    model.eval()
    counts = np.zeros(num_classes, dtype=np.int64)
    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            images = batch[0]
        else:
            images = batch
        images = images.to(device, non_blocking=True)
        logits = model(images)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        for p in preds:
            counts[int(p)] += 1
    if normalize:
        total = counts.sum()
        if total > 0:
            return counts / total
    return counts

def save_hist_to_csv(hist, csv_path, row_name):
    """保存直方图到 CSV"""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    header = ["name"] + [f"C{i}" for i in range(len(hist))]
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow([row_name] + [f"{x:.6f}" for x in hist.tolist()])



class LossData(Dataset):
    def __init__(self, forget_data, retain_data):
        super().__init__()
        self.forget_data = forget_data
        self.retain_data = retain_data
        self.forget_len = len(forget_data)
        self.retain_len = len(retain_data)

    def __len__(self):
        return self.forget_len + self.retain_len

    def __getitem__(self, index):
        if index < self.forget_len:
            x, y = self.forget_data[index]
            label = 1
        else:
            x, y = self.retain_data[index - self.forget_len]
            label = 0

        # 🔥 关键：全部返回 tensor 类型
        return x, torch.tensor(y).long(), torch.tensor(label).long()

import torch
import torch.nn.functional as F


def compute_forget_regularization_loss(logits, mode="data_kl", target_distribution="uniform"):
    """
    logits: shape (N, C)
    mode: ["data_kl", "data_entropy", "label_kl", "label_entropy"]
    target_distribution: currently only supports "uniform"
    """
    probs = F.softmax(logits, dim=1)

    if target_distribution == "uniform":
        uniform_target = torch.full_like(probs, 1.0 / probs.size(1))  # (N, C)
        uniform_label = torch.full((probs.size(1),), 1.0 / probs.size(1), device=probs.device)  # (C,)
    else:
        raise NotImplementedError("Only uniform target supported for now.")

    if mode == "data_kl":
        # KL for each sample: mean over batch
        log_probs = F.log_softmax(logits, dim=1)
        kl = F.kl_div(log_probs, uniform_target, reduction="batchmean")
        return kl

    elif mode == "data_entropy":
        log_probs = torch.log(probs + 1e-12)
        entropy = - torch.sum(probs * log_probs, dim=1)  # (N,)
        return entropy.mean()

    elif mode == "label_kl":
        mean_prob = probs.mean(dim=0)  # (C,)
        log_mean_prob = torch.log(mean_prob + 1e-12)
        kl = torch.sum(mean_prob * (log_mean_prob - torch.log(uniform_label)))
        return kl

    elif mode == "label_entropy":
        mean_prob = probs.mean(dim=0)  # (C,)
        log_mean_prob = torch.log(mean_prob + 1e-12)
        entropy = - torch.sum(mean_prob * log_mean_prob)
        return entropy

    else:
        raise ValueError(f"Unknown mode: {mode}")



def training_step_diverse(model, batch, criterion,epoch,coefficient_loss_retain,coefficient_loss_forget_ce,coefficient_loss_forget_data_kl,coefficient_loss_forget_label_kl,warmup_epoch):
    device = next(model.parameters()).device
    images, clabels, labels = batch
    images, clabels, labels = images.to(device), clabels.to(device), labels.to(device)

    out = model(images)

    retain_mask = (labels == 0)
    forget_mask = (labels == 1)

    retain_logits = out[retain_mask]
    retain_clabels = clabels[retain_mask]
    if retain_logits.shape[0] > 0:
        loss_retain = criterion(retain_logits, retain_clabels)
    else:
        loss_retain = torch.tensor(0.0, device=device, requires_grad=True)

    forget_logits = out[forget_mask]
    forget_clabels = clabels[forget_mask]
    if forget_logits.shape[0] > 0:
        loss_forget_ce = criterion(forget_logits, forget_clabels) # cross-entropy
        loss_forget_reg_data_kl = compute_forget_regularization_loss(forget_logits, mode='data_kl')
        loss_forget_reg_label_kl = compute_forget_regularization_loss(forget_logits, mode='label_kl')
    else:
        loss_forget_ce = torch.tensor(0.0, device=device, requires_grad=True)
        loss_forget_reg_data_kl = torch.tensor(0.0, device=device, requires_grad=True)
        loss_forget_reg_label_kl = torch.tensor(0.0, device=device, requires_grad=True)

    if epoch < warmup_epoch: #warmup
        loss = coefficient_loss_retain*loss_retain + coefficient_loss_forget_ce * loss_forget_ce
    else:
        loss = coefficient_loss_retain*loss_retain + coefficient_loss_forget_ce * loss_forget_ce + coefficient_loss_forget_data_kl*loss_forget_reg_data_kl+coefficient_loss_forget_label_kl*loss_forget_reg_label_kl

    return loss, loss_retain, loss_forget_ce,loss_forget_reg_data_kl,loss_forget_reg_label_kl



def FT_NegGradPlus_loss(
    num_epochs,
    model,
    train_retain_dl,
    train_forget_dl,
    test_poison_dl,
    test_clean_dl,
    device,
    learning_rate,
    forget_cover_clean_ratio,
    diverse_loss_type,
    coefficient_loss_retain,
    coefficient_loss_forget_ce,
    coefficient_loss_forget_data_kl,
    coefficient_loss_forget_label_kl,
    save_path='.',  # 默认当前目录保存日志
    model_type='neggrad_plus',
    clip_num=1,
    batch_size=128,
    warmup_epoch=2,
    mask=None,
    **kwargs,
):
    set_all_seeds(42)
    print(f"Unlearning ({model_type}) Result: \n")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    unlearning_data = LossData(forget_data=train_forget_dl.dataset, retain_data=train_retain_dl.dataset)
    generator = torch.Generator()
    generator.manual_seed(42)
    training_loader = DataLoader(unlearning_data, batch_size=batch_size, shuffle=True, pin_memory=True,generator=generator)

    log_csv_path = os.path.join(save_path, f"{model_type}_log_ratio_{forget_cover_clean_ratio}_{diverse_loss_type}_{coefficient_loss_forget_ce}_{coefficient_loss_forget_data_kl}_{coefficient_loss_forget_label_kl}.csv")
    # ✅ 创建并写入表头


    with open(log_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 'train_forget_acc',
            'test_poison_acc', 'test_clean_acc',
            'lr', 'epoch_time'
        ])

    base_log_path = os.path.join(save_path, "base_log_epoch-1.csv")
    if base_log_path and os.path.exists(base_log_path):
        with open(base_log_path, mode='r') as base_f:
            lines = list(csv.reader(base_f))
            if len(lines) >= 2:
                epoch_minus1_row = lines[1]
                with open(log_csv_path, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(epoch_minus1_row)
                print("[Info] Inserted epoch=-1 row from base_log.")
            else:
                print("[Warning] base_log_path does not contain epoch -1 row.")

    batch_log_csv_path = os.path.join(
        save_path,
        f"{model_type}_log_ratio_{forget_cover_clean_ratio}_{diverse_loss_type}_{coefficient_loss_forget_ce}_{coefficient_loss_forget_data_kl}_{coefficient_loss_forget_label_kl}_batch_loss_details.csv"
    )

    # 覆盖写入并写表头
    with open(batch_log_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 'batch',
            'total_loss', 'retain_ce', 'forget_ce',
            'forget_reg_data_kl', 'forget_reg_label_kl',
            'lr'
        ])

    hist_csv_path = os.path.join(
        save_path,
        f"{model_type}_forget_class_hist_{forget_cover_clean_ratio}_{diverse_loss_type}_{coefficient_loss_forget_ce}_{coefficient_loss_forget_data_kl}_{coefficient_loss_forget_label_kl}.csv"
    )
    hist_before = compute_pred_class_hist(model, train_forget_dl, device, num_classes=10, normalize=True)
    save_hist_to_csv(hist_before, hist_csv_path, row_name="before")

    for epoch in range(num_epochs):
        start = time.time()
        print("epoch", epoch)
        model.train()  # <--- 明确切成训练模式
        for i, batch in enumerate(training_loader):
            loss, loss_retain, loss_forget_ce, loss_forget_reg_data_kl,loss_forget_reg_label_kl = training_step_diverse(model, batch, criterion, epoch, coefficient_loss_retain,
                                  coefficient_loss_forget_ce,coefficient_loss_forget_data_kl,coefficient_loss_forget_label_kl,warmup_epoch)
            optimizer.zero_grad()
            loss.backward()

            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]
            if clip_num:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_num)

            if i % 100 == 0:
                print(
                    f"[Epoch {epoch} | Batch {i}] "
                    f"Total Loss: {loss.item():.4f} | "
                    f"Retain CE: {loss_retain.item():.4f} | "
                    f"Forget CE: {loss_forget_ce.item():.4f} | "
                    f"Forget Reg data kl: {loss_forget_reg_data_kl.item():.4f} | "
                    f"Forget Reg label kl: {loss_forget_reg_label_kl.item():.4f}"
                )

                with open(batch_log_csv_path, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        epoch, i,
                        f"{loss.item():.4f}",
                        f"{loss_retain.item():.4f}",
                        f"{loss_forget_ce.item():.4f}",
                        f"{loss_forget_reg_data_kl.item():.4f}",
                        f"{loss_forget_reg_label_kl.item():.4f}",
                        f"{optimizer.param_groups[0]['lr']:.6f}"
                    ])

            optimizer.step()

        scheduler.step()
        model.eval()  # <--- 明确切成评估模式
        # Accuracy 评估

        train_forget_acc = evaluate_acc(model, train_forget_dl, device)[1]
        test_poison_acc = evaluate_acc(model, test_poison_dl, device)[1]
        test_clean_acc = evaluate_acc(model, test_clean_dl, device)[1]

        # 打印结果
        print(f"After epoch {epoch}:\n"

              f"Train Forget Acc:  {train_forget_acc:.4f} \n"
              f"Test Poison Acc:   {test_poison_acc:.4f} \n"
              f"Test Clean Acc:    {test_clean_acc:.4f} \n"

              )


        # 写入日志
        with open(log_csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, train_forget_acc,
                test_poison_acc, test_clean_acc,
                optimizer.param_groups[0]['lr'],
                round(time.time() - start, 2),
            ])
    # 只在最后一个 epoch 统计一次
    hist_after = compute_pred_class_hist(model, train_forget_dl, device, num_classes=10, normalize=True)
    save_hist_to_csv(hist_after, hist_csv_path, row_name="after_with_def")
    plot_target_confidence_distribution(
                model, train_forget_dl, device, num_epochs,
                save_dir=save_path, dataset_name=model_type,
                label_mode='gt', threshold=0.9,y_limit=len(train_forget_dl.dataset),coefficient_loss_forget_ce=coefficient_loss_forget_ce,coefficient_loss_forget_data_kl=coefficient_loss_forget_data_kl,coefficient_loss_forget_label_kl=coefficient_loss_forget_label_kl
            )


def GA_loss(
    num_epochs,
    model,
    train_retain_dl,
    train_forget_dl,
    test_poison_dl,
    test_clean_dl,
    device,
    learning_rate,
    forget_cover_clean_ratio,
    diverse_loss_type,
    coefficient_loss_retain,
    coefficient_loss_forget_ce,
    coefficient_loss_forget_data_kl,
    coefficient_loss_forget_label_kl,
    save_path='.',  # 默认当前目录保存日志
    model_type='neggrad_plus',
    clip_num=1,
    batch_size=128,
    warmup_epoch=2,
    mask=None,
    **kwargs,
):
    set_all_seeds(42)
    print(f"Unlearning ({model_type}) Result: \n")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    unlearning_data = LossData(forget_data=train_forget_dl.dataset, retain_data=[])
    generator = torch.Generator()
    generator.manual_seed(42)
    training_loader = DataLoader(unlearning_data, batch_size=batch_size, shuffle=True, pin_memory=True,generator=generator)

    log_csv_path = os.path.join(save_path, f"{model_type}_log_ratio_{forget_cover_clean_ratio}_{diverse_loss_type}_{coefficient_loss_forget_ce}_{coefficient_loss_forget_data_kl}_{coefficient_loss_forget_label_kl}.csv")
    # ✅ 创建并写入表头


    with open(log_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 'train_forget_acc',
            'test_poison_acc', 'test_clean_acc',
            'lr', 'epoch_time'
        ])

    base_log_path = os.path.join(save_path, "base_log_epoch-1.csv")
    if base_log_path and os.path.exists(base_log_path):
        with open(base_log_path, mode='r') as base_f:
            lines = list(csv.reader(base_f))
            if len(lines) >= 2:
                epoch_minus1_row = lines[1]
                with open(log_csv_path, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(epoch_minus1_row)
                print("[Info] Inserted epoch=-1 row from base_log.")
            else:
                print("[Warning] base_log_path does not contain epoch -1 row.")

    batch_log_csv_path = os.path.join(
        save_path,
        f"{model_type}_log_ratio_{forget_cover_clean_ratio}_{diverse_loss_type}_{coefficient_loss_forget_ce}_{coefficient_loss_forget_data_kl}_{coefficient_loss_forget_label_kl}_batch_loss_details.csv"
    )

    # 覆盖写入并写表头
    with open(batch_log_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 'batch',
            'total_loss', 'retain_ce', 'forget_ce',
            'forget_reg_data_kl', 'forget_reg_label_kl',
            'lr'
        ])

    hist_csv_path = os.path.join(
        save_path,
        f"{model_type}_forget_class_hist_{forget_cover_clean_ratio}_{diverse_loss_type}_{coefficient_loss_forget_ce}_{coefficient_loss_forget_data_kl}_{coefficient_loss_forget_label_kl}.csv"
    )
    hist_before = compute_pred_class_hist(model, train_forget_dl, device, num_classes=10, normalize=True)
    save_hist_to_csv(hist_before, hist_csv_path, row_name="before")

    for epoch in range(num_epochs):
        start = time.time()
        print("epoch", epoch)
        model.train()  # <--- 明确切成训练模式
        for i, batch in enumerate(training_loader):
            loss, loss_retain, loss_forget_ce, loss_forget_reg_data_kl,loss_forget_reg_label_kl = training_step_diverse(model, batch, criterion, epoch, coefficient_loss_retain,
                                  coefficient_loss_forget_ce,coefficient_loss_forget_data_kl,coefficient_loss_forget_label_kl,warmup_epoch)
            optimizer.zero_grad()
            loss.backward()

            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]
            if clip_num:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_num)

            if i % 100 == 0:
                print(
                    f"[Epoch {epoch} | Batch {i}] "
                    f"Total Loss: {loss.item():.4f} | "
                    f"Retain CE: {loss_retain.item():.4f} | "
                    f"Forget CE: {loss_forget_ce.item():.4f} | "
                    f"Forget Reg data kl: {loss_forget_reg_data_kl.item():.4f} | "
                    f"Forget Reg label kl: {loss_forget_reg_label_kl.item():.4f}"
                )

                with open(batch_log_csv_path, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        epoch, i,
                        f"{loss.item():.4f}",
                        f"{loss_retain.item():.4f}",
                        f"{loss_forget_ce.item():.4f}",
                        f"{loss_forget_reg_data_kl.item():.4f}",
                        f"{loss_forget_reg_label_kl.item():.4f}",
                        f"{optimizer.param_groups[0]['lr']:.6f}"
                    ])

            optimizer.step()

        scheduler.step()
        model.eval()  # <--- 明确切成评估模式
        # Accuracy 评估

        train_forget_acc = evaluate_acc(model, train_forget_dl, device)[1]
        test_poison_acc = evaluate_acc(model, test_poison_dl, device)[1]
        test_clean_acc = evaluate_acc(model, test_clean_dl, device)[1]

        # 打印结果
        print(f"After epoch {epoch}:\n"

              f"Train Forget Acc:  {train_forget_acc:.4f} \n"
              f"Test Poison Acc:   {test_poison_acc:.4f} \n"
              f"Test Clean Acc:    {test_clean_acc:.4f} \n"

              )


        # 写入日志
        with open(log_csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, train_forget_acc,
                test_poison_acc, test_clean_acc,
                optimizer.param_groups[0]['lr'],
                round(time.time() - start, 2),
            ])
    # 只在最后一个 epoch 统计一次
    hist_after = compute_pred_class_hist(model, train_forget_dl, device, num_classes=10, normalize=True)
    save_hist_to_csv(hist_after, hist_csv_path, row_name="after_with_def")
    plot_target_confidence_distribution(
                model, train_forget_dl, device, num_epochs,
                save_dir=save_path, dataset_name=model_type,
                label_mode='gt', threshold=0.9,y_limit=len(train_forget_dl.dataset),coefficient_loss_forget_ce=coefficient_loss_forget_ce,coefficient_loss_forget_data_kl=coefficient_loss_forget_data_kl,coefficient_loss_forget_label_kl=coefficient_loss_forget_label_kl
            )

def RL_Salun_loss(
    num_epochs,
    model,
    train_retain_dl,
    train_forget_dl,
    test_poison_dl,
    test_clean_dl,
    device,
    learning_rate,
    forget_cover_clean_ratio,
    diverse_loss_type,
    coefficient_loss_retain,
    coefficient_loss_forget_ce,
    coefficient_loss_forget_data_kl,
    coefficient_loss_forget_label_kl,
    save_path='.',  # 默认当前目录保存日志
    model_type='neggrad_plus',
    clip_num=1,
    batch_size=128,
    warmup_epoch=2,
    mask=None,
    **kwargs,
):
    set_all_seeds(42)
    print(f"Unlearning ({model_type}) Result: \n")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    num_classes = 10
    unlearninglabels = list(range(num_classes))
    unlearning_trainset = []

    for sample in train_forget_dl.dataset:
        x, clabel = sample[0], sample[1].item()
        rnd = random.choice(unlearninglabels)
        while rnd == clabel:
            rnd = random.choice(unlearninglabels)
        # unlearning_trainset.append((x, rnd))
        unlearning_trainset.append((x, torch.tensor(rnd).long()))

    for sample in train_retain_dl.dataset:
        x, y = sample[0], sample[1].item()
        unlearning_trainset.append((x, torch.tensor(y, dtype=torch.long)))

    g = torch.Generator()
    g.manual_seed(42)
    unlearning_train_set_dl = DataLoader(
        unlearning_trainset, batch_size=batch_size, pin_memory=True, shuffle=True, generator=g
    )
    unlearning_data = LossData(forget_data=train_forget_dl.dataset, retain_data=unlearning_train_set_dl.dataset)

    generator = torch.Generator()
    generator.manual_seed(42)
    training_loader = DataLoader(unlearning_data, batch_size=batch_size, shuffle=True, pin_memory=True,generator=generator)

    log_csv_path = os.path.join(save_path, f"{model_type}_log_ratio_{forget_cover_clean_ratio}_{diverse_loss_type}_{coefficient_loss_forget_ce}_{coefficient_loss_forget_data_kl}_{coefficient_loss_forget_label_kl}.csv")
    # ✅ 创建并写入表头


    with open(log_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 'train_forget_acc',
            'test_poison_acc', 'test_clean_acc',
            'lr', 'epoch_time'
        ])

    base_log_path = os.path.join(save_path, "base_log_epoch-1.csv")
    if base_log_path and os.path.exists(base_log_path):
        with open(base_log_path, mode='r') as base_f:
            lines = list(csv.reader(base_f))
            if len(lines) >= 2:
                epoch_minus1_row = lines[1]
                with open(log_csv_path, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(epoch_minus1_row)
                print("[Info] Inserted epoch=-1 row from base_log.")
            else:
                print("[Warning] base_log_path does not contain epoch -1 row.")

    batch_log_csv_path = os.path.join(
        save_path,
        f"{model_type}_log_ratio_{forget_cover_clean_ratio}_{diverse_loss_type}_{coefficient_loss_forget_ce}_{coefficient_loss_forget_data_kl}_{coefficient_loss_forget_label_kl}_batch_loss_details.csv"
    )

    # 覆盖写入并写表头
    with open(batch_log_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 'batch',
            'total_loss', 'retain_ce', 'forget_ce',
            'forget_reg_data_kl', 'forget_reg_label_kl',
            'lr'
        ])

    hist_csv_path = os.path.join(
        save_path,
        f"{model_type}_forget_class_hist_{forget_cover_clean_ratio}_{diverse_loss_type}_{coefficient_loss_forget_ce}_{coefficient_loss_forget_data_kl}_{coefficient_loss_forget_label_kl}.csv"
    )
    hist_before = compute_pred_class_hist(model, train_forget_dl, device, num_classes=10, normalize=True)
    save_hist_to_csv(hist_before, hist_csv_path, row_name="before")

    for epoch in range(num_epochs):
        start = time.time()
        print("epoch", epoch)
        model.train()  # <--- 明确切成训练模式
        for i, batch in enumerate(training_loader):
            loss, loss_retain, loss_forget_ce, loss_forget_reg_data_kl,loss_forget_reg_label_kl = training_step_diverse(model, batch, criterion, epoch, coefficient_loss_retain,
                                  coefficient_loss_forget_ce,coefficient_loss_forget_data_kl,coefficient_loss_forget_label_kl,warmup_epoch)
            optimizer.zero_grad()
            loss.backward()

            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]
            if clip_num:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_num)

            if i % 100 == 0:
                print(
                    f"[Epoch {epoch} | Batch {i}] "
                    f"Total Loss: {loss.item():.4f} | "
                    f"Retain CE: {loss_retain.item():.4f} | "
                    f"Forget CE: {loss_forget_ce.item():.4f} | "
                    f"Forget Reg data kl: {loss_forget_reg_data_kl.item():.4f} | "
                    f"Forget Reg label kl: {loss_forget_reg_label_kl.item():.4f}"
                )

                with open(batch_log_csv_path, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        epoch, i,
                        f"{loss.item():.4f}",
                        f"{loss_retain.item():.4f}",
                        f"{loss_forget_ce.item():.4f}",
                        f"{loss_forget_reg_data_kl.item():.4f}",
                        f"{loss_forget_reg_label_kl.item():.4f}",
                        f"{optimizer.param_groups[0]['lr']:.6f}"
                    ])

            optimizer.step()

        scheduler.step()
        model.eval()  # <--- 明确切成评估模式
        # Accuracy 评估

        train_forget_acc = evaluate_acc(model, train_forget_dl, device)[1]
        test_poison_acc = evaluate_acc(model, test_poison_dl, device)[1]
        test_clean_acc = evaluate_acc(model, test_clean_dl, device)[1]

        # 打印结果
        print(f"After epoch {epoch}:\n"

              f"Train Forget Acc:  {train_forget_acc:.4f} \n"
              f"Test Poison Acc:   {test_poison_acc:.4f} \n"
              f"Test Clean Acc:    {test_clean_acc:.4f} \n"

              )


        # 写入日志
        with open(log_csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, train_forget_acc,
                test_poison_acc, test_clean_acc,
                optimizer.param_groups[0]['lr'],
                round(time.time() - start, 2),
            ])
    # 只在最后一个 epoch 统计一次
    hist_after = compute_pred_class_hist(model, train_forget_dl, device, num_classes=10, normalize=True)
    save_hist_to_csv(hist_after, hist_csv_path, row_name="after_with_def")
    plot_target_confidence_distribution(
                model, train_forget_dl, device, num_epochs,
                save_dir=save_path, dataset_name=model_type,
                label_mode='gt', threshold=0.9,y_limit=len(train_forget_dl.dataset),coefficient_loss_forget_ce=coefficient_loss_forget_ce,coefficient_loss_forget_data_kl=coefficient_loss_forget_data_kl,coefficient_loss_forget_label_kl=coefficient_loss_forget_label_kl
            )
def write_initial_log_entry(model, device, log_csv_path,
                            train_forget_dl, test_poison_dl, test_clean_dl):
    # 在追加模式下写入 epoch=-1
    with open(log_csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)

        train_forget_acc = evaluate_acc(model, train_forget_dl, device)[1]
        test_poison_acc = evaluate_acc(model, test_poison_dl, device)[1]
        test_clean_acc = evaluate_acc(model, test_clean_dl, device)[1]

        print(f"Before unlearning: \n"

              f"Train Forget Acc:  {train_forget_acc:.4f} \n"
              f"Test Poison Acc:   {test_poison_acc:.4f} \n"
              f"Test Clean Acc:    {test_clean_acc:.4f} \n"

              )
        writer.writerow([
            -1, train_forget_acc,
            test_poison_acc, test_clean_acc,
            0.0, 0.0
        ])



set_all_seeds(42)
model = generate_cls_model(model_name="preactresnet18", num_classes=10, image_size=32)
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint["model"])
model.to(device)
model.eval()

# log path
log_csv_path = os.path.join(save_path, f"base_log_epoch-1.csv")
# 写入表头 + epoch -1 行
with open(log_csv_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'epoch', 'train_forget_acc',
            'test_poison_acc', 'test_clean_acc',
            'lr', 'epoch_time'
    ])

write_initial_log_entry(model, device, log_csv_path,
                             train_forget_dl, test_poison_dl, test_clean_dl)


def training_step(model, batch, criterion, device):
    images, clabels = batch
    images, clabels = images.to(device), clabels.to(device)
    out = model(images)  # Generate predictions
    loss = criterion(out, clabels)  # Calculate loss
    _, pred = torch.max(out, 1)
    num_correct = (pred == clabels).sum()
    acc = num_correct.item() / len(clabels)
    return loss, acc, num_correct


def fit_one_cycle(
        epochs, model, train_retain_dl, train_forget_dl,
        test_poison_dl, test_clean_dl, device,
        lr, milestones, forget_cover_clean_ratio,
        mask=None, save_path='.', model_type='Missing', clip_num=None, probe_epochs=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=5e-4)

    scheduler = (
        torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
        if milestones else
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    )

    log_csv_path = os.path.join(save_path, f"{model_type}_log_ratio_{forget_cover_clean_ratio}.csv")
    with open(log_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 'train_forget_acc',
            'test_poison_acc', 'test_clean_acc',
            'lr', 'epoch_time'
        ])

    base_log_path = os.path.join(save_path, "base_log_epoch-1.csv")
    if base_log_path and os.path.exists(base_log_path):
        with open(base_log_path, mode='r') as base_f:
            lines = list(csv.reader(base_f))
            if len(lines) >= 2:
                epoch_minus1_row = lines[1]
                with open(log_csv_path, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(epoch_minus1_row)
                print("[Info] Inserted epoch=-1 row from base_log.")
            else:
                print("[Warning] base_log_path does not contain epoch -1 row.")

    for epoch in range(epochs):
        start = time.time()
        print("epoch", epoch)
        model.train()  # <--- 明确切成训练模式
        pbar = tqdm(train_retain_dl, total=len(train_retain_dl))

        for batch in pbar:
            loss, acc, correct = training_step(model, batch, criterion, device)
            loss.backward()

            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]

            if clip_num:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_num)

            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()
        model.eval()  # <--- 明确切成评估模式
        # Accuracy 评估

        train_forget_acc = evaluate_acc(model, train_forget_dl, device)[1]
        test_poison_acc = evaluate_acc(model, test_poison_dl, device)[1]
        test_clean_acc = evaluate_acc(model, test_clean_dl, device)[1]

        # 打印结果
        print(f"After epoch {epoch}:\n"

              f"Train Forget Acc:  {train_forget_acc:.4f} \n"
              f"Test Poison Acc:   {test_poison_acc:.4f} \n"
              f"Test Clean Acc:    {test_clean_acc:.4f} \n"

              )

            # 写入日志
        with open(log_csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, train_forget_acc,
                test_poison_acc, test_clean_acc,
                optimizer.param_groups[0]['lr'],
                round(time.time() - start, 2),
            ])

        # >>> 新增：如果设置了 probe_epochs，就在达到后 break
        if probe_epochs is not None and (epoch + 1) >= probe_epochs:
            print(f"[Probe Mode] Reached probe_epochs={probe_epochs}, stopping early.")
            break
        # <<< 新增结束




def finetune(
        epochs,
        model,
        train_retain_dl,
        train_forget_dl,
        test_poison_dl,
        test_clean_dl,

        device,
        lr,
        milestones,
        forget_cover_clean_ratio,
        mask=None,
        save_path='.',
        model_type='FT',
        clip_num=None,
        probe_epochs=None,

        **kwargs,
):
    # print(f"Unlearning (Finetune) Result: \n")
    fit_one_cycle(
        epochs=epochs, model=model, train_retain_dl=train_retain_dl, train_forget_dl=train_forget_dl,
        test_poison_dl=test_poison_dl, test_clean_dl=test_clean_dl, device=device,
        lr=lr, milestones=milestones, forget_cover_clean_ratio=forget_cover_clean_ratio, mask=mask,
        save_path=save_path, model_type=model_type, clip_num=clip_num, probe_epochs=probe_epochs
    )


def RL(
        epochs,
        model,
        train_retain_dl,
        train_forget_dl,
        test_poison_dl,
        test_clean_dl,
        device,
        lr,
        milestones,
        forget_cover_clean_ratio,
        mask=None,
        save_path='.',
        model_type='FT',
        clip_num=None,
        num_classes=10,
        batch_size=128,
        probe_epochs=None,
        **kwargs,
):
    import random
    from torch.utils.data import DataLoader

    unlearninglabels = list(range(num_classes))
    unlearning_trainset = []

    for sample in train_forget_dl.dataset:
        x, clabel = sample[0], sample[1].item()
        rnd = random.choice(unlearninglabels)
        while rnd == clabel:
            rnd = random.choice(unlearninglabels)
        unlearning_trainset.append((x, rnd))

    for sample in train_retain_dl.dataset:
        x, y = sample[0], sample[1].item()
        unlearning_trainset.append((x, y))

    unlearning_train_set_dl = DataLoader(
        unlearning_trainset, batch_size=batch_size, pin_memory=True, shuffle=True
    )
    train_retain_dl = unlearning_train_set_dl
    # print(f"Unlearning (RL) Result: \n")
    fit_one_cycle(
        epochs=epochs, model=model, train_retain_dl=train_retain_dl, train_forget_dl=train_forget_dl,
        test_poison_dl=test_poison_dl, test_clean_dl=test_clean_dl, device=device, lr=lr, milestones=milestones,
        forget_cover_clean_ratio=forget_cover_clean_ratio, mask=mask, save_path=save_path, model_type=model_type,
        clip_num=clip_num, probe_epochs=probe_epochs

    )


def GA(
        epochs, model, train_retain_dl, train_forget_dl,
        test_poison_dl, test_clean_dl, device,
        lr, milestones, forget_cover_clean_ratio,
        mask=None, save_path='.', model_type='Missing', clip_num=None, probe_epochs=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=5e-4)


    log_csv_path = os.path.join(save_path, f"{model_type}_log_ratio_{forget_cover_clean_ratio}.csv")
    with open(log_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 'train_forget_acc',
            'test_poison_acc', 'test_clean_acc',
            'lr', 'epoch_time'
        ])

    base_log_path = os.path.join(save_path, "base_log_epoch-1.csv")
    if base_log_path and os.path.exists(base_log_path):
        with open(base_log_path, mode='r') as base_f:
            lines = list(csv.reader(base_f))
            if len(lines) >= 2:
                epoch_minus1_row = lines[1]
                with open(log_csv_path, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(epoch_minus1_row)
                print("[Info] Inserted epoch=-1 row from base_log.")
            else:
                print("[Warning] base_log_path does not contain epoch -1 row.")

    for epoch in range(epochs):
        start = time.time()
        print("epoch", epoch)
        model.train()  # <--- 明确切成训练模式

        for i, (image, target) in enumerate(train_forget_dl):
            image = image.to(device)
            target = target.to(device)

            output_clean = model(image)
            loss = - criterion(output_clean, target)

            optimizer.zero_grad()
            loss.backward()
            if clip_num:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_num)
            optimizer.step()
        model.eval()  # <--- 明确切成评估模式
        # Accuracy 评估

        train_forget_acc = evaluate_acc(model, train_forget_dl, device)[1]
        test_poison_acc = evaluate_acc(model, test_poison_dl, device)[1]
        test_clean_acc = evaluate_acc(model, test_clean_dl, device)[1]

        # 打印结果
        print(f"After epoch {epoch}:\n"

              f"Train Forget Acc:  {train_forget_acc:.4f} \n"
              f"Test Poison Acc:   {test_poison_acc:.4f} \n"
              f"Test Clean Acc:    {test_clean_acc:.4f} \n"

              )

        # 写入日志
        with open(log_csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, train_forget_acc,
                test_poison_acc, test_clean_acc,
                optimizer.param_groups[0]['lr'],
                round(time.time() - start, 2),
            ])

        # >>> 新增：如果设置了 probe_epochs，就在达到后 break
        if probe_epochs is not None and (epoch + 1) >= probe_epochs:
            print(f"[Probe Mode] Reached probe_epochs={probe_epochs}, stopping early.")
            break
        # <<< 新增结束




class LossData(Dataset):
    def __init__(self, forget_data, retain_data):
        super().__init__()
        self.forget_data = forget_data
        self.retain_data = retain_data
        self.forget_len = len(forget_data)
        self.retain_len = len(retain_data)

    def __len__(self):
        return self.retain_len + self.forget_len

    def __getitem__(self, index):
        if index < self.forget_len:
            x, y = self.forget_data[index]
            label = 1
            return x, y, label
        else:
            x, y = self.retain_data[index - self.forget_len]
            label = 0
            return x, y, label


### ga_plus
def training_step_ga_plus(model, batch, criterion):
    device = next(model.parameters()).device
    images, clabels, labels = batch
    images, clabels, labels = images.to(device), clabels.to(device), labels.to(device)

    out = model(images)

    retain_mask = (labels == 0)
    forget_mask = (labels == 1)

    retain_logits = out[retain_mask]
    retain_clabels = clabels[retain_mask]
    loss_retain = criterion(retain_logits, retain_clabels)

    forget_logits = out[forget_mask]
    forget_clabels = clabels[forget_mask]
    loss_forget = criterion(forget_logits, forget_clabels)

    loss = loss_retain - 0.15 * loss_forget  # Calculate loss
    return loss, loss_retain, loss_forget


def neggrad_plus(
        epochs, model, train_retain_dl, train_forget_dl,
        test_poison_dl, test_clean_dl, device,
        lr, milestones, forget_cover_clean_ratio,
        mask=None, save_path='.', model_type='Missing', clip_num=None, probe_epochs=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    unlearning_data = LossData(forget_data=train_forget_dl.dataset, retain_data=train_retain_dl.dataset)
    training_loader = DataLoader(unlearning_data, batch_size=128, shuffle=True, pin_memory=True)


    log_csv_path = os.path.join(save_path, f"{model_type}_log_ratio_{forget_cover_clean_ratio}.csv")
    with open(log_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 'train_forget_acc',
            'test_poison_acc', 'test_clean_acc',
            'lr', 'epoch_time'
        ])

    base_log_path = os.path.join(save_path, "base_log_epoch-1.csv")
    if base_log_path and os.path.exists(base_log_path):
        with open(base_log_path, mode='r') as base_f:
            lines = list(csv.reader(base_f))
            if len(lines) >= 2:
                epoch_minus1_row = lines[1]
                with open(log_csv_path, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(epoch_minus1_row)
                print("[Info] Inserted epoch=-1 row from base_log.")
            else:
                print("[Warning] base_log_path does not contain epoch -1 row.")

    for epoch in range(epochs):
        start = time.time()
        print("epoch", epoch)
        model.train()  # <--- 明确切成训练模式
        for i, batch in enumerate(training_loader):
            loss, _, _ = training_step_ga_plus(model, batch, criterion)
            optimizer.zero_grad()
            loss.backward()
            if clip_num:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_num)
            optimizer.step()
        scheduler.step()
        model.eval()  # <--- 明确切成评估模式
        # Accuracy 评估

        train_forget_acc = evaluate_acc(model, train_forget_dl, device)[1]
        test_poison_acc = evaluate_acc(model, test_poison_dl, device)[1]
        test_clean_acc = evaluate_acc(model, test_clean_dl, device)[1]

        # 打印结果
        print(f"After epoch {epoch}:\n"

              f"Train Forget Acc:  {train_forget_acc:.4f} \n"
              f"Test Poison Acc:   {test_poison_acc:.4f} \n"
              f"Test Clean Acc:    {test_clean_acc:.4f} \n"

              )

        # 写入日志
        with open(log_csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, train_forget_acc,
                test_poison_acc, test_clean_acc,
                optimizer.param_groups[0]['lr'],
                round(time.time() - start, 2),
            ])

        # >>> 新增：如果设置了 probe_epochs，就在达到后 break
        if probe_epochs is not None and (epoch + 1) >= probe_epochs:
            print(f"[Probe Mode] Reached probe_epochs={probe_epochs}, stopping early.")
            break
        # <<< 新增结束




def save_gradient_ratio(data_loaders, model, criterion, args, clip_num, forget_cover_clean_ratio):
    """
    Generate gradient-based masks and save them.
    data_loaders: dict with keys 'forget' and optional 'retain'
    model: the torch model
    criterion: loss function (e.g., CrossEntropyLoss)
    args: Namespace or dict with attributes:
        unlearn_lr, momentum, weight_decay, save_dir
    """
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.unlearn_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    gradients = {}
    forget_loader = data_loaders["forget"]
    model.eval()

    for name, param in model.named_parameters():
        gradients[name] = 0

    for i, batch in enumerate(forget_loader):
        image = batch[0]
        target = batch[1]
        image = image.cuda()
        target = target.cuda()

        output_clean = model(image)
        loss = -criterion(output_clean, target)

        optimizer.zero_grad()
        loss.backward()
        if clip_num:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_num)
        optimizer.step()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gradients[name] += param.grad.data

    with torch.no_grad():
        for name in gradients:
            gradients[name] = torch.abs(gradients[name])

    # threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    threshold_list = [0.5]

    for i in threshold_list:
        sorted_dict_positions = {}
        hard_dict = {}

        all_elements = torch.cat([tensor.flatten() for tensor in gradients.values()])
        threshold_index = int(len(all_elements) * i)

        positions = torch.argsort(all_elements)
        ranks = torch.argsort(positions)

        start_index = 0
        for key, tensor in gradients.items():
            num_elements = tensor.numel()
            tensor_ranks = ranks[start_index: start_index + num_elements]
            sorted_positions = tensor_ranks.reshape(tensor.shape)
            sorted_dict_positions[key] = sorted_positions

            threshold_tensor = torch.zeros_like(tensor_ranks)
            threshold_tensor[tensor_ranks < threshold_index] = 1
            threshold_tensor = threshold_tensor.reshape(tensor.shape)
            hard_dict[key] = threshold_tensor
            start_index += num_elements

        all_gradients = torch.cat([gradient.flatten() for gradient in gradients.values()])
        sigmoid_gradients = torch.abs(2 * (torch.sigmoid(all_gradients) - 0.5))
        tanh_gradients = torch.abs(torch.tanh(all_gradients))

        sigmoid_soft_dict = {}
        tanh_soft_dict = {}

        start_idx = 0
        for net_name, gradient in gradients.items():
            num_params = gradient.numel()
            end_idx = start_idx + num_params

            sigmoid_gradient = sigmoid_gradients[start_idx:end_idx].reshape(gradient.shape)
            sigmoid_soft_dict[net_name] = sigmoid_gradient

            tanh_gradient = tanh_gradients[start_idx:end_idx].reshape(gradient.shape)
            tanh_soft_dict[net_name] = tanh_gradient

            start_idx = end_idx

        # Save masks
        os.makedirs(args.save_dir, exist_ok=True)
        torch.save(sigmoid_soft_dict, os.path.join(args.save_dir, f"sigmoid_soft_mask_{forget_cover_clean_ratio}.pt"))
        torch.save(tanh_soft_dict, os.path.join(args.save_dir, f"tanh_soft_mask_{forget_cover_clean_ratio}.pt"))
        torch.save(hard_dict, os.path.join(args.save_dir, f"hard_mask_{forget_cover_clean_ratio}.pt"))


def salun(
        epochs,
        model,
        train_retain_dl,
        train_forget_dl,
        test_poison_dl,
        test_clean_dl,
        device,
        lr,
        milestones,
        forget_cover_clean_ratio,
        mask=None,
        save_path='.',
        model_type='FT',
        clip_num=None,
        num_classes=10,
        batch_size=128,
        probe_epochs=None,
        **kwargs,
):
    import random
    from torch.utils.data import DataLoader

    unlearninglabels = list(range(num_classes))
    unlearning_trainset = []

    for sample in train_forget_dl.dataset:
        x, clabel = sample[0], sample[1].item()
        rnd = random.choice(unlearninglabels)
        while rnd == clabel:
            rnd = random.choice(unlearninglabels)
        unlearning_trainset.append((x, rnd))

    for sample in train_retain_dl.dataset:
        x, y = sample[0], sample[1].item()
        unlearning_trainset.append((x, y))

    unlearning_train_set_dl = DataLoader(
        unlearning_trainset, batch_size=batch_size, pin_memory=True, shuffle=True
    )
    train_retain_dl = unlearning_train_set_dl
    # print(f"Unlearning (salun) Result: \n")
    fit_one_cycle(
        epochs=epochs, model=model, train_retain_dl=train_retain_dl, train_forget_dl=train_forget_dl,
        test_poison_dl=test_poison_dl, test_clean_dl=test_clean_dl, device=device, lr=lr, milestones=milestones,
        forget_cover_clean_ratio=forget_cover_clean_ratio, mask=mask, save_path=save_path, model_type=model_type,
        clip_num=clip_num, probe_epochs=probe_epochs

    )


# ================== UPDATE: add clean-acc drop constraint in Phase-A ==================

phaseA_log_path = os.path.join(save_path, "phaseA_log.txt")


def log_and_print(msg):
    """既打印到屏幕，也写到 phaseA_log.txt"""
    print(msg)
    with open(phaseA_log_path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


import copy


def compute_F_score(acc_before, acc_after, num_classes):
    chance = 1.0 / float(num_classes)
    denom = max(acc_before - chance, 1e-8)
    return max(0.0, min(1.0, (acc_before - acc_after) / denom))


def evaluate_asr(model, test_poison_dl, device):
    return evaluate_acc(model, test_poison_dl, device)[1]  # 0~1


def evaluate_forget_acc(model, train_forget_dl, device):
    return evaluate_acc(model, train_forget_dl, device)[1]  # 0~1


def evaluate_clean_acc(model, test_clean_dl, device):
    return evaluate_acc(model, test_clean_dl, device)[1]  # 0~1


#
# # ================== END UPDATE ==================


# ================== 共享设置 ==================
shared_lr_candidates = [0.01,0.001,0.005, 0.02]
probe_epochs = 2
F_drop_thresh = 0.20
ASR_thresh = 0.50
CLEAN_MAX_DROP = 0.10
num_classes = 10

acc_before_forget = evaluate_forget_acc(model, train_forget_dl, device)
clean_acc_before = evaluate_clean_acc(model, test_clean_dl, device)

log_and_print(f"train_retain_dl: {len(train_retain_dl.dataset)}")
log_and_print(f"train_forget_dl: {len(train_forget_dl.dataset)}")
log_and_print(f"dl_poison_train: {len(dl_poison_train.dataset)}")
log_and_print(f"test_poison_dl: {len(test_poison_dl.dataset)}")
log_and_print(f"test_clean_dl: {len(test_clean_dl.dataset)}")
# ================== 工具：评估 + 探针函数（每种方法一份） ==================
def _eval_and_log(tag, lr_try):
    acc_after_forget = evaluate_forget_acc(model, train_forget_dl, device)
    clean_acc_after = evaluate_clean_acc(model, test_clean_dl, device)
    F = compute_F_score(acc_before_forget, acc_after_forget, num_classes)
    ASR = evaluate_asr(model, test_poison_dl, device)
    clean_drop_pp = clean_acc_before - clean_acc_after
    log_and_print(f"[Phase-A Probe][{tag}] lr={lr_try:.6f} | F={F:.2f} | ASR={ASR:.2f} | CleanΔ={clean_drop_pp:.2f}")
    ok = (F >= F_drop_thresh) and (ASR >= ASR_thresh) and (clean_drop_pp <= CLEAN_MAX_DROP)
    return ok


# ---------- FT ----------
def try_lr_list_FT(lr_list):
    for lr_try in lr_list:
        set_all_seeds(BASE_SEED)
        model.load_state_dict(init_state)
        finetune(
            epochs=10,  # 探针期内部仍用 probe_epochs 控制
            model=model,
            train_retain_dl=train_retain_dl,
            train_forget_dl=train_forget_dl,
            test_poison_dl=test_poison_dl,
            test_clean_dl=test_clean_dl,

            device=device,
            lr=lr_try,
            milestones=None,
            forget_cover_clean_ratio=forget_cover_clean_ratio,
            mask=None,
            save_path=save_path,
            model_type=f'FT-probe-lr{lr_try}',
            clip_num=2,
            probe_epochs=probe_epochs,
        )
        if _eval_and_log("FT", lr_try):
            log_and_print(f"[Phase-A SUCCESS][FT] lr* = {lr_try:.6f}")
            return lr_try
    return None


# ---------- RL ----------
def try_lr_list_RL(lr_list):
    for lr_try in lr_list:
        set_all_seeds(BASE_SEED)
        model.load_state_dict(init_state)
        RL(
            epochs=10,  # 探针期
            model=model,
            train_retain_dl=train_retain_dl,
            train_forget_dl=train_forget_dl,
            test_poison_dl=test_poison_dl,
            test_clean_dl=test_clean_dl,

            device=device,
            lr=lr_try,
            milestones=None,
            forget_cover_clean_ratio=forget_cover_clean_ratio,
            mask=None,
            save_path=save_path,
            model_type=f'RL-probe-lr{lr_try}',
            clip_num=2,
            probe_epochs=probe_epochs,
            num_classes=10,
            batch_size=128
        )
        if _eval_and_log("RL", lr_try):
            log_and_print(f"[Phase-A SUCCESS][RL] lr* = {lr_try:.6f}")
            return lr_try
    return None


# ---------- GA ----------
def try_lr_list_GA(lr_list):
    for lr_try in lr_list:
        set_all_seeds(BASE_SEED)
        model.load_state_dict(init_state)
        GA(
            epochs=5,  # 探针期（GA全程就是5）
            model=model,
            train_retain_dl=train_retain_dl,
            train_forget_dl=train_forget_dl,
            test_poison_dl=test_poison_dl,
            test_clean_dl=test_clean_dl,

            device=device,
            lr=lr_try,
            milestones=None,
            forget_cover_clean_ratio=forget_cover_clean_ratio,
            mask=None,
            save_path=save_path,
            model_type=f'GA-probe-lr{lr_try}',
            clip_num=2,
            probe_epochs=probe_epochs,
        )
        if _eval_and_log("GA", lr_try):
            log_and_print(f"[Phase-A SUCCESS][GA] lr* = {lr_try:.6f}")
            return lr_try
    return None


# ---------- NegGradPlus ----------
def try_lr_list_NegGradPlus(lr_list):
    for lr_try in lr_list:
        set_all_seeds(BASE_SEED)
        model.load_state_dict(init_state)
        neggrad_plus(
            epochs=5,  # 探针期（NegGradPlus全程就是5）
            model=model,
            train_retain_dl=train_retain_dl,
            train_forget_dl=train_forget_dl,
            test_poison_dl=test_poison_dl,
            test_clean_dl=test_clean_dl,

            device=device,
            lr=lr_try,
            milestones=None,
            forget_cover_clean_ratio=forget_cover_clean_ratio,
            mask=None,
            save_path=save_path,
            model_type=f'NegGradPlus-probe-lr{lr_try}',
            clip_num=2,
            probe_epochs=probe_epochs,
        )
        if _eval_and_log("NegGradPlus", lr_try):
            log_and_print(f"[Phase-A SUCCESS][NegGradPlus] lr* = {lr_try:.6f}")
            return lr_try
    return None


# ---------- SaLun ----------

def try_lr_list_SaLun(lr_list):
    for lr_try in lr_list:
        set_all_seeds(BASE_SEED)
        model.load_state_dict(init_state)
        salun(
            epochs=10,  # 探针期
            model=model,
            train_retain_dl=train_retain_dl,
            train_forget_dl=train_forget_dl,
            test_poison_dl=test_poison_dl,
            test_clean_dl=test_clean_dl,

            device=device,
            lr=lr_try,
            milestones=None,
            forget_cover_clean_ratio=forget_cover_clean_ratio,
            mask=mask_hard_mask,
            save_path=save_path,
            model_type=f'SaLun-probe-lr{lr_try}',
            clip_num=2,
            probe_epochs=probe_epochs,
            num_classes=10,
            batch_size=128
        )
        if _eval_and_log("SaLun", lr_try):
            log_and_print(f"[Phase-A SUCCESS][SaLun] lr* = {lr_try:.6f}")
            return lr_try
    return None


# ================== 四段“各自独立”的主流程 ==================

# --- FT ---

lr_star = try_lr_list_FT(shared_lr_candidates)
if lr_star is None:
    expanded_list = [v * 3.0 for v in shared_lr_candidates]
    log_and_print(f"[Phase-A][FT] No LR succeeded in {shared_lr_candidates}. Expanding ×3 -> {expanded_list}")
    lr_star = try_lr_list_FT(expanded_list)

if lr_star is None:
    log_and_print(f"[Phase-A FAIL][FT] No LR met all criteria.")
else:
    set_all_seeds(BASE_SEED)
    model.load_state_dict(init_state)
    finetune(
        epochs=10,  # 正式期 FT=10
        model=model,
        train_retain_dl=train_retain_dl,
        train_forget_dl=train_forget_dl,
        test_poison_dl=test_poison_dl,
        test_clean_dl=test_clean_dl,

        device=device,
        lr=lr_star,
        milestones=None,
        forget_cover_clean_ratio=forget_cover_clean_ratio,
        mask=None,
        save_path=save_path,
        model_type='FT',
        clip_num=2,
        probe_epochs=10,
    )
    set_all_seeds(BASE_SEED)
    model.load_state_dict(init_state)
    FT_NegGradPlus_loss(
        num_epochs=10,
        model=model,
        train_retain_dl=train_retain_dl,
        train_forget_dl=train_forget_dl,
        test_poison_dl=test_poison_dl,
        test_clean_dl=test_clean_dl,
        device=device,
        learning_rate=lr_star,
        forget_cover_clean_ratio=forget_cover_clean_ratio,
        diverse_loss_type='kl',
        coefficient_loss_retain=1,
        coefficient_loss_forget_ce=0,
        coefficient_loss_forget_data_kl=10,
        coefficient_loss_forget_label_kl=10,
        save_path=save_path,  # 默认当前目录保存日志
        model_type='FT',
        clip_num=2,
        batch_size=128, warmup_epoch=1
    )

# --- RL ---
lr_star = try_lr_list_RL(shared_lr_candidates)

if lr_star is None:
    expanded_list = [v * 3.0 for v in shared_lr_candidates]
    log_and_print(f"[Phase-A][RL] No LR succeeded in {shared_lr_candidates}. Expanding ×3 -> {expanded_list}")
    lr_star = try_lr_list_RL(expanded_list)

if lr_star is None:
    log_and_print(f"[Phase-A FAIL][RL] No LR met all criteria.")
else:
    set_all_seeds(BASE_SEED)
    model.load_state_dict(init_state)
    RL(
        epochs=10,  # 正式期 RL=10
        model=model,
        train_retain_dl=train_retain_dl,
        train_forget_dl=train_forget_dl,
        test_poison_dl=test_poison_dl,
        test_clean_dl=test_clean_dl,

        device=device,
        lr=lr_star,
        milestones=None,
        forget_cover_clean_ratio=forget_cover_clean_ratio,
        mask=None,
        save_path=save_path,
        model_type='RL',
        clip_num=2,
        probe_epochs=10,
        num_classes=10,
        batch_size=128
    )
    set_all_seeds(BASE_SEED)
    model.load_state_dict(init_state)

    RL_Salun_loss(
        num_epochs=10,
        model=model,
        train_retain_dl=train_retain_dl,
        train_forget_dl=train_forget_dl,
        test_poison_dl=test_poison_dl,
        test_clean_dl=test_clean_dl,
        device=device,
        learning_rate=lr_star,
        forget_cover_clean_ratio=forget_cover_clean_ratio,
        diverse_loss_type='kl',
        coefficient_loss_retain=1,
        coefficient_loss_forget_ce=0,
        coefficient_loss_forget_data_kl=10,
        coefficient_loss_forget_label_kl=10,
        save_path=save_path,  # 默认当前目录保存日志
        model_type='RL',
        clip_num=2,
        batch_size=128, warmup_epoch=1
    )

# --- GA ---
lr_star = try_lr_list_GA(shared_lr_candidates)

if lr_star is None:
    expanded_list = [v * 3.0 for v in shared_lr_candidates]
    log_and_print(f"[Phase-A][GA] No LR succeeded in {shared_lr_candidates}. Expanding ×3 -> {expanded_list}")
    lr_star = try_lr_list_GA(expanded_list)

if lr_star is None:
    log_and_print(f"[Phase-A FAIL][GA] No LR met all criteria.")
else:
    set_all_seeds(BASE_SEED)
    model.load_state_dict(init_state)
    GA(
        epochs=5,  # 正式期 GA=5
        model=model,
        train_retain_dl=train_retain_dl,
        train_forget_dl=train_forget_dl,
        test_poison_dl=test_poison_dl,
        test_clean_dl=test_clean_dl,

        device=device,
        lr=lr_star,
        milestones=None,
        forget_cover_clean_ratio=forget_cover_clean_ratio,
        mask=None,
        save_path=save_path,
        model_type='GA',
        clip_num=2,
        probe_epochs=5,
    )
    set_all_seeds(BASE_SEED)
    model.load_state_dict(init_state)
    GA_loss(
        num_epochs=5,
        model=model,
        train_retain_dl=train_retain_dl,
        train_forget_dl=train_forget_dl,
        test_poison_dl=test_poison_dl,
        test_clean_dl=test_clean_dl,
        device=device,
        learning_rate=lr_star,
        forget_cover_clean_ratio=forget_cover_clean_ratio,
        diverse_loss_type='kl',
        coefficient_loss_retain=0,
        coefficient_loss_forget_ce=-1,
        coefficient_loss_forget_data_kl=10,
        coefficient_loss_forget_label_kl=10,
        save_path=save_path,  # 默认当前目录保存日志
        model_type='GA',
        clip_num=2,
        batch_size=128, warmup_epoch=1
    )

# --- NegGradPlus ---
lr_star = try_lr_list_NegGradPlus(shared_lr_candidates)

if lr_star is None:
    expanded_list = [v * 3.0 for v in shared_lr_candidates]
    log_and_print(f"[Phase-A][NegGradPlus] No LR succeeded in {shared_lr_candidates}. Expanding ×3 -> {expanded_list}")
    lr_star = try_lr_list_NegGradPlus(expanded_list)

if lr_star is None:
    log_and_print(f"[Phase-A FAIL][NegGradPlus] No LR met all criteria.")
else:
    set_all_seeds(BASE_SEED)
    model.load_state_dict(init_state)
    neggrad_plus(
        epochs=5,  # 正式期 NegGradPlus=5
        model=model,
        train_retain_dl=train_retain_dl,
        train_forget_dl=train_forget_dl,
        test_poison_dl=test_poison_dl,
        test_clean_dl=test_clean_dl,

        device=device,
        lr=lr_star,
        milestones=None,
        forget_cover_clean_ratio=forget_cover_clean_ratio,
        mask=None,
        save_path=save_path,
        model_type='NegGradPlus',
        clip_num=2,
        probe_epochs=5,
    )
    set_all_seeds(BASE_SEED)
    model.load_state_dict(init_state)

    FT_NegGradPlus_loss(
        num_epochs=5,
        model=model,
        train_retain_dl=train_retain_dl,
        train_forget_dl=train_forget_dl,
        test_poison_dl=test_poison_dl,
        test_clean_dl=test_clean_dl,
        device=device,
        learning_rate=lr_star,
        forget_cover_clean_ratio=forget_cover_clean_ratio,
        diverse_loss_type='kl',
        coefficient_loss_retain=1,
        coefficient_loss_forget_ce=-0.15,
        coefficient_loss_forget_data_kl=10,
        coefficient_loss_forget_label_kl=10,
        save_path=save_path,  # 默认当前目录保存日志
        model_type='NegGradPlus',
        clip_num=2,
        batch_size=128, warmup_epoch=1
    )


# --- SaLun ---

# generate mask one time use GA's input

from types import SimpleNamespace

mask_args = SimpleNamespace(
    unlearn_lr=0.01,
    momentum=0.9,
    weight_decay=5e-4,
    save_dir=save_path,
)
criterion = nn.CrossEntropyLoss()
data_loaders = {"forget": train_forget_dl}
set_all_seeds(BASE_SEED)
model.load_state_dict(init_state)  # 你的 checkpoint 快照
model.to(device)
model.eval()
save_gradient_ratio(data_loaders, model, criterion, args=mask_args, clip_num=2,
                    forget_cover_clean_ratio=forget_cover_clean_ratio)
mask_path_hard_mask = os.path.join(save_path, f"hard_mask_{forget_cover_clean_ratio}.pt")
mask_hard_mask = torch.load(mask_path_hard_mask)

lr_star = try_lr_list_SaLun(shared_lr_candidates)

if lr_star is None:
    expanded_list = [v * 3.0 for v in shared_lr_candidates]
    log_and_print(f"[Phase-A][SaLun] No LR succeeded in {shared_lr_candidates}. Expanding ×3 -> {expanded_list}")
    lr_star = try_lr_list_SaLun(expanded_list)

if lr_star is None:
    log_and_print(f"[Phase-A FAIL][SaLun] No LR met all criteria.")
else:
    set_all_seeds(BASE_SEED)
    model.load_state_dict(init_state)
    salun(
                epochs=10,  # 正式期 salun=10
                model=model,
                train_retain_dl=train_retain_dl,
                train_forget_dl=train_forget_dl,
                test_poison_dl=test_poison_dl,
                test_clean_dl=test_clean_dl,

                device=device,
                lr=lr_star,
                milestones=None,
                forget_cover_clean_ratio=forget_cover_clean_ratio,
                mask=mask_hard_mask,
                save_path=save_path,
                model_type='SaLun',
                clip_num=2,
                probe_epochs=10,
                num_classes=10,
                batch_size=128
            )

    set_all_seeds(BASE_SEED)
    model.load_state_dict(init_state)
    RL_Salun_loss(
        num_epochs=10,
        model=model,
        train_retain_dl=train_retain_dl,
        train_forget_dl=train_forget_dl,
        test_poison_dl=test_poison_dl,
        test_clean_dl=test_clean_dl,
        device=device,
        learning_rate=lr_star,
        forget_cover_clean_ratio=forget_cover_clean_ratio,
        diverse_loss_type='kl',
        coefficient_loss_retain=1,
        coefficient_loss_forget_ce=0,
        coefficient_loss_forget_data_kl=10,
        coefficient_loss_forget_label_kl=10,
        save_path=save_path,  # 默认当前目录保存日志
        model_type='Salun',
        clip_num=2,
        batch_size=128, warmup_epoch=1, mask=mask_hard_mask
    )