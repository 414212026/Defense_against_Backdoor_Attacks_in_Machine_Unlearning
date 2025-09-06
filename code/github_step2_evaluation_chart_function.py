
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

