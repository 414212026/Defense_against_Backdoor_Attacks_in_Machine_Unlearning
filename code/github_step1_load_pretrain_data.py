
# Windows/Linux：Ctrl + Alt + S ->Build, Execution, Deployment → Console → Python Console ->Environment variables
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

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