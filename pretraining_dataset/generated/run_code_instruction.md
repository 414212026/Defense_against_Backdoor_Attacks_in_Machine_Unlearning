
This document provides step-by-step guidance on how to reproduce the **pretraining data generation** process using the [UBA-Inf repository](https://github.com/Huangzirui1206/UBA-Inf/releases/tag/v1.0).  
We strictly followed the authors’ official tutorial ([link to appendix](https://secartifacts.github.io/usenixsec2024/appendix-files/sec24winterae-final97.pdf)) and added notes based on our own experience (especially for running on Windows).  



---

## 1. Download the Repository
Clone or download the [UBA-Inf v1.0 release](https://github.com/Huangzirui1206/UBA-Inf/releases/tag/v1.0).  
This repository contains the codebase and configuration files needed to generate datasets.

---

## 2. Create the Conda Environment
Run the following commands in **Anaconda Prompt**:

```
conda create -n uba-inf python=3.8 -y
conda activate uba-inf
```
Your shell prompt should change from:

```
(base) C:\Users\YourName>
```
to:
```
(uba-inf) C:\Users\YourName>
```
---
## 3. Activate Conda in Git Bash
   
In the uba-inf folder, start Git Bash and run:
```
source ~/anaconda3/etc/profile.d/conda.sh
conda activate uba-inf
```
The prompt should also display (uba-inf) here.

---
## 4. Install Dependencies

Run the installation script:
```
sh ./sh/install.sh
```
For compatibility, in Python 3.8.10 environments, we recommend modifying the first line of install.sh as follows:

```
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 \
    -f https://download.pytorch.org/whl/torch_stable.html
```

---
## 5. Generate Pretraining Data

According to the official tutorial and demo, you can flexibly switch between different attacks, datasets, and models:

Replace badnet with blend or sig to try different attacks.

Replace cifar10 with tinyimagenet, etc.

Replace preactresnet18 with vgg16, etc.

---
## Example Commands

> ⚠️ Disclaimer: We did our best to reproduce the original environment, but please be aware that some inconsistencies in relative paths exist in the authors’ code. We kindly recommend using **absolute paths** to avoid “file not found” errors.

(a) Generate dataset
> 
```
python -m attack.badnet \
    --yaml_path config/attack/prototype/cifar10.yaml \
    --save_folder_name badnet_dataset \
    --add_cover 1 \
    --epoch 00 \
    --pratio 0.012 \
    --cratio 0.004 \
    --attack_target 6
```
(b) Construct UBA camouflages

```
python uba/uba_inf_cover.py \
    --dataset_folder C:/pycharm/pycharm_project_folder/20250609_1/uba-Inf/record/badnet_dataset \
    --device cuda:0 \
    --ft_epoch 60 \
    --ap_epochs 6
```

(c) Train the pre-unlearning model

```
python -m uba.perturb_attack \
    --yaml_path C:/pycharm/pycharm_project_folder/20250609_1/uba-Inf/config/attack/prototype/cifar10.yaml \
    --dataset_folder C:/pycharm/pycharm_project_folder/20250609_1/uba-Inf/record/badnet_dataset \
    --save_folder_name perturb_badnet_preactresnet \
    --epoch 120 \
    --random_seed 3407 \
    --batch_size 128 \
    --add_cover 1 \
    --model preactresnet18 \
    --device cuda:0
```

---

## Notes on Generated Files

During my own runs, I added some additional steps to **save intermediate experimental data**.  
As a result, the `generated/` folder in this repository contains **a few extra files** compared to the authors’ original outputs.  

Please note:  
- These additional files are **not required** in later experiments.  
- As long as you have all the file names included in the original authors’ folders, the reproduction of experiments will work correctly.  
- If you generate fewer files than shown in this repository, it will **not affect reproducibility**.  

This clarification is only to avoid confusion when comparing directory contents between this repository and your own generated results.
