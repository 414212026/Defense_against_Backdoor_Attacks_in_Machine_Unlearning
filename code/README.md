# Code Usage (PyCharm Only)

This folder contains **five sequential scripts** to be run in order inside **PyCharm**, plus a **Click & Run** script that wraps all five steps.

> No command-line inputs are required. All paths and hyperparameters are defined at the **top of each .py file** (constants section).  
> If you need to change anything (e.g., model, epochs, KL weight, output dir), edit those constants and click Run.

**Switching to Other Attacks**  
To run a different attack setup, simply modify the following constants inside the scripts:
- `save_path`: set to a new folder name for saving outputs  
- `model_name`: e.g., `"preactresnet18"` → `"vgg16"`  
- `num_classes`: adjust according to the dataset (e.g., `10` for CIFAR-10, `100` for CIFAR-100)  
- `image_size`: adjust according to the dataset (e.g., `32` for CIFAR, `64` for Tiny-ImageNet)  

---
## Before You Start
1) Make sure the pretraining data has been prepared (see `../pretraining_dataset/run_code_instruction.md`) and saved in the uba-Inf\record folder.  

## Run Order (Step 1 → Step 5)
Right-click each file and choose **Run**. Do them **in order**:

1. `github_step1_load_pretrain_data.py`  
   - Loads pretraining data.  

2. `github_step2_evaluation_chart_function.py`  
   - Utility functions for metrics & plotting 

3. `github_step3_unlearning_function.py`  
   - Runs **baseline unlearning**.  

4. `github_step4_defense_function.py`  
   - Runs **defense training** (e.g., KL-regularized unlearning).  

5. `github_step5_defense_result.py`  
   - Aggregates results, compares Baseline vs Defense, and plots.  
   - Final CSV/PNGs saved.

## One-Click (Click & Run)
- `github_defense_against_attacks_on_machine_unlearning_code.py`  
  - Runs **Step 1 → 5** automatically using the constants defined.  
  - Use this when you just want the end-to-end pipeline in one click.

## Reproducibility & Output Layout
- Seeds are fixed inside each script (constants).  
