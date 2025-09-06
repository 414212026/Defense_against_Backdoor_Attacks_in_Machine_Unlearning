# Experimental Protocol and Result Files

## Experimental Methodology
For each attack, we performed a probing stage across learning rates  
\[
[0.001, 0.003, 0.005, 0.01, 0.015, 0.02, 0.03, 0.06].
\]  
The pretraining learning rate was fixed at 0.01, but since most unlearning attempts failed under this setting, we probed additional values in the neighborhood of 0.01.  

The unlearning methods considered include **FT, RL, GA, SaLun, and NegGradPlus** (thanks to Yingdan [link](https://arxiv.org/pdf/2505.10859) for sharing the code).  
If probing showed both *forgetting* and *attack success* trends, we ran the full training schedule for that learning rate:  
- **10 epochs** for FT, RL, and SaLun  
- **5 epochs** for GA and NegGradPlus  

After completing all epochs, if the attack was successful (**ASR > 0.7**), we applied our defense method and evaluated its effectiveness in terms of both forgetting performance and the reduction of attack success.

---
## Results

This directory contains experiment outputs for different datasets, models, and attack types.  
Detailed findings are summarized separately in [findings.md](findings.md).

---
## File Descriptions

- **`phaseA_log.txt`**  
  Probe-stage logs, including forgetting and attack success checks at different learning rates.

- **`base_log_epoch-1.xlsx`**  
  Baseline evaluation *before unlearning*, including accuracy on the forget set, clean set, and ASR.

- **`FT/RL/GA/SaLun/NegGradPlus_log_ratio_1.xlsx`**  
  Results from running the full number of epochs (10 or 5) when probing indicated possible forgetting and attack success.  
  Metrics: accuracy on forget set, clean set, and ASR.

- **`FT/RL/GA/SaLun/NegGradPlus_log_ratio_1_kl_0_10_10.xlsx`**  
  Results with our defense method applied (after successful forgetting and attack).  
  Metrics: accuracy on forget set, clean set, and ASR.

- **`FT/RL/GA/SaLun/NegGradPlus_log_ratio_1_kl_0_10_10_batch_loss_details.xlsx`**  
  Batch-wise loss values during defense training.

- **`FT/RL/GA/SaLun/NegGradPlus_forget_class_hist_1_kl_0_10_10.xlsx`**  
  Label distribution of all forget-set samples after completing defense training.

- **`FT/RL/GA/SaLun/NegGradPlus_0_10_10_gt_epoch10.png`**  
  Distribution of model confidence on the ground-truth labels after defense training.

---

## Notes
- All result files are organized by dataset, model, and attack type in subdirectories under `result/`.
