# Defense against Backdoor Attacks in Machine Unlearning

This repository contains exploratory work on defending against camouflage-data-based backdoor attacks in machine unlearning scenarios.

If you have any questions or would like to reference this work, please contact:

Advisor: Ren Wang (rwang74@iit.edu)

For repository-related questions: Yi Qin (yiqin@umich.edu)

---------
## Method: Two-Phase KL-Regularized Unlearning

Our defense introduces a **two-phase unlearning scheme**:

- **Phase I (Warmup, 1 epoch):**  
  Perform *pure base unlearning* (FT / RL / GA / NegGrad+ / SaLun).  
  This ensures the forget set exhibits **observable forgetting** before applying any defense term.  

- **Phase II (Stabilized Unlearning):**  
  Continue unlearning under the same $\mathcal{L}_{base}$, but augment it with our KL regularizer:  



$$
\mathcal{L}_{\text{unlearn}} = \mathcal{L}_{\text{base}} + \lambda_d KL_{\text{data}} + \lambda_l KL_{\text{label}}
$$



**where:**


- $KL_{\text{data}} = \tfrac{1}{B}\sum_{i=1}^{B} KL\left(p(y \mid x_i)\parallel U\right)$  
  This term measures the divergence between each sample’s predictive distribution and the uniform distribution, averaged across the batch. It enforces **instance-level forgetting**, ensuring that every individual sample prediction approaches uniformity.


- $KL_{\text{label}} = KL\Big(\bar{p}(y)\parallel U\Big), \quad \bar{p}(y) = \tfrac{1}{B}\sum_{i=1}^B p(y \mid x_i)$  
  This term measures the divergence between the aggregated class distribution of the batch and the uniform distribution. It enforces **distribution-level forgetting**, ensuring that the overall class predictions remain balanced without collapsing into a single dominant class.

- $U$: the uniform distribution over $C$ classes ($U(y)=1/C$).  
- $\lambda_d, \lambda_l$: regularization weights for the data KL and label KL terms, respectively.

This design discourages the model from producing **over-confident predictions** on the forget set while stabilizing unlearning dynamics.

---------
## Contributions

1. Conducted experiments on several mainstream unlearning methods (FT / RL / GA / NegGrad+ / SaLun) and datasets, showing that our defense method is effective against certain types of attacks.

2. Released the first available defense code for successful camouflage-based poisoning cases.

3. Identified key bottlenecks (e.g., attack reproducibility, dependency on poison ratio), providing insights for future research.


---------
## Repository Structure
```
repo/
├── README.md                      # Main project description and navigation (this file)
│
├── literature_review/             # Survey of related work
│   └── README.md                  # Summary of attacks and defenses in prior work
│
├── pretraining_dataset/           # Pretraining datasets for attacks
│   ├── README.md                  # Overview of dataset sources
│   ├── UBA_inf_original/          # Original data from UBA-Inf authors
│   │   └── README.md              # Path issues and usage notes
│   └── generated/                 # Data generated with UBA-Inf code
│       ├── README.md              # Explains generated data and Google Drive links
│       └── run_code_instruction.md# Instructions for running code to regenerate data
│
├── code/                     # Our defense pipeline
│   ├── github_step1_load_pretrain_data.py
│   ├── github_step2_evaluation_chart_function.py
│   ├── github_step3_unlearning_function.py
│   ├── github_step4_defense_function.py
│   ├── github_step5_defense_result.py
│   ├── github_defense_against_attacks_on_machine_unlearning_code.py  # one-click run
│   └── README.md             # step-by-step instructions (PyCharm)
│
├── results/                       # Experimental results
│   ├── README.md                  # Experimental methodology overview, output file description, and link to findings
│   ├── findings_summary.xlsx      # Raw findings summary table (Excel)
│   ├── findings_summary.png       # Visualized findings summary
│   ├── findings.md                # Key observations and interpretation
│   ├── cifar10-parn18-badnet-0.12poison/   # Example experiment results
│   │   └── ...                    # Logs, charts, CSVs, etc.
│   ├── cifar100-vgg16-badnet-0.05poison/   # Another experiment
│   │   └── ...
│   └── ...                        # Other dataset/attack/model combinations
```



---------
## Citation   
If you find this repository helpful for your work, please consider citing or contacting us. 

---------
## Acknowledgments
- Parts of the implementation (e.g., unlearning baselines FT, RL, GA, SaLun, and NegGradPlus) were based on code kindly shared by Yingdan Shi [link](https://arxiv.org/pdf/2505.10859).
- Pretraining datasets were largely based on the original data released by the UBA-Inf authors [link](https://www.usenix.org/system/files/usenixsecurity24-huang-zirui.pdf).
