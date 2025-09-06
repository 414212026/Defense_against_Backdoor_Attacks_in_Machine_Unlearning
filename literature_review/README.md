# Literature Review: Backdoor Attacks and Defenses in Machine Unlearning

## Summary

## Attack Categories

•	Poison dataset + camouflage dataset: [1], [2], [3], [4], [7], [8], [9]

•	Non-poisoning (unlearning data without triggers): [3], [5]

•	Like traditional backdoor attack: [6]

•	Unlearning data not belonging to training data: [10]


## Defense Categories

•	Monitoring changes in model prediction confidence for each sample before and after unlearning: [2], [5]

----

## References

[1] Title: Hidden Poison: Machine Unlearning Enables Camouflaged Poisoning Attacks (NeurIPS 2023)

Authors: Jimmy Z. Di, Jack Douglas, Jayadev Acharya, Gautam Kamath, Ayush Sekhari

Link: https://openreview.net/forum?id=Isy7gl1Hqc&noteId=foS8ELk47F

Code: https://github.com/jimmy-di/camouflage-poisoning

Contribution: Attack (gray-box; clean-label)

Attack: Constructed poison + camouflage datasets using Gradient Matching (Geiping et al.).

[2] Title: Exploiting Machine Unlearning for Backdoor Attacks in Deep Learning System

Authors: Peixin Zhang, Jun Sun, Mingtian Tan, Xinyu Wang 

Link: https://arxiv.org/abs/2310.10659

Code: https://github.com/seartifacts/bau 

Contribution: Attack (black-box; dirty-label) + Defense

Attacks:

•	Poison + camouflage dataset based on input-targeted backdoor attacks.

•	Poison + camouflage dataset based on BadNets.

Defenses:
•	Uncertainty-based defense: poisoned and camouflage samples look similar, but labels differ. The model struggles, producing diffuse prediction distributions.

•	Sub-model similarity (SISA): only sub-models containing camouflage data learn correct labels, while others yield errors or low confidence, resulting in high prediction variance.

[3] Title: Backdoor attacks via machine unlearning (AAAI Conference 2024)

Authors: Zihao Liu, Tianhao Wang, Mengdi Huai, Chenglin Miao

Link: https://ojs.aaai.org/index.php/AAAI/article/view/29321

Contribution: Attack (black-box + white-box; clean-label)

Attacks:

•	Non-poisoning: unlearning data without triggers, selected such that clean samples minimize loss, trigger samples maximize misclassification, and the number of unlearning samples is minimized for stealth.

•	Poison + camouflage dataset: unlearning data includes triggers, constructed by applying fixed perturbations in mid-to-high frequency bands.

[4] Title: UBA-Inf: Unlearning Activated Backdoor Attack with Influence-Driven Camouflage (USENIX Security, 2024)

Authors: Zirui Huang, Yunlong Mao, Sheng Zhong 

Link: https://www.usenix.org/conference/usenixsecurity24/presentation/huang-zirui

Code: https://secartifacts.github.io/usenixsec2024/appendix-files/sec24winterae-final97.pdf

Contribution: Attack (gray-box + black-box)

Attack: Constructed camouflage datasets based on influence scores, sampled from public datasets given access to poisoned samples.

[5] Title: Releasing malevolence from benevolence: The menace of benign data on machine unlearning

Authors: Binhao Ma, Tianhang Zheng, Hongsheng Hu, Di Wang, Shuo Wang, Zhongjie Ba, Zhan Qin, Kui Ren

Link: https://arxiv.org/abs/2407.05112

Code:  https://github.com/Unlearning-Attack/Unlearning-Usability-Attack

Contribution: Attack (black-box; clean-label) + Defense

Attack: Non-poisoning—small amounts of unlearning data with high information density significantly harm model performance after unlearning.

Defenses:

•	Monitor forgetting cost per sample; if anomalously high, flag for manual inspection.

•	Maintain a validation set to assess model performance after unlearning.

[6] Title: Towards Understanding and Enhancing Robustness of Deep Learning Models against Malicious Unlearning Attacks (ACM 2023) 

Authors: Wei Qian , Chenxu Zhao , Wei Le, Meiyi Ma , Mengdi Huai 

Code: https://dl.acm.org/doi/abs/10.1145/3580305.3599526

Contribution: Attack (white-box + black-box; clean-label) + Defense

Attack: Adapted traditional poisoning by treating unlearning as “modifying” images (adding triggers) rather than deleting them.

Defense: Applied traditional backdoor defense (finding isolated medoids).

[7] Title: Trojan Attack on Machine Unlearning: Security Risk of The Right to Be Forgotten (IEEE 2025) 

Authors: Lefeng Zhang, Tianqing Zhu, Ping Xiong, Wanlei Zhou 

Link: https://www.computer.org/csdl/journal/tq/5555/01/10989746/26wpM8Gx7wc

Contribution: Attack (white-box; clean-label)

Attack: Improved Gradient Matching to construct poison + camouflage datasets.

[8] Title: ReVeil: Unconstrained Concealed Backdoor Attack on Deep Neural Networks using Machine Unlearning (DAC, 2025)  

Authors: Manaar Alam, Hithem Lamri, and Michail Maniatakos

Link: https://arxiv.org/pdf/2502.11687

Code: https://github.com/momalab/ReVeil 

Contribution: Attack (black-box)

Attack: Poison + camouflage dataset; camouflage constructed with poisoned samples plus Gaussian perturbations.

[9] Title: Malicious Unlearning in Ensemble Models (IEEE, 2024)

Authors: Huanyi Ye , Ziyao Liu , Yu Jiang , Jiale Guo , Kwok-Yan Lam 

Link: https://ieeexplore.ieee.org/document/10788066

Contribution: Attack (white-box; clean-label)

Attack: Applied Gradient Matching on SISA to construct poison + camouflage datasets.

[10] Title: Unlearn and Burn: Adversarial Machine Unlearning Requests Destroy Model Accuracy (ICLR 2025 Poster)

Authors: Yangsibo Huang, Daogao Liu, Lynn Chua, Badih Ghazi, Pritish Kamath, Ravi Kumar, Pasin Manurangsi, Milad Nasr, Amer Sinha, Chiyuan Zhang 

Link: https://openreview.net/forum?id=5xxGP9x5dZ&noteId=rDmJv1wa4t

Code: https://github.com/daogaoliu/unlearning-under-adversary

Contribution: Attack (white-box + black-box)

Attack: Crafted unlearning requests with data not belonging to training, which significantly degraded model performance.

