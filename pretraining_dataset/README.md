# Pretraining Dataset

This folder contains the **pretraining data** used in our experiments. The data comes from two main sources:

1. **Original data released by UBA-Inf authors**  
   - Downloaded directly from the authors' shared Google Drive https://drive.google.com/drive/u/0/folders/1NMrjpS7TqVHEBtKKF6HXZJSQ8BU8zMpU.  
   - Each folder includes the corresponding configuration text files provided by the authors.
   - Note: the configuration files saved by the authors contain **absolute paths**. To ensure compatibility for loading the pretraining data, subsequent unlearning and defense experiments, you must create the directory  
     ```
     C:\home\zrhuang\UBA-Inf-Artifacts\record
     ```  
     and copy all downloaded datasets (e.g., `badnet_dataset`, `perturb_badnet_preactresnet`, etc.) into this folder. This guarantees that the pretraining data can be properly loaded.  

2. **Data generated using UBA-Inf open-source code and instructions**  
   - We utilized the authors’ official implementation and generated the pretraining data by adjusting parameters such as poison ratio, cover ratio, attack type, dataset, and model.
   - Additional notes on **Windows-specific settings** and how to run the code are provided in `run_code_instruction.md`.
