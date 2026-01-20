# AugHyE
Official implementation of "AugHyE: flexible aligned unbound structure augmentation with hybrid encoder for robust protein binding interface prediction"

**TL;DR:** We propose AugHyE, a novel framework that integrates unbound structure augmentation with a
Hybrid Encoder to enhance model robustness by training on diverse unbound structures. 

## Abstract
Motivation: Protein binding interface prediction is fundamental to understanding biological processes and accelerating drug discovery. Recent deep learning methods have been developed to identify residues involved in protein interactions and have achieved significant performance improvements. However, their performance often degrades when applied to unbound structures, since these models are typically trained only on bound- state structures and insufficiently capture the conformational flexibility inherent to unbound structures. 

Results: We propose AugHyE, a novel framework that integrates unbound structure augmentation with a Hybrid Encoder to enhance model robustness by training on diverse unbound structures. Our approach leverages a protein language model to directly generate unbound structures from protein sequences and incorporates an alignment network to ensure biophysically plausible relative positioning between the independently generated unbound ligand and receptor structures. These aligned augmented unbound structures are combined with native bound structures as a unified bound-unbound dataset, which is used to train the Hybrid Encoder that integrates local geometric features with global structural context. Our method, evaluated on the Docking Benchmark 5.5 dataset, achieves state-of-the-art performance across multiple structural scenarios, demonstrating its robustness to diverse protein conformations.

## Docker:
### Requirements
- Docker Installation
- NVIDIA Container Toolkit (for GPU support)

### Step 1. Build Docker Image
Make sure to run the build command in the directory containing the Dockerfile.

You can simply build docker image by Dockerfile the following command:
```bash
docker build -t [IMAGE_NAME] .
```

### Step 2. Run Container (GPU)
```bash
docker run -it --gpus all -v [PROJECT_DIR]:/[WORKSPACE] [IMAGE_NAME]
```

### Step 3. Install mamba framework (GPU-version)
Install `mamba-ssm` and `causal-conv1d` **after running the container with GPU**  
(do NOT install during `docker build`).
```bash
pip install numpy==1.23.5
pip install --no-build-isolation causal-conv1d==1.4.0
pip install --no-build-isolation mamba-ssm[causal-conv1d]==2.2.2
```

## Conda activation:
A Conda virtual environment setup will be available.

## Notes on Reproducibility

This environment is tested with the following **key packages 
- Python 3.10 
- PyTorch 2.1.0 (CUDA 11.8)
- Torch_geometric
- dgl
- mamba-ssm 2.2.2
- numpy 1.23.5

## Experiments:
You can simply test run the AugHyE test code by using the following command:
```bash
python test.py
```
Data construction & unbound structure augmentation code will be availible.

## Dataset:
Our test dataset for this project can be found here: [Open in Colab](https://drive.google.com/drive/folders/1FDCqtaoE3U4c_k3Zzu75uGhORUABAnkS?usp=drive_link)

Training & validation dataset will be availible.

### Important Note
As of January 20, 2026, the test code has been updated and is working correctly.

If it does not work, please contact us by email. (deokjoong@korea.ac.kr)
