# ------------------------------------------------------------
# Base image
# ------------------------------------------------------------
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
WORKDIR /aughye
# ------------------------------------------------------------
# Install necessary packages
# ------------------------------------------------------------
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y tzdata\
    cmake \
    wget \
    vim \
    git \
    sudo \
    openssh-server \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    build-essential \
    libopenmpi-dev \
    curl \      
    gcc \
    python3.10 \
    python3.10-venv \
    python3.10-dev && \
    rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as the default python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# ------------------------------------------------------------
# Install Miniforge # ------------------------------------------------------------
    RUN curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" && \
    bash Miniforge3-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniforge3-Linux-x86_64.sh

ENV PATH=/opt/conda/bin:$PATH

# ------------------------------------------------------------
# Mamba installation and package setup
# ------------------------------------------------------------
RUN mamba install -y python=3.10
RUN mamba install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y


# ------------------------------------------------------------
# Copy requirements files
# ------------------------------------------------------------
# ------------------------------------------------------------
# Install pip dependencies
# ------------------------------------------------------------
RUN pip install --upgrade pip setuptools wheel
RUN pip install scikit-learn==1.2.2
RUN pip install wandb
RUN pip install higher
RUN pip install scikit-build
RUN pip install --no-cache-dir git+https://github.com/thomas-bouvier/horovod.git@compile-cpp17
RUN pip install horovod[pytorch]
RUN pip install torch_geometric
RUN pip install pyg_lib torch_scatter==2.1.2 torch_sparse==0.6.18 torch_cluster==1.6.2 torch_spline_conv==1.2.2 torch_geometric -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
RUN pip install transformers==4.38.2
RUN pip install pymol-open-source
RUN pip install nglview
RUN pip install py3Dmol
RUN pip install scipy==1.12.0
RUN pip install networkx==2.6.3
RUN pip install e3nn==0.5.0
RUN pip install pandas==1.5.1
RUN pip install fair-esm[esmfold]==2.0.0
RUN pip install pybind11==2.11.1
RUN pip install torchmetrics==0.11.0
RUN pip install pyyaml
RUN pip install biopython
RUN pip install spyrmsd
RUN pip install accelerate
RUN pip install lmdb
RUN pip install torchdrug
RUN pip install tensorboard
RUN pip install dgl -f https://data.dgl.ai/wheels/torch-2.1/cu118/repo.html
RUN pip install dglgo==0.0.2 dgllife==0.3.2 dill

RUN pip install einops
RUN pip install openmm
RUN pip install pyrosetta-help
RUN pip install hydra-core
RUN pip install omegaconf
RUN pip install rich
RUN pip install POT
RUN pip install biopandas==0.2.8
RUN pip install joblib==1.1.1

# ESM3
RUN pip install \
    cloudpathlib \
    boto3 \
    biotite \
    msgpack_numpy \
    ipywidgets \
    pygtrie \
    pytest \
    pydssp \
    dna_features_viewer \
    tenacity

# ------------------------------------------------------------
# Add cortex path
# --------------------------------------------------------- ---
WORKDIR /aughye

CMD [ "/bin/bash" ]
