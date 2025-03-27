#!/bin/bash
# 1. Install Miniforge
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh

# 2. Clone the repository
git clone https://github.com/chrisjihee/HybridGNER.git; cd HybridGNER*;

# 3. Create a new environment
conda search conda -c conda-forge
conda install -n base -c conda-forge conda=25.3.0 -y
conda create -n HybridGNER python=3.12 -y
conda activate HybridGNER
conda install -n HybridGNER cuda-libraries=11.8 cuda-libraries-dev=11.8 cudatoolkit=11.8 cuda-cudart=11.8 cuda-cudart-dev=11.8 cuda-nvrtc=11.8 cuda-nvrtc-dev=11.8 cuda-nvcc=11.8 cuda-cccl=11.8 cuda-runtime=11.8 libcusparse=11 libcusparse-dev=11 libcublas=11 libcublas-dev=11 -c nvidia -c pytorch -y
conda list > version-cuda.txt

# 4. Install the required packages
pip install -U -r requirements.txt; pip list | grep torch
rm -rf transformers; git clone https://github.com/chrisjihee/transformers.git; pip install -U -e transformers
rm -rf chrisbase;    git clone https://github.com/chrisjihee/chrisbase.git;    pip install -U -e chrisbase
rm -rf chrisdata;    git clone https://github.com/chrisjihee/chrisdata.git;    pip install -U -e chrisdata
rm -rf progiter;     git clone https://github.com/chrisjihee/progiter.git;     pip install -U -e progiter
export CUDA_HOME=""; DS_BUILD_FUSED_ADAM=1 pip install --no-cache deepspeed; ds_report
MAX_JOBS=40 pip install --no-cache --no-build-isolation --upgrade flash-attn;
pip list | grep -E "torch|transformer|accelerate|deepspeed|flash_attn|numpy|sentencepiece|eval|chris|prog|pydantic" > version-dep.txt

# 5. Prepare the necessary data
cd data/GNER; gzip -d -k pile-ner.jsonl.gz; cd ..
python chrisdata/scripts/ner_sample_jsonl.py
mkdir -p model_outputs
ln -s /dlfs/jhryu/output-dl012 model_outputs-dl012  # for dl012
ln -s /dlfs/jhryu/output-dl026 model_outputs-dl026  # for dl026

# 6. Link HF cache and login to HF
ln -s ~/.cache/huggingface/hub ./.cache
ln -s ~/.cache/huggingface ./.cache_hf
huggingface-cli whoami
huggingface-cli login
