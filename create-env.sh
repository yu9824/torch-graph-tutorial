#!/bin/bash

env_name="pytorch39"
conda create -y -n ${env_name} python=3.9 -c conda-forge
conda install -y -n ${env_name} -c pytorch pytorch::pytorch torchvision torchaudio
conda install -y -n ${env_name} -c conda-forge scikit-learn lightning jupyter umap-learn matplotlib seaborn pandas

$HOME/miniforge3/envs/${env_name}/bin/pip install torch_geometric
$HOME/miniforge3/envs/${env_name}/bin/pip install --no-deps japanize-matplotlib
