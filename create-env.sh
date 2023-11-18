#!/bin/bash

env_name="torch39"
conda create -y -n ${env_name} python=3.9 -c conda-forge
conda install -y -n ${env_name} -c pytorch pytorch::pytorch torchvision torchaudio
conda install -y -n ${env_name} -c conda-forge scikit-learn lightning jupyter umap-learn matplotlib seaborn pandas rdkit tqdm optuna graphviz torchinfo tabulate tensorboard

$HOME/miniforge3/envs/${env_name}/bin/pip install --no-deps japanize-matplotlib torch_geometric torchviz
