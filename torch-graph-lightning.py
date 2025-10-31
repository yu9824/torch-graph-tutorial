# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: pytorch39
#     language: python
#     name: python3
# ---

# %%
import os

# to avoid the following error when using MPS (GPU in ARM architecture)):
# NotImplementedError: The operator 'aten::scatter_reduce.two_out'
# is not currently implemented for the MPS device. If you want
# this op to be added in priority during the prototype phase of
# this feature, please comment on
# https://github.com/pytorch/pytorch/issues/77764.
# As a temporary fix, you can set the environment variable
# `PYTORCH_ENABLE_MPS_FALLBACK=1` to use the CPU as
# a fallback for this op.
# WARNING: this will be slower than running natively on MPS.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # before importing torch

print(os.cpu_count())


# %% [markdown]
# \# TODO
#
# - <s>BatchNormalization?</s>
# - ハイパーパラメータチューニング (optuna?)
# - <s>モデルの解釈 (tanhやpoolingについて)</s>
#   - tanh: 活性化関数
#   - pooling: すべての原子の情報を統合する。原子数が異なるためそれらを揃える役割も。
# - <s>dropoutの導入？←過学習対策</s>
# - autumentation?
# - モデルの途中保存 (エポック毎?)
# - エッジの重みも学習する (`nn.Parameter`?)
# - early-stopping
# - max-poolingの検討
# - 転移学習 (fine-tuning and feature-extraction)
# - [GIN層](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINConv.html#torch_geometric.nn.conv.GINConv)を試してみる

# %%
import lightning
import lightning.pytorch.callbacks.early_stopping
import lightning.pytorch.loggers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rdkit
import torch
import torch.utils.data
import torch_geometric
import torch_geometric.data
import torch_geometric.loader
import torch_geometric.nn
from rdkit import Chem
from rdkit.Chem import Descriptors
from torch_utils.data import GraphDataset
from torch_utils.lightning_utils import LightningGCN
from torch_utils.model import GCN
from torch_utils.torch_utils import (
    torch_seed,
)
from torch_utils.utils import yyplot


# %%
torch.__version__


# %%
torch_geometric.__version__


# %%
rdkit.__version__


# %%
seed = 334
batch_size = 64

torch_seed(seed)


# %%
# df_raw = pd.read_csv("./data/curated-solubility-dataset.csv", index_col=0)
df_raw = pd.read_csv("./data/logSdataset1290.csv", index_col=0)
# 計算時間短縮
# df_raw = df_raw.iloc[:1000]
print(df_raw.shape)
df_raw.head()


# %%
# smiles = df_raw["SMILES"]
# y = df_raw["Solubility"]
smiles = df_raw.index
y = df_raw["logS"]


# %%
# scaling
y_mean = y.mean()
y_std = y.std(ddof=1)


# %%
smiles = smiles.tolist()
y = torch.Tensor(((y - y_mean) / y_std).tolist()).view(-1, 1)


# %%
mols = map(Chem.MolFromSmiles, smiles)


# %%
dataset = GraphDataset(mols, y, n_jobs=-1)
dataset


# %%
dataset_train, dataset_test, dataset_val = torch.utils.data.random_split(
    dataset, [0.75, 0.10, 0.15], torch.Generator().manual_seed(seed)
)
print(len(dataset_train), len(dataset_val), len(dataset_test))


# %%
dataloader_train = torch_geometric.loader.DataLoader(
    dataset_train,
    batch_size=batch_size,
    # The 'train_dataloader' does not have many workers which may be a bottleneck.
    # Consider increasing the value of the `num_workers` argument` to `num_workers=7`
    # in the `DataLoader` to improve performance.
    num_workers=os.cpu_count(),
    # Consider setting `persistent_workers=True` in 'train_dataloader'
    # to speed up the dataloader worker initialization.
    persistent_workers=True,
    sampler=torch.utils.data.RandomSampler(
        dataset_train, generator=torch.Generator().manual_seed(seed)
    ),
)
dataloader_val = torch_geometric.loader.DataLoader(
    dataset_val,
    batch_size=batch_size,
    shuffle=False,
    num_workers=os.cpu_count(),
    persistent_workers=True,
)
dataloader_test = torch_geometric.loader.DataLoader(
    dataset_test,
    batch_size=batch_size,
    shuffle=False,
    num_workers=os.cpu_count(),
)


# %%
data = dataset_train[0]
num_features = data.x.shape[1]


# %%
model = GCN(in_channels=num_features, embedding_size=64)
print(model)


# %%
# Need 'tabulate' package
print(torch_geometric.nn.summary(model, data))


# %%
model_lightning = LightningGCN(
    model,
    optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
    criterion=torch.nn.MSELoss(),
)


# %%
early_stopping = lightning.pytorch.callbacks.early_stopping.EarlyStopping(
    monitor="val_loss", patience=5, min_delta=0, mode="min"
)


# %%
# The number of training batches (33) is smaller than the logging interval Trainer(log_every_n_steps=50).
# Set a lower value for log_every_n_steps if you want to see logs for the training epoch.
trainer = lightning.Trainer(
    accelerator="auto",
    max_epochs=100,
    callbacks=[early_stopping],
    logger=(
        lightning.pytorch.loggers.TensorBoardLogger(
            save_dir="logs", name="gcn_tb"
        ),
        lightning.pytorch.loggers.CSVLogger(save_dir="logs", name="gcn_csv"),
    ),
    log_every_n_steps=16,
)
trainer.fit(model_lightning, dataloader_train, dataloader_val)


# %%
trainer.test(model_lightning, dataloader_test)


# %%
dataloader_train_for_predict = torch_geometric.loader.DataLoader(
    dataset_train,
    batch_size=batch_size,
    shuffle=False,
    num_workers=os.cpu_count(),
    persistent_workers=True,
)


# %%
y_train_scaled = torch.cat(
    [_databatch.y for _databatch in dataloader_train_for_predict], dim=0
)
y_pred_train_scaled: torch.Tensor = torch.cat(
    trainer.predict(model_lightning, dataloader_train_for_predict), dim=0
)
y_pred_train_scaled.shape


# %%
y_val_scaled = torch.cat(
    [_databatch.y for _databatch in dataloader_val], dim=0
)
y_pred_val_scaled: torch.Tensor = torch.cat(
    trainer.predict(model_lightning, dataloader_val), dim=0
)
y_pred_val_scaled.shape


# %%
y_test_scaled = torch.cat(
    [_databatch.y for _databatch in dataloader_test], dim=0
)
y_pred_test_scaled: torch.Tensor = torch.cat(
    trainer.predict(model_lightning, dataloader_test), dim=0
)
y_pred_test_scaled.shape


# %%
ax = yyplot(
    y_train_scaled * y_std + y_mean,
    y_pred_train_scaled * y_std + y_mean,
    y_val_scaled * y_std + y_mean,
    y_pred_val_scaled * y_std + y_mean,
    y_test_scaled * y_std + y_mean,
    y_pred_test_scaled * y_std + y_mean,
)


# %%
df_history = pd.read_csv(
    os.path.join(trainer.loggers[1].log_dir, "metrics.csv")
)
df_history.head()


# %%
df_history = df_history.groupby("epoch").mean().reset_index()
df_history.head()


# %%
fig, ax = plt.subplots(facecolor="w")
ax.plot(df_history["epoch"], df_history["train_loss"], label="train")
ax.plot(df_history["epoch"], df_history["val_loss"], label="val")
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
ax.legend()
fig.tight_layout()

