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
import numpy as np
import pandas as pd
import rdkit
import torch
import torch_geometric
import torch_geometric.data
import torch_geometric.loader
import torch_geometric.nn
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from torch_utils.data import GraphDataset
from torch_utils.model import GCN
from torch_utils.torch_utils import (
    EarlyStopping,
    evaluate_history,
    fit,
    torch_seed,
)
from torch_utils.utils import check_tqdm, yyplot

# %%
torch.__version__


# %%
torch_geometric.__version__


# %%
rdkit.__version__


# %%
seed = 334
batch_size = 256

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
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
dataloader = torch_geometric.loader.DataLoader(
    dataset, batch_size=batch_size, shuffle=True
)
# the following is deprecated
# dataloader = torch_geometric.data.DataLoader(
#     dataset, batch_size=batch_size, shuffle=True
# )
dataloader


# %%
dataset_train, dataset_test = train_test_split(
    dataset, test_size=0.2, random_state=seed
)
dataset_train, dataset_val = train_test_split(
    dataset_train, test_size=0.1, random_state=seed
)
print(len(dataset_train), len(dataset_val), len(dataset_test))


# %%
dataloader_train = torch_geometric.loader.DataLoader(
    dataset_train, batch_size=batch_size, shuffle=True
)
dataloader_val = torch_geometric.loader.DataLoader(
    dataset_val, batch_size=batch_size, shuffle=False
)
dataloader_test = torch_geometric.loader.DataLoader(
    dataset_test, batch_size=batch_size, shuffle=False
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
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device


# %%
early_stopping = EarlyStopping(patience=5, verbose=True)

history = fit(
    model.to(device),
    optimizer=torch.optim.Adam(model.parameters(), lr=0.002),
    criterion=torch.nn.MSELoss(),
    train_loader=dataloader_train,
    val_loader=dataloader_val,
    # num_epochs=10,
    # num_epochs=100,
    num_epochs=50,
    device=device,
    early_stopping=early_stopping,
)


# %%
evaluate_history(history)


# %%
model.training


# %%
_tmp_y = []
_tmp_y_pred = []
for _batchdata_test in dataloader_test:
    _tmp_y.append(_batchdata_test.y)
    _tmp_y_pred.append(model(_batchdata_test.to(device)))


# %%
y_test_scaled = torch.cat(_tmp_y, dim=0)
y_pred_on_test_scaled = torch.cat(_tmp_y_pred, dim=0)


# %%
y_test = y_test_scaled * y_std + y_mean
y_pred_on_test = y_pred_on_test_scaled * y_std + y_mean


# %%
_tmp_y = []
_tmp_y_pred = []
for _batchdata_val in dataloader_val:
    _tmp_y.append(_batchdata_val.y)
    _tmp_y_pred.append(model(_batchdata_val.to(device)))


# %%
y_val_scaled = torch.cat(_tmp_y, dim=0)
y_pred_on_val_scaled = torch.cat(_tmp_y_pred, dim=0)


# %%
y_val = y_val_scaled * y_std + y_mean
y_pred_on_val = y_pred_on_val_scaled * y_std + y_mean


# %%
_tmp_y = []
_tmp_y_pred = []
for _batchdata_train in dataloader_train:
    _tmp_y.append(_batchdata_train.y)
    _tmp_y_pred.append(model(_batchdata_train.to(device)))


# %%
y_train_scaled = torch.cat(_tmp_y, dim=0)
y_pred_on_train_scaled = torch.cat(_tmp_y_pred, dim=0)


# %%
y_train = y_train_scaled * y_std + y_mean
y_pred_on_train = y_pred_on_train_scaled * y_std + y_mean


# %%
ax = yyplot(
    y_train.cpu().detach().numpy(),
    y_pred_on_train.cpu().detach().numpy(),
    y_val.cpu().detach().numpy(),
    y_pred_on_val.cpu().detach().numpy(),
    y_test.cpu().detach().numpy(),
    y_pred_on_test.cpu().detach().numpy(),
)
ax.figure.tight_layout()


# %%
# rdkit記述子を使ってRandomForestで予測した値と比較

dict_feat = dict()
for desc_name, desc_func in check_tqdm(Descriptors.descList):
    dict_feat[desc_name] = [
        desc_func(Chem.MolFromSmiles(smi)) for smi in smiles
    ]
X = pd.DataFrame.from_dict(dict_feat)
X.head()


# %%
X = X.loc[:, ~(X.isnull().any() | np.isinf(X).any())].astype(np.float32)
print(X.shape)
X.head()


# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, np.array(y) * y_std + y_mean, test_size=0.2, random_state=seed
)
rf = RandomForestRegressor(random_state=seed, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_on_test = rf.predict(X_test)
y_pred_on_train = rf.predict(X_train)
yyplot(y_train, y_pred_on_train, y_test, y_pred_on_test)
