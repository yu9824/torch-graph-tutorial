"""
This program is a copy and modification of the following program under Apache License Version 2.0.
- https://github.com/makaishi2/pythonlibs/blob/9f3d9314531d799803cc82565230822c794c69ee/torch_lib1/__init__.py

LICENSE:
- https://github.com/makaishi2/pythonlibs/blob/main/LICENSE
"""  # noqa: E501

from typing import TypeVar, Union
from collections.abc import Callable, Iterable
import pkgutil

import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib  # noqa: F401

import torch
import torch.nn
import torch.utils.data
import torch_geometric.data

# for type hint
T = TypeVar("T")
DataBatch = Union[torch_geometric.data.Batch, torch_geometric.data.Data]
DataLoader = Union[
    torch_geometric.data.DataLoader,
    Iterable[DataBatch],
]
LossFn = Union[
    torch.nn.modules.loss._Loss,
    Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
]


# 損失関数値計算用
def eval_loss(
    loader: DataLoader,
    net: torch.nn.Module,
    criterion: LossFn,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    # DataLoaderから最初の1セットを取得する
    data = next(iter(loader))

    # デバイスの割り当て
    data = data.to(device)

    # 予測値の計算
    y_pred = net(data)

    #  損失値の計算
    loss = criterion(y_pred, data.y)

    return loss


# 学習用関数
def fit(
    net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: LossFn,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device = torch.device("cpu"),
    num_epochs: int = 100,
    history: np.ndarray = np.empty((0, 3)),
    ipynb: bool = False,
) -> np.ndarray:
    base_epochs = len(history)

    for epoch in range(base_epochs, num_epochs + base_epochs):
        # 1エポックあたりの累積損失(平均化前)
        train_loss, val_loss = 0, 0
        # 1エポックあたりのデータ累積件数
        n_train, n_test = 0, 0

        # 訓練フェーズ
        net.train()

        if any(_module.name == "tqdm" for _module in pkgutil.iter_modules()):
            if ipynb:
                from tqdm.notebook import tqdm
            else:
                from tqdm import tqdm
        else:

            def tqdm(x: T) -> T:
                return x

        # for inputs, labels in tqdm(train_loader):
        for data_train in tqdm(train_loader):
            # 1バッチあたりのデータ件数
            train_batch_size = len(data_train)
            # 1エポックあたりのデータ累積件数
            n_train += train_batch_size

            # GPUヘ転送
            data_train = data_train.to(device)

            # 勾配の初期化
            optimizer.zero_grad()

            # 予測計算
            y_pred_train = net(data_train)

            # 損失計算
            loss = criterion(y_pred_train, data_train.y)

            # 勾配計算
            loss.backward()

            # パラメータ修正
            optimizer.step()

            # 平均前の損失と正解数の計算
            # lossは平均計算が行われているので平均前の損失に戻して加算
            train_loss += loss.item() * train_batch_size

        # 予測フェーズ
        net.eval()

        for data_test in test_loader:
            # 1バッチあたりのデータ件数
            test_batch_size = len(data_test)
            # 1エポックあたりのデータ累積件数
            n_test += test_batch_size

            # GPUヘ転送
            data_test = data_test.to(device)

            # 予測計算
            y_pred_test = net(data_test)

            # 損失計算
            loss_test = criterion(y_pred_test, data_test.y)

            #  平均前の損失と正解数の計算
            # lossは平均計算が行われているので平均前の損失に戻して加算
            val_loss += loss_test.item() * test_batch_size

        # 損失計算
        avg_train_loss = train_loss / n_train
        avg_val_loss = val_loss / n_test
        # 結果表示
        print(
            f"Epoch [{(epoch+1)}/{num_epochs+base_epochs}], "
            f"loss: {avg_train_loss:.5f} "
            f"val_loss: {avg_val_loss:.5f}, "
        )
        # 記録
        item = np.array([epoch + 1, avg_train_loss, avg_val_loss])
        history = np.vstack((history, item))
    return history


# 学習ログ解析
def evaluate_history(history: np.ndarray):
    # 損失と精度の確認
    print(f"初期状態: 損失: {history[0,2]:.5f}")
    print(f"最終状態: 損失: {history[-1,2]:.5f}")

    num_epochs = len(history)
    if num_epochs < 10:
        unit = 1
    else:
        unit = num_epochs / 10

    # 学習曲線の表示 (損失)
    plt.figure(figsize=(9, 8))
    plt.plot(history[:, 0], history[:, 1], "b", label="訓練")
    plt.plot(history[:, 0], history[:, 2], "k", label="検証")
    plt.xticks(np.arange(0, num_epochs + 1, unit))
    plt.xlabel("繰り返し回数")
    plt.ylabel("損失")
    plt.title("学習曲線(損失)")
    plt.legend()
    plt.show()


def torch_seed(seed: int = 123) -> None:
    """PyTorch乱数固定用

    Parameters
    ----------
    seed : int, optional
        , by default 123
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.mps.manual_seed(seed)  # for ARM
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
