from collections.abc import Sequence
from typing import SupportsIndex, Optional

import rdkit.Chem.rdchem
from joblib import Parallel, delayed
import torch.utils.data
import torch_geometric.data

from src.feat import mol2data


class GraphDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        mols: Sequence[rdkit.Chem.rdchem.Mol],
        n_jobs: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.mols = mols
        self.n_jobs = n_jobs

        self._data: list[torch_geometric.data.Data] = Parallel(
            n_jobs=self.n_jobs
        )(delayed(mol2data)(mol) for mol in self.mols)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: SupportsIndex) -> torch_geometric.data.Data:
        return self._data[index]
