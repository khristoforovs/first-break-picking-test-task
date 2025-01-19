import os
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import pandas as pd
from ..models.device import device
from torch import FloatTensor


@dataclass
class FirstBrakeDataset:
    x: npt.NDArray[np.float_]
    t: npt.NDArray[np.float_]
    validation_split: float
    seed: int = 0
    _train_indices: npt.NDArray[np.int_] = None
    _test_indices: npt.NDArray[np.int_] = None
    _rng: np.random.Generator = None

    @property
    def n_samples(self) -> int:
        return self.x.shape[0]

    @property
    def n_samples_train(self) -> int:
        return int(self.n_samples * self.validation_split)

    @property
    def n_samples_test(self) -> int:
        return self.n_samples - self.n_samples_train

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

        self.validation_split = np.clip(self.validation_split, 0.0, 1.0)

        indices = np.arange(self.n_samples)
        self._rng.shuffle(indices)

        self._train_indices = indices[: self.n_samples_train]
        self._test_indices = indices[self.n_samples_train :]

    def sample(
        self,
        batch_size: int,
        device: str = device,
        validation: bool = False,
        as_torch: bool = True,
    ) -> (np.ndarray | FloatTensor, np.ndarray | FloatTensor):
        i_mat = self._test_indices if validation else self._test_indices
        indices = self._rng.integers(low=0, high=i_mat.size, size=batch_size)

        xi = self.x[i_mat[indices], :, :].astype(np.float32)
        ti = self.t[i_mat[indices], :, :].astype(np.float32)
        if as_torch:
            return FloatTensor(xi).to(device), FloatTensor(ti).to(device)
        else:
            return xi, ti

    def sample_indices(
        self,
        indices: np.ndarray,
        device: str = device,
        as_torch: bool = True,
    ) -> (np.ndarray | FloatTensor, np.ndarray | FloatTensor):
        xi = self.x[indices, :].toarray().astype(np.float32)
        ti = self.t[indices, :].astype(np.float32)

        if as_torch:
            return FloatTensor(xi).to(device), FloatTensor(ti).to(device)
        return xi, ti

    @classmethod
    def load(cls, file_path: str, validation_split: float) -> "FirstBrakeDataset":
        df = pd.read_parquet(os.path.join(file_path, r"df.parquet.gzip"))
        df["data_array"] = df.apply(lambda row: row["data_array"].reshape(row["data_array_shape"]), axis=1)

        def make_zero_padding(xi: npt.NDArray[np.float_], n_traces: int):
            result = np.zeros((xi.shape[0], n_traces), dtype=xi.dtype)
            result[:, :xi.shape[1]] = xi
            return result

        df["spare1_index"] = df.apply(
            lambda row: np.argmin(np.abs(row["time_grid"][:, None] - row["spare1"][None, :]), axis=0),
            axis=1
        )

        def make_spare1_mask(row: pd.Series):
            xi = row["spare1_index"]
            shape = row["data_array"].shape
            result = np.zeros(shape, dtype=np.float_)
            for i, spare1i in enumerate(xi):
                result[spare1i:, i] = 1
            return result

        df["spare1_mask"] = df.apply(make_spare1_mask, axis=1)


        df["traces_number"] = df["data_array"].apply(lambda x: x.shape[1])
        max_traces_number = df["traces_number"].max()

        df["data_array"] = df["data_array"].apply(lambda x: make_zero_padding(x, max_traces_number))
        x = np.stack(df["data_array"].values)
        df["spare1_mask"] = df["spare1_mask"].apply(lambda x: make_zero_padding(x, max_traces_number))
        t = np.stack(df["spare1_mask"].values)

        x = 10.0 * np.clip(x, -0.1, 0.1)

        return cls(x=x[:, None, :, :], t=t[:, None, :, :], validation_split=validation_split)


pass
