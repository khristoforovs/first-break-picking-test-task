import h5py
import pandas as pd
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from tqdm import tqdm


path_to_file = r"D:\Storage\datasets\first-break-recognition-dataset\data\Halfmile3D_add_geom_sorted.hdf5"


f = dict(h5py.File(path_to_file)["TRACE_DATA/DEFAULT"])


def get_rec_ids() -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_], int]:
    rec = np.c_[f["REC_X"][:], f["REC_Y"][:]]
    unique_recs = np.unique(rec, axis=0)
    match_x = np.abs(rec[:, 0][:, None] - unique_recs[:, 0][None, :]) == 0
    match_y = np.abs(rec[:, 1][:, None] - unique_recs[:, 1][None, :]) == 0
    return rec, np.nonzero(np.logical_and(match_x, match_y))[1], len(unique_recs)


rec, rec_ids, n_unique_recs = get_rec_ids()


def prepare_data_array(receiver_ids: npt.NDArray[np.int_], n_unique_receivers: int) -> list[npt.NDArray[np.float_]]:
    data_array = f["data_array"][:]
    result = []
    for i in tqdm(range(n_unique_receivers)):
        result.append(data_array[receiver_ids == i, :])
    return result


def prepare_shot_id(receiver_ids: npt.NDArray[np.int_], n_unique_receivers: int):
    result = []
    for i in tqdm(range(n_unique_receivers)):
        result.append(f["SHOTID"][:][receiver_ids == i, :])
    return result


def prepare_spare1(receiver_ids: npt.NDArray[np.int_], n_unique_receivers: int):
    result = []
    for i in tqdm(range(n_unique_receivers)):
        result.append(f["SPARE1"][:][receiver_ids == i, :])
    return result


df = pd.DataFrame(
    {
        "data_array": prepare_data_array(rec_ids, n_unique_recs),
        "rec_coord": [rec[np.where(rec_ids==i)[0][0], :] for i in range(n_unique_recs)],
        "shot_id": prepare_shot_id(rec_ids, n_unique_recs),
        "spare1": prepare_spare1(rec_ids, n_unique_recs),
    }
)


pass








plt.figure()
plt.pcolor(data_array[:1000, :].T, cmap="gray", clim=(-0.5, 0.5))
plt.plot(f["SPARE1"][:][:1000], 'r')
plt.gca().invert_yaxis()
plt.colorbar()
plt.show()