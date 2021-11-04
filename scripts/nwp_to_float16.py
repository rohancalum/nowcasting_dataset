""" Change all nwp files to flaot16 """
import logging
from concurrent import futures

import numpy as np
import xarray as xr

from nowcasting_dataset.filesystem.utils import get_all_filenames_in_path

logging.basicConfig()
_LOG = logging.getLogger("nowcasting_dataset")
_LOG.setLevel(logging.DEBUG)

sets = ["train", "validation", "test"]
data_sources = ["gsp", "metadata", "nwp", "pv", "satellite", "sun", "topographic"]

LOCAL_PATH = (
    "/mnt/storage_b/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/"
    "prepared_ML_training_data/v10"
)

all_filenames = []
for dset in sets:
    data_source = "nwp"
    dir = f"{LOCAL_PATH}/{dset}/{data_source}"
    files = get_all_filenames_in_path(dir)
    files = sorted(files)
    # only get .nc files
    filenames = [file for file in files if ".nc" in file]
    print(f"There are {len(filenames)} to change")
    all_filenames = all_filenames + filenames


def one_file(local_file):
    """Change one nwp file to float 16"""
    # can use this index, only to copy files after a certain number

    print(local_file)

    nwp_data_raw = xr.load_dataset(filename_or_obj=local_file)

    nwp_data_raw.values = nwp_data_raw.values.astype(np.float16)

    encoding = {name: {"compression": "lzf"} for name in nwp_data_raw.data_vars}
    nwp_data_raw.to_netcdf(local_file, engine="h5netcdf", mode="w", encoding=encoding)


# test to see if it works
one_file(list(all_filenames.keys())[0], all_filenames[list(all_filenames.keys())[0]])

# loop over files
with futures.ThreadPoolExecutor(max_workers=2) as executor:
    # Submit tasks to the executor.
    future_examples_per_source = []
    for filename in all_filenames:
        task = executor.submit(one_file, local_file=filename)
        future_examples_per_source.append(task)
