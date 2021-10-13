from __future__ import annotations
from typing import Optional
from typing import Union, List

import numpy as np
import pandas as pd
import torch
import xarray as xr
from pydantic import BaseModel, Field

from nowcasting_dataset.config.model import Configuration


"""
xr.DataArray --> xr.Dataset --> Batch --> to_netcdf

from_netcdf --> Batch                       --> BatchML
                    - BatchDataSource1
                    - BatchDataSource2
"""


# class PydanticXArrayDataArray(xr.DataArray):
#     # Adapted from https://pydantic-docs.helpmanual.io/usage/types/#classes-with-__get_validators__
#
#     __slots__ = []
#
#     @classmethod
#     def __get_validators__(cls):
#         yield cls.validate
#
#     @classmethod
#     def validate(cls, v):
#         return v


class PydanticXArrayDataSet(xr.Dataset):
    # Adapted from https://pydantic-docs.helpmanual.io/usage/types/#classes-with-__get_validators__

    __slots__ = []

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        return v


if not hasattr(xr.DataArray, "torch"):

    @xr.register_dataarray_accessor("torch")
    class TorchAccessor:
        def __init__(self, xarray_obj):
            self._obj = xarray_obj

        def to_tensor(self):
            """Convert this DataArray to a torch.Tensor"""
            import torch

            return torch.tensor(self._obj.data)

        def to_named_tensor(self):
            """Convert this DataArray to a torch.Tensor with named dimensions"""
            import torch

            return torch.tensor(self._obj.data, names=self._obj.dims)


if not hasattr(xr.Dataset, "torch"):

    @xr.register_dataset_accessor("torch")
    class TorchAccessor:
        def __init__(self, xdataset_obj: xr.Dataset):
            self._obj = xdataset_obj

        def to_tensor(self, dims: List[str]) -> dict:
            """Convert this Dataset to dictionary of torch tensors"""

            torch_dict = {}

            for dim in dims:
                v = getattr(self._obj, dim)
                if "time" == dim:
                    v = v.astype(np.int32)

                torch_dict[dim] = v.torch.to_tensor()

            return torch_dict


def from_list_data_array_to_batch_dataset(image_data_arrays: List[xr.DataArray]) -> xr.Dataset:
    # might need to example dims here

    image_data_arrays = [
        convert_data_array_to_dataset(image_data_arrays[i]) for i in range(len(image_data_arrays))
    ]

    image_data_arrays = [
        image_data_arrays[i].expand_dims(dim="example").assign_coords(example=("example", [i]))
        for i in range(len(image_data_arrays))
    ]

    return xr.concat(image_data_arrays, dim="example")


def convert_data_array_to_dataset(data_xarray):

    dims = data_xarray.dims
    data = xr.Dataset({"data": data_xarray})

    for dim in dims:
        coord = data[dim]
        data[dim] = np.arange(len(coord))

        data = data.rename({dim: f"{dim}_index"})

        data[dim] = xr.DataArray(coord, coords=data[f"{dim}_index"].coords, dims=[f"{dim}_index"])

    return data


def create_image_array(dims=("time", "x", "y", "channels")):
    ALL_COORDS = {
        "time": pd.date_range("2021-01-01", freq="5T", periods=4),
        "x": np.random.randint(low=0, high=1000, size=8),
        "y": np.random.randint(low=0, high=1000, size=8),
        "channels": np.arange(5),
    }
    coords = [(dim, ALL_COORDS[dim]) for dim in dims]
    image_data_array = xr.DataArray(0, coords=coords)  # Fake data for testing!
    return image_data_array


def create_image_dataset(dims=("time", "x", "y", "channels")):
    data = create_image_array(dims=dims)

    return convert_data_array_to_dataset(data=data)


class Satellite(PydanticXArrayDataSet):
    # Use to store xr.Dataset data
    __slots__ = []


class SatelliteML(BaseModel):
    # Use to store data ready for ml
    data: torch.Tensor
    time: torch.Tensor
    x: torch.Tensor
    y: torch.Tensor

    class Config:
        arbitrary_types_allowed = True


class Batch(BaseModel):
    """A batch of xr.Datasets."""

    satellite: Optional[Satellite]
    # nwp
    # metadata

    def to_tensor(self):
        # loop through data_sources, and change to tensors
        pass

    def save_netcdf(self):
        # save to netcdf
        pass

    def load_netcdf(self):
        # load to netcdf
        pass


class BatchML(BaseModel):
    """A batch machine learning training examples."""

    satellite: Optional[SatelliteML]
    # nwp
    # metadata


sat_1 = create_image_array()
sat_2 = create_image_array()


satellite_batch = from_list_data_array_to_batch_dataset([sat_1, sat_2])

satellite_batch_ml = satellite_batch.torch.to_tensor(["data", "time", "x", "y"])
satellite_batch_ml = SatelliteML(**satellite_batch_ml)


batch_ml = Batch(satellite=satellite_batch_ml)


#
# class FakeDataset(torch.utils.data.Dataset):
#     """Fake dataset."""
#
#     def __init__(self, configuration: Configuration = Configuration(), length: int = 10):
#         """
#         Init
#
#         Args:
#             configuration: configuration object
#             length: length of dataset
#         """
#         self.batch_size = configuration.process.batch_size
#         self.seq_length_5 = (
#             configuration.process.seq_len_5_minutes
#         )  # the sequence data in 5 minute steps
#         self.seq_length_30 = (
#             configuration.process.seq_len_30_minutes
#         )  # the sequence data in 30 minute steps
#         self.satellite_image_size_pixels = configuration.process.satellite_image_size_pixels
#         self.nwp_image_size_pixels = configuration.process.nwp_image_size_pixels
#         self.number_sat_channels = len(configuration.process.sat_channels)
#         self.number_nwp_channels = len(configuration.process.nwp_channels)
#         self.length = length
#
#     def __len__(self):
#         """Number of pieces of data"""
#         return self.length
#
#     def per_worker_init(self, worker_id: int):
#         """Not needed"""
#         pass
#
#     def __getitem__(self, idx):
#         """
#         Get item, use for iter and next method
#
#         Args:
#             idx: batch index
#
#         Returns: Dictionary of random data
#
#         """
#
#         image_data = np.random.randn(
#             self.batch_size,
#             self.seq_length_5,
#             self.satellite_image_size_pixels,
#             self.satellite_image_size_pixels,
#             self.number_sat_channels,
#         )
#
#         sat = Satellite(image_data=xr.DataArray(data=image_data))
#         sat.to_named_tensor()
#         batch = Batch(satellite=sat, batch_size=self.batch_size)
#
#         # Note need to return as nested dict
#         return batch.dict()
#
#
# image_data = np.random.randn(5, 12, 32, 32, 10)
#
# sat = Satellite(image_data=xr.DataArray(data=image_data))
# sat.to_named_tensor()
#
# train = torch.utils.data.DataLoader(FakeDataset())
# i = iter(train)
# x = next(i)
#
# x = Batch(**x)
# # IT WORKS
# assert type(x.satellite.image_data) == torch.Tensor
