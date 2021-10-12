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
xr.DataArray --> xr.Dataset --> BatchDataSource --> to_netcdf

from_netcdf --> Batch                       --> to tensor
                    - BatchDataSource1
                    - BatchDataSource2


"""


class PydanticXArrayDataArray(xr.DataArray):
    # Adapted from https://pydantic-docs.helpmanual.io/usage/types/#classes-with-__get_validators__

    __slots__ = []

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        return v


class PydanticXArrayDataSet(xr.Dataset):
    # Adapted from https://pydantic-docs.helpmanual.io/usage/types/#classes-with-__get_validators__

    __slots__ = []

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        return v


class BatchDataSource(BaseModel):
    """Superclass for image data (satellite imagery, NWPs, etc.)"""

    data: PydanticXArrayDataSet

    def to_netcdf(self):
        pass

    def from_netcdf(self):
        pass


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


def convert_data_array_to_dataset(data):

    dims = data.dims
    data = xr.Dataset({"data": data})

    for dim in dims:
        coord = data[dim]
        data[dim] = np.arange(len(coord))

        data[f"{dim}_coords"] = xr.DataArray(coord, coords=[data[dim]], dims=[dim])

    return data


class Satellite(BaseModel):
    data: PydanticXArrayDataArray
    # can validate here satellite data


class BatchSatellite(BatchDataSource):
    data: PydanticXArrayDataSet


def create_image_dataset(dims=("time", "x", "y", "channels")):
    data = create_image_array(dims=dims)

    return convert_data_array_to_dataset(data=data)


class Example(BaseModel):
    """A single machine learning training example."""

    satellite: Optional[Satellite]
    # nwp
    # metadata


sat_1 = Satellite(data=create_image_array())
sat_2 = Satellite(data=create_image_array())


satellite_batch = BatchSatellite(
    data=from_list_data_array_to_batch_dataset([sat_1.data, sat_2.data])
)


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


class Batch(BaseModel):

    batch_size: int = Field(
        ...,
        g=0,
        description="The size of this batch. If the batch size is 0, "
        "then this item stores one data item",
    )

    satellite: BatchSatellite

    def to_tensor(self):
        # loop through data_sources, and change to tensors
        pass


batch = Batch(batch_size=2, satellite=satellite_batch)

#
# Array = Union[xr.DataArray, torch.Tensor]
#
#
# class Satellite(BaseModel):
#
#     image_data: Array = Field(
#         ...,
#         description="Satellites images. Shape: [batch_size,] seq_length, width, height, channel",
#     )
#
#     class Config:
#         arbitrary_types_allowed = True
#
#     def to_named_tensor(self):
#         """Convert this DataArray to a torch.Tensor with named dimensions"""
#         self.image_data = TorchAccessor(self.image_data).to_tensor()
#
#
class Batch(BaseModel):

    batch_size: int = Field(
        ...,
        g=0,
        description="The size of this batch. If the batch size is 0, "
        "then this item stores one data item",
    )

    satellite: Satellite


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
