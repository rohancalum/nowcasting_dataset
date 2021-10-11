from pydantic import BaseModel, Field, validator
from typing import Union
import numpy as np
import xarray as xr
import torch
from nowcasting_dataset.config.model import Configuration


Array = Union[xr.DataArray, torch.Tensor]


class Satellite(BaseModel):

    image_data: Array = Field(
        ...,
        description="Satellites images. Shape: [batch_size,] seq_length, width, height, channel",
    )

    class Config:
        arbitrary_types_allowed = True

    def to_named_tensor(self):
        """Convert this DataArray to a torch.Tensor with named dimensions"""
        self.image_data = self.image_data.torch.to_tensor()
        #  This is not ready yet, when running with 'torch.utils.data.DataLoader'
        # self.image_data = self.image_data.torch.to_named_tensor()


class Batch(BaseModel):

    batch_size: int = Field(
        ...,
        g=0,
        description="The size of this batch. If the batch size is 0, "
        "then this item stores one data item",
    )

    satellite: Satellite


class FakeDataset(torch.utils.data.Dataset):
    """Fake dataset."""

    def __init__(self, configuration: Configuration = Configuration(), length: int = 10):
        """
        Init

        Args:
            configuration: configuration object
            length: length of dataset
        """
        self.batch_size = configuration.process.batch_size
        self.seq_length_5 = (
            configuration.process.seq_len_5_minutes
        )  # the sequence data in 5 minute steps
        self.seq_length_30 = (
            configuration.process.seq_len_30_minutes
        )  # the sequence data in 30 minute steps
        self.satellite_image_size_pixels = configuration.process.satellite_image_size_pixels
        self.nwp_image_size_pixels = configuration.process.nwp_image_size_pixels
        self.number_sat_channels = len(configuration.process.sat_channels)
        self.number_nwp_channels = len(configuration.process.nwp_channels)
        self.length = length

    def __len__(self):
        """Number of pieces of data"""
        return self.length

    def per_worker_init(self, worker_id: int):
        """Not needed"""
        pass

    def __getitem__(self, idx):
        """
        Get item, use for iter and next method

        Args:
            idx: batch index

        Returns: Dictionary of random data

        """

        image_data = np.random.randn(
            self.batch_size,
            self.seq_length_5,
            self.satellite_image_size_pixels,
            self.satellite_image_size_pixels,
            self.number_sat_channels,
        )

        sat = Satellite(image_data=xr.DataArray(data=image_data))
        sat.to_named_tensor()
        batch = Batch(satellite=sat, batch_size=self.batch_size)

        # Note need to return as nested dict
        return batch.dict()


image_data = np.random.randn(5, 12, 32, 32, 10)

sat = Satellite(image_data=xr.DataArray(data=image_data))
sat.to_named_tensor()

train = torch.utils.data.DataLoader(FakeDataset())
i = iter(train)
x = next(i)

x = Batch(**x)
# IT WORKS
assert type(x.satellite.image_data) == torch.Tensor
