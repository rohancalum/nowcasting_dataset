# idea is to test to see if pydantic models in the ml forward step slow down the ml steps.
# worry that data will be moved back and forward to CPU and GPU, which will solve things down
# goes off these comments https://github.com/openclimatefix/nowcasting_dataset/issues/213
import numpy as np
from pydantic import BaseModel, Field
import torch
import time

# ***************
# set up base models
# ***************
class Satellite(BaseModel):

    image_data: torch.Tensor = Field(
        ...,
        description="Satellites images. Shape: [batch_size,] seq_length, width, height, channel",
    )

    class Config:
        arbitrary_types_allowed = True


class NWP(BaseModel):

    image_data: torch.Tensor = Field(
        ...,
        description="Satellites images. Shape: [batch_size,] seq_length, width, height, channel",
    )

    class Config:
        arbitrary_types_allowed = True


class Batch(BaseModel):

    batch_size: int = Field(
        ...,
        g=0,
        description="The size of this batch. If the batch size is 0, "
        "then this item stores one data item",
    )

    satellite: Satellite
    nwp: NWP


# ***************
# set up dataset
# ***************
class FakeDataset(torch.utils.data.Dataset):
    """Fake dataset."""

    def __init__(self, length: int = 10, return_pure_dict: bool = True):

        self.length = length
        self.return_pure_dict = return_pure_dict

        self.batch_size = 32
        self.seq_length_5 = 19
        self.image_size_pixels = 64
        self.number_channels = 10

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

        sat = Satellite(
            image_data=torch.rand(
                self.batch_size,
                self.seq_length_5,
                self.image_size_pixels,
                self.image_size_pixels,
                self.number_channels,
            ),
        )

        nwp = NWP(
            image_data=torch.rand(
                self.batch_size,
                self.seq_length_5,
                self.image_size_pixels,
                self.image_size_pixels,
                self.number_channels,
            ),
        )

        # Note need to return as nested dict
        if self.return_pure_dict:
            return {
                "batch_size": self.batch_size,
                "sat_data": sat.image_data,
                "nwp_data": nwp.image_data,
            }
        else:
            return Batch(satellite=sat, batch_size=self.batch_size, nwp=nwp).dict()


N = 10
# ***************
# test 1
# ***************
print("Pydantic")
dataloader = iter(FakeDataset(return_pure_dict=False))

times = []
i = 0
while i < N:
    t = time.time()
    x = next(dataloader)
    x = Batch(**x)
    t_seconds = time.time() - t
    print(t_seconds)

    i = i + 1
    times.append(t_seconds)

print("average")
print(np.mean(times))


# ***************
# test 2
# ***************
print("Dict")
dataloader = iter(FakeDataset(return_pure_dict=False))

times = []
i = 0
while i < N:
    t = time.time()
    x = next(dataloader)
    t_seconds = time.time() - t
    print(t_seconds)

    i = i + 1
    times.append(t_seconds)

print("average")
print(np.mean(times))
