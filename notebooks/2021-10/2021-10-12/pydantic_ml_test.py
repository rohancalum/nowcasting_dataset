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

    def __init__(self, length: int = 100, return_pure_dict: bool = True):

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

        if torch.cuda.is_available():
            nwp.image_data = nwp.image_data.cuda()
            sat.image_data = nwp.image_data.cuda()

        # Note need to return as nested dict
        if self.return_pure_dict:
            return {
                "batch_size": self.batch_size,
                "sat_data": sat.image_data,
                "nwp_data": nwp.image_data,
            }
        else:
            return Batch(satellite=sat, batch_size=self.batch_size, nwp=nwp).dict()


N = 50
# ***************
# test 1
# ***************
print("Pydantic")
dataloader = iter(FakeDataset(return_pure_dict=False))
x = next(dataloader)

times_pydantic = []
i = 0
while i < N:
    t = time.time()
    x = next(dataloader)
    x = Batch(**x)
    t_seconds = time.time() - t
    print(t_seconds)

    i = i + 1
    times_pydantic.append(t_seconds)

average_pydantic = np.mean(times_pydantic)
std_pydantic = np.std(times_pydantic)


# ***************
# test 2
# ***************
print("Dict")
dataloader = iter(FakeDataset(return_pure_dict=True))

# from torch.utils.data import DataLoader
# dataloader = iter(DataLoader(FakeDataset(return_pure_dict=False), batch_size=None))

times_dict = []
i = 0
while i < N:
    t = time.time()
    x = next(dataloader)
    t_seconds = time.time() - t
    print(t_seconds)

    i = i + 1
    times_dict.append(t_seconds)

average_dict = np.mean(times_dict)
std_dic = np.std(times_dict)

print("average")
print(f"Pydantic {average_pydantic}")
print(f"Dict {average_dict}")


# *************
# t test
# ********
# is pydantic slower than
# https://machinelearningmastery.com/how-to-code-the-students-t-test-from-scratch-in-python/

from scipy.stats import t

data1 = times_pydantic
data2 = times_dict
alpha = 0.05

# calculate means
mean1, mean2 = np.mean(data1), np.mean(data2)

# calculate sample standard deviations
std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)

# calculate standard errors
n1, n2 = len(data1), len(data2)
se1, se2 = std1 / np.sqrt(n1), std2 / np.sqrt(n2)

# standard error on the difference between the samples
sed = np.sqrt(se1 ** 2.0 + se2 ** 2.0)

# calculate the t statistic
t_stat = (mean1 - mean2) / sed
# degrees of freedom
df = len(data1) + len(data2) - 2
# calculate the critical value
cv = t.ppf(1.0 - alpha, df)
# calculate the p-value
p = (1.0 - t.cdf(abs(t_stat), df)) * 2.0
# return everything
print(t_stat, df, cv, p)

if p > alpha:
    print("Accept null hypothesis that the means are equal.")
else:
    print("Reject the null hypothesis that the means are equal.")
    print(f"Pydantic {average_pydantic}")
    print(f"Dict {average_dict}")


# *************
# results
# ********
# on GPU,  over 50 times,
# pydantic: 25663522243499753
# Dict: 0.2568572759628296

# std
