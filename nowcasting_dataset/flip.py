import numpy as np
from nowcasting_dataset.example import Example
import logging

logger = logging.getLogger(__name__)


def flip(data: Example):
    """
    Random flips both vertical and horizontal
    """

    vertical_p = np.random.uniform()
    horizontal_p = np.random.uniform()

    if vertical_p > 0.5:
        logger.debug("Flipping vertical")
        data = vertical_flip(data=data)

    if vertical_p > 0.5:
        logger.debug("Flipping horizontal")
        data = horizontal_flip(data=data)

    return data


def vertical_flip(data: Example) -> Example:
    """
    Vertical flip of satellite and nwp data
    """

    # sat_data
    # Shape: [batch_size,] seq_length, width, height, channel
    data["sat_data"] = np.flip(data["sat_data"], 3)

    # Numerical weather predictions (NWPs)
    #: Shape: [batch_size,] channel, seq_length, width, height
    data["nwp"] = np.flip(data["nwp"], 4)

    return data


def horizontal_flip(data: Example) -> Example:
    """
    Horizontal flip of satellite and nwp data
    """

    # sat_data
    # Shape: [batch_size,] seq_length, width, height, channel
    print()
    data["sat_data"] = np.flip(data["sat_data"], 2)

    # Numerical weather predictions (NWPs)
    #: Shape: [batch_size,] channel, seq_length, width, height
    data["nwp"] = np.flip(data["nwp"], 3)

    return data
