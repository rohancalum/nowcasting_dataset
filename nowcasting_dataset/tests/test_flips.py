from nowcasting_dataset.flip import flip, horizontal_flip, vertical_flip
import numpy as np


def test_is_vertical_flip():

    width = 10
    height = 10
    batch_size = 32
    seq_length = 19
    n_nwp_channels = 14

    data = {
        "sat_data": np.random.random((batch_size, seq_length, width, height, 3)),
        "nwp": np.random.random((batch_size, seq_length, n_nwp_channels, width, height)),
    }

    original_data = data.copy()

    data = vertical_flip(data)

    assert (data["sat_data"][0, 0, 0, -1, :] == original_data["sat_data"][0, 0, 0, 0, :]).all()
    assert (data["nwp"][0, 0, 0, :, -1] == original_data["nwp"][0, 0, 0, :, 0]).all()


def test_is_horizontal_flip():

    width = 10
    height = 10
    batch_size = 32
    seq_length = 19
    n_nwp_channels = 14

    data = {
        "sat_data": np.random.random((batch_size, seq_length, width, height, 3)),
        "nwp": np.random.random((batch_size, seq_length, n_nwp_channels, width, height)),
    }

    original_data = data.copy()

    data = horizontal_flip(data)

    assert (data["sat_data"][0, 0, -1, 0, :] == original_data["sat_data"][0, 0, 0, 0, :]).all()
    assert (data["nwp"][0, 0, 0, -1, :] == original_data["nwp"][0, 0, 0, 0, :]).all()


def test_flip():
    width = 10
    height = 10
    batch_size = 32
    seq_length = 19
    n_nwp_channels = 14

    data = {
        "sat_data": np.random.random((batch_size, seq_length, width, height, 3)),
        "nwp": np.random.random((batch_size, seq_length, n_nwp_channels, width, height)),
    }

    original_data = data.copy()

    _ = flip(data)

    old = original_data["nwp"][0, 0, 0]
    new = data["nwp"][0, 0, 0]

    # Either
    # no flip
    # flip horizontal
    # vertical flip
    # vertical and horizontal flip
    assert (
        (new[:, :] == old[:, :]).all()
        or (new[-1, :] == old[0, :]).all()
        or (new[:, -1] == old[:, -1]).all()
        or (new[-1, -1] == old[0, 0]).all()
    )
