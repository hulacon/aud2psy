import numpy as np
import pytest

from aud2psy.grid import Grid


def test_for_duration_counts_windows():
    assert Grid.for_duration(3.0, 0.5).n_windows == 6
    assert Grid.for_duration(3.1, 0.5).n_windows == 7
    assert Grid.for_duration(0.2, 0.5).n_windows == 1  # short clips get one window


def test_centers_are_window_midpoints():
    grid = Grid.for_duration(1.5, 0.5)
    np.testing.assert_allclose(grid.centers, [0.25, 0.75, 1.25])


def test_average_bins_by_window():
    grid = Grid(hop=1.0, n_windows=3)
    times = np.array([0.1, 0.6, 1.2, 2.5])
    values = np.array([1.0, 3.0, 5.0, 7.0])
    np.testing.assert_allclose(grid.average(times, values), [2.0, 5.0, 7.0])


def test_average_is_nan_aware():
    grid = Grid(hop=1.0, n_windows=2)
    times = np.array([0.2, 0.7, 1.5])
    values = np.array([2.0, np.nan, np.nan])
    out = grid.average(times, values)
    assert out[0] == 2.0  # NaN excluded from window mean
    assert np.isnan(out[1])  # all-NaN window stays NaN


def test_average_empty_window_is_nan():
    grid = Grid(hop=0.5, n_windows=4)
    out = grid.average(np.array([0.1]), np.array([1.0]))
    assert out[0] == 1.0
    assert np.isnan(out[1:]).all()


def test_average_clips_overhanging_frames():
    grid = Grid(hop=1.0, n_windows=2)
    out = grid.average(np.array([2.3]), np.array([9.0]))  # past the last edge
    assert out[1] == 9.0


def test_rate_counts_events_per_second():
    grid = Grid(hop=0.5, n_windows=4)
    rates = grid.rate(np.array([0.1, 0.2, 1.6]))
    np.testing.assert_allclose(rates, [4.0, 0.0, 0.0, 2.0])


def test_rate_empty():
    grid = Grid(hop=0.5, n_windows=3)
    np.testing.assert_allclose(grid.rate(np.array([])), np.zeros(3))


def test_bad_hop_raises():
    with pytest.raises(ValueError):
        Grid.for_duration(1.0, 0.0)


def test_window_max():
    grid = Grid(hop=1.0, n_windows=3)
    times = np.array([0.1, 0.6, 1.2, 2.5])
    values = np.array([1.0, 3.0, np.nan, 7.0])
    out = grid.window_max(times, values)
    assert out[0] == 3.0
    assert np.isnan(out[1])  # only a NaN native frame -> empty window
    assert out[2] == 7.0
