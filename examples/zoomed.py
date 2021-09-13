#!/usr/bin/env python
"""
Compute the comparison of the analytic and experimental PSD matrices.

This will generate Figure 1.
This is probably the only example that will run in a reasonable time without
a GPU.

For more details on the method see https://arxiv.org/abs/2106.13785.
"""

import numpy as np
import matplotlib.pyplot as plt
from bilby.core.utils import create_white_noise, create_frequency_series
from scipy.signal.windows import tukey
from scipy.interpolate import interp1d
from tqdm.auto import trange

from coarse_psd_matrix.utils import (
    compute_psd_matrix,
    fetch_psd_data,
)
from coarse_psd_matrix.plotting import plot_psd_matrix

from matplotlib import rcParams

rcParams["font.family"] = "serif"
rcParams["font.serif"] = "Computer Modern Roman"
rcParams["font.size"] = 20
rcParams["text.usetex"] = True
rcParams["grid.alpha"] = 0


if __name__ == "__main__":
    outdir = "zoomed"
    duration = 4
    medium_duration = 128
    sampling_frequency = 2048
    low_frequency = 16
    tukey_alpha = 0.1
    minimum_frequency = 480
    maximum_frequency = 530
    event = "GW170814"

    data = fetch_psd_data(
        interferometer_name="L1",
        event=event,
        duration=duration,
        sampling_frequency=sampling_frequency,
        low_frequency=low_frequency,
        tukey_alpha=tukey_alpha,
        medium_duration=medium_duration,
        outdir=outdir,
    )
    svd = compute_psd_matrix(
        interferometer_name="L1",
        event=event,
        duration=duration,
        sampling_frequency=sampling_frequency,
        low_frequency=low_frequency,
        tukey_alpha=tukey_alpha,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
        medium_duration=medium_duration,
        outdir=outdir,
    )
    psd = data["medium_psd"][: sampling_frequency // 2 * medium_duration + 1]
    original_frequencies = create_frequency_series(
        duration=medium_duration, sampling_frequency=sampling_frequency
    )
    new_frequencies = create_frequency_series(
        duration=256, sampling_frequency=sampling_frequency
    )
    psd = interp1d(original_frequencies, psd)(new_frequencies)

    short_window = tukey(duration * sampling_frequency, tukey_alpha)
    short_window /= np.mean(short_window ** 2) ** 0.5

    analytic_psd_matrix = (svd[0] * svd[1]) @ svd[2]
    estimated_psd_matrix = np.zeros_like(analytic_psd_matrix)

    nfft = duration * sampling_frequency
    start_idx = minimum_frequency * duration
    stop_idx = maximum_frequency * duration

    n_average = 1024 * 1024 // 64
    for _ in trange(n_average):
        white_noise, frequencies = create_white_noise(
            sampling_frequency=2048, duration=256
        )
        coloured_noise = white_noise * psd ** 0.5

        td_noise = np.fft.irfft(coloured_noise).reshape((-1, nfft))
        fd_noise = np.fft.rfft(td_noise * short_window)
        reduced_noise = fd_noise[:, start_idx : stop_idx + 1]

        estimated_psd_matrix += np.einsum(
            "ki,kj->ij", reduced_noise, reduced_noise.conjugate()
        )
    estimated_psd_matrix /= n_average
    estimated_psd_matrix /= sampling_frequency / 16

    fig, axes = plt.subplots(nrows=2, figsize=(10, 16))
    kwargs = dict(
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
        duration=duration,
        vmin=-53,
        vmax=-41.8,
        tick_step=10,
    )
    plot_psd_matrix(estimated_psd_matrix, axes[0], **kwargs)
    plot_psd_matrix(analytic_psd_matrix, axes[1], **kwargs)
    axes[0].text(-25, 190, "(a)")
    axes[1].text(-25, 190, "(b)")
    plt.tight_layout()
    plt.savefig("figure_1.pdf")
    plt.show()
    plt.close()
