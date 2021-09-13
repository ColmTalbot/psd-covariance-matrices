#!/usr/bin/env python
"""
Plot the maximum contamination per frequency bin for GW170814 data.

This will generate Figure 2.

For more details on the method see https://arxiv.org/abs/2106.13785.
"""

from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
from bilby.core.utils import create_frequency_series

from coarse_psd_matrix.utils import (
    create_parser,
    compute_psd_matrix,
    INTERFEROMETERS,
)

rcParams["font.family"] = "serif"
rcParams["font.serif"] = "Computer Modern Roman"
rcParams["font.size"] = 20
rcParams["text.usetex"] = True
rcParams["grid.alpha"] = 0


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    interferometers = INTERFEROMETERS[args.event]

    frequencies = create_frequency_series(
        sampling_frequency=args.sampling_frequency, duration=args.duration
    )
    keep = (frequencies >= args.minimum_frequency) & (
        frequencies <= args.maximum_frequency
    )
    frequencies = frequencies[keep]

    axis_labels = ["(a)", "(b)", "(c)"]
    fig, axes = plt.subplots(
        nrows=len(interferometers), figsize=(20, 5 * len(interferometers))
    )
    for ii, ifo in enumerate(interferometers):
        plt.sca(axes[ii])
        svd = compute_psd_matrix(
            interferometer_name=ifo,
            event=args.event,
            duration=args.duration,
            sampling_frequency=args.sampling_frequency,
            low_frequency=args.low_frequency,
            tukey_alpha=args.tukey_alpha,
            minimum_frequency=args.minimum_frequency,
            maximum_frequency=args.maximum_frequency,
            medium_duration=args.medium_duration,
            outdir=args.outdir,
        )

        psd_matrix = (svd[0] * svd[1]) @ svd[2]
        psd = abs(psd_matrix).diagonal()

        badness_matrix = psd_matrix.copy() / psd
        np.fill_diagonal(badness_matrix, 0)

        plt.semilogy(
            frequencies,
            np.max(abs(badness_matrix), axis=0),
            label=ifo,
            color=f"C{ii}",
        )
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("$\\Delta_{i}$")
        plt.text(-20, 80, axis_labels[ii])
        plt.xlim(args.minimum_frequency, args.maximum_frequency)
        plt.ylim(3e-2, 100)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("figure_2.pdf")
    plt.close()
