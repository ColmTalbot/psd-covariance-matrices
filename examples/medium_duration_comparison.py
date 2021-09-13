#!/usr/bin/env python
"""
Plot the coarsened PSD for different "infinite-duration" PSD segment lengths.

This will generate Figure 10.

For more details on the method see https://arxiv.org/abs/2106.13785.
"""

import matplotlib.pyplot as plt
from bilby.core.utils import create_frequency_series

from coarse_psd_matrix.utils import fetch_psd_data, create_parser

from matplotlib import rcParams

rcParams["font.family"] = "serif"
rcParams["font.serif"] = "Computer Modern Roman"
rcParams["font.size"] = 20
rcParams["text.usetex"] = True
rcParams["grid.alpha"] = 0

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    kwargs = dict(
        interferometer_name=args.interferometer,
        event=args.event,
        duration=args.duration,
        sampling_frequency=args.sampling_frequency,
        low_frequency=args.low_frequency,
        tukey_alpha=args.tukey_alpha,
        outdir=args.outdir,
    )
    reference_psd = fetch_psd_data(medium_duration=128, **kwargs)["psd"]
    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = "Computer Modern Roman"
    rcParams["font.size"] = 20
    rcParams["text.usetex"] = True
    rcParams["grid.alpha"] = 0
    frequencies = create_frequency_series(
        sampling_frequency=args.sampling_frequency, duration=args.duration
    )

    plt.figure(figsize=(8, 5))
    for ii, medium_duration in enumerate([8, 16, 32, 64]):
        label = medium_duration

        psd = fetch_psd_data(medium_duration=medium_duration, **kwargs)["psd"]
        rcParams["font.family"] = "serif"
        rcParams["font.serif"] = "Computer Modern Roman"
        rcParams["font.size"] = 20
        rcParams["text.usetex"] = True
        rcParams["grid.alpha"] = 0

        plt.plot(frequencies, abs(psd) / abs(reference_psd), label=f"$D={label}s$")
        plt.xlim(args.minimum_frequency, args.maximum_frequency)
        plt.ylim(1 / 3, 3)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel(f"$S_{{i}} / S^{{D={medium_duration}s}}_{{i}}$")
        plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("figure_10.pdf")
    plt.close()
