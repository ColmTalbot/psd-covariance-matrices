#!/usr/bin/env python
"""
Plot the SVD eigenvalues for GW170814.

This will generate Figure 3.

For more details on the method see https://arxiv.org/abs/2106.13785.
"""

import numpy as np
import matplotlib.pyplot as plt

from coarse_psd_matrix.utils import create_parser, compute_psd_matrix, INTERFEROMETERS

from matplotlib import rcParams

rcParams["font.family"] = "serif"
rcParams["font.serif"] = "Computer Modern Roman"
rcParams["font.size"] = 20
rcParams["text.usetex"] = True
rcParams["grid.alpha"] = 0

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    interferometers = INTERFEROMETERS[args.event]

    plt.figure(figsize=(8, 5))
    for ii, ifo in enumerate(interferometers):
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
        plt.semilogy(svd[1], label=ifo, color=f"C{ii}")
        psd_matrix = (svd[0] * svd[1]) @ svd[2]
        psd = abs(psd_matrix.diagonal())
        truncation = min(np.where(svd[1] < min(psd))[0])
        plt.axvline(truncation, color=f"C{ii}", linestyle=":")

    plt.axhline(-1, linestyle=":", color="k", label="min(PSD) Truncation")
    plt.axvline(
        int(len(svd[1]) * (1 - 5 * args.tukey_alpha / 8)),
        color="k",
        linestyle="--",
        label="Theoretical Truncation",
    )
    plt.xlim(0, len(svd[1]))
    plt.ylim(1e-47, 1e-40)
    plt.xlabel("Eigenmode number")
    plt.ylabel("Eigenvalue")
    plt.legend(loc="upper center")
    plt.tight_layout()
    plt.savefig("figure_3.pdf")
    plt.close()
