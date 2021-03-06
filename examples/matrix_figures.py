#!/usr/bin/env python
"""
Plot the relevant matrices for a given event.

This will generate Figures 11-13.

For more details on the method see https://arxiv.org/abs/2106.13785.
"""

import numpy as np
import matplotlib.pyplot as plt

from coarse_psd_matrix.utils import (
    compute_psd_matrix,
    create_parser,
    regularize_eigenvalues,
    regularized_inversion,
    INTERFEROMETERS,
)
from coarse_psd_matrix.plotting import plot_psd_matrix

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

    kwargs = dict(
        minimum_frequency=args.minimum_frequency,
        maximum_frequency=args.maximum_frequency,
        duration=args.duration,
        tick_step=125,
    )

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
        rcParams["font.family"] = "serif"
        rcParams["font.serif"] = "Computer Modern Roman"
        rcParams["font.size"] = 20
        rcParams["text.usetex"] = True
        rcParams["grid.alpha"] = 0
        psd_matrix = (svd[0] * svd[1]) @ svd[2]
        eigenvalues = regularize_eigenvalues(svd[1], fill_value=0)
        regularized_psd_matrix = (svd[0] * eigenvalues) @ svd[2]
        regularized_inverse_psd_matrix = regularized_inversion(svd)

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 16))
        plot_psd_matrix(psd_matrix, axes[0][0], **kwargs, label="C")
        plot_psd_matrix(regularized_psd_matrix, axes[0][1], **kwargs, label="\\bar{C}")
        plot_psd_matrix(svd[0].T, axes[1][0], **kwargs, label="U", origin="upper")
        plt.ylabel("Eigenmode number")
        plt.yticks(np.arange(0, len(svd[1]), 500), np.arange(0, len(svd[1]), 500))
        plot_psd_matrix(
            regularized_inverse_psd_matrix, axes[1][1], **kwargs, label="\\bar{C}^{-1}"
        )
        axes[0][0].text(-400, 2800, "(a)")
        axes[0][1].text(-400, 2800, "(b)")
        axes[1][0].text(-400, 250, "(c)")
        axes[1][1].text(-400, 2800, "(d)")
        plt.tight_layout()
        plt.savefig(f"figure_{11 + ii}.pdf")
        plt.close()
