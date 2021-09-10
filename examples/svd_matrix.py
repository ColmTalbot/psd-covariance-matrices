#!/usr/bin/env python
"""
Plot the relevant matrices for a given event.

This will generate Figure 4.

For more details on the method see https://arxiv.org/abs/2106.13785.
"""

import matplotlib.pyplot as plt

from coarse_psd_matrix.utils import compute_psd_matrix, create_parser
from coarse_psd_matrix.plotting import plot_psd_matrix, set_mpl_rc_params

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    kwargs = dict(
        minimum_frequency=args.minimum_frequency,
        maximum_frequency=args.maximum_frequency,
        duration=args.duration,
        tick_step=125,
    )

    svd = compute_psd_matrix(
        interferometer_name=args.interferometer,
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

    fig = plt.figure(figsize=(10, 8))
    plot_psd_matrix(svd[0], plt.gca(), **kwargs, label="U")
    set_mpl_rc_params()
    plt.tight_layout()
    plt.savefig("figure_4.pdf")
    plt.close()
