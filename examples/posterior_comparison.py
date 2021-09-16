#!/usr/bin/env python
"""
Make the comparison corner plots to visualize impact of off-diagonal terms.

This will generate Figures 5-7.

For more details on the method see https://arxiv.org/abs/2106.13785.
"""

import numpy as np
import matplotlib.lines as mpllines
import matplotlib.pyplot as plt
import pandas as pd
from corner import corner

from coarse_psd_matrix.utils import create_parser, reweight_posterior, INTERFEROMETERS

from matplotlib import rcParams

rcParams["font.family"] = "serif"
rcParams["font.serif"] = "Computer Modern Roman"
rcParams["font.size"] = 20
rcParams["text.usetex"] = True
rcParams["grid.alpha"] = 0

LABELS = dict(
    chirp_mass="$\\mathcal{M} \\, [M_{\\odot}]$",
    mass_ratio="$q$",
    mass_1_source="$m_1 \\, [M_{\\odot}]$",
    mass_2_source="$m_2 \\, [M_{\\odot}]$",
    chi_1="$\\chi_1$",
    chi_2="$\\chi_2$",
    chi_1_in_plane="$\\chi_{1\\perp}$",
    chi_2_in_plane="$\\chi_{2\\perp}$",
    chi_eff="$\\chi_{\\mathrm{eff}}$",
    chi_p="$\\chi_p$",
    phi_12="$\\phi_{12}$",
    phi_jl="$\\phi_{JL}$",
    luminosity_distance="$d_L \\, [\\mathrm{Mpc}]$",
    theta_jn="$\\theta_{JN}$",
    L1_time="$t_{L} \\, [s]$",
    zenith="$\\kappa$",
    azimuth="$\\epsilon$",
    psi="$\\psi$",
    delta_phase="$\\delta\\phi$",
    log_likelihood="$\\ln\\mathcal{L}$",
)


def js_divergence_from_weights(weights):
    normalized_weights = weights / np.mean(weights)
    return (
        np.mean(
            normalized_weights
            * np.log2(2 * normalized_weights / (normalized_weights + 1))
        )
        + np.mean(np.log2(2 / (normalized_weights + 1)))
    ) / 2


def load_posterior(event, args):
    posterior = pd.read_hdf(f"{event}/posterior_{args.medium_duration}.hdf5")
    ln_likelihoods = np.zeros(len(posterior))
    ln_likelihoods_diag = np.zeros(len(posterior))
    for ifo in ["H1", "L1", "V1"]:
        key = f"ln_weights_{ifo}"
        if key in posterior:
            ln_likelihoods += posterior[f"ln_likeilhood_ratio_svd_{ifo}"]
            ln_likelihoods_diag += posterior[f"ln_likeilhood_ratio_diagonal_{ifo}"]
            posterior[key] = (
                posterior[f"ln_likeilhood_ratio_svd_{ifo}"]
                - posterior[f"ln_likeilhood_ratio_diagonal_{ifo}"]
            )
    ln_weights = ln_likelihoods - ln_likelihoods_diag
    weights = np.exp(ln_weights)
    js = js_divergence_from_weights(weights=weights)

    print(
        f"Resampling efficiency: {np.sum(weights) ** 2 / np.sum(weights ** 2) / len(weights)}\n"
        f"Rejection sampling leaves {np.sum(weights) / max(weights)} samples."
        f"Total JS divergence: {js:.3e}\n"
        f"Total reduced JS divergence: {js / 15:.3e}"
    )
    return posterior, weights


def plot_corner(posterior, weights, keys):
    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = "Computer Modern Roman"
    rcParams["font.size"] = 20
    rcParams["text.usetex"] = True
    rcParams["grid.alpha"] = 0
    default_kwargs = dict(
        bins=40,
        smooth=0.9,
        label_kwargs=dict(fontsize=16),
        title_kwargs=dict(fontsize=16),
        quantiles=None,
        levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.0)),
        plot_density=False,
        plot_datapoints=True,
        fill_contours=True,
        hist_kwargs=dict(density=True),
        max_n_ticks=3,
    )

    default_kwargs["hist_kwargs"]["color"] = "C0"
    fig = corner(
        posterior[keys],
        labels=[LABELS[key] for key in keys],
        color="C0",
        **default_kwargs,
    )
    default_kwargs["hist_kwargs"]["color"] = "C1"
    corner(
        posterior[keys],
        labels=[LABELS[key] for key in keys],
        color="C1",
        **default_kwargs,
        fig=fig,
        weights=weights,
    )

    legends = ["Finite\nDuration", "Diagonal"]

    lines = [mpllines.Line2D([0], [0], color=f"C{ii}") for ii in [0, 1]]
    axes = fig.get_axes()
    axes[len(keys) - 1].legend(lines, legends, fontsize=18)
    return fig


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    keys = ["mass_1_source", "mass_2_source"]

    maximum_frequency = dict(GW190521=300)

    for ii, event in enumerate(["GW150914", "GW170814", "GW190521"]):
        for ifo in INTERFEROMETERS[event]:
            reweight_posterior(
                interferometer_name=ifo,
                event=event,
                maximum_frequency=maximum_frequency.get(event, 800),
                medium_duration=128,
                outdir=event,
            )
        posterior, weights = load_posterior(event, args)
        plot_corner(posterior, weights, keys)
        plt.savefig(f"figure_{5 + ii}.pdf")
        plt.close()
