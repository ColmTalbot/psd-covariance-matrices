#!/usr/bin/env python
"""
Perform the population demonstration test in Section IIIB.

This will generate Figures 8 and 9.

For more details on the method see https://arxiv.org/abs/2106.13785.
For more details on the analysis see https://arxiv.org/abs/1712.00688.
"""

import dill
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bilby.core.utils import create_frequency_series
from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters
from bilby.gw.detector import PowerSpectralDensity
from bilby.gw.source import lal_binary_black_hole
from bilby.gw.waveform_generator import WaveformGenerator
from scipy.signal.windows import tukey
from scipy.special import logsumexp
from tqdm.auto import trange

from coarse_psd_matrix.utils import (
    compute_psd_matrix,
    fetch_psd_data,
    regularized_inversion,
)
from coarse_psd_matrix.plotting import set_mpl_rc_params


def run_tbs(signal_model, signal_parameter, outdir):
    sampling_frequency = 2048
    duration = 128
    data_duration = 32
    short_duration = 4
    low_frequency = 16
    minimum_frequency = 20
    maximum_frequency = 800
    tukey_alpha = 0.1
    label = f"{signal_model}_{signal_parameter}"

    data = fetch_psd_data(
        interferometer_name="L1",
        event="GW170814",
        outdir="GW170814",
        medium_duration=duration,
        sampling_frequency=sampling_frequency,
        tukey_alpha=tukey_alpha,
        duration=short_duration,
        low_frequency=low_frequency,
    )
    psd = data["psd"]
    freqs_ = data["frequencies"]
    del data

    psd_ = PowerSpectralDensity.from_power_spectral_density_array(
        frequency_array=freqs_, psd_array=psd
    )

    short_window = tukey(sampling_frequency * short_duration, tukey_alpha)
    window = np.zeros(sampling_frequency * data_duration)
    window[: len(short_window)] = short_window
    window /= np.mean(window ** 2) ** 0.5

    long_frequencies = np.fft.fftfreq(
        sampling_frequency * data_duration, 1 / sampling_frequency
    )
    long_psd = psd_.power_spectral_density_interpolated(abs(long_frequencies))

    print("PSD matrix")
    data = dict(frequencies=freqs_, medium_psd=long_psd)
    data_file = f"{outdir}/GW170814_data_coarse_{data_duration}_{sampling_frequency}_{tukey_alpha}_L1.pkl"
    with open(data_file, "wb") as ff:
        dill.dump(data, ff)
    _svd = compute_psd_matrix(
        interferometer_name="L1",
        event="GW170814",
        duration=short_duration,
        sampling_frequency=sampling_frequency,
        low_frequency=low_frequency,
        tukey_alpha=tukey_alpha,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
        medium_duration=data_duration,
        outdir="outdir",
    )
    finite_psd = abs((_svd[0] * _svd[1]) @ _svd[2]).diagonal()

    regularization_method = "psd"
    cutoff = min(finite_psd)
    regularized_inverse = regularized_inversion(_svd, (regularization_method, cutoff))

    SIGNAL_MODELS = dict(cbc=generate_cbc_signal, gaussian=generate_gaussian_signal)
    short_frequencies = np.fft.rfftfreq(
        short_duration * sampling_frequency, 1 / sampling_frequency
    )
    mask = (short_frequencies >= minimum_frequency) & (
        short_frequencies <= maximum_frequency
    )
    signal = SIGNAL_MODELS[signal_model](
        signal_parameter,
        sampling_frequency=sampling_frequency,
        mask=mask,
        duration=short_duration,
    )
    signal = normalize_signal(signal=signal, psd=finite_psd)
    np.save(f"{outdir}/signal_{label}", signal)

    data = run_tbs_test(
        signal=signal,
        psd=psd_,
        sampling_frequency=sampling_frequency,
        duration=duration,
        short_duration=short_duration,
        short_psd=finite_psd,
        mask=mask,
        inverse=regularized_inverse,
        n_average=500,
    )
    data.to_hdf(f"{outdir}/ln_bfs_{label}.hdf5", key="bayes_factors")

    fig = plot_tbs_bayes_factors(data)
    fig.savefig(f"{outdir}/ln_bfs_{label}.png")
    plt.close(fig)

    fig = plot_tbs_posterior(data)
    fig.savefig(f"{outdir}/xi_post_{label}.png", transparent=True)
    plt.close(fig)


def generate_cbc_signal(mass, sampling_frequency, duration, mask):
    waveform_arguments = dict(
        waveform_approximant="IMRPhenomXPHM",
        minimum_frequency=10.0,
        reference_frequency=20.0,
        maximum_frequency=1024.0,
    )

    waveform_generator = WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=lal_binary_black_hole,
        parameter_conversion=convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=waveform_arguments,
    )

    injection_parameters = dict(
        mass_1=mass,
        mass_2=mass,
        a_1=0.0,
        a_2=0.0,
        tilt_1=0.0,
        tilt_2=0.0,
        phi_12=0.0,
        phi_jl=0.0,
        luminosity_distance=2000.0,
        theta_jn=0.4,
        phase=0,
    )
    signal = waveform_generator.frequency_domain_strain(injection_parameters)["plus"]
    signal *= np.exp(1j * 2 * np.pi * waveform_generator.frequency_array * 2)
    signal = signal[mask]
    return signal


def generate_gaussian_signal(peak_frequency, sampling_frequency, duration, mask):
    frequencies = create_frequency_series(
        sampling_frequency=sampling_frequency, duration=duration
    )
    signal = np.exp(-((frequencies - peak_frequency) ** 2) / 2 / 5 ** 2) * np.exp(
        1j * np.random.uniform(0, 2 * np.pi, len(frequencies))
    )
    signal = signal[mask]
    return signal


def normalize_signal(signal, psd, snr=4):
    signal /= np.sum(abs(signal) ** 2 / psd) ** 0.5
    signal *= snr ** 0.5
    return signal


def run_tbs_test(
    signal,
    psd,
    sampling_frequency,
    duration,
    short_duration,
    mask,
    short_psd,
    inverse,
    n_average=50,
):
    short_window = tukey(sampling_frequency * short_duration, 0.1)
    short_window /= np.mean(short_window ** 2) ** 0.5

    ln_bfs_1 = np.array([])
    ln_bfs_2 = np.array([])

    for _ in trange(n_average):
        noise, _ = psd.get_noise_realisation(
            sampling_frequency=sampling_frequency, duration=duration
        )
        fd_noise = np.fft.rfft(
            np.fft.irfft(noise).reshape(-1, sampling_frequency * short_duration)
            * short_window
        )
        fd_noise = fd_noise[:, mask]
        fd_noise[np.random.choice(len(fd_noise), 3, replace=False)] += signal
        fd_residual = fd_noise - signal

        ln_ls_1 = -np.sum((abs(fd_residual) ** 2 / short_psd).T, axis=0).real / 2
        ln_ln_1 = -np.sum((abs(fd_noise) ** 2 / short_psd).T, axis=0).real / 2
        ln_ls_2 = (
            -(fd_residual @ inverse @ fd_residual.conjugate().T).diagonal().real / 2
        )
        ln_ln_2 = -(fd_noise @ inverse @ fd_noise.conjugate().T).diagonal().real / 2

        ln_bfs_1 = np.concatenate([ln_bfs_1, ln_ls_1 - ln_ln_1])
        ln_bfs_2 = np.concatenate([ln_bfs_2, ln_ls_2 - ln_ln_2])

    data = pd.DataFrame()
    data["infinite_ln_bf"] = ln_bfs_1
    data["finite_ln_bf"] = ln_bfs_2
    return data


def plot_tbs_bayes_factors(data):
    set_mpl_rc_params()

    ln_bfs_1 = data["infinite_ln_bf"]
    ln_bfs_2 = data["finite_ln_bf"]

    fig = plt.figure(figsize=(8, 5))
    plt.hist(
        ln_bfs_1,
        bins=30,
        histtype="step",
        density=True,
        label="Finite duration diagonal",
    )
    plt.hist(
        ln_bfs_2,
        bins=30,
        histtype="step",
        density=True,
        label="Finite duration non-diagonal",
    )
    plt.legend(loc="best")
    plt.xlabel("$\\ln$ BF")
    plt.tight_layout()
    return fig


def compute_tbs_posterior(ln_bfs, xis):
    ln_post = np.sum(
        np.log(
            np.outer(xis, np.exp(ln_bfs - max(ln_bfs)))
            + np.outer(1 - xis, np.ones(len(ln_bfs)) * np.exp(-max(ln_bfs)))
        )
        + max(ln_bfs),
        axis=-1,
    )
    ln_post -= logsumexp(ln_post)
    ln_post -= np.log(np.trapz(np.exp(ln_post), xis))
    return np.exp(ln_post)


def plot_tbs_posterior(data):
    set_mpl_rc_params()

    ln_bfs_1 = data["infinite_ln_bf"]
    ln_bfs_2 = data["finite_ln_bf"]

    fig = plt.figure(figsize=(8, 5))
    xis = np.linspace(0.0, 1, 300)
    for ln_bfs, label in zip([ln_bfs_1, ln_bfs_2], ["Diagonal", "Finite Duration"]):
        post = compute_tbs_posterior(ln_bfs, xis)
        plt.plot(xis, post, label=label)
    plt.axvline(3 / 32, color="k", linestyle="--")
    plt.xlim(0, 1)
    plt.ylim(0, max(post) * 1.2)
    plt.xlabel("$\\xi$")
    plt.ylabel("$p(\\xi)$")
    plt.legend(loc="upper right")
    plt.tight_layout()
    return fig


def make_tbs_signal_plot(outdir, signals):
    set_mpl_rc_params()
    axis_labels = ["a", "b", "c", "d", "e", "f", "g", "h"]

    labels = list()
    for signal in signals:
        for signal_parameter in signals[signal]:
            labels.append(f"{signal}_{signal_parameter}")

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 10))

    frequencies = np.load(f"{outdir}/frequencies.npy")
    finite_psd = np.load(f"{outdir}/finite_psd.npy")

    bounds = [(20, 100), (100, 800)]
    for ii, bound in enumerate(bounds):
        plt.sca(axes[ii])
        plt.text(
            0.02,
            0.95,
            f"({axis_labels[ii]})",
            horizontalalignment="left",
            verticalalignment="top",
            transform=axes[ii].transAxes,
        )
        for label in labels:
            plt.loglog(frequencies, abs(np.load(f"{outdir}/signal_{label}.npy")))
        plt.loglog(frequencies, finite_psd ** 0.5)
        plt.xlim(*bound)
        plt.ylim(1e-26, 1e-20)
        plt.xscale("linear")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Amplitude Spectral Density [Hz$^{-1/2}$]")

    plt.tight_layout()
    plt.savefig("figure_7.pdf")
    plt.close()


def make_final_tbs_plot(outdir, signals):
    set_mpl_rc_params()
    axis_labels = ["a", "b", "c", "d", "e", "f", "g", "h"]

    labels = list()
    for signal in signals:
        for signal_parameter in signals[signal]:
            labels.append(f"{signal}_{signal_parameter}")

    fig, axes = plt.subplots(
        nrows=len(labels), ncols=1, figsize=(8, 3 * len(labels)), sharex=True
    )
    for ii, signal in enumerate(labels):
        plt.sca(axes[ii])
        plt.text(
            0.02,
            0.93,
            f"({axis_labels[ii]})",
            horizontalalignment="left",
            verticalalignment="top",
            transform=axes[ii].transAxes,
        )
        data = pd.read_hdf(f"{outdir}/ln_bfs_{signal}.hdf5")
        idxs = np.arange(len(data))
        ln_bfs_1 = data["infinite_ln_bf"][idxs]
        ln_bfs_2 = data["finite_ln_bf"][idxs]

        xis = np.linspace(0.08, 0.12, 300)
        for ln_bfs, label in zip([ln_bfs_1, ln_bfs_2], ["Diagonal", "Finite Duration"]):
            post = compute_tbs_posterior(ln_bfs, xis)
            plt.plot(xis, post, label=label)
        plt.axvline(3 / 32, color="k", linestyle="--", label="True Value")
        plt.ylim(0, 450)
        if ii == 0:
            plt.legend(loc="upper right")
        if ii == 1:
            plt.xlabel("$\\xi$")
        plt.ylabel("$p(\\xi)$")
    plt.xlim(0.08, 0.12)
    plt.tight_layout()
    plt.savefig("figure_8.pdf", transparent=True)
    plt.close()


if __name__ == "__main__":
    signals = dict(cbc=[30, 150], gaussian=[50, 500])
    outdir = "outdir"
    for signal_model in signals:
        for signal_parameter in signals[signal_model]:
            run_tbs(signal_model, signal_parameter, outdir=outdir)
    make_final_tbs_plot(outdir=outdir, signals=signals)
