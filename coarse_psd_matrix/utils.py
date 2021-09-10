import os
import time

import numpy as np

from . import coarse_psd, coarse_psd_matrix

INTERFEROMETERS = dict(
    GW190521=["H1", "L1", "V1"],
    GW170814=["H1", "L1", "V1"],
    GW150914=["H1", "L1"],
)


def create_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Run analyses for arXiv:2106.13785")
    parser.add_argument("--outdir", help="The output directory")
    parser.add_argument("--event", default="GW170814", help="GW event name")
    parser.add_argument(
        "--interferometer", default="L1", help="Interferometer to analyze"
    )
    parser.add_argument("--duration", type=int, help="The segment duration", default=4)
    parser.add_argument(
        "--medium-duration",
        type=int,
        help="The long segment duration to use for the 'infinite-duration' PSD",
        default=128,
    )
    parser.add_argument(
        "--sampling-frequency",
        type=float,
        help="The sampling frequency to use",
        default=2048,
    )
    parser.add_argument(
        "--low-frequency", type=float, help="The high pass filter frequency", default=16
    )
    parser.add_argument(
        "--tukey-alpha", type=float, help="Tukey window shape parameter", default=0.1
    )
    parser.add_argument(
        "--minimum-frequency",
        type=float,
        help="The minimum analyzed frequency",
        default=20,
    )
    parser.add_argument(
        "--maximum-frequency",
        type=float,
        help="The maximum analyzed frequency",
        default=800,
    )
    return parser


def time_average_psd(data, nfft, window, average="median", sampling_frequency=1):
    """
    Estimate a power spectral density (PSD) by averaging over non-overlapping
    shorter segments.

    This is different from many other implementations as it does not account
    for the window power loss factor (<window ** 2>)

    Parameters
    ----------
    data: np.ndarray
        The input data to use to estimate the PSD
    nfft: int
        The number of input elements per segment
    window: [str, tuple]
        Input arguments for scipy.signal.windows.get_window to specify the
        window.
    average: str
        Time averaging method, should be either "mean" or "median"
    sampling_frequency: float
        The sampling frequency of the input data, used to normalize the PSD
        estimate to have dimensions of 1 / Hz.

    Returns
    -------
    psd: np.ndarray
        The estimate PSD
    """
    from scipy.signal.spectral import _median_bias
    from scipy.signal.windows import get_window

    if not isinstance(window, np.ndarray):
        window = get_window(window, nfft)
    blocked_data = data.reshape(-1, nfft) * window
    blocked_psd = abs(np.fft.rfft(blocked_data, axis=-1) / sampling_frequency) ** 2
    if average == "median":
        normalization = 1 / _median_bias(len(blocked_data))
        func = np.median
    elif average == "mean":
        normalization = 1
        func = np.mean
    else:
        raise ValueError(f"PSD method should be mean or median, not {average}")
    psd = func(blocked_psd, axis=0) / 2 * normalization
    return psd


def band_pass_and_downsample(data, sampling_frequency, low_frequency):
    """
    Band pass and down sample the input data with high pass frequency
    `low_frequency` and new sampling rate `sampling_frequency`.

    Parameters
    ----------
    data: gwpy.timeseries.TimeSeries
        Input time series data
    sampling_frequency: float
        The sampling frequency after down sampling.
    low_frequency: float
        The high pass frequency

    Returns
    -------
    resampled: gwpy.timeseries.TimeSeries
        Result of band passing and down sampling.
    """
    high_frequency = sampling_frequency // 2
    if low_frequency > 0:
        high_passed = data.highpass(low_frequency)
    else:
        high_passed = data.copy()
    if high_frequency < sampling_frequency / 2:
        low_passed = high_passed.lowpass(high_frequency)
    else:
        low_passed = high_passed.copy()
    if sampling_frequency < low_passed.sample_rate.value:
        resampled = low_passed.resample(sampling_frequency)
    else:
        resampled = low_passed.copy()
    return resampled


def regularized_eigenvalues(input, method=("window", 0.1), fill_value=np.inf):
    output = input.copy()
    if method[0].lower() == "window":
        output[int(len(input) * (1 - 5 * method[1] / 8)) :] = fill_value
    elif method[0].lower() == "psd":
        output[input < method[1]] = fill_value
    else:
        raise ValueError("Regularization method should be either 'window' or 'psd'")
    return output


def regularized_inversion(svd, method=("window", 0.1)):
    regularized = regularized_eigenvalues(svd[1], method)
    regularized_inverse = svd[2].T @ np.nan_to_num(svd[0] / regularized).T
    return regularized_inverse


def fetch_psd_data(
    interferometer_name,
    event="GW170814",
    duration=4,
    sampling_frequency=2048,
    low_frequency=16,
    tukey_alpha=0.1,
    medium_duration=32,
    outdir="outdir",
):
    import dill
    from gwosc.datasets import event_gps
    from scipy.signal.windows import tukey

    trigger_time = event_gps(event)
    start_time = trigger_time + 2 - duration
    psd_start_time = start_time - 512
    window = tukey(sampling_frequency * duration, tukey_alpha)
    stride = medium_duration // duration
    normalization = medium_duration / np.mean(window ** 2) * 3 / 32

    data_file = f"{outdir}/{event}_data_coarse_{medium_duration}_{sampling_frequency}_{tukey_alpha}_{interferometer_name}.pkl"

    if os.path.isfile(data_file):
        print(f"Loading {data_file}")
        with open(data_file, "rb") as ff:
            data = dill.load(ff)
    else:
        from gwpy.timeseries import TimeSeries

        try:
            print("Fetching PSD data...")
            psd_data = TimeSeries.fetch_open_data(
                interferometer_name, psd_start_time, psd_start_time + 512
            )
        except ValueError:
            print(f"No data found for {event} for {interferometer_name}.")
            return
        print("Band passing and resampling PSD data...")
        resampled = band_pass_and_downsample(
            psd_data, sampling_frequency, low_frequency
        )
        print("Computing coarse PSD...")
        _window = tukey(sampling_frequency * medium_duration, 1)
        medium_psd = time_average_psd(
            resampled.value,
            len(_window),
            _window,
            sampling_frequency=sampling_frequency,
        )
        medium_psd = np.concatenate([medium_psd, medium_psd[1:-1][::-1]])
        full_window = np.zeros(len(medium_psd))
        full_window[: len(window)] = window
        full_window /= np.mean(full_window ** 2) ** 0.5
        _fd_window = np.fft.fft(full_window) / len(full_window)
        psd = coarse_psd(_fd_window, medium_psd, stride) / normalization

        print("Fetching analysis data...")
        analysis_data = TimeSeries.fetch_open_data(
            interferometer_name, start_time, start_time + duration
        )
        print("Band passing and resampling analysis data...")
        resampled = band_pass_and_downsample(
            analysis_data, sampling_frequency, low_frequency
        )
        fd_data = np.fft.rfft(resampled.value * window) / sampling_frequency

        freqs = np.fft.rfftfreq(sampling_frequency * duration, 1 / sampling_frequency)

        data = dict(
            strain=fd_data,
            psd=psd,
            medium_psd=medium_psd,
            frequencies=freqs,
            td_data=resampled,
        )
        with open(data_file, "wb") as ff:
            print(f"Saving PSD data to {data_file}")
            dill.dump(data, ff)
    return data


def compute_psd_matrix(
    interferometer_name,
    event="GW170814",
    duration=4,
    sampling_frequency=2048,
    low_frequency=16,
    tukey_alpha=0.1,
    minimum_frequency=20,
    maximum_frequency=800,
    medium_duration=32,
    outdir="outdir",
):
    import dill
    from scipy.signal.windows import tukey

    data = fetch_psd_data(
        interferometer_name=interferometer_name,
        event=event,
        duration=duration,
        sampling_frequency=sampling_frequency,
        low_frequency=low_frequency,
        tukey_alpha=tukey_alpha,
        medium_duration=medium_duration,
        outdir=outdir,
    )
    freqs = data["frequencies"]
    medium_psd = data["medium_psd"]

    window = tukey(sampling_frequency * duration, tukey_alpha)
    stride = medium_duration // duration
    normalization = medium_duration / np.mean(window ** 2) * 3 / 8 / 4

    frequency_mask = (freqs >= minimum_frequency) & (freqs <= maximum_frequency)

    svd_file = f"{outdir}/{event}_svd_coarse_{medium_duration}_{sampling_frequency}_{tukey_alpha}_{interferometer_name}_{int(minimum_frequency)}_{int(maximum_frequency)}.pkl"
    if os.path.isfile(svd_file):
        print(f"Loading SVD file {svd_file}...")
        with open(svd_file, "rb") as ff:
            svd = dill.load(ff)
    else:
        from gwpopulation.cupy_utils import xp, to_numpy
        from tqdm.auto import trange

        print("Computing PSD matrix...")
        full_window = np.zeros(len(medium_psd))
        full_window[: len(window)] = window
        full_window /= np.mean(full_window ** 2) ** 0.5
        _fd_window = np.fft.fft(full_window) / len(full_window)
        start = time.time()
        if xp == np:
            long_frequencies = np.fft.rfftfreq(
                len(full_window), d=1 / sampling_frequency
            )
            first_output = np.where(long_frequencies <= minimum_frequency)[0][-1]
            last_output = np.where(long_frequencies >= maximum_frequency)[0][0]
            kwargs = dict(
                fd_window=_fd_window,
                psd=medium_psd,
                stride=stride,
                first_output=first_output,
                last_output=last_output,
            )
            analytic_psd_matrix = 0 * coarse_psd_matrix(start=0, stop=10, **kwargs)
            for ii in trange(0, len(medium_psd), 16):
                stop = min(ii + 16, len(medium_psd) - 1)
                analytic_psd_matrix += coarse_psd_matrix(start=ii, stop=stop, **kwargs)
        else:
            analytic_psd_matrix = (
                coarse_psd_matrix(_fd_window, medium_psd, stride) / normalization
            )
            analytic_psd_matrix = analytic_psd_matrix[frequency_mask][:, frequency_mask]
            xp.cuda.Stream.null.synchronize()
        end = time.time()
        print(f"PSD matrix time: {end - start:.2f}s")

        print("Computing SVD...")
        start = time.time()
        svd = xp.linalg.svd(xp.asarray(analytic_psd_matrix))
        end = time.time()
        svd = (to_numpy(svd[0]), to_numpy(svd[1]), to_numpy(svd[2]))
        print(f"SVD time: {end - start:.2f}s")
        with open(svd_file, "wb") as ff:
            print(f"Saving SVD data to {svd_file}")
            dill.dump(svd, ff)

    return svd


def whiten_diagonal(array, psd):
    return array / psd ** 0.5


def whiten_svd(array, svd):
    return svd[0].T @ (array) / svd[1] ** 0.5


def likelihood(whitened_data, signal, whitener, whiten_method):
    whitened_signal = whiten_method(signal, whitener)
    return -np.sum(abs(whitened_data - whitened_signal) ** 2 / 2)


def reweight_posterior(
    interferometer_name,
    event="GW170814",
    duration=4,
    sampling_frequency=2048,
    low_frequency=16,
    tukey_alpha=0.1,
    minimum_frequency=20,
    maximum_frequency=800,
    reference_frequency=20,
    medium_duration=32,
    target="posterior",
    outdir="outdir",
):
    from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters
    from bilby.gw.detector import get_empty_interferometer, PowerSpectralDensity
    from bilby.gw.source import lal_binary_black_hole
    from bilby.gw.waveform_generator import WaveformGenerator
    from gwosc.datasets import event_gps
    from scipy.signal.windows import tukey
    from tqdm.auto import trange

    data = fetch_psd_data(
        interferometer_name=interferometer_name,
        event=event,
        duration=duration,
        sampling_frequency=sampling_frequency,
        low_frequency=low_frequency,
        tukey_alpha=tukey_alpha,
        medium_duration=medium_duration,
        outdir=outdir,
    )
    svd = compute_psd_matrix(
        interferometer_name=interferometer_name,
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
    trigger_time = event_gps(event)
    start_time = trigger_time + 2 - duration
    freqs = data["frequencies"]
    fd_data = data["strain"]
    psd = data["psd"]
    window = tukey(sampling_frequency * duration, tukey_alpha)

    frequency_mask = (freqs >= minimum_frequency) & (freqs <= maximum_frequency)
    target = f"{target}_{medium_duration}"
    cutoff = int(sum(frequency_mask) * np.mean(window ** 2))

    filename = f"{event}_{target}.hdf5"
    if os.path.isfile(filename):
        import pandas as pd

        posterior = pd.read_hdf(filename)
    else:
        from bilby.core.result import read_in_result

        result = read_in_result(f"./{event}/diagonal_wider_time_result.json")
        if "posterior" in target:
            posterior = result.posterior
            n_samples = min(len(posterior), 10000)
            posterior = posterior.sample(n_samples, replace=False)
        elif "nested" in target:
            posterior = result.nested_samples
        else:
            raise ValueError("Target should contain either 'posterior' or 'nested'")
    _posterior = posterior[
        [
            "chirp_mass",
            "mass_ratio",
            "a_1",
            "a_2",
            "tilt_1",
            "tilt_2",
            "phi_12",
            "phi_jl",
            "luminosity_distance",
            "theta_jn",
            "geocent_time",
            "ra",
            "dec",
            "phase",
            "psi",
        ]
    ]

    waveform_generator = WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=lal_binary_black_hole,
        parameter_conversion=convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=dict(
            reference_frequency=reference_frequency,
            maximum_frequency=maximum_frequency,
            minimum_frequency=minimum_frequency / 2,
            waveform_approximant="IMRPhenomXPHM",
        ),
    )
    ifo = get_empty_interferometer(interferometer_name)
    ifo.power_spectral_density = PowerSpectralDensity.from_power_spectral_density_array(
        frequency_array=freqs, psd_array=psd
    )
    ifo.set_strain_data_from_frequency_domain_strain(
        frequency_domain_strain=fd_data, frequency_array=freqs, start_time=start_time
    )
    ifo.minimum_frequency = minimum_frequency
    ifo.maximum_frequency = maximum_frequency

    ln_llr_svd = np.zeros(len(_posterior))
    ln_llr_diagonal = np.zeros(len(_posterior))

    fd_data = fd_data[frequency_mask]
    psd = psd[frequency_mask]

    svd = list(svd)
    diagonal_whitened = whiten_diagonal(fd_data, psd)
    svd_whitened = whiten_svd(fd_data, svd)
    noise_ln_l_diagonal = likelihood(
        diagonal_whitened, np.zeros_like(fd_data), psd, whiten_diagonal
    )
    noise_ln_l_non_diagonal_cut = likelihood(
        svd_whitened, np.zeros_like(fd_data), svd, whiten_svd
    )
    svd[1] = svd[1][:cutoff]
    svd[0] = svd[0][:, :cutoff]
    svd_whitened = whiten_svd(fd_data, svd)
    noise_ln_l_non_diagonal = likelihood(
        svd_whitened, np.zeros_like(fd_data), svd, whiten_svd
    )
    print(noise_ln_l_diagonal, noise_ln_l_non_diagonal, noise_ln_l_non_diagonal_cut)

    for ii in trange(len(_posterior)):
        parameters = dict(_posterior.iloc[ii])
        signal_polarizations = waveform_generator.frequency_domain_strain(parameters)
        signal = ifo.get_detector_response(signal_polarizations, parameters)
        signal = signal[frequency_mask]

        ln_llr_svd[ii] = float(
            likelihood(svd_whitened, signal, svd, whiten_svd) - noise_ln_l_non_diagonal
        )
        ln_llr_diagonal[ii] = float(
            likelihood(diagonal_whitened, signal, psd, whiten_diagonal)
            - noise_ln_l_diagonal
        )
    ln_weights = ln_llr_svd - ln_llr_diagonal
    posterior[f"ln_weights_{interferometer_name}"] = ln_weights
    posterior[f"ln_likeilhood_ratio_svd_{interferometer_name}"] = ln_llr_svd
    posterior[f"ln_likeilhood_ratio_diagonal_{interferometer_name}"] = ln_llr_diagonal
    posterior.to_hdf(filename, mode="w", key="posterior")
