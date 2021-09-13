import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.family"] = "serif"
rcParams["font.serif"] = "Computer Modern Roman"
rcParams["font.size"] = 20
rcParams["text.usetex"] = True
rcParams["grid.alpha"] = 0


def plot_psd_matrix(
    psd_matrix,
    axis,
    minimum_frequency,
    maximum_frequency,
    duration,
    vmin=None,
    vmax=None,
    tick_step=10,
    label="C",
    origin="lower",
):
    plt.sca(axis)
    plt.imshow(
        np.log10(np.abs(psd_matrix)),
        cmap="cividis",
        origin=origin,
        vmin=vmin,
        vmax=vmax,
    )
    cbar = plt.colorbar()
    cbar.set_label(f"$\\log_{{10}} |{label}_{{ij}}|$")
    for func in [plt.xticks, plt.yticks]:
        func(
            np.arange(
                0,
                (maximum_frequency - minimum_frequency) * duration,
                duration * tick_step,
            ),
            np.arange(minimum_frequency, maximum_frequency, tick_step, dtype=int),
        )
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Frequency [Hz]")
    return
