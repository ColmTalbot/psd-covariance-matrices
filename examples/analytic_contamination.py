#!/usr/bin/env python
"""
Plot the expected contamination for white Gaussian noise as a function of
Tukey parameter (alpha).

This will generate Figure 14.

For more details on the method see https://arxiv.org/abs/2106.13785.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal.windows import tukey

if __name__ == "__main__":
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = "Computer Modern Roman"
    mpl.rcParams["font.size"] = 20
    mpl.rcParams["text.usetex"] = True
    mpl.rcParams["grid.alpha"] = 0

    plt.figure(figsize=(8, 5))
    alphas = np.linspace(0, 1, 1000)
    values = np.empty(1000)
    for ii, tukey_alpha in enumerate(alphas):
        window = tukey(4 * 128, tukey_alpha)
        fd_window = np.fft.fft(window)
        limit = abs(np.mean(fd_window * np.roll(fd_window, 1).conjugate())) / np.sum(
            window ** 2
        )
        values[ii] = limit
    plt.plot(alphas, values)
    plt.xlim(0, 1)
    plt.ylim(0, 0.7)
    plt.xlabel("$\\alpha$")
    plt.ylabel("$\\Delta_{i}(\\alpha)$")
    plt.tight_layout()
    plt.savefig("figure_14.pdf")
    plt.close()
