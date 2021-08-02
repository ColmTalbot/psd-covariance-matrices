#cython: language_level=3

cimport numpy as np
import numpy as np
import cython


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef coarse_psd_matrix(np.ndarray fd_window, psd, int stride, int start=0, int stop=-1):
    """
    Compute the coarsened, windowed PSD. This is the leading diagonal of (16) in
    https://arxiv.org/abs/2106.13785.

    This is a double convolution between the window and the PSD evaluated a subset of
    the input frequencies.

    To speed up evaluation the upper right half matrix is generated first and then the
    lower left is taken as the Hermitian conjugate.

    The algorithm is:
    - for each input index `start <= k < stop` (this index is over greek indices):
        - choose row of window projection matrix as offset between indices
        - loop over output (roman) indices updating the output array
            - choose column of window projection matrix starting at row index
            - loop over output (roman) indices updating the output array

    Parameters
    ==========
    fd_window: array-like
        The two-sided frequency-domain window (\tilde{w}_{\mu}) at the fine
        frequency resolution.
    psd: array-like
        The two-sided fine resolution power spectral density (\mathcal{S}_{\mu}).
    stride: int
        The factor by which to down sample, e.g., to go from a 1 / 128 Hz
        resolution to a 1 / 4 Hz resolution stride = 128 / 4 = 32.
    start: int, optional
        The index of the input array to start at, default=0.
        This can be used to parallelize the computation.
    stop: int, optional
        The index of the input array to stop at, default=-1, i.e, the last entry.
        This can be used to parallelize the computation.

    Notes
    =====
    The two input arrays must be two-sided, i.e., contain positive and negative
    frequencies. However, the output is the one-sided PSD, i.e., it only contains
    positive frequency content.
    """
    cdef int ii, jj, kk, length, output_size, x_idx, y_idx
    cdef double temp_real, temp_psd
    cdef double wxr, wxi, wyr, wyi

    length = len(fd_window)
    if stop == -1:
        stop = length
    if stop > length:
        stop = length
    stop = min(stop, length)
    output_size = length // stride // 2 + 1
    
    temp_real = 0

    fd_window_real = np.real(fd_window)
    fd_window_imag = np.imag(fd_window)
    output_real = np.zeros((output_size, output_size))
    output_imag = np.zeros((output_size, output_size))

    cdef double[:, :] output_r = output_real
    cdef double[:, :] output_i = output_imag
    cdef double[:] psd_view = psd
    cdef double[:] window_real = fd_window_real
    cdef double[:] window_imag = fd_window_imag

    for kk in range(start, stop):
        temp_psd = psd_view[kk]
        if kk == 0:
            x_idx = 0
        else:
            x_idx = length - kk
        for ii in range(output_size):
            wxr = window_real[x_idx]
            wxi = window_imag[x_idx]
            wxr *= temp_psd
            wxi *= temp_psd
            y_idx = x_idx
            for jj in range(ii, output_size):
                wyr = window_real[y_idx]
                wyi = window_imag[y_idx]
                output_r[ii, jj] += wxr * wyr + wxi * wyi
                output_i[ii, jj] -= wxi * wyr - wxr * wyi
                y_idx += stride
                if y_idx >= length:
                    y_idx -= length
            x_idx += stride
            if x_idx >= length:
                x_idx -= length

    for ii in range(output_size):
        for jj in range(ii + 1, output_size):
            output_r[jj, ii] = output_r[ii, jj]
            output_i[jj, ii] = -output_i[ii, jj]

    return output_real + 1j * output_imag


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef coarse_psd(np.ndarray fd_window, psd, int stride, int start=0, int stop=-1):
    """
    Compute the coarsened, windowed PSD. This is the leading diagonal of (16) in
    https://arxiv.org/abs/2106.13785.

    This is essentially a convolution between the square magnitude of the window
    and the PSD evaluated a subset of input frequencies.

    The algorithm is:
    - for each input index `start <= k < stop` (this index is over greek indices):
        - choose row of window projection matrix as offset between indices
        - loop over output (roman) indices updating the output array

    Parameters
    ==========
    fd_window: array-like
        The two-sided frequency-domain window (\tilde{w}_{\mu}) at the fine
        frequency resolution.
    psd: array-like
        The two-sided fine resolution power spectral density (\mathcal{S}_{\mu}).
    stride: int
        The factor by which to down sample, e.g., to go from a 1 / 128 Hz
        resolution to a 1 / 4 Hz resolution stride = 128 / 4 = 32.
    start: int, optional
        The index of the input array to start at, default=0.
        This can be used to parallelize the computation.
    stop: int, optional
        The index of the input array to stop at, default=-1, i.e, the last entry.
        This can be used to parallelize the computation.

    Notes
    =====
    The two input arrays must be two-sided, i.e., contain positive and negative
    frequencies. However, the output is the one-sided PSD, i.e., it only contains
    positive frequency content.
    """
    cdef int ii, kk, length, output_size, x_idx
    cdef double temp_psd

    length = len(fd_window)
    if stop == -1:
        stop = length
    output_size = length // stride // 2 + 1
    
    temp_real = 0

    fd_window_power = np.abs(fd_window) ** 2
    output = np.zeros(output_size)

    cdef double[:] output_ = output
    cdef double[:] psd_view = psd
    cdef double[:] window_power = fd_window_power

    for kk in range(start, stop):
        temp_psd = psd_view[kk]
        if kk == 0:
            x_idx = 0
        else:
            x_idx = length - kk
        for ii in range(output_size):
            output_[ii] += temp_psd * window_power[x_idx]
            x_idx += stride
            if x_idx >= length:
                x_idx -= length

    return output
