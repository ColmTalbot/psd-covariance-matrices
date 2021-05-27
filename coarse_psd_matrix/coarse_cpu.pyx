#cython: language_level=3

cimport numpy as np
import numpy as np
import cython


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef coarse_psd_matrix(np.ndarray fd_window, psd, int stride, int start=0, int stop=-1):
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
    cdef int ii, jj, kk, length, output_size, x_idx, y_idx
    cdef double temp_real, temp_psd
    cdef double wxr, wxi, wyr, wyi

    length = len(fd_window)
    if stop == -1:
        stop = length
    output_size = length // stride // 2 + 1
    
    temp_real = 0

    fd_window_real = np.real(fd_window)
    fd_window_imag = np.imag(fd_window)
    output = np.zeros(output_size)

    cdef double[:] output_ = output
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
            temp_real = wxr * wxr + wxi * wxi
            output_[ii] += temp_psd * temp_real
            x_idx += stride
            if x_idx >= length:
                x_idx -= length

    return output
