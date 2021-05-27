import pyximport
pyximport.install(
    setup_args={"include_dirs":np.get_include()},
)
try:
    import cupy as cp
    CUPY = True
except ImportError:
    from coarsen_cpu import coarse_psd
    CUPY = False


if CUPY:
    def _coarse_psd_wrapper(func):

        def wrapped_kernel(fd_window, psd, stride, *args, **kwargs):
            array_1 = cp.atleast_1d(fd_window.copy())
            array_2 = cp.atleast_1d(psd.copy())
            length = array_1.size
            output_size = length // stride // 2 + 1
            output_real = cp.zeros((output_size, output_size))
            output_imag = cp.zeros((output_size, output_size))
            n_terms = output_size ** 2
            threads_per_block = min(
                func.max_threads_per_block, n_terms
            )
            grid_size = n_terms // threads_per_block
            if threads_per_block < n_terms:
                grid_size += 1
            func_args = (array_1, array_2, output_real, output_imag, stride, length, output_size)
            func((grid_size, ), (threads_per_block, ), func_args)
            output = output_real + 1j * output_imag
            if kwargs.get("output_type", "cupy") == "numpy":
                output = cp.asnumpy(output)
            return output

        return wrapped_kernel


    with open("coarsen_gpu.cu", "r") as ff:
        code = ff.read()
    kernel = cp.RawKernel(code, "coarse_psd_matrix")
    coarse_psd_matrix = _coarse_psd_wrapper(kernel)
else:
    from coarse_cpu import coarse_psd_matrix
