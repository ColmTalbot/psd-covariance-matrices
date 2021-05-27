extern "C" __global__
void coarse_psd_matrix(const double* window, const double* psd, double* output_real, double* output_imag, int stride, int length, int output_size){
    unsigned int ii, jj, kk, x_idx, y_idx;
    unsigned int x_idx_2, y_idx_2, len;
    double wxr, wxi, wyr, wyi;
    double total_real, total_imag;
    int tid, tid_2;
    unsigned long out_size;
    
    tid = blockDim.x * blockIdx.x + threadIdx.x;
    len = length;
    out_size = output_size;

    ii = tid % out_size;
    jj = tid / out_size;
    if (ii < jj) {
        return;
    } else {
        tid_2 = ii * out_size + jj;
    }
    
    x_idx = ii * stride;
    y_idx = jj * stride;
    total_real = 0;
    total_imag = 0;

    for (kk = 0; kk < len; kk++){
         if (x_idx >= kk)
            x_idx_2 = x_idx - kk;
        else
            x_idx_2 = x_idx + len - kk;
        if (y_idx >= kk)
            y_idx_2 = y_idx - kk;
        else
            y_idx_2 = y_idx + len - kk;
        wxr = window[2 * x_idx_2];
        wxi = window[2 * x_idx_2 + 1];
        wyr = window[2 * y_idx_2];
        wyi = window[2 * y_idx_2 + 1];
        total_real += psd[kk] * (wxr * wyr + wxi * wyi);
        total_imag += psd[kk] * (wxi * wyr - wxr * wyi);
    }
    output_real[tid] = total_real;
    output_imag[tid] = total_imag;
    if (ii > jj) {
        output_real[tid_2] = total_real;
        output_imag[tid_2] = -total_imag;
    }
}
