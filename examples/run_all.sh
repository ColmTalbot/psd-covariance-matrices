#!/bin/bash

python zoomed.py --outdir zoomed
python contamination.py --outdir GW170814
python eigenvalues.py --outdir GW170814
python svd_matrix.py --outdir GW170814
python posterior_comparison.py
python population.py
python medium_duration_comparison.py --outdir medium
python matrix_figures.py --outdir GW170814 --event GW170814
python analytic_contamination.py
