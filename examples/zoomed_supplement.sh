#!/bin/bash

mkdir variable_alpha_zoom
for alpha in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
    python zoomed.py --outdir variable_alpha_zoom --tukey-alpha $alpha
done

