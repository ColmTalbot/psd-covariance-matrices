To run the analysis in the paper simply run the following from this directory.

```console
$ bash run_all.sh
```

Note that this will take a very long time without a GPU.
The analysis was run on a single CPU core of a 64-core AMD Milan CPU and an NVIDIA A100-SXM4-80GB GPU in approximately two hours.
The parameter estimation analysis was run prior to this.

We also include a sciprt to generate the equivalent of Figure 1 with different Tukey alpha parameters.

```console
$ bash zoomed_supplement.sh
```

Other input paramters can be speficied from the command line.
For any script these can be queried with

```console
$ python $SCRIPT.py --help
```
