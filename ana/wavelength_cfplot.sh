#!/bin/bash -l 

dir=$(dirname $BASH_SOURCE)

ARG=${ARG:-15} ipython -i $dir/wavelength_cfplot.py 


