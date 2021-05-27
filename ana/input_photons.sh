#!/bin/bash -l 

ipython -i --pdb $(which input_photons.py) -- $*

