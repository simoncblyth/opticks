#!/bin/bash -l 
usage(){ cat << EOU

cxs.sh : CSGOptiXSimulateTest pyvista presentation of frame photons
===========================================================================

The input is created by cxs0.sh 

EOU
}

${IPYTHON:-ipython} -i tests/CSGOptiXSimulateTest.py 

