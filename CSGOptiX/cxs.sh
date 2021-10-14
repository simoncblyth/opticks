#!/bin/bash -l 
usage(){ cat << EOU

cxs.sh : CSGOptiX simulate 
================================================

The input is created by cxs0.sh 


EOU
}

${IPYTHON:-ipython} -i tests/CSGOptiXSimulate.py 

