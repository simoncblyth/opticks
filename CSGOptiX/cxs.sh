#!/bin/bash -l 
usage(){ cat << EOU

cxs.sh : CSGOptiX simulate 
================================================

EOU
}

${IPYTHON:-ipython} -i tests/CSGOptiXSimulate.py 

