#!/bin/bash -l

export FOLD=${FOLD:-/tmp/blyth/opticks/GeoChain/nmskTailOuter/G4CXSimtraceTest/ALL}

${IPYTHON:-ipython} --pdb -i $(dirname $BASH_SOURCE)/Values.py 
