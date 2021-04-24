#!/bin/bash -l 

dir=$(dirname $BASH_SOURCE)
name=$(basename $BASH_SOURCE)
stem=${name/.sh}

export OPTICKS_GGEO_SUPPRESS=${OPTICKS_GGEO_SUPPRESS:-HBeam,ixture,anchor,Steel2,Plane,Wall,Receiver,Strut0x,sBar0x}

cmd="${OPTICKS_PYTHON:-python3} $dir/$stem.py $*"

echo $cmd
eval $cmd 

