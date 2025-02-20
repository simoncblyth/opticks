#!/bin/bash
usage(){ cat << EOU
QSimTest_ONE.sh
=================

Thus script is intended to be symbolically linked with the
name of one TEST you want to make more convenient to run, eg::

    cd ~/o/qudarap/tests # qt 
    ln -s QSimTest_ONE.sh propagate_at_boundary_s_polarized.sh  
    chmod ugo+x ~/o/qudarap/tests/propagate_at_boundary_s_polarized.sh

This is useful for cycling on a failing test::

    ~/o/qudarap/tests/propagate_at_boundary_s_polarized.sh
    ~/o/qudarap/tests/propagate_at_boundary_s_polarized.sh dbg

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

name=$(basename $BASH_SOURCE)
name=${name/.sh}

vars="0 BASH_SOURCE PWD name"
for var in $vars ; do printf "%20s : %s\n" "$var" "${!var}" ; done 

TEST=$name ./QSimTest.sh $* 


