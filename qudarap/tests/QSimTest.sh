#!/bin/bash -l 

usage(){ cat << EOU
QSimTest.sh
=============

::

    TEST=fill_state_cf ./QSimTest.sh ana


EOU
}


arg=${1:-run_ana}

msg="=== $BASH_SOURCE :"

#export QBnd=INFO

#test=fill_state_0
#test=fill_state_1
#test=water
#test=propagate_to_boundary
test=rayleigh_scatter_align

export TEST=${TEST:-$test}

if [ "${arg/run}" != "$arg" ]; then 
   QSimTest 
   [ $? -ne 0 ] && echo $msg run error && exit 1 
fi


if [ "${arg/ana}" != "$arg" ]; then 

    # PYVISTA_KILL_DISPLAY envvar is observed to speedup exiting from ipython after pyvista plotting 
    # see https://github.com/pyvista/pyvista/blob/main/pyvista/plotting/plotting.py
    export PYVISTA_KILL_DISPLAY=1

    case $TEST in
       fill_state_0)  script=QSimTest_fill_state.py ;;
       fill_state_1)  script=QSimTest_fill_state.py ;;
       fill_state_cf) script=QSimTest_fill_state_cf.py ;;
                  *)  script=QSimTest_$TEST.py      ;;
    esac

    if [ -f "$script" ]; then
        echo $msg invoking analysis script $script
        ${IPYTHON:-ipython} --pdb -i $script
        [ $? -ne 0 ] && echo $msg ana error && exit 2
    else
        echo $msg there is no analysis script $script
    fi  

fi


exit 0 

