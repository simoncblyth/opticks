#!/bin/bash -l 

usage(){ cat << EOU
QSimTest.sh
=============

::

    ./QSimTest.sh 
        run the executable and invoke the python script  

    TEST=fill_state_cf ./QSimTest.sh ana
        just invoke the analysis script for the named TEST 

EOU
}


arg=${1:-run_ana}

msg="=== $BASH_SOURCE :"

#export QBnd=INFO

#test=rng_sequence

#test=fill_state_0
#test=fill_state_1

#test=water
#test=rayleigh_scatter_align

#test=propagate_to_boundary
test=propagate_at_boundary
#test=propagate_at_surface

#test=hemisphere_s_polarized
#test=hemisphere_p_polarized
#test=hemisphere_x_polarized

#test=propagate_at_boundary_s_polarized
#test=propagate_at_boundary_p_polarized
#test=propagate_at_boundary_x_polarized



M1=1000000
K2=100000

num=8
#num=$K2
#num=$M1

#nrm=0,0,1
nrm=0,0,-1

export NUM=${NUM:-$num}
export NRM=${NRM:-$nrm}
export TEST=${TEST:-$test}

if [ "${arg/run}" != "$arg" ]; then 
   if [ -n "$DEBUG" ]; then 
       lldb__ QSimTest
   else
       QSimTest
   fi 
   [ $? -ne 0 ] && echo $msg run error && exit 1 
fi


if [ "${arg/ana}" != "$arg" ]; then 

    # PYVISTA_KILL_DISPLAY envvar is observed to speedup exiting from ipython after pyvista plotting 
    # see https://github.com/pyvista/pyvista/blob/main/pyvista/plotting/plotting.py
    export PYVISTA_KILL_DISPLAY=1

    case $TEST in
       fill_state_0)           script=QSimTest_fill_state.py ;;
       fill_state_1)           script=QSimTest_fill_state.py ;;
       fill_state_cf)          script=QSimTest_fill_state_cf.py ;;

       hemisphere_s_polarized) script=QSimTest_hemisphere_polarized.py ;;
       hemisphere_p_polarized) script=QSimTest_hemisphere_polarized.py ;;
       hemisphere_x_polarized) script=QSimTest_hemisphere_polarized.py ;;

       propagate_at_boundary_s_polarized) script=QSimTest_propagate_at_boundary_polarized.py ;; 
       propagate_at_boundary_p_polarized) script=QSimTest_propagate_at_boundary_polarized.py ;; 
       propagate_at_boundary_x_polarized) script=QSimTest_propagate_at_boundary_polarized.py ;; 

                                       *) script=QSimTest_$TEST.py      ;;
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

