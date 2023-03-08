#!/bin/bash -l 

usage(){ cat << EOU

Formerly tries to kludge together despite different geometry on U4SimulateTest.sh::

    if [ "$LAYOUT" == "one_pmt" -a "$running_mode" == "SRM_G4STATE_RERUN" -a "$VERSION" == "1" ]; then

       ## when using natural geometry need to apply some burns to
       ## jump ahead in a way that corresponds to the consumption 
       ## for navigating the fake volumes in the old complex geomerty 

       ./UU_BURN.sh 
       export SEvt__UU_BURN=/tmp/UU_BURN.npy
    fi 

EOU

}

${IPYTHON:-ipython} --pdb UU_BURN.py 



