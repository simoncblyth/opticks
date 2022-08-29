#!/bin/bash -l 
usage(){ cat << EOU
nmskSolidMask.sh
==================

SELECTION envvar 
    provides simtrace indices to rerun on CPU using CSGQuery which runs
    the CUDA compatible intersect code on the CPU. 
    Without this envvar all the simtrace items ate rerun.

CSGSimtraceRerunTest requires a CSGFoundry geometry and corresponding 
simtrace intersect array. Create these with:: 

    geom_ ## set geom to "nmskSolidMask" :  a string understood by PMTSim::getSolid

    gc     ## cd ~/opticks/GeoChain
    ./translate.sh    ## translate PMTSim::getSolid Geant4 solid into CSGFoundry 

    gx     ## cd ~/opticks/g4cx 

    ./gxt.sh        # simtrace GPU run on workstation
    ./gxt.sh grab   # rsync back to laptop
    ./gxt.sh ana    # python plotting  

To temporatily look at a GEOM::    

    GEOM=nmskSolidMask ./gxt.sh ana 

EOU
}

echo $BASH_SOURCE 

export CSGFoundry=INFO

#export CSGRecord_ENABLED=1

#selection=176995,153452,459970
selection=176995

GEOM=nmskSolidMask SELECTION=$selection ./CSGSimtraceRerunTest.sh 

