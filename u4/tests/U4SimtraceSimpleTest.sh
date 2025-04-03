#!/bin/bash
usage(){ cat << EOU
U4SimtraceSimpleTest.sh
=========================

Split from U4SimtraceTest.sh as that has PMTSim_standalone complications

Uses U4SimtraceSimpleTest.cc which creates Geant4 geometry
with U4VolumeMaker::PV depending on GEOM envvar
and scans the geometry using U4SimtraceSimpleTest::scan
saving the intersects into the folder configured with FOLD envvar.


Commands
-----------

run/dbg
    simtrace intersects against geometry
ana
    presentation of simtrace intersects using python matplotlib OR pyvista
mpcap/pvcap
    screenshot current matplotlib/pyvista window with chrome cropped
mppub/pvpub
    publication by copying matplotlib/pyvista screenshot png into presentation tree



P[blyth@localhost tests]$ l $TMP/GEOM/RaindropRockAirWater/U4SimtraceSimpleTest/*_solid/B000/simtrace.npy
198752 -rw-rw-r--. 1 blyth blyth 203520128 Apr  3 17:26 /data/blyth/opticks/GEOM/RaindropRockAirWater/U4SimtraceSimpleTest/VACUUM_solid/B000/simtrace.npy
198752 -rw-rw-r--. 1 blyth blyth 203520128 Apr  3 17:26 /data/blyth/opticks/GEOM/RaindropRockAirWater/U4SimtraceSimpleTest/G4_Pb_solid/B000/simtrace.npy
198752 -rw-rw-r--. 1 blyth blyth 203520128 Apr  3 17:26 /data/blyth/opticks/GEOM/RaindropRockAirWater/U4SimtraceSimpleTest/G4_AIR_solid/B000/simtrace.npy
198752 -rw-rw-r--. 1 blyth blyth 203520128 Apr  3 17:26 /data/blyth/opticks/GEOM/RaindropRockAirWater/U4SimtraceSimpleTest/G4_WATER_solid/B000/simtrace.npy
P[blyth@localhost tests]$ 

P[blyth@localhost tests]$ cat /data/blyth/opticks/GEOM/RaindropRockAirWater/U4SimtraceSimpleTest/0/trs_names.txt 
VACUUM_solid
G4_Pb_solid
G4_AIR_solid
G4_WATER_solid


EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

DIR=$PWD
bin=U4SimtraceSimpleTest
geom=RaindropRockAirWater
evt=000
eye=0,1000,0   # +Y 1000mm

#cegs=16:0:9:1000   # default used from SFrameGenstep::MakeCenterExtentGensteps
cegs=16:0:9:5000    # increase photon count for more precise detail

#export GEOM=${GEOM:-$geom}
export GEOM=$geom

tmp=/tmp/$USER/opticks
export TMP=${TMP:-$tmp}

export BASE=$TMP/GEOM/$GEOM/$bin
export FOLD=$BASE   ## controls where the U4SimtraceSimpleTest.cc writes trs

export EVT=${EVT:-$evt}
export EYE=${EYE:-$eye}       # not extent scaled, just mm
export CEGS=${CEGS:-$cegs}
export CEHIGH_0=-1:1:0:0:7:8:1000:4


logging()
{
   type $FUNCNAME
   export U4VolumeMaker=INFO
}
[ -n "$LOG" ] && logging


log=$bin.log

defarg="run_ana"
arg=${1:-$defarg}


if [ "${arg/info}" != "$arg" ]; then
    vv="arg defarg GEOM FOLD"
    for v in $vv ; do printf "%20s : %s \n" "$v" "${!v}" ; done
fi 

if [ "${arg/run}" != "$arg" ]; then
    [ -f "$log" ] && rm $log
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1
    [ -f "$log" ] && echo $BASH_SOURCE rename log $log to logN $logN && mv $log $logN
fi

if [ "${arg/dbg}" != "$arg" ]; then
    [ -f "$log" ] && rm $log
    source $DIR/../../bin/dbg__.sh
    dbg__ $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 2
fi

if [ "${arg/ana}" != "$arg"  ]; then
    [ "$arg" == "nana" ] && export MODE=0
    ${IPYTHON:-ipython} --pdb -i $DIR/$bin.py
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 3
fi

if [ "$arg" == "pvcap" -o "$arg" == "pvpub" -o "$arg" == "mpcap" -o "$arg" == "mppub" ]; then
    export CAP_BASE=$FOLD/figs
    export CAP_REL=U4SimtraceSimpleTest
    export CAP_STEM=$GEOM
    case $arg in
       pvcap) source pvcap.sh cap  ;;
       mpcap) source mpcap.sh cap  ;;
       pvpub) source pvcap.sh env  ;;
       mppub) source mpcap.sh env  ;;
    esac
    if [ "$arg" == "pvpub" -o "$arg" == "mppub" ]; then
        source epub.sh
    fi
fi

exit 0
