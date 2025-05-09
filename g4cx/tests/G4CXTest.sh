#!/bin/bash
usage(){ cat << EOU
G4CXTest.sh : Standalone bi-simulation with G4CXApp::Main
===========================================================

::

    ~/opticks/g4cx/tests/G4CXTest.sh run

    PIDX=552 ~/opticks/g4cx/tests/G4CXTest.sh run ## run with single photon debug

    ~/opticks/g4cx/tests/G4CXTest.sh grab


    PICK=B MODE=2 ~/opticks/g4cx/tests/G4CXTest.sh ana
         2D (matplotlib) plot Geant4 photon histories



    MODE=2 ./G4CXTest.sh ana

    MODE=2 APID=62 ./G4CXTest.sh tra



Screen captures
-----------------

::

    ~/opticks/g4cx/tests/G4CXTest.sh grab  # from remote
    PICK=A MODE=2 APID=1000 FOCUS=0,0,80                  ~/opticks/g4cx/tests/G4CXTest.sh ana
    PICK=A MODE=2 APID=1000 FOCUS=0,0,80                  ~/opticks/g4cx/tests/G4CXTest.sh mpcap
    PICK=A MODE=2 APID=1000 FOCUS=0,0,80 PUB=Tub3_delta_1 ~/opticks/g4cx/tests/G4CXTest.sh mppub

Input scripts
---------------


~/opticks/u4/tests/FewPMT.sh
    configure geometry

~/opticks/u4/tests/storch_FillGenstep.sh
    configure torch photons, controlled via LAYOUT and CHECK envvars





EOU
}

SDIR=$(dirname $(realpath $BASH_SOURCE))
U4TDIR=$(realpath $SDIR/../../u4/tests)
BINDIR=$(realpath $SDIR/../../bin)

bin=G4CXTest
ana=$SDIR/G4CXTest.py
tra=$SDIR/G4CXSimtraceMinTest.py


#defarg="info_dbg_ana"
defarg="info_run_ana"
[ -n "$BP" ] && defarg="info_dbg_ana"
arg=${1:-$defarg}

source $HOME/.opticks/GEOM/GEOM.sh

geomscript=$U4TDIR/$GEOM.sh
if [ -f "$geomscript" ]; then
    source $geomscript
else
    echo $BASH_SOURCE : no geomscript $geomscript
fi


export VERSION=0

tmp=/tmp/$USER/opticks
export TMP=${TMP:-$tmp}

export OPTICKS_EVENT_NAME=DebugPhiloxShakedownIPH
export BASE=$TMP/GEOM/$GEOM/$bin
export EVT=000
export AFOLD=$BASE/ALL${VERSION}_${OPTICKS_EVENT_NAME}/A${EVT}
export BFOLD=$BASE/ALL${VERSION}_${OPTICKS_EVENT_NAME}/B${EVT}
export TFOLD=$BASE/0/p999

if [ -z "$APID" -a -z "$BPID" -a -n "$PIDX" ]; then
    echo $BASH_SOURCE : PIDX $PIDX is defined and APID BPID are both not defined so setting them to PIDX
    export APID=$PIDX
    export BPID=$PIDX
fi



#num_photons=1
#num_photons=10
#num_photons=100
#num_photons=1000      # 1k
#num_photons=10000    # 10k
#num_photons=50000    # 50k
num_photons=100000    # 100k
#num_photons=1000000  # 1M

NUM_PHOTONS=${NUM_PHOTONS:-$num_photons}

export G4CXOpticks__setGeometry_saveGeometry=$HOME/.opticks/GEOM/$GEOM

export OPTICKS_RUNNING_MODE=SRM_TORCH
export OPTICKS_MAX_BOUNCE=31
export OPTICKS_EVENT_MODE=DebugLite
export OPTICKS_INTEGRATION_MODE=3

export OPTICKS_MAX_PHOTON=${NUM_PHOTONS}
#export OPTICKS_MAX_PHOTON=100000

export SEvent_MakeGenstep_num_ph=${NUM_PHOTONS}

#check=rain_point_xpositive_100
#check=rain_line
#check=tub3_side_line
#check=circle_inwards_100
#check=circle_outwards_1
#check=rain_line_205
#check=rain_down_100
check=rectangle_inwards

export LAYOUT=one_pmt
export CHECK=${CHECK:-$check}
source $U4TDIR/storch_FillGenstep.sh
echo $BASH_SOURCE : CHECK $CHECK
env | grep storch

if [ "$storch_FillGenstep_type" == "" ]; then
    echo $BASH_SOURCE : FATAL : for CHECK $CHECK LAYOUT $LAYOUT GEOM $GEOM
    exit 1
fi


export U4SensorIdentifierDefault__GLOBAL_SENSOR_BOUNDARY_LIST=$(cat << EOL

    Pyrex/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/Vacuum
    Pyrex/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/Vacuum
    Pyrex/PMT_3inch_photocathode_logsurf2/PMT_3inch_photocathode_logsurf1/Vacuum
    Pyrex/PMT_20inch_veto_photocathode_logsurf2/PMT_20inch_veto_photocathode_logsurf1/Vacuum

    Pyrex/nnvt_photocathode_mirror_logsurf/nnvt_photocathode_mirror_logsurf/Vacuum

EOL
)




logging(){
   export Dummy=INFO
   #export G4CXOpticks=INFO
   #export X4PhysicalVolume=INFO   # look into sensor boundary to understand lpmtid -1
   #export QSim=INFO
   #export QEvent=INFO
   #export SSim__stree_level=2    # U4Tree/stree level   debugging U4Tree::identifySensitiveGlobals
   #export SEvt=INFO

   export U4Recorder__PIDX_ENABLED=1
   export C4OpBoundaryProcess__PIDX_ENABLED=1

}
logging

# dont need to redo the simtrace until geometry changed
#export U4Recorder__EndOfRunAction_Simtrace=1
#export CEGS=16:0:9:100


vars="BASH_SOURCE SDIR U4TDIR BINDIR GEOM bin ana tra geomscript BASE FOLD AFOLD BFOLD TFOLD PMTSimParamData_BASE"

if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done
fi

if [ "${arg/run}" != "$arg" ]; then
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 1
fi

if [ "${arg/dbg}" != "$arg" ]; then
    source dbg__.sh
    dbg__  $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : dbg error && exit 2
fi

if [ "${arg/grab}" != "$arg" ]; then
    source rsync.sh $BASE
    [ $? -ne 0 ] && echo $BASH_SOURCE : grab error && exit 3
fi

if [ "${arg/ana}" != "$arg" ]; then
    ${IPYTHON:-ipython} --pdb -i $ana
    [ $? -ne 0 ] && echo $BASH_SOURCE : ana error && exit 4
fi

if [ "${arg/pdb}" != "$arg" ]; then
    ${IPYTHON:-ipython} --pdb -i $tra
    [ $? -ne 0 ] && echo $BASH_SOURCE : tra error && exit 4
fi


if [ "$arg" == "pvcap" -o "$arg" == "pvpub" -o "$arg" == "mpcap" -o "$arg" == "mppub" ]; then
    cap_ctx=G4CXTest_${GEOM}_${LAYOUT}_${CHECK}
    case $PICK in
      A) cap_base=$AFOLD/figs ; cap_stem=${cap_ctx}_A${APID} ;;
      B) cap_base=$BFOLD/figs ; cap_stem=${cap_ctx}_B${BPID} ;;
    esac
    if [ -z "$cap_base" -o -z "$cap_stem" ]; then
       echo $BASH_SOURCE : ERROR :  pvcap/pvpub/mpcap/mppub require PICK=A or PICK=B : AB OR BA are not allowed
       exit 2
    fi
    export CAP_STEM=$cap_stem   # stem of the .png screencapture
    export CAP_BASE=$cap_base
    export CAP_REL=ntds3
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

