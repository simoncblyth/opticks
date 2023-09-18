#!/bin/bash -l 
usage(){ cat << EOU
U4SimulateTest.sh  (formerly U4PMTFastSimTest.sh)
===================================================

Covering the PMT*POM quadrants (POM:PMT Optical Model)::

    u4t

    N=0 POM=0 ./U4SimulateTest.sh   # unnatural geom , traditional POM 
    N=1 POM=0 ./U4SimulateTest.sh   # natural geom   , traditional POM

    N=0 POM=1 ./U4SimulateTest.sh   # unnatural geom , multifilm POM 
    N=1 POM=1 ./U4SimulateTest.sh   # natural geom   , multifilm POM


Analysis loading saved results::

    PID=726 ./U4SimulateTest.sh nana

    N=1 MODE=0 ./U4SimulateTest.sh ph  # no GUI with NumPy 
    N=1 MODE=2 ./U4SimulateTest.sh ph  # 2D GUI with matplotlib
    N=1 MODE=3 ./U4SimulateTest.sh ph  # 3D GUI with pyvista


Rerunning single photons off the same g4state is a bit delicate to arrange.
This incantation succeeds to rerun the N=0 big bouncer with N=1::

     u4t                       ## cd ~/opticks/u4/tests

     vi U4SimulateTest.sh      ## switch to running_mode=SRM_G4STATE_SAVE

     N=0 ./U4SimulateTest.sh   ## saves g4state into /tmp/blyth/opticks/GEOM/hamaLogicalPMT/U4SimulateTest/ALL0

     vi U4SimulateTest.sh      ## switch to running_mode=SRM_G4STATE_RERUN for PID 726 
    
     N=1 ./U4SimulateTest.sh   ## saves to /tmp/blyth/opticks/GEOM/hamaLogicalPMT/U4SimulateTest/SEL1


After that can compare timings::

    ./U4SimulateTest.sh cf 

EOU
}

DIR=$(cd $(dirname $BASH_SOURCE) && pwd)
bin=U4SimulateTest

#geom=V1J008
geom=FewPMT

export VERSION=${N:-0}
export GEOM=${GEOM:-$geom}

origin=$HOME/.opticks/GEOM/$GEOM/origin.gdml
if [ -f "$origin" ]; then
   export ${GEOM}_GDMLPath=$origin
   export U4VolumeMaker=INFO
fi 

geomscript=$DIR/$GEOM.sh 

if [ -f "$geomscript" ]; then  
    source $geomscript
else
    echo $BASH_SOURCE : no geomscript $geomscript
fi 

## moved LAYOUT and FAKES control inside geomscript so its in common 
## between U4SimulateTest.sh and U4SimtraceTest.sh 
## makes sense for everything that is very GEOM specific to be within the geomscript

_GEOMList=${GEOM}_GEOMList
GEOMList=${!_GEOMList}


export BASE=/tmp/$USER/opticks/GEOM/$GEOM/$bin

## process DISABLE/ENABLE controlling u4/tests/U4Physics.cc U4Physics::ConstructOp
export Local_G4Cerenkov_modified_DISABLE=1
export Local_DsG4Scintillation_DISABLE=1
#export G4OpAbsorption_DISABLE=1
#export G4OpRayleigh_DISABLE=1
#export G4OpBoundaryProcess_DISABLE=1

export U4App__PRIMARY_MODE=torch  # hmm seems iphoton and torch do same thing internally 
export BeamOn=${BeamOn:-1}
export U4Recorder__PIDX_ENABLED=1 
#export U4Recorder__UserSteppingAction_Optical_ClearNumberOfInteractionLengthLeft=1  ## makes easier to randmn align 



evt=n001
export EVT=${EVT:-$evt}       # used for FOLD envvars




# python ana level presentation 
export LOC=skip

log=${bin}.log
logN=${bin}_$VERSION.log

#num_photons=1
#num_photons=10
#num_photons=100
#num_photons=1000      # 1k
num_photons=10000    # 10k
#num_photons=50000    # 50k 
#num_photons=100000   # 100k
#num_photons=1000000  # 1M

NUM_PHOTONS=${NUM_PHOTONS:-$num_photons}

if [ -n "$RERUN" ]; then 
   export OPTICKS_G4STATE_RERUN=$RERUN
   running_mode=SRM_G4STATE_RERUN
else
   running_mode=SRM_G4STATE_SAVE  
fi 

case $running_mode in 
   SRM_G4STATE_SAVE)  reldir=ALL$VERSION ;; 
   SRM_G4STATE_RERUN) reldir=SEL$VERSION ;; 
esac

## sysrap/SEventConfig 
export OPTICKS_RUNNING_MODE=$running_mode   # see SEventConfig::RunningMode
export OPTICKS_MAX_BOUNCE=31                # can go upto 31 now that extended sseq.h 
export OPTICKS_EVENT_MODE=StandardFullDebug
export OPTICKS_G4STATE_SPEC=${NUM_PHOTONS}:38       # default is only 1000:38 to keep state files small 

export SEvent_MakeGensteps_num_ph=${NUM_PHOTONS}
source $DIR/storch_FillGenstep.sh 


loglevel(){
   export U4Recorder=INFO
   export U4Physics=INFO
   export junoPMTOpticalModel=INFO
   export junoPMTOpticalModelSimple=INFO
   #export SEvt=INFO     ## thus is exceedingly verbose
   export SEventConfig=INFO
   export InstrumentedG4OpBoundaryProcess=INFO
   export ShimG4OpAbsorption=INFO
   export ShimG4OpRayleigh=INFO
}


if [ "$running_mode" == "SRM_G4STATE_RERUN" ]; then 
   echo $BASH_SOURCE : switch on logging when doing single photon RERUN
   loglevel  
else
   #echo $BASH_SOURCE : switch on some logging anyhow : THIS WILL BE VERBOSE
   #export junoPMTOpticalModel=INFO
   #export CustomG4OpBoundaryProcess=INFO

   export U4Physics=INFO
   export U4Recorder=INFO
   #export SEvt=INFO
   export C4OpBoundaryProcess__PIDX_ENABLED=1
fi 


if [ -n "$SIMTRACE" ]; then
   export U4Recorder__EndOfRunAction_Simtrace=1  
   export SFrameGenstep=INFO
fi 


## analysis modes beginning with n: nfs/ncf/naf/nph are NumPy only (without matplotlib and pyvista)
## envout is used to communicate from some python scripts back into this bash script 
## this only makes sense for single N=0, N=1 running 

export ENVOUT=/tmp/$USER/opticks/U4SimulateTest/envout.sh
mkdir -p $(dirname $ENVOUT)

export FOLD=$BASE/$reldir/$EVT
export AFOLD=$BASE/ALL0/$EVT     ## for comparisons 
export BFOLD=$BASE/ALL1/$EVT




vars="BASH_SOURCE GEOM _GEOMList GEOMList BASE EVT FOLD NUM_PHOTONS script"










defarg="run_ph"
#defarg="run_mt"
[ -n "$BP" ] && defarg="dbg"

arg=${1:-$defarg}
[ "${arg:0:1}" == "n" ] && export MODE=0

if [ "${arg/fs}" != "$arg" -o "${arg/nfs}" != "$arg" ]; then
    script=$DIR/${bin}_fs.py  
elif [ "${arg/cf}" != "$arg" -o "${arg/ncf}" != "$arg" ]; then
    script=$DIR/${bin}_cf.py 
elif [ "${arg/af}" != "$arg" -o "${arg/naf}" != "$arg" ]; then
    script=$DIR/${bin}_af.py 
elif [ "${arg/ph}" != "$arg" -o "${arg/nph}" != "$arg" ]; then
    script=$DIR/${bin}_ph.py 
elif [ "${arg/mt}" != "$arg" -o "${arg/nmt}" != "$arg" ]; then
    script=$DIR/${bin}_mt.py 
elif [ "${arg/fk}" != "$arg" -o "${arg/nfk}" != "$arg" ]; then
    script=$DIR/${bin}_fk.py 
elif [ "${arg/ck}" != "$arg" -o "${arg/nck}" != "$arg" ]; then
    script=$DIR/${bin}_ck.py 
elif [ "${arg/pr}" != "$arg" -o "${arg/npr}" != "$arg" ]; then
    script=$DIR/${bin}_pr.py 
elif [ "${arg/tt}" != "$arg" -o "${arg/ntt}" != "$arg" ]; then
    script=$OPTICKS_HOME/sysrap/sevt_tt.py 
elif [ "${arg/__}" != "$arg" -o "${arg/n__}" != "$arg" ]; then
    script=$DIR/${bin}.py 
fi 


if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%-30s : %s \n" "$var" "${!var}" ; done
fi

if [ "${arg/run}" != "$arg" ]; then
    [ -f "$log" ] && rm $log 
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1 
    [ -f "$log" ] && echo $BASH_SOURCE rename log $log to logN $logN && mv $log $logN    
fi 


if [ "$arg" == "dbg" ]; then
   bp=MixMaxRng::flat
   #bp="$bp CustomG4OpBoundaryProcess::DielectricMetal"
   #bp="$bp CustomG4OpBoundaryProcess::ChooseReflection" 
   #bp="$bp CustomG4OpBoundaryProcess::DoAbsorption" 
   #bp="$bp CustomG4OpBoundaryProcess::DoReflection"
   #export BP=${BP:-$bp}
fi 

if [ "${arg/dbg}" != "$arg" ]; then
    [ -f "$log" ] && rm $log 
    case $(uname) in 
        Darwin) lldb__ $bin ;;
        Linux)   gdb__ $bin ;;
    esac
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 2
fi 

if [ -n "$script" -a -f "$script" ]; then 
    ${IPYTHON:-ipython} --pdb -i $script 
    [ $? -ne 0 ] && echo $BASH_SOURCE script $script error && exit 4
else
    echo $BASH_SOURCE no ana script $script is defined OR does not exist for arg $arg 
fi 


if [ -f "$ENVOUT" ]; then 
    if [ "$arg" == "pvcap" -o "$arg" == "pvpub" -o "$arg" == "mpcap" -o "$arg" == "mppub" ]; then
        echo $BASH_SOURCE : detected that python wrote ENVOUT $ENVOUT : sourcing this
        cat $ENVOUT
        source $ENVOUT
        env | grep ENVOUT 
    else
        echo $BASH_SOURCE : remove prior ENVOUT $ENVOUT 
        rm $ENVOUT
    fi 
fi


notes(){ cat << EON
ENVOUT COMMUNICATION FROM PYTHON BACK TO BASH
-----------------------------------------------

The U4SimulateTest_pr.py script writes ENVOUT 


NOTICE HOW ENVOUT COMMUNICATION RELIES ON OVERLAPPED RUNNING OF THIS BASH SCRIPT

1. IPYTHON PLOTTING RUNS AND WRITES THE ENVOUT FILE, POPS UP THE GUI WINDOW AND BLOCKS

2. THEN IN A DIFFERENT TAB THE MPCAP/MPPUB IS RUN THAT SOURCES THE ENVOUT
   IN ORDER TO CONFIGURE CAPTURE NAMING

3. FINALLY THE FIRST PYTHON PLOTTER SESSION IS EXITED THAT CLEANS UP THE ENVOUT FILE.  

So the ENVOUT file just contains the config for the currently displayed plot, and 
should only exist when a plot is being displayed. More than one plot of the same 
type displayed at the same time is not handled. 

EON
}

if [ "$arg" == "pvcap" -o "$arg" == "pvpub" -o "$arg" == "mpcap" -o "$arg" == "mppub" ]; then
    if [ -n "$ENVOUT_VERSION" ]; then
        echo $BASH_SOURCE picking up ENVOUT_VERSION $ENVOUT_VERSION 
        VERSION=$ENVOUT_VERSION
    fi 

    if [ -n "$ENVOUT_CAP_STEM" ]; then
        echo $BASH_SOURCE picking up ENVOUT_CAP_STEM $ENVOUT_CAP_STEM 
        export CAP_STEM=$ENVOUT_CAP_STEM
    else
        export CAP_STEM=$GEOM
    fi 

    export CAP_BASE=$BASE/$VERSION/figs
    export CAP_REL=U4SimulateTest
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

