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

    cd ~/opticks/u4/tests

     vi U4SimulateTest.sh      ## switch to running_mode=SRM_G4STATE_SAVE

     N=0 ./U4SimulateTest.sh   ## saves g4state into /tmp/blyth/opticks/GEOM/hamaLogicalPMT/U4SimulateTest/ALL0

     vi U4SimulateTest.sh      ## switch to running_mode=SRM_G4STATE_RERUN for PID 726 
    
     N=1 ./U4SimulateTest.sh   ## saves to /tmp/blyth/opticks/GEOM/hamaLogicalPMT/U4SimulateTest/SEL1


After that can compare timings::

    ./U4SimulateTest.sh cf 



EOU
}

bin=U4SimulateTest
export GEOM=FewPMT
export BASE=/tmp/$USER/opticks/GEOM/$GEOM/$bin

## process DISABLE/ENABLE controlling u4/tests/U4Physics.cc U4Physics::ConstructOp
export Local_G4Cerenkov_modified_DISABLE=1
export Local_DsG4Scintillation_DISABLE=1
#export G4OpAbsorption_DISABLE=1
#export G4OpRayleigh_DISABLE=1
#export G4OpBoundaryProcess_DISABLE=1
export G4FastSimulationManagerProcess_ENABLE=1  
## HMM: should FastSim process be switched off for N=1 running ? 

export U4RecorderTest__PRIMARY_MODE=torch  # hmm seems iphoton and torch do same thing internally 


## u4/tests/U4SimulateTest.cc
export BeamOn=${BeamOn:-1}

geomscript=$GEOM.sh 
export VERSION=${N:-0}

if [ -f "$geomscript" ]; then  
    source $geomscript $VERSION 
else
    echo $BASH_SOURCE : no geomscript $geomscript
fi 

## moved LAYOUT control inside geomscript so its in common 
## between U4SimulateTest.sh and U4SimtraceTest.sh 


if [ "$VERSION" == "0" ]; then 
    f0=Pyrex/Pyrex:AroundCircle0/hama_body_phys
    f1=Pyrex/Pyrex:hama_body_phys/AroundCircle0
    f2=Vacuum/Vacuum:hama_inner1_phys/hama_inner2_phys
    f3=Vacuum/Vacuum:hama_inner2_phys/hama_inner1_phys
    f4=Pyrex/Pyrex:AroundCircle1/nnvt_body_phys
    f5=Pyrex/Pyrex:nnvt_body_phys/AroundCircle1
    f6=Vacuum/Vacuum:nnvt_inner1_phys/nnvt_inner2_phys
    f7=Vacuum/Vacuum:nnvt_inner2_phys/nnvt_inner1_phys

    f8=Pyrex/Pyrex:nnvt_body_phys/nnvt_log_pv 

    case $LAYOUT in 
       two_pmt) fakes="$f0,$f1,$f2,$f3,$f4,$f5,$f6,$f7" ;;
       one_pmt) fakes="$f0,$f1,$f2,$f3,$f4,$f5,$f6,$f7,$f8" ;;
    esac
fi 

export U4Recorder__FAKES="$fakes"
export U4Recorder__FAKES_SKIP=1
export U4Recorder__PIDX_ENABLED=1 

# python ana level presentation 
export LOC=skip


log=${bin}.log
logN=${bin}_$VERSION.log




#num_ph=2
#num_ph=10
#num_ph=1000      #  1k
num_ph=10000    # 10k
#num_ph=100000   # 100k
#num_ph=1000000  # 1M




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
export OPTICKS_G4STATE_SPEC=$num_ph:38       # default is only 1000:38 to keep state files small 

# standalone access to PMT data 
export PMTSimParamData_BASE=$HOME/.opticks/GEOM/J007/CSGFoundry/SSim/juno



radius=250
#radius=0
[ $num_ph -lt 11  ] && radius=0



if [ "$LAYOUT" == "two_pmt" ]; then

    ttype=point
    case $ttype in 
      disc) pos=0,0,0 ;;
      line) pos=0,0,0 ;;
     point) pos=0,0,-120 ;;
    esac

    ## initial direction
    mom=-1,0,0   # with two_pmt layout -X is towards NNVT
    #mom=1,0,0     # with two_pmt layout +X is towards HAMA

elif [ "$LAYOUT" == "one_pmt" ]; then 

    ttype=point
    case $ttype in 
      disc) pos=0,0,0 ;;
      line) pos=0,0,0 ;;
     point) pos=0,0,100 ;;  # PMT upper mid-vacuum 
    esac
    mom=0,0,1     # +Z directly upwards
fi 

export SEvent_MakeGensteps_num_ph=$num_ph
export storch_FillGenstep_type=$ttype
export storch_FillGenstep_radius=$radius
export storch_FillGenstep_pos=$pos
export storch_FillGenstep_mom=$mom

loglevel(){
   export U4Recorder=INFO
   export U4Physics=INFO
   export junoPMTOpticalModel=INFO
   export junoPMTOpticalModelSimple=INFO
   #export SEvt=INFO
   export SEventConfig=INFO
   export InstrumentedG4OpBoundaryProcess=INFO
   export ShimG4OpAbsorption=INFO
   export ShimG4OpRayleigh=INFO
}


if [ "$running_mode" == "SRM_G4STATE_RERUN" ]; then 
   echo $BASH_SOURCE : switch on logging when doing single photon RERUN
   loglevel  
else
   echo $BASH_SOURCE : switch on some logging anyhow : THIS WILL BE VERBOSE
   #export junoPMTOpticalModel=INFO
   #export CustomG4OpBoundaryProcess=INFO
fi 


defarg="run_ph"
#defarg="run_mt"
[ -n "$BP" ] && defarg="dbg"

arg=${1:-$defarg}

if [ "$arg" == "dbg" ]; then
   bp=MixMaxRng::flat
   #bp="CustomG4OpBoundaryProcess::DielectricMetal CustomG4OpBoundaryProcess::ChooseReflection CustomG4OpBoundaryProcess::DoAbsorption CustomG4OpBoundaryProcess::DoReflection"
   #export BP=${BP:-$bp}
fi 



if [ "${arg/run}" != "$arg" ]; then
    [ -f "$log" ] && rm $log 
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1 

    [ -f "$log" ] && echo $BASH_SOURCE rename log $log to logN $logN && mv $log $logN    
    ## HMM: probably an envvar can change the logname directly ? 
fi 

if [ "${arg/dbg}" != "$arg" ]; then
    [ -f "$log" ] && rm $log 
    case $(uname) in 
        Darwin) lldb__ $bin ;;
        Linux)   gdb__ $bin ;;
    esac
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 2
fi 

## analysis modes beginning with n: nfs/ncf/naf/nph are NumPy only (without matplotlib and pyvista)

if [ "${arg/fs}" != "$arg" -o "${arg/nfs}" != "$arg" ]; then
    export FOLD=$BASE/$reldir
    [ "$arg" == "nfs" ] && export MODE=0
    ${IPYTHON:-ipython} --pdb -i ${bin}_fs.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE fs error && exit 3
fi 

if [ "${arg/cf}" != "$arg" -o "${arg/ncf}" != "$arg" ]; then
    [ "$arg" == "ncf" ] && export MODE=0

    export AFOLD=$BASE/ALL0
    export BFOLD=$BASE/ALL1
    ${IPYTHON:-ipython} --pdb -i ${bin}_cf.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE cf error && exit 4
fi 

if [ "${arg/af}" != "$arg" -o "${arg/naf}" != "$arg" ]; then
    [ "$arg" == "naf" ] && export MODE=0
    ${IPYTHON:-ipython} --pdb -i ${bin}_af.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE af error && exit 4
fi 

if [ "${arg/ph}" != "$arg" -o "${arg/nph}" != "$arg" ]; then
    export FOLD=$BASE/$reldir
    [ "$arg" == "nph" ] && export MODE=0
    ${IPYTHON:-ipython} --pdb -i ${bin}_ph.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE ph error && exit 5
fi 

if [ "${arg/mt}" != "$arg" -o "${arg/nmt}" != "$arg" ]; then
    export FOLD=$BASE/$reldir
    [ "$arg" == "nph" ] && export MODE=0
    ${IPYTHON:-ipython} --pdb -i ${bin}_mt.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE mt error && exit 5
fi 





exit 0 

