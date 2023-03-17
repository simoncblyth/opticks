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

DIR=$(dirname $BASH_SOURCE)
bin=U4SimulateTest

export VERSION=${N:-0}
export GEOM=FewPMT
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

vars="GEOM _GEOMList GEOMList"
for var in $vars ; do printf "%-30s : %s \n" "$var" "${!var}" ; done

#exit 0

export BASE=/tmp/$USER/opticks/GEOM/$GEOM/$bin

## process DISABLE/ENABLE controlling u4/tests/U4Physics.cc U4Physics::ConstructOp
export Local_G4Cerenkov_modified_DISABLE=1
export Local_DsG4Scintillation_DISABLE=1
#export G4OpAbsorption_DISABLE=1
#export G4OpRayleigh_DISABLE=1
#export G4OpBoundaryProcess_DISABLE=1

export U4RecorderTest__PRIMARY_MODE=torch  # hmm seems iphoton and torch do same thing internally 

export BeamOn=${BeamOn:-1}
export U4Recorder__PIDX_ENABLED=1 

# python ana level presentation 
export LOC=skip


log=${bin}.log
logN=${bin}_$VERSION.log




#num_ph=1
#num_ph=10
#num_ph=100
#num_ph=1000      # 1k
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


#check=rain_disc
check=rain_line
#check=up_rain_line
#check=escape
#check=rain_dynode
#check=lhs_window_line
#check=lhs_reflector_line
#check=lhs_reflector_point

export CHECK=${CHECK:-$check}

if [ "$LAYOUT" == "one_pmt" ]; then 
    if [ "$CHECK" == "rain_disc" ]; then

        ttype=disc 
        pos=0,0,195
        mom=0,0,-1
        radius=250
        # approx PMT extents : xy -255:255, z -190:190

    elif [ "$CHECK" == "rain_line" ]; then

        ttype=line
        pos=0,0,195    ## 190 grazes HAMA apex, somehow causing "TO TO SD" 
        radius=260     # standand for line from above,  280 hangsover  
        mom=0,0,-1   

    elif [ "$CHECK" == "up_rain_line" ]; then

        ttype=line
        radius=260
        pos=0,0,-195  
        mom=0,0,1        

    elif [ "$CHECK" == "escape" ]; then

        ttype=point
        pos=0,0,100 
        mom=0,0,1
        radius=0

    elif [ "$CHECK" == "rain_dynode" ]; then

        ttype=line
        radius=120    # focus on HAMA dynode
        pos=0,0,-50
        mom=0,0,-1

    elif [ "$CHECK" == "lhs_window_line" ]; then

        ttype=line
        radius=95     
        pos=-300,0,95   ## line from (-300,0,0) to (-300,0,190) 
        mom=1,0,0

    elif [ "$CHECK" == "lhs_reflector_line" ]; then

        ttype=line
        radius=95
        pos=-300,0,-95   ## line from (-300,0,0) to (-300,0,-190)
        mom=1,0,0        

    elif [ "$CHECK" == "lhs_reflector_point" ]; then

        ttype=point
        pos=-300,0,-10     ## PMT left below cathode at Z=0, for shooting the reflector 
        mom=1,0,0
        radius=0

    else
         echo $BASH_SOURCE : ERROR LAYOUT $LAYOUT CHECK $CHECK IS NOT HANDLED
    fi 


elif [ "$LAYOUT" == "two_pmt" ]; then

    echo $BASH_SOURCE : ERROR LAYOUT $LAYOUT CHECK $CHECK IS NOT HANDLED

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

## envout is used to communicate from some python scripts back into this bash script 
## this only makes sense for single N=0, N=1 running 
export ENVOUT=/tmp/$USER/opticks/U4SimulateTest/envout.sh
mkdir -p $(dirname $ENVOUT)


if [ "${arg/fs}" != "$arg" -o "${arg/nfs}" != "$arg" ]; then
    export FOLD=$BASE/$reldir
    [ "$arg" == "nfs" ] && export MODE=0
    ${IPYTHON:-ipython} --pdb -i $DIR/${bin}_fs.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE fs error && exit 3
fi 

if [ "${arg/cf}" != "$arg" -o "${arg/ncf}" != "$arg" ]; then
    [ "$arg" == "ncf" ] && export MODE=0
    export AFOLD=$BASE/ALL0
    export BFOLD=$BASE/ALL1
    ${IPYTHON:-ipython} --pdb -i $DIR/${bin}_cf.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE cf error && exit 4
fi 

if [ "${arg/af}" != "$arg" -o "${arg/naf}" != "$arg" ]; then
    [ "$arg" == "naf" ] && export MODE=0
    ${IPYTHON:-ipython} --pdb -i $DIR/${bin}_af.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE af error && exit 4
fi 

if [ "${arg/ph}" != "$arg" -o "${arg/nph}" != "$arg" ]; then
    export FOLD=$BASE/$reldir
    [ "$arg" == "nph" ] && export MODE=0
    ${IPYTHON:-ipython} --pdb -i $DIR/${bin}_ph.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE ph error && exit 5
fi 

if [ "${arg/mt}" != "$arg" -o "${arg/nmt}" != "$arg" ]; then
    export FOLD=$BASE/$reldir
    [ "$arg" == "nmt" ] && export MODE=0
    ${IPYTHON:-ipython} --pdb -i $DIR/${bin}_mt.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE mt error && exit 5
fi 

if [ "${arg/fk}" != "$arg" -o "${arg/nfk}" != "$arg" ]; then
    export FOLD=$BASE/$reldir
    [ "$arg" == "nfk" ] && export MODE=0
    ${IPYTHON:-ipython} --pdb -i $DIR/${bin}_fk.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE fk error && exit 6
fi 

if [ "${arg/ck}" != "$arg" -o "${arg/nck}" != "$arg" ]; then
    export FOLD=$BASE/$reldir
    [ "$arg" == "nck" ] && export MODE=0
    ${IPYTHON:-ipython} --pdb -i $DIR/${bin}_ck.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE ck error && exit 7
fi 

if [ "${arg/pr}" != "$arg" -o "${arg/npr}" != "$arg" ]; then
    export AFOLD=$BASE/ALL0
    export BFOLD=$BASE/ALL1
    [ "$arg" == "nck" ] && export MODE=0
    ${IPYTHON:-ipython} --pdb -i $DIR/${bin}_pr.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE pr error && exit 8
fi 

if [ "${arg/__}" != "$arg" -o "${arg/n__}" != "$arg" ]; then
    export FOLD=$BASE/$reldir
    [ "$arg" == "n__" ] && export MODE=0
    ${IPYTHON:-ipython} --pdb -i $DIR/${bin}.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE ck error && exit 9
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

NOTICE HOW ENVOUT COMMUNICATION RELIES ON OVERLAPPED RUNNING OF THIS BASH SCRIPT

1. IPYTHON PLOTTING RUNS AND WRITES THE ENVOUT FILE, POPS UP THE GUI WINDOW AND BLOCKS

2. THEN IN A DIFFERENT TAB THE MPCAP/MPPUB IS RUN THAT SOURCES THE ENVOUT
   IN ORDER TO CONFIGURE CAPTURE NAMING

3. FINALLY THE FIRST PYTHON PLOTTER SESSION IS EXITED THAT CLEANS UP THE ENVOUT FILE.  


So the ENVOUT file just contains the config for the currently displayed plot. 

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

