#!/bin/bash
usage(){ cat << EOU
cxt_min.sh : Simtrace Geometry Intersect Creation and Plotting 
===============================================================

Former label "CSGOptiXTMTest Simtrace minimal executable and script for shakedown"
but has become the standard simtrace script.
Uses CSGOptiXTMTest which just does "CSGOptiX::SimtraceMain()".

Envar controls:

GEOM
   picks the geometry together with ${GEOM}_CFBaseFromGEOM

MOI
   specify the frame in which simtrace gensteps will be generated, 
   which sets the region where most of the intersects will be found

CEGS
   "CE-center-extent-Gensteps" specifies orientatation of the grid of 
   gensteps and the number of simtrace rays from each genstep origin   


Usage pattern assuming analysis python environment with 
matplotlib and/or pyvista on laptop::

   cxt_min.sh info_run               ## on workstation
   cxt_min.sh grab                   ## on laptop
   NOGRID=1 MODE=2 cxt_min.sh ana    ## on laptop


Command arguments:

info
   output many vars
fold 
   output just $FOLD where simtrace SEvt are persisted
run
   executes the CSGOptiXTMTest executable with main being just "CSGOptiX::SimtraceMain()"
   
   1. loads envvar configured geometry
   2. generates envvar configured simtrace gensteps
   3. runs CSGOptiX:simtrace which uses OptiX on GPU to intersect
      simtrace rays with the gometry, saving them into the simtrace array
   4. saves SEVt including simtrace array to $FOLD

dbg
   runs the above under gdb 

brab
   old grab for rsyncing the SEvt FOLD between machines
grab 
   rsync an SEvt FOLD from remote to local machine


pdb
   ${IPYTHON:-ipython} plotting of simtrace intersects, typically 
   giving 2D cross section through geometry with matplotlib (MODE:2)
   OR 3D intersect view with pyvista (MODE:3)

ana
   as above but with ${PYTHON:-python} 


About IPYTHON/PYTHON ipython/python overrides
-----------------------------------------------

"Official" python environments might not include the 
matplotlib/pyvista plotting packages which 
will cause MODE 2/3 to give errors for "pdb" and "ana". 
Workaround this by defining IPYTHON/PYTHON envvars
to pick a python install (eg from miniconda) 
which does include the plotting libraries.



EOU
}

cd $(dirname $(realpath $BASH_SOURCE))
script=$(realpath $PWD/cxt_min.py)   ## use python script that is sibling to the bash script

allarg="info_fold_run_dbg_brab_grab_ana"

defarg=run_info
[ -n "$BP" ] && defarg="info_dbg"
arg=${1:-$defarg}


bin=CSGOptiXTMTest
which_bin=$(which $bin)

External_CFBaseFromGEOM=${GEOM}_CFBaseFromGEOM
if [ -n "$GEOM" -a -n "${!External_CFBaseFromGEOM}" -a -d "${!External_CFBaseFromGEOM}" -a -f "${!External_CFBaseFromGEOM}/CSGFoundry/prim.npy" ]; then
    ## distributed usage : where have one fixed geometry for each distribution
    echo $BASH_SOURCE - External GEOM setup detected
    vv="External_CFBaseFromGEOM ${External_CFBaseFromGEOM}"
    for v in $vv ; do printf "%40s : %s \n" "$v" "${!v}" ; done
else
    ## development source tree usage : where need to often switch between geometries 
    source ~/.opticks/GEOM/GEOM.sh   # sets GEOM envvar, use GEOM bash function to setup/edit
    export ${GEOM}_CFBaseFromGEOM=$HOME/.opticks/GEOM/$GEOM
    source ~/.opticks/GEOM/MOI.sh   # sets MOI envvar, use MOI bash function to setup/edit
fi


tmp=/tmp/$USER/opticks
TMP=${TMP:-$tmp}

export EVT=${EVT:-A000}
export BASE=$TMP/GEOM/$GEOM
export BINBASE=$BASE/$bin
export LOGDIR=$BINBASE/$MOI
export FOLD=$TMP/GEOM/$GEOM/$bin/${MOI:-0}/$EVT
export SCRIPT=$(basename $BASH_SOURCE)

version=1
VERSION=${VERSION:-$version}

export OPTICKS_EVENT_MODE=DebugLite
export OPTICKS_INTEGRATION_MODE=1

mkdir -p $LOGDIR
cd $LOGDIR
LOGNAME=$bin.log


# pushing this too high tripped M3 max photon limit
# 16*9*2000 = 0.288 
export CEGS=16:0:9:2000   # XZ default
#export CEGS=16:0:9:1000   # XZ default
#export CEGS=16:0:9:100     # XZ reduce rays for faster rsync
#export CEGS=16:9:0:1000    # try XY

## base photon count without any CEHIGH for 16:0:9:2000 is (2*16+1)*(2*9+1)*2000 = 1,254,000

#export CE_OFFSET=CE    ## offsets the grid by the CE


logging(){
    type $FUNCNAME
    export CSGOptiX=INFO
    export QEvent=INFO
    #export QSim=INFO
    #export SFrameGenstep=INFO
    #export CSGTarget=INFO
    #export SEvt=INFO
    export SEvt__LIFECYCLE=INFO
}
[ -n "$LOG" ] && logging


vars="BASH_SOURCE script bin which_bin allarg defarg arg GEOM ${GEOM}_CFBaseFromGEOM FOLD MOI LOG LOGDIR BASE CUDA_VISIBLE_DEVICES CEGS"


if [ "${arg/info}" != "$arg" ]; then
   for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done
fi

if [ "${arg/fold}" != "$arg" ]; then
    echo $FOLD
fi

if [ "${arg/run}" != "$arg" -o "${arg/dbg}" != "$arg" ]; then

   if [ -f "$LOGNAME" ]; then
       echo $BASH_SOURCE : run/dbg : delete prior LOGNAME $LOGNAME
       rm "$LOGNAME"
   fi

   if [ "${arg/run}" != "$arg" ]; then
       $bin
   elif [ "${arg/dbg}" != "$arg" ]; then
       source dbg__.sh
       dbg__ $bin
   fi
   [ $? -ne 0 ] && echo $BASH_SOURCE run/dbg error && exit 1
fi


if [ "${arg/brab}" != "$arg" -o "${arg/list}" != "$arg" -o "${arg/pub}" != "$arg" ]; then
    ## THIS OLD BASE_grab.sh SYNCS TOO MUCH : BUT IT DOES OTHER THINGS LIKE list and pub
    source BASE_grab.sh $arg
fi

if [ "${arg/grab}" != "$arg" ]; then
    source rsync.sh $FOLD
fi

if [ "${arg/pdb}" != "$arg" ]; then
    ${IPYTHON:-ipython} --pdb -i $script
fi

if [ "${arg/ana}" != "$arg" ]; then
    ${PYTHON:-python} $script
fi


