#!/bin/bash
usage(){ cat << EOU
cxt_min.sh : Simtrace Geometry Intersect Creation and Plotting
===============================================================

To make select intersects to show use comma delimted KEY::

     PRIMTAB=1 cxt_min.sh pdb
     PRIMTAB=1 KEY=hotpink,honeydew,darkmagenta cxt_min.sh pdb

     PRIMTAB=1 KEY=honeydew,deeppink cxt_min.sh pdb     ## WORKS FOR 3D GRID TOO


     PRIMTAB=1 KEY=yellow cxt_min.sh pdb             ## looks disjoint, extensive coincidence ?
     PRIMTAB=1 KEY=green cxt_min.sh pdb              ## neck only goes up a little
     PRIMTAB=1 KEY=yellow,turquoise cxt_min.sh pdb
     PRIMTAB=1 GSGRID=1 KEY=yellow,turquoise,green cxt_min.sh pdb
     PRIMTAB=1 GSGRID=1 KEY=yellow,turquoise,green,darkmagenta cxt_min.sh pdb


     GSGRID=0 KEY=blue,orange,lightblue cxt_min.sh pdb
     GSGRID=0 GRID=1 GLOBAL=1 XKEY=blue,orange,lightblue cxt_min.sh pdb
     GSGRID=0 GRID=1 GLOBAL=1 KEY=blue,lightblue,cornflowerblue cxt_min.sh pdb
     GSGRID=0 GRID=0 GLOBAL=0 KEY=blue,lightblue,cornflowerblue,orange cxt_min.sh pdb

     GSGRID=0 GRID=0 GLOBAL=0 XKEY=blue,lightblue,cornflowerblue,orange KEY=magenta cxt_min.sh pdb
     GSGRID=0 GRID=0 GLOBAL=0 KEY=blue,lightblue,cornflowerblue,orange,magenta,tomato cxt_min.sh pdb
         ## illuminating re the Tyvek:magenta,tomato L shape at top of LowerChimney

     PRIMTAB=1 NORMAL=1 cxt_min.sh pdb

     PRIMTAB=1 NORMAL=1 NORMAL_FILTER=100 KEY=~yellow,green ./cxt_min.sh pdb
           ## inverted key selection


TODO: add option to draw a spinkle of intersect normal direction arrows


OVERLAP Check
--------------

::

    PRIMTAB=1 OVERLAP=1 KEY=deeppink,honeydew cxt_min.sh pdb
    PRIMTAB=1 OVERLAP=1 KEY=deeppink,honeydew HIDE=1 cxt_min.sh pdb

    PRIMTAB=1 OVERLAP=1 BOXSEL=-2000,2000,-2000,2000,-1000,1000 KEY=deeppink,honeydew KEYOFF=honeydew:0,0,-27 cxt_min.sh pdb      ##  overlap_pt 1
    PRIMTAB=1 OVERLAP=1 BOXSEL=-2000,2000,-2000,2000,-1000,1000 KEY=deeppink,honeydew KEYOFF=honeydew:0,0,-28 cxt_min.sh pdb      ## NO OVERLAP




Issue
------

LOG=1 shows are writing to different dir from MFOLD where python looks::

    /data1/blyth/tmp/GEOM/J25_4_0_opticks_Debug/CSGOptiXTMTest/ALL0_no_opticks_event_name/A000
    /data1/blyth/tmp/GEOM/J25_4_0_opticks_Debug/CSGOptiXTMTest/PMT_20inch_veto:0:1000/A000



Coincident surface checking
------------------------------

Take a very close look, will see color variation as zoom
in/out when there are coincident intersects with different
boundaries



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
   GRID=1 MODE=2 cxt_min.sh ana      ## on laptop


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

matplotlib wayland warning but pdb succeeds to plot
------------------------------------------------------

::

   qt.qpa.plugin: Could not find the Qt platform plugin "wayland" in ""


EOU
}

cd $(dirname $(realpath $BASH_SOURCE))
script=$(realpath $PWD/cxt_min.py)   ## use python script that is sibling to the bash script

allarg="info_fold_run_dbg_brab_grab_ana"

defarg=info_run_info_pdb
[ -n "$BP" ] && defarg="info_dbg"
arg=${1:-$defarg}
arg2=$2

bin=CSGOptiXTMTest   ## just calls CSGOptiX::SimtraceMain()
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
fi

source $HOME/.opticks/GEOM/MOI.sh 2>/dev/null  ## optionally sets MOI envvar, use MOI bash function to setup/edit
source $HOME/.opticks/GEOM/CUR.sh 2>/dev/null  ## optionally define CUR_ bash function, for controlling directory for screenshots
source $HOME/.opticks/GEOM/EVT.sh 2>/dev/null  ## optionally define AFOLD and/or BFOLD for adding event tracks to simtrace plots

vars=""


if [ -n "$USE_ELV" ]; then
    ## NOT NORMAL TO EXCLUDE GEOMETRY FOR SIMTRACE : BUT SOMETIMES USEFUL TO DO SO
    source $HOME/.opticks/GEOM/ELV.sh

    ## USING DYNAMIC ELV SELECTED GEOMETRY COMES WITH COMPLICATIONS - IT FORCES ALT SAVING OF THE NON-STANDARD GEOMETRY
    cfbase_alt=/tmp/$USER/.opticks/ELV_Dynamic/GEOM/$GEOM
    mkdir -p $cfbase_alt
    export CFBASE_ALT=$cfbase_alt
    export CSGFoundry_Load_saveAlt=1

fi
vars="$vars USE_ELV CFBASE_ALT CSGFoundry_Load_saveAlt"


tmp=/tmp/$USER/opticks
TMP=${TMP:-$tmp}

mode=3
eye=0,10000,0

export MODE=${MODE:-$mode}
export EYE=${EYE:-$eye}


: user can pick EVT but it must be of expected form - otherwise override to A000
evt=A000
if ! [[ $EVT =~ ^[AB][0-9]{3}$ ]]; then
    echo $BASH_SOURCE - WARNING EVT $EVT NOT OF EXPECTED FORM - A000 A001 ... B000 B001 ... - OVERRIDE TO default evt $evt
    EVT=$evt
fi
export EVT=${EVT:-$evt}



export BASE=$TMP/GEOM/$GEOM
export BINBASE=$BASE/$bin
export LOGDIR=$BINBASE/$MOI

#rel=${MOI:-0}
rel=ALL0_no_opticks_event_name   ## SOMEHOW THE DIRECTORY WRITTEN TO HAS CHANGED ?
export MFOLD=$TMP/GEOM/$GEOM/$bin/$rel/$EVT


export SCRIPT=$(basename $BASH_SOURCE)
SCRIPT=${SCRIPT/.sh}

version=1
VERSION=${VERSION:-$version}

export OPTICKS_EVENT_MODE=DebugLite
export OPTICKS_INTEGRATION_MODE=1

mkdir -p $LOGDIR
cd $LOGDIR
LOGNAME=$bin.log


## see SFrameGenstep::StandardizeCEGS for CEGS/CEHIGH [4]/[7]/[8] layouts

#export CEGS=16:0:9:2000   # [4] XZ default
#export CEGS=16:0:9:1000  # [4] XZ default
#export CEGS=16:0:9:100   # [4] XZ reduce rays for faster rsync
#export CEGS=16:9:0:1000  # [4] try XY

#export CEGS=16:9:9:100    # [4] try a 3D grid
export CEGS=16:9:9:500    # [4] try a 3D grid
export CEGS_NPY=/tmp/overlap_pt.npy   # see SFrameGenstep::MakeCenterExtentGenstep_FromFrame


if [ "$CEGS" == "16:0:9:2000" ]; then
    export CEHIGH_0=-16:16:0:0:-4:4:2000:4
    export CEHIGH_1=-16:16:0:0:4:8:2000:4
    #export CEHIGH_0=16:0:9:0:0:10:2000     ## [7] dz:10 aim to land another XZ grid above in Z 16:0:9:2000
    #export CEHIGH_1=-4:4:0:0:-9:9:2000:5   ## [8]
    #export CEHIGH_2=-4:4:0:0:10:28:2000:5  ## [8]
fi



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

debug()
{
    type $FUNCNAME
    export CSGFoundry__getFrameE_VERBOSE=1
    export SEvt__FRAME=1
}

[ "$LOG" == "1" ] && logging
[ "$DBG" == "1" ] && debug


_CUR=GEOM/$GEOM/$SCRIPT/${MOI//:/_}

vars="$vars BASH_SOURCE script bin which_bin allarg defarg arg GEOM ${GEOM}_CFBaseFromGEOM MFOLD MOI SCRIPT _CUR LOG LOGDIR BASE CUDA_VISIBLE_DEVICES CEGS TITLE"

## define TITLE based on ana/pdb control envvars
title="cxt_min.sh pdb"
ee="LINE GLOBAL PRESEL KEY NORMAL NORMAL_FILTER GRID GSGRID PRIMTAB"
for e in $ee ; do
   #printf "%20s : %s \n" "$e" "${!e}"
   [ -n "${!e}" ] && title="$e=${!e} $title"
done
export TITLE="$title"


if [ "${arg/info}" != "$arg" ]; then
   for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done
fi

if [ "${arg/fold}" != "$arg" ]; then
    echo $FOLD
fi

if [ "${arg/ls}" != "$arg" ]; then
    ff="MFOLD"
    for f in $ff ; do printf "%20s : ls -alst %s \n" "$f" "${!f}"  && ls -alst ${!f} ; done
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
    source rsync.sh $MFOLD
fi



if [ "${arg/open}" != "$arg" ]; then
    # open to define current context string which controls where screenshots are copied to
    CUR_open ${_CUR}
fi

if [ "${arg/pdb}" != "$arg" ]; then
    ${IPYTHON:-ipython} --pdb -i $script
fi

if [ "${arg/ana}" != "$arg" ]; then
    ${PYTHON:-python} $script
fi

if [ "${arg/close}" != "$arg" ]; then
    # close to invalidate the context
    CUR_close
fi

if [ "$arg" == "touch"  ]; then
    if [ -n "$arg2" ]; then
        CUR_touch "$arg2"
    else
        echo $BASH_SOURCE:touch needs arg2 datetime accepted by CUR_touch eg "cxt_min.sh touch 11:00"
    fi
fi

if [ "${arg/cfg}" != "$arg" ]; then
    cfgpath=$(realpath ~/.opticks/GEOM/cxt_min.ini)
    cmd="vim $cfgpath"
    echo $cmd
    eval  $cmd
fi

exit 0

