#!/bin/bash -l 
usage(){ cat << EOU
CSGFoundry_MakeCenterExtentGensteps_Test.sh
===============================================

::

   ~/o/CSG/tests/CSGFoundry_MakeCenterExtentGensteps_Test.sh
   LOG=1 ~/o/CSG/tests/CSGFoundry_MakeCenterExtentGensteps_Test.sh

This checks the center-extent-gensteps used in CSGOptiX/cxs.sh 
by generating some photons on CPU from them and loading into python. 

EOU
}


SDIR=$(dirname $(realpath $BASH_SOURCE))
tmp=/tmp/$USER/opticks
TMP=${TMP:-$tmp}

name="CSGFoundry_MakeCenterExtentGensteps_Test"
script=$SDIR/$name.py 

export PYVISTA_KILL_DISPLAY=1
export FOLD=$TMP/$name


customgeom()
{
    #local geom=HamaXZ_0
    local geom=HamaXZ_1000
    #local geom=XJfixtureConstruction_0
    #local geom=sWorld_XZ

    export GEOM=${GEOM:-$geom}
    ce_offset=0,0,0
    ce_scale=1   
    gridscale=0.10

    if [ "$GEOM" == "sWorld_XZ" ]; then

        moi=sWorld
        cegs=16:0:9:-24   

    elif [ "$GEOM" == "HamaXZ_0" ]; then

        moi=Hama
        cegs=16:0:9:-24

    elif [ "$GEOM" == "HamaXZ_1000" ]; then

        moi=Hama:0:1000
        cegs=16:0:9:-24   
        #gridscale=0.10
        ce_offset=0,-666.6,0

    elif [ "$GEOM" == "XJfixtureConstruction_0" ]; then


        ## see CSGTarget::getCenterExtent

        #iidx=0       # default 
        #iidx=-1
        #iidx=-2
        iidx=-3       # model2world_rtpw = translate * scale * rotate 
        #iidx=-4
        #iidx=-5

        moi="solidXJfixture:10:$iidx"

        #cegs=16:0:9:100                # XZ/RP     (XYZ)->(RTP) 
        cegs=0:16:9:-24                 # YZ/TP
        gridscale=0.05
    fi 
}

source $HOME/.opticks/GEOM/GEOM.sh 


moi=sChimneyAcrylic:0:0
cegs=16:0:9:100
ce_offset=CE


export MOI=${MOI:-$moi}
export CEGS=${CEGS:-$cegs}
export CE_OFFSET=${CE_OFFSET:-$ce_offset}
export CE_SCALE=${CE_SCALE:-$ce_scale}
export GRIDSCALE=${GRIDSCALE:-$gridscale}


logging(){
   export SFrameGenstep=INFO
}
[ -n "$LOG" ] && logging 



defarg="into_run_ana"
arg=${1:-$defarg}

vars="BASH_SOURCE SDIR MOI CEGS CE_OFFSET CE_SCALE GRIDSCALE"

if [ "${arg/info}" != "$arg" ]; then  
   for var in $vars ; do printf "%25s : %s \n" "$var" "${!var}" ; done 
fi

if [ "${arg/run}" != "$arg" ]; then  
   $name
   [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1 
fi

if [ "${arg/dbg}" != "$arg" ]; then  
   dbg__ $name
   [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 2 
fi

if [ "${arg/ana}" != "$arg" ]; then  
   ${IPYTHON:-ipython} --pdb -i $script
   [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 3
fi

exit 0 

