#!/bin/bash -l 

msg="=== $BASH_SOURCE :"
usage(){ cat << EOU
csg.sh : using test/CSGIntersectSolidTest.cc tests/CSGIntersectSolidTest.py without GEOM envvar 
=================================================================================================

TODO: factor off common parts of csg_geochain.sh and this script csg.sh 

EOU
}





dx=0
dy=0
dz=0
pho=${PHO:--100} 

case $pho in
  -*)  echo $msg using regular bicycle spoke photon directions ;; 
   *)  echo $msg using random photon directions                ;;
esac


#sopr=0:0_YZ
#sopr=0:0_XYZ
sopr=0:3_XY

export VERBOSE=1
export SOPR=${SOPR:-$sopr}


case $SOPR in  
   *_XZ) cegs=16:0:9:$dx:$dy:$dz:$pho  ;;  
   *_YZ) cegs=0:16:9:$dx:$dy:$dz:$pho  ;;  
   *_XY) cegs=16:9:0:$dx:$dy:$dz:$pho  ;;  
   *_ZX) cegs=9:0:16:$dx:$dy:$dz:$pho  ;;  
   *_ZY) cegs=0:9:16:$dx:$dy:$dz:$pho  ;;  
   *_YX) cegs=9:16:0:$dx:$dy:$dz:$pho  ;;  
   *_XYZ) cegs=9:16:9:$dx:$dy:$dz:$pho ;;  
       *) echo $msg UNEXPECTED SUFFIX FOR SOPR $SOPR WHICH DOES NOT END WITH ONE OF : _XZ _YZ _XY _ZX _ZY _YX _XYZ  && exit 1   ;; 
esac

export CEGS=${CEGS:-$cegs}


bin=CSGIntersectSolidTest
script=tests/CSGIntersectSolidTest.py 


arg=${1:-run_ana}

if [ "${arg/dump}" != "$arg" ]; then 
   echo $msg CSGGeometryTest dump
   DUMP=1 $bin 
   exit 0 

elif [ "${arg/run}" != "$arg" ]; then

    if [ -n "$DEBUG" ]; then 
        echo $msg running binary $bin under debugger
        if [ "$(uname)" == "Darwin" ]; then
            lldb__ $bin
        else
            gdb $bin
        fi 
        [ $? -ne 0 ] && echo $msg error while running binary $bin under debugger  && exit 1
    else
        echo $msg running binary $bin
        $bin
        [ $? -ne 0 ] && echo $msg error while running binary $bin  && exit 1
    fi 
fi



if [ "${arg/ana}" != "$arg" ]; then

    cfbase_sh=/tmp/CFBASE.sh
    source $cfbase_sh

    [ -z "$CFBASE" ] && echo $msg ERROR no CFBASE && exit 1 
    [ -n "$CFBASE" ] && echo $msg using CFBASE $CFBASE from cfbase_sh $cfbase_sh 

    echo $msg running script $script
    ${IPYTHON:-ipython} --pdb -i $script
    [ $? -ne 0 ] && echo $msg script error && exit 2
fi







echo 0 

