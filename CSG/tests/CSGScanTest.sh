#!/bin/bash 
usage(){ cat << EOU
CSGScanTest.sh
===============

::

    ~/o/CSG/tests/CSGScanTest.sh


rbin
    runs locally built bin /tmp/$USER/opticks/${name}.build/$name
run
    runs om standardly built name $name

EOU
}


cd $(dirname $(realpath $BASH_SOURCE))

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}   # just use some CUDA headers, not using GPU 

#defarg="info_build_run_ana"
defarg="info_run_ana"
[ -n "$BP" ] && defarg="info_dbg" 

arg=${1:-$defarg}



gdb__ () 
{ 
    if [ -z "$BP" ]; then
        H="";
        B="";
        T="-ex r";
    else
        H="-ex \"set breakpoint pending on\"";
        B="";
        for bp in $BP;
        do
            B="$B -ex \"break $bp\" ";
        done;
        T="-ex \"info break\" -ex r";
    fi;
    local runline="gdb $H $B $T --args $* ";
    echo $runline;
    date;
    eval $runline;
    date
}


name=CSGScanTest 
tmpd=/tmp/$USER/opticks/$name
bdir=/tmp/$USER/opticks/${name}.build
mkdir -p $bdir
bin=$bdir/$name
script=CSGScanTest.py 


#geom=JustOrb
geom=DifferenceBoxSphere
#geom=UnionBoxSphere

export GEOM=${GEOM:-$geom}
export FOLD=$tmpd/$GEOM
export BASE=$FOLD   # need BASE envvar for grab 

export CSGScanTest__init_SAVEFOLD=$HOME/.opticks/GEOM/$GEOM   ## define to save the CSGMaker solid CSGFoundry

vars="FOLD GEOM defarg arg SAVEFOLD"

if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi 


if [ "${arg/build}" != "$arg" ]; then

    opts=""
    #opts="-DDEBUG=1"    ## very verbose 

    cui=../CSGScan.cu
    cuo=$bdir/CSGScan_cu.o
    nvcc -c $cui \
             -std=c++11 -lstdc++ \
             -I.. \
             -I$OPTICKS_PREFIX/include/SysRap \
             -o $cuo 
    [ $? -ne 0 ] && echo $BASH_SOURCE : nvcc compile error cui $cui cuo $cuo  && exit 1

    srcs="$name.cc 
          ../CSGFoundry.cc 
          ../CSGImport.cc 
          ../CSGSolid.cc 
          ../CSGCopy.cc 
          ../CSGPrim.cc 
          ../CSGNode.cc 
          ../CSGScan.cc 
          ../CSGTarget.cc 
          ../CSGMaker.cc 
          ../CU.cc 
          $cuo
          "

    gcc \
        $srcs \
        -I.. \
        -std=c++11 -lm \
        $opts \
        -I${CUDA_PREFIX}/include \
        -I${OPTICKS_PREFIX}/externals/glm/glm \
        -I${OPTICKS_PREFIX}/include/SysRap \
        -I${OPTICKS_PREFIX}/externals/plog/include \
        -L${CUDA_PREFIX}/lib64 -lcudart -lstdc++ \
        -L${OPTICKS_PREFIX}/lib64 \
        -lSysRap \
        -DWITH_CHILD \
        -o $bin
        #-DWITH_VERBOSE \

    [ $? -ne 0 ] && echo build error && exit 1
fi 


if [ "${arg/rbin}" != "$arg" ]; then 
    echo $BASH_SOURCE - GEOM $GEOM $arg
    $bin
    [ $? -ne 0 ] && echo rbin error && exit 2
fi 

if [ "${arg/run}" != "$arg" ]; then 
    echo $BASH_SOURCE - GEOM $GEOM $arg
    $name
    [ $? -ne 0 ] && echo run error && exit 2
fi 

if [ "${arg/dbg}" != "$arg" ]; then 
    echo $BASH_SOURCE - GEOM $GEOM 
    gdb__ $name
    [ $? -ne 0 ] && echo dbg error && exit 2
fi 

if [ "${arg/grab}" != "$arg" ]; then 
    source ../../bin/BASE_grab.sh $arg
    [ $? -ne 0 ] && echo grab error && exit 4
fi 

if [ "${arg/ana}" != "$arg" ]; then 
    ${IPYTHON:-ipython} -i --pdb $script
    [ $? -ne 0 ] && echo ana error && exit 4
fi 

exit 0 
