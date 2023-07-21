#!/bin/bash -l 
usage(){ cat << EOU
QPMT_Test.sh : standalone build variant of standardly build QPMTTest.sh 
==============================================================================================

* standalone builds are useful for testing and to see exactly
  what are depending on 

* note that the build is using libSysRap, it would be lower level to 
  directly operate from the sysrap sources : but the tree of dependencies
  needs pruning to make that workable in addition to making more of 
  sysrap header only 
   
EOU
}

SCRIPT=$(basename $BASH_SOURCE)
export SCRIPT

REALDIR=$(cd $(dirname $BASH_SOURCE) && pwd)
REALFOLD=$(dirname $REALDIR)

name=QPMT_Test
export FOLD=/tmp/$name
bin=$FOLD/$name

source $HOME/.opticks/GEOM/GEOM.sh # define GEOM envvar 

defarg="build_run_ana"
arg=${1:-$defarg}

logging(){
   export QPMT=INFO
}
logging


custom4_prefix=${OPTICKS_PREFIX}_externals/custom4/0.1.6
CUSTOM4_PREFIX=${CUSTOM4_PREFIX:-$custom4_prefix}

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}

vars="BASH_SOURCE REALDIR REALFOLD FOLD GEOM name CUSTOM4_PREFIX CUDA_PREFIX"


if [ "${arg/info}" != "$arg" ]; then 
    for var in $vars ; do printf "%30s : %s \n" "$var" "${!var}" ; done 
fi

if [ "${arg/build}" != "$arg" ]; then  

    mkdir -p $FOLD

    cus="QPMT QProp"
    for cu in $cus 
    do
        cui=$REALFOLD/$cu.cu
        cuo=$FOLD/${cu}_cu.o  
        nvcc -c $cui \
             -std=c++11 -lstdc++ \
             -I.. \
             -DWITH_THRUST \
             -DWITH_CUSTOM4 \
             -I$CUSTOM4_PREFIX/include/Custom4 \
             -I$OPTICKS_PREFIX/include/SysRap \
             -o $cuo
        [ $? -ne 0 ] && echo $BASH_SOURCE : nvcc compile error cu $cu  && exit 1
        echo $BASH_SOURCE : cui $cui cuo $cuo
    done 

    ccs="QPMT QProp QU"
    for cc in $ccs 
    do 
        cci=$REALFOLD/$cc.cc
        cco=$FOLD/${cc}_cc.o  
        gcc -c $cci \
        -g \
        -std=c++11 \
        -I.. \
        -DWITH_CUSTOM4 \
        -DWITH_THRUST \
        -I$CUSTOM4_PREFIX/include/Custom4 \
        -I$OPTICKS_PREFIX/include/SysRap \
        -I$OPTICKS_PREFIX/include/OKConf \
        -I$OPTICKS_PREFIX/externals/plog/include \
        -I$OPTICKS_PREFIX/externals/glm/glm \
        -I$CUDA_PREFIX/include \
        -o $cco   
       [ $? -ne 0 ] && echo $BASH_SOURCE : gcc compile error cc $cc  && exit 2
       echo $BASH_SOURCE : cci $cci cco $cco
    done

    gcc  $name.cc \
         $FOLD/QPMT_cu.o \
         $FOLD/QProp_cu.o \
         $FOLD/QPMT_cc.o \
         $FOLD/QProp_cc.o \
         $FOLD/QU_cc.o \
         -g \
        -std=c++11 -lstdc++ \
        -I.. \
        -DWITH_CUSTOM4 \
        -DWITH_THRUST \
        -I$CUSTOM4_PREFIX/include/Custom4 \
        -I$OPTICKS_PREFIX/include/SysRap \
        -I$OPTICKS_PREFIX/include/OKConf \
        -I$OPTICKS_PREFIX/externals/plog/include \
        -I$OPTICKS_PREFIX/externals/glm/glm \
        -I$CUDA_PREFIX/include \
        -L$OPTICKS_PREFIX/lib  \
        -lOKConf \
        -lSysRap \
        -L$CUDA_PREFIX/lib \
        -lcudart \
        -o $bin
       [ $? -ne 0 ] && echo $BASH_SOURCE : gcc comple link error  && exit 2
 
       echo $BASH_SOURCE : compiled and linked bin : $bin 
fi 
 
if [ "${arg/run}" != "$arg" ]; then  
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 3
fi 

if [ "${arg/dbg}" != "$arg" ]; then  
    case $(uname) in
       Darwin) lldb__ $bin ;;
       Linux) gdb__ $bin ;;
    esac
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 4
fi 

if [ "${arg/ana}" != "$arg" ]; then  
    ${IPYTHON:-ipython} --pdb -i $REALDIR/QPMTTest.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 2 
fi 

if [ "$arg" == "mpcap" -o "$arg" == "mppub" ]; then
    export CAP_BASE=$FOLD/figs
    export CAP_REL=QPMTTest
    export CAP_STEM=QPMTTest
    case $arg in  
       mpcap) source mpcap.sh cap  ;;  
       mppub) source mpcap.sh env  ;;  
    esac

    if [ "$arg" == "mppub" ]; then 
        source epub.sh 
    fi  
fi 

exit 0 
