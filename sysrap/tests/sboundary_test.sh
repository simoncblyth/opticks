#!/bin/bash -l 
usage(){ cat << EOU
sboundary_test.sh
===================

::

    N=160 POLSCALE=10 AOI=BREWSTER ./sboundary_test.sh 
    N=160 POLSCALE=10 AOI=45 ./sboundary_test.sh 

    N=4 POLSCALE=10 AOI=BREWSTER  ./sboundary_test.sh 

EOU
}

name=sboundary_test
export FOLD=/tmp/$USER/opticks/$name
mkdir -p $FOLD

n=16
force=R  # R/T/N
aoi=CRITICAL
n1=1.5
n2=1.0
b=1

export N=${N:-$n}
export FORCE=${FORCE:-$force}
export AOI=${AOI:-$aoi}
export N1=${N1:-$n1}
export N2=${N2:-$n2}
export B=${B:-$b}

topline="opticks/sysrap/tests/sboundary_test.sh N:$N N1:$N1 N2:$N2 FORCE:$FORCE AOI $AOI B:$B "
case $AOI in 
   BREWSTER) topline="$topline (Polarizing angle)" ;; 
   CRITICAL) topline="$topline (TIR: Total Internal Reflection, ie no refract)" ;; 
          *) topline="$topline" ;;
esac
topline="$topline EYE $EYE LOOK $LOOK"
export TOPLINE=${TOPLINE:-$topline}
export GEOM=AOI_${AOI}


defarg=build_run_ana
arg=${1:-$defarg}

if [ "${arg/build}" != "$arg" ]; then 
   gcc $name.cc \
          -std=c++11 -lstdc++ \
          -I.. \
          -DMOCK_CURAND \
          -I/usr/local/cuda/include \
          -I$OPTICKS_PREFIX/externals/glm/glm \
          -I$OPTICKS_PREFIX/externals/plog/include \
          -o $FOLD/$name 

   [ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then 
   $FOLD/$name
   [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2 
fi 

if [ "${arg/ana}" != "$arg" ]; then 
   ${IPYTHON:-ipython} --pdb -i $name.py 
   [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 3 
fi 

if [ "$arg" == "pvcap" -o "$arg" == "pvpub" -o "$arg" == "mpcap" -o "$arg" == "mppub" ]; then
    export CAP_BASE=$FOLD/figs
    export CAP_REL=sboundary_test
    export CAP_STEM=$GEOM
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

