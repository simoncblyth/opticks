#!/bin/bash -l 


SDIR=$(cd $(dirname $BASH_SOURCE) && pwd)
cd $SDIR

name=S4OpBoundaryProcessTest

clhep-
g4-

defarg="info_build_run_ana"
arg=${1:-$defarg}


BASE=/tmp/$name
bin=$BASE/$name

check=smear_normal_sigma_alpha
#check=smear_normal_polish
export CHECK=${CHECK:-$check}

export FOLD=$BASE/$CHECK
mkdir -p $FOLD

#num=1
#num=1000
num=100000
export NUM=${NUM:-$num}

opt=-DMOCK_CUDA_DEBUG


vars="BASH_SOURCE name SDIR BASE FOLD CHECK bin NUM opt CHECK"

if [ "${arg/info}" != "$arg" ]; then 
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/build}" != "$arg" ]; then 
    gcc $name.cc \
       -I.. \
        $opt \
       -I$(clhep-prefix)/include \
       -I$(g4-prefix)/include/Geant4  \
       -L$(clhep-prefix)/lib \
       -lCLHEP \
       -std=c++11 -lstdc++ \
       -o $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : build error && exit 1 
fi

if [ "${arg/run}" != "$arg" ]; then 
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : build error && exit 2
fi

if [ "${arg/ana}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $name.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE : ana error && exit 3
fi

exit 0 


