#!/bin/bash -l 


SDIR=$(cd $(dirname $BASH_SOURCE) && pwd)
cd $SDIR

name=S4OpBoundaryProcessTest

clhep-
g4-

defarg="info_build_run_ana"
arg=${1:-$defarg}

export FOLD=/tmp/$name
mkdir -p $FOLD
bin=$FOLD/$name

export OTHER=/tmp/QSim_MockTest

#num=1000
num=100000
export NUM=${NUM:-$num}

vars="BASH_SOURCE name SDIR FOLD bin NUM"

if [ "${arg/info}" != "$arg" ]; then 
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/build}" != "$arg" ]; then 
    gcc $name.cc \
       -I.. \
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



