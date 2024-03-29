#!/bin/bash -l 
usage(){ cat << EOU
~/o/examples/Geant4/ScintGenStandalone/DsG4ScintillationTest.sh
================================================================

::

    BP=G4Track::CalculateVelocityForOpticalPhoton  ~/o/examples/Geant4/ScintGenStandalone/DsG4ScintillationTest.sh
    BP=G4Track::CalculateVelocity  ~/o/examples/Geant4/ScintGenStandalone/DsG4ScintillationTest.sh

    ~/o/examples/Geant4/ScintGenStandalone/DsG4ScintillationTest.sh ana

    In [9]: f.p.shape
    Out[9]: (964, 4, 4)

    In [2]: f.gs.shape
    Out[2]: (4, 6, 4)

    In [8]: f.gs[:,5,0]                ## four gs differ in bottom left float  
    Out[8]: array([  4.6,  15.1,  76.1, 397. ], dtype=float32)

    In [4]: f.gs.view(np.int32)[:,0]    ## and top right int (photon count)
    Out[4]:
    array([[  5,   0,   0, 656],
           [  5,   0,   0, 206],
           [  5,   0,   0,  73],
           [  5,   0,   0,  29]], dtype=int32)

    In [5]: f.gs.view(np.int32)[:,0,3]
    Out[5]: array([656, 206,  73,  29], dtype=int32)

    In [6]: f.gs.view(np.int32)[:,0,3].sum()
    Out[6]: 964



EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

msg="=== $BASH_SOURCE :"

srcs=(
    DsG4ScintillationTest.cc
    DsG4Scintillation.cc
     ../CerenkovStandalone/OpticksUtil.cc
     ../CerenkovStandalone/OpticksRandom.cc
   )

name=${srcs[0]}
name=${name/.cc}

for src in $srcs ; do echo $msg $src ; done

source $HOME/.opticks/GEOM/GEOM.sh
export FOLD=/tmp/$USER/opticks/DsG4ScintillationTest
mkdir -p $FOLD
LOGDIR=/tmp/$name
bin=$LOGDIR/$name
script=$name.py

vars="BASH_SOURCE FOLD LOGDIR bin script"

for var in $vars ; do printf "%20s : %s\n" "$var" "${!var}" ; done



export DsG4Scintillation_verboseLevel=0   # 0/1/2
export SEvt=INFO
export U4Material=INFO



g4-
clhep-
#boost-
cuda-   # just for vector_functions.h

standalone-compile(){
    local name=$1
    name=${name/.cc}
    mkdir -p $LOGDIR

    local opt="-DSTANDALONE "

    cat << EOC
    gcc \
        $* \
         -std=c++11 \
       -I. \
       -I../CerenkovStandalone \
       -I../../../sysrap \
       -I../../../u4 \
       -I../../../extg4 \
       -I../../../qudarap \
       $opt \
       -g \
       -I$(cuda-prefix)/include  \
       -I$(opticks-prefix)/externals/plog/include \
       -I$(opticks-prefix)/externals/glm/glm \
       -I$(g4-prefix)/include/Geant4 \
       -I$(clhep-prefix)/include \
       -L$(g4-prefix)/lib \
       -L$(clhep-prefix)/lib \
       -L$(opticks-prefix)/lib \
       -lstdc++ \
       -lSysRap \
       -lU4 \
       -lG4global \
       -lG4materials \
       -lG4particles \
       -lG4track \
       -lG4tracking \
       -lG4processes \
       -lG4geometry \
       -lG4intercoms \
       -lCLHEP \
       -o $bin
EOC
}

arg=${1:-build_run_ana}

[ -n "$BP" ] && echo $BASH_SOURCE : as BP defined override arg to dbg && arg=dbg


if [ "${arg/build}" != "$arg" ]; then
    standalone-compile ${srcs[@]}
    eval $(standalone-compile ${srcs[@]})
    [ $? -ne 0 ] && echo $msg compile error && exit 1
fi



iwd=$PWD


cd $LOGDIR

if [ "${arg/dbg}" != "$arg" ]; then
    bp=DsG4Scintillation::PostStepDoIt
    BP=${BP:-$bp} dbg__ $bin
    [ $? -ne 0 ] && echo $msg dbg error && exit 2
fi

if [ "${arg/run}" != "$arg" ]; then
    $bin
    [ $? -ne 0 ] && echo $msg run error && exit 2
fi

cd $iwd

if [ "${arg/ana}" != "$arg" ]; then
    echo $msg ana script $script FOLD $FOLD

    if [ -f "$script" ]; then
        ${IPYTHON:-ipython} --pdb -i $script
        [ $? -ne 0 ] && echo $msg ana error && exit 3
        echo $msg ana script $script FOLD $FOLD
    else
        echo $msg ERROR script $script does not exist && exit 4
    fi
fi

exit 0




