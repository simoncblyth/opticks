#!/bin/bash -l 
usage(){ cat << EOU

~/o/examples/Geant4/OpticalApp/OpticalAppTest.sh 

~/o/examples/Geant4/OpticalApp/OpticalAppTest.sh ana


BP=G4ParticleChange::DumpInfo ~/o/examples/Geant4/OpticalApp/OpticalAppTest.sh

BP=G4OpBoundaryProcess::PostStepDoIt ~/o/examples/Geant4/OpticalApp/OpticalAppTest.sh

REC=3 ~/o/examples/UseGeometryShader/run.sh 



EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

name=OpticalAppTest
export FOLD=/tmp/$name
mkdir -p $FOLD

bin=$FOLD/$name
#script=$name.py 
script=${name}_ok.py 

vars="BASH_SOURCE name bin FOLD"


#export OpticalApp__GeneratePrimaries_DEBUG_GENIDX=50000
#export OpticalApp__PreUserTrackingAction_UseGivenVelocity_KLUDGE=1 


# -Wno-deprecated-copy \

defarg=info_build_run_ana

[ -n "$BP" ] && defarg=dbg 

arg=${1:-$defarg}

if [ "${arg/info}" != "$arg" ]; then
   for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi

if [ "${arg/build}" != "$arg" ]; then
    gcc $name.cc \
            -I. \
            -g \
            $(geant4-config --cflags) \
            $(geant4-config --libs) \
             -lstdc++ \
            -o $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : build error && exit 1 
fi

if [ "${arg/run}" != "$arg" ]; then
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2
fi

if [ "${arg/dbg}" != "$arg" ]; then
    dbg__ $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : dbg error && exit 3
fi

if [ "${arg/ana}" != "$arg" ]; then
    ${IPYTHON:-ipython} --pdb -i $script 
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2
fi

exit 0 

