#!/bin/bash -l
usage(){ cat << EOU
go.sh
=====

See also build.sh which doesnt use cmake 

EOU
}

arg=${1:-build_run}
sdir=$(pwd)
name=$(basename $sdir)

source ~/.opticks_config

export SHADER_FOLD=$sdir/rec_flying_point

#export ARRAY_FOLD=/tmp/$USER/opticks/GeoChain/BoxedSphere/CXRaindropTest
#export ARRAY_FOLD=/tmp/blyth/opticks/GeoChain/BoxedSphere/CXRaindropTest/SCVD0/70000
#export ARRAY_FOLD=/tmp/$USER/opticks/QSimTest/mock_propagate
#export ARRAY_FOLD=/tmp/blyth/opticks/U4RecorderTest/ShimG4OpAbsorption_FLOAT_ShimG4OpRayleigh_FLOAT/hama_body_log
#export ARRAY_FOLD=/tmp/blyth/opticks/GEOM/V1J011/ntds3/ALL1/p001
export ARRAY_FOLD=/tmp/sphoton_test


if [ "${arg/build}" != "$arg" ] ; then 

    opticks-
    oe-
    om-
    bdir=/tmp/$USER/opticks/$name/build 
    rm -rf $bdir && mkdir -p $bdir && cd $bdir && pwd 
    om-cmake $sdir
    make
    [ $? -ne 0 ] && exit 1

    make install   
fi 

if [ "${arg/run}" != "$arg" ] ; then 
    echo $BASH_SOURCE : run $name
    EYE=0,-3,0,1 $name 
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2
fi 

if [ "${arg/dbg}" != "$arg" ] ; then 
    echo $BASH_SOURCE : dbg $name
    dbg__ $name 
    [ $? -ne 0 ] && echo $BASH_SOURCE : dbg error && exit 3
fi

exit 0 

