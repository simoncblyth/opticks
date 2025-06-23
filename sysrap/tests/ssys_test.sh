#!/bin/bash
usage(){ cat << EOU
ssys_test.sh
=============

::

    ~/opticks/sysrap/tests/ssys_test.sh

EOU
}
cd $(dirname $(realpath $BASH_SOURCE))

name=ssys_test
bin=${TMP:-/tmp/$USER/opticks}/$name
mkdir -p $(dirname $bin)

defarg="build_run"
arg=${1:-$defarg}



export MULTILINE=$(cat << EOV

red
green
blue
cyan
magenta
   yellow
   pink

   puce

    Pyrex/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/Vacuum
    Pyrex/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/Vacuum
    Pyrex/PMT_3inch_photocathode_logsurf2/PMT_3inch_photocathode_logsurf1/Vacuum
    Pyrex/PMT_20inch_veto_photocathode_logsurf2/PMT_20inch_veto_photocathode_logsurf1/Vacuum


EOV
)


#i=-1 u=2 f=101.3 d=-202.5 /tmp/$name

export GEOM=FewPMT
export ${GEOM}_GEOMList=hamaLogicalPMT
export VERSION=214
export COMMANDLINE="Some-COMMANDLINE-with-spaces after the spaces start"


export GEOM=J_2024aug27
export stree__force_triangulate_solid='filepath:$HOME/.opticks/GEOM/${GEOM}_meshname_stree__force_triangulate_solid.txt'


export ssys_test__getenviron_SIGINT=1



if [ "${arg/build}" != "$arg" ]; then
    echo [ $BASH_SOURCE build
    gcc $name.cc -g -std=c++11 -lstdc++ -I.. -o $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE compile error && exit 1
    echo ] $BASH_SOURCE build
fi

if [ "${arg/run}" != "$arg" ]; then
    echo [ $BASH_SOURCE run
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2
    echo ] $BASH_SOURCE run
fi

if [ "${arg/dbg}" != "$arg" ]; then
   source dbg__.sh
   dbg__ $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 3
fi


exit 0

