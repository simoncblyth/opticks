#!/bin/bash -l 
usage(){ cat << EOU
ssys_test.sh
=============

::

    ~/opticks/sysrap/tests/ssys_test.sh 

EOU
}

name=ssys_test 
bin=${TMP:-/tmp/$USER/opticks}/$name 
mkdir -p $(dirname $bin)

defarg="build_run"
arg=${1:-$defarg}

cd $(dirname $BASH_SOURCE)

if [ "${arg/build}" != "$arg" ]; then 
    gcc $name.cc -g -std=c++11 -lstdc++ -I.. -o $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE compile error && exit 1 
fi 



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

if [ "${arg/run}" != "$arg" ]; then 
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2 
fi 

if [ "${arg/dbg}" != "$arg" ]; then 
   case $(uname) in 
     Darwin) lldb__ $bin ;;
     Linux)  gdb__ $bin ;;
   esac
   [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 3 
fi 


exit 0 

