#!/bin/bash
usage(){ cat << EOU
SClientSimulatorTest.sh
=========================

See also the standalone version of this SClientSimulator_test.sh

Build and start the endpoint "server" on GPU node::

   lo  ## using full opticks env (NOT lo_client)
   ~/opticks/CSGOptiX/tests/CSGOptiXService_FastAPI_test/CSGOptiXService_FastAPI_test.sh

Run this test, not necessarily on GPU node::

    lo_client  ## NB "Client" config is required for the libcurl version needed by NP_CURL.h
    oid        ## check are within "Client" config
    ~/o/sysrap/tests/SClientSimulatorTest.sh


EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

name=SClientSimulatorTest
bin=$name

vv="BASH_SOURCE PWD name tmp TMP FOLD bin"

defarg="info_run"
arg=${1:-$defarg}
vv="$vv defarg arg"

source $HOME/.opticks/GEOM/GEOM.sh

tmp=/tmp/$USER/opticks
export TMP=${TMP:-$tmp}
export FOLD=$TMP/SEvt__createInputGenstep_configuredTest

export OPTICKS_INPUT_GENSTEP=$FOLD/gs.npy

vv="$vv GEOM ${GEOM}_CFBaseFromGEOM tmp TMP FOLD OPTICKS_INPUT_GENSTEP"

if [[ $arg =~ info ]]; then
    for v in $vv ; do printf "%30s : %s\n" "$v" "${!v}" ; done
fi

if [[ $arg =~ run ]]; then
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2
fi

if [[ $arg =~ dbg ]]; then
    source dbg__.sh
    dbg__ $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2
fi

exit 0
