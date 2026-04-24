#!/bin/bash
usage(){ cat << EOU
SClientSimulator_test.sh
=========================

HMM: cannot here depend on Geant4 - so simply load gensteps into SEvt
and then access them and use them with NP_CURL.h


Build and start the endpoint "server" on GPU node::

   lo  ## using full opticks env (NOT lo_client)
   ~/opticks/CSGOptiX/tests/CSGOptiXService_FastAPI_test/CSGOptiXService_FastAPI_test.sh


Run this test, not necessarily on GPU node::

    lo  ## OR lo_client
    ~/o/sysrap/tests/SClientSimulator_test.sh


NEXT: Add non-standalone SClientSimulatorTest.cc
that can be used from Client build

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

name=SClientSimulator_test

tmp=/tmp/$USER/opticks
TMP=${TMP:-$tmp}
BINFOLD=$TMP/$name
mkdir -p $BINFOLD
bin=$BINFOLD/$name
vv="BASH_SOURCE PWD name tmp TMP BINFOLD bin"

defarg="info_chk_gcc_run"
arg=${1:-$defarg}
vv="$vv defarg arg"

LIBCURL_VERSION=$(curl-config --version)

source $HOME/.opticks/GEOM/GEOM.sh

export FOLD=$TMP/SEvt__createInputGenstep_configuredTest

vv="$vv LIBCURL_VERSION GEOM ${GEOM}_CFBaseFromGEOM tmp TMP FOLD"

if [ "${arg/info}" != "${arg}" ]; then
    for v in $vv ; do printf "%30s : %s\n" "$v" "${!v}" ; done
fi

if [ "${arg/chk}" != "$arg" ]; then

    checkfor="8.12.1"

    if [ -n "$VERBOSE" ]; then
        which curl-config
        curl-config --libs
        curl-config --cflags
        curl-config --vernum
        curl-config --version
    fi

    if curl-config --checkfor "$checkfor" >/dev/null 2>&1; then
        echo "$BASH_SOURCE - OK - $(curl-config --version) meets requirements."
    else
        echo "$BASH_SOURCE - FAIL - Need curl-config with checkfor and version of at least $checkfor, but have $(curl-config --version)."
        echo "$BASH_SOURCE - TRY FIRST ACTIVATING YOUR CONDA ENV TO USE ITS LIBCURL eg with \"lo_client\" OR \"lo\" "
        exit 1
    fi
fi




if [ "${arg/inc}" != "${arg}" ]; then
    echo $BASH_SOURCE - inc - dump headers with curl.h in name actually used by compiler
    gcc -std=c++17 -I.. -DWITH_SEVT_MOCK $(curl-config --cflags) -E -H SClientSimulator_test.cc 2>&1 | grep curl.h
    # -E : just preprocessing
    # -H : print header paths
fi

if [ "${arg/gcc}" != "${arg}" ]; then


    gcc $name.cc ../s_pa.cc ../s_tv.cc ../s_bb.cc ../sn.cc ../s_csg.cc \
        -o $bin \
        -std=c++17 -lstdc++ -lm -g \
        -I.. \
        -DWITH_CHILD \
        -DWITH_SEVT_MOCK \
        $(curl-config --cflags) \
        $(curl-config --libs)

    [ $? -ne 0 ] && echo $BASH_SOURCE gcc error && exit 1
fi

if [ "${arg/run}" != "${arg}" ]; then
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2
fi

if [[ "$arg" =~ dbg ]]; then
    source dbg__.sh
    dbg__ $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 3
fi





exit 0

