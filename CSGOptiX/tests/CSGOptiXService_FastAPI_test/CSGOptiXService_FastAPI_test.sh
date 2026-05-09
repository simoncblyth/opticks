#!/usr/bin/env bash

usage(){ cat << EOU
CSGOptiXService_FastAPI_test.sh
================================

This requires the *uv* python package+venv tool::

    https://github.com/astral-sh/uv

Build and start the FastAPI HTTP server, on first run dependencies
are downloaded from pypi into the virtual env .venv directory::

    lo   ## full opticks environment - "lo_client" is not sufficient
    ~/opticks/CSGOptiX/tests/CSGOptiXService_FastAPI_test/CSGOptiXService_FastAPI_test.sh

Make HTTP POST requests to the endpoint::

     ~/np/tests/np_curl_test/np_curl_test.sh
     LEVEL=1 MULTIPART=0  ~/np/tests/np_curl_test/np_curl_test.sh
     LEVEL=1 MULTIPART=1  ~/np/tests/np_curl_test/np_curl_test.sh


CAUTION : the server consumes significant VRAM
-----------------------------------------------

Some opticks-t tests FAIL with VRAM OOM if they are run
while the server is running. Due to this have added check
before running tests for this does not run tests when
VRAM is low.


How to rebuild the server
---------------------------

::

    lo
    lco   ## need miniconda python env with nanobind
    cx
    om-cleaninstall


Caution, using lco means the server will see the miniconda libcurl
which may be different to the one used by the client build.



Check the .so of the python module
-----------------------------------

::

    (ok) [lo] A[blyth@localhost CSGOptiX]$ find $OPTICKS_PREFIX/py -name "opticks_CSGOptiX*.so" -exec ls -alst {} \;
    3136 -rwxr-xr-x. 1 blyth blyth 3207576 May  8 15:32 /data1/blyth/local/opticks_Debug/py/opticks_CSGOptiX.cpython-313-x86_64-linux-gnu.so

    (ok) [lo] A[blyth@localhost CSGOptiX]$ find $OPTICKS_PREFIX/build -name "opticks_CSGOptiX*.so" -exec ls -alst {} \;
    3136 -rwxr-xr-x. 1 blyth blyth 3207576 May  8 15:32 /data1/blyth/local/opticks_Debug/build/CSGOptiX/opticks_CSGOptiX.cpython-313-x86_64-linux-gnu.so



subcommands
------------

clean
   delete __pycache__ and .venv from source directory

info
   dump variables

check
   check fastapi is in environment

venv
   use uv to initialize the virtual environment and write .gitignore

activate
   source the venv setup

run
   start the server

pdb
   invoke check script under ipython

ldd
   check dependencies of the python module


EOU
}


cd $(dirname $(realpath $BASH_SOURCE))


name=CSGOptiXService_FastAPI_test
main_script=$(realpath main.py)
check_script=$(realpath check.py)

source $HOME/.opticks/GEOM/GEOM.sh

## BELOW setting done by oj_geomsetup of Examples/Tutorial/share/opticks_juno.sh
export OPTICKS_HIT_MASK=EC

if [ -n "$SAVE" ]; then
    ## SAVING IS APPROPRIATE ONLY DURING DEBUG/VALIDATION
    echo $BASH_SOURCE - SAVE - CONFIG ARRAY SAVING
    export OPTICKS_MODE_SAVE=SMS_RELATIVE
    export OPTICKS_EVENT_MODE=Hit
fi

tmp=/tmp/$USER/opticks
TMP=${TMP:-$tmp}
LOGDIR=$TMP/$name
mkdir -p $LOGDIR

WHICH_FASTAPI=$(which fastapi 2>/dev/null)

defarg="info_check_venv_activate_run"
arg=${1:-$defarg}

vv="BASH_SOURCE PWD defarg arg GEOM main_script check_script tmp TMP LOGDIR WHICH_FASTAPI SAVE"


if [[ "$arg" =~ clean ]]; then
   rm -rf __pycache__
   rm -rf .venv
fi

if [[ "$arg" =~ info ]]; then
    for v in $vv ; do printf "%30s : %s\n" "$v" "${!v}" ; done
fi

if [[ "$arg" =~ check ]]; then
    if command -v fastapi >/dev/null 2>&1; then
        echo "$BASH_SOURCE - check - fastapi CLI is installed at $(command -v fastapi)"
    else
        echo "$BASH_SOURCE - check - fastapi CLI not found - try miniconda python environment via bash function \"lco\" "
        exit 1
    fi
fi

if [[ "$arg" =~ venv ]]; then
    if [ ! -d ".venv" ]; then
        echo $BASH_SOURCE - installing dependencies
        echo .venv > .gitignore
        echo __pycache__ >> .gitignore
        uv venv
        uv pip install "fastapi[standard]" numpy ipython
    else
        echo $BASH_SOURCE - using existing .venv
    fi
fi

if [[ "$arg" =~ activate ]]; then
    if [ -f .venv/bin/activate ]; then
       source .venv/bin/activate
       [ $? -ne 0 ] && echo $BASH_SOURCE - failed to activate venv && exit 1
    else
       echo $BASH_SOURCE - no .venv - EXIT HERE && exit 0
    fi
fi

if [[ "$arg" =~ run ]]; then
    (
        cd $LOGDIR  || exit 1
        which fastapi
        fastapi dev $main_script
    )
    [ $? -ne 0 ] && echo $BASH_SOURCE - failed to fastapi dev && exit 2
fi

if [[ "$arg" =~ pdb ]]; then
    which ipython
    ${ipython:-ipython} -i --pdb $check_script
    [ $? -ne 0 ] && echo $bash_source - failed to pdb && exit 3
fi

if [[ "$arg" =~ ldd ]]; then
    lib=$(find $OPTICKS_PREFIX/py -name "opticks_CSGOptiX*.so")
    ls -alst $lib
    ldd $lib
fi






exit 0
