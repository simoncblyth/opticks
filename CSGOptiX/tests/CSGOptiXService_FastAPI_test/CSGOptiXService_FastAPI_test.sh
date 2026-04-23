#!/usr/bin/env bash

usage(){ cat << EOU
CSGOptiXService_FastAPI_test.sh
================================

This requires the *uv* python package+venv tool::

    https://github.com/astral-sh/uv

Build and start the FastAPI HTTP server, on first run dependencies
are downloaded from pypi into the virtual env .venv directory::

    ~/opticks/CSGOptiX/tests/CSGOptiXService_FastAPI_test/CSGOptiXService_FastAPI_test.sh

Make HTTP POST requests to the endpoint::

     ~/np/tests/np_curl_test/np_curl_test.sh
     LEVEL=1 MULTIPART=0  ~/np/tests/np_curl_test/np_curl_test.sh
     LEVEL=1 MULTIPART=1  ~/np/tests/np_curl_test/np_curl_test.sh

EOU
}


cd $(dirname $(realpath $BASH_SOURCE))


name=CSGOptiXService_FastAPI_test
main_script=$(realpath main.py)
check_script=$(realpath check.py)

source $HOME/.opticks/GEOM/GEOM.sh


## TODO: MOVE THIS CONFIG INTO CODE ?
export OPTICKS_EVENT_MODE=Hit
export OPTICKS_HIT_MASK=EC


tmp=/tmp/$USER/opticks
TMP=${TMP:-$tmp}
LOGDIR=$TMP/$name
mkdir -p $LOGDIR


defarg="info_check_venv_run"
arg=${1:-$defarg}

if [ "${arg/clean}" != "$arg" ]; then
   rm -rf __pycache__
   rm -rf .venv
fi

if [ "${arg/info}" != "$arg" ]; then
    vv="BASH_SOURCE PWD defarg arg GEOM main_script check_script tmp TMP LOGDIR"
    for v in $vv ; do printf "%30s : %s\n" "$v" "${!v}" ; done
fi

if [ "${arg/check}" != "$arg" ]; then
    if command -v fastapi >/dev/null 2>&1; then
        echo "$BASH_SOURCE - fastapi CLI is installed at $(command -v fastapi)"
    else
        echo "$BASH_SOURCE - fastapi CLI not found - try base environment with ipython such as \"lo\" "
        exit 1
    fi
fi

if [ "${arg/venv}" != "$arg" ]; then
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

if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
   [ $? -ne 0 ] && echo $BASH_SOURCE - failed to activate venv && exit 1
else
   echo $BASH_SOURCE - no .venv - EXIT HERE && exit 0
fi

if [ "${arg/run}" != "$arg" ]; then
    (
        cd $LOGDIR  || exit 1
        which fastapi
        fastapi dev $main_script
    )
    [ $? -ne 0 ] && echo $BASH_SOURCE - failed to fastapi dev && exit 2
fi

if [ "${arg/pdb}" != "$arg" ]; then
    which ipython
    ${IPYTHON:-ipython} -i --pdb $check_script
    [ $? -ne 0 ] && echo $BASH_SOURCE - failed to pdb && exit 3
fi


exit 0
