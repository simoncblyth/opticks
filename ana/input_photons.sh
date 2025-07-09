#!/bin/bash
usage(){ cat << EOU
input_photons.sh
===================

Running this script creates several input photon .npy arrays
into the directory ~/.opticks/InputPhotons::

   ~/o/ana/input_photons.sh   # create input photon arrays (non interactive)
   ~/o/ana/iinput_photons.sh  # create input photon arrays (interactive)

Note that the python script runs twice with DTYPE envvar
as np.float32 and np.float64 in order to create the arrays
in both single and double precision.

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))
script=input_photons.py

defarg="info_numpy_run_ls"
[ "$OPT" == "-i" ] && defarg="info_numpy_dbg_ls"

arg=${1:-$defarg}

dtypes="np.float32 np.float64"
vars="PWD script defarg arg dtypes"

if [ "${arg/info}" != "$arg" ]; then
   for var in $vars ; do printf "%30s : %s \n" "$var" "${!var}" ; done
fi

if [ "${arg/numpy}" != "$arg" ]; then
    python -c "import numpy as np" 2>/dev/null
    [ $? -ne 0 ] && echo $BASH_SOURCE numpy package is not within your python - cannot generate input photons && exit 2
fi

if [ "${arg/dbg}" != "$arg" ]; then
    for dtype in $dtypes ; do
        DTYPE=$dtype ${IPYTHON:-ipython} --pdb $OPT $script -- $*
        [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 1
    done
fi


if [ "${arg/run}" != "$arg" ]; then
    for dtype in $dtypes ; do
        DTYPE=$dtype python $script
        [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2
    done
fi

if [ "${arg/ls}" != "$arg" ]; then
   echo [ ls -alst ~/.opticks/InputPhotons
   ls -alst ~/.opticks/InputPhotons
   echo ] ls -alst ~/.opticks/InputPhotons
fi


if [ "${arg/tar}" != "$arg" ]; then

   cd ~/.opticks
   if [ ! -d InputPhotons ]; then
      echo $BASH_SOURCE tar error - create InputPhotons first && exit 3
   fi

   TAR=InputPhotons.tar

   if [ ! -f "$TAR" ]; then
      tar cvf $TAR InputPhotons/
   else
      echo $BASH_SOURCE TAR $TAR exists already
   fi
   ls -alst $TAR
   du -h $TAR
fi




