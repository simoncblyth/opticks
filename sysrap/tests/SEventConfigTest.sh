#!/bin/bash -l
usage(){ cat << EOU
SEventConfigTest.sh
=====================

::

    ~/opticks/sysrap/tests/SEventConfigTest.sh

    OEM=DebugLite ~/opticks/sysrap/tests/SEventConfigTest.sh 
    OEM=DebugLite ~/opticks/sysrap/tests/SEventConfigTest.sh 



EOU
}

cd $(dirname $BASH_SOURCE)
name=SEventConfigTest

#oem=DebugLite
#oem=Default
oem=Minimal
export OPTICKS_EVENT_MODE=${OEM:-$oem}

case $OPTICKS_EVENT_MODE in 
   Default|Minimal|DebugLite|DebugHeavy|HitOnly|HitAndPhoton)   ok=1 ;; 
   *) ok=0 ;;  
esac

if [ $ok -eq  1 ]; then
    echo $BASH_SOURCE : OPTICKS_EVENT_MODE $OPTICKS_EVENT_MODE IS VALID
else
    echo $BASH_SOURCE : ERROR : OPTICKS_EVENT_MODE $OPTICKS_EVENT_MODE NOT VALID 
    exit 1  
fi 


omb=31
export OPTICKS_MAX_BOUNCE=${OMB:-$omb}



#export OPTICKS_OUT_FOLD=${TMP:-/tmp/$USER/opticks}/$name/out_fold
#export OPTICKS_OUT_NAME=organized/relative/dir/tree/out_name

export OPTICKS_INPUT_PHOTON=/some/path/to/name.npy 
export OPTICKS_INPUT_PHOTON_FRAME=Hama:0:1000
export FOLD=${TMP:-/tmp/$USER/opticks}/$name

defarg="info_run"
arg=${1:-$defarg}


vars="0 BASH_SOURCE name mode OPTICKS_EVENT_MODE OPTICKS_INPUT_PHOTON OPTICKS_INPUT_PHOTON_FRAME"

if [ "${arg/info}" != "$arg" ]; then 
   for var in $vars ; do printf "%25s : %s \n" "$var" "${!var}" ; done 
fi

if [ "${arg/run}" != "$arg" ]; then 
   $name
   [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1 
fi 

if [ "${arg/dbg}" != "$arg" ]; then 
   dbg__ $name
   [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 2 
fi 

if [ "${arg/ana}" != "$arg" ]; then 
   ${IPYTHON:-ipython} --pdb -i $name.py 
   [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 3 
fi 

exit 0 

