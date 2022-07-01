#!/bin/bash -l
usage(){ cat << EOU
U4VolumeMakerTest.sh
=====================

u4t::

   ./U4VolumeMakerTest.sh

HMM when the GEOM has manager prefix hama/nnvt/hmsk/nmsk
U4VolumeMaker::Make currently assumes the rest of the name
if of an LV. 

Running without the "-l" on shebang line fails for lack of boost::

    epsilon:tests blyth$ ./U4VolumeMakerTest.sh
    dyld: Library not loaded: libboost_system.dylib
      Referenced from: /usr/local/opticks/lib/U4VolumeMakerTest
      Reason: image not found
    ./U4VolumeMakerTest.sh: line 20: 51758 Abort trap: 6           U4VolumeMakerTest

EOU
}

arg=${1:-run}


#geom=BoxOfScintillator  # default 
geom=hama_body_log

export GEOM=${GEOM:-$geom}

vars="BASH_SOURCE arg GEOM"
for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 

if [ "${arg/run}" != "${arg}" ]; then 
    U4VolumeMakerTest 
    [ $? -ne 0 ] && echo $msg run error && exit 1 
fi 

if [ "${arg/dbg}" != "${arg}" ]; then 
    case $(uname) in 
       Darwin) lldb__ U4VolumeMakerTest ;;
       Linux)  gdb U4VolumeMakerTest ;;
    esac  
    [ $? -ne 0 ] && echo $msg dbg error && exit 2 
fi 

exit 0 
