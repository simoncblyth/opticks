#!/bin/bash -l
usage(){ cat << EOU
U4VolumeMakerTest.sh
=====================

u4t::

   ./U4VolumeMakerTest.sh


Running without the "-l" on shebang line fails for lack of boost::

    epsilon:tests blyth$ ./U4VolumeMakerTest.sh
    dyld: Library not loaded: libboost_system.dylib
      Referenced from: /usr/local/opticks/lib/U4VolumeMakerTest
      Reason: image not found
    ./U4VolumeMakerTest.sh: line 20: 51758 Abort trap: 6           U4VolumeMakerTest

EOU
}

geom=BoxOfScintillator  # default 
export GEOM=${GEOM:-$geom}

vars="BASH_SOURCE GEOM"
for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 


U4VolumeMakerTest 

