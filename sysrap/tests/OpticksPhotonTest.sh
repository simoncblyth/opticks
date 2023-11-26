#!/bin/bash -l 
usage(){ cat << EOU
OpticksPhotonTest.sh
======================

::

    ~/opticks/sysrap/tests/OpticksPhotonTest.sh

EOU
}

source $HOME/.opticks/GEOM/GEOM.sh 
OpticksPhotonTest 


