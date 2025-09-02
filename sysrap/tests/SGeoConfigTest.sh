#!/bin/bash
usage(){ cat << EOU
SGeoConfigTest.sh
===================

::

   ~/o/sysrap/tests/SGeoConfigTest.sh

   EMM=0,1,2,3,4,5,6,7 ~/o/sysrap/tests/SGeoConfigTest.sh
   EMM=t0,1,2,3,4,5,6,7 ~/o/sysrap/tests/SGeoConfigTest.sh


Example ELV::

    sWorld,sTarget
    tsWorld,sTarget
    t:sWorld,sTarget
    ~sWorld,sTarget
    \~sWorld,sTarget


EOU
}


source $HOME/.opticks/GEOM/GEOM.sh
source $HOME/.opticks/GEOM/ELV.sh

export SBitSet__level=1
export SGeoConfig__ELVSelection_VERBOSE=1

#test=ELVString
test=ELVSelection

export TEST=$test

SGeoConfigTest


