#!/bin/bash
usage(){ cat << EOU
OpticksPhotonTest.sh
======================

::

    ~/opticks/sysrap/tests/OpticksPhotonTest.sh

EOU
}

source $HOME/.opticks/GEOM/GEOM.sh

#test=GetHitMask
test=AbbrevToFlag
export TEST=${TEST:-$test}

OpticksPhotonTest


