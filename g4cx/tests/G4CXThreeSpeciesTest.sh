#!/bin/bash
usage(){ cat << EOU
G4CXThreeSpeciesTest.sh : standalone JUNO three-species LS scintillation test
==============================================================================

Setup the opticks environment with a converted geometry, then::

    ~/o/g4cx/tests/G4CXThreeSpeciesTest.sh
    ~/o/g4cx/tests/G4CXThreeSpeciesTest.sh info
    ~/o/g4cx/tests/G4CXThreeSpeciesTest.sh dbg

    OPTICKS_INTEGRATION_MODE=2 ~/o/g4cx/tests/G4CXThreeSpeciesTest.sh    ## ECPU only

Settings
---------

OPTICKS_USE_THREE_SPECIES_LS_MODEL=1
    switches the GPU kernel to the JUNO new LS optical model

U4Physics__ConstructOp_UseScintThree=1
    U4Physics uses JUNO new LS optical model for scintillation

U4Physics__ConstructOp_Cerenkov_DISABLE=1
    isolate the LS scintillation, so that all recorded photons come from
    the three-species model

Integration modes
------------------

    OPTICKS_INTEGRATION_MODE=2 : ECPU only, photons into B
    OPTICKS_INTEGRATION_MODE=3 : ECPU + EGPU, EGPU into A and ECPU into B

Output goes to::

    \$TMP/GEOM/\$GEOM/G4CXThreeSpeciesTest/ALL.._../(B000 = ECPU, A000 = EGPU)

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

name=G4CXThreeSpeciesTest
bin=$name

defarg="info_run"
arg=${1:-$defarg}

if [ -z "$GEOM" ]; then
    echo "$BASH_SOURCE : FATAL GEOM MUST BE SET : source the opticks env first" >&2
    exit 1
fi

geomvar=${GEOM}_CFBaseFromGEOM
if [ -z "${!geomvar}" ]; then
    echo "$BASH_SOURCE : FATAL $geomvar MUST BE SET : convert the geometry first" >&2
    exit 1
fi

export OPTICKS_USE_THREE_SPECIES_LS_MODEL=1
export U4Physics__ConstructOp_UseScintThree=1
export U4Physics__ConstructOp_Cerenkov_DISABLE=1

export OPTICKS_NUM_EVENT=${OPTICKS_NUM_EVENT:-10}
export OPTICKS_INTEGRATION_MODE=${OPTICKS_INTEGRATION_MODE:-3}
export OPTICKS_EVENT_MODE=${OPTICKS_EVENT_MODE:-DebugLite}
export OPTICKS_MAX_SLOT=${OPTICKS_MAX_SLOT:-M1}

export G4CXOpticks__SaveGeometry_DIR=$TMP/GEOM/$GEOM/$name/${name}_geom

vars="BASH_SOURCE GEOM $geomvar TMP OPTICKS_USE_THREE_SPECIES_LS_MODEL U4Physics__ConstructOp_UseScintThree U4Physics__ConstructOp_Cerenkov_DISABLE OPTICKS_NUM_EVENT OPTICKS_INTEGRATION_MODE OPTICKS_EVENT_MODE OPTICKS_MAX_SLOT G4CXOpticks__SaveGeometry_DIR"

if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%45s : %s \n" "$var" "${!var}" ; done
fi

if [ "${arg/run}" != "$arg" ]; then
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 1
fi

if [ "${arg/dbg}" != "$arg" ]; then
    source dbg__.sh
    dbg__ $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : dbg error && exit 2
fi

if [ "${arg/usage}" != "$arg" ]; then
    usage
fi

exit 0
