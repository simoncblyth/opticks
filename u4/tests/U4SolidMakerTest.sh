#!/bin/bash
usage(){ cat << EOU

~/o/u4/tests/U4SolidMakerTest.sh

WaterDistributer
   multiunion of multiple-multiunions is untranslatable to Opticks listnode
   and anyhow the multiunions contain sn::CutCylinder which has no
   intersection impl .. so need to devise some placeholder nodes
   which when detected force triangulation

AltWaterDistributer
   everything in one MultiUnion is simpler,
   but it takes forever to polgonize, so unworkable


To see this solid within container universe box
converted into CSGFoundry geometry use::

   ~/o/g4cx/tests/G4CX_U4TreeCreateCSGFoundryTest.sh


EOU
}

name=U4SolidMakerTest

cd $(dirname $(realpath $BASH_SOURCE))


#solid=WaterDistributer
#solid=AltWaterDistributer
#solid=WaterDistributorPartIIIUnion
#solid=OuterReflectorOrbSubtraction

#solid=R12860_PMTSolid
#solid=R12860_PMTSolidU   # Union with torus, instead of Subtraction, to see the position
solid=R12860_PMTSolidK    # KLUDGE : dont subtract torus to see the tube "scarf"





export SOLID=$solid
export FOLD=/tmp/$USER/opticks/$name
mkdir -p $FOLD


if [ "$SOLID" == "OuterReflectorOrbSubtraction" ]; then
   export U4SolidMaker__OuterReflectorOrbSubtraction_radOuterReflector=20722.1
fi

bin=$name
script=$name.py

vv="BASH_SOURCE name PWD bin script FOLD SOLID arg defarg"

defarg="info_run_pdb"
arg=${1:-$defarg}


if [ "${arg/info}" != "$arg" ]; then
    for v in $vv ; do printf "%30s : %s\n" "$v" "${!v}" ; done
fi

if [ "${arg/run}" != "$arg" ]; then\
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE - run error && exit 1
fi

if [ "${arg/dbg}" != "$arg" ]; then\
    source dbg__.sh
    dbg__ $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE - dbg error && exit 1
fi

if [ "${arg/ana}" != "$arg" ]; then\
    ${PYTHON:-python} $script
    [ $? -ne 0 ] && echo $BASH_SOURCE - ana error && exit 2
fi

if [ "${arg/pdb}" != "$arg" ]; then\
    ${IPYTHON:-ipython} -i --pdb $script
    [ $? -ne 0 ] && echo $BASH_SOURCE - pdb error && exit 3
fi

exit 0

