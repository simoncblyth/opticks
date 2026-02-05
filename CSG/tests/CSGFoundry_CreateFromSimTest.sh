#!/bin/bash
usage(){ cat << EOU
CSGFoundry_CreateFromSimTest.sh testing CSGFoundry::CreateFromSim
=====================================================================

This is a technical test that are able to copy one CSGFoundry
to another, rather than a comparison of two routes.
This is because when using U4TreeCreate start from gdml
and create SSim that creates the CSGFoundry.
So will need to go upwards to gxt/G4CXOpticks_setGeometry_Test.sh
for a more complete test.

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))
bin=CSGFoundry_CreateFromSimTest

script=CSGFoundryAB.py



srcd=$HOME/.opticks/GEOM

External_CFBaseFromGEOM=${GEOM}_CFBaseFromGEOM
if [ -z "$NOXGEOM" -a -n "$GEOM" -a -n "${!External_CFBaseFromGEOM}" -a -d "${!External_CFBaseFromGEOM}" -a -f "${!External_CFBaseFromGEOM}/CSGFoundry/prim.npy" ]; then
    GEOM_METHOD="External GEOM setup : use NOXGEOM=1 to disable externally configured GEOM"
else
    source $srcd/GEOM.sh  ## sets GEOM envvar, use GEOM bash function to setup/edit
    GEOM_METHOD="local sourcing of ~/.opticks/GEOM/GEOM.sh"
fi

CFB=${GEOM}_CFBaseFromGEOM  ## cannot use External_CFBaseFromGEOM because GEOM not defined up there
export A_CFBASE=${!CFB}

export FOLD=/tmp/$USER/opticks/$bin
mkdir -p $FOLD

# env for CSGFoundryAB comparison
export B_CFBASE=$FOLD

lvid=119
ndid=0
export LVID=${LVID:-$lvid}
export NDID=${NDID:-$ndid}
#export scsg_level=1

vv="GEOM_METHOD GEOM NOXGEOM External_CFBaseFromGEOM ${External_CFBaseFromGEOM} A_CFBASE B_CFBASE"

vv="$vv BASH_SOURCE bin GEOM BASE FOLD check script LVID NDID scsg_level"

loglevel(){
   export CSGFoundry=INFO
   #export CSGImport=INFO
}
loglevel


defarg=info_run_ana
arg=${1:-$defarg}



if [ "${arg/info}" != "$arg" ]; then
    for v in $vv ; do printf "%20s : %s \n" "$v" "${!v}" ; done
fi

if [ "${arg/run}" != "$arg" ]; then
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1
fi

if [ "${arg/dbg}" != "$arg" ]; then
    dbg__ $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 2
fi

if [ "${arg/pdb}" != "$arg" ]; then
   ${IPYTHON:-ipython} --pdb -i $script
   [ $? -ne 0 ] && echo $BASH_SOURCE : pdb error && exit 1
fi

if [ "${arg/ana}" != "$arg" ]; then
   ${PYTHON:-python} $script
   [ $? -ne 0 ] && echo $BASH_SOURCE : ana error && exit 1
fi

exit 0

