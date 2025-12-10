#!/bin/bash
usage(){ cat << EOU
U4Mesh_test_py.sh
===================

::

    oje
    SOLID=Waterdistributor_1 ~/o/u4/tests/U4Mesh_test_py.sh

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))
script=U4Mesh_test.py


External_CFBaseFromGEOM=${GEOM}_CFBaseFromGEOM
if [ -n "$GEOM" -a -n "${!External_CFBaseFromGEOM}" -a -d "${!External_CFBaseFromGEOM}" -a -f "${!External_CFBaseFromGEOM}/CSGFoundry/prim.npy" ]; then
    echo $BASH_SOURCE - External GEOM setup detected
    vv="External_CFBaseFromGEOM ${External_CFBaseFromGEOM}"
    for v in $vv ; do printf "%41s : %s \n" "$v" "${!v}" ; done
    export CFBase=${!External_CFBaseFromGEOM}
else
    echo $BASH_SOURCE - NOT EXTERNAL GEOM
    source ~/.opticks/GEOM/GEOM.sh  ## sets GEOM envvar, use GEOM bash function to setup/edit

    if [ -n "$GEOM" -a -n "${GEOM}_CFBaseFromGEOM" ]; then
        _CFBase=${GEOM}_CFBaseFromGEOM
        export CFBase=${!_CFBase}
    fi
fi

export FOLD=$CFBase/CSGFoundry/SSim/stree/mesh

solid=Waterdistributor_0
export SOLID=${SOLID:-$solid}





${IPYTHON:-ipython} --pdb -i $script



