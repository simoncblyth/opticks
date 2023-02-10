#!/bin/bash -l 


cfabdir=$(dirname $BASH_SOURCE)

#a_cfbase=/Users/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/41c046fe05b28cb70b1fc65d0e6b7749/1/CSG_GGeo
#b_cfbase=/tmp/blyth/opticks/J000/G4CXSimtraceTest

a_cfbase=$HOME/.opticks/GEOM/J007
b_cfbase=/tmp/$USER/opticks/CSGImportTest


export A_CFBASE=${A_CFBASE:-$a_cfbase}
export B_CFBASE=${B_CFBASE:-$b_cfbase}

${IPYTHON:-ipython} --pdb -i $cfabdir/CSGFoundryAB.py 


