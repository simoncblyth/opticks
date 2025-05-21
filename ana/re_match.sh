#!/bin/bash 
usage(){ cat << EOU


PTN='Water.*LatticedShellSteel' ./re_match.sh 

EOU
}

source $HOME/.opticks/GEOM/GEOM.sh

defarg="info_pdb"
arg=${1:-$defarg}

vv="BASH_SOURCE GEOM script"

script=re_match.py 

if [ "${arg/info}" != "$arg" ]; then
   for v in $vv ; do printf "%30s : %s\n" "$v" "${!v}" ; done
fi 

if [ "${arg/pdb}" != "$arg" ]; then
   ${IPYTHON:-ipython} --pdb -i $script
fi



