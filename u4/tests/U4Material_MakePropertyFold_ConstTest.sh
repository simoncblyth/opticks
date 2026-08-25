#!/bin/bash
usage(){ cat << EOU

~/o/u4/tests/U4Material_MakePropertyFold_ConstTest.sh

EOU
}

name=U4Material_MakePropertyFold_ConstTest

cd $(dirname $(realpath $BASH_SOURCE))

defarg=info_run_pdb
arg=${1:-$defarg}

bin=$name
script=$name.py

tmp=/tmp/$USER/opticks
TMP=${TMP:-$tmp}

export FOLD=$TMP/U4Material_MakePropertyFold_ConstTest
## FOLD is used from the python script by "Fold.Load(symbol='f')"


vv="BASH_SOURCE name PWD defarg arg bin script tmp TMP FOLD"

if [[ "$arg" =~ info ]]; then
   for v in $vv ; do printf "%30s : %s\n" "$v" "${!v}" ; done
fi

if [[ "$arg" =~ run ]]; then
   $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE - ERROR from run && exit 1
fi

if [[ "$arg" =~ pdb ]]; then
   ${IPYTHON:-ipython} -i --pdb $script
   [ $? -ne 0 ] && echo $BASH_SOURCE - ERROR from pdb && exit 2
fi

if [[ "$arg" =~ ana ]]; then
   ${PYTHON:-python} $script
   [ $? -ne 0 ] && echo $BASH_SOURCE - ERROR from ana && exit 3
fi

exit 0

