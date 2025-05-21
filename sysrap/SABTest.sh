#!/bin/bash
usage(){ cat << EOU
SABTest.sh : A-B comparison of persisted Opticks SEvt
========================================================

A:B comparison of SEvt and plotting is currently done by very 
many Opticks scripts. Aims of SABTest.sh:

1. replace most of the other A-B comparisons to decrease duplication
2. work from release, ie do not depend on having source tree

Started from ~/j/InputPhotonsCheck/InputPhotonsCheck.sh

EOU
}

vv="BASH_SOURCE"
source $HOME/.opticks/GEOM/GEOM.sh
vv="$vv GEOM"
source $HOME/.opticks/GEOM/EVT.sh    
vv="$vv AFOLD BFOLD"

anascript=SABTest.py

defarg="info_ls_pdb"
arg=${1:-$defarg}

if [ "${arg/info}" != "$arg" ]; then
    for v in $vv ; do printf "%30s :  %s\n" "$v" "${!v}" ; done
fi

if [ "${arg/ls}" != "$arg" ]; then
    ff="AFOLD BFOLD"
    for f in $ff ; do echo $f ${!f} && ls -alst ${!f} ; done
fi

if [ "${arg/pdb}" != "$arg" ]; then
   ${IPYTHON:-ipython} --pdb -i $(which $anascript)
   [ $? -ne 0 ] && echo $BASH_SOURCE pdb error
fi

if [ "${arg/ana}" != "$arg" ]; then
   ${PYTHON:-python} $(which $anascript)
   [ $? -ne 0 ] && echo $BASH_SOURCE ana error
fi

if [ "${arg/chi2}" != "$arg" ] ; then
   sseq_index_test.sh info_run_ana
fi

if [ "${arg/dhi2}" != "$arg" ] ; then
   # development version of "chi2"
   unset sseq_index_test__DEBUG
   #export sseq_index_test__DEBUG=1
   DEV=1 $HOME/opticks/sysrap/tests/sseq_index_test.sh info_build_run_ana
fi


