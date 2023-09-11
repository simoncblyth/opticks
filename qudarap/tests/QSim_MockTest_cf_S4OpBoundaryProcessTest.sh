#!/bin/bash -l 

cd $(dirname $BASH_SOURCE) 
name=QSim_MockTest_cf_S4OpBoundaryProcessTest

defarg="info_ana"
arg=${1:-$defarg}

export AFOLD=/tmp/QSim_MockTest
export BFOLD=/tmp/S4OpBoundaryProcessTest

#check=SmearNormal_SigmaAlpha
check=SmearNormal_Polish
export CHECK=${CHECK:-$check}


vars="BASH_SOURCE name arg AFOLD BFOLD CHECK"

if [ "${arg/info}" != "$arg" ]; then 
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi

if [ "${arg/ana}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $name.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE : ana error && exit 1 
fi

exit 0 

