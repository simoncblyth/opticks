#!/bin/bash -l 

cd $(dirname $BASH_SOURCE) 
name=QSim_MockTest_cf_S4OpBoundaryProcessTest

defarg="info_ana"
arg=${1:-$defarg}


check=smear_normal_sigma_alpha
#check=smear_normal_polish
export CHECK=${CHECK:-$check}


export AFOLD=/tmp/QSim_MockTest/$CHECK
export BFOLD=/tmp/S4OpBoundaryProcessTest/$CHECK
export CFOLD=/tmp/QSimTest/$CHECK


vars="BASH_SOURCE name arg CHECK AFOLD BFOLD CFOLD"

if [ "${arg/info}" != "$arg" ]; then 
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi

if [ "${arg/ana}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $name.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE : ana error && exit 1 
fi

exit 0 

