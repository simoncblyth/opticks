#!/bin/bash -l 
usage(){ cat << EOU
sysrap/tests/gaussQTables.sh
============================

Related scripts:

sysrap/tests/gaussQTables.sh 
   get familiar with Geant4 gaussTable 

sysrap/tests/S4MTRandGaussQTest.sh
   using the table to implement S4MTRandGaussQTest::transformQuick 

sysrap/tests/erfcinvf_Test.sh 
   getting good match to S4MTRandGaussQTest::transformQuick on device with erfcinvf 

EOU
}



CDAT=/usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/global/HEPRandom/src/gaussQTables.cdat
export CDAT 

SDIR=$(cd $(dirname $BASH_SOURCE) && pwd )
cd $SDIR

name=gaussQTables

defarg="info_ana"
arg=${1:-$defarg}

vars="name arg CDAT SDIR"


if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/ana}" != "$arg" ]; then
    ${IPYTHON:-ipython} -i --pdb $name.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE : ana error && exit 1 
fi 

exit 0 

