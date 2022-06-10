#!/bin/bash -l 
usage(){ cat << EOU
U4RecorderTest.sh 
====================

::

    cd ~/opticks/u4/tests
    ./U4RecorderTest.sh 


EOU
}
msg="=== $BASH_SOURCE :"

arg=${1:-run_ana}

srcdir=$PWD
logdir=/tmp/$USER/opticks/U4RecorderTest
mkdir -p $logdir 

export FOLD=/tmp/$USER/opticks/U4RecorderTest
#export DsG4Scintillation_verboseLevel=3
#export G4Cerenkov_verboseLevel=3
export DsG4Scintillation_opticksMode=3  # 3:0b11 collect gensteps and do Geant4 generation loop too 

#export G4Cerenkov_DISABLE=1
#export DsG4Scintillation_DISABLE=1

#pidx=0
#export PIDX=${PIDX:-$pidx}

#gidx=117
#export GIDX=${GIDX:-$gidx}


if [ "${arg/run}" != "${arg}" ]; then 
    cd $logdir 
    #export BP=DsG4Scintillation::PostStepDoIt
    #lldb__ U4RecorderTest
    U4RecorderTest
    echo $msg logdir $logdir
fi 

if [ "${arg/dbg}" != "${arg}" ]; then 
    cd $logdir 
    #export BP=DsG4Scintillation::PostStepDoIt
    lldb__ U4RecorderTest
    echo $msg logdir $logdir
fi 

if [ "${arg/clean}" != "${arg}" ]; then
   cd $FOLD 
   pwd
   ls -l *.npy *.txt *.log
   read -p "$msg Enter YES to delete these : " ans
   if [ "$ans" == "YES" ] ; then 
       echo $msg proceeding
       rm *.npy *.txt *.log
   else
       echo $msg skip 
   fi 
fi 

if [ "${arg/ana}" != "${arg}" ]; then 
    cd $srcdir 
    pwd

    ${IPYTHON:-ipython} --pdb -i U4RecorderTest.py 
fi 


