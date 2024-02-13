#!/bin/bash -l 

usage(){ cat << EOU
U4HitTest.sh
==============

::

   ~/o/u4/tests/U4HitTest.sh


Temporary fixup::

    N[blyth@localhost J23_1_0_rc3_ok0]$ mkdir -p U4HitTest/ALL0/A000/
    N[blyth@localhost J23_1_0_rc3_ok0]$ cp jok-tds/ALL0/A000/* U4HitTest/ALL0/A000/

EOU
}


cd $(dirname $(realpath $BASH_SOURCE)) 

#export SOpticksResource_ExecutableName=G4CXSimulateTest

source $HOME/.opticks/GEOM/GEOM.sh 

name=U4HitTest 
script=$name.py 
msg="=== $BASH_SOURCE :"

#defarg="run_ana"
defarg="run"
arg=${1:-$defarg}

export SEvt=info

if [ "${arg/run}" != "$arg" ]; then 
   $name
   [ $? -ne 0 ] && echo $msg run $name error && exit 1
fi 

if [ "${arg/dbg}" != "$arg" ]; then 
   dbg__ $name 
   [ $? -ne 0 ] && echo $msg dbg $name error && exit 2
fi 

if [ "${arg/ana}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $script
   [ $? -ne 0 ] && echo $msg ana error && exit 3
fi 

exit 0 




