#!/bin/bash -l 

usage(){ cat << EOU
U4HitTest.sh
==============

::

   ~/o/u4/tests/U4HitTest.sh


Temporary fix for hit positioning, workstation::

    N[blyth@localhost J23_1_0_rc3_ok0]$ mkdir -p U4HitTest/ALL0/A000/
    N[blyth@localhost J23_1_0_rc3_ok0]$ cp jok-tds/ALL0/A000/* U4HitTest/ALL0/A000/

Temporary fix for hit positioning, laptop::

    epsilon:tests blyth$ GEOM tmp
    cd /data/blyth/opticks/GEOM/J23_1_0_rc3_ok0

    epsilon:J23_1_0_rc3_ok0 blyth$ pwd
    /data/blyth/opticks/GEOM/J23_1_0_rc3_ok0

    epsilon:J23_1_0_rc3_ok0 blyth$ mkdir -p U4HitTest/ALL0/A000/
    epsilon:J23_1_0_rc3_ok0 blyth$ cp jok-tds/ALL0/A000/* U4HitTest/ALL0/A000/



EOU
}


cd $(dirname $(realpath $BASH_SOURCE)) 

#export SOpticksResource_ExecutableName=G4CXSimulateTest
# THIS SETTING NO LONGER CHANGING SEvt PATH ? 

source $HOME/.opticks/GEOM/GEOM.sh 


name=U4HitTest 
script=$name.py 
msg="=== $BASH_SOURCE :"

path=$TMP/$name/$name.txt
mkdir -p $(dirname $path)


#defarg="run_ana"
defarg="run"
arg=${1:-$defarg}

export SEvt=info


vars="name path"

if [ "${arg/info}" != "$arg" ]; then 
   for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi 


if [ "${arg/run}" != "$arg" ]; then 
   $name
   [ $? -ne 0 ] && echo $msg run $name error && exit 1
fi 
if [ "${arg/cat}" != "$arg" ]; then 

   cmds="head tail"
   for cmd in $cmds 
   do 
       echo $cmd -4 $path
       eval $cmd -4 $path
   done 

   [ $? -ne 0 ] && echo $msg cat $name error && exit 1
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



