#!/bin/bash -l 
usage(){ cat << EOU
nmskSolidMaskVirtual.sh
=========================

With the nudging, currently ON by default, see sprinkle along z=98 and nothing along z=0, and clear geometry changed "shoulder"::

    ./nmskSolidMaskVirtual.sh withnudge_ct_ana
    ./nmskSolidMaskVirtual.sh ana

Disabling nudging get very clear coincidence lines at z=97 and z=0::

    ./nmskSolidMaskVirtual.sh skipnudge_ct_ana
    ./nmskSolidMaskVirtual.sh ana

After running eg skipnudge_ct the ana appearance will stay the same 
until running eg withnudge_ct 

EOU
}

defarg=withnudge_ct_ana
#defarg=skipnudge_ct_ana

arg=${1:-$defarg}

export ZZ=0,98,291.1
export XX=0
export GEOM=nmskSolidMaskVirtual

loglevels(){
    export GeoChain=INFO
    export SArgs=INFO
    export ncylinder=INFO
}
loglevels


if [ "${arg/withnudge}" != "$arg" ]; then 
   gc
   ./translate.sh
   [ $? -ne 0 ] && echo $BASH_SOURCE withnudge translate error && exit 1 
fi 
if [ "${arg/skipnudge}" != "$arg" ]; then 
   gc
   OPTICKS_OPTS="--x4nudgeskip 0" ./translate.sh
   [ $? -ne 0 ] && echo $BASH_SOURCE skipnudge translate error && exit 1 
fi 
if [ "${arg/ct}" != "$arg" ]; then 
   c
   ./ct.sh run
   [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2 
fi 
if [ "${arg/ana}" != "$arg" ]; then 
   c
   ./ct.sh ana
   [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 3 
fi 
if [ "${arg/unx}" != "$arg" ]; then 
   c
   XDIST=500 NOLEGEND=1 UNEXPECTED=1 ./ct.sh ana
   [ $? -ne 0 ] && echo $BASH_SOURCE unx error && exit 3 
fi 

if [ "${arg/sample}" != "$arg" ]; then 
   c 
   ./CSGSimtraceSampleTest.sh 
fi


exit 0 

