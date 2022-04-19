#!/bin/bash -l 
source $PWD/../bin/GEOM.sh trim   ## sets GEOM envvar based on GEOM.txt file 
msg="=== $BASH_SOURCE :"
arg=${1:-run_ana}

usage(){ cat << EOU
cxsim.sh : CSGOptiXSimulateTest combining CFBASE_LOCAL simple test geometry with standard CFBASE basis geometry  
=================================================================================================================


EOU
}

if [ "$(uname)" == "Linux" ]; then
    cfname_local=GeoChain/$GEOM    
else
    cfname_local=GeoChain_Darwin/$GEOM    
fi
export CFBASE_LOCAL=/tmp/$USER/opticks/$cfname_local
unset GEOM   # MUST unset GEOM to get CSGFoundry::Load_ to load the OPTICKS_KEY basis geometry 

logdir=/tmp/$USER/opticks/CSGOptiXSimulateTest 
mkdir -p $logdir
cd $logdir

CSGOptiXSimulateTest 
[ $? -ne 0 ] && echo $msg RUN ERROR && exit 1 

echo $msg logdir $logdir 

exit 0 
