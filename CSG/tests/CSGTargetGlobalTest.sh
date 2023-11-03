#!/bin/bash -l 


bin=CSGTargetGlobalTest

source $HOME/.opticks/GEOM/GEOM.sh 

defarg="info_run"
arg=${1:-$defarg}

logging(){
   export CSGTarget=INFO
}
logging



vars="BASH_SOURCE bin GEOM" 

if [ "${arg/info}" != "$arg" ]; then
   for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/run}" != "$arg" ]; then
   $bin 
   [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 1 
fi 

if [ "${arg/dbg}" != "$arg" ]; then
   case $(uname) in 
     Darwin) lldb__ $bin ;;
     Linux)  gdb__ $bin ;; 
   esac 
   [ $? -ne 0 ] && echo $BASH_SOURCE : dbg error && exit 2
fi 


exit 0 


#moi=solidXJfixture:64
#export MOI=${MOI:-$moi}
#
#${IPYTHON:-ipython} -i CSGTargetGlobalTest.py 


