#!/bin/bash -l 

bin=CSGMakerTest

logging(){
   export CSGFoundry=INFO
}
logging


#geom=JustOrb
#geom=BoxedSphere
#export CSGMakerTest_GEOM=$geom


arg=${1:-run}

if [ "$arg" == "run" ]; then 
   $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1 
fi 

if [ "$arg" == "dbg" ]; then 

   case $(uname) in 
      Darwin) lldb__ $bin ;;
      Linux)  gdb__ $bin ;;
   esac 
   [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 2 
fi 

exit 0 









