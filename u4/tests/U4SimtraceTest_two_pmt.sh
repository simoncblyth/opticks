#!/bin/bash -l 

arg=${1:-ana}

sym=A
SYM=${SYM:-$sym}

case $SYM in 
  A) N=0 ;;
  B) N=1 ;;
esac

if [ "$SYM" == "A" ]; then 

  pid=813

elif [ "$SYM" == "B" ]; then 

  #pid=748
  pid=892

fi 


#opt=ast,nrm
opt=idx,nrm



export TOPLINE="N=$N ${SYM}PID=$pid ${SYM}OPT=$opt BGC=yellow FOCUS=0,0,255 ./U4SimtraceTest.sh"
TOPLINE="$TOPLINE $arg"
eval $TOPLINE


