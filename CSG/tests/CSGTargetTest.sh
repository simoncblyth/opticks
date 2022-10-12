#!/bin/bash -l 

moi=Hama:0:1000
moi=NNVT:0:1000


export MOI=$moi
export METHOD=getFrame

loglevel()
{
    export CSGTarget=INFO
}
loglevel


bin=CSGTargetTest

defarg=run
arg=${1:-$defarg}


if [ "$arg" == "run" ]; then 
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1 
fi 

if [ "$arg" == "dbg" ]; then 
    case $(uname) in 
       Darwin) lldb__ $bin ;;
       Linux)   gdb__ $bin ;;
    esac
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 2 
fi 

if [ "$arg" == "ana" ]; then 
    ${IPYTHON:-ipython} --pdb -i $bin.py
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 3 
fi 



exit 0 


