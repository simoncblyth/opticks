#!/bin/bash -l 

msg="=== $BASH_SOURCE :"

#export QBnd=INFO

#test=fill_state_0
test=fill_state_1
#test=water

export TEST=${TEST:-$test}

QSimTest 

case $test in
   fill_state_*) script=QSimTest_fill_state.py ;;
              *) script=QSimTest_$test.py      ;;
esac

if [ -f "$script" ]; then
    echo $msg invoking analysis script $script
    ${IPYTHON:-ipython} --pdb -i $script
else
    echo $msg there is no analysis script $script
fi  

