#!/bin/bash -l

cmdline="$*"

npy-
ggeo- 

dbg=0
if [ "${cmdline/--dbg}" != "${cmdline}" ]; then
   dbg=1
fi


cfg4-
case $dbg in
    0)  cfg4-run $* ;;
    1)  cfg4-dbg $* ;;
esac



