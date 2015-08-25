#!/bin/bash -l

cmdline="$*"
ggeoview-

cmp=0
if [ "${cmdline/--cmp}" != "${cmdline}" ]; then
   cmp=1
fi

dbg=0
if [ "${cmdline/--dbg}" != "${cmdline}" ]; then
   dbg=1
fi



if [ "${cmdline/--juno}" != "${cmdline}" ]; then
   export GGEOVIEW_DETECTOR=DAE_NAME_JUNO
elif [ "${cmdline/--jpmt}" != "${cmdline}" ]; then
   export GGEOVIEW_DETECTOR=DAE_NAME_JPMT
elif [ "${cmdline/--jtst}" != "${cmdline}" ]; then
   export GGEOVIEW_DETECTOR=DAE_NAME_JTST
elif [ "${cmdline/--dyb}" != "${cmdline}" ]; then
   export GGEOVIEW_DETECTOR=DAE_NAME_DYB
else
   export GGEOVIEW_DETECTOR=DAE_NAME_DYB
fi



if [ "$cmp" == "0" ]; then 
    case $dbg in
       0)  ggeoview-run $*  ;;
       1)  ggeoview-dbg $*  ;;
    esac
else
    ggeoview-compute $*


fi

