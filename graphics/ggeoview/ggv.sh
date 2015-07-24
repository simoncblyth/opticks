#!/bin/bash -l

cmdline="$*"
ggeoview-

cmp=0
if [ "${cmdline/--cmp}" != "${cmdline}" ]; then
   cmp=1
fi


if [ "${cmdline/--juno}" != "${cmdline}" ]; then
   export GGEOVIEW_DETECTOR=DAE_NAME_JUNO
elif [ "${cmdline/--dyb}" != "${cmdline}" ]; then
   export GGEOVIEW_DETECTOR=DAE_NAME_DYB
else
   export GGEOVIEW_DETECTOR=DAE_NAME_DYB
fi


if [ "$cmp" == "0" ]; then 
    ggeoview-run $*
else
    ggeoview-compute $*
fi

