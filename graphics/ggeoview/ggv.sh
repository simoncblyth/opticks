#!/bin/bash -l

cmdline="$*"
ggeoview-

if [ "${cmdline/--juno}" != "${cmdline}" ]; then
   export GGEOVIEW_DETECTOR=DAE_NAME_JUNO
elif [ "${cmdline/--dyb}" != "${cmdline}" ]; then
   export GGEOVIEW_DETECTOR=DAE_NAME_DYB
else
   export GGEOVIEW_DETECTOR=DAE_NAME_DYB
fi

ggeoview-run $*


