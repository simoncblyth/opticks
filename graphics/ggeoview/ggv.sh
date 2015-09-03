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

cd=0
if [ "${cmdline/--cd}" != "${cmdline}" ]; then
   cd=1
fi


make=0
if [ "${cmdline/--make}" != "${cmdline}" ]; then
   make=1
   cu=$(ggeoview-sdir)/cu/generate.cu 
   touch $cu
   ls -l $cu
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

if [ "${cmdline/--oac}" != "${cmdline}" ]; then
   export OPTIX_API_CAPTURE=1
fi

if [ "$make" == "1" ]; then
    ggeoview-install 
fi


if [ "$cd" == "1" ]; then 
    idp=$(ggeoview-run $* --idp)
    ls -l $idp 
    cd $idp 
elif [ "$cmp" == "0" ]; then 
    case $dbg in
       0)  ggeoview-run $*  ;;
       1)  ggeoview-dbg $*  ;;
    esac
else
    ggeoview-compute $*

fi

