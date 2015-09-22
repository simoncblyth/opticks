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
   optixrap-
   cu=$(optixrap-sdir)/cu/generate.cu 
   touch $cu
   ls -l $cu
fi



if [ "${cmdline/--juno}" != "${cmdline}" ]; then

   export GGEOVIEW_GEOKEY=DAE_NAME_JUNO
   export GGEOVIEW_QUERY="range:1:50000"
   export GGEOVIEW_CTRL=""

elif [ "${cmdline/--jpmt}" != "${cmdline}" ]; then

   export GGEOVIEW_GEOKEY=DAE_NAME_JPMT
   export GGEOVIEW_QUERY="range:1:289734"  # 289733+1 all test3.dae volumes
   export GGEOVIEW_CTRL=""

elif [ "${cmdline/--jtst}" != "${cmdline}" ]; then

   export GGEOVIEW_GEOKEY=DAE_NAME_JTST
   export GGEOVIEW_QUERY="range:1:50000" 
   export GGEOVIEW_CTRL=""

elif [ "${cmdline/--dyb}" != "${cmdline}" ]; then

   export GGEOVIEW_GEOKEY=DAE_NAME_DYB
   export GGEOVIEW_QUERY="range:3153:12221"
   export GGEOVIEW_CTRL="volnames"
   #export GGEOVIEW_QUERY="range:3153:4814"     #  transition to 2 AD happens at 4814 
   #export GGEOVIEW_QUERY="range:3153:4813"     #  this range constitutes full single AD
   #export GGEOVIEW_QUERY="range:3161:4813"      #  push up the start to get rid of plain outer volumes, cutaway view: udp.py --eye 1.5,0,1.5 --look 0,0,0 --near 5000
   #export GGEOVIEW_QUERY="index:5000"
   #export GGEOVIEW_QUERY="index:3153,depth:25"
   #export GGEOVIEW_QUERY="range:5000:8000"

elif [ "${cmdline/--idyb}" != "${cmdline}" ]; then

   export GGEOVIEW_GEOKEY=DAE_NAME_DYB
   export GGEOVIEW_QUERY="range:3161:4813"      # this misses out the IAV 
   export GGEOVIEW_CTRL=""

else

   export GGEOVIEW_GEOKEY=DAE_NAME_DYB
   export GGEOVIEW_QUERY="range:3153:12221"
   export GGEOVIEW_CTRL="volnames"

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

