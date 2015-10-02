#!/bin/bash -l

cmdline="$*"
ggeoview-

dbg=0
if [ "${cmdline/--dbg}" != "${cmdline}" ]; then
   dbg=1
fi

if [ "${cmdline/--oac}" != "${cmdline}" ]; then
   export OPTIX_API_CAPTURE=1
fi


if [ "${cmdline/--cmp}" != "${cmdline}" ]; then
   export GGEOVIEW_BINARY=$(ggeoview-compute-bin)
elif [ "${cmdline/--loader}" != "${cmdline}" ]; then
   export GGEOVIEW_BINARY=$(ggeoview-loader-bin)
else
   unset GGEOVIEW_BINARY 
fi



ggeoview_defaults(){

   export GGEOVIEW_GEOKEY=DAE_NAME_DYB
   export GGEOVIEW_QUERY="range:3153:12221"
   export GGEOVIEW_CTRL="volnames"
   #export GGEOVIEW_MESHFIX="iav,oav"
   export GGEOVIEW_MESHFIX="iav"

   #export GGEOVIEW_QUERY="range:3153:4814"     #  transition to 2 AD happens at 4814 
   #export GGEOVIEW_QUERY="range:3153:4813"     #  this range constitutes full single AD
   #export GGEOVIEW_QUERY="range:3161:4813"      #  push up the start to get rid of plain outer volumes, cutaway view: udp.py --eye 1.5,0,1.5 --look 0,0,0 --near 5000
   #export GGEOVIEW_QUERY="index:5000"
   #export GGEOVIEW_QUERY="index:3153,depth:25"
   #export GGEOVIEW_QUERY="range:5000:8000"

}



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

   ggeoview_defaults

elif [ "${cmdline/--idyb}" != "${cmdline}" ]; then

   export GGEOVIEW_GEOKEY=DAE_NAME_DYB
   #export GGEOVIEW_QUERY="range:3161:4813"      # this misses out the IAV 
   export GGEOVIEW_QUERY="range:3158:3160"       # just 2 volumes (python style range) __dd__Geometry__AD__lvLSO--pvIAV0xc2d0348, __dd__Geometry__AD__lvIAV--pvGDS0xbf6ab00 
   export GGEOVIEW_CTRL="volnames"

elif [ "${cmdline/--jdyb}" != "${cmdline}" ]; then

   export GGEOVIEW_GEOKEY=DAE_NAME_DYB
   export GGEOVIEW_QUERY="range:3158:3159"       # just 1 volume __dd__Geometry__AD__lvLSO--pvIAV0xc2d0348
   export GGEOVIEW_CTRL="volnames"
   
elif [ "${cmdline/--kdyb}" != "${cmdline}" ]; then

   export GGEOVIEW_GEOKEY=DAE_NAME_DYB
   export GGEOVIEW_QUERY="range:3159:3160"       # just 1 volume __dd__Geometry__AD__lvIAV--pvGDS0xbf6ab00 
   export GGEOVIEW_CTRL="volnames"

else

   ggeoview_defaults

fi









if [ "${cmdline/--make}" != "${cmdline}" ]; then
   optixrap-
   cu=$(optixrap-sdir)/cu/generate.cu 
   touch $cu
   ls -l $cu
   ggeoview-install 
fi


if [ "${cmdline/--idp}" != "${cmdline}" ]; then

    echo $(ggeoview-run $*)

elif [ "${cmdline/--assimp}" != "${cmdline}" ]; then

    assimpwrap-
    ggeoview-export
    $(assimpwrap-bin) GGEOVIEW_

else
    case $dbg in
       0)  ggeoview-run $*  ;;
       1)  ggeoview-dbg $*  ;;
    esac
fi

