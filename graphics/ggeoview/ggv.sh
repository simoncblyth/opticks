#!/bin/bash -l
cmdline="$*"

dump(){
  local IFS="$1" ; shift  
  local elements
  read -ra elements <<< "$*" 
  local elem 
  for elem in "${elements[@]}"; do
      >&2 printf "   %s\n" $elem
  done 
}


>&2 echo $0 dumping cmdline arguments

for arg in $cmdline 
do
   if [ "${arg/=}" == "${arg}" ]; then  
       >&2 printf "%s\n" $arg
   else
       dump _ $arg
   fi
done

#exit


npy-
ggeo- 
ggeoview-
assimpwrap-
openmeshrap-
cfg4-
opticks-
optixrap-


dbg=0
if [ "${cmdline/--dbg}" != "${cmdline}" ]; then
   dbg=1
fi

# cfg4- doesnt handle visualization of loaded NumpyEvt so pass to ggv-
load=0
if [ "${cmdline/--load}" != "${cmdline}" ]; then
   load=1
fi


if [ "${cmdline/--oac}" != "${cmdline}" ]; then
   export OPTIX_API_CAPTURE=1
fi

export OPTICKS_ARGS="$*"

if [ "${cmdline/--cmp}" != "${cmdline}" ]; then
   export OPTICKS_BINARY=$(ggeoview-bindir)/computeTest
elif [ "${cmdline/--boundaries}" != "${cmdline}" ]; then
   export OPTICKS_BINARY=$(ggeo-bin BoundariesNPYTest)
elif [ "${cmdline/--cfg4}" != "${cmdline}" ]; then

   if [ "$load" == "0" ]; then
       cfg4-export
       export OPTICKS_BINARY=$(cfg4-bin)
   fi
elif [ "${cmdline/--cproplib}" != "${cmdline}" ]; then
   cfg4-export
   export OPTICKS_BINARY=$(cfg4-tbin CPropLibTest)

elif [ "${cmdline/--cdetector}" != "${cmdline}" ]; then
   cfg4-export
   export OPTICKS_BINARY=$(cfg4-tbin CDetectorTest)

elif [ "${cmdline/--recs}" != "${cmdline}" ]; then
   export OPTICKS_BINARY=$(ggeo-bin RecordsNPYTest)
elif [ "${cmdline/--tracer}" != "${cmdline}" ]; then
   export OPTICKS_BINARY=$(ggeoview-bindir)/OTracerTest
elif [ "${cmdline/--lookup}" != "${cmdline}" ]; then
   export OPTICKS_BINARY=$(ggeo-bin LookupTest)
elif [ "${cmdline/--bnd}" != "${cmdline}" ]; then
   export OPTICKS_BINARY=$(ggeo-bin GBndLibTest)
elif [ "${cmdline/--itemlist}" != "${cmdline}" ]; then
   export OPTICKS_BINARY=$(ggeo-bin GItemListTest)

elif [ "${cmdline/--gsource}" != "${cmdline}" ]; then
   export OPTICKS_BINARY=$(ggeo-bin GSourceTest)
elif [ "${cmdline/--gsrclib}" != "${cmdline}" ]; then
   export OPTICKS_BINARY=$(ggeo-bin GSourceLibTest)

elif [ "${cmdline/--resource}" != "${cmdline}" ]; then
   export OPTICKS_BINARY=$(opticks-bin OpticksResourceTest)
elif [ "${cmdline/--opticks}" != "${cmdline}" ]; then
   export OPTICKS_BINARY=$(opticks-bin OpticksTest)

elif [ "${cmdline/--pybnd}" != "${cmdline}" ]; then
   export OPTICKS_BINARY=$(ggeo-tdir)/GBndLibTest.py
elif [ "${cmdline/--mat}" != "${cmdline}" ]; then
   export OPTICKS_BINARY=$(ggeo-bin GMaterialLibTest)
elif [ "${cmdline/--mm}" != "${cmdline}" ]; then
   export OPTICKS_BINARY=$(ggeo-bin GMergedMeshTest)
elif [ "${cmdline/--testbox}" != "${cmdline}" ]; then
   export OPTICKS_BINARY=$(ggeo-bin GTestBoxTest)
elif [ "${cmdline/--geolib}" != "${cmdline}" ]; then
   export OPTICKS_BINARY=$(ggeo-bin GGeoLibTest)
elif [ "${cmdline/--geotest}" != "${cmdline}" ]; then
   export OPTICKS_BINARY=$(ggeo-bin GGeoTestTest)
elif [ "${cmdline/--gmaker}" != "${cmdline}" ]; then
   export OPTICKS_BINARY=$(ggeo-bin GMakerTest)
elif [ "${cmdline/--pmt}" != "${cmdline}" ]; then
   export OPTICKS_BINARY=$(ggeo-bin GPmtTest)
   export OPTICKS_ARGS=${cmdline/--pmt}
elif [ "${cmdline/--attr}" != "${cmdline}" ]; then
   export OPTICKS_BINARY=$(ggeo-bin GAttrSeqTest)
elif [ "${cmdline/--surf}" != "${cmdline}" ]; then
   export OPTICKS_BINARY=$(ggeo-bin GSurfaceLibTest)
   export OPTICKS_ARGS=${cmdline/--surf}
elif [ "${cmdline/--tscint}" != "${cmdline}" ]; then
   export OPTICKS_BINARY=$(ggeo-bin GScintillatorLibTest)
   export OPTICKS_ARGS=${cmdline/--scint}
elif [ "${cmdline/--oscint}" != "${cmdline}" ]; then
   export OPTICKS_BINARY=$(optixrap-bin OScintillatorLibTest)
   export OPTICKS_ARGS=${cmdline/--oscint}

elif [ "${cmdline/--flags}" != "${cmdline}" ]; then
   export OPTICKS_BINARY=$(ggeo-bin GFlagsTest)
elif [ "${cmdline/--gbuffer}" != "${cmdline}" ]; then
   export OPTICKS_BINARY=$(ggeo-bin GBufferTest)
elif [ "${cmdline/--meta}" != "${cmdline}" ]; then
   export OPTICKS_BINARY=$(ggeo-bin GBoundaryLibMetadataTest)
elif [ "${cmdline/--sensor}" != "${cmdline}" ]; then
   export OPTICKS_BINARY=$(ggeo-bin GSensorListTest)
elif [ "${cmdline/--ggeo}" != "${cmdline}" ]; then
   export OPTICKS_BINARY=$(ggeo-bin GGeoTest)
elif [ "${cmdline/--assimp}" != "${cmdline}" ]; then
   export OPTICKS_BINARY=$(assimpwrap-bin)
elif [ "${cmdline/--openmesh}" != "${cmdline}" ]; then
   export OPTICKS_BINARY=$(openmeshrap-bin) 
elif [ "${cmdline/--torchstep}" != "${cmdline}" ]; then
   export OPTICKS_BINARY=$(ggeo-bin TorchStepNPYTest)
elif [ "${cmdline/--hits}" != "${cmdline}" ]; then
   export OPTICKS_BINARY=$(npy-bin HitsNPYTest)
else
   unset OPTICKS_BINARY 
fi


#echo OPTICKS_BINARY $OPTICKS_BINARY



ggeoview_defaults_dyb(){

   export OPTICKS_GEOKEY=DAE_NAME_DYB
   export OPTICKS_QUERY="range:3153:12221"
   export OPTICKS_CTRL="volnames"
   export OPTICKS_MESHFIX="iav,oav"
   export OPTICKS_MESHFIX_CFG="100,100,10,-0.999"   # face barycenter xyz alignment and dot face normal cuts for faces to be removed 

   #export OPTICKS_QUERY="range:3153:4814"     #  transition to 2 AD happens at 4814 
   #export OPTICKS_QUERY="range:3153:4813"     #  this range constitutes full single AD
   #export OPTICKS_QUERY="range:3161:4813"      #  push up the start to get rid of plain outer volumes, cutaway view: udp.py --eye 1.5,0,1.5 --look 0,0,0 --near 5000
   #export OPTICKS_QUERY="index:5000"
   #export OPTICKS_QUERY="index:3153,depth:25"
   #export OPTICKS_QUERY="range:5000:8000"
}


if [ "${cmdline/--juno}" != "${cmdline}" ]; then

   export OPTICKS_GEOKEY=DAE_NAME_JUNO
   export OPTICKS_QUERY="range:1:50000"
   export OPTICKS_CTRL=""

elif [ "${cmdline/--jpmt}" != "${cmdline}" ]; then

   export OPTICKS_GEOKEY=DAE_NAME_JPMT
   export OPTICKS_QUERY="range:1:289734"  # 289733+1 all test3.dae volumes
   export OPTICKS_CTRL=""

elif [ "${cmdline/--jtst}" != "${cmdline}" ]; then

   export OPTICKS_GEOKEY=DAE_NAME_JTST
   export OPTICKS_QUERY="range:1:50000" 
   export OPTICKS_CTRL=""

elif [ "${cmdline/--dpib}" != "${cmdline}" ]; then

   export OPTICKS_GEOKEY=DAE_NAME_DPIB
   export OPTICKS_QUERY="" 
   export OPTICKS_CTRL=""

elif [ "${cmdline/--dpmt}" != "${cmdline}" ]; then

   export OPTICKS_GEOKEY=DAE_NAME_DPIB
   export OPTICKS_QUERY="range:1:6"   # exclude the box at first slot   
   export OPTICKS_CTRL=""

elif [ "${cmdline/--dyb}" != "${cmdline}" ]; then

   ggeoview_defaults_dyb

elif [ "${cmdline/--idyb}" != "${cmdline}" ]; then

   ggeoview_defaults_dyb
   export OPTICKS_QUERY="range:3158:3160"       # 2 volumes : pvIAV and pvGDS

elif [ "${cmdline/--jdyb}" != "${cmdline}" ]; then

   ggeoview_defaults_dyb
   export OPTICKS_QUERY="range:3158:3159"       # 1 volume : pvIAV
   
elif [ "${cmdline/--kdyb}" != "${cmdline}" ]; then

   ggeoview_defaults_dyb
   export OPTICKS_QUERY="range:3159:3160"       # 1 volume : pvGDS

elif [ "${cmdline/--ldyb}" != "${cmdline}" ]; then

   ggeoview_defaults_dyb
   export OPTICKS_QUERY="range:3156:3157"       # 1 volume : pvOAV

elif [ "${cmdline/--mdyb}" != "${cmdline}" ]; then

   ggeoview_defaults_dyb
   export OPTICKS_QUERY="range:3201:3202,range:3153:3154"   # 2 volumes : first pmt-hemi-cathode and ADE  

   #  range:3154:3155  SST  Stainless Steel/IWSWater not a good choice for an envelope, just get BULK_ABSORB without going anywhere

else

   ggeoview_defaults_dyb

fi




if [ "${cmdline/--make}" != "${cmdline}" ]; then
   optixrap-
   cu=$(optixrap-sdir)/cu/generate.cu 
   touch $cu
   ls -l $cu
   ggeoview-install 
fi


# TODO: make binary for this to avoid the exception
if [ "${cmdline/--idp}" != "${cmdline}" ]; then
    echo $(ggeoview-run ${OPTICKS_ARGS})
else
    case $dbg in
       0)  ggeoview-run ${OPTICKS_ARGS}  ;;
       1)  ggeoview-dbg ${OPTICKS_ARGS}  ;;
    esac
fi

