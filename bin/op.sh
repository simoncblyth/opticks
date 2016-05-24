#!/bin/bash -l
[ "$0" == "$BASH_SOURCE" ] && sauce=0 || sauce=1
if [ "$sauce" == "1" ]; then 
   op-(){   . $BASH_SOURCE ; } 
   op-vi(){ vi $BASH_SOURCE ; } 
fi

cmdline="$*"

op-usage(){ cat << EOU
op : Opticks Operations
===========================

This script pre-parses arguments at bash level, determining:

* binary to launch 
* geometry envvars to define

op.sh is intended to replace ggv.sh using 
simplifications possible following the
move to the superbuild approach.

TODO
-----

* consider moving geometry envvar config into C++ OpticksResource(?)
  as users tend to trip over envvars


EOU
}

op-binary-name-default(){ echo GGeoView ; }
op-binary-name()
{
   case $1 in 
           --cfg4) echo CG4Test ;;
         --tracer) echo OTracerTest ;;
            --mat) echo GMaterialLibTest ;;
           --surf) echo GSurfaceLibTest ;;
            --bnd) echo GBndLibTest ;;
  --ctestdetector) echo CTestDetectorTest ;;
  --cgdmldetector) echo CGDMLDetectorTest ;;
       --cproplib) echo CPropLibTest ;;
     --ngunconfig) echo NGunConfigTest ;;


     --boundaries) echo BoundariesNPYTest ;;
           --recs) echo RecordsNPYTest ;;
         --lookup) echo LookupTest ;;
       --itemlist) echo GItemListTest ;;
        --gsource) echo GSourceTest ;;
        --gsrclib) echo GSourceLibTest ;;
       --resource) echo OpticksResourceTest ;;
        --opticks) echo OpticksTest ;;
          --pybnd) echo GBndLibTest.py ;;
             --mm) echo GMergedMeshTest ;;
        --testbox) echo GTestBoxTest ;;
         --geolib) echo GGeoLibTest ;;
        --geotest) echo GGeoTestTest ;;
         --gmaker) echo GMakerTest ;;
            --pmt) echo GPmtTest ;;
           --attr) echo GAttrSeqTest ;;
         --tscint) echo GScintillatorLibTest ;;
         --oscint) echo OScintillatorLibTest ;;
          --flags) echo GFlagsTest ;;
        --gbuffer) echo GBufferTest ;;
           --meta) echo GBoundaryLibMetadataTest ;;
         --sensor) echo GSensorListTest ;;
           --ggeo) echo GGeoTest ;;
         --assimp) echo AssimpRapTest ;;
       --openmesh) echo OpenMeshRapTest ;;
      --torchstep) echo TorchStepNPYTest ;;  
           --hits) echo HitsNPYTest ;;  
   esac 
   # no default as its important this return blank for unidentified commands
   #      *) echo $(op-binary-name-default) ;;
      
}

op-binary-desc()
{
   case $1 in 
           --cfg4) echo "Geant4 comparison simulation of simple test geometries. Requires g4-export environment. " ;; 
         --tracer) echo "Fast OpenGL viz and OptiX tracing, NO propagation. From ggeoview-/tests. Used for simple geometry/machinery checking"  ;;
            --mat) echo "Dump properties of material identified by 0-based index , eg op --mat 0 " ;;
           --surf) echo "Dump properties of surface identified by 0-based index , eg op --surf 0 " ;;
            --bnd) echo "Dump boundaries of a geometry, eg op --bnd --jpmt " ;; 
  --ctestdetector) echo "Test Geant4 simple detector construction using class cfg4-/CTestDetector " ;; 
  --cgdmldetector) echo "Test Geant4 GDML full detector construction using cfg4-/CGDMLDetector " ;; 
       --cproblib) echo "Test Opticks/Geant4 material property library/converter cfg4-/CPropLib " ;; 
     --ngunconfig) echo "Test Geant4 Gun configuration with npy-/NGunConfigTest ";;
   esac 
}

op-geometry-name()
{
   case $1 in 
       --dyb)  echo DYB ;; 
       --idyb) echo IDYB ;; 
       --jdyb) echo JDYB ;; 
       --kdyb) echo KDYB ;; 
       --ldyb) echo LDYB ;; 
       --mdyb) echo MDYB ;; 
       --juno) echo JUNO ;; 
       --jpmt) echo JPMT ;; 
       --jtst) echo JTST ;; 
       --dpib) echo DPIB ;; 
       --dpmt) echo DPMT ;; 
   esac
}

op-geometry-setup()
{
    local geo=${OPTICKS_GEO:-DYB}
    op-geometry-unset 
    case $geo in 
       DYB|IDYB|JDYB|KDYB|LDYB|MDYB) op-geometry-setup-dyb  $geo  ;;
                     JUNO|JPMT|JTST) op-geometry-setup-juno $geo  ;;
                          DPIB|DPMT) op-geometry-setup-dpib $geo  ;;
    esac
}

op-geometry-query-dyb()
{
    case $1 in 
        DYB)  echo "range:3153:12221"  ;;
       IDYB)  echo "range:3158:3160" ;;  # 2 volumes : pvIAV and pvGDS
       JDYB)  echo "range:3158:3159" ;;  # 1 volume : pvIAV
       KDYB)  echo "range:3159:3160" ;;  # 1 volume : pvGDS
       LDYB)  echo "range:3156:3157" ;;  # 1 volume : pvOAV
       MDYB)  echo "range:3201:3202,range:3153:3154"  ;;  # 2 volumes : first pmt-hemi-cathode and ADE  
    esac
    # range:3154:3155  SST  Stainless Steel/IWSWater not a good choice for an envelope, just get BULK_ABSORB without going anywhere
}

op-geometry-setup-dyb()
{
    local geo=${1:-DYB}
    export OPTICKS_GEOKEY=DAE_NAME_DYB
    export OPTICKS_QUERY=$(op-geometry-query-dyb $geo) 
    export OPTICKS_CTRL="volnames"
    export OPTICKS_MESHFIX="iav,oav"
    export OPTICKS_MESHFIX_CFG="100,100,10,-0.999"   # face barycenter xyz alignment and dot face normal cuts for faces to be removed 
}
op-geometry-setup-juno()
{
   local geo=${1:-JPMT}
   if [ "$geo" == "JUNO" ]; then 
       export OPTICKS_GEOKEY=DAE_NAME_JUNO
       export OPTICKS_QUERY="range:1:50000"
       export OPTICKS_CTRL=""
   elif [ "$geo" == "JPMT" ]; then
       export OPTICKS_GEOKEY=DAE_NAME_JPMT
       export OPTICKS_QUERY="range:1:289734"  # 289733+1 all test3.dae volumes
       export OPTICKS_CTRL=""
   elif [ "$geo" == "JTST" ]; then
       export OPTICKS_GEOKEY=DAE_NAME_JTST
       export OPTICKS_QUERY="range:1:50000" 
       export OPTICKS_CTRL=""
   fi
}
op-geometry-setup-dpib()
{
   local geo=${1:-DPIB}
   if [ "$geo" == "DPIB" ]; then
       export OPTICKS_GEOKEY=DAE_NAME_DPIB
       export OPTICKS_QUERY="" 
       export OPTICKS_CTRL=""
    elif [ "$geo" == "DPMT" ]; then
       export OPTICKS_GEOKEY=DAE_NAME_DPIB
       export OPTICKS_QUERY="range:1:6"   # exclude the box at first slot   
       export OPTICKS_CTRL=""
   fi 
}
op-geometry-unset()
{
    unset OPTICKS_GEOKEY
    unset OPTICKS_QUERY 
    unset OPTICKS_CTRL
    unset OPTICKS_MESHFIX
    unset OPTICKS_MESHFIX_CFG
}


op-binary-names(){ type op-binary-name | perl -ne 'm,--(\w*)\), && print "$1\n" ' - ; } 
op-help(){
   local cmd
   local bin
   local hlp
   op-binary-names | while read cmd ; do
      bin=$(op-binary-name "--$cmd")
      desc=$(op-binary-desc "--$cmd")
      hlp=$(printf "%s (%s)" "$desc" $bin)  
      printf " %10s : %s \n" $cmd "$hlp"
   done
}


op-cmdline-dump()
{
    >&2 echo $0 $FUNCNAME
    local arg
    for arg in $cmdline 
    do
       if [ "${arg/=}" == "${arg}" ]; then  
           >&2 printf "%s\n" $arg
       else
           op-dump _ $arg
       fi
    done
}
op-dump(){
  local IFS="$1" ; shift  
  local elements
  read -ra elements <<< "$*" 
  local elem 
  for elem in "${elements[@]}"; do
      >&2 printf "   %s\n" $elem
  done 
}


op-cmdline-specials()
{
   unset OPTICKS_DBG 
   unset OPTICKS_LOAD
   unset OPTIX_API_CAPTURE

   if [ "${cmdline/--dbg}" != "${cmdline}" ]; then
       export OPTICKS_DBG=1
   fi
   if [ "${cmdline/--load}" != "${cmdline}" ]; then
       export OPTICKS_LOAD=1
   fi
   if [ "${cmdline/--oac}" != "${cmdline}" ]; then
       export OPTIX_API_CAPTURE=1
   fi
}

op-cmdline-binary-match()
{
    local msg="=== $FUNCNAME : finds 1st argument with associated binary :"
    local arg
    local bin
    unset OPTICKS_CMD

    for arg in $cmdline 
    do
       bin=$(op-binary-name $arg)
       #echo arg $arg bin $bin geo $geo 
       if [ "$bin" != "" ]; then 
           export OPTICKS_CMD=$arg
           return 
       fi
    done
}


op-binary-setup()
{
    local msg="=== $FUNCNAME :"
    local cfm=$OPTICKS_CMD
    local bin=$(op-binary-name $cfm) 
    local def=$(op-binary-name-default)

    if [ "$bin" == "" ]; then
       bin=$def
    fi 

    if [ "$OPTICKS_LOAD" == "1" ]; then 
        echo $msg OPTICKS_LOAD overrides binary from $bin to default $def as cfg4 doesnt handle visualization of loaded NumpyEvt 
        bin=$def
    fi 

    #echo $msg cfm $cfm bin $bin def $def
    unset OPTICKS_BINARY 
    unset OPTICKS_ARGS

    if [ "$bin" != "" ]; then
       export OPTICKS_BINARY=$(opticks-bindir)/$bin
       # some commands should not be removed from the commandline
       # as they are needed by the binary 
       case $cfm in 
         --surf|--scint|--oscint|--pmt) export OPTICKS_ARGS=${cmdline/$cfm}   ;;
                                     *) export OPTICKS_ARGS=$cmdline ;;
       esac
    fi 
}


op-cmdline-geometry-match()
{
    local msg="=== $FUNCNAME : finds 1st argument with associated geometry :"
    local arg
    local geo
    unset OPTICKS_GEO
    for arg in $cmdline 
    do
       geo=$(op-geometry-name $arg)
       #echo arg $arg geo $geo 
       if [ "$geo" != "" ]; then 
           export OPTICKS_GEO=$geo
           return 
       fi
    done
}


op-cmdline-parse()
{
    #op-cmdline-dump
    op-cmdline-specials

    op-cmdline-binary-match
    op-cmdline-geometry-match

    op-binary-setup
    op-geometry-setup
}


op-export()
{
   export-
   export-export
}

op-runline()
{
   local runline
   if [ "${OPTICKS_BINARY: -3}" == ".py" ]; then
      runline="python ${OPTICKS_BINARY} ${OPTICKS_ARGS} "
   elif [ "${OPTICKS_DBG}" == "1" ]; then 
      case $(uname) in
          Darwin) runline="lldb ${OPTICKS_BINARY} -- ${OPTICKS_ARGS} " ;;
               *) runline="gdb  ${OPTICKS_BINARY} -- ${OPTICKS_ARGS} " ;;
      esac
   else
      runline="${OPTICKS_BINARY} ${OPTICKS_ARGS}" 
   fi
   echo $runline
}



opticks-
op-cmdline-parse
runline=$(op-runline)
op-export

if [ "$sauce" == "1" ]; then
   #echo sauce detected : assume are debugging this script
   echo -n
elif [ "${cmdline/--ophelp}" != "${cmdline}" ]; then
   op-help
else
   >&2 ls -alst ${OPTICKS_BINARY}
   env | >&2 grep OPTICKS
   >&2 echo proceeding : $runline
   eval $runline
fi 


