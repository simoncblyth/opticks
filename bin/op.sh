#!/bin/bash -l
[ "$0" == "$BASH_SOURCE" ] && sauce=0 || sauce=1

op-(){    . $(which op.sh) ; } 
op-vi(){ vi $(which op.sh) ; } 


cmdline="$*"

op-usage(){ cat << \EOU

op.sh : Opticks Launching Script
===================================

The **op.sh** script launches different Opticks executables
or scripts depending on the arguments provided. It also 
sets environment variables picking a detector geometry
and selecting volumes within the geometry.

Most usage of Opticks should use this script.

To see the options specific to particular scripts or
executables use "-h" rather than the "--help" 
that provides this text.


Profile Setup 
---------------

To save typing add the below bash function to your .bash_profile::

   op(){ op.sh $* ; }


Commandline Help
------------------

Lists of geometry config arguments and executable selection are provided by the help

::

    op --help




EOU
}

#op-binary-name-default(){ echo GGeoViewTest ; }
#op-binary-name-default(){ echo OpticksMgrTest ; }
op-binary-name-default(){ echo OKTest ; }

op-binary-names(){ type op-binary-name | perl -ne 'm,--(\w*)\), && print "$1\n" ' - ; } 
op-binary-name()
{
   case $1 in 
         --idpath) echo OpticksIDPATH ;;
           --keys) echo InteractorKeys ;;
          --tcfg4) echo CG4Test ;;
           --okg4) echo OKG4Test ;;
         --tracer) echo OTracerTest ;;
      --gdml2gltf) echo gdml2gltf.py ;;
            --mat) echo GMaterialLibTest ;;
           --cmat) echo CMaterialLibTest ;;
           --surf) echo GSurfaceLibTest ;;
           --snap) echo OpTest ;;
            --bnd) echo GBndLibTest ;;
  --ctestdetector) echo CTestDetectorTest ;;
  --cgdmldetector) echo CGDMLDetectorTest ;;
     --ngunconfig) echo NGunConfigTest ;;
   --gpropertymap) echo GPropertyMapTest ;;
  --gscintillatorlib) echo GScintillatorLibTest ;;
   --opticksquery) echo OpticksQueryTest ;;  
    --gitemindex) echo GItemIndexTest ;;  
        --nindex) echo IndexTest ;;  
      --tindexer) echo IndexerTest ;;  
      --tevtload) echo EvtLoadTest ;;  
--topticksgeometry) echo OpticksGeometryTest ;;  

     --boundaries) echo BoundariesNPYTest ;;
           --recs) echo RecordsNPYTest ;;
         --lookup) echo LookupTest ;;
       --itemlist) echo GItemListTest ;;
        --gsource) echo GSourceTest ;;
        --gsrclib) echo GSourceLibTest ;;
       --gmeshlib) echo GMeshLibTest ;;
       --resource) echo OpticksResourceTest ;;
        --opticks) echo OpticksTest ;;
          --pybnd) echo GBndLibTest.py ;;
         --extras) echo extras.py ;;
             --mm) echo GMergedMeshTest ;;
        --testbox) echo GTestBoxTest ;;
         --geolib) echo GGeoLibTest ;;
        --geotest) echo GGeoTestTest ;;
         --gmaker) echo GMakerTest ;;
            --pmt) echo GPmtTest ;;
           --attr) echo GAttrSeqTest ;;
         --oscint) echo OScintillatorLibTest ;;
          --flags) echo GFlagsTest ;;
        --gbuffer) echo GBufferTest ;;
           --meta) echo GBoundaryLibMetadataTest ;;
         --sensor) echo NSensorListTest ;;
          --tggeo) echo GGeoTest ;;
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
         --idpath) echo "Emit to stdout the path of the geocache directory for the geometry selected by arguments" ;;
           --keys) echo "List key controls available in GGeoViewTest " ;;
          --tcfg4) echo "Geant4 comparison simulation of simple test geometries " ;; 
           --okg4) echo "Integrated Geant4/Opticks runing allowing G4GUN steps to be directly Opticks GPU propagated. " ;; 
         --tracer) echo "Fast OpenGL viz and OptiX tracing, NO propagation. From ggeoview-/tests. Used for simple geometry/machinery checking"  ;;
      --gdml2gltf) echo "Once only geometry conversion of GDML input into GLTF file needed for analytic geocache creation" ;;
            --mat) echo "Dump properties of material identified by 0-based index , eg op --mat 0 " ;;
           --surf) echo "Dump properties of surface identified by 0-based index , eg op --surf 0 " ;;
           --snap) echo "Pure compute raytrace snapshot of geometry. " ;;
            --bnd) echo "Dump boundaries of a geometry, eg op --bnd --jpmt " ;; 
  --ctestdetector) echo "Test Geant4 simple detector construction using class cfg4-/CTestDetector " ;; 
  --cgdmldetector) echo "Test Geant4 GDML full detector construction using cfg4-/CGDMLDetector " ;; 
       --cproblib) echo "Test Opticks/Geant4 material property library/converter cfg4-/CPropLib " ;; 
     --ngunconfig) echo "Test Geant4 Gun configuration with npy-/NGunConfigTest ";;
   --gpropertymap) echo "Test creation and persisting of GPropertyMap " ;;
  --gscintillatorlib) echo "Test loading of GScintillatorLib " ;;
  --opticksquery) echo "Test parsing of OPTICKS_QUERY geometry selection"  ;;  
    --gitemindex) echo "Test presentation of item indices, such as seqmat and seqhis photon flag sequences " ;;  
        --nindex) echo "npy-/Index test " ;;  
      --tindexer) echo "optickscore-/IndexerTest " ;;  
      --tevtload) echo "optickscore-/EvtLoadTest " ;;  
--topticksgeometry) echo "opticksgeo-/OpticksGeometryTest"  ;;  
   esac 
}






op-geometry-unset()
{
    unset OPTICKS_GEOKEY
    unset OPTICKS_QUERY 
    unset OPTICKS_CTRL
    unset OPTICKS_MESHFIX
    unset OPTICKS_MESHFIX_CFG
}

op-geometry-setup-notes(){ cat << EON

FUNCTIONS
~~~~~~~~~~

*op-geometry-unset*

    unset the output envvars 

*op-geometry-name arg*

    checks arg to see if it is a geometry selection arg, if so emits 
    the corresponding tag eg DYB for argument --dyb or JPMT for argument --jpmt

*op-cmdline-geometry-match*

    loops over commandline arguments checking for geometry selection
    args, the first tag found eg DYB or JPMT is set as the 
    value of the OPTICKS_GEO envvar  

    Multiple similar tags typically correspond to different geometry selections 
    within a single detector geometry .dae file.

    * DYB,IDYB,JDYB,KDYB,..  
    * JUNO,JPMT,JTST

*op-geometry-setup tag*

    OPTICKS_GEO envvar (or tag argument) is a high level tag 
    that identifies a detector and potentially a selection of that detectors 
    geometrical volumes.

    The value of the tag leads to the setting of envvars which are 
    specific to the particular geometry.

ENVVARS
~~~~~~~~~

OPTICKS_GEO

    mainly for internal use of op.sh, set based on commandline arguments
    its value results in the setting of the other envvars listed below

OPTICKS_GEOKEY

    names another envvar that contains the path to the .dae
    eg OPTICKSDATA_DAEPATH_DYB

    The OPTICKSDATA_ envvars are internally set via an ini file
    \$OPTICKS_INSTALL_PREFIX/opticksdata/config/opticksdata.ini
    The indirection is used to isolate paths which may be different
    for every installation from general handling.

OPTICKS_QUERY

    volume selection, eg range:3153:12221

OPTICKS_CONTROL

    influnces the geometry import, currently

OPTICKS_MESHFIX

    names of meshes to be fixed eg iav, oav
  
OPTICKS_MESHFIX_CFG

     configuration of meshfixing 


Questions
~~~~~~~~~~~~~

Do I recall correctly ? Geometry selection arguments are baked 
into the geocache director name digest so changing geometry selection 
arguments like the OPTICKS_QUERY volume range will require 
a rebuild of the geocache, unless that configuration 
has been used previously.


EON
}

op-geometry-names(){ type op-geometry-name | perl -ne 'm,--(\w*)\), && print "$1\n" ' - ; } 
op-geometry-name()
{
   case $1 in 
       --dyb)  echo DYB ;; 
       --dlin) echo DLIN ;; 
       --dfar) echo DFAR ;; 
       --dpib) echo DPIB ;; 
       --dsst) echo DSST ;; 
       --dsst2) echo DSST2 ;; 
       --drv3153) echo DRV3153 ;; 
       --drv3155) echo DRV3155 ;; 
       --dlv17) echo DLV17 ;; 
       --dlv30) echo DLV30 ;; 
       --dlv46) echo DLV46 ;; 
       --dlv55) echo DLV55 ;; 
       --dlv56) echo DLV56 ;; 
       --dlv65) echo DLV65 ;; 
       --dlv66) echo DLV66 ;; 
       --dlv67) echo DLV67 ;; 
       --dlv68) echo DLV68 ;; 
       --dlv103) echo DLV103 ;; 
       --dlv140) echo DLV140 ;; 
       --dlv185) echo DLV185 ;; 
       --jpmt) echo JPMT ;; 
       --j1707) echo J1707 ;; 
       --lxe)  echo LXE ;; 

       --idyb) echo IDYB ;; 
       --jdyb) echo JDYB ;; 
       --kdyb) echo KDYB ;; 
       --ldyb) echo LDYB ;; 
       --mdyb) echo MDYB ;; 
       --juno) echo JUNO ;; 
       --jtst) echo JTST ;; 
       --dpmt) echo DPMT ;; 
   esac
}

op-geometry-desc()
{
   case $1 in 
      --dyb)  echo "DayaBay Near Site" ;; 
      --dlin) echo "DayaBay LingAo Site" ;; 
      --dfar) echo "DayaBay Far Site" ;; 
      --dpib) echo "DayaBay PMT in Box of Mineral Oil Test Geometry" ;;
      --dsst) echo "DYB debugging SST rib impingement" ;;
      --dsst2) echo "DYB debugging SST rib impingement" ;;
      --dlv65) echo "DYB cycybobo lvid 65 " ;;
      --jpmt) echo "JUNO with PMTs" ;;
      --lxe)  echo "Geant4 LXe Liquid Xenon example" ;; 
      --j1707)  echo "JUNO with ~36k 3inch PMTs, ~18k 20inch PMTs, torus guide tube " ;; 
   esac
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



op-geometry-setup()
{
    local geo=${OPTICKS_GEO:-DYB}
    op-geometry-unset 
    case $geo in 
     DYB|IDYB|JDYB|KDYB|LDYB|MDYB|DLIN|DFAR|DSST2|DLV*|DRV*) op-geometry-setup-dyb  $geo  ;;
                             JUNO|JPMT|JTST|J1707) op-geometry-setup-juno $geo  ;;
                              DPIB|DPMT|LXE) op-geometry-setup-misc $geo  ;;
    esac
}
op-geometry-query-dyb()
{
    case $1 in 
   DYB|DLIN)  echo "range:3153:12221"  ;;
       DFAR)  echo "range:4686:18894"   ;;  #  
       IDYB)  echo "range:3158:3160" ;;  # 2 volumes : pvIAV and pvGDS
       JDYB)  echo "range:3158:3159" ;;  # 1 volume : pvIAV
       KDYB)  echo "range:3159:3160" ;;  # 1 volume : pvGDS
       LDYB)  echo "range:3156:3157" ;;  # 1 volume : pvOAV
       MDYB)  echo "range:3201:3202,range:3153:3154"  ;;  # 2 volumes : first pmt-hemi-cathode and ADE  
       DSST2)  echo "range:3155:3156,range:4440:4448" ;;    # large BBox discrep
       DRV3153) echo "index:3153,depth:13" ;;  
       DRV3155) echo "index:3155,depth:20" ;;  
       DLV17)  echo "range:3155:3156,range:2436:2437" ;;    # huh just see the cylinder
       DLV30)  echo "range:3155:3156,range:3167:3168" ;;    #
       DLV46)  echo "range:3155:3156,range:3200:3201" ;;    #
       DLV55)  echo "range:3155:3156,range:4357:4358" ;;    #
       DLV56)  echo "range:3155:3156,range:4393:4394" ;;    #
       DLV65)  echo "range:3155:3156,range:4440:4441" ;;  
       DLV66)  echo "range:3155:3156,range:4448:4449" ;;  
       DLV67)  echo "range:3155:3156,range:4456:4457" ;; 
       DLV68)  echo "range:3155:3156,range:4464:4465" ;;    # 
      DLV103)  echo "range:3155:3156,range:4543:4544" ;;    #
      DLV140)  echo "range:3155:3156,range:4606:4607" ;;    #
      DLV185)  echo "range:3155:3156,range:4799:4800" ;;    #
    esac
    # range:3154:3155  SST  Stainless Steel/IWSWater not a good choice for an envelope, just get BULK_ABSORB without going anywhere
}
op-geometry-setup-dyb()
{
    local geo=${1:-DYB}
    local geokey
    case $geo in 
         DYB) geokey=OPTICKSDATA_DAEPATH_DYB ;;
        DLIN) geokey=OPTICKSDATA_DAEPATH_DLIN ;;
        DFAR) geokey=OPTICKSDATA_DAEPATH_DFAR ;;
           *) geokey=OPTICKSDATA_DAEPATH_DYB ;;
    esac

    export OPTICKS_GEOKEY=$geokey
    export OPTICKS_QUERY=$(op-geometry-query-dyb $geo) 
    export OPTICKS_CTRL="volnames"
    export OPTICKS_MESHFIX="iav,oav"
    export OPTICKS_MESHFIX_CFG="100,100,10,-0.999"   # face barycenter xyz alignment and dot face normal cuts for faces to be removed 
}
op-geometry-setup-juno()
{
   local geo=${1:-JPMT}
   if [ "$geo" == "JUNO" ]; then 
       export OPTICKS_GEOKEY=OPTICKSDATA_DAEPATH_JUNO
       export OPTICKS_QUERY="range:1:50000"
       export OPTICKS_CTRL=""
   elif [ "$geo" == "JPMT" ]; then
       export OPTICKS_GEOKEY=OPTICKSDATA_DAEPATH_JPMT
       export OPTICKS_QUERY="range:1:289734"  # 289733+1 all test3.dae volumes
       export OPTICKS_CTRL=""
   elif [ "$geo" == "JTST" ]; then
       export OPTICKS_GEOKEY=OPTICKSDATA_DAEPATH_JTST
       export OPTICKS_QUERY="range:1:50000" 
       export OPTICKS_CTRL=""
   elif [ "$geo" == "J1707" ]; then
       export OPTICKS_GEOKEY=OPTICKSDATA_DAEPATH_J1707
       export OPTICKS_QUERY="all" 
       export OPTICKS_CTRL="volnames"
   fi
}
op-geometry-setup-misc()
{
   local geo=${1:-DPIB}
   if [ "$geo" == "DPIB" ]; then
       export OPTICKS_GEOKEY=OPTICKSDATA_DAEPATH_DPIB
       export OPTICKS_QUERY="" 
       export OPTICKS_CTRL=""
    elif [ "$geo" == "DPMT" ]; then
       export OPTICKS_GEOKEY=OPTICKSDATA_DAEPATH_DPIB
       export OPTICKS_QUERY="range:1:6"   # exclude the box at first slot   
       export OPTICKS_CTRL=""
    elif [ "$geo" == "LXE" ]; then
       export OPTICKS_GEOKEY=OPTICKSDATA_DAEPATH_LXE
       export OPTICKS_QUERY="" 
       export OPTICKS_CTRL=""
    fi 
}

op-geometry-setup-analytic()
{
    local msg="=== $FUNCNAME :"
    echo $msg 
    env | grep OPTICKS_
}



op-help(){
   local cmd
   local bin
   local hlp

   op-usage

   printf "\nGEOMETRY SELECTION ARGUMENTS \n\n" ;
   op-geometry-names | while read cmd ; do
      bin=$(op-geometry-name "--$cmd")
      desc=$(op-geometry-desc "--$cmd")
      [ -z "$desc" ] && continue ; 
      printf " %20s : %25s : %s \n" "--$cmd"  $bin  "$desc"
   done

   printf "\nBINARY SELECTION ARGUMENTS \n\n" ;
   op-binary-names | while read cmd ; do
      bin=$(op-binary-name "--$cmd")
      desc=$(op-binary-desc "--$cmd")
   #   [ -z "$desc" ] && continue ; 
      printf " %20s : %25s : %s \n" "--$cmd"  $bin  "$desc"
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

   if [ "${cmdline/--malloc}" != "${cmdline}" ]; then
       export OPTICKS_MALLOC=1
   fi
   if [ "${cmdline/--debugger}" != "${cmdline}" ]; then
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
       #echo arg $arg bin $bin  
       if [ "$bin" != "" ]; then 
           export OPTICKS_CMD=$arg
           echo $msg $arg
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

    local inbin=$bin

    if [ "$OPTICKS_LOAD" == "1" ]; then 
        case $bin in
           OKTest|OKG4Test)  echo -n ;;
                         *)  bin=$def ;; 
        esac 
    fi 

    if [ "$bin" != "$inbin" ]; then
        echo $msg OPTICKS_LOAD overrides binary from $inbin to default $bin as not all binaries support OpticksEvent loading
    fi

    #echo $msg cfm $cfm bin $bin def $def
    unset OPTICKS_BINARY 
    unset OPTICKS_ARGS

    if [ "$bin" != "" ]; then

       local ubin
       if [ "${bin: -3}" == ".py" ]; then
           ubin=$(which $bin)
       else
           ubin=$(opticks-bindir)/$bin
       fi

       export OPTICKS_BINARY=$ubin
       # some commands should not be removed from the commandline
       # as they are needed by the binary 
       echo ubin $ubin cfm $cfm cmdline $cmdline

       case $cfm in 
         --surf|--scint|--oscint|--pmt|--j1707) export OPTICKS_ARGS=${cmdline/$cfm}   ;;
                                             *) export OPTICKS_ARGS=$cmdline ;;
       esac
    fi 

    local presentation=""
    case $(uname) in
       Linux) presentation="--size 1920,1080,1 " ;;
      MINGW*) presentation="--size 1920,1080,1 --fullscreen" ;;
           *) presentation="" ;;
    esac

    OPTICKS_ARGS="$presentation ${OPTICKS_ARGS} " 


    if [ "${OPTICKS_ARGS/rendermode}" == "${OPTICKS_ARGS}" ]
    then 
        OPTICKS_ARGS="--rendermode +global,+in0,+axis" 
    fi 

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
   local msg="=== $FUNCNAME :"

   # TODO: avoid need for any envvars (other than PATH) 
   #opticksdata-
   #opticksdata-export

   ## but python scrips need IDPATH 

   echo $msg OPTICKS_BINARY ${OPTICKS_BINARY}


   if [ "${OPTICKS_BINARY: -3}" == ".py" ]; then
       IDPATH=$(OpticksIDPATH ${OPTICKS_ARGS}  2>&1 > /dev/null)
       export IDPATH
       echo $msg IDPATH $IDPATH
   fi 

}


op-windows-debug(){ cat << \EOM

Windows debugging from commandline not yet implemented.
Instead try Visual Studio, from Powershell with 
vs-export in profile run:

   opticks-vs 

Select "RelWithDebInfo" config 

Then select the desired target as the startup project

TODO: work out way of passing commandline args into Visual Studio
      (perhaps using devenv ?)


EOM
}

op-runline()
{
   local runline
   if [ "${OPTICKS_BINARY: -3}" == ".py" ]; then
      runline="python ${OPTICKS_BINARY} ${OPTICKS_ARGS} "
   elif [ "${OPTICKS_DBG}" == "1" ]; then 
      case $(uname) in
          Darwin) runline="lldb ${OPTICKS_BINARY} -- ${OPTICKS_ARGS} " ;;
           MING*) runline="     ${OPTICKS_BINARY} -- ${OPTICKS_ARGS} " ;; 
               *) runline="gdb  --args ${OPTICKS_BINARY} ${OPTICKS_ARGS} " ;;
      esac
   else
      runline="${OPTICKS_BINARY} ${OPTICKS_ARGS}" 
   fi
   echo $runline
}

op-malloc()
{
   #if [ "${OPTICKS_MALLOC}" == "1" ]; then 

       export MallocStackLoggingNoCompact=1   # all allocations are logged
       export MallocScribble=1     # free sets each byte of every released block to the value 0x55.
       export MallocPreScribble=1  # sets each byte of a newly allocated block to the value 0xAA
       export MallocGuardEdges=1   # adds guard pages before and after large allocations
       export MallocCheckHeapStart=1 
       export MallocCheckHeapEach=1 
   #fi
}


opticks-
op-cmdline-parse

runline=$(op-runline)




op-export

if [ "$sauce" == "1" ]; then
   #echo sauce detected : assume are debugging this script
   echo -n
elif [ "${cmdline/--help}" != "${cmdline}" ]; then
   op-help

elif [ "${cmdline/--idpath}" != "${cmdline}" ]; then

   IDPATH=$(OpticksIDPATH ${OPTICKS_ARGS}  2>&1 > /dev/null)
   echo IDPATH $IDPATH

else
  
   if [ "${OPTICKS_BINARY/OpticksIDPATH}" == "${OPTICKS_BINARY}" ]; then 
       export OPTICKS_QUIET=1
   else
       unset OPTICKS_QUIET 
   fi 

   IDPATH=$(OpticksIDPATH ${OPTICKS_ARGS}  2>&1 > /dev/null)  

   ## capture only stderr, the directory 
   ## NB Opticks executables do not need IDPATH envvar 
   ## but python analysis scripts needing access to geocache need this

   if [ -n "$OPTICKS_QUIET" ]; then 

       if [ ${OPTICKS_BINARY/\/} == ${OPTICKS_BINARY} ]; then 
           >&2 ls -alst $(which ${OPTICKS_BINARY})
       else 
           >&2 ls -alst ${OPTICKS_BINARY}
       fi  

      # env | >&2 grep OPTICKS_ | sort  
       >&2 echo proceeding.. : $runline
   fi 


   #op-malloc 

   eval $runline
   RC=$?
   echo $0 RC $RC
   exit $RC

   cat << EOC
# geocache directory corresponding to OPTICKS_ARGS ${OPTICKS_ARGS} 
export IDPATH=$IDPATH
EOC

fi 



