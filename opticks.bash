opticks-(){         source $(opticks-source) && opticks-env $* ; }
opticks-src(){      echo opticks.bash ; }
opticks-source(){   echo $BASH_SOURCE ; }
opticks-vi(){       vi $(opticks-source) ; }
opticks-help(){ opticks-usage ; }
opticks-usage(){   cat << \EOU

OPTICKS BASH FUNCTIONS
========================

*opticks-rdocs*
     open browser on the remote html documentation
     https://simoncblyth.bitbucket.io/opticks/index.html

*opticks-docs*
     open browser on the local html documentation

*opticks-docs-make*
     sphinx-build the docs

*opticks-notes*
     open browser on the local html development notes 

*opticks-notes-make*
     sphinx-build the notes


Most Opticks development notes are kept in the env repository, 
use start from 

   env-;opticksdev-;opticksdev-vi


EOU
}

opticks-env(){      
   # dont pollute : otherwise will get infinite loops : as opticks is used in many other -env
   . $(opticks-home)/externals/externals.bash   ## just precursors
}

opticks-env-info(){

   cat << EOI

       uname   : $(uname -a)
       HOME    : $HOME
       VERBOSE : $VERBOSE
       USER    : $USER

EOI
   env | grep OPTICKS
   
}


olocal-()
{
   echo -n # transitional standin for olocal-
}

opticks-home-default(){ echo $(dirname $(opticks-source)) ; }
opticks-home(){   echo ${OPTICKS_HOME:-$(opticks-home-default)} ; }  ## input from profile 
opticks-name(){   basename $(opticks-home) ; }


opticks-pretty(){  cat ${1:-some.json} | python -m json.tool ; }

opticks-idfold(){ echo $(dirname $IDPATH) ; }
opticks-srcpath(){ echo $(opticks-idpath2srcpath $IDPATH) ; }
opticks-srcfold(){ echo $(dirname $(opticks-srcpath)) ; }
#opticks-srcextras(){ echo $(opticks-idfold)/extras ; }   # layout 0
opticks-srcextras(){ echo $(opticks-srcfold)/extras ; }  # layout 1

opticks-join(){ local ifs=$IFS ; IFS="$1"; shift; echo "$*" ; IFS=$ifs ;  }

opticks-paths(){ cat << EON

$FUNCNAME
===============================

The srcpath if obtained from the IDPATH envvar using 
opticks-idpath2srcpath  which is the bash equivalant 
of the C++ brap-/BPath and python base/bpath.py 

    IDPATH          : $IDPATH

    opticks-srcpath : $(opticks-srcpath)
    opticks-srcfold : $(opticks-srcfold)


    opticks-srcextras     : $(opticks-srcextras)
    opticks-tbool-path 0  : $(opticks-tbool-path 0)
    opticks-nnt-path 0    : $(opticks-nnt-path 0)

EON
}

opticks-idpath2srcpath()
{
   local idpath=$1
   local ifs=$IFS
   local elem
   IFS="/"
   declare -a elem=($idpath)
   IFS=$ifs 

   local nelem=${#elem[@]}
   local last=${elem[$nelem-1]}   ## -ve indices requires bash 4.3+
   #echo nelem $nelem last $last 

   IFS="." 
   declare -a bits=($last)
   IFS=$ifs 
   local nbits=${#bits[@]}
 
   local idfile
   local srcdigest 
   local idname
   local prefix

   if [ "$nbits" == "3" ] ; then

      idfile=$(opticks-join . ${bits[0]} ${bits[2]}) 
      srcdigest=${bits[1]}
      idname=${elem[$nelem-2]}
      prefix=$(opticks-join / ${elem[@]:0:$nelem-4})

      #echo triple idfile $idfile srcdigest $srcdigest idname $idname prefix $prefix 
   else
      srcdigest=${elem[$nelem-2]}
      idfile=${elem[$nelem-3]}
      idname=${elem[$nelem-4]}
      prefix=$(opticks-join / ${elem[@]:0:$nelem-5}) 

      #echo not triple idfile $idfile srcdigest $srcdigest idname $idname prefix $prefix   
   fi  
   local srcpath=$(opticks-join / "" $prefix "opticksdata" "export" $idname $idfile)
   IFS=$ifs 

   echo $srcpath
}

opticks-idpath2srcpath-test-one()
{
   local v=$IDPATH
   local s=$(opticks-idpath2srcpath $v)
   printf "%40s %40s \n" $v $s 
   local s2=$(opticks-idpath2srcpath $v)
   printf "%40s %40s \n" $v $s2 
}

opticks-idpath2srcpath-test()
{
    local ifs=$IFS
    local line
    local kv
    env | grep IDPATH | while read line  
    do    
       IFS="="
       declare -a kv=($line) 
       IFS=$ifs

       if [ ${#kv[@]} == "2" ]; then 

           local k=${kv[0]}
           local v=${kv[1]}

           local s=$(opticks-idpath2srcpath $v)
           printf "%10s %40s %40s \n" $k $v $s 
       fi 
    done

}


opticks-tbool-info(){ cat << EOI

$FUNCNAME
======================



EOI
}

opticks-tbool-path(){ 
   local lvid=${1:-0} 
   local extras=$(opticks-srcextras)
   local path=$extras/${lvid}/tbool${lvid}.bash
   echo $path 
}
opticks-nnt-path(){ 
   local lvid=${1:-0} 
   local extras=$(opticks-srcextras)
   local path=$extras/${lvid}/NNodeTest_${lvid}.cc
   echo $path 
}
opticks-nnt-paths(){
    local arg
    for arg in $* 
    do
        echo $(opticks-nnt-path $arg)
    done
}

opticks-nnt-vi(){ 
   [[ $# -eq 0 ]] && echo expecting one or more lvidx integer arguments && return 
   local paths=$(opticks-nnt-paths $*)
   vi $paths
}

opticks-nnt-(){ 

   local path=${1:-/tmp/some/path/to/NNodeTest_LVID.cc}
   local name=$(basename $path)   
   local stem=${name/.cc}
   local bindir=$(opticks-bindir)
   local bin=$bindir/$stem

   npy-
   sysrap-
   glm-
   plog-
 
   cat << EOL

   clang -std=c++11  $path \
            -I$(npy-sdir) \
            -I$(sysrap-sdir) \
            -I$(glm-dir) \
            -I$(plog-dir)/include \
             -lNPY \
             -lSysRap \
             -lc++ \
             -L$(opticks-bindir) \
             -Wl,-rpath,$(opticks-bindir) \
             -o $bin
   
EOL
}

opticks-nnt()
{
   local msg="$FUNCNAME :"
   local lvid=${1:-0} 
   local path=$(opticks-nnt-path $lvid)
   local name=$(basename $path)   
   local stem=${name/.cc}

   [ ! -f $path ] && echo $msg no such path && return

   echo $msg compiling $path 

   eval $($FUNCNAME- $path)  && $stem

   which $stem

   

}




opticks-tbool-vi(){ 
   local path=$(opticks-tbool-path ${1:-0})
   vi $path
}

opticks-tbool(){ 
   local msg="$FUNCNAME :"
   local lvid=${1:-0} 
   local path=$(opticks-tbool-path $lvid)
   echo $msg sourcing $path 
   [ ! -f $path ] && echo $msg no such path && return
   . $path
   tbool${lvid}
}

opticks-tbool-(){ 
   local msg="$FUNCNAME :"
   local lvid=${1:-0} 
   local path=$(opticks-tbool-path $lvid)
   echo $msg sourcing $path 
   [ ! -f $path ] && echo $msg no such path && return
   . $path
   tbool${lvid}-
}




# reliance on envvars is appropriate for debugging only, not "production" 
opticks-debugging-idpath(){ echo $IDPATH ; }
opticks-debugging-idfold(){ echo $(dirname $(opticks-debugging-idpath)) ; }

#opticks-tscan-dir(){ echo  $TMP/tgltf/extras/${1:-0} ; }
opticks-tscan-dir(){ echo  $(opticks-debugging-idfold)/extras/${1:-0} ; }

opticks-tscan-all(){ opticks-tscan / ; }
opticks-tscan(){
   local msg="$FUNCNAME :"
   local lvid=${1:-0} 
   local dir=$(opticks-tscan-dir $lvid)
   [ ! -d $dir ] && echo $msg no such dir $dir && return
   echo $msg scanning $dir 
   NScanTest $dir
}




opticks-suffix(){
   case $(uname) in
      MING*) echo .exe ;;
          *) echo -n  ;;   
   esac
}

opticks-day(){ date +"%Y%m%d" ; }

opticks-fold(){ 
   # when LOCAL_BASE unset rely instead on finding an installed binary from PATH  

   if [ -n "$OPTICKS_GREENFIELD_TEST" ]; then
       echo /tmp/$USER/opticks$(opticks-day) 
   else
       if [ -z "$LOCAL_BASE" ]; then 
          echo $(dirname $(dirname $(which OpticksTest$(opticks-suffix))))
       else
          echo ${LOCAL_BASE}/$(opticks-name) ;
       fi
   fi
}



opticks-sdir(){   echo $(opticks-home) ; }
opticks-scd(){  cd $(opticks-sdir)/$1 ; }
opticks-ncd(){  opticks-scd notes/issues ;  }

opticks-prefix(){ echo $(opticks-fold)  ; }

opticks-dir(){    echo $(opticks-prefix) ; }
opticks-idir(){   echo $(opticks-prefix) ; }
opticks-bdir(){   echo $(opticks-prefix)/build ; }
opticks-bindir(){ echo $(opticks-prefix)/lib ; }   ## use lib for executables for simplicity on windows

opticks-xdir(){   echo $(opticks-fold)/externals ; }  ## try putting externals above the build identity 

opticks-installcachedir(){ echo $(opticks-fold)/installcache ; }


opticks-c(){    cd $(opticks-dir) ; }
opticks-cd(){   cd $(opticks-dir) ; }
opticks-icd(){  cd $(opticks-idir); }
opticks-bcd(){  cd $(opticks-bdir); }
opticks-xcd(){  cd $(opticks-xdir); }




opticks-optix-install-dir(){ echo ${OPTICKS_OPTIX_INSTALL_DIR:-$($FUNCNAME-)} ; }
opticks-optix-install-dir-(){
   local t=$NODE_TAG
   case $t in 
      E) echo /Developer/OptiX_501 ;;
      D_400) echo /Developer/OptiX_400 ;;
      D) echo /Developer/OptiX_380 ;;
   RYAN) echo /Developer/OptiX_380 ;;
    GTL) echo ${MYENVTOP}/OptiX ;;
   H5H2) echo ${MYENVTOP}/OptiX ;;
      X) echo /usr/local/optix-3.8.0/NVIDIA-OptiX-SDK-3.8.0-linux64 ;;
   #SDUGPU) echo /root/NVIDIA-OptiX-SDK-4.1.1-linux64 ;;
   SDUGPU) echo /home/simon/NVIDIA-OptiX-SDK-4.1.1-linux64 ;;
      *) echo /tmp ;;
   esac
} 


opticks-compute-capability(){ echo ${OPTICKS_COMPUTE_CAPABILITY:-$($FUNCNAME-)} ; }
opticks-compute-capability-()
{
    local t=$NODE_TAG
    case $t in 
       E) echo 30 ;;
       D) echo 30 ;;
    RYAN) echo 30 ;;
     GTL) echo 30 ;;
    H5H2) echo 50 ;;
       X) echo 52 ;; 
  SDUGPU) echo 30 ;; 
       *) echo  0 ;;
    esac
}

opticks-externals(){ cat << EOL
glm
glfw
glew
gleq
imgui
assimp
openmesh
plog
opticksdata
oimplicitmesher
odcs
oyoctogl
ocsgbsp
EOL
}

opticks-optionals(){ cat << EOL
xercesc
g4
EOL
}

opticks-possibles(){ cat << EOL
oof
EOL
}


opticks-externals-notes(){ cat << EON

oimplicitmesher
    requires glm, finds it using opticks/cmake/Modules/FindGLM.cmake
    this means use common prefix with opticks
odcs
    requires glm, finds it using opticks/cmake/Modules/FindGLM.cmake
    this means use common prefix with opticks



EON
}


opticks-externals-install(){ echo $FUNCNAME ; opticks-externals | -opticks-installer ; }
opticks-externals-url(){     echo $FUNCNAME ; opticks-externals | -opticks-url ; }
opticks-externals-dist(){    echo $FUNCNAME ; opticks-externals | -opticks-dist ; }

opticks-optionals-install(){ echo $FUNCNAME ; opticks-optionals | -opticks-installer ; }
opticks-optionals-url(){     echo $FUNCNAME ; opticks-optionals | -opticks-url ; }
opticks-optionals-dist(){    echo $FUNCNAME ; opticks-optionals | -opticks-dist ; }

opticks-possibles-install(){ echo $FUNCNAME ; opticks-possibles | -opticks-installer ; }
opticks-possibles-url(){     echo $FUNCNAME ; opticks-possibles | -opticks-url ; }
opticks-possibles-dist(){    echo $FUNCNAME ; opticks-possibles | -opticks-dist ; }


opticks-full()
{
    local msg="=== $FUNCNAME :"
    echo $msg START $(date)
    opticks-info

    if [ ! -d "$(opticks-prefix)/externals" ]; then

        echo $msg installing the below externals into $(opticks-prefix)/externals
        opticks-externals 
        opticks-externals-install


    else
        echo $msg using preexisting externals from $(opticks-prefix)/externals
    fi 

    opticks-configure

    opticks--

    opticks-prepare-installcache

    echo $msg DONE $(date)
}



-opticks-installer(){
   local msg="=== $FUNCNAME :"
   echo $msg START $(date)
   local ext
   while read ext 
   do
        echo $msg $ext
        $ext-
        $ext--
   done
   echo $msg DONE $(date)
}

-opticks-url(){
   local ext
   while read ext 
   do
        $ext-
        printf "%30s :  %s \n" $ext $($ext-url) 
   done
}

-opticks-dist(){
   local ext
   local dist
   while read ext 
   do
        $ext-
        dist=$($ext-dist 2>/dev/null)
        printf "%30s :  %s \n" $ext $dist
   done
}


opticks-locations(){ cat << EOL

$FUNCNAME
==================

      opticks-source   :   $(opticks-source)
      opticks-home     :   $(opticks-home)
      opticks-name     :   $(opticks-name)

      opticks-fold     :   $(opticks-fold)

      opticks-sdir     :   $(opticks-sdir)
      opticks-idir     :   $(opticks-idir)
      opticks-bdir     :   $(opticks-bdir)
      opticks-xdir     :   $(opticks-xdir)
      ## cd to these with opticks-scd/icd/bcd/xcd

      opticks-installcachedir   :  $(opticks-installcachedir)
      opticks-bindir            :  $(opticks-bindir)
      opticks-optix-install-dir :  $(opticks-optix-install-dir)

EOL
}


opticks-info(){
   opticks-locations
   opticks-env-info
   opticks-externals-url
   opticks-externals-dist
   opticks-optionals-url
   opticks-optionals-dist
}



opticks-cmake-generator()
{
    if [ "$NODE_TAG" == "M" ]; then
       echo MSYS Makefiles 
    else  
       case $(uname -s) in
         MINGW64_NT*)  echo Visual Studio 14 2015 ;;
                   *)  echo Unix Makefiles ;;
       esac                          
    fi
}

opticks-cmake-info(){ g4- ; xercesc- ; cat << EOI

$FUNCNAME
======================

       NODE_TAG                   :  $NODE_TAG

       opticks-sdir               :  $(opticks-sdir)
       opticks-bdir               :  $(opticks-bdir)
       opticks-cmake-generator    :  $(opticks-cmake-generator)
       opticks-compute-capability :  $(opticks-compute-capability)
       opticks-prefix             :  $(opticks-prefix)
       opticks-optix-install-dir  :  $(opticks-optix-install-dir)
       g4-cmake-dir               :  $(g4-cmake-dir)
       xercesc-library            :  $(xercesc-library)
       xercesc-include-dir        :  $(xercesc-include-dir)

EOI
}


opticks-cmakecache(){ echo $(opticks-bdir)/CMakeCache.txt ; }
opticks-cmakecache-grep(){ grep ${1:-COMPUTE_CAPABILITY} $(opticks-cmakecache) ; }
opticks-cmakecache-vars-(){  cat << EOV
CMAKE_BUILD_TYPE
COMPUTE_CAPABILITY
CMAKE_INSTALL_PREFIX
OptiX_INSTALL_DIR
Geant4_DIR
XERCESC_LIBRARY
XERCESC_INCLUDE_DIR
EOV
}

opticks-cmakecache-vars(){ 
   local var 
   $FUNCNAME- | while read var ; do
       opticks-cmakecache-grep $var 
   done    
}


opticks-cmake(){
   local msg="=== $FUNCNAME : "
   local iwd=$PWD
   local bdir=$(opticks-bdir)

   echo $msg configuring installation

   mkdir -p $bdir
   [ -f "$bdir/CMakeCache.txt" ] && echo $msg configured already use opticks-configure to wipe build dir and re-configure && return  

   opticks-bcd

   g4- 
   xercesc-

   opticks-cmake-info 

   cmake \
        -G "$(opticks-cmake-generator)" \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCOMPUTE_CAPABILITY=$(opticks-compute-capability) \
       -DCMAKE_INSTALL_PREFIX=$(opticks-prefix) \
       -DOptiX_INSTALL_DIR=$(opticks-optix-install-dir) \
       -DGeant4_DIR=$(g4-cmake-dir) \
       -DXERCESC_LIBRARY=$(xercesc-library) \
       -DXERCESC_INCLUDE_DIR=$(xercesc-include-dir) \
       $* \
       $(opticks-sdir)

   cd $iwd
}

opticks-cmake-modify-ex1(){
  local msg="=== $FUNCNAME : "
  local bdir=$(opticks-bdir)
  local bcache=$bdir/CMakeCache.txt
  [ ! -f "$bcache" ] && echo $msg requires a preexisting $bcache from prior opticks-cmake run && return 
  opticks-bcd
  g4-
  xercesc- 

  cmake \
       -DGeant4_DIR=$(g4-cmake-dir) \
       -DXERCESC_LIBRARY=$(xercesc-library) \
       -DXERCESC_INCLUDE_DIR=$(xercesc-include-dir) \
          . 
}

opticks-cmake-modify-ex2(){
  local msg="=== $FUNCNAME : "
  local bdir=$(opticks-bdir)
  local bcache=$bdir/CMakeCache.txt
  [ ! -f "$bcache" ] && echo $msg requires a preexisting $bcache from prior opticks-cmake run && return 
  opticks-bcd

  cmake \
       -DCOMPUTE_CAPABILITY=$(opticks-compute-capability) \
          . 
}

opticks-cmake-modify-ex3(){

  local msg="=== $FUNCNAME : "
  local bdir=$(opticks-bdir)
  local bcache=$bdir/CMakeCache.txt
  [ ! -f "$bcache" ] && echo $msg requires a preexisting $bcache from prior opticks-cmake run && return 
  opticks-bcd

  echo $msg opticks-cmakecache-vars BEFORE MODIFY 
  opticks-cmakecache-vars 

  cmake \
       -DOptiX_INSTALL_DIR=/Developer/OptiX_380 \
       -DCOMPUTE_CAPABILITY=30 \
          . 

  echo $msg opticks-cmakecache-vars AFTER MODIFY 
  opticks-cmakecache-vars 

}




opticks-wipe(){
  local msg="=== $FUNCNAME : "
   local bdir=$(opticks-bdir)
   echo $msg wiping build dir $bdir
   rm -rf $bdir
}

opticks-configure()
{
   opticks-wipe

   case $(opticks-cmake-generator) in
       "Visual Studio 14 2015") opticks-configure-local-boost $* ;;
                             *) opticks-configure-system-boost $* ;;
   esac
}

opticks-configure-system-boost()
{
   opticks-cmake $* 
}

opticks-configure-local-boost()
{
    local msg="=== $FUNCNAME :"
    boost-

    local prefix=$(boost-prefix)
    [ ! -d "$prefix" ] && type $FUNCNAME && return  
    echo $msg prefix $prefix

    opticks-cmake \
              -DBOOST_ROOT=$prefix \
              -DBoost_USE_STATIC_LIBS=1 \
              -DBoost_USE_DEBUG_RUNTIME=0 \
              -DBoost_NO_SYSTEM_PATHS=1 \
              -DBoost_DEBUG=0 

    # vi $(cmake-find-package Boost)
}


  


opticks_config_cflags()
{
    echo -I
}
opticks_config_libs()
{
    echo -L
}

opticks_config()
{
   local arg
   for arg in $* ; do 
       case $arg in
          --cflags)  opticks_config_cflags ;; 
          --libs)    opticks_config_libs   ;; 
       esac
   done
}


#opticks-config-type(){ echo Debug ; }
opticks-config-type(){ echo RelWithDebInfo ; }
opticks--(){     

   local msg="$FUNCNAME : "
   local iwd=$PWD

   local bdir=$1
   shift
   [ -z "$bdir" -o "$bdir" == "." ] && bdir=$(opticks-bdir) 
   [ ! -d "$bdir" ] && echo $msg bdir $bdir does not exist && return 

   cd $bdir

   cmake --build . --config $(opticks-config-type) --target ${1:-install}

   cd $iwd
}



opticks-prepare-installcache()
{
    local msg="=== $FUNCNAME :"
    echo $msg generating RNG seeds into installcache 

    cudarap-
    cudarap-prepare-installcache

    OpticksPrepareInstallCache  
}

opticks-check-installcache()
{
    local msg="=== $FUNCNAME :"
    local rc=0
    local iwd=$PWD
    local dir=$(opticks-installcachedir)
    if [ ! -d "$dir" ]; then
        echo $msg missing dir $dir 
        rc=100
    else
        cd $dir
        [ ! -d "$dir/PTX" ] && echo $msg $PWD : missing PTX && rc=101
        [ ! -d "$dir/RNG" ] && echo $msg $PWD : missing RNG && rc=102
        [ ! -d "$dir/OKC" ] && echo $msg $PWD : missing OKC && rc=103
    fi
    cd $iwd
    return $rc
}


opticks-ti(){ opticks-t- $* --interactive-debug-mode 1 ; }
opticks-t(){  opticks-t- $* --interactive-debug-mode 0 ; }



opticks-t-()
{
   # 
   # Basic environment (PATH and envvars to find data) 
   # should happen at profile level (or at least be invoked from there) 
   # not here (and there) for clarity of a single location 
   # where smth is done.
   #
   # Powershell presents a challenge to this principal,
   # TODO:find a cross platform way of doing envvar setup 
   #
   #
   local msg="=== $FUNCNAME : "
   local iwd=$PWD

   local rc=0
   opticks-check-installcache 
   rc=$?
   [ "$rc" != "0" ] && echo $msg ABORT : missing installcache components : create with opticks-prepare-installcache && return $rc


   ## if 1st arg is a directory, cd there to run ctest     
   ## otherwise run from the top level bdir
   local arg=$1
   if [ "${arg:0:1}" == "/" -a -d "$arg" ]; then
       bdir=$arg
       shift
   else
       bdir=$(opticks-bdir) 
   fi


   cd $bdir

   local log=ctest.log
   #opticks-t-- $*

   date          | tee $log
   ctest $* 2>&1 | tee -a $log
   date          | tee -a $log

   cd $iwd
   echo $msg use -V to show output, ctest output written to $bdir/ctest.log
}


opticks-tl()
{
   local arg=$1
   if [ "${arg:0:1}" == "/" -a -d "$arg" ]; then
       bdir=$arg
       shift
   else
       bdir=$(opticks-bdir) 
   fi
   ls -l $bdir/ctest.log
   cat $bdir/ctest.log
}


opticks-ts()
{
   ## list tests taking longer than 1 second
   local arg=$1
   if [ "${arg:0:1}" == "/" -a -d "$arg" ]; then
       bdir=$arg
       shift
   else
       bdir=$(opticks-bdir) 
   fi
   #perl -n -e 'm,[123456789]\.\d{2} sec, && print  ' $bdir/ctest.log   ## this missins 10.00 20.00
   grep " sec" $bdir/ctest.log | grep -v " 0.* sec" - 

}



opticks-t--()
{
   [ "$(which ctest 2>/dev/null)" == "" ] && ctest3 $* || ctest $*
}


opticks-lfind(){ opticks-find $1 -l ; }
opticks-f(){ opticks-find $* ; }
opticks-find(){
   local str=${1:-ENV_HOME}
   local opt=${2:--H}

   local iwd=$PWD
   opticks-scd

   find . -name '*.sh' -exec grep $opt $str {} \;
   find . -name '*.bash' -exec grep $opt $str {} \;
   find . -name '*.cu' -exec grep $opt $str {} \;
   find . -name '*.cc' -exec grep $opt $str {} \;
   find . -name '*.hh' -exec grep $opt $str {} \;
   find . -name '*.cpp' -exec grep $opt $str {} \;
   find . -name '*.hpp' -exec grep $opt $str {} \;
   find . -name '*.h' -exec grep $opt $str {} \;
   find . -name '*.txt' -exec grep $opt $str {} \;
   find . -name '*.py' -exec grep $opt $str {} \;

   #cd $iwd
}


opticks-unset--()
{
   local pfx=${1:-OPTICKS_}
   local kv
   local k
   local v
   env | grep $pfx | while read kv ; do 

       k=${kv/=*}
       v=${kv/*=}

       #printf "%50s %s \n" $k $v  
       echo unset $k 
   done
}

opticks-unset-()
{
   opticks-unset-- OPTICKS_
   opticks-unset-- DAE_
   opticks-unset-- IDPATH
}
opticks-unset()
{
   local tmp=/tmp/unset.sh
   opticks-unset- >  $tmp

   echo unset with : . $tmp
}




# below need to be precursor names
opticks-all-projs-(){ cat << EOP
sysrap
brap
npy
okc
ggeo
assimprap
openmeshrap
okg
oglrap

cudarap
thrap
oxrap
okop
opticksgl

ok
cfg4
okg4
EOP
}


opticks-cuda-projs-(){ cat << EOP
cudarap
thrap
oxrap
okop
opticksgl
EOP
}


opticks---(){ 
   local arg=${1:-all}
   local proj
   opticks-${arg}-projs- | while read proj ; do
      [ -z "$proj" ] && continue  
      $proj-
      $proj--
   done
} 

opticks----(){ 
   ## proj--- touches the API header and then does $proj-- : this forcing recompilation of everything 
   local arg=${1:-all}
   local proj
   
   opticks-${arg}-projs- | while read proj ; do
      [ -z "$proj" ] && continue  
      $proj-
      echo proj $proj
      $proj---
   done

} 

opticks-list()
{
   local arg=${1:-all}
   local proj
   opticks-${arg}-projs- | while read proj ; do
      [ -z "$proj" ] && continue  
      echo proj $proj
   done
}



opticks-log(){ find $(opticks-home) -name '*.log' ; }
opticks-rmlog(){ find $(opticks-home) -name '*.log' -exec rm -f {} \; ; }
opticks-nuclear(){   rm -rf $LOCAL_BASE/opticks/* ; }
opticks-distclean(){ opticks-rmdirs- bin build gl include lib ptx  ; }
opticks-fullclean(){ opticks-rmdirs- bin build gl include lib ptx externals  ; }
opticks-rmdirs-(){
   local base=$(opticks-dir)
   local msg="# $FUNCNAME : "
   echo $msg pipe to sh to do the deletion
   local name
   for name in $*
   do 
      local dir=$base/$name
      [ -d "$dir" ] && echo rm -rf $dir ;
   done
}

opticks-cleanbuild()
{
   opticks-distclean 
   opticks-distclean | sh 
   opticks-full 
}


opticks-make-()
{
    local iwd=$PWD
    local bdir=$1
    shift
    cd $bdir
    make $*
    cd $iwd
}


########## runtime setup ########################

opticks-path(){ echo $PATH | tr ":" "\n" ; }
opticks-path-add(){
  local dir=$1 
  #[ ! -d "$dir" ] && return  
  [ "${PATH/$dir}" == "${PATH}" ] && export PATH=$dir:$PATH
}

opticks-export()
{
   opticks-path-add $(opticks-prefix)/lib
   opticks-path-add $(opticks-home)/bin
   opticks-path-add $(opticks-home)/ana

   opticksdata-
   opticksdata-export

   case $(uname -s) in
      MINGW*) opticks-export-mingw ;;
   esac
}
opticks-export-mingw()
{
  local dirs="lib externals/bin externals/lib"
  local dir
  for dir in $dirs 
  do
      opticks-path-add $(opticks-prefix)/$dir
  done 

  # see brap-/fsutil
  export OPTICKS_PATH_PREFIX="C:\\msys64" 
}



########## building opticks docs 

opticks-bb(){      opticks-open  http://simoncblyth.bitbucket.io/opticks/index.html ; } 
opticks-rdocs(){   opticks-open  http://simoncblyth.bitbucket.io/opticks/index.html ; } 
opticks-rnotes(){  opticks-open  http://simoncblyth.bitbucket.io/opticks_notes/index.html ; } 
opticks-docs(){    opticks-open  $(opticks-docs-htmldir)/index.html ; } 
opticks-docs-htmldir(){ 
   local htmldirbb=$HOME/simoncblyth.bitbucket.org/opticks 
   [ -d "$htmldirbb" ] && echo $htmldirbb || echo $(opticks-prefix)/html 
}
opticks-docs-make()
{
   local iwd=$PWD
   opticks-scd
   sphinx-build -b html  . $(opticks-docs-htmldir)
   cd $iwd 
   opticks-docs

   echo $msg publish the docs via bitbucket commit/push from $(opticks-docs-htmldir)

}

########## building opticks dev notes




opticks-notes-cd(){ cd $(opticks-home)/notes/issues/$1 ; }

opticks-notes-image(){
    find $(opticks-home)/notes -name '*.rst' -exec grep -H image {} \;
}


opticks-notes-notes(){ cat << EON

Planted a top level link::

   simon:/ blyth$ sudo ln -s /Users/blyth/simoncblyth.bitbucket.org/env 

To allow default file links like the below to resolve from 
local html::

   file:///env/graphics/ggeoview/issues/offset_bottom/dpib.png


EON
}

opticks-notes(){ opticks-open  $(opticks-notes-htmldir)/index.html ; } 
opticks-notes-htmldir(){ 
   local htmldirbb=$HOME/simoncblyth.bitbucket.org/opticks_notes 
   [ -d "$htmldirbb" ] && echo $htmldirbb || echo $(opticks-prefix)/notes/html 
}
opticks-notes-make()
{
   local iwd=$PWD
   opticks-scd "notes"
   sphinx-build -b html  . $(opticks-notes-htmldir)
   cd $iwd 
   opticks-notes
}

##############



opticks-open()
{
  local url=$1
  case $(uname) in
      Linux) firefox $url ;;
     Darwin) open $url ;;
      MING*) chrome $url ;;
  esac  
}

### opticks projs ###  **moved** all projs into top level folders

sysrap-(){          . $(opticks-home)/sysrap/sysrap.bash && sysrap-env $* ; }
brap-(){            . $(opticks-home)/boostrap/brap.bash && brap-env $* ; }
npy-(){             . $(opticks-home)/opticksnpy/npy.bash && npy-env $* ; }
okc-(){             . $(opticks-home)/optickscore/okc.bash && okc-env $* ; }

ggeo-(){            . $(opticks-home)/ggeo/ggeo.bash && ggeo-env $* ; }
assimprap-(){       . $(opticks-home)/assimprap/assimprap.bash && assimprap-env $* ; }
openmeshrap-(){     . $(opticks-home)/openmeshrap/openmeshrap.bash && openmeshrap-env $* ; }
okg-(){             . $(opticks-home)/opticksgeo/okg.bash && okg-env $* ; }

oglrap-(){          . $(opticks-home)/oglrap/oglrap.bash && oglrap-env $* ; }
cudarap-(){         . $(opticks-home)/cudarap/cudarap.bash && cudarap-env $* ; }
thrap-(){           . $(opticks-home)/thrustrap/thrap.bash && thrap-env $* ; }
oxrap-(){           . $(opticks-home)/optixrap/oxrap.bash && oxrap-env $* ; }

okop-(){            . $(opticks-home)/okop/okop.bash && okop-env $* ; }
opticksgl-(){       . $(opticks-home)/opticksgl/opticksgl.bash && opticksgl-env $* ; }
ok-(){              . $(opticks-home)/ok/ok.bash && ok-env $* ; }
cfg4-(){            . $(opticks-home)/cfg4/cfg4.bash && cfg4-env $* ; }
okg4-(){            . $(opticks-home)/okg4/okg4.bash && okg4-env $* ; }
ana-(){             . $(opticks-home)/ana/ana.bash && ana-env $*  ; }
cfh-(){             . $(opticks-home)/ana/cfh.bash && cfh-env $*  ; }
tests-(){           . $(opticks-home)/tests/tests.bash && tests-env $*  ; }
tools-(){           . $(opticks-home)/tools/tools.bash && tools-env $*  ; }
notes-(){           . $(opticks-home)/notes/notes.bash && notes-env $*  ; }
pmt-(){             . $(opticks-home)/ana/pmt/pmt.bash && pmt-env $* ; }
# override old original from env, $(env-home)/nuwa/detdesc/pmt/pmt.bash


### opticks launchers ########

oks-(){             . $(opticks-home)/bin/oks.bash && oks-env $* ; }
ggv-(){             . $(opticks-home)/bin/ggv.bash && ggv-env $* ; }
vids-(){            . $(opticks-home)/bin/vids.bash && vids-env $* ; }
op-(){              . $(opticks-home)/bin/op.sh ; }

#### opticks top level tests ########

tviz-(){       . $(opticks-home)/tests/tviz.bash      && tviz-env $* ; }
tpmt-(){       . $(opticks-home)/tests/tpmt.bash      && tpmt-env $* ; }
trainbow-(){   . $(opticks-home)/tests/trainbow.bash  && trainbow-env $* ; }
tnewton-(){    . $(opticks-home)/tests/tnewton.bash   && tnewton-env $* ; }
tprism-(){     . $(opticks-home)/tests/tprism.bash    && tprism-env $* ; }
tbox-(){       . $(opticks-home)/tests/tbox.bash      && tbox-env $* ; }
treflect-(){   . $(opticks-home)/tests/treflect.bash  && treflect-env $* ; }
twhite-(){     . $(opticks-home)/tests/twhite.bash    && twhite-env $* ; }
tlens-(){      . $(opticks-home)/tests/tlens.bash     && tlens-env $* ; }
tg4gun-(){     . $(opticks-home)/tests/tg4gun.bash    && tg4gun-env $* ; }
tlaser-(){     . $(opticks-home)/tests/tlaser.bash    && tlaser-env $* ; }
tboxlaser-(){  . $(opticks-home)/tests/tboxlaser.bash && tboxlaser-env $* ; }
tdefault-(){   . $(opticks-home)/tests/tdefault.bash  && tdefault-env $* ; }
tconcentric-(){   . $(opticks-home)/tests/tconcentric.bash  && tconcentric-env $* ; }
tboolean-(){      . $(opticks-home)/tests/tboolean.bash  && tboolean-env $* ; }
tboolean-bib-(){      . $(opticks-home)/tests/tboolean-bib.bash  && tboolean-bib-env $* ; }
tjuno-(){      . $(opticks-home)/tests/tjuno.bash  && tjuno-env $* ; }
tgltf-(){         . $(opticks-home)/tests/tgltf.bash  && tgltf-env $* ; }

####### below functions support analysis on machines without a full opticks install
####### by copying some parts of an opticks install to corresponding local locations 

opticks-host(){ echo ${OPTICKS_HOST:-192.168.1.101} ; }
opticks-user(){ echo ${OPTICKS_USER:-$USER} ; }

opticks-scp(){
   local msg="=== $FUNCNAME : "
   local host=$(opticks-host)
   local user=$(opticks-user)
   local src=$1
   local dst=$2
   mkdir -p $(dirname $dst) 

   [ -d "$dst" ] && echo $msg dst $dst exists already && return 

   local cmd="scp -r $user@$host:$src $(dirname $dst)"
   echo $msg \"$cmd\"
   
   local ans
   read -p "$msg Proceed with the above command ? [Yy] "  ans
   if [ "$ans" == "Y" -o "$ans" == "y" ]; then
       echo $msg OK PROCEEDING
       eval $cmd
   else
       echo $msg OK SKIP
   fi
}


opticks-prefix-ref(){ echo ${OPTICKS_PREFIX_REF:-/home/simonblyth/local/opticks} ; }
opticks-prefix-loc(){ echo  $HOME/local/opticks ; }

opticks-okc-ref(){    echo $(opticks-prefix-ref)/installcache/OKC ; }
opticks-okc-loc(){    echo $(opticks-prefix-loc)/installcache/OKC ; }

opticks-geo-ref(){ echo $(opticks-prefix-ref)/opticksdata/export/$(opticks-geopath-ref) ; }
opticks-geo-loc(){ echo $(opticks-prefix-loc)/opticksdata/export/$(opticks-geopath-ref) ; }

opticks-geopath-ref(){ echo DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae ; }


opticks-geo-get(){
   local user=${1:-$USER}
   [ ! -d "$(opticks-prefix-loc)/opticksdata" ] && echo YOU NEED TO optickdata-get FIRST && return 1

   opticks-scp $(opticks-geo-ref) $(opticks-geo-loc) 
   opticks-scp $(opticks-okc-ref) $(opticks-okc-loc)    
}

opticks-evt-ref(){ echo ${OPTICKS_EVT_REF:-/tmp/simonblyth/opticks/evt} ; }
opticks-evt-loc(){ echo /tmp/$USER/opticks/evt ; }
opticks-evt-get()
{
    local subd=${1:-reflect}
    opticks-scp $(opticks-evt-ref)/$subd $(opticks-evt-loc)/$subd 
}


opticks-analysis-only-setup()
{
   opticksdata-
   opticksdata-get

   opticks-geo-get

   opticks-evt-get rainbow
}

opticks-analysis-only-check()
{
   cat << EOC

   PATH : $PATH 
 
       This needs to include directories where the tools are installed, eg 

           hg (Mercurial)
           python
           ipython

   LOCAL_BASE : $LOCAL_BASE   

       Should be \$HOME/local ie $HOME/local 
       This is used by opticksdata- functions to 
       identify the location for the opticksdata repository clone

   PYTHONPATH : $PYTHONPATH   

       Should be same as \$HOME ie $HOME
       This allows you to : 
           python -c "import opticks" 

EOC

}


opticks-lib-ext()
{
   case $(uname) in 
     Linux) echo so ;; 
     Darwin) echo dylib ;; 
   esac
}


opticks-lib-ls()
{
    local ext=$(opticks-lib-ext)

    cd $(opticks-prefix)/lib
    ls -1 *.$ext | perl -pe "s/.$ext//" - | perl -pe "s/^lib//" - 



}


opticks-find-types()
{
    opticks-find Types.hpp
}

opticks-find-typ()
{
    opticks-find Typ.hpp
}

opticks-find-flags()
{
    opticks-find OpticksFlags.hh  
}




opticks-cls () 
{ 
    opticks-cls- "." $*
}
opticks-cls-() 
{ 
    local iwd=$PWD;
    opticks-scd;
    local base=${1:-.};
    local name=${2:-DsG4OpBoundaryProcess};
    local h=$(find $base -name "$name.h");
    local hh=$(find $base -name "$name.hh");
    local cc=$(find $base -name "$name.cc");
    local icc=$(find $base -name "$name.icc");
    local vcmd="vi -R $h $hh $icc $cc";
    echo $vcmd;
    eval $vcmd;
    cd $iwd
}


opticks-cmake-vi-(){ cat << EOF
CMakeLists.txt
cmake/Modules/OpticksConfigureConfigScript.cmake
cmake/Templates/opticks-config.in
cmake/Modules/OpticksConfigureCMakeHelpers.cmake
cmake/Templates/OpticksConfig.cmake.in
EOF
}

opticks-cmake-vi(){ opticks-scd ; vi $(opticks-cmake-vi-) ; }

