##
## Copyright (c) 2019 Opticks Team. All Rights Reserved.
##
## This file is part of Opticks
## (see https://bitbucket.org/simoncblyth/opticks).
##
## Licensed under the Apache License, Version 2.0 (the "License"); 
## you may not use this file except in compliance with the License.  
## You may obtain a copy of the License at
##
##   http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software 
## distributed under the License is distributed on an "AS IS" BASIS, 
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
## See the License for the specific language governing permissions and 
## limitations under the License.
##

opticks-(){         source $(opticks-source) && opticks-env $* ; }
opticks-source(){   echo $BASH_SOURCE ; }
opticks-ldir(){     echo $(dirname $BASH_SOURCE) ; }
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
   . $(opticks-ldir)/externals/externals.bash       ## just precursors
   . $(opticks-ldir)/integration/integration.bash   ## just precursors
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


opticks-metadata-export()
{
    onvidia-
    onvidia-export
}





olocal-()
{
   echo -n # transitional standin for olocal-
}

opticks-home-default(){ echo $(dirname $(opticks-source)) ; }
opticks-home(){   echo ${OPTICKS_HOME:-$(opticks-home-default)} ; }  ## input from profile 
opticks-name(){   basename $(opticks-home) ; }





opticks-tboolean-shortcuts(){ 


   : **simulate** : aligned bi-simulation creating OK+G4 events 
   ts(){  LV=$1 tboolean.sh ${@:2} ; } 

   : **visualize** : load events and visualize the propagation
   tv(){  LV=$1 tboolean.sh --load ${@:2} ; } 

   : **visualize** the geant4 propagation 
   tv4(){  LV=$1 tboolean.sh --load --vizg4 ${@:2} ; } 

   : **analyse interactively** : load events and analyse the propagation in ipython
   ta(){  LV=$1 tboolean.sh --ip ${@:2} ; } 

   : **analyse** : load events and run python analysis script the propagation
   tp(){  LV=$1 tboolean.sh --py ${@:2} ; } 

}


opticks-shared-cache-prefix-default(){ echo $HOME/.opticks ; }
opticks-user-cache-prefix-default(){   echo $HOME/.opticks ; }



opticks-shared-cache-prefix(){ echo ${OPTICKS_SHARED_CACHE_PREFIX:-$(opticks-shared-cache-prefix-default)} ; } 
opticks-user-cache-prefix(){   echo ${OPTICKS_USER_CACHE_PREFIX:-$(opticks-user-cache-prefix-default)} ; } 


opticks-geocachedir(){ echo $(opticks-shared-cache-prefix)/geocache ; } 
opticks-rngcachedir(){ echo $(opticks-shared-cache-prefix)/rngcache ; }
opticks-rngdir(){      echo $(opticks-rngcachedir)/RNG ; }
opticks-rngdir-cd(){   cd $(opticks-rngdir) ; }


opticks-cache-info(){ cat << EON
$FUNCNAME
=====================

    opticks-prefix                      : $(opticks-prefix)
    opticks-installcachedir             : $(opticks-installcachedir)


    OPTICKS_SHARED_CACHE_PREFIX         : $OPTICKS_SHARED_CACHE_PREFIX
    OPTICKS_USER_CACHE_PREFIX           : $OPTICKS_USER_CACHE_PREFIX


    opticks-shared-cache-prefix-default : $(opticks-shared-cache-prefix-default)
    opticks-shared-cache-prefix         : $(opticks-shared-cache-prefix)

    opticks-geocachedir                 : $(opticks-geocachedir)
    opticks-rngcachedir                 : $(opticks-rngcachedir)

    opticks-rngdir                      : $(opticks-rngdir)


    opticks-user-cache-prefix-default   : $(opticks-user-cache-prefix-default)
    opticks-user-cache-prefix           : $(opticks-user-cache-prefix)


shared-cache-prefix
   geocache, rngcache

user-cache-prefix
   runcache
  

Changes to the cache prefix layouts or envvars names 
need to be done in triplicate in bash/py/C++ in::

    opticks.bash 
    ana/geocache.bash
    ana/key.py
    boostrap/BOpticksResource.cc

EON
}




opticks-id(){ cat << EOI

  opticks-home   : $(opticks-home)
  opticks-prefix : $(opticks-prefix)
  opticks-prefix-tmp : $(opticks-prefix-tmp)
  opticks-name   : $(opticks-name)

EOI
}

opticks-cmakecache(){ echo $(opticks-bdir)/CMakeCache.txt ; }

opticks-pretty(){  cat ${1:-some.json} | python -m json.tool ; }

opticks-key2idpath(){ local dir=$(OpticksIDPATH --envkey --fatal 2>&1) ; echo $dir ; } 
opticks-kcd(){  local dir=$(opticks-key2idpath) && cd $dir && pwd && echo OPTICKS_KEY=$OPTICKS_KEY ; }

opticks-key2idpath-notes(){ cat << EON

[blyth@localhost opticks]$ OPTICKS_KEY=OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.699463ea0065185a7ffaf10d4935fc61 opticks-key2idpath
/home/blyth/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/699463ea0065185a7ffaf10d4935fc61/1
[blyth@localhost opticks]$ l /home/blyth/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/699463ea0065185a7ffaf10d4935fc61/1/
total 8
-rw-rw-r--.  1 blyth blyth  177 May 10 16:13 cachemeta.json
drwxrwxr-x. 42 blyth blyth 4096 May 10 16:13 GMeshLib
drwxrwxr-x.  8 blyth blyth   60 May 10 16:13 GParts
drwxrwxr-x.  8 blyth blyth   60 May 10 16:13 GMergedMesh
drwxrwxr-x.  2 blyth blyth   30 May 10 15:34 GBndLib

EON
}




opticks-idfold(){ echo $(dirname $IDPATH) ; }
opticks-srcpath(){ echo $(opticks-idpath2srcpath $IDPATH) ; }
opticks-srcfold(){ echo $(dirname $(opticks-srcpath)) ; }
#opticks-srcextras(){ echo $(opticks-idfold)/extras ; }   # layout 0
opticks-srcextras(){ echo $(opticks-srcfold)/extras ; }  # layout 1

opticks-join(){ local ifs=$IFS ; IFS="$1"; shift; echo "$*" ; IFS=$ifs ;  }

opticks-paths(){ cat << EON

$FUNCNAME
===============================

NB THIS IS THE DEPRECATED OLD WAY OF DOING THINGS

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


   opticks-tbool-path 0 : $(opticks-tbool-path 0)
   opticks-nnt-path 0   : $(opticks-nnt-path 0)


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
#opticks-fold-tmp(){  echo /tmp/local/$(opticks-name) ; } 
opticks-fold-tmp(){  echo ${LOCAL_BASE}/$(opticks-name)-tmp ; } 


opticks-sdir(){   echo $(opticks-home) ; }
opticks-scd(){  cd $(opticks-sdir)/$1 ; }
opticks-ncd(){  opticks-scd notes/issues ;  }

opticks-buildtype(){ echo Debug  ; }
opticks-prefix(){ echo $(opticks-fold)  ; }
opticks-prefix-tmp(){ echo $(opticks-fold-tmp)  ; }

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



opticks-optix-install-dir(){ echo ${OPTICKS_OPTIX_INSTALL_DIR:-$(opticks-prefix)/externals/OptiX} ; }


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

opticks-externals(){ 
: emits to stdout the names of the bash precursors that download and install the externals 
  cat << EOL
bcm
glm
glfw
glew
gleq
imgui
assimp
openmesh
plog
opticksaux
oimplicitmesher
odcs
oyoctogl
ocsgbsp
xercesc
g4
EOL
}

opticks-optionals(){ cat << EOL
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




opticks-cmake-version(){  cmake --version | perl -ne 'm/(\d*\.\d*\.\d*)/ && print "$1" ' - ; }
opticks-externals-info(){  cat << EOI
$FUNCNAME
============================

    opticks-cmake-version  : $(opticks-cmake-version)

EOI
}

opticks-externals-install(){ echo $FUNCNAME ; opticks-externals | -opticks-installer ; }
opticks-externals-url(){     echo $FUNCNAME ; opticks-externals | -opticks-url ; }
opticks-externals-dist(){    echo $FUNCNAME ; opticks-externals | -opticks-dist ; }
opticks-externals-dir(){     echo $FUNCNAME ; opticks-externals | -opticks-dir ; }
opticks-externals-status(){  echo $FUNCNAME ; opticks-externals | -opticks-status ; }

opticks-optionals-install(){ echo $FUNCNAME ; opticks-optionals | -opticks-installer ; }
opticks-optionals-url(){     echo $FUNCNAME ; opticks-optionals | -opticks-url ; }
opticks-optionals-dist(){    echo $FUNCNAME ; opticks-optionals | -opticks-dist ; }

opticks-possibles-install(){ echo $FUNCNAME ; opticks-possibles | -opticks-installer ; }
opticks-possibles-url(){     echo $FUNCNAME ; opticks-possibles | -opticks-url ; }
opticks-possibles-dist(){    echo $FUNCNAME ; opticks-possibles | -opticks-dist ; }


# for finding system boost 
opticks-boost-includedir(){ echo ${OPTICKS_BOOST_INCLUDEDIR:-/tmp} ; }
opticks-boost-libdir(){     echo ${OPTICKS_BOOST_LIBDIR:-/tmp} ; }
opticks-boost-info(){ cat << EOI
$FUNCNAME
===================

   opticks-boost-includedir : $(opticks-boost-includedir)
   opticks-boost-libdir     : $(opticks-boost-libdir)

EOI
}

opticks-cmake-generator(){ om- ; om-cmake-generator ; } 

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

    #opticks-configure
    #opticks--

    om-
    cd $(om-home)
    om-install


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

-opticks-dir(){
   local ext
   local dir
   while read ext 
   do
        $ext-
        dir=$($ext-dir 2>/dev/null)
        printf "%30s :  %s \n" $ext $dir
   done
}
-opticks-status(){
   local ext
   local dir
   local iwd=$PWD 
   while read ext 
   do
        $ext-
        dir=$($ext-dir 2>/dev/null)
        printf "\n\n%30s :  %s \n" $ext $dir
        if [ -d "$dir" ]; then
            cd $dir 
            [ -d ".hg" ]  && hg paths -v && hg status . 
            [ -d ".git" ] && git remote -v && git status . 
        else
            echo no such dir : maybe system repo install 
        fi 
   done
   cd $iwd
}


opticks-g4-clean-build()
{
    local arg="extg4:"

    om-
    om-subs $arg 
    om-clean $arg 

    type $FUNCNAME
    read -p "$FUNCNAME : enter YES to proceed to to clean and build : " ans

    [ "$ans" != "YES" ] && echo skip && return 

    om-clean $arg | sh 
    om-conf $arg
    om-make $arg
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
   opticks-externals-info
   opticks-locations
   opticks-env-info
   opticks-externals-url
   opticks-externals-dist
   opticks-optionals-url
   opticks-optionals-dist
}




opticks-wipe(){
  local msg="=== $FUNCNAME : "
   local bdir=$(opticks-bdir)
   echo $msg wiping build dir $bdir
   rm -rf $bdir
}


opticks--(){     
   local bdir=$1
   if [ "$bdir" == "" ]; then
      bdir=$(opticks-home) 
   fi 
   shift 
   local iwd=$(pwd)
   cd $bdir
   om- 
   om-make $1

   cd $iwd 
}



opticks-prepare-installcache-notes(){ cat << EON

$FUNCNAME
===================================

Name of the function becoming vestigial.


Only PTX remains in the installcache.

OKC has been eliminated, the below should NOT be run::

    OpticksPrepareInstallCacheTest '$INSTALLCACHE_DIR/OKC'

RNG has been relocated from the installcache to the rngcache 
which is positioned under OPTICKS_SHARED_CACHE_PREFIX

The default RNG dir is ~/.opticks/rngcache/RNG but this is expected
to be moved to a shared location, eg for use on a GPU cluster, 
using envvar OPTICKS_SHARED_CACHE_PREFIX which positions 
the RNG dir at $OPTICKS_SHARED_CACHE_PREFIX/rngcache/RNG

EON
}


opticks-prepare-installcache()
{
    local msg="=== $FUNCNAME :"
    echo $msg generating RNG seeds into installcache 

    cudarap-
    cudarap-prepare-rng
    cudarap-check-rng
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
        [ ! -d "PTX" ] && echo $msg $PWD : missing PTX : compiled OptiX programs created when building oxrap-  && rc=101
    fi

    cd $iwd
    return $rc
}


opticks-t(){  opticks-t- $* ; }   ## see om-test-one for details of ctest arguments
opticks-t0(){ CUDA_VISIBLE_DEVICES=0 opticks-t $* ; }
opticks-t1(){ CUDA_VISIBLE_DEVICES=1 opticks-t $* ; }

opticks-i(){  
   local iwd=$PWD
   om-
   local bdir=$(om-bdir integration)
   cd $bdir
   om-test -V
   cd $iwd
}



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

   om-
   om-test 

   cd $iwd
}

opticks-t-notes(){ cat << EON
$FUNCNAME
=====================

*opticks-t-* is invoked by the subproj test functions such as *oxrap-t* 
with the corresponding build dir as the first argument.


EON
}


opticks-tl()
{
   om-
   om-testlog
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


opticks-ifind(){ opticks-find "$1" -Hi ; }
opticks-findl(){ opticks-find "$1" -l ; }
opticks-find(){
   local str="${1:-ENV_HOME}"
   local opt=${2:--H}

   local iwd=$PWD
   opticks-scd

   find . -name '*.sh' -exec grep $opt "$str" {} \;
   find . -name '*.bash' -exec grep $opt "$str" {} \;
   find . -name '*.cu' -exec grep $opt "$str" {} \;
   find . -name '*.cc' -exec grep $opt "$str" {} \;
   find . -name '*.hh' -exec grep $opt "$str" {} \;
   find . -name '*.cpp' -exec grep $opt "$str" {} \;
   find . -name '*.hpp' -exec grep $opt "$str" {} \;
   find . -name '*.h' -exec grep $opt "$str" {} \;
   find . -name '*.txt' -exec grep $opt "$str" {} \;
   find . -name '*.py' -exec grep $opt "$str" {} \;
   find . -name '*.cmake' -exec grep $opt "$str" {} \;

   #cd $iwd
}



opticks-if(){ opticks-f "$1" -Hi ; }   
opticks-fl(){ opticks-f "$1" -l ; }   
opticks-f(){   
   local str="${1:-ENV_HOME}"
   local opt=${2:--H}

   local iwd=$PWD
   opticks-scd

   find . \
        \( \
       -name '*.sh' -or \
       -name '*.bash' -or \
       -name '*.cu' -or \
       -name '*.cc' -or \
       -name '*.hh' -or \
       -name '*.cpp' -or \
       -name '*.hpp' -or \
       -name '*.h' -or \
       -name '*.txt' -or \
       -name '*.cmake' -or \
       -name '*.py' \
        \) \
       -exec grep $opt "$str" {} \;

}


opticks-c(){   
   local str="${1:-ENV_HOME}"
   local opt=${2:--H}

   local iwd=$PWD
   opticks-scd

   find . \
        \( \
       -name '*.cc' -or \
       -name '*.hh' -or \
       -name '*.cpp' -or \
       -name '*.hpp' -or \
       -name '*.h' \
        \) \
       -exec grep $opt "$str" {} \;
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
asirap
openmeshrap
okg
oglrap

cudarap
thrap
oxrap
okop
okgl

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
okgl
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


opticks-llp-(){ 
    local prefix=${1:-$(opticks-prefix)}
    cat << EOL
$prefix/lib
$prefix/lib64
$prefix/externals/lib
$prefix/externals/lib64
$prefix/externals/optix/lib64
EOL
}
opticks-join(){ local ifs=$IFS ; IFS="$1"; shift; echo "$*" ; IFS=$ifs ;  }
opticks-llp(){  opticks-join : $($FUNCNAME- $*) ; } 




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


opticks-okdist-dirlabel-notes(){ cat << EON
opticks-okdist-dirlabel-notes
-------------------------------

    opticks-okdist-dirlabel : $(opticks-okdist-dirlabel)

Examples:: 

   x86_64-centos7-gcc48-geant4_10_04_p02-dbg

The label is used by okdist- for naming directories that contain 
Opticks binary distributions.

Note that the below versions are not included in this directory label as
they are encompassed by the Opticks version.

* OptiX version
* CUDA Version
* NVIDIA Driver Version 


CUDA is treated separately and lib access is from LD_LIBRARY_PATH
so perhaps it belongs in the name ? 


EON
}

opticks-okdist-mode(){ echo dbg ; }
opticks-okdist-dirlabel(){ g4- ; echo $(arch)-$(opticks-linux-release)-$(opticks-compiler-version)-$(g4-nom)-$(opticks-okdist-mode) ; }
opticks-gcc-version(){  gcc --version |  perl -ne 'm/(\d)\.(\d)/ && print "$1$2" ' ; }   
opticks-compiler-version(){  echo gcc$(opticks-gcc-version) ; }  ## TODO: clang 
opticks-linux-release-(){   cat /etc/redhat-release | perl -ne 'm/(\w*) Linux release (\d)\.(\d)/ && print "${1}/${2}" ' ; }
opticks-linux-release()
{
   ## TODO: Ubuntu 
   if [ -f /etc/redhat-release ]; then
        local rr=$(opticks-linux-release-) 
        case $(dirname $rr) in 
           Scientific)  echo slc$(basename $rr)    ;;
               CentOS)  echo centos$(basename $rr) ;;
        esac 
   else 
       echo UNKNOWN
   fi 
}






########### bitbucket commits

opticks-co(){      opticks-open  https://bitbucket.org/simoncblyth/opticks/commits/all ; } 
opticks-co2(){     opticks-open  https://bitbucket.org/simoncblyth/opticks-cmake-overhaul/commits/all ; } 



########## building opticks docs 

opticks-bb(){      opticks-open  http://simoncblyth.bitbucket.io/opticks/index.html ; } 
opticks-rdocs(){   opticks-open  http://simoncblyth.bitbucket.io/opticks/index.html ; } 
opticks-rnotes(){  opticks-open  http://simoncblyth.bitbucket.io/opticks_notes/index.html ; } 
opticks-docs(){    opticks-open  $(opticks-docs-htmldir)/index.html ; } 
opticks-docs-htmldir(){ 
   local htmldirbb=$HOME/simoncblyth.bitbucket.io/opticks 
   [ -d "$htmldirbb" ] && echo $htmldirbb || echo $(opticks-prefix)/html 
}

opticks-docs-make-info(){ cat << EOI

$FUNCNAME
========================

opticks-docs-htmldir : $(opticks-docs-htmldir)

EOI
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




## [WIP] modern CMake proj-by-proj style building 

om-(){       . $(opticks-home)/om.bash      && om-env $* ; }
opnovice-(){ . $(opticks-home)/notes/geant4/opnovice.bash      && opnovice-env $* ; }



### opticks CMake projects all residing in top level folders ##

okconf-(){          . $(opticks-home)/okconf/okconf.bash && okconf-env $* ; }
sysrap-(){          . $(opticks-home)/sysrap/sysrap.bash && sysrap-env $* ; }
brap-(){            . $(opticks-home)/boostrap/brap.bash && brap-env $* ; }
npy-(){             . $(opticks-home)/npy/npy.bash && npy-env $* ; }
okc-(){             . $(opticks-home)/optickscore/okc.bash && okc-env $* ; }

ggeo-(){            . $(opticks-home)/ggeo/ggeo.bash && ggeo-env $* ; }
asirap-(){          . $(opticks-home)/assimprap/asirap.bash && asirap-env $* ; }
openmeshrap-(){     . $(opticks-home)/openmeshrap/openmeshrap.bash && openmeshrap-env $* ; }
okg-(){             . $(opticks-home)/opticksgeo/okg.bash && okg-env $* ; }

oglrap-(){          . $(opticks-home)/oglrap/oglrap.bash && oglrap-env $* ; }
cudarap-(){         . $(opticks-home)/cudarap/cudarap.bash && cudarap-env $* ; }
thrap-(){           . $(opticks-home)/thrustrap/thrap.bash && thrap-env $* ; }
oxrap-(){           . $(opticks-home)/optixrap/oxrap.bash && oxrap-env $* ; }

okop-(){            . $(opticks-home)/okop/okop.bash && okop-env $* ; }
okgl-(){            . $(opticks-home)/opticksgl/okgl.bash && okgl-env $* ; }
ok-(){              . $(opticks-home)/ok/ok.bash && ok-env $* ; }
cfg4-(){            . $(opticks-home)/cfg4/cfg4.bash && cfg4-env $* ; }
okg4-(){            . $(opticks-home)/okg4/okg4.bash && okg4-env $* ; }
   
g4ok-(){            . $(opticks-home)/g4ok/g4ok.bash && g4ok-env $* ; }
x4-(){              . $(opticks-home)/extg4/x4.bash  && x4-env $* ; }
x4gen-(){           . $(opticks-home)/extg4/x4gen.bash  && x4gen-env $* ; }
yog-(){             . $(opticks-home)/yoctoglrap/yog.bash && yog-env $* ; }

integration-(){     . $(opticks-home)/integration/integration.bash && integration-env $* ; }


## opticks misc including python analysis/debugging ##

ana-(){             . $(opticks-home)/ana/ana.bash && ana-env $*  ; }
cfh-(){             . $(opticks-home)/ana/cfh.bash && cfh-env $*  ; }
tests-(){           . $(opticks-home)/tests/tests.bash && tests-env $*  ; }
tools-(){           . $(opticks-home)/tools/tools.bash && tools-env $*  ; }
notes-(){           . $(opticks-home)/notes/notes.bash && notes-env $*  ; }
pmt-(){             . $(opticks-home)/ana/pmt/pmt.bash && pmt-env $* ; }
ab-(){              . $(opticks-home)/bin/ab.bash      && ab-env $* ; }
abe-(){             . $(opticks-home)/bin/abe.bash     && abe-env $* ; }
ev-(){              . $(opticks-home)/bin/ev.bash      && ev-env $* ; }
scan-(){            . $(opticks-home)/bin/scan.bash    && scan-env $* ; }
hh-(){              . $(opticks-home)/bin/hh.bash      && hh-env $* ; }
vbx-(){             . $(opticks-home)/bin/vbx.bash     && vbx-env $* ; }
ptx-(){             . $(opticks-home)/bin/ptx.bash     && ptx-env $* ; }
ezgdml-(){          . $(opticks-home)/bin/ezgdml.bash  && ezgdml-env $* ; }
odocker-(){         . $(opticks-home)/bin/odocker.bash && odocker-env $* ; }
olxd-(){            . $(opticks-home)/bin/olxd.bash    && olxd-env $* ; }
onvidia-(){         . $(opticks-home)/bin/onvidia.bash && onvidia-env $* ; }
nsight-(){          . $(opticks-home)/bin/nsight.bash  && nsight-env $* ; }

# override old original from env, $(env-home)/nuwa/detdesc/pmt/pmt.bash

### opticks g4 examples ########
g4x-(){             . $(opticks-home)/examples/g4x.bash && g4x-env $* ; }

### opticks launchers ########

okr-(){             . $(opticks-home)/bin/okr.bash && okr-env $* ; }
okdist-(){          . $(opticks-home)/bin/okdist.bash && okdist-env $* ; }
oks-(){             . $(opticks-home)/bin/oks.bash && oks-env $* ; }
winimportlib-(){    . $(opticks-home)/bin/winimportlib.bash && winimportlib-env $* ; }
ggv-(){             . $(opticks-home)/bin/ggv.bash && ggv-env $* ; }
vids-(){            . $(opticks-home)/bin/vids.bash && vids-env $* ; }
op-(){              . $(opticks-home)/bin/op.sh ; }
fn-(){              . $(opticks-home)/bin/fn.bash && fn-env $* ; }

#### opticks top level tests ########





geocache-(){      . $(opticks-home)/ana/geocache.bash  && geocache-env $* ; }
ckm-(){           . $(opticks-home)/examples/Geant4/CerenkovMinimal/ckm.bash  && ckm-env $* ; }


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
    local hpp=$(find $base -name "$name.hpp");
    local cc=$(find $base -name "$name.cc");
    local cpp=$(find $base -name "$name.cpp");
    local icc=$(find $base -name "$name.icc");
    local tcc=$(find $base -name "${name}Test.cc");
    local vcmd="vi  $h $hh $hpp $icc $cpp $cc $tcc";
    echo $vcmd;
    eval $vcmd;
    cd $iwd
}


opticks-cmake-projs-vi-(){    ls -1  $(opticks-home)/CMakeLists.txt $(opticks-home)/*/CMakeLists.txt ; }
opticks-cmake-examples-vi-(){ ls -1 $(opticks-home)/examples/*/CMakeLists.txt ; }
opticks-cmake-tests-vi-(){    ls -1 $(opticks-home)/*/tests/CMakeLists.txt ; }

opticks-cmake-vi-()
{  
    opticks-cmake-projs-vi-
    opticks-cmake-examples-vi- 
    opticks-cmake-tests-vi-
}

opticks-cmake-projs-vi(){     vi $($FUNCNAME-) ; }
opticks-cmake-examples-vi(){  vi $($FUNCNAME-) ; }
opticks-cmake-tests-vi(){     vi $($FUNCNAME-) ; }
opticks-cmake-vi(){           vi $($FUNCNAME-) ; }




opticks-h()
{
   local msg="=== $FUNCNAME $(pwd) :"
 
   echo $msg header includes 
   grep -h ^#include *.hh *.hpp 2>/dev/null | sort | uniq 
   
   #echo $msg implementation includes 
   #grep -h ^#include *.cc *.cpp 2>/dev/null | sort | uniq 
}


opticks-rpath(){ grep RPATH $(opticks-home)/examples/*/CMakeLists.txt ; }

opticks-bcm-deploy-(){  find . -name CMakeLists.txt -exec grep -l bcm_deploy {} \; ; }
opticks-bcm-deploy(){ vi $(opticks-bcm-deploy-) ; }

opticks-deps(){ CMakeLists.py $* ; }
opticks-deps-vi(){ vi $(opticks-home)/bin/CMakeLists.py ; }

opticks-executables(){  find . -type f -perm +111 -print | grep -v dylib | grep -v a.out | grep -v .bin | grep -v .cmake | grep -v opticks-config ; }




opticks-cmake-check(){
   opticks-scd 
   find . -name CMakeLists.txt -exec grep -l OpticksBuildOptions {} \; | wc -l 
   find . -name CMakeLists.txt | wc -l 

   ## see bin/CMakeLists.py for easier way of making such consistency checks 

}

opticks-pdoc(){ o ; vi okop/OKOP.rst opticksgeo/OKGEO.rst optixrap/OXRAP.rst thrustrap/THRAP.rst g4ok/G4OK.rst ; }



opticks-linecount(){

   opticks-scd
   find . -path ./.hg -prune -o -name '*.*'  | xargs wc -l 

}

