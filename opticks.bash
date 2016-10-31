opticks-(){         source $(opticks-source) && opticks-env $* ; }
opticks-src(){      echo opticks.bash ; }
opticks-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(opticks-src)} ; }
opticks-vi(){       vi $(opticks-source) ; }
opticks-usage(){   cat << \EOU

OPTICKS BASH FUNCTIONS
========================

*opticks-docs*
     open browser on the local html documentation

*opticks-docs-make*
     sphinx-build the docs

*opticks-notes*
     open browser on the local html development notes 

*opticks-notes-make*
     sphinx-build the notes


EOU
}

opticks-env(){      
   # dont pollute : otherwise will get infinite loops : as opticks is used in many other -env
   . $(opticks-home)/externals/externals.bash   ## just precursors
}

olocal-()
{
   echo -n # transitional standin for olocal-
}

opticks-home(){   echo ${OPTICKS_HOME:-$HOME/opticks} ; }  ## input from profile 

opticks-suffix(){
   case $(uname) in
      MING*) echo .exe ;;
          *) echo -n  ;;   
   esac
}

opticks-fold(){ 
   # when LOCAL_BASE unset rely instead on finding an installed binary from PATH  
   if [ -z "$LOCAL_BASE" ]; then 
      echo $(dirname $(dirname $(which OpticksTest$(opticks-suffix))))
   else
      echo ${LOCAL_BASE}/opticks ;
   fi
}

opticks-sdir(){   echo $(opticks-home) ; }
opticks-scd(){  cd $(opticks-sdir)/$1 ; }
opticks-ncd(){  opticks-scd notes/issues ;  }

#opticks-bid(){    echo $(optix-vernum) ; }    ## build identity 
#opticks-prefix(){ echo $(opticks-fold)/$(opticks-bid) ; }
opticks-prefix(){ echo $(opticks-fold)  ; }

opticks-dir(){    echo $(opticks-prefix) ; }
opticks-idir(){   echo $(opticks-prefix) ; }
opticks-bdir(){   echo $(opticks-prefix)/build ; }
opticks-bindir(){ echo $(opticks-prefix)/lib ; }   ## use lib for executables for simplicity on windows

opticks-xdir(){   echo $(opticks-fold)/externals ; }  ## try putting externals above the build identity 

opticks-cd(){   cd $(opticks-dir) ; }
opticks-icd(){  cd $(opticks-idir); }
opticks-bcd(){  cd $(opticks-bdir); }
opticks-xcd(){  cd $(opticks-xdir); }



opticks-optix-install-dir(){ 
    local t=$NODE_TAG
    case $t in 
       D_400) echo /Developer/OptiX_400 ;;
       D) echo /Developer/OptiX_380 ;;
     GTL) echo ${MYENVTOP}/OptiX ;;
    H5H2) echo ${MYENVTOP}/OptiX ;;
       X) echo /usr/local/optix-3.8.0/NVIDIA-OptiX-SDK-3.8.0-linux64 ;;
       *) echo /tmp ;;
    esac
}

opticks-compute-capability(){
    local t=$NODE_TAG
    case $t in 
       D) echo 30 ;;
     GTL) echo 30 ;;
    H5H2) echo 50 ;;
       X) echo 52 ;; 
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
EOL
}

opticks-optionals(){ cat << EOL
xercesc
g4
EOL
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

opticks-externals-install(){ opticks-externals | -opticks-installer ; }
opticks-externals-url(){     opticks-externals | -opticks-url ; }
opticks-externals-dist(){    opticks-externals | -opticks-dist ; }

opticks-optionals-install(){ opticks-optionals | -opticks-installer ; }
opticks-optionals-url(){     opticks-optionals | -opticks-url ; }
opticks-optionals-dist(){    opticks-optionals | -opticks-dist ; }

opticks-info(){
   echo externals-url
   opticks-externals-url
   echo externals-dist
   opticks-externals-dist
   echo optionals-url
   opticks-optionals-url
   echo optionals-dist
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

opticks-cmake(){
   local msg="=== $FUNCNAME : "
   local iwd=$PWD
   local bdir=$(opticks-bdir)

   mkdir -p $bdir
   [ -f "$bdir/CMakeCache.txt" ] && echo $msg configured already use opticks-configure to reconfigure  && return  

   opticks-bcd

   g4- 
   xercesc-


   echo $msg opticks-prefix:$(opticks-prefix)
   echo $msg opticks-optix-install-dir:$(opticks-optix-install-dir)
   echo $msg g4-cmake-dir:$(g4-cmake-dir)
   echo $msg xercesc-library:$(xercesc-library)
   echo $msg xercesc-include-dir:$(xercesc-include-dir)

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

opticks-cmake-modify(){
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


opticks-wipe(){
   local bdir=$(opticks-bdir)
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


  

#opticks-config(){ echo Debug ; }
opticks-config(){ echo RelWithDebInfo ; }
opticks--(){     

   local msg="$FUNCNAME : "
   local iwd=$PWD

   local bdir=$1
   shift
   [ -z "$bdir" -o "$bdir" == "." ] && bdir=$(opticks-bdir) 
   [ ! -d "$bdir" ] && echo $msg bdir $bdir does not exist && return 

   cd $bdir

   cmake --build . --config $(opticks-config) --target ${1:-install}

   cd $iwd
}


opticks-prepare-installcache()
{
    cudarap-
    cudarap-prepare-installcache

    OpticksPrepareInstallCache  
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

   local msg="$FUNCNAME : "
   local iwd=$PWD

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

   opticks-t-- $*

   cd $iwd
   echo $msg use -V to show output 
}

opticks-t--()
{
   [ "$(which ctest 2>/dev/null)" == "" ] && ctest3 $* || ctest $*
}


opticks-lfind(){ opticks-find $1 -l ; }
opticks-find(){
   local str=${1:-ENV_HOME}
   local opt=${2:--H}

   local iwd=$PWD
   opticks-scd

   find . -name '*.bash' -exec grep $opt $str {} \;
   find . -name '*.cu' -exec grep $opt $str {} \;
   find . -name '*.cc' -exec grep $opt $str {} \;
   find . -name '*.hh' -exec grep $opt $str {} \;
   find . -name '*.cpp' -exec grep $opt $str {} \;
   find . -name '*.hpp' -exec grep $opt $str {} \;
   find . -name '*.h' -exec grep $opt $str {} \;
   find . -name '*.txt' -exec grep $opt $str {} \;

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

opticks-full()
{
    local msg="=== $FUNCNAME :"
    echo $msg START $(date)

    if [ ! -d "$(opticks-prefix)/externals" ]; then
         opticks-externals-install
    fi 

    opticks-configure

    opticks--

    echo $msg DONE $(date)
}

opticks-cleanbuild()
{
   opticks-distclean 
   opticks-distclean | sh 
   opticks-full 
}


########## runtime setup ########################

opticks-path(){ echo $PATH | tr ":" "\n" ; }
opticks-path-add(){
  local dir=$1 
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

opticks-docs(){ opticks-open  $(opticks-docs-htmldir)/index.html ; } 
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
}

########## building opticks dev notes


opticks-notes-cd(){ cd $(opticks-home)/notes/issues/geant4_opticks_integration ; }

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
tests-(){           . $(opticks-home)/tests/tests.bash && tests-env $*  ; }

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



