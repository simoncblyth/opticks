om-source(){ echo $BASH_SOURCE ; }
om-vi(){ vi $(om-source) ; }
om-env(){  olocal- ; opticks- ; }
om-usage(){ cat << EOU

OM : Opticks Minimal Approach to Configuring and Building
===========================================================

The below subproj functions configure, build, install or test 
either one, all or a range of Opticks subprojects 
in the appropriate dependency order.

The commands are sensitive to the invoking directory, performing
actions for all subprojects when invoked from top level 
directories. Also an argument can be used to select a 
starting subproject, eg "cudarap:" starts from there 
or ":" starts from the subproject of the current directory.

When building for the first time it is necessary to 
use "om-install" as later subprojects cannot be configured 
until earlier ones have been installed.

Usage Examples
---------------

om-subs okconf:opticksgeo
    list the subprojects in this inclusive range

om-subs :opticksgeo
    same as the above, as first is implicit

om-test :opticksgeo
    run the tests for all subprojects from the first okconf upto opticksgeo    


SUBPROJ FUNCTIONS 
-----------------

om-subs
   list subprojects in dependency order

om-conf
   configures using cmake 

om-make
   build and install 

om--
   shorter name for om-make

om-install
    configures, builds and installs by doing both om-conf and om-make 

om-test
   runs ctests

om-visit 
   visits each subproj in turn
 
om-echo
   echos the pwd of each subproj
 

OTHER FUNCTIONS
-----------------

om-cd
   cd from a source tree directory to the corresponding 
   directory in the build tree or vice versa


FUNCTIONS INSENSITIVE TO INVOKING DIRECTORY
-------------------------------------------------

om-visit-all
om-conf-all
om-make-all
om-test-all

om-testlog
    parse test logs from previous test runs and present summary totals 


    
EOU
} 

om-home-default(){  echo $(dirname $(om-source)) ; } 
om-home(){   echo ${OPTICKS_HOME:-$(om-home-default)} ; }
om-local(){  echo ${LOCAL_BASE:-/usr/local} ; }
om-name(){   echo $(basename $(om-home)) ; }
om-fold(){   echo $(om-local)/$(om-name) ; }
om-prefix(){ echo $(om-fold) ; }
om-cmake-generator(){ echo ${OPTICKS_CMAKE_GENERATOR:-Unix Makefiles} ; }
om-bdir(){  
   local gen=$(om-cmake-generator)
   case $gen in 
      "Unix Makefiles") echo $(om-prefix)/build/$1 ;;
               "Xcode") echo $(om-prefix)/build_xcode/$1 ;;
   esac
}
om-sdir(){   echo $(om-home)/$1 ; }

om-url(){ echo http://bitbucket.org/simoncblyth/$(om-name)/src/$(om-rdir) ; }
om-open(){ open $(om-url) ; }

om-info(){ cat << EOI

   om-home-default    : $(om-home-default)
   om-home            : $(om-home)  from OPTICKS_HOME envvar if defined 
   om-name            : $(om-name)
   om-local           : $(om-local)  from LOCAL_BASE envvar if defined
   om-fold            : $(om-fold)
   om-prefix          : $(om-prefix)
   om-cmake-generator : $(om-cmake-generator)
   om-bdir            : $(om-bdir)
   om-sdir            : $(om-sdir)

EOI
}

om-subs--all(){ cat << EOS
# to update this list : opticks-deps --subdirs
okconf
sysrap
boostrap
npy
yoctoglrap
optickscore
ggeo
assimprap
openmeshrap
opticksgeo
cudarap
thrustrap
optixrap
okop
oglrap
opticksgl
ok
extg4
cfg4
okg4
g4ok
EOS
}


om-subs--partial(){  cat << EOS
#boostrap
#optickscore
ggeo
extg4
EOS
}

om-subs--()
{
   om-subs--all
   #om-subs--partial
}


om-subs-(){ om-subs-- | grep -v ^\# ; }



om-subs-notes(){ cat << EON
om-subs
--------

Default with no argument lists all subproject directory names 
in build dependency order. Arguments can be used to select 
ranges of names.
 
Shorthand forms  ":" and "+" only work from within a subproj
which is taken as the first, all other forms work from anywhere.

:
    subproj from current onwards  
+
    subproj from next onwards  

:last
    subprojects from the first "okconf" to the last specified inclusive, 
    eg ":ok" lists all up to and including "ok" 

first:last
    subprojects from the "first" to "last" inclusive 

EON
}

om-subs(){

   local arg=$1
   [ -z "$arg" ] && om-subs- && return 
 
   local iwd=$(pwd)
   local name=$(basename $iwd)

   local mode=":"
   local first
   local last 

   ## expand shorthand argument into full 
   [ "$arg" == ":" ] && arg="${name}:" && mode=":"
   [ "$arg" == "+" ] && arg="${name}:" && mode="+"

   ## form ":last" means from okconf (1st project)  
   [ "${arg:0:1}" == ":" ] && first="okconf" || first=${arg/:*}
   last=${arg/*:}

   #echo $FUNCNAME arg $arg name $name first $first last $last  

   local sel=0 
   local sub
   om-subs- | while read sub ; do
       [ "${sub}" == "$first" -a "${mode}" == ":" ] && sel=1 
       [ "${sub}" == "$first" -a "${mode}" == "+" ] && sel=2 
       [ "$sel" == "1" ] && echo $sub
       [ "${sub}" == "$last" -a "${mode}" == ":" ] && sel=0
       [ "$sel" == "2" ] && sel=1
   done 
}


om-subs0(){
  local arg=$1

  local iwd=$(pwd)
  local name=$(basename $iwd)

  if [ -z "$arg" ]; then 
      om-subs-
  else
      [ "$arg" == ":" ] && arg="${name}:"
      [ "$arg" == "+" ] && arg="${name}+"
      local sel=0 
      local sub
      om-subs- | while read sub ; do
         [ "${sub}:" == "$arg" ] && sel=1 
         [ "${sub}+" == "$arg" ] && sel=2 
         [ "$sel" == "1" ] && echo $sub
         [ "$sel" == "2" ] && sel=1
      done 
  fi
}





om-reldir()
{
   local cwd=$(pwd)
   local sdir=$(om-sdir)
   local bdir=$(om-bdir)
   case $cwd in
      $sdir*) echo ${cwd/$sdir} ;; 
      $bdir*) echo ${cwd/$bdir} ;; 
   esac
}
om-bcd(){ cd $(om-bdir $(om-reldir)) ; pwd ;  }
om-scd(){ cd $(om-sdir $(om-reldir)) ; pwd ;  }


om-visit-all(){     om-all ${FUNCNAME/-all} $* ; }
om-conf-all(){      om-all ${FUNCNAME/-all} $* ; }
om-make-all(){      om-all ${FUNCNAME/-all} $* ; }
om-install-all(){   om-all ${FUNCNAME/-all} $* ; }
om-test-all(){      om-all ${FUNCNAME/-all} $* ; om-testlog ; }
om-echo-all(){      om-all ${FUNCNAME/-all} $* ; }
om-clean-all(){     om-all ${FUNCNAME/-all} $* ; }
om-find-all(){      om-all ${FUNCNAME/-all} $* ; }

om-testlog(){      CTestLog.py $(om-bdir)  ; }


om-conf-xcode(){ OPTICKS_CMAKE_GENERATOR=Xcode om-conf ; }

om-conf(){    om-one-or-all conf $* ; }
om-make(){    om-one-or-all make $* ; }
om-install(){ om-one-or-all install $* ; }
om-visit(){   om-one-or-all visit $* ; }
om-test(){    om-one-or-all test $* ; }
om-echo(){    om-one-or-all echo $* ; }
om-clean(){   om-one-or-all clean $* ; }
om-find(){    om-one-or-all find $* ; }

om--(){       om-make $* ; }     ## just this proj
om---(){      om-make-all : ; }  ## all projs from this one onwards 
om----(){     om-make-all + ; }  ## all projs following this one 


om-check()
{
    local msg="=== $FUNCNAME :"
    local bdir=$(om-bdir)
    local rc=0
    if [ ! -d "$bdir" ]; then 
        echo $msg top level bdir $bdir does not exist : creating it 
        mkdir -p $bdir  
    fi 
    if [ ! -d "$bdir" ]; then 
        echo $msg top level bdir $bdir does not exist : create failed 
        rc=1
    fi 
    return $rc 
}



om-all()
{
    local rc
    local iwd=$(pwd)
    local func=$1
    local msg="=== $FUNCNAME $func :"
    shift

    om-check
    rc=$?
    [ "$rc" != "0" ] && echo $msg ERROR om-check failed && return $rc

    local name
    om-subs $* | while read name 
    do 
        local sdir=$(om-sdir $name)
        local bdir=$(om-bdir $name)
        mkdir -p $bdir
        local udir
        case $func in
          om-find) udir=$sdir ;;
                *) udir=$bdir ;;
        esac
        
        cd $udir

        $func
        rc=$?
        [ "$rc" != "0" ] && echo $msg ERROR bdir $bdir : non-zero rc $rc && return $rc
    done 
    cd $iwd
    return $rc
}

om-one-or-all()
{
    local rc=0 
    local func=$1
    local arg=$2  # not normally used
    local iwd=$(pwd)
    local msg="=== $FUNCNAME $func :"

    om-check
    rc=$?
    [ "$rc" != "0" ] && echo $msg ERROR om-check failed && om-info && return $rc 

    if [ "${iwd}/" == "$(om-sdir)" -o "${iwd}/" == "$(om-bdir)" -o  "${arg/:}" != "$arg" ]; then
        om-$func-all $arg 
        rc=$?
    else
        om-$func-one $arg 
        rc=$?
    fi

    if [ "$rc" != "0" ]; then
       echo $msg non-zero rc $rc 
    fi 
    cd $iwd
    return $rc 
}

om-echo-one(){ echo $(pwd) ; }

om-install-one()
{
    om-visit-one $*
    om-conf-one $*
    om-make-one $*
}

om-visit-one()
{
    local msg="=== $FUNCNAME :"
    local iwd=$(pwd)
    local name=$(basename $iwd)
    local sdir=$(om-sdir $name)
    local bdir=$(om-bdir $name)

    if [ ! -d "$bdir" ]; then
         echo $msg bdir $bdir does not exist : creating it
         mkdir -p $bdir
    fi 
    cd $bdir
    printf "%s %-15s %-60s %-60s \n"  "$msg" $name $sdir $bdir
}

om-conf-one()
{
    local arg=$1
    local iwd=$(pwd)

    local name=$(basename ${iwd/tests})   # trim tests to get name of subproj from tests folder or subproj folder
    local sdir=$(om-sdir $name)
    local bdir=$(om-bdir $name)

    if [ "$arg" == "clean" ]; then
         echo $msg removed bdir $bdir as directed by clean argument
         rm -rf $bdir
    fi 

    if [ ! -d "$bdir" ]; then
         echo $msg bdir $bdir does not exist : creating it
         mkdir -p $bdir
    fi 

    cd $bdir
    printf "%s %-15s %-60s %-60s \n"  "$msg" $name $sdir $bdir

    local rc=0
    if [ "$name" == "okconf" ]; then     
        om-cmake-okconf $sdir
        rc=$?
    else
        om-cmake $sdir
        rc=$?
    fi
    return $rc
}


om-cmake-okconf()
{
    local sdir=$1  
    local bdir=$PWD
    [ "$sdir" == "$bdir" ] && echo ERROR sdir and bdir are the same $sdir && return 1000

    local rc 
    cmake $sdir \
       -G "$(om-cmake-generator)" \
       -DCMAKE_BUILD_TYPE=$(opticks-buildtype) \
       -DCMAKE_PREFIX_PATH=$(om-prefix)/externals \
       -DCMAKE_INSTALL_PREFIX=$(om-prefix) \
       -DCMAKE_MODULE_PATH=$(om-home)/cmake/Modules \
       -DOptiX_INSTALL_DIR=$(opticks-optix-install-dir) \
       -DCOMPUTE_CAPABILITY=$(opticks-compute-capability)

    rc=$?
    return $rc
}


om-cmake()
{
    local sdir=$1  
    local bdir=$PWD
    [ "$sdir" == "$bdir" ] && echo ERROR sdir and bdir are the same $sdir && return 1000

    local rc 
    cmake $sdir \
       -G "$(om-cmake-generator)" \
       -DCMAKE_BUILD_TYPE=$(opticks-buildtype) \
       -DCMAKE_PREFIX_PATH=$(om-prefix)/externals \
       -DCMAKE_INSTALL_PREFIX=$(om-prefix) \
       -DCMAKE_MODULE_PATH=$(om-home)/cmake/Modules 

    rc=$?
    return $rc
#   -DBOOST_INCLUDEDIR=$(opticks-boost-includedir) \
#   -DBOOST_LIBRARYDIR=$(opticks-boost-libdir)
}

om-cmake-info(){ cat << EOI

$FUNCNAME
===============

   om-cmake-generator         : $(om-cmake-generator)
   opticks-buildtype          : $(opticks-buildtype)
   om-prefix                  : $(om-prefix)

   opticks-optix-install-dir  : $(opticks-optix-install-dir)
   OPTICKS_OPTIX_INSTALL_DIR  : $OPTICKS_OPTIX_INSTALL_DIR
 
   opticks-compute-capability : $(opticks-compute-capability)
   OPTICKS_COMPUTE_CAPABILITY : $OPTICKS_COMPUTE_CAPABILITY 

   NODE_TAG                   : $NODE_TAG

EOI
}


om-nproc(){ echo ${OM_NPROC:-$(nproc)} ; }
    

om-make-one()
{
    local rc=0
    local iwd=$(pwd)
    local msg="=== $FUNCNAME :"
    local name=$(basename ${iwd/tests})   # trim tests to get name of subproj from tests folder or subproj folder
    local sdir=$(om-sdir $name)
    local bdir=$(om-bdir $name)
    printf "%s %-15s %-60s %-60s \n"  "$msg" $name $sdir $bdir

    if [ ! -d $bdir ]; then
       echo $msg ERROR bdir $bdir does not exist : you need to om-install OR om-conf once before using om-make or the om-- shortcut  
       rc=1
       return $rc
    fi 

    cd $bdir
    local t0=$(date +"%s")

    local pr=$(om-nproc)
    
    if [ $pr -gt 1 ]; then 
        make all -j$pr
    else
        cmake --build .  --target all
    fi

    rc=$?
    local t1=$(date +"%s")
    local d1=$(( t1 - t0 ))

    [ "$rc" != "0" ] && cd $iwd && return $rc

    #echo d1 $d1
    [ "$(uname)" == "Darwin" -a $d1 -lt 1 ] && echo $msg kludge sleep 2s : make time $d1 && sleep 2  

    if [ $pr -gt 1 ]; then 
        make install -j$pr
    else
        cmake --build .  --target install
    fi


    rc=$?
    return $rc
    [ "$rc" != "0" ] && cd $iwd && return $rc
}

om-test-one()
{
    local iwd=$(pwd)
    local name=$(basename $iwd)

    local msg="=== $FUNCNAME :"
    local name=$(basename ${iwd/tests})   # trim tests to get name of subproj from tests folder or subproj folder
    local sdir=$(om-sdir $name)
    local bdir=$(om-bdir $name)
    printf "%s %-15s %-60s %-60s \n"  "$msg" $name $sdir $bdir
    cd $bdir
    local log=ctest.log
    date          | tee $log
    ctest $* --interactive-debug-mode 0 2>&1 | tee -a $log
    date          | tee -a $log

    cd $iwd
}

om-clean-one()
{
    local iwd=$(pwd)
    local name=$(basename $iwd)

    local msg="=== $FUNCNAME :"
    local name=$(basename ${iwd/tests})   # trim tests to get name of subproj from tests folder or subproj folder
    local sdir=$(om-sdir $name)
    local bdir=$(om-bdir $name)
    cd $sdir
    local cmd="rm -rf $bdir && mkdir -p $bdir" 
    echo $cmd
    eval $cmd

    cd $iwd
}





om-find-one(){   
   local str="${1:-ENV_HOME}"
   local opt=${2:--H}
   local iwd=$(pwd)
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
       -name '*.rst' -or \
       -name '*.py' \
        \) \
       -exec grep $opt "$str" {} \;

    cd $iwd
}








om-bdir-()
{
    local iwd=$(pwd)
    local name=$(basename ${iwd/tests})   # trim tests to get name of subproj from tests folder or subproj folder
    local bdir=$(om-bdir $name)
    echo $bdir  
}

om-test-log(){ echo $(om-bdir-)/ctest.log ; }
om-tl(){ cat $(om-test-log) ; }




om-divider(){ cat << EOD
//////////////////////////////////////////////////////////////////////////////////

My bash functions tend to grow, in order to avoid confusing users 
trying to install Opticks should try to distinguish between 

1. essentials for building 
2. useful additions for developers

This divider attempts to make that split.  Possibly relocate the below into 
an omu- to make that distinction stronger.

//////////////////////////////////////////////////////////////////////////////////
EOD
}


om-mk-notes(){ cat << EON
om-mk 
=======

om-mk allows running commands from the parallel build tree, 
particularly useful for quickly building and running/debugging 
single test executables. This is especially useful with projects 
like NPY which is quite slow to build fully, as it allows just the 
needed code to be built for the test executable.

Note this works so long as the source changes are all within the 
current subproj. If there are changes in other projs this will not 
detect that::

   npy-c 
   cd tests

   om-mk "make help | grep Test"

   om-mk "make NTreeBuilderTest && ./NTreeBuilderTest"

   om-mk "make NConvexPolyhedronSegmentTest  && ./NConvexPolyhedronSegmentTest"

       ## build and run a single test 
    
   TEST=NSceneLoadTest om-t


Note that this does not install ... and 
there seems no way to install just the one test ?

EON
}


#om-tt(){ echo ${TEST:-NTreePositiveTest} ; }
#om-tt(){ echo ${TEST:-NTreeBalanceTest} ; }
#om-tt(){ echo ${TEST:-NCSGSaveTest} ; }
om-tt(){ echo ${TEST:-X4PolyconeTest} ; }

om-t(){  om-t- $(om-tt) ; }
om-t-(){ om-mk "make $1  && ./$1" ; }

om-d(){  om-d- $(om-tt) ; }
om-d-(){ om-mk "make $1  && lldb ./$1" ; }

om-mk()
{
    local msg="=== $FUNCNAME :"
    local iwd=$(pwd)
    local rdir=$(om-reldir)   #  relative dir, when invoked from within source or build trees
    [ "${rdir/tests}" == "${rdir}" ] && rdir="$rdir/tests"  
    local bdir=$(om-bdir $rdir)
    cd $bdir
    echo $msg bdir $bdir rdir $rdir : $1 
    eval $1
    cd $iwd
}

om-pdir() 
{ 
    local here=$(pwd -P);
    local stop=$(om-sdir);
    local btop=$(om-bdir);
    stop=${stop%/}  # remove trailing slash 
    btop=${btop%/}   
    case $here in 
        $stop)  echo $btop ;;
        $btop)  echo $stop ;;
        $stop*) echo $btop/${here/$stop\/} ;;
        $btop*) echo $stop/${here/$btop\/} ;;
             *) echo "" ;;
    esac
    return 0 
}

om-rdir()
{
    local here=$(pwd -P);
    local stop=$(om-sdir);
    local btop=$(om-bdir);
    stop=${stop%/}  # remove trailing slash 
    btop=${btop%/}   
    case $here in 
        $stop)  echo "" ;;
        $btop)  echo "" ;;
        $stop*) echo ${here/$stop\/} ;;
        $btop*) echo ${here/$btop\/} ;;
             *) echo "" ;;
    esac
    return 0 
}

om-cd()
{
    local msg="=== $FUNCNAME :"
    local iwd=$(pwd -P)
    local pdir=$(om-pdir);
    [ -z "$pdir" ] && echo pwd $iwd is not inside Opticks source tree or its counterpart build tree && return 1;
    [ -n "$pdir" ] && cd $pdir
    #printf "%s %-60s to %-60s \n"  "$msg" $name $iwd $pdir
    pwd
}



om-gen()
{
   cd $(opticks-home)

   local rel=$1
   rel=${rel/.bash}

   local nam=$2
   [ -z "$nam" ] && nam=$(basename $rel)

   local dir=$(dirname $rel)
   mkdir -p $dir 

   om-gen- $nam $rel > $rel.bash

   . $rel.bash

   $nam-vi
}


om-gen-(){ cat << EOT

$1-source(){ echo \$BASH_SOURCE ; }
$1-vi(){ vi \$($1-source) om.bash opticks.bash externals/externals.bash ; }
$1-env(){  olocal- ; opticks- ; }
$1-usage(){ cat << EOU

$1 Usage 
===================

Generate a file for bash precursor functions or notes using om-gen like this::

   om-gen yoctoglrap/yog         # makes precursor yog-
   om-gen notes/geant4/opnovice 

Hook up a line like the below to opticks.bash or externals/externals.bash::
  
   $1-(){ . \$(opticks-home)/$2.bash      && $1-env \$* ; }


EOU
}

$1-dir(){ echo \$(dirname \$($1-source)) ; }
$1-cd(){  cd \$($1-dir) ; } 
$1-c(){   cd \$($1-dir) ; } 
$1--(){   opticks-- \$($1-dir) ; } 

EOT
}





om-tst-(){ cat << EOT

#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);


    return 0 ; 
}

EOT

}
