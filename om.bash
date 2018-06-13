om-source(){ echo $BASH_SOURCE ; }
om-vi(){ vi $(om-source) ; }
om-env(){  olocal- ; opticks- ; }
om-usage(){ cat << EOU

OM : Opticks Mimimal Approach to Configuring and Building
===========================================================

The below functions configure, build and test the Opticks subprojects
in the appropriate dependency order.


SUBPROJ FUNCTIONS 
-----------------

om-subs
   list subprojects in dependency order,
   an argument can be used to select a
   starting subproject, eg "cudarap:" starts from there or ":" 
   starts from the current directory

   Such arguments can be used for all the subproj functions

om-conf
   runs cmake to configure one/all/range of subprojects 

om-make
   builds and install one/all/range of subprojects 

om-test
   runs ctest for one/all/range of subprojects

om-visit 
   debugging
 
om-echo
   debugging
 


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

om-subs--all(){ cat << EOS
okconf
sysrap
boostrap
npy
yoctoglrap
extg4
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
   #om-subs--all
   om-subs--partial
}


om-subs-(){ om-subs-- | grep -v ^\# ; }

om-subs(){
  local arg=$1

  local iwd=$(pwd)
  local name=$(basename $iwd)

  if [ -z "$arg" ]; then 
      om-subs-
  else
      [ "$arg" == ":" ] && arg="${name}:"
      local sel=0 
      local sub
      om-subs- | while read sub ; do
         [ "${sub}:" == "$arg" ] && sel=1 
         [ "$sel" == "1" ] && echo $sub
      done 
  fi
}


om-sdir(){  echo $(opticks-home)/$1 ; }
om-bdir(){  
   local gen=$(opticks-cmake-generator)
   case $gen in 
      "Unix Makefiles") echo $(opticks-prefix)/build/$1 ;;
               "Xcode") echo $(opticks-prefix)/build_xcode/$1 ;;
   esac
}

om-visit-all(){     om-all ${FUNCNAME/-all} $* ; }
om-conf-all(){      om-all ${FUNCNAME/-all} $* ; }
om-make-all(){      om-all ${FUNCNAME/-all} $* ; }
om-test-all(){      om-all ${FUNCNAME/-all} $* ; om-testlog ; }
om-echo-all(){      om-all ${FUNCNAME/-all} $* ; }

om-testlog(){      CTestLog.py $(om-bdir)  ; }

om-all()
{
    local rc
    local iwd=$(pwd)
    local func=$1
    shift
    local msg="=== $FUNCNAME $func :"
    local name
    om-subs $* | while read name 
    do 
        local bdir=$(om-bdir $name)
        cd $bdir
        $func
        rc=$?
        [ "$rc" != "0" ] && echo $msg ERROR bdir $bdir : non-zero rc $rc && return $rc
    done 
    cd $iwd
    return $rc
}


om-echo(){ echo $(pwd) ; }

om-visit()
{
    local arg=$1  # not normally used
    local msg="=== $FUNCNAME :"
    local iwd=$(pwd)

    if [ "${iwd}/" == "$(om-sdir)" -o "${iwd}/" == "$(om-bdir)" -o "${arg/:}" != "$arg" ]; then

        om-visit-all $arg

    else
        local name=$(basename $iwd)
        local sdir=$(om-sdir $name)
        local bdir=$(om-bdir $name)
        cd $bdir
        printf "%s %-15s %-60s %-60s \n"  "$msg" $name $sdir $bdir
    fi
}



om-conf-xcode(){ OPTICKS_CMAKE_GENERATOR=Xcode om-conf ; }

om-conf()
{
    local rc 
    local arg=$1  # not normally used
    local iwd=$(pwd)
    local msg="=== $FUNCNAME :"

    if [ "${iwd}/" == "$(om-sdir)" -o "${iwd}/" == "$(om-bdir)" -o  "${arg/:}" != "$arg" ]; then

        om-conf-all $arg 

    else
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

        # TODO: hmm cleaner just to use same invokation for all pkgs 
 
        if [ "$name" == "okconf" ]; then     
            cmake $sdir \
               -G "$(opticks-cmake-generator)" \
               -DCMAKE_BUILD_TYPE=$(opticks-buildtype) \
               -DCMAKE_PREFIX_PATH=$(opticks-prefix)/externals \
               -DCMAKE_INSTALL_PREFIX=$(opticks-prefix) \
               -DCMAKE_MODULE_PATH=$(opticks-home)/cmake/Modules \
               -DOptiX_INSTALL_DIR=$(opticks-optix-install-dir) \
               -DCOMPUTE_CAPABILITY=$(opticks-compute-capability)
            rc=$?
        else
            cmake $sdir \
               -G "$(opticks-cmake-generator)" \
               -DCMAKE_BUILD_TYPE=$(opticks-buildtype) \
               -DCMAKE_PREFIX_PATH=$(opticks-prefix)/externals \
               -DCMAKE_INSTALL_PREFIX=$(opticks-prefix) \
               -DCMAKE_MODULE_PATH=$(opticks-home)/cmake/Modules 

            rc=$?
        fi
    fi

    if [ "$rc" != "0" ]; then
       echo $msg non-zero rc $rc 
    fi 
    cd $iwd
    return $rc 
}


om--(){ om-make $* ; }

om-make()
{   
    local rc=0
    local arg=$1  # not normally used
    local iwd=$(pwd)

    if [ "${iwd}/" == "$(om-sdir)" -o "${iwd}/" == "$(om-bdir)" -o "${arg/:}" != "$arg" ]; then

        om-make-all $arg

    else
        local msg="=== $FUNCNAME :"
        local name=$(basename ${iwd/tests})   # trim tests to get name of subproj from tests folder or subproj folder
        local sdir=$(om-sdir $name)
        local bdir=$(om-bdir $name)
        printf "%s %-15s %-60s %-60s \n"  "$msg" $name $sdir $bdir
        cd $bdir
        local t0=$(date +"%s")
        cmake --build .  --target all
        rc=$?
        local t1=$(date +"%s")
        local d1=$(( t1 - t0 ))

        [ "$rc" != "0" ] && cd $iwd && return $rc

        #echo d1 $d1
        [ "$(uname)" == "Darwin" -a $d1 -lt 1 ] && echo $msg kludge sleep 2s : make time $d1 && sleep 2  
        cmake --build .  --target install
        rc=$?
        [ "$rc" != "0" ] && cd $iwd && return $rc
    fi 
    cd $iwd
    return $rc
}



om-test()
{
    local iwd=$(pwd)
    local name=$(basename $iwd)

    if [ "${iwd}/" == "$(om-sdir)" -o "${iwd}/" == "$(om-bdir)" -o "${arg/:}" != "$arg" ]; then

        om-test-all $arg

    else
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
    fi
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


om-url(){ echo http://bitbucket.org/simoncblyth/$(opticks-name)/src/$(om-rdir) ; }
om-open(){ open $(om-url) ; }

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
   local nam=$(basename $rel)
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

   om-gen notes/geant4/opnovice 

Hook up a line like the below to opticks.bash or externals/externals.bash::
  
   $1-(){ . \$(opticks-home)/$2.bash      && $1-env \$* ; }


EOU
}

EOT
}


om-find()
{ 
    local str=${1:-ENV_HOME}
    local opt=${2:--H}
    local iwd=$PWD
    cd $(opticks-home) 
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
}




om-tst-(){ cat << EOT

#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG_COLOR__(argc, argv);


    return 0 ; 
}

EOT
}
