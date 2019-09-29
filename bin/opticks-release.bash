opticks-release-source(){ echo $BASH_SOURCE ; }
opticks-release-vi(){     vi $BASH_SOURCE ; } 
opticks-release-(){       source $BASH_SOURCE ; }

opticks-release-dir(){    echo $(dirname $BASH_SOURCE) ; }
opticks-release-prefix(){ echo $(dirname $(dirname $BASH_SOURCE)) ; }
opticks-release-cd(){     cd $(opticks-release-prefix) ; }


opticks-release-usage(){ cat << EOU
$FUNCNAME
=======================

This opticks-release.bash script sets up a minimal environment for the 
use of an Opticks binary distribution.

Use it by including a line in your ~/.bashrc similar to one of the below, 
corresponding to your operating system architecture and package versions::

   source /cvmfs/opticks.ihep.ac.cn/ok/releases/Opticks-0.0.0_alpha/x86_64-slc7-gcc48-geant4_10_04_p02-dbg/bin/release.bash
   source /cvmfs/opticks.ihep.ac.cn/ok/releases/Opticks-0.0.0_alpha/x86_64-centos7-gcc48-geant4_10_04_p02-dbg/bin/release.bash
   
By default Opticks binary distributions do not include Geant4 libs or data, 
those are assumed to be separately provided and environment configured.

The Opticks binary distribution also does not include the shared cache.
The shared cache which contains geocache and rngcache is much larger than the
binary distribution. The administrator of your GPU cluster should instruct 
you regarding the path to the script that will setup access to 
the shared cache. 

FUNCTIONS
----------

opticks-release-main
    environment setup function which is run on sourcing this script

opticks-release-check
    dumps environment 

opticks-release-test
    runs more than 400 tests of the Opticks distribution using ctest
    roughly 20% of them will fail if Geant4 libraries or data are not found.
    Many tests will also fail if access to the shared cache with 
    geocache and rngcache is not configured. 

EOU
}

opticks-release-info(){ cat << EOI
$FUNCNAME
=====================

  opticks-release-source : $(opticks-release-source)
  opticks-release-dir    : $(opticks-release-dir)
  opticks-release-prefix : $(opticks-release-prefix)

EOI
}

opticks-release-check(){ 
   local msg="=== $FUNCNAME :"

   echo $msg PATH 
   echo $PATH | tr ":" "\n" ; 

   echo $msg LD_LIBRARY_PATH
   echo $LD_LIBRARY_PATH | tr ":" "\n" ; 

   echo $msg PYTHONPATH
   echo $PYTHONPATH | tr ":" "\n" ; 

   echo $msg OPTICKS env
   env | grep OPTICKS

   echo $msg G4 env
   env | grep G4

   echo $msg ctest $(which ctest)

}

opticks-release-main(){

   export OPTICKS_INSTALL_PREFIX=$(opticks-release-prefix)   

   export PATH=$(opticks-release-prefix)/bin:$(opticks-release-prefix)/lib:$(opticks-release-prefix)/py/opticks/ana:$PATH 

   export PYTHONPATH=$(opticks-release-prefix)/py:$PYTHONPATH

   source $(opticks-release-prefix)/integration/integration.bash   ## bash precursor hookup, eg tboolean- which is used by some tests

   olocal-(){ echo -n ; }   ## many bash env functions expect this function to be defined
}

opticks-release-logdir(){ echo $HOME/.opticks/logs ; }

opticks-release-test()
{
    local proj=$1
    local msg="== $FUNCNAME :"
    local iwd=$PWD

    local ldir=$(opticks-release-logdir)
    local tdir=$(opticks-release-prefix)/tests
    local tlog

    if [ -n "$proj" -a -d "$tdir/$proj" ]; then
       tdir=$tdir/$proj
       tlog=$ldir/$FUNCNAME-$proj.log
    else
       tlog=$ldir/$FUNCNAME.log
    fi

    cd $tdir
    local tbeg=$(date)
    mkdir -p $(dirname $tlog)

    echo $msg tdir $tdir | tee $tlog
    echo $msg tlog $tlog | tee $tlog
    echo $msg tbeg $tbeg | tee $tlog
    echo $msg tlog $tlog | tee $tlog

    ctest $* --interactive-debug-mode 0 --output-on-failure 2>&1 | tee -a $tlog

    local tend=$(date)

    echo $msg tbeg $tbeg | tee $tlog
    echo $msg tend $tend | tee $tlog
    echo $msg tdir $tdir | tee $tlog
    echo $msg tlog $tlog | tee $tlog

    cd $iwd
}


opticks-release-main

