release-source(){ echo $BASH_SOURCE ; }
release-vi(){     vi $BASH_SOURCE ; } 
release-(){       source $BASH_SOURCE ; }

release-dir(){    echo $(dirname $BASH_SOURCE) ; }
release-prefix(){ echo $(dirname $(dirname $BASH_SOURCE)) ; }
release-cd(){     cd $(release-prefix) ; }


release-usage(){ cat << EOU
$FUNCNAME
============

This release.bash script sets up a minimal environment for the 
use of an Opticks binary distribution.

Use it by including a line similar to the below in your ~/.bashrc::

   source /cvmfs/opticks.ihep.ac.cn/ok/releases/Opticks-0.0.0_alpha/x86_64-slc7-gcc48-geant4_10_04_p02-dbg/bin/release.bash

By default Opticks binary distributions do not include Geant4 libs or data, 
those are assumed to be separately provided and environment configured.


FUNCTIONS
----------

release-main
    environment setup function which is run on sourcing this script

release-check
    dumps environment 

release-ctest
    runs more than 400 tests of the Opticks distribution using ctest
    roughly 20% of them will fail if Geant4 libraries or data are not found  

EOU
}

release-info(){ cat << EOI
$FUNCNAME
============

  release-source : $(release-source)
  release-dir    : $(release-dir)
  release-prefix : $(release-prefix)

EOI
}

release-check(){ 
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

release-main(){

   export OPTICKS_INSTALL_PREFIX=$(release-prefix)   

   export PATH=$(release-prefix)/bin:$(release-prefix)/lib:$(release-prefix)/py/opticks/ana:$PATH 

   export PYTHONPATH=$(release-prefix)/py:$PYTHONPATH

   source $(release-prefix)/integration/integration.bash   ## bash precursor hookup, eg tboolean- which is used by some tests

   olocal-(){ echo -n ; }   ## many bash env functions expect this function to be defined
}

release-logdir(){ echo $HOME/.opticks/logs ; }

release-ctest()
{
    local proj=$1
    local msg="== $FUNCNAME :"
    local iwd=$PWD

    local ldir=$(release-logdir)
    local tdir=$OPTICKS_INSTALL_PREFIX/tests
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


release-main

