opticks-release-source(){ echo $BASH_SOURCE ; }
opticks-release-vi(){     vi $BASH_SOURCE ; } 
opticks-release-(){       source $BASH_SOURCE ; }

opticks-release-dir(){    echo $(dirname $BASH_SOURCE) ; }
opticks-release-prefix(){ echo $(dirname $(dirname $BASH_SOURCE)) ; }
opticks-release-cd(){     cd $(opticks-release-prefix) ; }

opticks-release-usage(){ cat << EOU
bin/opticks-release.bash
=========================

This is sourced from bin/opticks-site.bash with a 
path customized in bin/opticks-site-local.bash to configure
the installation prefix of the binary release. 

The primary role of this script is to define the prefix directory
based on the directory in which this script is installed PREFIX/bin/opticks-release.bash

So this script is intended to be sourced in its installed location::

    . /usr/local/opticks/bin/opticks-release.bash
    opticks-release-info
    opticks-release-check


FUNCTIONS
----------

opticks-release-main
    environment setup function which is run on sourcing this script

opticks-release-check
    dumps environment 

opticks-release-test
    runs hundreds of tests using ctest 

EOU
}

opticks-release-info(){ cat << EOI
bin/opticks-release.bash opticks-release-info
================================================

  opticks-release-source : $(opticks-release-source)
  opticks-release-dir    : $(opticks-release-dir)
  opticks-release-prefix : $(opticks-release-prefix)


  HOME                      : $HOME  
  OPTICKS_USER_HOME         : $OPTICKS_USER_HOME         SEEMS NO LONGER USED ?
      optional envvar that overrides HOME within Opticks

  opticks-release-user-home : $(opticks-release-user-home)

  opticks-release-logdir    : $(opticks-release-logdir)

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

   olocal-(){ echo -n ; }   ## many bash env functions expect this function to be defined
}


opticks-release-user-home(){ echo ${OPTICKS_USER_HOME:-$HOME} ; }

opticks-release-logdir(){ echo $(opticks-release-user-home)/.opticks/logs ; }

opticks-release-cd(){  cd $(opticks-release-prefix) ; } 


opticks-release-test()
{
    # THIS MAKES MORE SENSE UP IN opticks-site perhaps ? 
    # AS THE ENVIRONMENT IS NOT SETUP HERE ?

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

    echo $msg tdir $tdir | tee    $tlog
    echo $msg tlog $tlog | tee -a $tlog
    echo $msg tbeg $tbeg | tee -a $tlog
    echo $msg tlog $tlog | tee -a $tlog

    ctest $* --interactive-debug-mode 0 --output-on-failure 2>&1 | tee -a $tlog

    local tend=$(date)

    echo $msg tbeg $tbeg | tee -a $tlog
    echo $msg tend $tend | tee -a $tlog
    echo $msg tdir $tdir | tee -a $tlog
    echo $msg tlog $tlog | tee -a $tlog

    cd $iwd
}


opticks-release-main

