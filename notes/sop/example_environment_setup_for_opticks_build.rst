example_environment_setup_for_opticks_build
==============================================

As Opticks is intended to be integrated with detector simulation frameworks
it does not by default install many external libs, such as CLHEP,XercesC,Geant4,Boost.
Those externals are assumed to be provided by the detector simulation framework mechanisms
and having mutiple such externals would cause version shear issues. 

Below is a real world example of a bash shell setup for installing Opticks.  
Note that many paths may not be appropriate for your machine, but something
similar can probably be made for work for you. 

The idea is to show the approach, you will need to customize for your situation. 

* NB the command "vip" which opens all the config files listed by vip-paths into vim 


.bash_profile
--------------

::

    # .bash_profile

    source ~/.bashrc

    vip-paths(){ sed 's/#.*//' << EOP
    $HOME/.bash_profile
       $HOME/.bashrc 
          $HOME/.local.bash               # gcc binutils gdb
          $HOME/.python_config

          $HOME/.opticks_externals_config   # CMAKE_PREFIX_PATH : Boost XercesC CLHEP Geant4 (Custom4) 
          $HOME/.opticks_build_config
          $HOME/.opticks_usage_config

    EOP
    }

    vip(){ vim $(vip-paths) ; } 
    gip(){ grep $1 $(vip-paths) ; } 
    ini(){ source ~/.bash_profile ; } 


.bash_profile > .bashrc
------------------------
::

    # .bash_profile > .bashrc
    # notice the modal split between "build" and "usage" : this is for elucidation/clarity 

    # Source global definitions
    if [ -f /etc/bashrc ]; then
       . /etc/bashrc
    fi

    l(){ ls -l $* ; } 
    #t(){ type $* ; }
    t(){ typeset -f $* ; }

    rc(){ RC=$? ; echo RC $RC ; return $RC ; } 
    v(){ vi $(which $1) ; } 
    eo(){ env | grep OPTICKS ; } 
    x(){ exit ; } 

    makepath(){ cat << EOP | grep -v \#
    /usr/local/bin
    /usr/local/sbin
    /usr/bin
    /usr/sbin
    /bin
    /sbin
    /usr/local/cuda-11.7/bin
    EOP
    }
    #/usr/local/cuda-10.1/bin

    join(){ local ifs=$IFS ; IFS="$1"; shift; echo "$*" ; IFS=$ifs ;  }
    PATH=$(join : $(makepath))   ## absolute setting of PATH, to avoid growth 

    source $HOME/.local.bash
    source ~/.python_config

    #mode=none
    mode=build
    #mode=usage

    source ~/.opticks_externals_config

    case $mode in  
       none)    echo -n ;;  
      build)    source ~/.opticks_build_config ;;
      usage)    source ~/.opticks_usage_config ;;
    esac

    if [ -n "$OPTICKS_PREFIX" ]; then
       orp(){ cd $OPTICKS_PREFIX/$1 ; pwd ;  }
       ort(){ cd $OPTICKS_PREFIX/tests/$1 ; pwd ; } 
    fi


    paths () 
    {
        : .bashrc
        local vars="VIP_MODE CMAKE_PREFIX_PATH PKG_CONFIG_PATH PATH LD_LIBRARY_PATH DYLD_LIBRARY_PATH PYTHONPATH MANPATH CPATH";
        local var;
        for var in $vars;
        do
            echo $var;
            echo ${!var} | tr ":" "\n";
            echo;
        done
    }



.bash_profile > .bashrc > .local.bash
----------------------------------------

::

    # .bash_profile > .bashrc > .local.bash : toolchain setup

    local-notes(){ cat << EON

    Using the same compiler and debugger versions as those 
    used for your detector framework is recommended.


    EON
    }

    source /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/gcc/11.2.0/bashrc
    source /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bashrc
    source /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/gdb/12.1/bashrc





.bash_profile > .bashrc > .python_config
--------------------------------------------

::

    # .bash_profile > .bashrc > .python_config
    # THIS IS PROBABLY NO LONGER NEEDED

    ip(){ local py=${1:-dummy.py}; ipython --pdb -i -- $(which $py) ${@:2} ; }
    i(){ ipython $* ; }

    export LC_ALL="en_US.UTF-8"
    export LC_CTYPE="en_US.UTF-8"

    #export PYTHONPATH=$HOME
    #source ~blyth/.miniconda3_config # py37


.bash_profile > .bashrc > .opticks_externals_config
--------------------------------------------------------

::

    # .bash_profile > .bashrc > .opticks_externals_config

    # PATH envvars control the externals that opticks/CMake will build against 
    unset CMAKE_PREFIX_PATH
    unset PKG_CONFIG_PATH

    #source /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/Pre-Release/J23.1.x/ExternalLibs/Boost/1.78.0/bashrc
    #source /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/Pre-Release/J23.1.x/ExternalLibs/Xercesc/3.2.3/bashrc
    #source /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/Pre-Release/J23.1.x/ExternalLibs/CLHEP/2.4.1.0/bashrc
    #source /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/Pre-Release/J23.1.x/ExternalLibs/Geant4/10.04.p02.juno/bashrc 

    #source /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/Pre-Release/J23.1.0-rc3.dc1/ExternalLibs/custom4/0.1.8/bashrc
    # COMMENT ABOVE FOR NOT WITH_CUSTOM4 TEST

    source /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/Pre-Release/J24.1.x-g411/ExternalLibs/Boost/1.82.0/bashrc
    source /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/Pre-Release/J24.1.x-g411/ExternalLibs/Xercesc/3.2.4/bashrc
    source /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/Pre-Release/J24.1.x-g411/ExternalLibs/CLHEP/2.4.7.1/bashrc
    source /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/Pre-Release/J24.1.x-g411/ExternalLibs/Geant4/11.2.0/bashrc




.bash_profile > .bashrc > .opticks_build_config
--------------------------------------------------


* notice the setting of OPTICKS_DOWNLOAD_CACHE to a /cvmfs path this avoids the need to navigate thru firewalls

::

    # .bash_profile > .bashrc > .opticks_build_config   (workstation/simon)
    usage(){ cat << EOU
    ~/.opticks_build_config
    ========================

    Build config should provide the environment to complete opticks-full. 
    Note how sourcing the bashrc rather than using opticks-prepend-prefix 
    allows more intuitive ordering. 

    EOU
    }

    # config system level pre-requisites 
    export OPTICKS_CUDA_PREFIX=/usr/local/cuda-11.7
    export OPTICKS_OPTIX_PREFIX=/cvmfs/opticks.ihep.ac.cn/external/OptiX_750
    export OPTICKS_COMPUTE_CAPABILITY=70

    # config opticks build : wheres the source, where to install etc..
    #export OPTICKS_DOWNLOAD_CACHE=/data/opticks_download_cache
    export OPTICKS_DOWNLOAD_CACHE=/cvmfs/opticks.ihep.ac.cn/opticks_download_cache
    export OPTICKS_HOME=$HOME/opticks

    #export OPTICKS_BUILDTYPE=Release
    export OPTICKS_BUILDTYPE=Debug
    export OPTICKS_PREFIX=/data/simon/local/opticks_${OPTICKS_BUILDTYPE}
      
    export PYTHONPATH=$(dirname $OPTICKS_HOME)     ## HMM FIX: SOURCE TREE?, STOMPING,  TUCK AWAY 

    opticks-(){  [ -r $OPTICKS_HOME/opticks.bash ] && . $OPTICKS_HOME/opticks.bash && opticks-env $* ; }
    opticks-



.bash_profile > .bashrc > .opticks_usage_config
---------------------------------------------------


::

    # .bash_profile > .bashrc > .opticks_usage_config

    source /data/simon/local/opticks_Debug/bashrc
    #source /data/simon/local/opticks_Release/bashrc
    #source /data/simon/local/opticks_release/Opticks-0.0.1_alpha/x86_64-CentOS7-gcc1120-geant4_10_04_p02-dbg/bashrc
    #source /cvmfs/opticks.ihep.ac.cn/ok/releases/Opticks-v0.2.1/x86_64-CentOS7-gcc1120-geant4_10_04_p02-dbg/bashrc

    export TMP=/data/simon/opticks   # override default TMP of /tmp/$USER/opticks
    mkdir -p $TMP                    # whether override or not, need to create 

    export CUDA_VISIBLE_DEVICES=1

    ## NOT: USAGE BUT TOO USEFUL..
    export OPTICKS_HOME=$HOME/opticks
    opticks-(){  [ -r $OPTICKS_HOME/opticks.bash ] && . $OPTICKS_HOME/opticks.bash && opticks-env $* ; }
    opticks-


    export PYTHONPATH=$HOME
    export IPYTHON=/home/blyth/local/env/tools/conda/miniconda3/bin/ipython




