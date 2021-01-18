Opticks Setup Script
=============================


.. contents:: Table of Contents
   :depth: 2


Summary
----------

* path `$(opticks-prefix)/bin/opticks-setup.sh`
* appends to the envvars necessary for building Opticks including CMAKE_PREFIX_PATH
* generated within `opticks-full` by `opticks-full-externals`.
* sourced by the `oe-env` bash function that is invoked by the **om** build function.  


How important is the setup script ?
--------------------------------------

The installation of the Opticks sub-projects with *om-install* 
will immediately fail with a configuration error related to not finding 
the BCM package if the setup script has not been sourced.
The script sets up the below vital PATH envvars::  

    CMAKE_PREFIX_PATH
    LD_LIBRARY_PATH
    DYLD_LIBRARY_PATH
    PKG_CONFIG_PATH  
    PATH

The script also runs the Geant4 setup script setting up
envvars starting with G4 that are needed by Geant4 to find its datafiles.



How is the setup script generated ?
----------------------------------------

The *opticks-full* bash function that installs Opticks first installs
the "automated" externals (not the foreign ones) with *opticks-full-externals* then 
it proceeds to *opticks-full-make* as you can see by looking at the bash function:: 


    epsilon:docs blyth$ type opticks-full
    opticks-full () 
    { 
        local msg="=== $FUNCNAME :";
        opticks-info;
        if [ ! -d "$(opticks-prefix)/externals" ]; then
            opticks-full-externals;
        else
            echo $msg using preexisting externals from $(opticks-prefix)/externals;
        fi;
        opticks-full-make
    }


The *opticks-full-make* bash function starts by running *opticks-setup-generate* 
which generates the setup script at the path given by *opticks-setup-path*.::

    epsilon:docs blyth$ type opticks-setup-path
    opticks-setup-path () 
    { 
        echo $(opticks-prefix)/bin/opticks-setup.sh
    }


If the setup script has somehow not been generated the *opticks-full-make* 
function will abort.  If the script is present then the opticks sub projects 
are built with *om-install*. This is all readily apparent by instrospecting 
the bash function::

    epsilon:docs blyth$ type opticks-full-make
    opticks-full-make () 
    { 
        local msg="=== $FUNCNAME :";
        echo $msg START $(date);
        local rc;
        echo $msg generating setup script;
        opticks-setup-generate;
        rc=$?;
        [ $rc -ne 0 ] && return $rc;
        local setup=$(opticks-setup-path);
        [ ! -f "$setup" ] && echo $msg ABORT missing opticks setup script $setup && return 1;
        om-;
        cd_func $(om-home);
        om-install;
        rc=$?;
        [ $rc -ne 0 ] && return $rc;
        opticks-prepare-installation;
        rc=$?;
        [ $rc -ne 0 ] && return $rc;
        echo $msg DONE $(date);
        return 0
    }
    epsilon:docs blyth$ 




Example showing CMAKE_PREFIX_PATH before and after sourcing opticks-setup.sh
-------------------------------------------------------------------------------

::

    epsilon:~ blyth$ echo $CMAKE_PREFIX_PATH | tr ":" "\n"
    /usr/local/opticks_externals/boost
    /usr/local/opticks_externals/g4
    /usr/local/opticks_externals/xercesc
    /usr/local/opticks_externals/clhep

    epsilon:~ blyth$ source $OPTICKS_PREFIX/bin/opticks-setup.sh
    === opticks-setup.sh : build time OPTICKS_PREFIX /usr/local/opticks is consistent with HERE_OPTICKS_PREFIX /usr/local/opticks
    === opticks-setup.sh : consistent CMAKE_PREFIX_PATH between build time and usage
    === opticks-setup.sh :         CMAKE_PREFIX_PATH 
    /usr/local/opticks_externals/boost
    /usr/local/opticks_externals/g4
    /usr/local/opticks_externals/xercesc
    /usr/local/opticks_externals/clhep
    === opticks-setup.sh : consistent PKG_CONFIG_PATH between build time and usage
    === opticks-setup.sh :           PKG_CONFIG_PATH 
    /usr/local/opticks_externals/boost/lib/pkgconfig
    /usr/local/opticks_externals/g4/lib/pkgconfig
    /usr/local/opticks_externals/xercesc/lib/pkgconfig
    /usr/local/opticks_externals/clhep/lib/pkgconfig
    === opticks-setup-        add     append                 PATH /usr/local/cuda/bin
    === opticks-setup-        add     append                 PATH /usr/local/opticks/bin
    === opticks-setup-        add     append                 PATH /usr/local/opticks/lib
    === opticks-setup-        add     append    CMAKE_PREFIX_PATH /usr/local/opticks
    === opticks-setup-        add     append    CMAKE_PREFIX_PATH /usr/local/opticks/externals
    === opticks-setup-        add     append    CMAKE_PREFIX_PATH /usr/local/optix
    === opticks-setup-        add     append      PKG_CONFIG_PATH /usr/local/opticks/lib/pkgconfig
    === opticks-setup-      nodir     append      PKG_CONFIG_PATH /usr/local/opticks/lib64/pkgconfig
    === opticks-setup-        add     append      PKG_CONFIG_PATH /usr/local/opticks/externals/lib/pkgconfig
    === opticks-setup-      nodir     append      PKG_CONFIG_PATH /usr/local/opticks/externals/lib64/pkgconfig
    === opticks-setup-        add     append    DYLD_LIBRARY_PATH /usr/local/opticks/lib
    === opticks-setup-      nodir     append    DYLD_LIBRARY_PATH /usr/local/opticks/lib64
    === opticks-setup-        add     append    DYLD_LIBRARY_PATH /usr/local/opticks/externals/lib
    === opticks-setup-      nodir     append    DYLD_LIBRARY_PATH /usr/local/opticks/externals/lib64
    === opticks-setup-        add     append    DYLD_LIBRARY_PATH /usr/local/cuda/lib
    === opticks-setup-      nodir     append    DYLD_LIBRARY_PATH /usr/local/cuda/lib64
    === opticks-setup-      nodir     append    DYLD_LIBRARY_PATH /usr/local/optix/lib
    === opticks-setup-        add     append    DYLD_LIBRARY_PATH /usr/local/optix/lib64
    epsilon:~ blyth$ 
    epsilon:~ blyth$ 
    epsilon:~ blyth$ echo $CMAKE_PREFIX_PATH | tr ":" "\n"
    /usr/local/opticks_externals/boost
    /usr/local/opticks_externals/g4
    /usr/local/opticks_externals/xercesc
    /usr/local/opticks_externals/clhep
    /usr/local/opticks
    /usr/local/opticks/externals
    /usr/local/optix
    epsilon:~ blyth$ 


The above is from my macOS laptop. On Linux LD_LIBRARY_PATH is used rather than DYLD_LIBRARY_PATH.



How is the setup script used ?
-------------------------------

Tracing the bash functions that normally do the above source of the setup::

    epsilon:~ blyth$ alias t     ## using an alias for "type"
    alias t='type'

    epsilon:misc blyth$ t om 
    om () 
    { 
        om-;
        om-- $*
    }
    epsilon:misc blyth$ t om-
    om- () 
    { 
        . $(opticks-home)/om.bash && om-env $*
    }
    epsilon:misc blyth$ t om-env    # nothing defined yet
    epsilon:misc blyth$ om-         # run the precursor function
    epsilon:misc blyth$ t om-env    # now its defined
    om-env () 
    { 
        olocal-;
        opticks-;
        local msg="=== $FUNCNAME :";
        if [ "$1" == "quiet" -o "$1" == "q" -o -n "$OPTICKS_QUIET" ]; then
            oe- 2> /dev/null;
        else
            echo $msg normal running;
            oe-;
        fi
    }
    epsilon:misc blyth$ t oe-       
    oe- () 
    { 
        . $(opticks-home)/oe.bash && oe-env $*
    }
    epsilon:misc blyth$ t oe-env
    oe-env () 
    { 
        olocal-;
        opticks-;
        source $OPTICKS_PREFIX/bin/opticks-setup.sh 1>&2
    }
    epsilon:misc blyth$ 





