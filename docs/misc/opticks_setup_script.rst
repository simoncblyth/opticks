Opticks Setup Script
=============================

* path `$(opticks-prefix)/bin/opticks-setup.sh`
* appends to the envvars necessary for building Opticks including CMAKE_PREFIX_PATH
* generated within `opticks-full` by `opticks-full-externals`.
* sourced by the `oe-env` bash function that is invoked by the **om** build function.  



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


How the setup script is used
------------------------------

Tracing the bash functions that normally do the above source of the setup::

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



