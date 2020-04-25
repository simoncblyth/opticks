
oe-source(){ echo $BASH_SOURCE ; }
oe-vi(){  vi $BASH_SOURCE ; }
oe-usage(){ cat <<EOU

OE : Opticks Environment Setup
==================================

::

   oe-
       

When opticks is treated as an external to JUNO sim framework 
the junoenv mechanism takes over the envvar setup.
This should not be used when using Opticks as
a JUNO external. In that case the JUNOTOP/bashrc
sets up the paths.


TODO : move PATH setup into here, from opticks-export 

TODO : avoid build needin HOME in PYTHONPATH

* by controlling the environment of the CMake custom command
  used in okc independently from this enviroment : then 
  this script needs to know nothing of the source tree

* All python scripts used in testing will need to be installed, 
  already ?

EOU

}

oe-name(){ echo $(basename $(oe-home)) ; }
oe-local(){ echo ${LOCAL_BASE:-/usr/local} ; }
oe-fold(){   echo $(oe-local)/$(oe-name) ; }
oe-prefix(){ echo $(oe-fold) ; }

oe-home(){ echo $(dirname $BASH_SOURCE) ; }

oe-env() 
{ 
    olocal-;
    opticks-;
    oe-export
}


oe-export-mode(){ echo ${OE_EXPORT_MODE:-append} ; }

oe-export- () 
{ 
    local msg="=== $FUNCNAME :";
    local pfx;
    local libdir;
    local mode=$(oe-export-mode);
    for pfx in $*;
    do
        oe-$mode CMAKE_PREFIX_PATH $pfx;
        local libs="lib lib64";
        for lib in $libs;
        do
            libdir=$pfx/$lib;
            if [ -d "$libdir" ]; then
                oe-$mode PKG_CONFIG_PATH $libdir/pkgconfig;
                [ "$(uname)" == "Linux" ] && oe-$mode LD_LIBRARY_PATH $libdir;
                [ "$(uname)" == "Darwin" ] && oe-$mode DYLD_LIBRARY_PATH $libdir;
            fi;
        done;
    done;
    export CMAKE_PREFIX_PATH;
    export PKG_CONFIG_PATH;
    [ "$(uname)" == "Linux" ] && export LD_LIBRARY_PATH;
    [ "$(uname)" == "Darwin" ] && export DYLD_LIBRARY_PATH
}


oe-export-setup-artificial-env () 
{ 
    : transient testing environment setup;
    : in real usage the detector simulation framework should define the paths, especially : CMAKE_PREFIX_PATH PKG_CONFIG_PATH;
    unset CMAKE_PREFIX_PATH;
    unset PKG_CONFIG_PATH;
    unset LD_LIBRARY_PATH;
    unset DYLD_LIBRARY_PATH;
    unset PYTHONPATH;
    unset CPATH;
    unset MANPATH;
    if [ "$(uname)" == "Darwin" ]; then
        OE_EXPORT_MODE=prepend oe-export- /usr/local/foreign;
    else
        if [ "$(uname)" == "Linux" ]; then
            local sh=$JUNOTOP/bashrc.sh;
            [ -f "$sh" ] && source $sh;
        fi;
    fi
}



oe-export() 
{ 
    : appends standard prefixes to CMAKE_PREFIX_PATH and pkgconfig paths to PKG_CONFIG_PATH and exports them;
    oe-export-setup-artificial-env;
    OE_EXPORT_MODE=append oe-export- $(oe-prefix) $(oe-prefix)/externals;
    OE_EXPORT_MODE=append oe-export- $(oe-prefix)/externals/OptiX


    oe-prepend PATH $(oe-prefix)/lib
    oe-prepend PATH $(oe-prefix)/bin
    oe-prepend PATH $(oe-home)/bin
    oe-prepend PATH $(oe-home)/ana
    # nasty source tree dirs in PATH
}


oe-append () 
{ 
    local var=${1:-PATH};
    local dir=${2:-/tmp};
    local l=${3:-:};
    if [ -z "${!var}" ]; then
        eval $var=$dir;
    else
        [[ "$l${!var}$l" != *"$l${dir}$l"* ]] && eval $var=${!var}$l$dir;
    fi
}

oe-prepend () 
{ 
    local var=${1:-PATH};
    local dir=${2:-/tmp};
    local l=${3:-:};
    if [ -z "${!var}" ]; then
        eval $var=$dir;
    else
        [[ "$l${!var}$l" != *"$l${dir}$l"* ]] && eval $var=$dir$l${!var};
    fi
}






