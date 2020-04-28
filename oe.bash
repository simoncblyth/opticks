
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

    #oe-export
    source $OPTICKS_PREFIX/bin/opticks-setup.sh 
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
    : warning : significant changes here almost certainly force a cleaninstall of opticks as this changes the externals

    unset CMAKE_PREFIX_PATH;
    unset PKG_CONFIG_PATH;
    unset LD_LIBRARY_PATH;
    unset DYLD_LIBRARY_PATH;
    unset PYTHONPATH;
    unset CPATH;
    unset MANPATH;
    if [ "$(uname)" == "Darwin" ]; then

       echo -n
       # OE_EXPORT_MODE=prepend oe-export- /usr/local/foreign;
    else
        if [ "$(uname)" == "Linux" ]; then
            local sh=$JUNOTOP/bashrc.sh;
            [ -f "$sh" ] && source $sh;
        fi;
    fi
}

oe-find()
{
   local pkg=${1:-Geant4}
   echo pkg_config.py $pkg
        pkg_config.py $pkg
   echo find_package.py $pkg
        find_package.py $pkg

}



oe-export-geant4()
{
    if [ -z "$G4ENSDFSTATEDATA" ]; then 
        local g4prefix=$(opticks-config --prefix geant4) 
        local g4sh=$g4prefix/bin/geant4.sh 
        if [ -f "$g4sh" ]; then 
            echo $msg g4prefix $g4prefix g4sh $g4sh 
            source $g4sh 
        else
            echo $msg g4prefix $g4prefix g4sh $g4sh  FAILED
        fi 
    fi     
}

oe-export-misc()
{
    export TMP=/tmp/$USER/opticks
    export OPTICKS_EVENT_BASE=$TMP
}

oe-export-cuda()
{
    # system PATH is assumed to have nvcc in it prior to opticks 

    [ "$(which nvcc)" == "" ] && return 
    local prefix=$(dirname $(dirname $(which nvcc)))

    case $(uname) in
       Linux) oe-append LD_LIBRARY_PATH   $prefix/lib64 $prefix/lib ;; 
      Darwin) oe-append DYLD_LIBRARY_PATH $prefix/lib64 $prefix/lib ;; 
    esac
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

    oe-export-geant4
    oe-export-cuda
    oe-export-misc
}

oe-export-notes () 
{ 
    cat <<EON

append
    less precedence to the dir
preprend
    more precedence to the dir

previously opticks-export did the below ...
with PATHs from both installed and source trees
with source tree ana and bin taking precedence. 
This seems unhealthy.

    opticks-path-add $(opticks-prefix)/lib;
    opticks-path-add $(opticks-prefix)/bin;
    opticks-path-add $(opticks-home)/bin;
    opticks-path-add $(opticks-home)/ana;
    opticksdata-;
    opticksdata-export

EON

}


oe-unset() 
{ 
    unset CMAKE_PREFIX_PATH;
    unset PKG_CONFIG_PATH;
    unset LD_LIBRARY_PATH;
    unset DYLD_LIBRARY_PATH;
    unset PYTHONPATH;
    unset CPATH;
    unset MANPATH;
}


oe-append()
{
    local var=$1
    shift 
    local dir
    for dir in $* ; do 
       [ -d "$dir" ] && oe-append- $var $dir
    done
}

oe-prepend()
{
    local var=$1
    shift 
    local dir
    for dir in $* ; do 
       [ -d "$dir" ] && oe-prepend- $var $dir
    done
}

oe-append-() 
{ 
    local var=${1:-PATH}
    local dir=${2:-/tmp}
    local l=:
    if [ -z "${!var}" ]; then
        eval $var=$dir;
    else
        [[ "$l${!var}$l" != *"$l${dir}$l"* ]] && eval $var=${!var}$l$dir;
    fi
}

oe-prepend-() 
{ 
    local var=${1:-PATH};
    local dir=${2:-/tmp};
    local l=:
    if [ -z "${!var}" ]; then
        eval $var=$dir;
    else
        [[ "$l${!var}$l" != *"$l${dir}$l"* ]] && eval $var=$dir$l${!var};
    fi
}


oe-info () 
{ 
    echo CMAKE_PREFIX_PATH;
    echo $CMAKE_PREFIX_PATH | tr ":" "\n";
    echo PKG_CONFIG_PATH;
    echo $PKG_CONFIG_PATH | tr ":" "\n";
    if [ "$(uname)" == "Linux" ]; then
        echo LD_LIBRARY_PATH;
        echo $LD_LIBRARY_PATH | tr ":" "\n";
    else
        if [ "$(uname)" == "Darwin" ]; then
            echo DYLD_LIBRARY_PATH;
            echo $DYLD_LIBRARY_PATH | tr ":" "\n";
        fi;
    fi;
    echo PYTHONPATH;
    echo $PYTHONPATH | tr ":" "\n";
    echo PATH;
    echo $PATH | tr ":" "\n"
}

