#!/bin/bash
## NB : no -l : so not accessing the login environment : trying to do most everything in here 

arg=$1

NAME=$(basename $BASH_SOURCE)
MSG="=== $NAME :" 

if [ "$BASH_SOURCE" == "$0" ]; then
   echo $MSG ERROR the $BASH_SOURCE file needs to be sourced not executed
   exit 1   
fi 

opticks-setup- () 
{ 
    local mode=${1:-prepend};
    local var=${2:-PATH};
    local dir=${3:-/tmp};
    local st="";
    : dir exists and is not in the path variable already;
    if [ -d "$dir" ]; then
        if [[ ":${!var}:" != *":${dir}:"* ]]; then
            if [ -z "${!var}" ]; then
                export $var=$dir;
                st="new";
            else
                st="add";
                case $mode in 
                    prepend)
                        export $var=$dir:${!var}
                    ;;
                    append)
                        export $var=${!var}:$dir
                    ;;
                esac;
            fi;
        else
            st="skip";
        fi;
    else
        st="nodir";
    fi;
    printf "=== %s %10s %10s %20s %s\n" $FUNCNAME $st $mode $var $dir
}
opticks-setup-info- () 
{ 
    for var in $*;
    do
        echo $var;
        echo ${!var} | tr ":" "\n";
        echo;
    done
}
opticks-setup-info () 
{ 
    opticks-setup-info- PATH CMAKE_PREFIX_PATH PKG_CONFIG_PATH LD_LIBRARY_PATH DYLD_LIBRARY_PATH;
    echo "env | grep OPTICKS";
    env | grep OPTICKS
}


export OM_SUBS=minimal 
export OPTICKS_HOME=$HOME/opticks
export OPTICKS_PREFIX=/usr/local/opticks_minimal
export OPTICKS_CUDA_PREFIX=/usr/local/cuda
export OPTICKS_OPTIX_PREFIX=/usr/local/optix
export OPTICKS_COMPUTE_CAPABILITY=30
#export OPTICKS_GEOCACHE_PREFIX=/usr/local/opticks  # default is $HOME/.opticks

export TMP=${TMP:-/tmp/$USER/opticks}  

opticks-setup- append PATH $OPTICKS_CUDA_PREFIX/bin   ## nvcc
opticks-setup- append PATH $OPTICKS_PREFIX/bin
opticks-setup- append PATH $OPTICKS_PREFIX/lib

opticks-setup- append CMAKE_PREFIX_PATH $OPTICKS_PREFIX
opticks-setup- append CMAKE_PREFIX_PATH $OPTICKS_PREFIX/externals
opticks-setup- append CMAKE_PREFIX_PATH $OPTICKS_OPTIX_PREFIX

opticks-setup- append PKG_CONFIG_PATH $OPTICKS_PREFIX/lib/pkgconfig
opticks-setup- append PKG_CONFIG_PATH $OPTICKS_PREFIX/lib64/pkgconfig
opticks-setup- append PKG_CONFIG_PATH $OPTICKS_PREFIX/externals/lib/pkgconfig
opticks-setup- append PKG_CONFIG_PATH $OPTICKS_PREFIX/externals/lib64/pkgconfig

# opticks-setup-libpaths-  

if [ "$(uname)" == "Linux" ]; then
    llp=LD_LIBRARY_PATH
else
    llp=DYLD_LIBRARY_PATH
fi 

opticks-setup- append $llp $OPTICKS_PREFIX/lib
opticks-setup- append $llp $OPTICKS_PREFIX/lib64
opticks-setup- append $llp $OPTICKS_PREFIX/externals/lib
opticks-setup- append $llp $OPTICKS_PREFIX/externals/lib64

opticks-setup- append $llp $OPTICKS_CUDA_PREFIX/lib
opticks-setup- append $llp $OPTICKS_CUDA_PREFIX/lib64

opticks-setup- append $llp $OPTICKS_OPTIX_PREFIX/lib
opticks-setup- append $llp $OPTICKS_OPTIX_PREFIX/lib64

#opticks-(){  source $OPTICKS_HOME/opticks.bash && opticks-env $* ; }
#opticks-

opticks-buildtype(){ echo Debug ; }
opticks-optix-prefix(){       echo $OPTICKS_OPTIX_PREFIX ; }
opticks-compute-capability(){ echo $OPTICKS_COMPUTE_CAPABILITY ; }
opticks-prefix(){ echo $OPTICKS_PREFIX ; }  
opticks-git-clone(){  git clone $* ; }
opticks-curl(){  curl -L -O $* ; }

olocal-(){ echo -n ; }
opticks-(){ echo -n ; }


if [ "$arg" == "build" ]; then

    source $OPTICKS_HOME/om.bash 
    source $OPTICKS_HOME/externals/externals.bash 

    bcm-
    bcm--

    glm-
    glm--

    plog-
    plog--

    nljson-
    nljson--

    cd $OPTICKS_HOME

    om-install

else
    echo $msg use arg build to do so 
fi 

usage(){ cat << EOU

Test by running: 

/usr/local/opticks_minimal/lib/CSGOptiXRenderTest
$OPTICKS_PREFIX/lib/CSGOptiXRenderTest

OPTICKS_KEY : $OPTICKS_KEY 
OPTICKS_GEOCACHE_PREFIX : $OPTICKS_GEOCACHE_PREFIX    (default is $HOME/.opticks)

After setting OPTICKS_KEY and copying in the geocache

For scripts that use this executable see::

   cd $OPTICKS_HOME/CSGOptix
   ls -l cxr*.sh

For text rendering it is necessary to define envvar that points to a font file, eg::

   export OPTICKS_STTF_PATH=/usr/local/opticks/externals/imgui/imgui/extra_fonts/Cousine-Regular.ttf


EOU
}

usage

