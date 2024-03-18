#!/bin/bash -l
usage(){ cat << EOU
examples/UseInstance/go.sh
============================

Minimal example of OpenGL instancing, default test pops up a window with 8 instanced triangles::

    ~/o/examples/UseInstance/go.sh
    TEST=UseInstanceTest ~/o/examples/UseInstance/go.sh run

    TEST=OneTriangleTest ~/o/examples/UseInstance/go.sh run

Issue
------

Find RPATH not working for this on Darwin, BUT it works for other tests ?
So have to manually define the DYLD_LIBRARY_PATH like on Linux with LD_LIBRARY_PATH

EOU
}

opticks-
oe-
om-


path=$(realpath $BASH_SOURCE)
sdir=$(dirname $path)
name=$(basename $sdir)
bdir=/tmp/$USER/opticks/$name/build 

defarg="info_build_run"
arg=${1:-$defarg}

#test=OneTriangleTest 
test=UseInstanceTest
TEST=${TEST:-$test}

vars="BASH_SOURCE TEST arg bdir"

if [ "${arg/info}" != "$arg" ]; then
     for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi

if [ "${arg/build}" != "$arg" ]; then

    rm -rf $bdir && mkdir -p $bdir && cd $bdir && pwd 
    
    om-cmake $sdir 
    make
    [ $? -ne 0 ] && echo $BASH_SOURCE : make error && exit 1 
    make install   
    [ $? -ne 0 ] && echo $BASH_SOURCE : install error && exit 2 
fi

if [ "${arg/manual}" != "$arg" ]; then

      rm -rf $bdir && mkdir -p $bdir && cd $bdir && pwd 

      cmake $sdir 
             \
            -DCMAKE_BUILD_TYPE=$(opticks-buildtype) \
            -DOPTICKS_PREFIX=$(opticks-prefix) \
            -DCMAKE_PREFIX_PATH=$(opticks-prefix)/externals \
            -DCMAKE_INSTALL_PREFIX=$(opticks-prefix) \
            -DCMAKE_MODULE_PATH=$(opticks-home)/cmake/Modules 

      make
      make install   
fi

if [ "${arg/run}" != "$arg" ]; then
    echo executing $TEST
    #om-
    #om-run $TEST
    $TEST
fi

if [ "${arg/lib}" != "$arg" ]; then
    case $(uname) in 
       Darwin) otool -L $(which $TEST) ;;
       Linux) ldd  $(which $TEST) ;;
   esac
fi






