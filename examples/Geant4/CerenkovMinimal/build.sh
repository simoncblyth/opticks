#!/bin/bash -l
msg="=== $BASH_SOURCE :"
usage(){ cat << EOU
::

   ./build.sh         ## build using existing build directory 
   ./build.sh clean   ## full build by first delecting build directory 

EOU
}

./check.sh 
[ $? -ne 0 ] && echo check failed && exit 1

sdir=$(pwd)
name=$(basename $sdir)
bdir=/tmp/$USER/opticks/examples/$name/build 

[ "$1" == "clean" ] && echo $msg removing bdir $bdir && rm -rf $bdir
 
if [ ! -d "$bdir" -o ! -f "$bdir/Makefile" ]; then 
   mkdir -p $bdir && cd $bdir 
   cmake $sdir \
      -DCMAKE_BUILD_TYPE=Debug \
      -DOPTICKS_PREFIX=$OPTICKS_PREFIX \
      -DCMAKE_INSTALL_PREFIX=$OPTICKS_PREFIX \
      -DCMAKE_MODULE_PATH=$OPTICKS_HOME/cmake/Modules 
else
   cd $bdir 
fi 

pwd
make
[ $? -ne 0 ] && echo $msg make error && exit 1 
make install   

