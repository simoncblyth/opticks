#!/bin/bash -l

usage(){ cat << EOU
~/o/examples/UseOptiX7GeometryInstancedGASCompDyn/go.sh
=========================================================


The common prefix change to avoid repeated download of glm has issue of breaking ptx path consistency::

    /tmp/blyth/opticks/examples/                            ptx/UseOptiX7GeometryInstancedGASCompDyn_generated_UseOptiX7GeometryInstancedGASCompDyn.cu.ptx
    /tmp/blyth/opticks/UseOptiX7GeometryInstancedGASCompDyn/ptx/UseOptiX7GeometryInstancedGASCompDyn_generated_UseOptiX7GeometryInstancedGASCompDyn.cu.ptx

Fixed that by not using the common "examples" prefix. 

TODO: instead use opticks externals glm 


EOU
}


optix-prefix(){ echo ${OPTICKS_OPTIX_PREFIX} ; }
[ ! -d "$(optix-prefix)" ] && echo no optix-prefix dir $(optix-prefix) && exit 0 


sdir=$(pwd)
name=$(basename $sdir)


prefix=/tmp/$USER/opticks/$name
#prefix=/tmp/$USER/opticks/examples

export PREFIX=$prefix
export PATH=$PREFIX/bin:$PATH

bdir=$prefix/build 
echo bdir $bdir name $name prefix $prefix


rm -rf $bdir && mkdir -p $bdir 
[ ! -d $bdir ] && exit 1

cd $bdir && pwd 


glm-dir(){  echo $prefix/externals/glm/$(glm-name) ; }
glm-version(){ echo 0.9.9.5 ; }
glm-name(){    echo glm-$(glm-version) ; }
glm-url(){    echo https://github.com/g-truc/glm/releases/download/$(glm-version)/$(glm-name).zip ; }
glm-dist(){    echo $(dirname $(glm-dir))/$(basename $(glm-url)) ; }

glm-get(){
   local msg="=== $FUNCNAME :"
   local iwd=$PWD
   local dir=$(dirname $(glm-dir)) &&  mkdir -p $dir && cd $dir
   local url=$(glm-url)
   local zip=$(basename $url)
   local nam=$(glm-name)
   local opt=$( [ -n "${VERBOSE}" ] && echo "" || echo "-q" )

   local hpp=$nam/glm/glm/glm.hpp
   echo $msg nam $nam PWD $PWD hpp $hpp
   ## curiously directories under /tmp being emptied but directory structure
   ## remains, so have to check file rather than directory existance  

   [ ! -f "$zip" ] && curl -L -O $url
   [ ! -f "$hpp" ] && unzip $opt $zip -d $nam
   ln -sfnv $(glm-name)/glm glm 
   echo symbolic link for access without version in path

   cd $iwd
}


glm-get

 
cmake $sdir \
     -DCMAKE_BUILD_TYPE=Debug \
     -DOptiX_INSTALL_DIR=$(optix-prefix) \
     -DCMAKE_MODULE_PATH=$(optix-prefix)/SDK/CMake \
     -DCMAKE_INSTALL_PREFIX=$prefix

rm -rf $prefix/ptx
rm -rf $prefix/bin
mkdir -p $prefix/{ptx,bin} 

make
[ $? -ne 0 ] && echo $0 : make FAIL && exit 1
make install   
[ $? -ne 0 ] && echo $0 : install FAIL && exit 2

cd $sdir
./run.sh $*

