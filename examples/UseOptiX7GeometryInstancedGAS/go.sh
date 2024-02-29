#!/bin/bash -l

optix-prefix(){ echo $OPTICKS_OPTIX_PREFIX ; }
[ ! -d "$(optix-prefix)" ] && echo no optix-prefix dir $(optix-prefix) && exit 0 


sdir=$(pwd)
name=$(basename $sdir)


prefix=/tmp/$USER/opticks/$name

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
rm -rf $prefix/ppm

mkdir -p $prefix/{ptx,bin,ppm} 

make
make install   

which $name
$name

#open $prefix/ppm/$name.ppm



