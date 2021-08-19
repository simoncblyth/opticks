#!/bin/bash 

source ./env.sh 

bdir=$CSG_PREFIX/build 
echo $msg bdir $bdir 

rm -rf $bdir && mkdir -p $bdir 
[ ! -d $bdir ] && exit 1

cd $bdir && pwd 


glm-dir(){  echo $CSG_PREFIX/externals/glm/$(glm-name) ; }
#glm-version(){ echo 0.9.9.5 ; }
glm-version(){ echo 0.9.9.8 ; }
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

   [ ! -f "$zip" ] && curl -L -O $url
   [ ! -f "$hpp" ] && unzip $opt $zip -d $nam

   if [ ! -L glm ]; then 
       ln -sfnv $(glm-name)/glm glm 
       echo symbolic link for access without version in path
   fi

   cd $iwd
}
glm-get



cmake $sdir \
     -DCMAKE_BUILD_TYPE=Debug \
     -DCMAKE_INSTALL_PREFIX=$CSG_PREFIX


rm -rf   $CSG_PREFIX/lib
mkdir -p $CSG_PREFIX/lib 

make
[ $? -ne 0 ] && echo $0 : make FAIL && exit 1
make install   
[ $? -ne 0 ] && echo $0 : install FAIL && exit 2

exit 0

