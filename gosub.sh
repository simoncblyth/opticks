#!/bin/bash -l

opticks-
opticks-id

home=$(pwd)

gosub()
{
   local sdir=$(pwd)
   local name=$(basename $sdir)
   local bdir=$(opticks-prefix)/build/$name

   #rm -rf $bdir 
   mkdir -p $bdir && cd $bdir && pwd 

   cmake $sdir \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_PREFIX_PATH=$(opticks-prefix)/externals \
       -DCMAKE_INSTALL_PREFIX=$(opticks-prefix) \
       -DCMAKE_MODULE_PATH=$(opticks-home)/cmake/Modules \
       -DOptiX_INSTALL_DIR=$(opticks-optix-install-dir)

   make

   if [ "$(uname)" == "Darwin" ]; then
      echo "kludge sleeping for 2s"
      sleep 2
   fi 

   make install   
}




opticks-deps --subdirs 2>/dev/null | while read sub ; do 
   echo $sub 
   [ ! -d $sub ] && echo missing sub $dir && exit 1

   cd $home/$sub

   gosub

   rc=$?
   [ "$rc" != "0" ] && echo $msg $(pwd) non-zero rc $rc && exit 1

   cd $home 
done

