#!/bin/bash -l

opticks-
opticks-id

go-sdir(){ echo $(opticks-home)/${1:-name} ; }
go-bdir(){ echo $(opticks-prefix)/build/${1:-name} ; }

go-build-()
{
   local sdir=$1 
   local bdir=$2
   local name=$(basename $sdir)
   local rc
   [ ! -d "$sdir" ] && echo $msg missing sdir $sdir && exit 1
   [ ! -d "$bdir" ] && echo $msg missing bdir $bdir && exit 1

   local extra
   case $name in
      optixrap) extra="-DOptiX_INSTALL_DIR=$(opticks-optix-install-dir)" ;;
             *) extra="" ;;
   esac

   cmake $sdir \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_PREFIX_PATH=$(opticks-prefix)/externals \
       -DCMAKE_INSTALL_PREFIX=$(opticks-prefix) \
       -DCMAKE_MODULE_PATH=$(opticks-home)/cmake/Modules \
       $extra

   rc=$?
   [ "$rc" != "0" ] && echo $msg $(pwd) non-zero rc $rc && exit 1

   make
   [ "$rc" != "0" ] && echo $msg $(pwd) non-zero rc $rc && exit 1

   if [ "$(uname)" == "Darwin" ]; then
      echo "kludge sleeping for 2s"
      sleep 2
   fi 

   make install   
   [ "$rc" != "0" ] && echo $msg $(pwd) non-zero rc $rc && exit 1
}

go-build()
{
   local iwd=$PWD
   local sub
   opticks-deps --subdirs 2>/dev/null | while read sub 
   do 
       echo $sub
       local sdir=$(go-sdir $sub)
       local bdir=$(go-bdir $sub)

       #rm -rf $bdir 
       mkdir -p $bdir 
       cd $bdir

       go-build- $sdir $bdir
   done
   cd $iwd
}

go-test-()
{
   local sdir=$1 
   local bdir=$2

   [ ! -d "$sdir" ] && echo $msg missing sdir $sdir && exit 1
   [ ! -d "$bdir" ] && echo $msg missing bdir $bdir && exit 1

   opticks-t $bdir  
}
go-test()
{
   local iwd=$PWD
   local sub
   opticks-deps --subdirs 2>/dev/null | while read sub 
   do 
       echo $sub
       local sdir=$(go-sdir $sub)
       local bdir=$(go-bdir $sub)

       go-test- $sdir $bdir
   done
   cd $iwd
}


go-build
#go-test


