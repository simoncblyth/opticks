standalone-vi(){ vi $BASH_SOURCE ; }

standalone-usage(){ cat << EOU



EOU
}


standalone--(){
   local sdir=$(pwd)
   local name=$(basename $sdir)
   local prefix=/tmp/$USER/opticks/$name
   mkdir -p $prefix

   export STANDALONE_NAME=$name
   export STANDALONE_PREFIX=$prefix
   export PATH=$STANDALONE_PREFIX/bin:$PATH
   standalone-setup 

   source ../Standalone/glm-setup.bash

   local bdir=$prefix/build 
   echo bdir $bdir name $name prefix $prefix

   rm -rf $bdir 
   mkdir -p $bdir 
   cd $bdir && pwd 

   standalone-build $sdir $prefix

   cd $sdir
}


standalone-libvar(){
    case $(uname) in 
      Darwin) echo DYLD_LIBRARY_PATH ;;
       Linux)  echo LD_LIBRARY_PATH ;;
    esac
}
standalone-libdir(){
    local var=$1 
    local prefix=${!var} 
    if [ -d "$prefix/lib64" ]; then
        echo $prefix/lib64
    elif [ -d "$prefix/lib" ]; then
        echo $prefix/lib
    fi
}

standalone-vars(){ echo OPTICKS_CUDA_PREFIX OPTICKS_OPTIX_PREFIX OPTICKS_COMPUTE_CAPABILITY ; }

standalone-check()
{
    local msg="=== $FUNCNAME :"
    local vars=$(standalone-vars)
    local var 
    for var in $vars ; do 
       [ -z "${!var}" ] && echo $msg MISSING envvar $var ${!var} && return 1
       printf "%30s : %s \n" $var ${!var}
    done
}
standalone-optix-prefix(){ echo $OPTICKS_OPTIX_PREFIX ; }
standalone-cuda-prefix(){  echo $OPTICKS_CUDA_PREFIX ; }
standalone-compute-capability(){ echo $OPTICKS_COMPUTE_CAPABILITY ;  }

standalone-setup()
{
    standalone-check
    local cudalibdir=$(standalone-libdir OPTICKS_CUDA_PREFIX)   
    local optixlibdir=$(standalone-libdir OPTICKS_OPTIX_PREFIX)   

    local libvar=$(standalone-libvar) 
    export $libvar=$cudalibdir:$optixlibdir:${!libvar}
    echo $libvar : ${!libvar}
}

standalone-info(){ 
   local libvar=$(standalone-libvar)
   standalone-check 
   cat << EOI

   standalone-optix-prefix       : $(standalone-optix-prefix)
   standalone-cuda-prefix        : $(standalone-cuda-prefix)
   standalone-compute-capability : $(standalone-compute-capability)

   standalone-libdir OPTICKS_CUDA_PREFIX  : $(standalone-libdir OPTICKS_CUDA_PREFIX)
   standalone-libdir OPTICKS_OPTIX_PREFIX : $(standalone-libdir OPTICKS_OPTIX_PREFIX)

   $libvar 
EOI
   echo ${!libvar} | tr ":" "\n"    
}


standalone-build()
{
    : this expects to be invoked from the build dir 
    pwd 
    ls -l 
    local msg="=== $FUNCNAME :"
    local sdir=$1 
    local prefix=$2  

    echo $msg sdir $sdir prefix $prefix

    rm -rf $prefix/{ptx,bin,ppm}
    mkdir -p $prefix/{ptx,bin,ppm} 

    cmake $sdir \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_PREFIX_PATH=$prefix/externals \
       -DCMAKE_INSTALL_PREFIX=$prefix \
       -DCMAKE_MODULE_PATH=$(standalone-optix-prefix)/SDK/CMake \
       -DOptiX_INSTALL_DIR=$(standalone-optix-prefix) \
       -DCOMPUTE_CAPABILITY=$(standalone-compute-capability)

    make
    make install   
}



standalone-run()
{
   local msg="=== $FUNCNAME :"
   local sdir=$(pwd)
   local name=$(basename $sdir)
   local path=$(which $name)
   local prefix=$(dirname $(dirname $path))  # assumes prefix/bin/$name
   local rc
   echo $msg name $name path $path prefix $prefix
   $name
   rc=$?
   [ ! $rc -eq 0 ] && echo non-zero RC && return 1

   if [ -n "$PPM" ]; then 
       local ppm=$prefix/ppm/$name.ppm
       [ ! -f "$ppm" ] && echo $msg failed to write ppm $ppm && return 1
       echo ppm $ppm

       ls -l $ppm
       open $ppm    ## create an open function such as "gio open" if using gnome
   fi 
}




