#!/bin/bash

cd $(dirname $(realpath $BASH_SOURCE))

defarg="test_nvcc_line"
arg=${1:-$defarg}

if [[ "$arg" =~ get_nvcc_line ]]; then
    touch CSGOptiX7.cu
    ( cd $OPTICKS_PREFIX/build/CSGOptiX ; VERBOSE=1 make | grep nvcc )
fi

if [[ "$arg" =~ test_nvcc_line ]]; then

   # -D_LIBCUDACXX_HAS_NO_INT128=1 \
   # -DCCCL_DISABLE_INT128_SUPPORT=1 \
   
  #name=CSGOptiX7
  name=CSGOptiX7_optixir_check

  ( cd $OPTICKS_PREFIX/build/CSGOptiX ; /usr/local/cuda-13.1/bin/nvcc \
          -forward-unknown-to-host-compiler \
          -DCONFIG_Release \
          -DOPTICKS_CSGOPTIX \
          -DPRODUCTION \
          -DRNG_PHILOX \
          -DWITH_CUSTOM4 \
          -DWITH_RENDER \
          -DWITH_SER \
          -DWITH_SIMTRACE \
          -DWITH_SIMULATE \
          -DWITH_THRUST \
          --options-file CMakeFiles/CSGOptiX_OPTIX.dir/includes_CUDA.rsp \
          --expt-extended-lambda -O3 -DNDEBUG -std=c++17 "--generate-code=arch=compute_89,code=[compute_89,sm_89]" \
          -Xcompiler=-fPIC --use_fast_math -Xcompiler -fPIC -lineinfo \
          -MD -MT CMakeFiles/CSGOptiX_OPTIX.dir/CSGOptiX7.optixir \
          -MF CMakeFiles/CSGOptiX_OPTIX.dir/CSGOptiX7.optixir.d \
          -x cu -rdc=true -optix-ir /home/blyth/opticks/CSGOptiX/$name.cu -o CMakeFiles/CSGOptiX_OPTIX.dir/CSGOptiX7.optixir
   ) 
fi

if [[ "$arg" =~ ls ]]; then

    dirs="$OPTICKS_PREFIX/optix/objects-Release/CSGOptiX_OPTIX $OPTICKS_PREFIX/optix/objects-Debug/CSGOptiX_OPTIX"
    for dir in $dirs ; do
       if [ -d "$dir" ]; then
          echo $dir
          ls -alst $dir
       fi 
    done

fi


