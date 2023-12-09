#!/bin/bash -l 
usage(){ cat << EOU
preprocessor.sh 
=================

This applies just the preprocessor to flatten a 
source file with a tree of includes into a single 
file with no includes. 

The reason to do this is to have easy search access 
to all the code that is being compiled. For example
to find inadvertent printf or double. 


https://gcc.gnu.org/onlinedocs/gcc/Preprocessor-Options.html

-E 
    nothing is done except preprocessing    
-C
    dont discard comments
-CC
    dont discard comments even in macros, also changes C++ style comments into C ones

-P 
    inhibit generation of line markers


Navigation Tips
=================

* 91036 lines 

Opticks headers comments have RST underlines::

    :^=======
    :^-----




EOU
}


cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}

opticks_optix_prefix=/usr/local/OptiX_750
OPTICKS_OPTIX_PREFIX=${OPTICKS_OPTIX_PREFIX:-$opticks_optix_prefix}

src=$OPTICKS_HOME/CSGOptiX/CSGOptiX7.cu

gcc \
   -E -C -P \
   -D__CUDACC__ \
   -DPRODUCTION \
   -I$OPTICKS_OPTIX_PREFIX/include \
   -I$CUDA_PREFIX/include \
   -I$OPTICKS_PREFIX/include/SysRap \
   -I$OPTICKS_PREFIX/include/QUDARap \
   -I$OPTICKS_PREFIX/include/CSG \
   -I$OPTICKS_PREFIX/include/CSGOptiX \
   -std=c++11 \
   -xc++ $src

