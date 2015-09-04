# === func-gen- : cuda/nvcc/nvcc fgp cuda/nvcc/nvcc.bash fgn nvcc fgh cuda/nvcc
nvcc-src(){      echo cuda/nvcc/nvcc.bash ; }
nvcc-source(){   echo ${BASH_SOURCE:-$(env-home)/$(nvcc-src)} ; }
nvcc-vi(){       vi $(nvcc-source) ; }
nvcc-env(){      elocal- ; }
nvcc-usage(){ cat << EOU

NVCC Experience
=================


thrust
-------

Lots of Thrust will compile when in .cpp or .cu files BUT 
often what you get doesnt work how you want it to when compiled from .cpp 
(main example is optix/cuda/thrust interop).
Pragmatic solution to Thrust problems is to move 
as much as possible into .cu

Normally that means tedious development of headers that both compilers
can stomach to provide a bridge between the worlds.  But that is slow

Heterogenous C++ Class definition
-----------------------------------

By this I mean classes with some methods compiled by the host compiler
and some by the NVIDIA compiler.
The advantage is that do not need to laboriously bridge between the worlds
with separate headers (other than the  header of the class itself) 
can just use class members directly.

The bridging is using the implicit this parameter. 


boost issues
--------------

* http://stackoverflow.com/questions/8138673/why-does-nvcc-fails-to-compile-a-cuda-file-with-boostspirit

nvcc sometimes has trouble compiling complex template code such as is found in
Boost, even if the code is only used in __host__ functions.

When a file's extension is .cpp, nvcc performs no parsing itself and instead
forwards the code to the host compiler, which is why you observe different
behavior depending on the file extension.

If possible, try to quarantine code which depends on Boost into .cpp files
which needn't be parsed by nvcc.


hiding things from compilers
-----------------------------

Pimpl/opaque_pointer https://en.wikipedia.org/wiki/Opaque_pointer







EOU
}
nvcc-dir(){ echo $(local-base)/env/cuda/nvcc/cuda/nvcc-nvcc ; }
nvcc-cd(){  cd $(nvcc-dir); }
nvcc-mate(){ mate $(nvcc-dir) ; }
nvcc-get(){
   local dir=$(dirname $(nvcc-dir)) &&  mkdir -p $dir && cd $dir

}
