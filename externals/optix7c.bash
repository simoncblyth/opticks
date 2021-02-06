
optix7c-source(){ echo $BASH_SOURCE ; }
optix7c-vi(){ vi $(optix7c-source) om.bash opticks.bash externals/externals.bash ; }
optix7c-env(){  olocal- ; opticks- ; }
optix7c-usage(){ cat << EOU

optix7c Usage 
===================

See Also
-----------

* https://github.com/search?q=optix7&type=


Build
--------

::

   [blyth@localhost build]$ cmake .. -DOptiX_INSTALL_DIR=$(opticks-prefix)/externals/OptiX_700


Issue 1 : not using C++11 by default
--------------------------------------

Gives:: 

    error: ‘uint32_t’ does not name a type

Fix by adding to CMakeLists.txt::

   set(CMAKE_CXX_STANDARD 14)
   set(CMAKE_CXX_STANDARD_REQUIRED YES)


Issue 2 : nvcc commandline not using C++11 
----------------------------------------------

Fix by adding to CMakeLists.txt after configure_optix is included::

    39 list(APPEND CUDA_NVCC_FLAGS "-std=c++11")


Issue 3 : error: ‘runtime_error’ is not a member of ‘std’
---------------------------------------------------------------

Fixed by adding header to example02_pipelineAndRayGen/optix7.h::

   #include <stdexcept>

Issue 4
-----------

Fix split the line in common/gdt/gdt/math/vec.h

::

    In file included from /home/blyth/local/opticks/externals/optix7c/optix7course/common/gdt/gdt/math/LinearSpace.h:37:0,
                     from /home/blyth/local/opticks/externals/optix7c/optix7course/common/gdt/gdt/math/AffineSpace.h:37,
                     from /home/blyth/local/opticks/externals/optix7c/optix7course/common/glfWindow/GLFWindow.h:20,
                     from /home/blyth/local/opticks/externals/optix7c/optix7course/common/glfWindow/GLFWindow.cpp:17:
    /home/blyth/local/opticks/externals/optix7c/optix7course/common/gdt/gdt/math/vec.h: In instantiation of ‘gdt::vec_t<T, 3> gdt::normalize(const gdt::vec_t<T, 3>&) [with T = float]’:
    /home/blyth/local/opticks/externals/optix7c/optix7course/common/glfWindow/GLFWindow.h:99:86:   required from here
    /home/blyth/local/opticks/externals/optix7c/optix7course/common/gdt/gdt/math/vec.h:308:19: error: no match for ‘operator/’ (operand types are ‘gdt::vec_t<float, 3>’ and ‘double’)
         return v * 1.f/gdt::sqrt(dot(v,v));





Background on bin2c
----------------------

bin2c is a binary tool that comes with CUDA which
creates c code of an array from arbitrary input data.::

    epsilon:1 blyth$ which bin2c
    /usr/local/cuda/bin/bin2c

    epsilon:1 blyth$ pwd
    /tmp/blyth/opticks/evt/g4live/natural/1
    epsilon:1 blyth$ bin2c gs.npy

    #ifdef __cplusplus
    extern "C" {
    #endif

    unsigned char imageBytes[] = {
    0x93,0x4e,0x55,0x4d,0x50,0x59,0x01,0x00,0x46,0x00,0x7b,0x27,0x64,0x65,0x73,0x63,
    ...

Observation : bin2c embedding of ptx inside executable
---------------------------------------------------------

* bin2c appears is a trick to embed ptx within an executable ?

example02_pipelineAndRayGen/CMakeLists.txt::

    cuda_compile_and_embed(embedded_ptx_code devicePrograms.cu)

::

    [blyth@localhost example02_pipelineAndRayGen]$ grep embedded_ptx_code *.*
    CMakeLists.txt:cuda_compile_and_embed(embedded_ptx_code devicePrograms.cu)
    CMakeLists.txt:  ${embedded_ptx_code}
    SampleRenderer.cpp:  extern "C" char embedded_ptx_code[];
    SampleRenderer.cpp:    const std::string ptxCode = embedded_ptx_code;


common/gdt/cmake/configure_optix.cmake::

     49 macro(cuda_compile_and_embed output_var cuda_file)
     50   set(c_var_name ${output_var})
     51   cuda_compile_ptx(ptx_files ${cuda_file})
     52   list(GET ptx_files 0 ptx_file)
     53   set(embedded_file ${ptx_file}_embedded.c)
     54 #  message("adding rule to compile and embed ${cuda_file} to \"const char ${var_name}[];\"")
     55   add_custom_command(
     56     OUTPUT ${embedded_file}
     57     COMMAND ${BIN2C} -c --padd 0 --type char --name ${c_var_name} ${ptx_file} > ${embedded_file}
     58     DEPENDS ${ptx_file}
     59     COMMENT "compiling (and embedding ptx from) ${cuda_file}"
     60     )
     61   set(${output_var} ${embedded_file})
     62 endmacro()

::

    --const (-c)     
        Specify if the generated data array is constant.

    --padd <byte>,...          (-p)     
        Specify trailing bytes.

    --type <type>              (-t)     
        Specify the base type of the generated data array.
        Allowed values for this option:  'char','int','longlong','short'.
        Default value:  'char'.

     --name <name>              (-n)     
        Specify the name of the generated data array.
        Default value:  'imageBytes'.



ssh to gitlab is blocked from IHEP Gold workstation
----------------------------------------------------

* workaround is to route ssh via port 443 
* https://about.gitlab.com/blog/2016/02/18/gitlab-dot-com-now-supports-an-alternate-git-plus-ssh-port/

~/.ssh/config::

    host gitlab.com
      Hostname altssh.gitlab.com
      User git 
      Port 443 
      PreferredAuthentications publickey
      IdentityFile ~/.ssh/id_rsa


crytek sponza model
---------------------

* https://casual-effects.com/g3d/data10/index.html#


Mysteries
-----------

* appears that __align__ is being accepted by gcc in example02_pipelineAndRayGen/SampleRenderer.cpp ?


"struct __align__(8)" Alignment Specifiers
-------------------------------------------

* https://stackoverflow.com/questions/13205742/how-to-specify-alignment-for-global-device-variables-in-cuda
* CUDA guide "Size and Alignment Requirement"

Global memory instructions support reading or writing words of size equal to 1,
2, 4, 8, or 16 bytes. Any access (via a variable or a pointer) to data residing
in global memory compiles to a single global memory instruction if and only if
the size of the data type is 1, 2, 4, 8, or 16 bytes and the data is naturally
aligned (i.e., its address is a multiple of that size).

If this size and alignment requirement is not fulfilled, the access compiles to
multiple instructions with interleaved access patterns that prevent these
instructions from fully coalescing. It is therefore recommended to use types
that meet this requirement for data that resides in global memory.

::

    struct __align__(16) {
        float x;
        float y;
        float z;
    };





Note that examples are not using OpenGL/CUDA interop : so inefficient drawing
--------------------------------------------------------------------------------


* example 4 : optixTrace device code passes pointer to PRD decomposed into two uints
 
  * seems a bizarre when consider are millions of these ? Also it feels like referencing a local about to go out of scope.
  * why not just pass the PRD color values ?
  * payload only stretches to 8 uints so presumably doing this to handle when need 
    to communicate more than this



CUDA_CHECK adds the cuda prefix to the call
----------------------------------------------

::

     26 #define CUDA_CHECK(call)                            \
     27     {                                   \
     28       cudaError_t rc = cuda##call;                                      \
     29       if (rc != cudaSuccess) {                                          \
     30         std::stringstream txt;                                          \
     31         cudaError_t err =  rc; /*cudaGetLastError();*/                  \
     32         txt << "CUDA Error " << cudaGetErrorName(err)                   \
     33             << " (" << cudaGetErrorString(err) << ")";                  \
     34         throw std::runtime_error(txt.str());                            \
     35       }                                                                 \
     36     }






EOU
}


optix7c-name(){ echo optix7course ; }
optix7c-dir(){  echo $(opticks-prefix)/externals/optix7c/$(optix7c-name) ; }
optix7c-bdir(){ echo $(optix7c-dir).build ; } 


optix7c-cd(){  cd $(optix7c-dir) ; } 
optix7c-c(){   cd $(optix7c-dir) ; } 
optix7c-bcd(){  cd $(optix7c-bdir) ; } 


optix7c-url(){ 
  case $USER in 
     blyth) echo git@gitlab.com:simoncblyth/optix7course.git ;; 
         *) echo https://gitlab.com/simoncblyth/optix7course.git ;;
  esac
}


optix7c-get()
{
   local dir=$(dirname $(optix7c-dir)) &&  mkdir -p $dir && cd $dir
   local nam=$(optix7c-name) 
   local url=$(optix7c-url)

   [ ! -d "$nam" ] && git clone $url

}


optix7c-cmake()
{
   local bdir=$(optix7c-bdir)

   rm -rf $bdir
   mkdir -p $bdir && cd $bdir
   cmake $(optix7c-dir) -DOptiX_INSTALL_DIR=$(opticks-prefix)/externals/OptiX_700
}

optix7c--()
{
   optix7c-cmake
   make
}

optix7c-bin(){ cat << EOB
ex01_helloOptix
ex02_pipelineAndRayGen
EOB
}


optix7c-run()
{
   optix7c-bcd
   local bin
   optix7c-bin | while read bin ; do 
      echo $bin
      [ ! -x "$bin" ] && echo not executable && break
      ./$bin
      [ $? -ne 0 ] && echo non-zero return code && break
   done
}


