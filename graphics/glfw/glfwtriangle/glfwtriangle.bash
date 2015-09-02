# === func-gen- : graphics/glfw/glfwtriangle/glfwtriangle fgp graphics/glfw/glfwtriangle/glfwtriangle.bash fgn glfwtriangle fgh graphics/glfw/glfwtriangle
glfwtriangle-src(){      echo graphics/glfw/glfwtriangle/glfwtriangle.bash ; }
glfwtriangle-source(){   echo ${BASH_SOURCE:-$(env-home)/$(glfwtriangle-src)} ; }
glfwtriangle-vi(){       vi $(glfwtriangle-source) ; }
glfwtriangle-usage(){ cat << EOU

GLFWTRIANGLE : minimalist environment for interop testing
===========================================================

Evolution

*glfwtriangle.cc* 
     Pure OpenGL/GLFW3 drawing a triangle 

     Build with *glfwtriangle-make*

*glfwtriangle.cu* 
     First attempt at interop with Thrust, by resorting to 
     compiling everything with nvcc : not a scalable approach.
     Thrust modifies VBO within the render loop, 
     changing the size of the triangle by changing actual vertex data.

     Build with *glfwtriangle-cu-make*

*glfwtriangle_split.cu,InteropBuffer.hh*
     First try at splitting into nvcc and clang compilation units not successful, 
     but did succeed to tidy up the interop using InteropBuffer header 
     and a transform method that takes a functor argument.  

     Problem is that the functor cannot be exposed to clang
     as it uses CUDA specific __device__ __host__ 

     Build with *glfwtriangle-split-make*


*glfwtriangle_split2.cc,InteropBuffer.hh,GrowBuffer.hh,GrowBuffer.cu*
     Succeed to partition into nvcc and clang units by hiding nvcc specifics
     in the *GrowBuffer.cu* implementation, so the header remains
     clean and acceptable to both compilers.

     Build with: 

     *glfwtriangle-split2-growbuffer-make*
     *glfwtriangle-split2-make*


*glfwtriangle_gto.cc,GTOBuffer.hh,GTOBufferImp.hh,GTOBufferImp.cu*
     Bringing OptiX into the interop mix in GTOBuffer.hh causes complications 
     for compilation of thrust functors... so keep dependencies 
     separate.  

     ABORTED DOING ALL THREE GTO TOGETHER 

     Build/test with:

     *glfwtriangle-gto-make*       nvcc compiles optix program to ptx
     *glfwtriangle-gtoimp-make*    nvcc compiles CUDA/thrust imp specifics to obj
     *glfwtriangle-gtobin-make*    clang compile and link with the obj 
     *glfwtriangle-gtobin-run*


*glfwtriangle_cgb.cc,CudaGLBuffer.hh,callgrow.hh,callgrow.cu*

     Instead move the optix handling into the main
 
     Build/test with: *glfwtriangle-cgb*


Refs
-----

* throgl-
* optixminimal-

* http://antongerdelan.net/opengl/hellotriangle.html
* http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097?pgno=2
* https://gist.github.com/dangets/2926425
* https://groups.google.com/forum/#!searchin/thrust-users/gl/thrust-users/nI34k3laV_E/X6HUm7nRhisJ
* https://groups.google.com/forum/#!topicsearchin/thrust-users/subject$3AOpenGL


Intended Workflow for photon generation and hit handling
----------------------------------------------------------

Aiming at handling large photon/record buffers without 
ever allocating them on the host : which looks to be a bottleneck.

* two modes : OpenGL backed, bare compute 

OpenGL backed
~~~~~~~~~~~~~~~

* gen OpenGL buffer of fixed size depending on known photon/record count with NULL data 
* create OptiX buffer fromGLBO

  * there seems to be no special mapping or registering actions needed to use an OpenGL buffer from OptiX   


Complications
~~~~~~~~~~~~~~

* compiling everything with nvcc is not a scalable thing to do, 
  need to split off the Thrust functors and anything else that 
  needs nvcc into .cu and link those objects with the normal clang/gcc
  compiled obj





EOU
}
glfwtriangle-dir(){ echo $(env-home)/graphics/glfw/glfwtriangle ; }
glfwtriangle-cd(){  cd $(glfwtriangle-dir); }

glfwtriangle-env(){      
   elocal- 
   glew-
   glfw-
}

glfwtriangle-make()
{
   glfwtriangle-cd

   local name=glfwtriangle
   local bin=/tmp/$name

   clang $name.cc -o $bin \
        -I$(glew-prefix)/include \
        -I$(glfw-prefix)/include \
        -L$(glew-prefix)/lib -lglew  \
        -L$(glfw-prefix)/lib -lglfw.3  \
        -framework OpenGL
}


glfwtriangle-cu-make()
{
   local msg="$FUNCNAME : "

   glfwtriangle-cd
   cuda- 

   local name=glfwtriangle
   local bin=/tmp/$name

   echo $msg making bin $bin

   nvcc -ccbin /usr/bin/clang $name.cu -o $bin \
        -I$(glew-prefix)/include \
        -I$(glfw-prefix)/include \
        -I$(cuda-prefix)/include \
        -L$(glew-prefix)/lib -lglew  \
        -L$(glfw-prefix)/lib -lglfw.3  \
        -L$(cuda-prefix)/lib -lcudart.7.0  \
        -L/System/Library/Frameworks/OpenGL.framework/Libraries -lGL

   # nvcc cannot handle -framework option
}


glfwtriangle-interop-make()
{
   glfwtriangle-cd
   cuda- 

   local name=InteropBuffer
   clang $name.cc -c -o /tmp/$name.obj \
        -I$(glew-prefix)/include \
        -I$(glfw-prefix)/include \
        -I$(cuda-prefix)/include 
}



glfwtriangle-split-make()
{
   local msg="$FUNCNAME : "

   glfwtriangle-cd
   cuda- 

   local name=glfwtriangle_split
   local bin=/tmp/$name

   echo $msg making bin $bin

   nvcc -ccbin /usr/bin/clang $name.cu -o $bin \
        -I$(glew-prefix)/include \
        -I$(glfw-prefix)/include \
        -I$(cuda-prefix)/include \
        -L$(glew-prefix)/lib -lglew  \
        -L$(glfw-prefix)/lib -lglfw.3  \
        -L$(cuda-prefix)/lib -lcudart.7.0  \
        -L/System/Library/Frameworks/OpenGL.framework/Libraries -lGL

}





glfwtriangle-split2-growbuffer-make()
{
   glfwtriangle-cd
   cuda- 

   local name=GrowBuffer
   nvcc $name.cu -c -o /tmp/$name.obj \
        -I$(glew-prefix)/include \
        -I$(glfw-prefix)/include \
        -I$(cuda-prefix)/include 
}

glfwtriangle-split2-make()
{
   local msg="$FUNCNAME : "

   glfwtriangle-cd
   cuda- 

   local name=glfwtriangle_split2
   local bin=/tmp/$name

   echo $msg making bin $bin

   clang $name.cc /tmp/GrowBuffer.obj -o $bin \
        -I$(glew-prefix)/include \
        -I$(glfw-prefix)/include \
        -I$(cuda-prefix)/include \
        -L$(glew-prefix)/lib -lglew  \
        -L$(glfw-prefix)/lib -lglfw.3  \
        -L$(cuda-prefix)/lib -lcudart.7.0  \
        -L/System/Library/Frameworks/OpenGL.framework/Libraries -lGL \
        -lc++
}




glfwtriangle-gtobuffer-make()
{
   glfwtriangle-cd
   cuda- 
   optix- 

   local name=GTOBuffer
   clang $name.cc -c -o /tmp/$name.obj \
        -I$(glew-prefix)/include \
        -I$(glfw-prefix)/include \
        -I$(cuda-prefix)/include \
        -I$(optix-prefix)/include 
}



glfwtriangle-ptxdir(){ echo /tmp/glfwtriangleptx ; }
glfwtriangle-cgb-make()
{
   glfwtriangle-cd
   cuda- 
   optix- 

   local ptxdir=$(glfwtriangle-ptxdir)
   mkdir -p $ptxdir

   local name=cgb
   nvcc -ptx $name.cu -o $ptxdir/$name.ptx \
        -I$(optix-prefix)/include 
}

glfwtriangle-callgrow-make()
{
   glfwtriangle-cd
   cuda- 
   local name=callgrow
   nvcc $name.cu -c -o /tmp/$name.o 
}

glfwtriangle-cgbbin-make()
{
   local msg="$FUNCNAME : "

   glfwtriangle-cd
   cuda- 
   optix-

   local name=glfwtriangle_cgb
   local bin=/tmp/$name
   local obj=/tmp/callgrow.o

   echo $msg making bin $bin

   clang $name.cc $obj -o $bin \
        -I$(glew-prefix)/include \
        -I$(glfw-prefix)/include \
        -I$(cuda-prefix)/include \
        -I$(optix-prefix)/include \
        -L$(glew-prefix)/lib -lglew  \
        -L$(glfw-prefix)/lib -lglfw.3  \
        -L$(cuda-prefix)/lib -lcudart.7.0  \
        -L$(optix-prefix)/lib64 -loptix.3.8.0 -loptixu.3.8.0  \
        -L/System/Library/Frameworks/OpenGL.framework/Libraries -lGL \
        -lc++ \
        -Xlinker -rpath -Xlinker $(cuda-prefix)/lib \
        -Xlinker -rpath -Xlinker $(optix-prefix)/lib64
}

glfwtriangle-cgbbin-run()
{
    local name=glfwtriangle_cgb
    local bin=/tmp/$name
    PTXDIR=$(glfwtriangle-ptxdir) $bin
}


glfwtriangle-cgb()
{
    glfwtriangle-cgb-make 
    glfwtriangle-callgrow-make 
    glfwtriangle-cgbbin-make 
    glfwtriangle-cgbbin-run
}

