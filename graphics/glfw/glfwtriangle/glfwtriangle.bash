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


Intended Buffer Flows
----------------------

Debug
~~~~~~

In debug mode primary buffers are OpenGL ones, secondary OptiX "buffers" and
tertiary CUDA/Thrust "buffers" just reference the primaries.

* input geometry buffers loaded from file into primary buffers
* input genstep loaded from file/network into primary buffers 
* primary photon/record/sequence/recsel/phosel buffers created with size based on gensteps 
* photon/record/sequence buffers are populated via an OptiX program launch  
* sequence buffers are indexed using Thrust populating recsel and phosel buffers
  providing photon and record selection based on material or history sequences
* recsel/phosel buffers used by OpenGL/GLSL shaders to select photons/records
* geometry/genstep/photons/record buffers used by OpenGL to visualize 
  the simulation within a render loop

Production
~~~~~~~~~~~

In production mode primary buffers are OptiX ones, secondary CUDA/Thrust "buffers"
just reference the primaries.

* input geometry buffers loaded from file into primary buffers
* input genstep loaded from file/network into primary buffers 
* primary photon buffers created with size based on gensteps 
* photon buffers are populated via an OptiX program launch  


Left Field
----------

* https://github.com/nvpro-samples/gl_optix_composite/blob/master/src/render_optix.cpp

::

    int vboid = buffer->getGLBOId ();
    glBindBuffer ( GL_PIXEL_UNPACK_BUFFER, vboid );     // Bind to the optix buffer

    rtBufferGetGLBOId stores the OpenGL buffer object id in *gl_id if buffer was created with rtBufferCreateFromGLBO. 
    If buffer was not created from an OpenGL Buffer Object *gl_id will be 0 after the call and RT_ERROR_INVALID_VALUE is returned.


Where are the OptiX buffers stored ?
---------------------------------------

* https://devtalk.nvidia.com/default/topic/539422/?comment=3782061

From the Programming Guide chapter 9: In multi-GPU environments 

* INPUT_OUTPUT and OUTPUT buffers are stored on the host.
* But with the RT_BUFFER_GPU_LOCAL flag this is not the case, 
  which is why you saw a significant speedup using it.




Interop including OpenGL : debug mode
----------------------------------------

Interop including OpenGL is intended for visualization to allow
efficient debugging, thus the simplifying assumption of a single
GPU is OK. 

OpenGL must come first, as interop from it is 
in one direction only, so there are two possible approaches.

CudaGLBuffer.hh::


    OpenGL --- CUDA --- Thrust 
                 \
                  \
                   OptiX 

OptiXGLBuffer.hh::

    OpenGL ---  OptiX
                   \
                    \
                     CUDA --- Thrust 


Initially tried to put all three layers into a single class
but found problems with nvcc compilation of thrust functors
which including optix headers, so avoiding the complication by 
attempting to keep thrust and optix as separate as possible. 


Which is best ? 

* Need to try, measure and see
* OptiXGLBuffer may have some advantage of being closer to the without OpenGL situation



Interop without OpenGL : production mode
-----------------------------------------

Such an arrangement aims for efficient multi-GPU usage that is 
workable on compute only nodes. As OptiX provides transparent 
multi-GPU support its best for it to:

* setup CUDA context 
* manage buffer creation 

::

     OptiX
         \
          \
           CUDA --- Thrust 


The converse arrangement of setting up buffers with CUDA first 
is disfavored when multi-GPU operation is intended. 

TODO: find way to check that the CUDAWrap curand loading is being done after OptiX 
has setup its CUDA contexts



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

glfwtriangle-grow-make()
{
   glfwtriangle-cd
   cuda- 
   local name=grow
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
   #local obj=/tmp/grow.o

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

