# === func-gen- : graphics/glfw/glfwtriangle/glfwtriangle fgp graphics/glfw/glfwtriangle/glfwtriangle.bash fgn glfwtriangle fgh graphics/glfw/glfwtriangle
glfwtriangle-src(){      echo graphics/glfw/glfwtriangle/glfwtriangle.bash ; }
glfwtriangle-source(){   echo ${BASH_SOURCE:-$(env-home)/$(glfwtriangle-src)} ; }
glfwtriangle-vi(){       vi $(glfwtriangle-source) ; }
glfwtriangle-usage(){ cat << EOU

GLFWTRIANGLE : minimalist environment for interop testing
===========================================================

Evolution

*glfwtriangle.cc* 
     pure OpenGL/GLFW3 drawing a triangle 

     Build with *glfwtriangle-make*

*glfwtriangle.cu* 
     first attempt at interop with Thrust, by resorting to 
     compiling everything with nvcc : not a scalable approach.
     Thrust modifies VBO within the render loop, 
     changing the size of the triangle by changing actual vertex data.

     Build with *glfwtriangle-cu-make*

*glfwtriangle_split.cc,grow_split.cu*
     try splitting with only the functor getting compiled by nvcc

     Build with *glfwtriangle-split-make*


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


glfwtriangle-split-make()
{
   local msg="$FUNCNAME : "

   glfwtriangle-cd
   cuda- 

   local name=glfwtriangle_split
   local bin=/tmp/$name
   local cu_name=grow_split
   local cu_obj=/tmp/$cu_name.obj

   echo $msg making obj $obj

   nvcc -ccbin /usr/bin/clang $cu_name.cu -o $cu_obj \
        -I$(cuda-prefix)/include \
        -L$(cuda-prefix)/lib -lcudart.7.0 

   clang $name.cc -o $bin \
        -I$(glew-prefix)/include \
        -I$(glfw-prefix)/include \
        -I$(cuda-prefix)/include \
        -L$(glew-prefix)/lib -lglew  \
        -L$(glfw-prefix)/lib -lglfw.3  \
        -L$(cuda-prefix)/lib -lcudart.7.0  \
        -framework OpenGL


}





