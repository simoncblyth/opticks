# === func-gen- : graphics/glfw/glfwtriangle/glfwtriangle fgp graphics/glfw/glfwtriangle/glfwtriangle.bash fgn glfwtriangle fgh graphics/glfw/glfwtriangle
glfwtriangle-src(){      echo graphics/glfw/glfwtriangle/glfwtriangle.bash ; }
glfwtriangle-source(){   echo ${BASH_SOURCE:-$(env-home)/$(glfwtriangle-src)} ; }
glfwtriangle-vi(){       vi $(glfwtriangle-source) ; }
glfwtriangle-env(){      elocal- ; }
glfwtriangle-usage(){ cat << EOU


* http://antongerdelan.net/opengl/hellotriangle.html

* http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097?pgno=2

* https://gist.github.com/dangets/2926425

* throgl-

* https://groups.google.com/forum/#!searchin/thrust-users/gl/thrust-users/nI34k3laV_E/X6HUm7nRhisJ

* https://groups.google.com/forum/#!topicsearchin/thrust-users/subject$3AOpenGL


EOU
}
glfwtriangle-dir(){ echo $(env-home)/graphics/glfw/glfwtriangle ; }
glfwtriangle-cd(){  cd $(glfwtriangle-dir); }
glfwtriangle-mate(){ mate $(glfwtriangle-dir) ; }
glfwtriangle-get(){
   local dir=$(dirname $(glfwtriangle-dir)) &&  mkdir -p $dir && cd $dir

}


glfwtriangle-make()
{
   glfwtriangle-cd

   glew-
   glfw-

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

   glew-
   glfw-
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



