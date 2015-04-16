# === func-gen- : graphics/glm/glm fgp graphics/glm/glm.bash fgn glm fgh graphics/glm
glm-src(){      echo graphics/glm/glm.bash ; }
glm-source(){   echo ${BASH_SOURCE:-$(env-home)/$(glm-src)} ; }
glm-vi(){       vi $(glm-source) ; }
glm-env(){      elocal- ; }
glm-usage(){ cat << EOU

GLM : OpenGL Mathematics
==========================

* http://glm.g-truc.net/0.9.6/index.html

OpenGL Mathematics (GLM) is a header only C++ mathematics library for graphics
software based on the OpenGL Shading Language (GLSL) specifications.

GLM provides classes and functions designed and implemented with the same
naming conventions and functionalities than GLSL so that when a programmer
knows GLSL, he knows GLM as well which makes it really easy to use.

This project isn't limited to GLSL features. An extension system, based on the
GLSL extension conventions, provides extended capabilities: matrix
transformations, quaternions, data packing, random numbers, noise, etc...


See also
---------

* env/cmake/Modules/FindGLM.cmake

EOU
}
glm-dir(){  echo $(local-base)/env/graphics/glm/$(glm-name) ; }
glm-idir(){  echo $(glm-dir)/glm ; }
glm-sdir(){  echo $(env-home)/graphics/glm ; }
glm-tdir(){ echo $(glm-dir)/_test ; }
glm-cd(){   cd $(glm-dir); }
glm-tcd(){  cd $(glm-tdir); }
glm-icd(){  cd $(glm-idir); }
glm-scd(){  cd $(glm-sdir) ; }

glm-version(){ echo 0.9.6.3 ; }
glm-name(){    echo glm-$(glm-version) ; }
glm-url(){     echo http://downloads.sourceforge.net/project/ogl-math/$(glm-name)/$(glm-name).zip ; }

glm-get(){
   local dir=$(dirname $(glm-dir)) &&  mkdir -p $dir && cd $dir
   local url=$(glm-url)
   local zip=$(basename $url)
   local nam="glm"
   [ ! -f "$zip" ] && curl -L -O $url
   [ ! -d "$nam" ] && unzip $zip
   ln -sfnv $nam $(glm-name) 
   echo WARNING : symbolic link workaround for non-standard unziping to unversioned glm directory 
}

glm-doc(){ open file://$(glm-dir)/doc/api/modules.html ; }
glm-pdf(){ open file://$(glm-dir)/doc/glm.pdf ; }

glm-find()
{
   glm-icd
   find . -name '*.hpp' -exec grep -H ${1:-scale} {} \;
   find . -name '*.inl' -exec grep -H ${1:-scale} {} \;
}

glm-test-cmake(){
   local iwd=$PWD
   local tdir=$(glm-tdir)
   mkdir -p $tdir
   glm-tcd
   cmake -DGLM_TEST_ENABLE=ON $(glm-dir) 
   cd $iwd
}

glm-test-make(){
   local iwd=$PWD
   glm-tcd
   make $*
   cd $iwd
}

glm--()
{
   glm-test-cmake
   glm-test-make
   glm-test-make test
}


glm-lookat(){
   clang++ -I$(glm-dir) $(glm-sdir)/lookat.cc -o /tmp/lookat && /tmp/lookat 
}


