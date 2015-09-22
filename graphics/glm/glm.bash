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


Compare glm::frustum and glm::ortho sources
--------------------------------------------

Scaling left, right, bottom, top by 1/near looks like it 
might bring things closer.

::

    150     template <typename T>
    151     GLM_FUNC_QUALIFIER tmat4x4<T, defaultp> ortho
    152     (
    153         T left,
    154         T right,
    155         T bottom,
    156         T top,
    157         T zNear,
    158         T zFar
    159     )
    160     {
    161         tmat4x4<T, defaultp> Result(1);
    162         Result[0][0] = static_cast<T>(2) / (right - left);
    163         Result[1][1] = static_cast<T>(2) / (top - bottom);
    164         Result[2][2] = - static_cast<T>(2) / (zFar - zNear);
    165         Result[3][0] = - (right + left) / (right - left);
    166         Result[3][1] = - (top + bottom) / (top - bottom);
    167         Result[3][2] = - (zFar + zNear) / (zFar - zNear);
    168         return Result;
    169     }

    ///     essentially this is mapping onto a canonical box
    ///     in the normal symmetric case
    ///
    ///          right = -left
    ///          top   = -bottom
    ///
    ///    | 2/w  0   0        0             |
    ///    |  0  2/h  0        0             |
    ///    |  0   0  -2/(f-n)  -(f+n)/(f-n)  |
    ///    |  0   0   0        1             |
    ///
    ///


    189     template <typename T>
    190     GLM_FUNC_QUALIFIER tmat4x4<T, defaultp> frustum
    191     (
    192         T left,
    193         T right,
    194         T bottom,
    195         T top,
    196         T nearVal,
    197         T farVal
    198     )
    199     {
    200         tmat4x4<T, defaultp> Result(0);
    201         Result[0][0] = (static_cast<T>(2) * nearVal) / (right - left);
    202         Result[1][1] = (static_cast<T>(2) * nearVal) / (top - bottom);
    203         Result[2][0] = (right + left) / (right - left);
    204         Result[2][1] = (top + bottom) / (top - bottom);
    205         Result[2][2] = -(farVal + nearVal) / (farVal - nearVal);
    206         Result[2][3] = static_cast<T>(-1);
    207         Result[3][2] = -(static_cast<T>(2) * farVal * nearVal) / (farVal - nearVal);
    208         return Result;
    209     }





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
   #cmake -DGLM_TEST_ENABLE=ON $(glm-dir) 
   cmake $(glm-dir) 
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
   #glm-test-make test
}


glm-lookat(){
   clang++ -I$(glm-dir) $(glm-sdir)/lookat.cc -o /tmp/lookat && /tmp/lookat 
}


