##
## Copyright (c) 2019 Opticks Team. All Rights Reserved.
##
## This file is part of Opticks
## (see https://bitbucket.org/simoncblyth/opticks).
##
## Licensed under the Apache License, Version 2.0 (the "License"); 
## you may not use this file except in compliance with the License.  
## You may obtain a copy of the License at
##
##   http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software 
## distributed under the License is distributed on an "AS IS" BASIS, 
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
## See the License for the specific language governing permissions and 
## limitations under the License.
##

glm-src(){      echo externals/glm.bash ; }
glm-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(glm-src)} ; }
glm-vi(){       vi $(glm-source) ; }
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


Compare glm::frustum and glm::ortho sources
--------------------------------------------

Scaling left, right, bottom, top by 1/near looks like it 
might bring things closer.

* http://www.songho.ca/opengl/gl_projectionmatrix.html


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
    ///          right = -left     = w/2 
    ///          top   = -bottom   = h/2
    ///
    ///    | 2/w  0   0        -(r+l)/(r-l)   |
    ///    |  0  2/h  0        -(t+b)/(t-b)   |
    ///    |  0   0  -2/(f-n)  -(f+n)/(f-n)   |
    ///    |  0   0   0          1            |
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

    ///
    ///    |   2n/w    0     (r+l)/(r-l)    0              | 
    ///    |    0     2n/h   (t+b)/(t-b)    0              |
    ///    |    0      0    -(f+n)/(f-n)   -2 f n/(f-n)    |
    ///    |    0      0         -1           0            |
    ///


In [28]: simplify(BAP)
Out[28]: 
[ 2*n              l + r         ]
[------    0       -----      0  ]
[-l + r            l - r         ]
[                                ]
[         2*n      b + t         ]
[  0     ------    -----      0  ]
[        -b + t    b - t         ]
[                                ]
[                -(f + n)   2*f*n]
[  0       0     ---------  -----]
[                  f - n    f - n]
[                                ]
[  0       0         1        0  ]

i
In [34]: simplify(BAN)
Out[34]: 
[ 2*n          l + r          ]
[-----    0    ------     0   ]
[l - r         -l + r         ]
[                             ]
[        2*n   b + t          ]
[  0    -----  ------     0   ]
[       b - t  -b + t         ]
[                             ]
[              f + n   -2*f*n ]
[  0      0    -----   -------]
[              f - n    f - n ]
[                             ]
[  0      0      -1       0   ]







EOU
}
glm-env(){     opticks- ;  }
glm-dir(){  echo $(opticks-prefix)/externals/glm/$(glm-name) ; }
glm-dir2(){  echo $(opticks-prefix)/externals/glm/glm ; }
glm-idir(){  echo $(glm-dir)/glm ; }
glm-sdir(){  echo $(opticks-home)/graphics/glm ; }


glm-pc-path(){ echo $(opticks-prefix)/externals/lib/pkgconfig/GLM.pc ; }


glm-tdir(){ echo $(glm-dir)/_test ; }
glm-cd(){   cd $(glm-dir); }
glm-tcd(){  cd $(glm-tdir); }
glm-icd(){  cd $(glm-idir); }
glm-scd(){  cd $(glm-sdir) ; }

glm-edit(){  vi $(opticks-home)/cmake/Modules/FindGLM.cmake ; }

#glm-version(){ echo 0.9.6.3 ; }
glm-version(){ echo 0.9.9.5 ; }   # loadsa warnings with gcc 8.3.1 and CUDA 10.1 nvcc
#glm-version(){ echo 0.9.9.8 ; }  # still loadsa warning, see glm-nvcc-test

glm-current(){ echo $(readlink $(opticks-prefix)/externals/glm/glm) ; }

glm-releases(){ open https://github.com/g-truc/glm/releases ; } ## on Linux : define open in shell  eg for Gnome : open(){ gio open $* ; } 
glm-releases-notes(){ cat << EON

0.9.9.5
    

0.9.6.3
    used until CUDA 10.1 update see 

    * notes/issues/glm-0.9.6.3-nvcc-warnings-dereferencing-type-punned-pointer-will-break-strict-aliasing-rules.rst

EON
}

glm-name(){    echo glm-$(glm-version) ; }
#glm-url(){     echo http://downloads.sourceforge.net/project/ogl-math/$(glm-name)/$(glm-name).zip ; }
glm-url(){    echo https://github.com/g-truc/glm/releases/download/$(glm-version)/$(glm-name).zip ; }
glm-dist(){    echo $(dirname $(glm-dir))/$(basename $(glm-url)) ; }

glm-get(){
   local msg="=== $FUNCNAME :"
   local dir=$(dirname $(glm-dir)) &&  mkdir -p $dir && cd $dir
   local url=$(glm-url)
   local zip=$(basename $url)
   local nam=$(glm-name)
   local opt=$( [ -n "${VERBOSE}" ] && echo "" || echo "-q" )

   local cmd="curl -L -O $url"
   [ ! -s "$zip" ] && echo $cmd && eval $cmd 
   [ ! -f "$zip" ] && echo $msg fFAILED TO DOWNLOAD FROM $url && return 1
   [ ! -s "$zip" ] && echo $msg sFAILED TO DOWNLOAD FROM $url && return 1
   [ ! -d "$nam" ] && unzip $opt $zip -d $nam
   [ ! -d "$nam" ] && echo $msg FAILED TO UNZIP $zip && return 2

   if [ -d glm -a ! -L glm ]; then
      echo $msg ERROR cannot plant symbolic link as a non-link glm directory exists in PWD $PWD
      echo $msg solution is to delete is to delete the dir and rerun glm-- BUT not automating that as want to fix the root cause
   else
      ln -sfnv $(glm-name)/glm glm 
      echo $msg planting symbolic link for access without version in path
   fi  
   return 0 
}

glm-get-notes(){ cat << 'EON'

The zip contains top directory glm without version, so extract into a versioned dir
and then symbolic link to it : this is to allow easier version switching when updating  

Note that when the zips and versioned dirs are already present changing the 
glm-version and rerunning glm-- just changes the symbolic link


Need to use consistent glm version across everything to avoid::

     62%] Linking CXX executable NYJSONTest
    ../libNPY.so: undefined reference to `ImplicitMesherBase::setParam(int, glm::vec<3, float, (glm::qualifier)0> const&, glm::vec<3, float, (glm::qualifier)0> const&, int)'
    ../libNPY.so: undefined reference to `OctreeNode::GenerateVertexIndices(OctreeNode*, std::vector<glm::vec<3, float, (glm::qualifier)0>, std::allocator<glm::vec<3, float, (glm::qualifier)0> > >&, std::vector<glm::vec<3, float, (glm::qualifier)0>, std::allocator<glm::vec<3, float, (glm::qualifier)0> > >&, FGLite*)'


That means on changing glm version must::

    oimplictmesher-;oimplicitmesher--
    odcs-;odcs--    


EON
}


glm-info(){ cat << EOI

glm-name  : $(glm-name)
glm-url   : $(glm-url)
glm-dist  : $(glm-dist)
glm-dir   : $(glm-dir)
glm-idir  : $(glm-idir)
glm-sdir  : $(glm-sdir)

EOI
}

glm-doc(){ open file://$(glm-dir)/glm/doc/api/modules.html ; }
glm-api(){ open file://$(glm-dir)/glm/doc/api/index.html ; }
glm-pdf(){ open file://$(glm-dir)/glm/doc/manual.pdf ; }


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
   glm-get
   [ $? -ne 0 ] && return 1
   glm-pc
   [ $? -ne 0 ] && return 2
   return 0
}

glm-test(){
   glm-test-cmake
   glm-test-make
   #glm-test-make test
}


glm-lookat(){
   clang++ -I$(glm-dir) $(glm-sdir)/lookat.cc -o /tmp/lookat && /tmp/lookat 
}



glm-find(){ find $(glm-idir) -type f -exec grep -H ${1:-rotate} {} \; ; }
glm-lfind(){ find $(glm-idir) -type f -exec grep -l ${1:-rotate} {} \; ; }









glm-nvcc-notes(){ cat << EON

New compilation warnings with CUDA 10.1 nvcc and glm when optimization is switched on
=======================================================================================

See also 

* thrap-glm-test 
* notes/issues/glm-0.9.6.3-nvcc-warnings-dereferencing-type-punned-pointer-will-break-strict-aliasing-rules.rst


/usr/local/cuda-10.1/bin/nvcc
/home/blyth/local/opticks/externals/glm/glm/glm/detail/func_packing.inl: In function ‘glm::uint glm::packUnorm2x16(const vec2&)’:
/home/blyth/local/opticks/externals/glm/glm/glm/detail/func_packing.inl:42:46: warning: dereferencing type-punned pointer will break strict-aliasing rules [-Wstrict-aliasing]
   return reinterpret_cast<uint const &>(Topack);

EON
}

glm-nvcc-(){ cat << EOC
#include <iostream>
//#include <glm/glm.hpp>
#include <glm/packing.hpp>

int main(int argc, char** argv)
{

/*
    std::cout << argv[0] << std::endl ; 
    std::cout << "$FUNCNAME" << std::endl ; 

    glm::ivec2 iv2(1,2);

    glm::vec2 v2(0.f,1.f) ; 
    glm::vec4 v4(0.f,1.f,2.f,3.f) ; 
    assert( v2.y == 1.f );  
    assert( v4.y == 1.f );  
*/
    return 42 ; 
}
EOC
}

glm-nvcc(){
   local iwd=$PWD 
   local tmp=/tmp/$USER/opticks/$FUNCNAME
   rm -rf $tmp && mkdir -p $tmp && cd $tmp

   $FUNCNAME- 
   $FUNCNAME- > $FUNCNAME.cu

   which nvcc

   #nvcc -I$LOCAL_BASE/opticks/externals/glm/glm $FUNCNAME.cu -o $FUNCNAME -O2 -Xcompiler "-Wall,-Wno-comment,-Wno-strict-aliasing"
   nvcc -I$LOCAL_BASE/opticks/externals/glm/glm $FUNCNAME.cu -o $FUNCNAME -O2 -Xcompiler "-Wall,-Wno-comment"

   # switching off strict-aliasing warnings makes it go away 

   ./$FUNCNAME

   cd $iwd
}


glm-prefix(){  echo $(opticks-prefix)/externals/glm/glm ; }


glm-pc-(){ cat << EOP

# use --define-prefix to set prefix to the grandparent of the pcfiledir, eg /usr/local/opticks/externals 
# hmm when using the xlib trick to make the two pkgconfig dirs at the same level, need the externals 

prefix=$(opticks-prefix)
includedir=\${prefix}/externals/glm/glm

Name: GLM
Description: Mathematics 
Version: 0.1.0

Cflags:  -I\${includedir}
Libs: -lstdc++
Requires: 

EOP
}


glm-pc(){ 
   local msg="=== $FUNCNAME :"
   local path=$(glm-pc-path)
   local dir=$(dirname $path)

   [ ! -d "$dir" ] && echo $msg creating dir $dir && mkdir -p $dir 

   glm-pc- > $path 
}


glm-setup(){ cat << EOS
# $FUNCNAME 
EOS
}



glm-nvcc-test-(){ cat << EOC
#include <glm/glm.hpp>
EOC
}
glm-nvcc-test-notes(){ cat << EON

See notes/issues/glm_anno_warnings_with_gcc_831.rst::

   warning: __device__ annotation is ignored on a function("vec") that is explicitly defaulted on its first declaration

suppress the warning with::

   -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored 

EON
}


glm-nvcc-test(){
   : see notes/issues/glm_anno_warnings_with_gcc_831.rst 
   local tmpdir=/tmp/$USER/$FUNCNAME
   mkdir -p $tmpdir
   cd $tmpdir
   local capability=${OPTICKS_COMPUTE_CAPABILITY:-70} 

   $FUNCNAME- > tglm.cu

   local ccbin=$(which cc)
   echo ccbin $ccbin

   nvcc tglm.cu -c -ccbin $ccbin -m64 -I$(glm-dir2) \
  -Xcompiler ,\"-fvisibility=hidden\",\"-fvisibility-inlines-hidden\",\"-fdiagnostics-show-option\",\"-Wall\",\"-Wno-unused-function\",\"-Wno-comment\",\"-Wno-deprecated\",\"-Wno-shadow\",\"-fPIC\",\"-g\" \
  -Xcompiler -fPIC -gencode=arch=compute_$capability,code=sm_$capability -std=c++11 -O2 --use_fast_math -DNVCC \
   -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored 

}


