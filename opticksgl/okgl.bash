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

okgl-source(){   echo ${BASH_SOURCE} ; }
okgl-vi(){       vi $(okgl-source) ; }
okgl-usage(){ cat << EOU

OpticksGL
==========

Classes depending on OptiX optixrap- and OpenGL

Classes
--------

OpViz
    High level connector between opticksop-/OpEngine and oglrap-/Scene
    with *render* method that performs OptiX ray trace and 
    pushes results to OpenGL texture

ORenderer
    connection between OpenGL renderer and OptiX ray trace

OFrame
    mechanics of OpenGL texture and OptiX buffer handling 
    and pushing from buffer to texture
    TODO: maybe dont provide public header for this




EOU
}


okgl-sdir(){ echo $(opticks-home)/opticksgl ; }
okgl-tdir(){ echo $(opticks-home)/opticksgl/tests ; }
okgl-idir(){ echo $(opticks-idir) ; }
okgl-bdir(){ echo $(opticks-bdir)/opticksgl ; }

okgl-c(){    cd $(okgl-sdir); }
okgl-cd(){   cd $(okgl-sdir); }
okgl-scd(){  cd $(okgl-sdir); }
okgl-tcd(){  cd $(okgl-tdir); }
okgl-icd(){  cd $(okgl-idir); }
okgl-bcd(){  cd $(okgl-bdir); }


okgl-env(){  
   olocal- 
   opticks-
}


okgl-name(){ echo OpticksGL ; }
okgl-tag(){ echo OKGL ; }

okgl-wipe(){ local bdir=$(okgl-bdir) ; rm -rf $bdir ; } 

okgl-apihh(){  echo $(okgl-sdir)/$(okgl-tag)_API_EXPORT.hh ; }
okgl---(){     touch $(okgl-apihh) ; okgl--  ; } 



okgl-t(){ okgl- ; okgl-cd ; om- ; om-test ; }

okgl--(){                   opticks-- $(okgl-bdir) ; } 
okgl-ctest(){               opticks-ctest $(okgl-bdir) $* ; } 
okgl-genproj() { okgl-scd ; opticks-genproj $(okgl-name) $(okgl-tag) ; } 
okgl-gentest() { okgl-tcd ; opticks-gentest ${1:-Example} $(okgl-tag) ; } 
okgl-txt(){ vi $(okgl-sdir)/CMakeLists.txt $(okgl-tdir)/CMakeLists.txt ; } 



