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

oof-src(){      echo externals/oof.bash ; }
oof-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(oof-src)} ; }
oof-vi(){       vi $(oof-source) ; }
oof-usage(){ cat << EOU

OpenFlipper
=============

Refs
----

* https://www.openflipper.org/plugins/
* https://www.graphics.rwth-aachen.de/media/papers/moebius_2013_searis1.pdf


Building on Mac
-----------------

* http://openflipper.org/Documentation/latest/a00099.html



1. qt4-mac +debug
2. glew
3. glut


qt4
~~~~

::

    -- Could not find QT5, searching for QT4
    -- Found unsuitable Qt version "" from NOTFOUND
    Qt QTCORE library not found.
    CMake Error at CMakeLists.txt:137 (message):
      Could not found any QT Version.  Please specify QT5_INSTALL_PATH (path to
      bin and include dir) to build with QT5 or specify the QT_QMAKE_EXECUTABLE
      to build with QT4

    simon:cmake blyth$ port contents qt4-mac | grep bin/qmake
      /opt/local/libexec/qt4/bin/qmake


boost python
~~~~~~~~~~~~~~~

    - Looking for Boost Python -- found
    -- Checking the Boost Python configuration
    Checking the Boost Python configuration failed!
    Reason: An error occurred while running a small Boost Python test project.
    Make sure that your Python and Boost Python libraries match.
    Skipping Python Bindings.







Code View
-----------

/usr/local/opticks/externals/openflipper/OpenFlipper-3.1/MeshTools/MeshNavigationT.cc::

     86 template < typename MeshT >
     87 inline
     88 typename MeshT::VertexHandle
     89 findClosestBoundary(MeshT* _mesh , typename MeshT::VertexHandle _vh){
     90 
     91   //add visited property
     92   OpenMesh::VPropHandleT< bool > visited;
     93   _mesh->add_property(visited,"Visited Property" );
     94 
     95   //init visited property
     96   typename MeshT::VertexIter v_it, v_end = _mesh->vertices_end();
     97   for( v_it = _mesh->vertices_begin(); v_it != v_end; ++v_it )
     98     _mesh->property( visited, *v_it ) = false;
     99 
    100   std::queue< typename MeshT::VertexHandle > queue;
    101   queue.push( _vh );
    102 
    103   while(!queue.empty()){
    104     typename MeshT::VertexHandle vh = queue.front();
    105     queue.pop();
    106     if (_mesh->property(visited, vh)) continue;
    107 
    108     for (typename MeshT::VertexOHalfedgeIter voh_it(*_mesh,vh); voh_it.is_valid(); ++voh_it){
    109 
    110       if ( _mesh->is_boundary(*voh_it) ){
    111         _mesh->remove_property(visited);
    112         return _mesh->to_vertex_handle(*voh_it);
    113       }else{
    114         queue.push( _mesh->to_vertex_handle(*voh_it) );
    115       }
    116     }
    117     _mesh->property(visited, vh) = true;
    118   }
    119 
    120   _mesh->remove_property(visited);
    121   return typename MeshT::VertexHandle(-1);
    122 }






EOU
}

oof-env(){  olocal- ; opticks- ; }

oof-info(){ cat << EOI

    name : $(oof-name)
    dist : $(oof-dist)



EOI
}


oof-vers(){ echo 3.1 ; }
oof-name(){ echo OpenFlipper-$(oof-vers) ; }
oof-url(){  echo http://www.openflipper.org/media/Releases/$(oof-vers)/$(oof-name).tar.gz ; }

oof-dist(){ echo $(dirname $(oof-dir))/$(basename $(oof-url)) ; }

oof-app(){ echo $(opticks-prefix)/externals/OpenFlipper.app ; }
oof-run(){ open $(oof-app) ; }

oof-base(){ echo $(opticks-prefix)/externals/openflipper ; }
oof-prefix(){ echo $(opticks-prefix)/externals ; }
oof-idir(){ echo $(oof-prefix) ; }

oof-dir(){  echo $(oof-base)/$(oof-name) ; }
oof-bdir(){ echo $(oof-base)/$(oof-name).build ; }

oof-ecd(){  cd $(oof-edir); }
oof-cd(){   cd $(oof-dir)/$1 ; }
oof-bcd(){  cd $(oof-bdir); }
oof-icd(){  cd $(oof-idir); }


oof-get(){
   local dir=$(dirname $(oof-dir)) &&  mkdir -p $dir && cd $dir
   local url="$(oof-url)"
   local tgz=$(basename $url)
   local nam=${tgz/.tar.gz}
   [ ! -f "$tgz" ] && curl -L -O $url
   [ ! -d "$nam" ] && tar zxf $tgz 
}

oof-doc(){ oof-html ; }
oof-html(){ open $(oof-dir)/DeveloperHelp/index.html ; }


oof-find(){ oof-cd ; find . -type f -exec grep -H ${1:-DefaultTraits} {} \; ; }


oof-wipe(){
  local bdir=$(oof-bdir)
  rm -rf $bdir 

}

oof-edit(){ vi $(opticks-home)/cmake/Modules/FindOpenFlipper.cmake ; }

oof-cmake(){
  local iwd=$PWD
  local bdir=$(oof-bdir)
  mkdir -p $bdir

  #[ -f "$bdir/CMakeCache.txt" ] && echo $msg already configured : oof-configure to reconfigure && return 

  oof-bcd


  # -G "$(opticks-cmake-generator)" \

  local qmake=/opt/local/libexec/qt4/bin/qmake 

  cmake $(oof-dir) \
      -DQT_QMAKE_EXECUTABLE=$qmake \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=$(oof-prefix) 

  cd $iwd
}

oof-configure()
{
   oof-wipe
   oof-cmake $*
}


oof-make(){
  local iwd=$PWD
  oof-bcd

  cmake --build . --config Release --target ${1:-install}

  cd $iwd
}

oof--(){
  oof-get 
  oof-cmake
  oof-make install
}




