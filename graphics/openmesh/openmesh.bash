# === func-gen- : graphics/openmesh/openmesh fgp graphics/openmesh/openmesh.bash fgn openmesh fgh graphics/openmesh
openmesh-src(){      echo graphics/openmesh/openmesh.bash ; }
openmesh-source(){   echo ${BASH_SOURCE:-$(env-home)/$(openmesh-src)} ; }
openmesh-vi(){       vi $(openmesh-source) ; }
openmesh-usage(){ cat << EOU

OpenMesh
==========

cmake::

    -- Checking the Boost Python configuration
    Checking the Boost Python configuration failed!
    Reason: An error occurred while running a small Boost Python test project.
    Make sure that your Python and Boost Python libraries match.
    Skipping Python Bindings.


Boundary handling
------------------

From docs:

In order to efficiently classify a boundary vertex, the outgoing halfedge of
these vertices must be a boundary halfedge (see OpenMesh::PolyMeshT::is_boundary()).  

Whenever you modify the topology using low-level topology changing functions, 
be sure to guarantee this behaviour (see OpenMesh::PolyMeshT::adjust_outgoing_halfedge())

Related
--------

* https://github.com/memononen/libtess2

Books
------

Geometric Tools for Computer Graphics

* https://books.google.com.tw/books?id=3Q7HGBx1uLIC

  * p340: connected meshes, an algo to split mesh into connected components 

M. Botsch et al. / Geometric Modeling Based on Triangle Meshes

* http://lgg.epfl.ch/publications/2006/botsch_2006_GMT_eg.pdf
* fig10 is most useful 


Splitting Mesh into Connected Components
------------------------------------------

* http://stackoverflow.com/questions/21502416/splitting-mesh-into-connected-components-in-openmesh
* http://www.openflipper.org/media/Documentation/OpenFlipper-1.0.2/MeshInfoT_8cc_source.html

Usage
-------

* http://www.hao-li.com/cs599-ss2015/exercises/exercise1.pdf

Good Starting points
----------------------

TriMesh

* file:///usr/local/env/graphics/openmesh/OpenMesh-4.1/Documentation/a00276.html


Alternates
-----------

* cgal
* vcg/meshlab
* http://gfx.cs.princeton.edu/proj/trimesh2/

EOU
}

openmesh-env(){  elocal- ; opticks- ; }
openmesh-vers(){ echo 4.1 ; }
openmesh-name(){ echo OpenMesh-$(openmesh-vers) ; }
openmesh-url(){  echo http://www.openmesh.org/media/Releases/$(openmesh-vers)/$(openmesh-name).tar.gz ; }

openmesh-edir(){ echo $(opticks-home)/graphics/openmesh ; }
openmesh-old-base(){ echo $(local-base)/env/graphics/openmesh ; }
openmesh-base(){ echo $(opticks-prefix)/externals/openmesh ; }
openmesh-prefix(){ echo $(openmesh-base)/$(openmesh-vers) ; }

openmesh-dir(){  echo $(openmesh-base)/$(openmesh-name) ; }
openmesh-idir(){ echo $(openmesh-base)/$(openmesh-vers) ; }
openmesh-bdir(){ echo $(openmesh-base)/$(openmesh-name).build ; }

openmesh-ecd(){  cd $(openmesh-edir); }
openmesh-cd(){   cd $(openmesh-dir); }
openmesh-bcd(){  cd $(openmesh-bdir); }
openmesh-icd(){  cd $(openmesh-idir); }

openmesh-get(){
   local dir=$(dirname $(openmesh-dir)) &&  mkdir -p $dir && cd $dir
   local url="$(openmesh-url)"
   local tgz=$(basename $url)
   local nam=${tgz/.tar.gz}
   [ ! -f "$tgz" ] && curl -L -O $url
   [ ! -d "$nam" ] && tar zxf $tgz 
}

openmesh-html(){ open $(openmesh-dir)/Documentation/index.html ; }

openmesh-wipe(){
  local bdir=$(openmesh-bdir)
  rm -rf $bdir 

}
openmesh-cmake(){
  local iwd=$PWD
  local bdir=$(openmesh-bdir)
  mkdir -p $bdir

  [ -f "$bdir/CMakeCache.txt" ] && echo $msg already configured : openmesh-wipe then openmesh-cmake to reconfigure && return 

  openmesh-bcd

  cmake $(openmesh-dir) \
       -G "$(opticks-cmake-generator)" \
      -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_INSTALL_PREFIX=$(openmesh-idir) \
      -DBUILD_APPS=OFF 

  cd $iwd
}

openmesh-make(){
  local iwd=$PWD
  openmesh-bcd
  make $* 
  cd $iwd
}

openmesh--(){
  openmesh-get 
  openmesh-cmake
  openmesh-make
  openmesh-make install
}

