# === func-gen- : graphics/openmesh/openmesh fgp externals/openmesh.bash fgn openmesh fgh graphics/openmesh
openmesh-src(){      echo externals/openmesh.bash ; }
openmesh-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(openmesh-src)} ; }
openmesh-vi(){       vi $(openmesh-source) ; }
openmesh-usage(){ cat << EOU

OpenMesh
==========


* http://openmesh.org/Documentation/OpenMesh-Doc-Latest/a00030.html


* https://mailman.rwth-aachen.de/mailman/listinfo/openmesh
  
  Argh unsearchable mailing list 


* https://graphics.rwth-aachen.de:9000/OpenMesh/OpenMesh

  Web interface to git repo, issue tracker etc..


* https://graphics.rwth-aachen.de:9000/OpenMesh/OpenMesh/commit/338086152b8d5cfce75580c76e445c1de9d80381
* https://graphics.rwth-aachen.de:9000/OpenMesh/OpenMesh/blob/master/src/Unittests/unittests_delete_face.cc

  gtest unittests 


* https://www.openmesh.org/media/Documentations/OpenMesh-4.1-Documentation/index.html




cmake::

    -- Checking the Boost Python configuration
    Checking the Boost Python configuration failed!
    Reason: An error occurred while running a small Boost Python test project.
    Make sure that your Python and Boost Python libraries match.
    Skipping Python Bindings.



Flag Consistency
-----------------

::

    (Link target) ->
      OpenMeshCored.lib(BaseProperty.obj) : error LNK2038: mismatch detected for '_ITERATOR_DEBUG_LEVEL': 
      value '2' doesn't match value '0' in MESHRAP_LOG.obj [C:\usr\local\opticks\build\graphics\openmeshrap\OpenMeshRap.vcxproj]

      OpenMeshCored.lib(BaseProperty.obj) : error LNK2038: mismatch detected for 'RuntimeLibrary':
      value 'MDd_DynamicDebug' doesn't match value 'MD_DynamicRelease' in MESHRAP_LOG.obj [C:\usr\local\opticks\build\graphics\openmeshrap\OpenMeshRap.vcxproj]

Warnings
--------------

::

    "C:\usr\local\opticks\build\graphics\openmeshrap\OpenMeshRap.vcxproj" (default target) (88) ->
    (ClCompile target) ->
      C:\usr\local\opticks\externals\include\OpenMesh/Core/Mesh/AttribKernelT.hh(140): warning C4127: conditional expression is constant [C:\usr\local\opticks\build\graphics\openmeshrap\OpenMeshRap.vcxproj]
      C:\usr\local\opticks\externals\include\OpenMesh/Core/Mesh/AttribKernelT.hh(143): warning C4127: conditional expression is constant [C:\usr\local\opticks\build\graphics\openmeshrap\OpenMeshRap.vcxproj]
      C:\usr\local\opticks\externals\include\OpenMesh/Core/Mesh/AttribKernelT.hh(146): warning C4127: conditional expression is constant [C:\usr\local\opticks\build\graphics\openmeshrap\OpenMeshRap.vcxproj]
      C:\usr\local\opticks\externals\include\OpenMesh/Core/Mesh/AttribKernelT.hh(149): warning C4127: conditional expression is constant [C:\usr\local\opticks\build\graphics\openmeshrap\OpenMeshRap.vcxproj]
      C:\usr\local\opticks\externals\include\OpenMesh/Core/Mesh/AttribKernelT.hh(152): warning C4127: conditional expression is constant [C:\usr\local\opticks\build\graphics\openmeshrap\OpenMeshRap.vcxproj]
      C:\usr\local\opticks\externals\include\OpenMesh/Core/Mesh/AttribKernelT.hh(155): warning C4127: conditional expression is constant [C:\usr\local\opticks\build\graphics\openmeshrap\OpenMeshRap.vcxproj]
      C:\usr\local\opticks\externals\include\OpenMesh/Core/Mesh/AttribKernelT.hh(158): warning C4127: conditional expression is constant [C:\usr\local\opticks\build\graphics\openmeshrap\OpenMeshRap.vcxproj]
      C:\usr\local\opticks\externals\include\OpenMesh/Core/Mesh/AttribKernelT.hh(161): warning C4127: conditional expression is constant [C:\usr\local\opticks\build\graphics\openmeshrap\OpenMeshRap.vcxproj]
      C:\usr\local\opticks\externals\include\OpenMesh/Core/Mesh/AttribKernelT.hh(164): warning C4127: conditional expression is constant [C:\usr\local\opticks\build\graphics\openmeshrap\OpenMeshRap.vcxproj]
      C:\usr\local\opticks\externals\include\OpenMesh/Core/Mesh/AttribKernelT.hh(167): warning C4127: conditional expression is constant [C:\usr\local\opticks\build\graphics\openmeshrap\OpenMeshRap.vcxproj]
      C:\usr\local\opticks\externals\include\OpenMesh/Core/Mesh/AttribKernelT.hh(170): warning C4127: conditional expression is constant [C:\usr\local\opticks\build\graphics\openmeshrap\OpenMeshRap.vcxproj]
      C:\usr\local\opticks\externals\include\OpenMesh/Core/Mesh/AttribKernelT.hh(173): warning C4127: conditional expression is constant [C:\usr\local\opticks\build\graphics\openmeshrap\OpenMeshRap.vcxproj]
      C:\usr\local\opticks\externals\include\OpenMesh/Core/Mesh/AttribKernelT.hh(176): warning C4127: conditional expression is constant [C:\usr\local\opticks\build\graphics\openmeshrap\OpenMeshRap.vcxproj]
      C:\usr\local\opticks\externals\include\OpenMesh/Core/Mesh/AttribKernelT.hh(179): warning C4127: conditional expression is constant [C:\usr\local\opticks\build\graphics\openmeshrap\OpenMeshRap.vcxproj]
      C:\usr\local\opticks\externals\include\OpenMesh/Core/Mesh/AttribKernelT.hh(182): warning C4127: conditional expression is constant [C:\usr\local\opticks\build\graphics\openmeshrap\OpenMeshRap.vcxproj]
      C:\usr\local\opticks\externals\include\OpenMesh/Core/Mesh/AttribKernelT.hh(185): warning C4127: conditional expression is constant [C:\usr\local\opticks\build\graphics\openmeshrap\OpenMeshRap.vcxproj]
      C:\usr\local\opticks\externals\include\OpenMesh/Core/Mesh/AttribKernelT.hh(188): warning C4127: conditional expression is constant [C:\usr\local\opticks\build\graphics\openmeshrap\OpenMeshRap.vcxproj]
      C:\usr\local\opticks\externals\include\OpenMesh/Core/Mesh/AttribKernelT.hh(191): warning C4127: conditional expression is constant [C:\usr\local\opticks\build\graphics\openmeshrap\OpenMeshRap.vcxproj]
      c:\usr\local\opticks\externals\include\openmesh\core\utils\property.hh(156): warning C4702: unreachable code [C:\usr\local\opticks\build\graphics\openmeshrap\OpenMeshRap.vcxproj]
      c:\usr\local\opticks\externals\include\openmesh\core\utils\property.hh(166): warning C4702: unreachable code [C:\usr\local\opticks\build\graphics\openmeshrap\OpenMeshRap.vcxproj]
      c:\usr\local\opticks\externals\include\openmesh\core\utils\property.hh(156): warning C4702: unreachable code [C:\usr\local\opticks\build\graphics\openmeshrap\OpenMeshRap.vcxproj]
      c:\usr\local\opticks\externals\include\openmesh\core\utils\property.hh(166): warning C4702: unreachable code [C:\usr\local\opticks\build\graphics\openmeshrap\OpenMeshRap.vcxproj]
      c:\usr\local\opticks\externals\include\openmesh\core\utils\property.hh(166): warning C4702: unreachable code [C:\usr\local\opticks\build\graphics\openmeshrap\OpenMeshRap.vcxproj]
      c:\usr\local\opticks\externals\include\openmesh\core\utils\property.hh(156): warning C4702: unreachable code [C:\usr\local\opticks\build\graphics\openmeshrap\OpenMeshRap.vcxproj]
      c:\usr\local\opticks\externals\include\openmesh\core\utils\property.hh(166): warning C4702: unreachable code [C:\usr\local\opticks\build\graphics\openmeshrap\OpenMeshRap.vcxproj]
      c:\usr\local\opticks\externals\include\openmesh\core\utils\property.hh(156): warning C4702: unreachable code [C:\usr\local\opticks\build\graphics\openmeshrap\OpenMeshRap.vcxproj]




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

openmesh-env(){  olocal- ; opticks- ; }
openmesh-vers(){ echo 4.1 ; }
#openmesh-vers(){ echo 6.1 ; }

openmesh-name(){ echo OpenMesh-$(openmesh-vers) ; }
openmesh-url(){  echo http://www.openmesh.org/media/Releases/$(openmesh-vers)/$(openmesh-name).tar.gz ; }



openmesh-edir(){ echo $(opticks-home)/graphics/openmesh ; }
openmesh-old-base(){ echo $(local-base)/env/graphics/openmesh ; }
openmesh-base(){ echo $(opticks-prefix)/externals/openmesh ; }

openmesh-prefix(){ echo $(opticks-prefix)/externals ; }
openmesh-idir(){ echo $(openmesh-prefix) ; }

openmesh-dir(){  echo $(openmesh-base)/$(openmesh-name) ; }
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

openmesh-edit(){ vi $(opticks-home)/cmake/Modules/FindOpenMesh.cmake ; }

openmesh-cmake(){
  local iwd=$PWD
  local bdir=$(openmesh-bdir)
  mkdir -p $bdir

  [ -f "$bdir/CMakeCache.txt" ] && echo $msg already configured : openmesh-configure to reconfigure && return 

  openmesh-bcd


  # -G "$(opticks-cmake-generator)" \

  cmake $(openmesh-dir) \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=$(openmesh-prefix) \
      -DBUILD_APPS=OFF 

  cd $iwd
}

openmesh-configure()
{
   openmesh-wipe
   openmesh-cmake $*
}


openmesh-make(){
  local iwd=$PWD
  openmesh-bcd

  cmake --build . --config Release --target ${1:-install}

  cd $iwd
}

openmesh--(){
  openmesh-get 
  openmesh-cmake
  openmesh-make install
}




