# === func-gen- : graphics/openmesh/openmesh fgp externals/openmesh.bash fgn openmesh fgh graphics/openmesh
openmesh-src(){      echo externals/openmesh.bash ; }
openmesh-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(openmesh-src)} ; }
openmesh-vi(){       vi $(openmesh-source) ; }
openmesh-usage(){ cat << EOU

OpenMesh
==========


* http://openmesh.org/Documentation/OpenMesh-Doc-Latest/a00030.html
* https://www.openmesh.org/media/Documentations/OpenMesh-4.1-Documentation/index.html


Mailing List 
---------------

* https://mailman.rwth-aachen.de/mailman/listinfo/openmesh
  
  Argh unsearchable mailing list 

Repo
-------

* https://graphics.rwth-aachen.de:9000/OpenMesh/OpenMesh

  Web interface to git repo, issue tracker etc..


* https://graphics.rwth-aachen.de:9000/OpenMesh/OpenMesh/commit/338086152b8d5cfce75580c76e445c1de9d80381
* https://graphics.rwth-aachen.de:9000/OpenMesh/OpenMesh/blob/master/src/Unittests/unittests_delete_face.cc

  gtest unittests 


PyMesh
--------

* http://pymesh.readthedocs.io/en/latest/basic.html

OpenVolumeMesh
----------------

* http://www.openvolumemesh.org/Documentation/OpenVolumeMesh-Doc-Latest/concepts.html


Mesh Basics
-------------

* https://www.cs.mtu.edu/~shene/COURSES/cs3621/SLIDES/Mesh.pdf


Does Edge splitting make a mesh non-manifold ?
-----------------------------------------------

* https://en.wikipedia.org/wiki/Euler_operator
* https://en.wikipedia.org/wiki/Manifold



Building
------------

cmake::

    -- Checking the Boost Python configuration
    Checking the Boost Python configuration failed!
    Reason: An error occurred while running a small Boost Python test project.
    Make sure that your Python and Boost Python libraries match.
    Skipping Python Bindings.


Refs
-------


Geometric Modeling Based on Triangle Meshes (EUROGRAPHICS 2006)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

106 page tutorial covering mesh operations

* https://graphics.ethz.ch/Downloads/Publications/Tutorials/2006/Bot06b/eg06-tutorial.pdf
* ~/opticks_refs/Mesh_Geometric_Modelling_Botsch_eg06-tutorial.pdf  


Mesh Basics
-------------

* http://graphics.stanford.edu/courses/cs468-10-fall/LectureSlides/02_Basics.pdf

Mesh Navigation : following a boundary 
-------------------------------------------

* http://www.openmesh.org/Documentation/OpenMesh-2.0-Documentation/mesh_navigation.html

Mesh Editing
-------------

* garbage collection invalidates vertex handles


Splitting Edges
~~~~~~~~~~~~~~~~

* https://mailman.rwth-aachen.de/pipermail/openmesh/2012-November/000802.html

as long as you don't call garbage_collection on your mesh, all handles
will stay valid and point to the correct vertices (even after the split
operation.


No, if you tag the vertices before your operation, the handle will point
to the correct vertex and this information is stored on the vertex. Of
course, if you do a garbage collection, your old handle will be
invalidated (You can use the advanced garbage collection to let OM
update the handles for you). But the information is still stored for the
correct vertex.

Splitting Faces
~~~~~~~~~~~~~~~~~~

* https://stackoverflow.com/questions/41008298/openmesh-face-split

Deleting faces 
~~~~~~~~~~~~~~~~~~

* https://mailman.rwth-aachen.de/pipermail/openmesh/2012-August/000764.html

That is how garbage collection is working. The deleted elements are
removed from the storage and the vectors are compacted, which
invalidates the handles. Therefore, garbage collection should be used
after your algorithm is completed.

The entities which are deleted, are marked as
deleted in their status flag. If you are using the mesh for rendering,
you can simply use the skipping iterators ( mesh->vertices_sbegin() )
which will skip deleted elements in the iteration.

Another possibility is the newly introduced garbage_collection with
handle updates. You have to give the garbage collection arrays of
pointers to handles which should get updated. It's in the svn repo and
already included in the daily builds.


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

* https://stackoverflow.com/questions/21502416/splitting-mesh-into-connected-components-in-openmesh
* http://www.openflipper.org/media/Documentation/OpenFlipper-1.0.2/MeshInfoT_8cc_source.html

Usage
-------

* http://www.hao-li.com/cs599-ss2015/exercises/exercise1.pdf
* ~/opticks_refs/OpenMesh_Introduction_HaoLi.pdf


Good Starting points
----------------------

TriMesh

* file:///usr/local/env/graphics/openmesh/OpenMesh-4.1/Documentation/a00276.html

Halfedge structure
-------------------

* https://www.openmesh.org/Daily-Builds/Doc/a03947.html

The data structure used in this project is the so called halfedge data
structure . While face-based structures store their connectivity in faces
referencing their vertices and neighbors, edge-based structures put the
connectivity information into the edges. Each edge references its two vertices,
the faces it belongs to and the two next edges in these faces. If one now
splits the edges (i.e. an edge connecting vertex A and vertex B becomes two
directed halfeges from A to B and vice versa) one gets a halfedge-based data
structure. The following figure illustrates the way connectivity is stored in
this structure:


Introduction to Halfedge data structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://www.flipcode.com/archives/The_Half-Edge_Data_Structure.shtml

Another Halfedge intro
~~~~~~~~~~~~~~~~~~~~~~~~~

* http://kaba.hilvi.org/homepage/blog/halfedge/halfedge.htm

What makes the half-edge structure interesting is its ability to efficiently answer such adjacency queries as:

* What edges are connected to this vertex?
* What polygons are connected to this vertex?
* What vertices are connected to this polygon?

and the fact that it achieves this by using only a constant amount of data for each element.

There are two important restrictions on the relationships that can be described:

* No non-orientable surfaces (e.g. Moebius strip).
* No non-manifold surfaces (e.g. two cubes sharing a corner vertex).

The second restriction implies an analogous statement for edges and polygons.
The quad-edge structure is a minimal upgrade to the half-edge structure that
enables representation of non-orientable surfaces. The partial entity structure
extends the idea of halving edges to halving all elements and handles also
non-manifold surfaces.


Another Intro
~~~~~~~~~~~~~~~

* https://fgiesen.wordpress.com/2012/02/21/half-edge-based-mesh-representations-theory/

The reason for focusing on edges is quite simple: Face->Edge, Face->Vertex,
Vertex->Edge and Vertex->Face are all one-to-many-type relationships in a general
polygon mesh, but each edge links exactly 2 vertices and lies (in a closed,
2-manifold mesh) between exactly 2 faces – a fixed number of pointers (or
indices), much nicer. 

Half-edges make this even more uniform by using directed edges 
(in the direction of winding order). Each half-edge is incident to one
face and one vertex, and a pair of half-edges makes up a full edge, hence the
name.



Efficient Following of boundaries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Its a fundamental feature of the halfedge data structure.


Alternates
-----------

* cgal
* vcg/meshlab
* http://gfx.cs.princeton.edu/proj/trimesh2/


Code Review
--------------


/usr/local/opticks/externals/openmesh/OpenMesh-4.1/src/OpenMesh/Core/Mesh/ArrayItems.hh::


    ///  purely private internal classes, only accessible via ArrayKernel 

    .63 namespace OpenMesh {
     64 
     65 
     66 //== CLASS DEFINITION =========================================================
     67 
     68 
     69 /// Definition of mesh items for use in the ArrayKernel
     70 struct ArrayItems
     71 {
     72 
     73   //------------------------------------------------------ internal vertex type
     74 
     75   /// The vertex item
     76   class Vertex
     77   {
     78     friend class ArrayKernel;
     79     HalfedgeHandle  halfedge_handle_;
     80   };
     81 
     82 
     83   //---------------------------------------------------- internal halfedge type
     84 
     85 #ifndef DOXY_IGNORE_THIS
     86   class Halfedge_without_prev
     87   {
     88     friend class ArrayKernel;
     89     FaceHandle      face_handle_;
     90     VertexHandle    vertex_handle_;
     91     HalfedgeHandle  next_halfedge_handle_;
     92   };
     93 #endif
     94 
     95 #ifndef DOXY_IGNORE_THIS
     96   class Halfedge_with_prev : public Halfedge_without_prev
     97   {
     98     friend class ArrayKernel;
     99     HalfedgeHandle  prev_halfedge_handle_;
    100   };
    101 #endif
    102 
    103   //TODO: should be selected with config.h define
    104   typedef Halfedge_with_prev                Halfedge;
    105   typedef GenProg::Bool2Type<true>          HasPrevHalfedge;
    106 
    107   //-------------------------------------------------------- internal edge type
    108 #ifndef DOXY_IGNORE_THIS
    109   class Edge
    110   {
    111     friend class ArrayKernel;
    112     Halfedge  halfedges_[2];
    113   };
    114 #endif
    115 
    116   //-------------------------------------------------------- internal face type
    117 #ifndef DOXY_IGNORE_THIS
    118   class Face
    119   {
    120     friend class ArrayKernel;
    121     HalfedgeHandle  halfedge_handle_;
    122   };
    123 };
    124 #endif






/usr/local/opticks/externals/openmesh/OpenMesh-4.1/src/OpenMesh/Core/Mesh/PolyMeshT.hh

::

    093 template <class Kernel>
     94 class PolyMeshT : public Kernel
     95 {

    188   PolyMeshT() {}
    ...
    195    // --- creation ---
    196   inline VertexHandle new_vertex()
    197   { return Kernel::new_vertex(); }
    198 
    199   inline VertexHandle new_vertex(const Point& _p)
    200   {
    201     VertexHandle vh(Kernel::new_vertex());
    202     this->set_point(vh, _p);
    203     return vh;
    204   }
    205 
    206   inline VertexHandle add_vertex(const Point& _p)
    207   { return new_vertex(_p); }


/usr/local/opticks/externals/openmesh/OpenMesh-4.1/src/OpenMesh/Core/Mesh/Handles.hh

Handles are just ints::


    .66 /// Base class for all handle types
     67 class BaseHandle
     68 {
     69 public:
     70 
     71   explicit BaseHandle(int _idx=-1) : idx_(_idx) {}
     72 
     73   /// Get the underlying index of this handle
     74   int idx() const { return idx_; }
     75 
     76   /// The handle is valid iff the index is not equal to -1.
     77   bool is_valid() const { return idx_ != -1; }
     78 
     79   /// reset handle to be invalid
     80   void reset() { idx_=-1; }
     81   /// reset handle to be invalid
     82   void invalidate() { idx_ = -1; }
     83 
     84   bool operator==(const BaseHandle& _rhs) const {
     85     return (this->idx_ == _rhs.idx_);
     86   }
     87 
    ...
    102 private:
    103 
    104   int idx_;
    105 };
    ...
    111 inline std::ostream& operator<<(std::ostream& _os, const BaseHandle& _hnd)
    112 {
    113   return (_os << _hnd.idx());
    114 }
    115 
    121 struct VertexHandle : public BaseHandle
    122 {
    123   explicit VertexHandle(int _idx=-1) : BaseHandle(_idx) {}
    124 };
    128 struct HalfedgeHandle : public BaseHandle
    129 {
    130   explicit HalfedgeHandle(int _idx=-1) : BaseHandle(_idx) {}
    131 };
    135 struct EdgeHandle : public BaseHandle
    136 {
    137   explicit EdgeHandle(int _idx=-1) : BaseHandle(_idx) {}
    138 };
    142 struct FaceHandle : public BaseHandle
    143 {
    144   explicit FaceHandle(int _idx=-1) : BaseHandle(_idx) {}
    145 };

    * https://stackoverflow.com/questions/121162/what-does-the-explicit-keyword-mean-in-c

     "explicit" ctors prevents compiler from using single parameter implicit conversions ... 
     otherwise methods that accept handles would be able to implicitly magic them into 
     existance : which would lead to ambiguity. 
      

/usr/local/opticks/externals/openmesh/OpenMesh-4.1/src/OpenMesh/Core/Mesh/ArrayKernel.hh::


    092 class OPENMESHDLLEXPORT ArrayKernel : public BaseKernel, public ArrayItems
     93 {
     94 public:
     95 
     96   // handles
     97   typedef OpenMesh::VertexHandle            VertexHandle;
     98   typedef OpenMesh::HalfedgeHandle          HalfedgeHandle;
     99   typedef OpenMesh::EdgeHandle              EdgeHandle;
    100   typedef OpenMesh::FaceHandle              FaceHandle;
    101   typedef Attributes::StatusInfo            StatusInfo;
    102   typedef VPropHandleT<StatusInfo>          VertexStatusPropertyHandle;
    103   typedef HPropHandleT<StatusInfo>          HalfedgeStatusPropertyHandle;
    104   typedef EPropHandleT<StatusInfo>          EdgeStatusPropertyHandle;
    105   typedef FPropHandleT<StatusInfo>          FaceStatusPropertyHandle;
    106 
    107 public:
    108 
    109   // --- constructor/destructor ---
    110   ArrayKernel();
    111   virtual ~ArrayKernel();
    112 
    113   /** ArrayKernel uses the default copy constructor and assignment operator, which means
    114       that the connectivity and all properties are copied, including reference
    115       counters, allocated bit status masks, etc.. In contrast assign_connectivity
    116       copies only the connectivity, i.e. vertices, edges, faces and their status fields.
    117       NOTE: The geometry (the points property) is NOT copied. Poly/TriConnectivity
    118       override(and hide) that function to provide connectivity consistence.*/
    119   void assign_connectivity(const ArrayKernel& _other);
    120 
    121   // --- handle -> item ---
    122   VertexHandle handle(const Vertex& _v) const;
    123 
    124   HalfedgeHandle handle(const Halfedge& _he) const;
    125 
    126   EdgeHandle handle(const Edge& _e) const;
    127 
    128   FaceHandle handle(const Face& _f) const;
    ...
    144   // --- item -> handle ---
    145   const Vertex& vertex(VertexHandle _vh) const
    146   {
    147     assert(is_valid_handle(_vh));
    148     return vertices_[_vh.idx()];
    149   }
    150 
    151   Vertex& vertex(VertexHandle _vh)
    152   {
    153     assert(is_valid_handle(_vh));
    154     return vertices_[_vh.idx()];
    155   }
    156 
    157   const Halfedge& halfedge(HalfedgeHandle _heh) const
    158   {
    159     assert(is_valid_handle(_heh));
    160     return edges_[_heh.idx() >> 1].halfedges_[_heh.idx() & 1];
    161   }
    162 
    163   Halfedge& halfedge(HalfedgeHandle _heh)
    164   {
    165     assert(is_valid_handle(_heh));
    166     return edges_[_heh.idx() >> 1].halfedges_[_heh.idx() & 1];
    167   }
    168 
    169   const Edge& edge(EdgeHandle _eh) const
    170   {
    171     assert(is_valid_handle(_eh));
    172     return edges_[_eh.idx()];
    173   }
    174 
    175   Edge& edge(EdgeHandle _eh)
    176   {
    177     assert(is_valid_handle(_eh));
    178     return edges_[_eh.idx()];
    179   }
    180 
    181   const Face& face(FaceHandle _fh) const
    182   {
    183     assert(is_valid_handle(_fh));
    184     return faces_[_fh.idx()];
    185   }
    186 
    187   Face& face(FaceHandle _fh)
    188   {
    189     assert(is_valid_handle(_fh));
    190     return faces_[_fh.idx()];
    191   }
    192 
    193   // --- get i'th items ---
    194 
    195   VertexHandle vertex_handle(unsigned int _i) const
    196   { return (_i < n_vertices()) ? handle( vertices_[_i] ) : VertexHandle(); }
    197 
    198   HalfedgeHandle halfedge_handle(unsigned int _i) const
    199   {
    200     return (_i < n_halfedges()) ?
    201       halfedge_handle(edge_handle(_i/2), _i%2) : HalfedgeHandle();
    202   }
    203 
    204   EdgeHandle edge_handle(unsigned int _i) const
    205   { return (_i < n_edges()) ? handle(edges_[_i]) : EdgeHandle(); }
    206 
    207   FaceHandle face_handle(unsigned int _i) const
    208   { return (_i < n_faces()) ? handle(faces_[_i]) : FaceHandle(); }
    209 



/usr/local/opticks/externals/openmesh/OpenMesh-4.1/src/OpenMesh/Core/Mesh/ArrayKernel.cc::

    105 VertexHandle ArrayKernel::handle(const Vertex& _v) const^M
    106 {^M
    107    return VertexHandle( int( &_v - &vertices_.front()));^M
    108 }^M
    109 ^M
    110 HalfedgeHandle ArrayKernel::handle(const Halfedge& _he) const^M
    111 {^M
    112   // Calculate edge belonging to given halfedge^M
    113   // There are two halfedges stored per edge^M
    114   // Get memory position inside edge vector and devide by size of an edge^M
    115   // to get the corresponding edge for the requested halfedge^M
    116   size_t eh = ( (char*)&_he - (char*)&edges_.front() ) /  sizeof(Edge)  ;^M
    117   assert((&_he == &edges_[eh].halfedges_[0]) ||^M
    118          (&_he == &edges_[eh].halfedges_[1]));^M
    119   return ((&_he == &edges_[eh].halfedges_[0]) ?^M
    120                     HalfedgeHandle( int(eh)<<1) : HalfedgeHandle((int(eh)<<1)+1));^M
    121 }^M
    122 ^M
    123 EdgeHandle ArrayKernel::handle(const Edge& _e) const^M
    124 {^M
    125   return EdgeHandle( int(&_e - &edges_.front() ) );^M
    126 }^M
    127 ^M
    128 FaceHandle ArrayKernel::handle(const Face& _f) const^M
    129 {^M
    130   return FaceHandle( int(&_f - &faces_.front()) );^M
    131 }^M

    /// handle ints are pointer arithmetic offsets from the container front

    ...
    212   inline VertexHandle new_vertex()
    213   {
    214     vertices_.push_back(Vertex());
    215     vprops_resize(n_vertices());//TODO:should it be push_back()?
    216 
    217     return handle(vertices_.back());
    218   }




EOU
}

openmesh-env(){  olocal- ; opticks- ; }
openmesh-vers(){ echo 4.1 ; }
#openmesh-vers(){ echo 6.1 ; }

openmesh-name(){ echo OpenMesh-$(openmesh-vers) ; }
openmesh-url(){  echo http://www.openmesh.org/media/Releases/$(openmesh-vers)/$(openmesh-name).tar.gz ; }

openmesh-dist(){ echo $(dirname $(openmesh-dir))/$(basename $(openmesh-url)) ; }

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

openmesh-doc(){ openmesh-html ; }
openmesh-html(){ open $(openmesh-dir)/Documentation/index.html ; }

openmesh-find(){ openmesh-cd ; find src -type f -exec grep -H ${1:-DefaultTraits} {} \; ; }


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




