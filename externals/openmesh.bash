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

# === func-gen- : graphics/openmesh/openmesh fgp externals/openmesh.bash fgn openmesh fgh graphics/openmesh
openmesh-src(){      echo externals/openmesh.bash ; }
openmesh-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(openmesh-src)} ; }
openmesh-vi(){       vi $(openmesh-source) ; }


openmesh-usage(){ cat << EOU

OpenMesh
==========


* http://openmesh.org/Documentation/OpenMesh-Doc-Latest/a00030.html
* https://www.openmesh.org/media/Documentations/OpenMesh-4.1-Documentation/index.html



gcc 7.3 compilation failure on Ubuntu
------------------------------------------

OpenMesh compilation fail, add header to OpenMesh-6.3/src/OpenMesh/Tools/Utils/conio.cc::

   #include <sys/time.h>  /* timeval struct */

* https://bugs.archlinux.org/task/56522 

::

    [ 77%] Building CXX object src/OpenMesh/Tools/CMakeFiles/OpenMeshToolsStatic.dir/Utils/conio.cc.o
    /usr/local/opticks/externals/openmesh/OpenMesh-6.3/src/OpenMesh/Tools/Utils/conio.cc: In function ‘int OpenMesh::Utils::kbhit()’:
    /usr/local/opticks/externals/openmesh/OpenMesh-6.3/src/OpenMesh/Tools/Utils/conio.cc:122:20: error: aggregate ‘OpenMesh::Utils::kbhit()::timeval tv’ has incomplete type and cannot be defined
         struct timeval tv;
                        ^~
    /usr/local/opticks/externals/openmesh/OpenMesh-6.3/src/OpenMesh/Tools/Utils/conio.cc:127:5: error: ‘select’ was not declared in this scope
         select(1, NULL, NULL, NULL, &tv);
         ^~~~~~
    /usr/local/opticks/externals/openmesh/OpenMesh-6.3/src/OpenMesh/Tools/Utils/conio.cc:127:5: note: suggested alternative: ‘sleep’
         select(1, NULL, NULL, NULL, &tv);
         ^~~~~~
         sleep




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


Joining Meshes
--------------

* http://www.lsis.org/louruding/doc/LOU_TMCE2008.pdf
* ~/opticks_refs/direct_merging_of_meshes_EDF_LOU_TMCE2008.pdf


CSG BSP
-------

Web service for mesh booleans

* https://www.graphics.rwth-aachen.de/webbsp/


Connectivity Traverse 
-------------------------

Sqrt3 subdiv appears to need face order of the traverse the mesh in ever increasing circles
of connected faces. 


// https://mailman.rwth-aachen.de/pipermail/openmesh/2015-April/001083.html
// https://mailman.rwth-aachen.de/pipermail/openmesh/2015-April/001084.html

On Wed, Apr 1, 2015 at 9:42 AM, Junwei Huang <jwhuang1982 at gmail.com> wrote:

> Hello,
> I wonder if there are efficient algorithms/ideas that allow me to visit
> every vertex or face of a mesh by connectivity. Methods like *for vh in
> mesh.vertices()* can visit all vertices, but it does not honor the
> connectivity. For example, the next vertex in the list may not connected at
> all with the previous vertex. Same problem with *for fh in mesh.faces()*.
> Other circulators such as *for vh in mesh.vv(vh1)* honor the connectivity
> for one step only (one ring). How can I get a vertex handle and then access
> its one-ring neighborhood, 2-ring, 3-ring, till all vertices are visited?
> Many thanks.


You can try a breadth first search. A non-recursive traversal might looks
something like:

VertexHandle start = ...

enum Color { White, Gray, Black };
std::vector<Color> vcolor( mesh.n_vertices(), White );
std::deque<VHandle> Q;

vcolor[start.idx()] = Gray;
Q.push_back( start );

visit( start );

while( !Q.empty() )
{
  VHandle Vi = Q.front(); Q.pop_front();

  for( VOHIter vohit = mesh.voh_iter(Vi); vohit.is_valid(); ++vohit )
  {
    VHandle Vj = mesh.to_vertex_handle( *vohit );
    if( vcolor[Vj.idx()] == White )
    {
      vcolor[Vj.idx()] = Gray;
      visit( Vj );
      Q.push_back(Vj);
    }
    else { ... }
  } // for halfedge one-ring

  vcolor[Vi.idx()] = Black;
}

This is based on the implementation of boost::graph::breadth_first_visit in
the boost graph library and there are plenty of other visiting events that
can be triggered during this traversal, see their code for reference.

Ron Griswold
Software Engineer
Digital Domain
rgriswold at d2.com


Graph Traversal : Tricolor Algorithm : Breadth First Search
--------------------------------------------------------------

* http://www.cs.cornell.edu/courses/cs2112/2012sp/lectures/lec24/lec24-12sp.html

* http://hhoppe.com/edgetrav.pdf
* https://www.researchgate.net/publication/2274908_A_Breadth-First_Approach_To_Efficient_Mesh_Traversal

* http://oa.upm.es/21749/1/INVE_MEM_2011_114411.pdf
* ~/opticks_refs/mesh_traversal_and_sorting_INVE_MEM_2011_114411.pdf



Refs
-------

Euler Formula
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://math.stackexchange.com/questions/425968/eulers-formula-for-triangle-mesh

Nathan Reed: 

Since we're talking about a triangle mesh, there is a fixed relationship
between the number of edges and the number of faces. To derive this it's
helpful to think of the mesh as being made of half-edges. 

A half-edge is a pair of an edge and a face it borders. 

The total number of half-edges in the mesh is 2E, 
since each edge has two halves; and it's also 3F, since each face touches 
three half-edges and this counts all the half-edges exactly once. 
Therefore 2E=3F.

By solving for E and substituting into the formula V−E+F=0, we can easily derive your two facts:

* E = 3/2 F
* V - 3/2 F + F ~ 0 
* V - 1/2 F ~ 0  
* F ~ 2V


* F = 2/3E 
* V - E + 2/3E = V - 1/3E ~ 0 
* E ~ 3V




Geometric Modeling Based on Triangle Meshes (EUROGRAPHICS 2006)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

106 page tutorial covering mesh operations

* https://graphics.ethz.ch/Downloads/Publications/Tutorials/2006/Bot06b/eg06-tutorial.pdf
* ~/opticks_refs/Mesh_Geometric_Modelling_Botsch_eg06-tutorial.pdf  

Mesh Basics
-------------

* http://graphics.stanford.edu/courses/cs468-10-fall/LectureSlides/02_Basics.pdf

PyMesh
--------

* http://pymesh.readthedocs.io/en/latest/basic.html

OpenVolumeMesh
----------------

* http://www.openvolumemesh.org/Documentation/OpenVolumeMesh-Doc-Latest/concepts.html

Search
--------

* :google:`single face mesh subdivision`
* :google:`local mesh subdivision`


Subdivision Basics
~~~~~~~~~~~~~~~~~~~~

* http://www.cs.cmu.edu/afs/cs/academic/class/15462-s14/www/lec_slides/Subdivision.pdf

Incremental Mesh Subdivision
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://pdfs.semanticscholar.org/398d/5a2c15eedfc93969109277bf5002652231f3.pdf
* ~/opticks_refs/incremental_mesh_subdivision_5a2c15eedfc93969109277bf5002652231f3.pdf


A Remeshing Approach to Multiresolution Modeling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mario Botsch, Leif Kobbelt
Computer Graphics Group, RWTH Aachen University

* http://www.lsi.upc.edu/~pere/PapersWeb/SGI/Kobbelt2.pdf


Mesh Optimization
~~~~~~~~~~~~~~~~~~~

* http://hhoppe.com/meshopt.pdf


CGAL Subdiv
~~~~~~~~~~~~~~~

* http://doc.cgal.org/latest/Subdivision_method_3/


sqrt(3)-subdivision
~~~~~~~~~~~~~~~~~~~~~

* https://www.graphics.rwth-aachen.de/media/papers/sqrt31.pdf
* ~/opticks_refs/sqrt3_mesh_subdivision_kobbelt_sqrt31.pdf


* https://people.eecs.berkeley.edu/~sequin/CS284/PAPERS/root3subdiv.pdf
* ~/opticks_refs/interpolatory_sqrt3_subdiv_root3subdiv.pdf



Reference from OpenMesh source

* /usr/local/opticks/externals/openmesh/OpenMesh-4.1/src/OpenMesh/Tools/Subdivider/Adaptive/Composite/CompositeT.hh
* :google:`API Design for Adaptive Subdivision schemes`
* ~/opticks_refs/OpenSG2003_sovakar_APIDesign_slides.pdf


* http://www.multires.caltech.edu/pubs/sqrt3.pdf
* ~/opticks_refs/composite_primal_dual_sqrt3_subdiv.pdf


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



Adaptive Composite Rule for Tvv3 : topological split face into 3 by adding vertex at centroid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/usr/local/opticks/externals/openmesh/OpenMesh-4.1/src/OpenMesh/Tools/Subdivider/Adaptive/Composite/RulesT.cc

::

     102 template<class M>
     103 void
     104 Tvv3<M>::raise(typename M::FaceHandle& _fh, state_t _target_state)
     105 {
     ...
     139     // interior face
     140     if (!Base::mesh_.is_boundary(_fh) || MOBJ(_fh).final()) {
     141 
     142       // insert new vertex
     143       vh = Base::mesh_.new_vertex();
     144 
     145       Base::mesh_.split(_fh, vh);
     146 
     147       typename M::Scalar valence(0.0);
     148 
     149       // calculate display position for new vertex
     150       for (vv_it = Base::mesh_.vv_iter(vh); vv_it.is_valid(); ++vv_it)
     151       {
     152         position += Base::mesh_.point(*vv_it);
     153         valence += 1.0;
     154       }
     155 
     156       position /= valence;
     157 
     158       // set attributes for new vertex
     159       Base::mesh_.set_point(vh, position);

     ///    above is same as add_vertex at centroid



     088 #define MOBJ Base::mesh_.data
      89 #define FH face_handle
      90 #define VH vertex_handle
      91 #define EH edge_handle
      92 #define HEH halfedge_handle
      93 #define NHEH next_halfedge_handle
      94 #define PHEH prev_halfedge_handle
      95 #define OHEH opposite_halfedge_handle
      96 #define TVH  to_vertex_handle
      97 #define FVH  from_vertex_handle
      98 


     160       MOBJ(vh).set_position(_target_state, zero_point);
     161       MOBJ(vh).set_state(_target_state);
     162       MOBJ(vh).set_not_final();
     163 


     ///     outgoing halfedges from the new splitting vertex
     ///     nheh goes around ccw, so the oheh face gets adjacent faces
     ///
     ///   
           
                      +
                     / \
                    / . \  ^ 
                0n /  ^  \  \
               /  /   0   \  1n
              v  /    +    \
                /  v.   v   \
               / .2       1 .\
              +---------------+
                    -> 2n

     164       typename M::VertexOHalfedgeIter      voh_it;
     165       // check for edge flipping
     166       for (voh_it = Base::mesh_.voh_iter(vh); voh_it.is_valid(); ++voh_it) {
     167 
     168         if (Base::mesh_.FH(*voh_it).is_valid()) {
     169 
     170           MOBJ(Base::mesh_.FH(*voh_it)).set_state(_target_state);
     171           MOBJ(Base::mesh_.FH(*voh_it)).set_not_final();
     172           MOBJ(Base::mesh_.FH(*voh_it)).set_position(_target_state - 1, face_position);
     173 
     174 
     175           for (state_t j = 0; j < _target_state; ++j) {
     176             MOBJ(Base::mesh_.FH(*voh_it)).set_position(j, MOBJ(_fh).position(j));
     177           }
     178 
     179           if (Base::mesh_.FH(Base::mesh_.OHEH(Base::mesh_.NHEH(*voh_it))).is_valid()) {
     180 
     181             if (MOBJ(Base::mesh_.FH(Base::mesh_.OHEH(Base::mesh_.NHEH(*voh_it)))).state() == _target_state) {
     182 
     183               if (Base::mesh_.is_flip_ok(Base::mesh_.EH(Base::mesh_.NHEH(*voh_it)))) {
     184 
     185                 edge_vector.push_back(Base::mesh_.EH(Base::mesh_.NHEH(*voh_it)));
     186               }
     187             }
     188           }
     189         }
     190       }
     191     }





Flipping an edge
~~~~~~~~~~~~~~~~~~~~

::

    341 void TriConnectivity::flip(EdgeHandle _eh)
    342 {
    343   // CAUTION : Flipping a halfedge may result in
    344   // a non-manifold mesh, hence check for yourself
    345   // whether this operation is allowed or not!
    346   assert(is_flip_ok(_eh));//let's make it sure it is actually checked
    347   assert(!is_boundary(_eh));
    348 
    349   HalfedgeHandle a0 = halfedge_handle(_eh, 0);
    350   HalfedgeHandle b0 = halfedge_handle(_eh, 1);
    351 
    352   HalfedgeHandle a1 = next_halfedge_handle(a0);   // all 3 halfedges (ccw)  a0,a1,a2  and b0,b1,b2
    353   HalfedgeHandle a2 = next_halfedge_handle(a1);
    354 
    355   HalfedgeHandle b1 = next_halfedge_handle(b0);
    356   HalfedgeHandle b2 = next_halfedge_handle(b1);
    357 
    358   VertexHandle   va0 = to_vertex_handle(a0);
    359   VertexHandle   va1 = to_vertex_handle(a1);
    360 
    361   VertexHandle   vb0 = to_vertex_handle(b0);
    362   VertexHandle   vb1 = to_vertex_handle(b1);
    363 
    364   FaceHandle     fa  = face_handle(a0);
    365   FaceHandle     fb  = face_handle(b0);


::

                   /\                                          /\                     
                  /  \                                        /  \
                 / vb1\                                      / vb1\
                / /  \ \                                    / /||\ \
               / /    \ \                                  / / || \ \ 
              / /      \ \                                / /  ||  \ \
             / /        \ \                              / /   ||   \ \
            / /        ^ \ \                            / /    ||  ^ \ \
           / /         b1 \ \                          / /     ||  b1 \ \
          / / b2           \ \                        / / b2   ||  |   \ \
         / / v              \ \                      / / v     ||  |    \ \
        / /     fb           \ \                    / /        ||  fa    \ \ 
       / /        \           \ \                  / /        ^|| /       \ \
      / /----------b0>--------vb0\                / va0      b0||a0        \ \
      \ va0-------<a0----------  /                \ \       /  ||v        vb0/
       \ \          \         / /                  \ \     fb  ||         / /
        \ \          fa      / /                    \ \    |   ||        / /
         \ \                / /                      \ \   |   ||       / /
          \ \            ^ / /                        \ \  |   ||    ^ / /
           \ \ a1       a2/ /                          \ \ a1  ||   a2/ /
            \ \ v        / /                            \ \ v  ||    / /
             \ \        / /                              \ \   ||   / /
              \ \      / /                                \ \  ||  / /
               \ \    / /                                  \ \ || / /
                \ \  / /                                    \ \||/ /
                 \ va1/                                      \ va1/
                  \  /                                        \  /
                   \/                                          \/                  


    341 void TriConnectivity::flip(EdgeHandle _eh)
    ...
    366 
    367   set_vertex_handle(a0, va1);
    368   set_vertex_handle(b0, vb1);
    ///   
    ///   changing "to" vertex of both halfedges
    ///
    369 
    370   set_next_halfedge_handle(a0, a2);   // a0-a2-b1-a0 ccw next cycle
    371   set_next_halfedge_handle(a2, b1);
    372   set_next_halfedge_handle(b1, a0);
    373 
    374   set_next_halfedge_handle(b0, b2);   // b0-b2-a1-b0 ccw next cycle
    375   set_next_halfedge_handle(b2, a1);
    376   set_next_halfedge_handle(a1, b0);
    377 
    378   set_face_handle(a1, fb);    //  a1 changes its face
    379   set_face_handle(b1, fa);    //  

    380 
    381   set_halfedge_handle(fa, a0);
    382   set_halfedge_handle(fb, b0);

    ///  remember in halfedge data structure a "face" is just identified by one of  
    ///  its halfedges ... (i think any halfedge will do)

    383 
    384   if (halfedge_handle(va0) == b0)
    385     set_halfedge_handle(va0, a1);
    386   if (halfedge_handle(vb0) == a0)
    387     set_halfedge_handle(vb0, b1);
    388 }







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


/usr/local/opticks/externals/openmesh/OpenMesh-4.1/src/OpenMesh/Core/Mesh/ArrayKernel.hh::

    339   // --- halfedge connectivity ---
    340   VertexHandle to_vertex_handle(HalfedgeHandle _heh) const
    341   { return halfedge(_heh).vertex_handle_; }
    342 
    343   VertexHandle from_vertex_handle(HalfedgeHandle _heh) const
    344   { return to_vertex_handle(opposite_halfedge_handle(_heh)); }
    345 
    346   void set_vertex_handle(HalfedgeHandle _heh, VertexHandle _vh)
    347   {
    348 //     assert(is_valid_handle(_vh));
    349     halfedge(_heh).vertex_handle_ = _vh;
    350   }


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
     ///  * Each vertex references one outgoing halfedge, i.e. a halfedge that starts at this vertex (1).
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

     ///    halfedge : face, vertex and next halfedge 
     ///
     ///    * vertex is the "to_vertex"  (see ArrayKernel.hh)
     ///   
     ///

     /// Each halfedge provides a handle to
     ///
     ///  1. the vertex it points to (3),
     ///  2. the face it belongs to (4)
     ///  3. the next halfedge inside the face (ordered counter-clockwise) (5),
     ///     the opposite halfedge (6),
     ///
     ///     (optionally: the previous halfedge in the face (7)).
     ///
     ///   ArrayKernel.hh opposite is derived from the index, not stored    
     ///
     ///  427   HalfedgeHandle opposite_halfedge_handle(HalfedgeHandle _heh) const
     ///  428   { return HalfedgeHandle((_heh.idx() & 1) ? _heh.idx()-1 : _heh.idx()+1); }
     ///  429 


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

    /// edge : just two halfedges

    114 #endif
    115 
    116   //-------------------------------------------------------- internal face type
    117 #ifndef DOXY_IGNORE_THIS
    118   class Face
    119   {
    120     friend class ArrayKernel;
    121     HalfedgeHandle  halfedge_handle_;
    122   };

    ///  face : just holds onto one halfedge (which ? is it symmetric)
    ///
    /// Each face references one of the halfedges bounding it (2).


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
#openmesh-vers(){ echo 4.1 ; }
openmesh-vers(){ echo 6.3 ; }

openmesh-info(){ cat << EOI

    name : $(openmesh-name)
    dist : $(openmesh-dist)

    dir  : $(openmesh-dir)
    bdir : $(openmesh-bdir)
    pfx  : $(openmesh-prefix)

EOI
}


openmesh-name(){ echo OpenMesh-$(openmesh-vers) ; }
openmesh-url(){  echo http://www.openmesh.org/media/Releases/$(openmesh-vers)/$(openmesh-name).tar.gz ; }

openmesh-dist(){ echo $(dirname $(openmesh-dir))/$(basename $(openmesh-url)) ; }

openmesh-edir(){ echo $(opticks-home)/graphics/openmesh ; }
openmesh-old-base(){ echo $(local-base)/env/graphics/openmesh ; }
openmesh-base(){ echo $(opticks-prefix)/externals/openmesh ; }

openmesh-prefix(){ echo $(opticks-prefix)/externals ; }
openmesh-idir(){ echo $(openmesh-prefix) ; }
openmesh-pc-path(){ echo $(opticks-prefix)/externals/lib/pkgconfig/OpenMesh.pc ; }

openmesh-dir(){  echo $(openmesh-base)/$(openmesh-name) ; }
openmesh-bdir(){ echo $(openmesh-base)/$(openmesh-name).build ; }

openmesh-ecd(){  cd $(openmesh-edir); }
openmesh-cd(){   cd $(openmesh-dir)/$1 ; }
openmesh-bcd(){  cd $(openmesh-bdir); }
openmesh-icd(){  cd $(openmesh-idir); }

openmesh-mcd(){  openmesh-cd src/OpenMesh/Core/Mesh ; }

openmesh-get(){
   local dir=$(dirname $(openmesh-dir)) &&  mkdir -p $dir && cd $dir
   local url="$(openmesh-url)"
   local tgz=$(basename $url)
   local nam=${tgz/.tar.gz}
   [ ! -f "$tgz" ] && curl -L -O $url
   [ ! -d "$nam" ] && tar zxf $tgz 
   return 0 
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
  local rc
  openmesh-bcd

  cmake --build . --config Release --target ${1:-install}
  rc=$? 

  cd $iwd
  return $rc
}

openmesh--(){

  local msg="=== $FUNCNAME :"

  openmesh-get 
  [ $? -ne 0 ] && echo $msg get FAIL && return 1

  openmesh-cmake
  [ $? -ne 0 ] && echo $msg cmake FAIL && return 2

  openmesh-make install
  [ $? -ne 0 ] && echo $msg make FAIL && return 3

  openmesh-pc
  [ $? -ne 0 ] && echo $msg pc FAIL && return 4

  return 0 
}


openmesh-libs(){
  ls -l $(openmesh-prefix)/lib/libOpenMesh*
}

openmesh-pc-(){ 
   oc-
   oc-variables-
   cat << EOP

# $FUNCNAME $(date)

Name: OpenMesh
Description: Mesh Traversal and Manipulations
Version: $(openmesh-vers)
Libs: -L\${libdir} -lOpenMeshTools -lOpenMeshCore -lstdc++ -lm
Cflags: -I\${includedir}

EOP
}

openmesh-pc(){ 
   local msg="=== $FUNCNAME :"
   local path=$(openmesh-pc-path)
   local dir=$(dirname $path)

   [ ! -d "$dir" ] && echo $msg creating dir $dir && mkdir -p $dir 
   echo $msg $path 
   openmesh-pc- > $path 
}

openmesh-setup(){ cat << EOS
# $FUNCNAME
EOS
}
