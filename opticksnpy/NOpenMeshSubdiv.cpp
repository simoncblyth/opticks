#include <sstream>

#include "PLOG.hh"

#include "NOpenMeshProp.hpp"
#include "NOpenMeshDesc.hpp"
#include "NOpenMeshFind.hpp"
#include "NOpenMeshBuild.hpp"
#include "NOpenMeshSubdiv.hpp"


using namespace OpenMesh::Subdivider;


template <typename T>
NOpenMeshSubdiv<T>::NOpenMeshSubdiv( 
    T& mesh, 
    const NOpenMeshProp<T>& prop, 
    const NOpenMeshDesc<T>& desc, 
    const NOpenMeshFind<T>& find, 
          NOpenMeshBuild<T>& build 
    )
    : 
    mesh(mesh),
    prop(prop),
    desc(desc),
    find(find),
    build(build),
    subdivider(new subdivider_t(mesh)) 
{
    init();
} 


template <typename T>
void NOpenMeshSubdiv<T>::init()
{
    //init_subdivider();
}
 

template <typename T>
void NOpenMeshSubdiv<T>::init_subdivider()
{
   typedef typename Adaptive::RuleInterfaceT<T>  Rule ; 

   typedef typename Adaptive::Tvv3<T> Tvv3_r ; 
   typedef typename Adaptive::VF<T>   VF_r ; 
   typedef typename Adaptive::FF<T>   FF_r ; 
   typedef typename Adaptive::FVc<T>  FVc_r ; 

   // http://www.multires.caltech.edu/pubs/sqrt3.pdf

   subdivider->template add<Tvv3_r>();
   subdivider->template add<VF_r>();
   subdivider->template add<FF_r>();
   subdivider->template add<FVc_r>();

   LOG(info) << "NOpenMeshSubdiv<T>::init()"
             << " " << brief()
             ;

   assert(subdivider->initialize()); 
}

template <typename T>
std::string NOpenMeshSubdiv<T>::brief()
{
    std::stringstream ss ; 

    ss << subdivider->rules_as_string() ;

    return ss.str();
}
 
template <typename T>
void NOpenMeshSubdiv<T>::refine(typename T::FaceHandle fh)
{
    subdivider->refine(fh); 
}









template <typename T>
void NOpenMeshSubdiv<T>::manual_subdivide_face_creating_soup(typename T::FaceHandle fh, const nnode* other, int verbosity, float epsilon )
{
/*
Uniform subdivision of single triangle face

* add three new vertices at midpoints of original triangle edges 
* delete face, leaving triangular hole, retain vertices
* add 4 new faces

Hmm cannot do this to single face without regard to the rest of the 
connected faces... so although this appears to work it 
creates triangle soup, hindering further operations.  

Can only do this when apply to entire mesh ?

Suspect deleting and adding faces is not the way OpenMesh expects 
subdiv to be done, the below used mesh.split(eh, midpoint)

   /usr/local/opticks/externals/openmesh/OpenMesh-4.1/src/OpenMesh/Tools/Subdivider/Uniform/LongestEdgeT.hh



                     o[2]
                     +
                    / \
                   /   \
                  /     \
                 /  f[2] \
                /         \
         m[2]  /           \  m[1]
              +-------------+ 
             / \           / \
            /   \  f[3]   /   \
           /     \       /     \
          /  f[0] \     / f[1]  \
         /         \   /         \
        /           \ /           \
       +-------------+-------------+
      o[0]          m[0]           o[1]


*/


    typedef typename T::FaceHandle          FH ; 
    typedef typename T::VertexHandle        VH ; 
    typedef typename T::HalfedgeHandle      HEH ; 
    typedef typename T::ConstFaceHalfedgeIter FHI ; 
    typedef typename T::Point               P ; 

    VH o[3] ; 
    VH m[3] ; 
    FH f[4] ; 


    if(verbosity > 1)
    LOG(info) << "subdivide_face" << " fh " << fh ; 
                           

    unsigned n(0) ; 
    for(FHI fhe=mesh.cfh_iter(fh) ; fhe.is_valid() ; fhe++) 
    {
        const HEH& heh = *fhe ; 

        const VH& fr_ = mesh.from_vertex_handle( heh );
        const VH& to_ = mesh.to_vertex_handle( heh );

        const P& fr = mesh.point( fr_ );
        const P& to = mesh.point( to_ );
        P mi = (fr + to)/2.f ; 

        bool added(false); 
        o[n] = fr_ ;
        m[n] = build.add_vertex_unique(mi, added, epsilon);  

        //assert(added == true);   
        // not always added, as edges (and midpoints) are shared 
        
        if(verbosity > 2)
        {
          std::cout 
                    << " splitting heh " << heh 
                    << " n " << n 
                    << " o[n] " << o[n]
                    << " m[n] " << m[n]
                    << " fr " << desc(fr)
                    << " to " << desc(to)
                    << " mi " << desc(mi)
                    << std::endl ; 
        } 
 

        n++ ; 
    }
    assert( n == 3);

    bool delete_isolated_vertices = false ; 
    mesh.delete_face( fh, delete_isolated_vertices );

    f[0] = build.add_face_( o[0], m[0], m[2], verbosity ); 
    f[1] = build.add_face_( o[1], m[1], m[0], verbosity ); 
    f[2] = build.add_face_( m[1], o[2], m[2], verbosity ); 
    f[3] = build.add_face_( m[0], m[1], m[2], verbosity ); 

    if(other)
    {
        build.mark_face(f[0], other);
        build.mark_face(f[1], other);
        build.mark_face(f[2], other);
        build.mark_face(f[3], other);
    }
}
 






template <typename T>
void NOpenMeshSubdiv<T>::manual_subdivide_face(typename T::FaceHandle fh, const nnode* /*other*/, int verbosity, float epsilon)
{
/*

First tried just centroid subdivision, 
this works but leads to skinny triangles, that 
after border edge skipping yields sliver cracks.

Hmm there must be some better way to refine border tris of the mesh.
Yes, now trying to implement sqrt(3) subdivision

* https://www.graphics.rwth-aachen.de/media/papers/sqrt31.pdf
* ~/opticks_refs/sqrt3_mesh_subdivision_kobbelt_sqrt31.pdf

Thats:

* centroid face split 
* flip original edges. 

But its currently scrambling the mesh... maybe:

* as flipping "feature" edges ?
* and/or implementation error.  

TODO: study how other adaptive subdiv are done, eg how to hold on to old verts.

/usr/local/opticks/externals/openmesh/OpenMesh-4.1/src/OpenMesh/Tools/Subdivider/Adaptive/Composite/CompositeT.cc



                     o[2]
                     +
                    / \
                   / | \
                  /     \
                 /   |   \
                /         \
               /     |     \
              /             \
             /       |       \
            /       .  .      \
           /      .      .     \
          /     .           .   \
         /   .                 . \
        / .                      .\
       +---------------------------+
      o[0]                         o[1]


https://stackoverflow.com/questions/41008298/openmesh-face-split


*/
    if(verbosity > 2) LOG(info) << "subdivide_face " << fh  ;  

    std::cout << "before subdivide_face " << brief() << std::endl ;   
    //std::cout << "fh(before) " << desc(fh) << std::endl ;  


    typedef typename T::Point                P ; 
    typedef typename T::VertexHandle        VH ; 
    typedef typename T::FaceHandle          FH ; 
    typedef typename T::EdgeHandle          EH ; 
    typedef typename T::HalfedgeHandle      HEH ; 
    typedef typename T::VertexFaceIter      VFI ; 
    typedef typename T::FaceEdgeIter        FEI ; 
    typedef typename T::VertexOHalfedgeIter   VOHI ;

    P centroid = mesh.calc_face_centroid( fh );

    bool added(false) ;
    VH cvh = build.add_vertex_unique( centroid, added, epsilon );
    assert(added);

    if(verbosity > 2)
    {
        std::cout << "centroid " << desc(centroid)
                  << "cvh " << cvh 
                  << "added " << added
                  << std::endl ; 
    }


    mesh.split( fh, cvh ); 


/*

  Using approach from Tvv3 of 
  /usr/local/opticks/externals/openmesh/OpenMesh-4.1/src/OpenMesh/Tools/Subdivider/Adaptive/Composite/RulesT.cc  

  outgoing halfedges from the new splitting vertex
  nheh goes around ccw, so the oheh face gets adjacent faces
     
           
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

*/


/*
    std::vector<EH>  edge_vector;

    for( VOHI vohi=mesh.voh_iter(cvh); vohi.is_valid(); ++vohi) 
    {
        HEH heh = *vohi ; 
        FH  nfh = mesh.face_handle(heh) ;  // new faces 

        if (!nfh.is_valid()) continue ;

        HEH nheh = mesh.next_halfedge_handle(heh);
        HEH oheh = mesh.opposite_halfedge_handle(nheh) ;  // same edge, opposite direction to give access to adjacent face
    
        FH  ofh = mesh.face_handle(oheh) ;  // adjacent faces   

        if(ofh.is_valid())
        {
             EH eh = mesh.edge_handle(nheh) ; 
             if(mesh.is_flip_ok(eh))
             {
                 edge_vector.push_back(eh); 
             } 
        }
    }


    // flip edges
    while (!edge_vector.empty()) 
    {
        EH eh = edge_vector.back();
        edge_vector.pop_back();
          
        assert(mesh.is_flip_ok(eh));

        mesh.flip(eh);

        if(other)
        {
            HEH a0 = mesh.halfedge_handle(eh, 0);
            HEH a1 = mesh.halfedge_handle(eh, 1);

            FH  f0 = mesh.face_handle(a0);
            FH  f1 = mesh.face_handle(a1);

            mark_face(f0, other);
            mark_face(f1, other);
        }
   } 
*/



/*

   suspect this is a higher level 
   way of doing the above


    VFI vfbeg = mesh.vf_begin(cvh) ;
    VFI vfend = mesh.vf_end(cvh) ;

    if(other)
    {
        unsigned n(0);
        for( VFI vfi=vfbeg ; vfi != vfend ; vfi++ ) // over 3 new faces
        {
            FH f = *vfi ; 
            mark_face(f, other);
            n++ ; 
        }
        if( n != 3) LOG(warning) << "NOpenMesh::subdivide_face : unexpected face count around added vertex " << n ; 
        //assert( n == 3); 
    }

*/

    //std::cout << "desc(after)" << desc.desc()  << std::endl ; 
    std::cout << "after subdivide_face " << brief() << std::endl ;   
    std::cout << "fh(after) " << desc(fh) << std::endl ;  




}








template struct NOpenMeshSubdiv<NOpenMeshType> ;

