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
    NOpenMeshProp<T>& prop, 
    const NOpenMeshDesc<T>& desc, 
    const NOpenMeshFind<T>& find, 
          NOpenMeshBuild<T>& build,
    int verbosity,
    float epsilon
    )
    : 
    mesh(mesh),
    prop(prop),
    desc(desc),
    find(find),
    build(build),
    verbosity(verbosity),
    epsilon(epsilon),
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
void NOpenMeshSubdiv<T>::create_soup(typename T::FaceHandle fh, const nnode* other  )
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
typename T::VertexHandle NOpenMeshSubdiv<T>::centroid_split_face(typename T::FaceHandle fh )
{
/*
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
    typedef typename T::Point                P ; 
    typedef typename T::VertexHandle        VH ; 

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
    return cvh ; 
}


template <typename T>
void NOpenMeshSubdiv<T>::sqrt3_split_r(typename T::FaceHandle fh, const nnode* other )
{
/*

  Using approach from Tvv3 of 
  /usr/local/opticks/externals/openmesh/OpenMesh-4.1/src/OpenMesh/Tools/Subdivider/Adaptive/Composite/RulesT.cc  

  Trying to implement sqrt(3) subdivision

   * https://www.graphics.rwth-aachen.de/media/papers/sqrt31.pdf
   * ~/opticks_refs/sqrt3_mesh_subdivision_kobbelt_sqrt31.pdf

   * centroid face split 
   * flip original edges. 

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


But its currently scrambling the mesh... maybe:

* as flipping "feature" edges ?
* and/or implementation error.  

TODO: study how other adaptive subdiv are done, eg how to hold on to old verts.

/usr/local/opticks/externals/openmesh/OpenMesh-4.1/src/OpenMesh/Tools/Subdivider/Adaptive/Composite/CompositeT.cc

*/
    if(verbosity > 2) LOG(info) << "subdivide_face " << fh  ;  


    typedef typename T::Point                P ; 
    typedef typename T::VertexHandle        VH ; 
    typedef typename T::FaceHandle          FH ; 
    typedef typename T::EdgeHandle          EH ; 
    typedef typename T::HalfedgeHandle      HEH ; 
    typedef typename T::VertexFaceIter      VFI ; 
    typedef typename T::FaceEdgeIter        FEI ; 
    typedef typename T::VertexOHalfedgeIter   VOHI ;


    int base_id = prop.get_identity( fh ); 
    int base_gen = prop.get_generation( fh ); 
    int new_gen = base_gen + 1 ; 

    bool even = base_gen % 2 == 0 ; 

    if(even)
    {
        VH cvh = centroid_split_face( fh );

        int iface = 0 ; 
        for( VOHI vohi=mesh.voh_iter(cvh); vohi.is_valid(); ++vohi) 
        {
            HEH heh = *vohi ; 
            FH  nfh = mesh.face_handle(heh) ;  // new faces 
            assert( nfh.is_valid() );

            iface++ ; 
            prop.set_identity( nfh, base_id*100 + iface ) ;
            prop.set_generation(nfh, new_gen ) ;

            HEH nheh = mesh.next_halfedge_handle(heh);
            HEH oheh = mesh.opposite_halfedge_handle(nheh) ;  // same edge, opposite direction to give access to adjacent face
            FH  ofh = mesh.face_handle(oheh) ;  // adjacent faces   
         
            if(ofh.is_valid())
            {
                int adjacent_gen = prop.get_generation(ofh) ;
                if( adjacent_gen == new_gen ) // its been split 
                {
                     EH eh = mesh.edge_handle(nheh) ; 
                     if(mesh.is_flip_ok(eh))
                     {
                         mesh.flip(eh);

                         HEH a0 = mesh.halfedge_handle(eh, 0);
                         HEH a1 = mesh.halfedge_handle(eh, 1);
                         FH  f0 = mesh.face_handle(a0);
                         FH  f1 = mesh.face_handle(a1);

                         prop.increment_generation(f0);
                         prop.increment_generation(f1);
                     } 
                }
            }
            else
            {
                std::cout << "invalid ofh : " 
                          << " base_id " << base_id 
                          << " iface " << iface
                          << std::endl ; 
            }
        }
    }
    else
    {
        HEH heh = mesh.halfedge_handle(fh); 
        HEH nheh = mesh.next_halfedge_handle(heh);
        HEH oheh = mesh.opposite_halfedge_handle(nheh) ;  // same edge, opposite direction to give access to adjacent face
        FH  ofh = mesh.face_handle(oheh) ;  // adjacent faces   
 
        if(ofh.is_valid())
        {
            int adjacent_gen = prop.get_generation(ofh) ; 
            if(adjacent_gen == base_gen - 2)
            {
                sqrt3_split_r(ofh, other) ;  
            }
        }
    }
}


template struct NOpenMeshSubdiv<NOpenMeshType> ;

