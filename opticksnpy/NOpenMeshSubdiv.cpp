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
void NOpenMeshSubdiv<T>::sqrt3_split_r(typename T::FaceHandle fh, const nnode* other, int depth )
{
/*
  Using approach from Tvv3 of 
  /usr/local/opticks/externals/openmesh/OpenMesh-4.1/src/OpenMesh/Tools/Subdivider/Adaptive/Composite/RulesT.cc  
  /usr/local/opticks/externals/openmesh/OpenMesh-4.1/src/OpenMesh/Tools/Subdivider/Adaptive/Composite/CompositeT.cc

  Trying to implement sqrt(3) subdivision : centroid face split, flip original edges 

  NB the recursion is limited, it does not spread over the entire mesh, it just applies splits to neighbouring 
  faces when that is necessary for mesh conformance/balancing 

   * https://www.graphics.rwth-aachen.de/media/papers/sqrt31.pdf
   * ~/opticks_refs/sqrt3_mesh_subdivision_kobbelt_sqrt31.pdf

           
                      +               0,1,2    : outgoing halfedges from the new splitting vertex
                     / \              0n,1n,2n : nheh halfedges (ccw), oheh to give adjacent faces
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
    typedef typename T::FaceHandle          FH ; 
    typedef typename T::EdgeHandle          EH ; 
    typedef typename T::HalfedgeHandle      HEH ; 
    typedef typename T::VertexFaceIter      VFI ; 
    typedef typename T::FaceEdgeIter        FEI ; 
    typedef typename T::VertexOHalfedgeIter   VOHI ;

    int base_id = prop.get_identity( fh ); 
    int base_gen = prop.get_generation( fh ); 

    bool even = base_gen % 2 == 0 ; 

    if(verbosity > 2) 
    std::cout << "sqrt3_split_r" 
              << " face fh " << std::setw(4) << fh 
              << " base_gen " << std::setw(4) << base_gen 
              << " base_id " << std::setw(4) << base_id
              << " depth " << std::setw(4) << depth
              << std::endl 
               ;  

    if(even)
    {
        P centroid = mesh.calc_face_centroid( fh );
        bool added(false) ;
        VH cvh = build.add_vertex_unique( centroid, added, epsilon );
        assert(added && "centroid face splitting should always add a new vertex??");
        mesh.split( fh, cvh ); 

        int iface = 0 ; 

        // outgoing halfedges from the splitting vertex
        for( VOHI vohi=mesh.voh_iter(cvh); vohi.is_valid(); ++vohi) 
        {
            HEH heh = *vohi ; 
            FH  nfh = mesh.face_handle(heh) ;  // new faces 
            assert( nfh.is_valid() );
            iface++ ; 

            int newface_id = base_id*100 + iface  ;
            int newface_gen = base_gen + 1 ; 

            prop.set_identity( nfh, newface_id ) ;
            prop.set_generation( nfh, newface_gen ) ;

            HEH nheh = mesh.next_halfedge_handle(heh);
            HEH oheh = mesh.opposite_halfedge_handle(nheh) ;  
            FH  ofh = mesh.face_handle(oheh) ; 
            // next then opposite_halfedge face is adjacent 
            bool adjacent_valid = ofh.is_valid() ; 

            if(verbosity > 2)
            std::cout 
                   << " (even) " 
                   << " newface_id " << std::setw(4) << newface_id 
                   << " newface_gen " << std::setw(4) << newface_gen 
                   << " adjacent_valid " << ( adjacent_valid ? "Y" : "N" )
                   ;
 
            if(adjacent_valid)
            {
                int adjacent_id = prop.get_identity(ofh) ;  
                int adjacent_gen = prop.get_generation(ofh) ;
                bool do_flip = adjacent_gen == newface_gen ;

                if(verbosity > 2)
                std::cout 
                   << " adjacent_id " << std::setw(4) << adjacent_id 
                   << " adjacent_gen " << std::setw(4) << adjacent_gen
                   << " do_flip " << ( do_flip ? "YES" : "NO" )
                   << std::endl ; 

                if( do_flip ) 
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
                     else
                     {
                         LOG(warning) << "flip not ok,  fh " << fh ;  
                     }
                }
            }
            else
            {
                if(verbosity > 2)
                std::cout 
                       << std::endl ; 
                       ;
                     
            }
        }
    }
    else
    {
        if(verbosity > 2)
        std::cout 
                << " (odd) " 
                << std::endl ; 
 

        HEH heh = mesh.halfedge_handle(fh); 
        HEH nheh = mesh.next_halfedge_handle(heh);
        HEH oheh = mesh.opposite_halfedge_handle(nheh) ;  
        FH  ofh = mesh.face_handle(oheh) ; 

        // next then opposite_halfedge face is adjacent 
        // (hmm perhaps this depends on the vertex->halfedge ordering of the original mesh?) 
 
        if(ofh.is_valid())
        {
            int adjacent_id = prop.get_identity(ofh) ;  
            int adjacent_gen = prop.get_generation(ofh) ;
            bool two_gen = adjacent_gen == base_gen - 2 ;

            if(verbosity > 2)
            std::cout 
                << " (odd) " 
                << " adjacent_id " << adjacent_id
                << " adjacent_gen " << adjacent_gen
                << " two_gen " << ( two_gen ? "YES" : "NO" )
                << std::endl ; 

            if(two_gen)
            {
                sqrt3_split_r(ofh, other, depth+1) ;  
            }
            //sqrt3_split_r(ofh, other, depth+1) ;  

        }
    }
}


template struct NOpenMeshSubdiv<NOpenMeshType> ;

