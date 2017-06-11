#include <sstream>

#include "PLOG.hh"

#include "NOpenMeshCfg.hpp"
#include "NOpenMeshProp.hpp"
#include "NOpenMeshDesc.hpp"
#include "NOpenMeshFind.hpp"
#include "NOpenMeshBuild.hpp"
#include "NOpenMeshSubdiv.hpp"


/*
  Using approach from Tvv3 of 
  /usr/local/opticks/externals/openmesh/OpenMesh-4.1/src/OpenMesh/Tools/Subdivider/Adaptive/Composite/RulesT.cc  
  /usr/local/opticks/externals/openmesh/OpenMesh-4.1/src/OpenMesh/Tools/Subdivider/Adaptive/Composite/CompositeT.cc

  Trying to implement sqrt(3) subdivision pseudocode : centroid face split, flip original edges 

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


    tripatch testing

    6  verts :                                 4 faces
                                                                     
                 5                                +           
                / \                              /4\
               2---4                            +---+
              / \ / \                          /2\1/3\
             0---1---3                        +---+---+

    add_face_(v[1],v[4],v[2], 1);
    add_face_(v[1],v[2],v[0], 2);
    add_face_(v[4],v[1],v[3], 3);
    add_face_(v[2],v[4],v[5], 4);

    add_face_ order dictates subdiv order...
    
         putting the middle face last leads to no flips until placing this
         last one ... whence get wierd issue of seeing 4 heh from the 
         added centroid ?? which leads to an extra flip 
    
         * how is this possible, generation check should avoid ? 
           YES but : flipping all edges of a triangle leads to confusion 
       
         * moral of story : dont flip all edges of a triangle
           at one subdiv if you wish to maintain sanity, instead try 
           to pace the flips so one flip happens 
    
           putting the middle face first enables flips to happen for every 
           centroid subdiv as there is always an adjacent other that has been split
           already to flip with 
      
           seems that subdiv face order needs to contiguous, so need to 
           arrange face order like that... around in circles ?
    
      how to generate a list of faces ... 
      need to traverse the mesh in ever increasing circles
      of connected faces 
          https://mailman.rwth-aachen.de/pipermail/openmesh/2015-April/001083.html
          https://mailman.rwth-aachen.de/pipermail/openmesh/2015-April/001084.html

      Hmm it may be possible to do this with test meshes, but with real world
      ones this isnt practical, try instead to avoid face order sensitivity by 
      doing the split and the flip is separate passes.



*contiguous* 
    problems with contiguous are

    1. contiguous face order is needed
    2. must avoid last piece in jigsaw issue (ie refining a face that
       has already been refined on all three other sides leads to 
       failed treble flipping). Potentially there are other similar 
       issues from failed double flips...

*phased*
    problems with phased are

    1. near to last flips, mess up : issue with flipping edges that have 
       already undergone a flip 


*/


using namespace OpenMesh::Subdivider;


template <typename T>
NOpenMeshSubdiv<T>::NOpenMeshSubdiv( 
    T& mesh, 
    const NOpenMeshCfg* cfg, 
    NOpenMeshProp<T>& prop, 
    const NOpenMeshDesc<T>& desc, 
    const NOpenMeshFind<T>& find, 
          NOpenMeshBuild<T>& build
    )
    : 
    mesh(mesh),
    cfg(cfg),
    prop(prop),
    desc(desc),
    find(find),
    build(build),
    verbosity(cfg->verbosity),
    subdivider(new subdivider_t(mesh))
{
    init();
} 


template <typename T>
void NOpenMeshSubdiv<T>::init()
{
    if(verbosity > 3)
    LOG(info) << "NOpenMeshSubdiv<T>::init()" ;
    //std::cout << cfg.brief() << std::endl ;

    //init_subdivider();
}
 

template <typename T>
void NOpenMeshSubdiv<T>::init_subdivider()
{
   assert(0 && "thus composite divider aint working, just hangs " );

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


/*
Doing splits then immediatelt flips introduces a face ordering 
sensitivity, as the faces need to be contiguous with other formerly refined ones
in order that the flipping will find a same fgeneration pair.
  
Instead if the splits and the flips could be done in two passes, I think 
the face ordering sensitivity can be eliminated, as all splits will
have been done when flipping.

Hmm, can the flips be done in any order though ?


But that means need a way to identity the edges that need to be flipped...

* given a list of faces after a split, 
  how to find the splitting vertex in order to find the edges to flip ?
  they will all be odd fgeneration, so that doesnt help

* ... could use a vgeneration ?  so then just iterate over all vertices finding odd vgeneration 
  that need flips

*/


template <typename T>
void NOpenMeshSubdiv<T>::sqrt3_split_r( FH fh, int depth )
{
    int base_id = prop.get_identity( fh ); 
    int base_gen = prop.get_fgeneration( fh ); 
    bool even = base_gen % 2 == 0 ; 

    if(verbosity > 2) std::cout << desc_face(fh, "sqrt3_split") << " depth " << std::setw(4) << depth << std::endl ;  

    if(even)
    {
        P centroid = mesh.calc_face_centroid( fh );
        bool added(false) ;
        VH cvh = build.add_vertex_unique( centroid, added );
        if(!added) LOG(warning) << " fh " << fh << " base_id " << base_id << " NON_UNIQUE CENTROID ? " << desc(centroid)   ;

        mesh.split( fh, cvh ); 

        int iface = 0 ; 
        for( VOHI vohi=mesh.voh_iter(cvh); vohi.is_valid(); ++vohi) // outgoing halfedges from the splitting vertex
        {
            HEH heh       = *vohi  ;                         assert( heh.is_valid() );
            FH  adj_face  = next_opposite_face(heh) ;       
            HEH next_heh  = mesh.next_halfedge_handle(heh);  assert( next_heh.is_valid() ) ; 
            FH  new_face  = mesh.face_handle(heh) ;          assert( new_face.is_valid() );

            iface++ ; 
            prop.set_identity(   new_face, base_id*100 + iface ) ;
            prop.set_fgeneration( new_face, base_gen + 1 ) ;

            if(verbosity > 2) std::cout << desc_face(new_face, "even:nf") << std::endl  ;
            if(!adj_face.is_valid()) continue ;          // no adj_face on border

            bool same_generation = prop.get_fgeneration(adj_face) == base_gen + 1 ;
            if(verbosity > 2) std::cout << desc_face(adj_face, "adj_face") << ( same_generation ? " DO_FLIP" : "" ) << std::endl ; 
            if( same_generation ) sqrt3_flip_edge( next_heh );
        }
    }
    else
    {
        HEH heh      = mesh.halfedge_handle(fh);     assert( heh.is_valid() );
        FH  adj_face = next_opposite_face(heh) ;
        if(adj_face.is_valid())           // no adj_face on border
        { 
            bool adj_behind = prop.get_fgeneration(adj_face) == base_gen - 2 ;
            if(verbosity > 2) std::cout << desc_face(adj_face, "(odd)") << ( adj_behind ? " RECURSE" : "" ) << std::endl ;   
            if(adj_behind) sqrt3_split_r(adj_face, depth+1) ;  
        }
    }
}



template <typename T>
void NOpenMeshSubdiv<T>::sqrt3_refine_phased( const std::vector<FH>& target )
{
    bool split = cfg->split > 0 ;
    bool flip = cfg->flip > 0 ;

    LOG(info) << "sqrt3_refine_phased" 
              << ( split ? " SPLIT " : " " )
              << " verbosity " << verbosity
              << " cfg " << cfg->desc()
              << " n_target " << target.size()
              << desc.desc_euler()
              ;

    std::vector<VH> centroid_vertices ; 
    if(split)
    {
        for(unsigned i=0 ; i < target.size() ; i++) 
        {
            FH fh = target[i] ;
            sqrt3_centroid_split_face(fh, centroid_vertices);
        }
        LOG(info) << " centroid_vertices " << centroid_vertices.size()  ; 
    }

    if(flip)
    {
        int numflip = centroid_vertices.size() ;
        if( cfg->numflip < 0 )     numflip += cfg->numflip ;  // -ve numflip, reduces the total 
        else if( cfg->numflip > 0) numflip = cfg->numflip ;   // +ve numflip, set absolute value
        else if( cfg->numflip == 0) assert( numflip > 0 );   // leave asis 
        int maxflip = cfg->maxflip ; 

        LOG(info) << "FLIP centroid_vertices " << centroid_vertices.size()
                  << " cfg.numflip " << cfg->numflip
                  << " numflip " << numflip
                  << " maxflip " << maxflip
                  ;


        for( int i=0 ; i < numflip ; i++)
        {
            VH cvh = centroid_vertices[i] ; 
            if(mesh.valence(cvh)== 3) 
            {
                sqrt3_flip_adjacent_edges(cvh, maxflip);
               // vertex valence:3 indicates a freshly added splitting vertex 
               // without any flips yet, avoid multi-flipping by requiring this
            }
            else
            {
                std::cout << " valence skip-flip " << desc_vertex(cvh, "cvh") << std::endl ;  
            }
        } 
    }
}

template <typename T>
void NOpenMeshSubdiv<T>::sqrt3_refine_contiguous( std::vector<FH>& target )
{
    find.sort_faces_contiguous( target );

    if(verbosity > 0)
    LOG(info) << "NOpenMeshSubdiv<T>::sqrt3_refine_contiguous START"
              << " verbosity " << verbosity 
              << " target " << target.size()
              ;

    for(unsigned i=0 ; i < target.size() ; i++) 
    {
        FH fh = target[i] ;
        sqrt3_split_r(fh, 0);
    }

    if(verbosity > 0)
    LOG(info) << "NOpenMeshSubdiv<T>::sqrt3_refine_contiguous DONE"
              << " verbosity " << verbosity 
              << " target " << target.size()
              ;

}



template <typename T>
void NOpenMeshSubdiv<T>::sqrt3_refine( NOpenMeshFindType select, int param  )
{
    std::vector<FH> target ; 
    find.find_faces( target, select,  param );

    bool contiguous = cfg->contiguous > 0 ; 
    bool phased = cfg->phased > 0 ; 

    assert( contiguous ^ phased );

    if(phased)
    {
         sqrt3_refine_phased(target);
    }
    else
    {
         sqrt3_refine_contiguous(target);
    }

    mesh.garbage_collection();  // NB this invalidates handles, so dont hold on to them across this point
}


template <typename T>
void NOpenMeshSubdiv<T>::sqrt3_centroid_split_face(FH fh, std::vector<VH>& centroid_vertices)
{
    int base_id = prop.get_identity( fh ); 
    int base_gen = prop.get_fgeneration( fh ); 
    assert( base_gen % 2 == 0 && "expecting even fgeneration" ); 

    P centroid = mesh.calc_face_centroid( fh );

    bool added(false) ;

    VH cvh = build.add_vertex_unique( centroid, added );

    if(!added) LOG(fatal) << desc_face(fh, "DUPE-CENTROID?") << desc(centroid) ;
    assert(added); 

    mesh.split( fh, cvh ); 
    centroid_vertices.push_back(cvh);

    int iface = 0 ; 
    for( VOHI vohi=mesh.voh_iter(cvh); vohi.is_valid(); ++vohi) // outgoing halfedges from the splitting vertex
    {
        HEH heh       = *vohi  ;                         assert( heh.is_valid() );
        HEH next_heh  = mesh.next_halfedge_handle(heh);  assert( next_heh.is_valid() ) ; 
        FH  new_face  = mesh.face_handle(heh) ;          assert( new_face.is_valid() );

        iface++ ; 
        prop.set_identity(   new_face, base_id*100 + iface ) ;
        prop.set_fgeneration( new_face, base_gen + 1 ) ;
    } 
}

template <typename T>
void NOpenMeshSubdiv<T>::sqrt3_flip_adjacent_edges(const VH cvh, int maxflip)
{
    int nflip(0);
    for( VOHI vohi=mesh.voh_iter(cvh); vohi.is_valid(); ++vohi) // outgoing halfedges from the splitting vertex
    {
        HEH heh       = *vohi  ;                         assert( heh.is_valid() );
        FH  new_face  = mesh.face_handle(heh) ;          assert( new_face.is_valid() );
        HEH next_heh  = mesh.next_halfedge_handle(heh);  assert( next_heh.is_valid() ) ; 

        FH  adj_face  = next_opposite_face(heh) ;       

        if(verbosity > 2) std::cout << desc_face(new_face, "even:nf") << std::endl  ;
        if(!adj_face.is_valid()) continue ;   // no adj_face on border

        bool same_fgeneration = prop.get_fgeneration(adj_face) == prop.get_fgeneration(new_face) ;

        if(same_fgeneration)
        { 
            bool flip_limited = maxflip != 0 && nflip >= maxflip ; 
            if(flip_limited)
            {    
                if(verbosity > 2) 
                std::cout << desc_face(adj_face, "adj_face") 
                          << " FLIP_LIMITED " 
                          << " nflip " << nflip 
                          << " maxflip " << maxflip 
                          << std::endl ; 
            }
            else
            {
                if(verbosity > 2) 
                std::cout << desc_face(adj_face, "adj_face") 
                          << " DO_FLIP " 
                          << " nflip " << nflip 
                          << " maxflip " << maxflip 
                          << std::endl ; 
 
                sqrt3_flip_edge( next_heh );
                nflip++ ; 
            }

        }
    }
}

template <typename T>
void NOpenMeshSubdiv<T>::sqrt3_flip_edge(typename T::HalfedgeHandle heh)
{
     EH eh = mesh.edge_handle(heh) ; 
     if(mesh.is_flip_ok(eh))
     {
         mesh.flip(eh);  

         HEH a0 = mesh.halfedge_handle(eh, 0);
         HEH a1 = mesh.halfedge_handle(eh, 1);
         FH  f0 = mesh.face_handle(a0);
         FH  f1 = mesh.face_handle(a1);

         prop.increment_fgeneration(f0);
         prop.increment_fgeneration(f1);

         int f0gen =  prop.get_fgeneration(f0) ;
         int f1gen =  prop.get_fgeneration(f1) ;

         if(verbosity > 2)
         std::cout 
             << " sqrt3_flip_edge " << eh
             << " f0gen " << f0gen 
             << " f1gen " << f1gen
             << std::endl ;  

     } 
     else
     {
         LOG(warning) << "sqrt3_flip_edge CANNOT FLIP : eh " << eh ;  
     }

}





template <typename T>
typename T::FaceHandle NOpenMeshSubdiv<T>::next_opposite_face(HEH heh)
{
    // next then opposite_halfedge face is adjacent 
    // (hmm perhaps this depends on the vertex->halfedge ordering of the original mesh?) 

    HEH nheh = mesh.next_halfedge_handle(heh);
    HEH oheh = mesh.opposite_halfedge_handle(nheh) ;  
    FH  ofh = mesh.face_handle(oheh) ; 
    return ofh ; 
}

template <typename T>
std::string NOpenMeshSubdiv<T>::desc_face(const FH fh, const char* label)
{
     int id = prop.get_identity(fh) ;  
     int fgen = prop.get_fgeneration(fh) ;

     std::stringstream ss ; 
     ss 
       << std::setw(10) << label
       << " fh " << std::setw(4) << fh 
       << " id " << std::setw(4) << id 
       << " fgen " << std::setw(1) << fgen
       ; 

     return ss.str();
}
 
template <typename T>
std::string NOpenMeshSubdiv<T>::desc_vertex(const VH vh, const char* label)
{
     std::stringstream ss ; 
     ss 
       << std::setw(10) << label
       << " vh " << std::setw(4) << vh 
       << " val " << mesh.valence(vh)
       ; 
     return ss.str();
}

template <typename T>
std::string NOpenMeshSubdiv<T>::desc_edge(const EH eh, const char* label)
{
     std::stringstream ss ; 
     ss 
       << std::setw(10) << label
       << " eh " << std::setw(4) << eh 
       ; 
     return ss.str();
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
        m[n] = build.add_vertex_unique(mi, added );  

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
 




template struct NOpenMeshSubdiv<NOpenMeshType> ;

