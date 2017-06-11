#include "NOpenMeshFind.hpp"

#include <deque>  
#include <algorithm>  
#include <iostream>
#include <iomanip>
#include <sstream>

#include <boost/lexical_cast.hpp>

#include "PLOG.hh"
#include "SBase36.hh"

#include "Nuv.hpp"

#include "NOpenMeshCfg.hpp"
#include "NOpenMeshProp.hpp"
#include "NOpenMesh.hpp"
#include "NOpenMeshFind.hpp"


#include "NOpenMeshTraverse.hpp"


template <typename T>
NOpenMeshFind<T>::NOpenMeshFind( T& mesh, 
                                  const NOpenMeshCfg* cfg, 
                                  NOpenMeshProp<T>& prop, 
                                  const nnode* node
                                )
    :
    mesh(mesh),
    cfg(cfg),
    prop(prop),
    verbosity(cfg->verbosity),
    node(node)
{} 



template <typename T>
struct NOpenMeshOrder
{
    typedef typename T::FaceHandle FH  ; 

    NOpenMeshOrder(NOpenMeshOrderType order) : order(order) {} ;  

    bool operator() (const FH a, const FH b) const 
    {
        bool cmp = false ; 
        switch(order)
        {
           case ORDER_DEFAULT_FACE: cmp = a < b     ;break;
           case ORDER_REVERSE_FACE: cmp = !(a < b)  ;break;
        }
        return cmp  ;
    }

    NOpenMeshOrderType order ; 
};





template <typename T>
bool NOpenMeshFind<T>::is_selected(const FH fh, NOpenMeshFindType sel, int param) const 
{
    bool selected = false ; 
    switch(sel)
    {
       case FIND_ALL_FACE      :    selected = true                             ; break ; 
       case FIND_IDENTITY_FACE :    selected = prop.is_identity_face(fh, param) ; break ;
       case FIND_EVENGEN_FACE  :    selected = prop.is_even_fgeneration(fh, param) ; break ;
       case FIND_ODDGEN_FACE   :    selected = prop.is_odd_fgeneration(fh, param) ; break ;
       case FIND_FACEMASK_FACE :    selected = prop.is_facemask_face(fh, param) ; break ;
       case FIND_REGULAR_FACE  :    selected = is_regular_face(fh, param)  ; break ; 
       case FIND_INTERIOR_FACE :    selected = is_interior_face(fh, param) ; break ; 
       case FIND_NONBOUNDARY_FACE : selected = is_numboundary_face(fh, 0)  ; break ; 
       case FIND_BOUNDARY_FACE :    selected = is_numboundary_face(fh, -1)  ; break ; 
       case FIND_CORNER_FACE      : selected = is_numboundary_face(fh, 2)  ; break ; 
       case FIND_SIDE_FACE        : selected = is_numboundary_face(fh, 1)  ; break ; 
       case FIND_SIDECORNER_FACE  : selected = is_side_or_corner_face(fh)  ; break ; 
    }
    return selected ; 
}


template <typename T>
typename T::FaceHandle NOpenMeshFind<T>::first_face(NOpenMeshFindType sel, int param) const 
{
    typedef typename T::ConstFaceIter        FI ; 
    FH first ;  
    for( FI fi=mesh.faces_begin() ; fi != mesh.faces_end(); ++fi ) 
    {
        FH fh = *fi ;  
        bool selected = is_selected(fh, sel, param) ; 
        if(selected) 
        {
            first = fh ;  
            break ; 
        }
    } 
    return first ;  
}

template <typename T>
typename T::FaceHandle NOpenMeshFind<T>::first_face(const std::vector<FH>& faces, NOpenMeshFindType sel, int param) const 
{
    typedef typename T::ConstFaceIter        FI ; 
    FH first ;  
    for( unsigned i=0 ; i < faces.size() ; i++)
    {
        FH fh = faces[i] ; ;  
        bool selected = is_selected(fh, sel, param) ; 
        if(selected) 
        {
            first = fh ;  
            break ; 
        }
    } 
    return first ;   
}

template <typename T>
void NOpenMeshFind<T>::find_faces(std::vector<FH>& faces, NOpenMeshFindType sel, int param) const 
{
    typedef typename T::ConstFaceIter        FI ; 

    faces.clear();

    unsigned totface(0); 

    for( FI fi=mesh.faces_begin() ; fi != mesh.faces_end(); ++fi ) 
    {
        totface++ ; 
        FH fh = *fi ;  
        bool selected = is_selected(fh, sel, param) ; 
        if(selected) faces.push_back(fh);
    }

    if(verbosity > 0)
    LOG(info) << "NOpenMeshFind<T>::find_faces  "
              << " FindType " << NOpenMeshEnum::FindType(sel)
                  << " param " << param
                  << " count " << faces.size()
                  << " totface " << totface
                  << " cfg.reversed " << cfg->reversed
                  ; 

}

template <typename T>
void NOpenMeshFind<T>::sort_faces(std::vector<FH>& faces) const 
{
    if(cfg->sortcontiguous > 0)
    {
        sort_faces_contiguous(faces);
        //sort_faces_contiguous_monolithic(faces);
    }
    else
    {
        NOpenMeshOrderType face_order = ORDER_DEFAULT_FACE ;
        if(cfg->reversed > 0)
        {
            LOG(warning) << "using reversed face order" ; 
            face_order = ORDER_REVERSE_FACE ;         
        }
        NOpenMeshOrder<T> order(face_order) ; 
        std::sort( faces.begin(), faces.end(), order );
    }
}

template <typename T>
bool NOpenMeshFind<T>::are_contiguous(const FH a, const FH b) const 
{
    for (CFFI cffi = mesh.cff_iter(a); cffi.is_valid(); ++cffi) 
    {
        const FH fh = *cffi ; 
        if( fh == b ) return true ; 
    }
    return false ; 
}


template <typename T>
void NOpenMeshFind<T>::dump_contiguity( const std::vector<FH>& faces ) const 
{
    unsigned nf = faces.size(); 
    LOG(info) << "NOpenMeshFind<T>::dump_contiguity"
              << " nf " << nf 
               ; 


    std::cout << std::setw(2) << "  " ;  
    for(unsigned i=0 ; i < nf ; i++) std::cout << std::setw(3) << prop.get_identity_b36(faces[i]) ; 
    std::cout << std::endl ;

    for(unsigned i=0 ; i < nf ; i++) 
    {
        for(unsigned j=0 ; j < nf ; j++) 
        {
            if( j == 0 ) std::cout << std::setw(3) << prop.get_identity_b36(faces[i]) ;  // left column of each line of table

            if( i == j )
            {
                std::cout << "   " ; 
            }
            else
            {
                bool contig = are_contiguous(faces[i],faces[j]) ;
                std::cout << ( contig ? " C " : " . " ) ;
            } 
        }
        std::cout << std::endl ;
    }
}



template <typename T>
std::string NOpenMeshFind<T>::desc_face_i(const FH fh) const 
{
    std::stringstream ss ; 
    ss 
       << " fh(" << std::setw(2) << fh << ")"
       << " id(" << std::setw(1) << prop.get_identity_b36(fh) << ")"
       ;
    return ss.str();
}
template <typename T>
std::string NOpenMeshFind<T>::desc_face_v(const FH fh) const 
{
    std::vector<VH> vhs ; 
    for (CFVI fvi=mesh.cfv_iter(fh); fvi.is_valid(); ++fvi)
    {
        VH vh = *fvi ; 
        vhs.push_back(vh);
   }

    std::stringstream ss ; 
    ss << " {" ; 
    unsigned nv=vhs.size();
    for(unsigned i=0 ; i < nv ; i++)
    {
        ss << prop.b36(vhs[i].idx()) ; 
        if( i < nv - 1) ss << "," ; 
    }
    ss << "} " ; 
    return ss.str();
}

template <typename T>
std::string NOpenMeshFind<T>::desc_face(const FH fh) const 
{
    std::stringstream ss ; 
    ss << desc_face_i(fh) << desc_face_v(fh) ;
    return ss.str();
}

template <typename T>
void NOpenMeshFind<T>::dump_faces(std::vector<FH>& faces) const 
{
    LOG(info) << "NOpenMeshFind<T>::dump_faces"
              << " verbosity " << verbosity
              << " nf " << faces.size()
              ; 

    for(unsigned i=0 ; i < faces.size() ; i++)
         std::cout 
            << std::setw(4) << i 
            << " "
            << desc_face( faces[i] ) 
            << std::endl
             ;

}



template <typename T>
void NOpenMeshFind<T>::sort_faces_contiguous(std::vector<FH>& faces) const 
{
    if(faces.size()==0) return ; 

    if(verbosity > 0)
    LOG(info) << "NOpenMeshFind<T>::sort_faces_contiguous"
              << " faces " << faces.size()
              ;


    

    FH seed = first_face( faces, FIND_SIDECORNER_FACE, -1 ); 

    if(!mesh.is_valid_handle(seed))
    {
        LOG(warning) << "failed to find SIDECORNER seed, trying FIND_ALL_FACE ";
        seed = first_face( faces, FIND_ALL_FACE, -1 ); 
    }


    assert( mesh.is_valid_handle(seed)  ); 



    NOpenMeshTraverse<T> trav(mesh, *this, faces, seed, verbosity) ; 

} 

template <typename T>
void NOpenMeshFind<T>::sort_faces_contiguous_monolithic(std::vector<FH>& faces) const 
{
/*
    This does not yield a nice continuous traversal of 
    the mesh, the places where the algo gets stuck and needs a kick
    with the offset cause jumps in the sequence ...

    However in hexpatch testing it appears to be good enough to 
    avoid the too many flips at once issue with the contiguous mode
    sqrt3 subdiv. 

    Also the set of faces are intersested in refining 
    are CSG sub-object "grey" tris which cross the 
    frontier between CSG sub-objects, those should be (or can
    be made to be a continuous loop).

    If this needs to be improved, need to investigate 
  
    * :google:`mesh breadth first traversal`
    * graph traversals : tricolor algorithm 
    * http://www.cs.cornell.edu/courses/cs2112/2012sp/lectures/lec24/lec24-12sp.html


In addition to being contiguous, also need to avoid ending on an interior
face to avoid the last piece in jigsaw 3-flips issue, need to end on side or corner face.

... hmm difficult to arrange to end somewhere so instead start there
and reverse the sequence as a post step

*/


    assert( 0 && "dont use this use the NOpenMeshTraverse one "); 


    unsigned nfaces = faces.size() ;

    if(nfaces ==0) return ; 

    // pick seed face at border
    FH seed = first_face( faces, FIND_SIDECORNER_FACE, -1 ); 

    bool valid = mesh.is_valid_handle(seed) ;
    if(!valid)
    {
        LOG(warning) << "sort_faces_contiguous failed to FIND_SIDECORNER_FACE " ;
        return ; 
    } 

    LOG(info) << "sort_faces_contiguous"
              << " nf " << faces.size()
              << " seed " << desc_face(seed) 
              ;

    // queue the faces and remove the seed

    std::deque<FH> q(faces.begin(), faces.end()); 
    typedef typename std::deque<FH>::iterator DFHI ;  
    DFHI it = std::find(q.begin(), q.end(), seed );
    assert( it != q.end() );
    q.erase(it) ; 
    assert( q.size() == faces.size() - 1 );

    std::deque<FH> contiguous  ; 
    contiguous.push_back(seed);

    unsigned steps(0); 
    unsigned since(0); 
    unsigned offset = 0 ;  

    // places where gets stuck lead to jumps 

    while(!q.empty() && steps < 100)
    {
        FH candidate = q.back(); q.pop_back() ; 
        FH tip = contiguous[contiguous.size() - 1 - offset] ; 

        bool is_contiguous = are_contiguous(tip, candidate) ;
        bool is_stuck = since > q.size() ; 

        if(verbosity > 4)
        std::cout
            << " q " << std::setw(3) << q.size()
            << " c " << std::setw(3) << contiguous.size()
            << " since " << std::setw(3) << since
            << " o " << std::setw(3) << offset
            << "      "
            << " tip " << desc_face(tip)  
            << "      "
            << " can " << desc_face(candidate) 
            << "      "
            << ( is_contiguous ? " CONTIGUOUS " : "" )
            << ( is_stuck ? " STUCK " : "" )
            << std::endl 
            ;

        if(is_contiguous)
        {
            contiguous.push_back(candidate);
            since = 0 ; 
            offset = 0 ; 
        }  
        else
        { 
            q.push_front(candidate);
            since++ ;
        }

        assert( contiguous.size() + q.size() == nfaces );


        if(is_stuck && offset + 1 < contiguous.size() )
        {
            offset++ ; 
            since = 0 ; 
        }

        steps++ ; 
    } 

    assert( contiguous.size() == faces.size() );   

    LOG(info) << "NOpenMeshFind<T>::sort_faces_contiguous"
              << " faces " << faces.size()
              ;

    faces.assign( contiguous.rbegin(), contiguous.rend() );
    for(unsigned i=0 ; i < faces.size() ; i++)
    {
         std::cout  << desc_face( faces[i] ) << std::endl ;
    }


}






template <typename T>
unsigned NOpenMeshFind<T>::get_num_boundary(const FH fh) const 
{
    typedef typename T::ConstFaceEdgeIter   FEI ; 
    typedef typename T::EdgeHandle          EH ; 
    unsigned num_boundary(0) ;
    for(FEI fei=mesh.cfe_iter(fh) ; fei.is_valid() ; fei++) 
    {
        EH eh = *fei ; 
        if( mesh.is_boundary(eh) ) num_boundary++ ;
    }
    return num_boundary ; 
}


template <typename T>
bool NOpenMeshFind<T>::is_numboundary_face(const FH fh, int numboundary) const 
{
    int n_boundary = get_num_boundary(fh); 
    return numboundary > -1 ? n_boundary == numboundary : ( n_boundary > 0 )  ;  
}
 

template <typename T>
bool NOpenMeshFind<T>::is_side_or_corner_face(const FH fh) const 
{
    int n_boundary = get_num_boundary(fh); 
    return n_boundary > 0 ;  
}



template <typename T>
bool NOpenMeshFind<T>::is_regular_face(const FH fh, int valence ) const 
{
    // defining a "regular" face as one with all three vertices having 
    // valence equal to the argument value

    unsigned tot(0) ; 
    unsigned miss(0) ; 
  
    for (CFVI fvi=mesh.cfv_iter(fh); fvi.is_valid(); ++fvi)
    {
        VH vh = *fvi ; 
        unsigned vhv = mesh.valence(vh) ;
        tot++ ; 

        if(vhv != unsigned(valence)) miss++ ; 
    }

    bool is_regular = miss == 0 ; 
    return is_regular ; 
}
 



template <typename T>
bool NOpenMeshFind<T>::is_interior_face(const FH fh, int margin ) const 
{
    for (CFVI fvi=mesh.cfv_iter(fh); fvi.is_valid(); ++fvi)
    {
        VH vh = *fvi ; 
        nuv uv = prop.get_uv(vh) ; 

        if(!uv.is_interior(unsigned(margin))) return false ; 
    }
    return true ;  
}



template <typename T>
void NOpenMeshFind<T>::dump_boundary_loops(const char* msg, bool detail)
{
    LOG(info) << msg ; 
    unsigned nloop = get_num_boundary_loops();
    std::cout << " nloop " << nloop << std::endl ;
    for(unsigned i=0 ; i < nloop ; i++)
    {
        NOpenMeshBoundary<T>& loop = get_boundary_loop(i);
        std::cout << loop.desc() << std::endl ; 
    }

    if(detail)
    {
        for(unsigned i=0 ; i < nloop ; i++)
        {
            NOpenMeshBoundary<T>& loop = get_boundary_loop(i);
            loop.dump();
        }
    }
}


template <typename T>
unsigned  NOpenMeshFind<T>::get_num_boundary_loops()
{
    return loops.size();
}
template <typename T>
NOpenMeshBoundary<T>& NOpenMeshFind<T>::get_boundary_loop(unsigned i)
{
    return loops[i] ; 
}



template <typename T>
int NOpenMeshFind<T>::find_boundary_loops()
{
    //LOG(info) << "find_boundary_loops" ; 

    typedef typename T::FaceHalfedgeIter    FHI ; 
    typedef typename T::ConstEdgeIter       CEI ; 
    typedef typename T::EdgeHandle          EH ; 
    typedef typename T::HalfedgeHandle      HEH ; 

    typedef std::vector<HEH>              VHEH ; 

    unsigned he_bnd[3] ; 
    he_bnd[0] = 0 ; 
    he_bnd[1] = 0 ; 
    he_bnd[2] = 0 ; 

    loops.clear();

    prop.set_hbloop_all( 0 );  

    // initialize to 0, then subsequenntly set 
    // to =1-based loop indices
    //
    // NB the hbl labelling as loops are found prevents the loop finding 
    // from being repeated for every heh, instead it is only done 
    // for each distinct loop

    for(CEI e=mesh.edges_begin() ; e != mesh.edges_end() ; ++e)
    {
        EH eh = *e ; 
        if(mesh.status(eh).deleted()) continue ; 

        for(int i=0 ; i < 2 ; i++)
        {
            HEH heh = mesh.halfedge_handle(eh, i);
            if(mesh.status(heh).deleted()) continue ; 

            int hbl = prop.get_hbloop(heh) ;
 
            if(mesh.is_boundary(heh) && hbl == 0)  
            {
                he_bnd[i]++ ; 

                NOpenMeshBoundary<T> bnd(mesh, cfg, prop, heh, node); 
                loops.push_back(bnd);            

                int loop_index = loops.size() ;
                bnd.set_loop_index( loop_index ); // sets hbl for all heh in the loop
            }
        }

        he_bnd[2]++ ; 
    }

/*
    if(verbosity > 0)
    LOG(info) << "is_boundary stats for halfedges"
              << " heh(0) " << he_bnd[0]      
              << " heh(1) " << he_bnd[1]      
              << " all " << he_bnd[2]
              << " loops " << loops.size()
              ;      
*/

    return loops.size();
}
           


template <typename T>
typename T::VertexHandle NOpenMeshFind<T>::find_vertex_exact(P pt) const 
{
    typedef typename T::VertexHandle   VH ;
    typedef typename T::VertexIter     VI ; 

    VH result ;

    VI beg = mesh.vertices_begin() ;
    VI end = mesh.vertices_end() ;

    for (VI vit=beg ; vit != end ; ++vit) 
    {
        VH vh = *vit ; 
        const P& p = mesh.point(vh); 
        if( p == pt )
        {
            result = vh ; 
            break ;          
        }
    }
    return result ; 
}



template <typename T>
typename T::VertexHandle NOpenMeshFind<T>::find_vertex_closest(P pt, float& distance ) const 
{
    typedef typename T::VertexHandle   VH ;
    typedef typename T::VertexIter     VI ; 

    VI beg = mesh.vertices_begin() ;
    VI end = mesh.vertices_end() ;

    VH closest ;

    for (VI vit=beg ; vit != end ; ++vit) 
    {
        VH vh = *vit ; 
        const P& p = mesh.point(vh); 

        P d = p - pt ; 
        float dlen = d.length();

        if(dlen < distance )
        {
            closest = vh ;
            distance = dlen ;  
        }
    }
    return closest ; 
}


template <typename T>
typename T::VertexHandle NOpenMeshFind<T>::find_vertex_epsilon(P pt, const float epsilon_ ) const 
{
    float distance = std::numeric_limits<float>::max() ;
    float epsilon = epsilon_ < 0 ? cfg->epsilon : epsilon_ ; 

    VH empty ; 
    VH closest = find_vertex_closest(pt, distance );
         
    return distance < epsilon ? closest : empty ;  
}





template struct NOpenMeshFind<NOpenMeshType> ;

