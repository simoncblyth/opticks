#include "NOpenMeshFind.hpp"

#include <deque>  
#include <algorithm>  
#include <iostream>
#include <iomanip>
#include <sstream>

#include <boost/lexical_cast.hpp>

#include "PLOG.hh"

#include "Nuv.hpp"

#include "NOpenMeshCfg.hpp"
#include "NOpenMeshProp.hpp"
#include "NOpenMesh.hpp"
#include "NOpenMeshFind.hpp"


template <typename T>
NOpenMeshFind<T>::NOpenMeshFind( T& mesh, 
                                  const NOpenMeshCfg& cfg, 
                                  const NOpenMeshProp<T>& prop, 
                                  int verbosity )
    :
    mesh(mesh),
    cfg(cfg),
    prop(prop),
    verbosity(verbosity)
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
void NOpenMeshFind<T>::find_faces(std::vector<FH>& faces, NOpenMeshFindType sel, int param) const 
{
    typedef typename T::ConstFaceIter        FI ; 

    faces.clear();

    unsigned totface(0); 

    for( FI fi=mesh.faces_begin() ; fi != mesh.faces_end(); ++fi ) 
    {
        totface++ ; 

        FH fh = *fi ;  

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
        }
        if(selected) faces.push_back(fh);
    }

    if(verbosity > 0)
    LOG(info) << "NOpenMeshFind<T>::find_faces  "
              << " FindType " << NOpenMeshEnum::FindType(sel)
              << " param " << param
              << " count " << faces.size()
              << " totface " << totface
              << " cfg.reversed " << cfg.reversed
              ; 

    if(totface > 1)
    {
        sort_faces(faces);
    }
}

template <typename T>
void NOpenMeshFind<T>::sort_faces(std::vector<FH>& faces) const 
{
    if(cfg.sortcontiguous > 0)
    {
        sort_faces_contiguous(faces);
    }
    else
    {
        NOpenMeshOrderType face_order = ORDER_DEFAULT_FACE ;
        if(cfg.reversed > 0)
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


    std::cout << std::setw(2) << "  " ;  
    for(unsigned i=0 ; i < faces.size() ; i++) std::cout << std::setw(3) << faces[i] ; 
    std::cout << std::endl ;

    for(unsigned i=0 ; i < faces.size() ; i++) 
    {
        for(unsigned j=0 ; j < faces.size() ; j++) 
        {
            if( j == 0 ) std::cout << std::setw(3) << i ;  
            if( i == j )
            {
                std::cout << " - " ; 
            }
            else
            {
                const FH a = faces[i] ; 
                const FH b = faces[j] ; 
                bool contig = are_contiguous(a,b) ;

                //int a_id = prop.get_identity(a);
                //int b_id = prop.get_identity(b);


                std::cout << ( contig ? " C " : " . " ) ;
            } 
        }
        std::cout << std::endl ;
    }
}



 

template <typename T>
void NOpenMeshFind<T>::sort_faces_contiguous(std::vector<FH>& faces) const 
{
    if(faces.size()==0) return ; 

    // hmm is direct contuguity to the last face needed ? or just to the clump of faces so far ?

    std::deque<FH> q(faces.begin(), faces.end()); 
    std::vector<FH> contiguous(faces.size()) ; 

    FH cursor = q[0] ; 
    int cursor_id = prop.get_identity(cursor) ;
    contiguous.push_back(cursor);

    std::map<int, std::string> idc ; 
    for(int i=0 ; i < 10 ; i++) idc[i] = boost::lexical_cast<std::string>(i) ; 
    //for(int i=10 ; i < 24 ; i++) idc[i] = boost::lexical_cast<std::string>( i-10 + 'a' ) ; 

    idc[10] = "a" ; 
    idc[11] = "b" ; 
    idc[12] = "c" ; 
    idc[13] = "d" ; 
    idc[14] = "e" ; 
    idc[15] = "f" ; 
    idc[16] = "g" ; 
    idc[17] = "h" ; 
    idc[18] = "i" ; 
    idc[19] = "j" ; 
    idc[20] = "k" ; 
    idc[21] = "l" ; 
    idc[22] = "m" ; 
    idc[23] = "n" ; 
    idc[24] = "o" ; 
    idc[25] = "p" ; 
    idc[26] = "q" ; 
    idc[27] = "r" ; 
    idc[27] = "r" ; 



    // keep pulling faces from the queue, 
    // if they are contiguous with prior face 
    // proceed to add them, if not try with the next...
    // continue until all faces are added
    // ... this of course assumes they are actually all
    // contiguous 

    int steps(0); 

    while(!q.empty() && steps < 200)
    {
        FH candidate = q.back(); q.pop_back() ; 
        int candidate_id = prop.get_identity(candidate) ; 



        bool is_contiguous = are_contiguous(cursor, candidate) ;

        std::cout 
            << " q " << std::setw(3) << q.size()
            << " c " << std::setw(3) << contiguous.size()
            << " cursor    " << std::setw(3) << cursor
            << " candidate " << std::setw(3) << candidate
            << " cursor_id    " << std::setw(3) << cursor_id
            << " candidate_id " << std::setw(3) << candidate_id
            << " idc " << std::setw(3) << idc[candidate_id]
            << ( is_contiguous ? " is_contiguous " : "" )
            << std::endl
            ;

        if(is_contiguous)
        {
            contiguous.push_back(candidate);
            cursor = candidate ; 
            cursor_id = candidate_id ; 
        }  
        else
        { 
            q.push_front(candidate);
        }

        steps++ ; 

    } 

    assert( contiguous.size() == faces.size() );   

    LOG(info) << "NOpenMeshFind<T>::sort_faces_contiguous"
              << " faces " << faces.size()
              ;

    for(unsigned i=0 ; i < faces.size() ; i++)
    {
         std::cout 
             << "  fh:" << std::setw(10) << faces[i]
             << " cfh:" << std::setw(10) << contiguous[i]
             << std::endl ; 
    }




}





template <typename T>
bool NOpenMeshFind<T>::is_numboundary_face(const FH fh, int numboundary) const 
{
    typedef typename T::ConstFaceEdgeIter   FEI ; 
    typedef typename T::EdgeHandle          EH ; 

    int n_edge(0) ;
    int n_boundary(0) ;

    for(FEI fei=mesh.cfe_iter(fh) ; fei.is_valid() ; fei++) 
    {
        EH eh = *fei ; 
        if( mesh.is_boundary(eh) ) n_boundary++ ;
        n_edge++ ; 
    }
    
    return numboundary > -1 ? n_boundary == numboundary : ( n_boundary > 0 )  ;  
}
 



template <typename T>
bool NOpenMeshFind<T>::is_regular_face(const FH fh, int valence ) const 
{
    // defining a "regular" face as one with all three vertices having 
    // valence equal to the argument value

    unsigned tot(0) ; 
    unsigned miss(0) ; 
  
    for (FVI fvi=mesh.cfv_iter(fh); fvi.is_valid(); ++fvi)
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
    for (FVI fvi=mesh.cfv_iter(fh); fvi.is_valid(); ++fvi)
    {
        VH vh = *fvi ; 
        nuv uv = prop.get_uv(vh) ; 

        if(!uv.is_interior(unsigned(margin))) return false ; 
    }
    return true ;  
}




template <typename T>
int NOpenMeshFind<T>::find_boundary_loops()
{
    //LOG(info) << "find_boundary_loops" ; 

    typedef typename T::FaceHalfedgeIter    FHI ; 
    typedef typename T::ConstEdgeIter       EI ; 
    typedef typename T::EdgeHandle          EH ; 
    typedef typename T::HalfedgeHandle      HEH ; 

    typedef std::vector<HEH>              VHEH ; 
    typedef typename VHEH::const_iterator VHEHI ; 


    unsigned he_bnd[3] ; 
    he_bnd[0] = 0 ; 
    he_bnd[1] = 0 ; 
    he_bnd[2] = 0 ; 

    loops.clear();

    for(EI e=mesh.edges_begin() ; e != mesh.edges_end() ; ++e) 
    {
        EH eh = *e ; 
        for(int i=0 ; i < 2 ; i++)
        {
            HEH heh = mesh.halfedge_handle(eh, i);
            mesh.property(prop.h_boundary_loop, heh) = 0 ; 
        }
    }


    // label halfedges with 1-based loop indices

    for(EI e=mesh.edges_begin() ; e != mesh.edges_end() ; ++e)
    {
        EH eh = *e ; 
        if(mesh.status(eh).deleted()) continue ; 

        for(int i=0 ; i < 2 ; i++)
        {
            HEH heh = mesh.halfedge_handle(eh, i);
            if(mesh.status(heh).deleted()) continue ; 

            int hbl = mesh.property(prop.h_boundary_loop, heh) ; 

            if(mesh.is_boundary(heh) && hbl == 0) 
            {
                he_bnd[i]++ ; 

                NOpenMeshBoundary<T> bnd(&mesh, heh); 
                loops.push_back(bnd);            

                for(VHEHI it=bnd.loop.begin() ; it != bnd.loop.end() ; it++) mesh.property(prop.h_boundary_loop, *it) = loops.size()  ; 
            }
        }

        he_bnd[2]++ ; 
    }

    if(verbosity > 0)
    LOG(info) << "is_boundary stats for halfedges"
              << " heh(0) " << he_bnd[0]      
              << " heh(1) " << he_bnd[1]      
              << " all " << he_bnd[2]
              << " loops " << loops.size()
              ;      

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
typename T::VertexHandle NOpenMeshFind<T>::find_vertex_epsilon(P pt, const float epsilon ) const 
{
    float distance = std::numeric_limits<float>::max() ;

    VH empty ; 
    VH closest = find_vertex_closest(pt, distance );
         
    return distance < epsilon ? closest : empty ;  
}





template struct NOpenMeshFind<NOpenMeshType> ;

