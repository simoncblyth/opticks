#include "NOpenMeshFind.hpp"


#include <iostream>
#include <iomanip>
#include <sstream>

#include "PLOG.hh"

#include "Nuv.hpp"

#include "NOpenMeshFind.hpp"
#include "NOpenMeshProp.hpp"
#include "NOpenMesh.hpp"


template <typename T>
NOpenMeshFind<T>::NOpenMeshFind( T& mesh, const NOpenMeshProp<T>& prop, int verbosity )
    :
    mesh(mesh),
    prop(prop),
    verbosity(verbosity)
 {} 



template <typename T>
void NOpenMeshFind<T>::find_faces(std::vector<FH>& faces, NOpenMeshFindType sel, int param)
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
              ; 
}




template <typename T>
bool NOpenMeshFind<T>::is_numboundary_face(const FH fh, int numboundary)
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
bool NOpenMeshFind<T>::is_regular_face(const FH fh, int valence )
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
bool NOpenMeshFind<T>::is_interior_face(const FH fh, int margin )
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

