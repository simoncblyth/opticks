
#include <iostream>
#include <iomanip>
#include <sstream>

#include "Nuv.hpp"

#include "NOpenMeshDesc.hpp"
#include "NOpenMeshBoundary.hpp"
#include "NOpenMesh.hpp"



template <typename T>
NOpenMeshDesc<T>::NOpenMeshDesc( const T* mesh ) : mesh(mesh) {} 


template <typename T>
std::string NOpenMeshDesc<T>::desc() const 
{
    std::stringstream ss ; 
 
    ss << "desc.vertices" << std::endl << vertices() << std::endl ; 
    ss << "desc.faces" << std::endl << faces() << std::endl ; 
    ss << "desc.edges" << std::endl << edges() << std::endl ; 

    return ss.str();
}


template <typename T>
std::string NOpenMeshDesc<T>::vertices() const 
{
    typedef typename T::VertexHandle   VH ; 
    typedef typename T::ConstVertexIter     VI ; 

    VI beg = mesh->vertices_begin() ;
    VI end = mesh->vertices_end() ;

    std::stringstream ss ; 

    for (VI vi=beg ; vi != end ; ++vi) 
    {
        VH vh = *vi ;
        ss << (*this)(vh) << std::endl ; 
    }

    return ss.str();
}



template <typename T>
std::string NOpenMeshDesc<T>::faces() const 
{
    typedef typename T::VertexHandle   VH ; 
    typedef typename T::FaceHandle     FH ; 
    typedef typename T::ConstFaceIter  FI ; 

    FI beg = mesh->faces_begin() ;
    FI end = mesh->faces_end() ;

    std::stringstream ss ; 

    for (FI fi=beg ; fi != end ; ++fi) 
    {
        FH fh = *fi ;
        ss << (*this)(fh) << std::endl ; 
    }

    return ss.str();
}



template <typename T>
std::string NOpenMeshDesc<T>::edges() const 
{
    typedef typename T::VertexHandle   VH ; 
    typedef typename T::EdgeHandle     EH ; 
    typedef typename T::ConstEdgeIter  EI ; 

    EI beg = mesh->edges_begin() ;
    EI end = mesh->edges_end() ;

    std::stringstream ss ; 

    for (EI ei=beg ; ei != end ; ++ei) 
    {
        EH eh = *ei ;
        ss << (*this)(eh) << std::endl ; 
    }

    return ss.str();
}







template <typename T>
std::string NOpenMeshDesc<T>::operator()(const typename T::VertexHandle vh) const 
{
    typedef typename T::Point            P ; 
    typedef typename T::FaceHandle      FH ; 
    typedef typename T::HalfedgeHandle HEH ; 


    OpenMesh::VPropHandleT<nuv> v_parametric;
    assert(mesh->get_property_handle(v_parametric, NOpenMesh<T>::V_PARAMETRIC));

    nuv uv = mesh->property(v_parametric, vh) ; 

    P pt = mesh->point(vh);
    HEH heh = mesh->halfedge_handle(vh); 
    //bool heh_valid = mesh->is_valid_handle(heh);
    FH fh = mesh->face_handle(heh);

    std::stringstream ss ; 
    ss 
       << " vh:" << std::setw(4) << vh 
       << (*this)(pt)
       << uv.desc()
       << "  fh: " << std::setw(5) << fh  
       << (*this)(heh)  
       ;  


    return ss.str();
}


template <typename T>
std::string NOpenMeshDesc<T>::operator()(const std::vector<typename T::HalfedgeHandle> loop, unsigned mx) const 
{
    std::stringstream ss ; 
    unsigned nhe = loop.size() ;
    ss << " loop: " ;
    for(unsigned i=0 ; i < std::min(nhe,mx) ; i++) ss << " " << loop[i] ; 
    if( nhe > mx ) ss << "..." ;
    ss << " (" << nhe << ")" ;  
    return ss.str();
}


template <typename T>
std::string NOpenMeshDesc<T>::operator()(const typename T::EdgeHandle eh) const 
{
    typedef typename T::HalfedgeHandle      HEH ; 
    typedef typename T::VertexHandle        VH ; 
    typedef typename T::FaceHandle          FH ; 

    std::stringstream ss ; 

    HEH heh0 = mesh->halfedge_handle(eh,0);
    HEH heh1 = mesh->halfedge_handle(eh,1);

    FH fh0 = mesh->face_handle( heh0 ); 
    FH fh1 = mesh->face_handle( heh1 ); 

    ss << " eh " << std::setw(4) << eh << std::endl 
       <<  (*this)(heh0) << std::endl 
       <<  (*this)(heh1) << std::endl 
       ;

    ss 
        << (*this)(fh0) << std::endl 
        << (*this)(fh1) << std::endl  
        ;

    return ss.str();
}


template <typename T>
std::string NOpenMeshDesc<T>::operator()(const typename T::HalfedgeHandle heh) const 
{
    typedef typename T::HalfedgeHandle      HEH ; 
    typedef typename T::VertexHandle        VH ; 
    typedef typename T::FaceHandle          FH ; 

    FH fh = mesh->face_handle( heh ); 
    VH vfr = mesh->from_vertex_handle( heh );
    VH vto = mesh->to_vertex_handle( heh );

    std::vector<HEH> loop ; 
    NOpenMeshBoundary<T>::CollectLoop( mesh, heh, loop );

    std::stringstream ss ; 
    ss 
       << " heh " << std::setw(5) << heh 
       << " fh " << std::setw(5) << fh 
       << " vfr-to: " << vfr << "-" << vto  
       << (*this)(loop, 10) 
        ;

    return ss.str();
}

template <typename T>
std::string NOpenMeshDesc<T>::operator()(const typename T::FaceHandle fh) const 
{
    typedef typename T::ConstFaceHalfedgeIter   FHI ; 
    typedef typename T::HalfedgeHandle      HEH ; 
    typedef typename T::VertexHandle        VH ; 

    std::stringstream ss ; 
    ss << "fh " << std::setw(4) << fh << std::endl ;
 
    std::vector<VH> vtos ; 
 
    for(FHI fhe=mesh->cfh_iter(fh) ; fhe.is_valid() ; fhe++) 
    {
        HEH heh = *fhe ; 
        VH vto = mesh->to_vertex_handle( heh );
        vtos.push_back(vto);
        ss <<  (*this)( heh ) << std::endl ; 
    }

    for(unsigned i=0 ; i < vtos.size() ; i++) ss << (*this)(vtos[i]) << std::endl ; 

    return ss.str();
}


template <typename T>
std::string NOpenMeshDesc<T>::operator()(const typename T::Point& pt) const 
{
    std::stringstream ss ; 
    ss 
        << " (" 
        << std::setw(15) << std::setprecision(3) << std::fixed << pt[0]
        << "," 
        << std::setw(15) << std::setprecision(3) << std::fixed << pt[1]
        << "," 
        << std::setw(15) << std::setprecision(3) << std::fixed << pt[2]
        << ") " 
        ;

    return ss.str();
}


template struct NOpenMeshDesc<NOpenMeshType> ;

