
#include <iostream>
#include <iomanip>
#include <sstream>

#include "PLOG.hh"

#include "Nuv.hpp"

#include "NOpenMeshProp.hpp"
#include "NOpenMeshDesc.hpp"
//#include "NOpenMeshBoundary.hpp"
#include "NOpenMesh.hpp"




template <typename T>
NOpenMeshDesc<T>::NOpenMeshDesc( const T& mesh, const NOpenMeshProp<T>& prop )
    :
    mesh(mesh),
    prop(prop)
 {} 



template <typename T>
int NOpenMeshDesc<T>::euler_characteristic() const 
{
/*
https://www.cs.mtu.edu/~shene/COURSES/cs3621/NOTES/model/euler.html


   V - E + F - (L - F) - 2(S - G) = 0

V: the number of vertices

E: the number of edges

F: the number of faces

G: the number of holes that penetrate the solid, usually referred to as genus in topology

S: the number of shells. A shell is an internal void of a solid. A shell is bounded by a 2-manifold surface, 
   which can have its own genus value. 
   Note that the solid itself is counted as a shell. 
   Therefore, the value for S is at least 1.

L: the number of loops, all outer and inner loops of faces are counted.


* http://www.cad.zju.edu.cn/home/zhx/GM/015/00-ism.pdf

*/

    unsigned n_faces    = std::distance( mesh.faces_begin(),    mesh.faces_end() );
    unsigned n_vertices = std::distance( mesh.vertices_begin(), mesh.vertices_end() );
    unsigned n_edges    = std::distance( mesh.edges_begin(),    mesh.edges_end() );

    assert( n_faces    == mesh.n_faces() );
    assert( n_vertices == mesh.n_vertices() );
    assert( n_edges    == mesh.n_edges() );

    int euler = n_vertices - n_edges + n_faces  ;
    return euler ; 
}

template <typename T>
std::string NOpenMeshDesc<T>::desc_euler() const 
{
    std::stringstream ss ; 
    ss 
        << " V " << mesh.n_vertices()
        << " E " << mesh.n_edges()
        << " F " << mesh.n_faces()
        << " Euler [(V - E + F)] " << euler_characteristic() 
        ;
    return ss.str();
}



template <typename T>
std::string NOpenMeshDesc<T>::desc() const 
{
    std::stringstream ss ; 

    ss << desc_euler() << std::endl ;  
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

    VI beg = mesh.vertices_begin() ;
    VI end = mesh.vertices_end() ;

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

    FI beg = mesh.faces_begin() ;
    FI end = mesh.faces_end() ;

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

    EI beg = mesh.edges_begin() ;
    EI end = mesh.edges_end() ;

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

    //OpenMesh::VPropHandleT<nuv> v_parametric;
    //assert(mesh.get_property_handle(v_parametric, NOpenMesh<T>::V_PARAMETRIC));

    nuv uv = mesh.property(prop.v_parametric, vh) ; 

    P pt = mesh.point(vh);
    HEH heh = mesh.halfedge_handle(vh); 
    //bool heh_valid = mesh.is_valid_handle(heh);
    FH fh = mesh.face_handle(heh);

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

    HEH heh0 = mesh.halfedge_handle(eh,0);
    HEH heh1 = mesh.halfedge_handle(eh,1);

    FH fh0 = mesh.face_handle( heh0 ); 
    FH fh1 = mesh.face_handle( heh1 ); 

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
    EH eh = mesh.edge_handle( heh ); 
    FH fh = mesh.face_handle( heh ); 
    VH vfr = mesh.from_vertex_handle( heh );
    VH vto = mesh.to_vertex_handle( heh );

    //std::vector<HEH> loop ; 
    //NOpenMeshBoundary<T>::CollectLoop( &mesh, heh, loop );

    std::stringstream ss ; 
    ss 
       << " heh " << std::setw(5) << heh 
       << " eh " << std::setw(5) << eh 
       << " fh " << std::setw(5) << fh 
       << " v( " << vfr << ":" << vto << ")"
      // << (*this)(loop, 10) 
        ;

    return ss.str();
}

template <typename T>
std::string NOpenMeshDesc<T>::operator()(const typename T::FaceHandle fh) const 
{
    typedef typename T::ConstFaceHalfedgeIter   FHI ; 

    bool valid = mesh.is_valid_handle(fh) ;
    int id = valid ? prop.get_identity(fh) : -1 ; 

    std::stringstream ss ; 
    ss 
       << " fh " << std::setw(4) << fh 
       << " id " << std::setw(4) << id 
       << std::endl ;


    if(valid)
    {
        std::vector<VH> vtos ; 
        for(FHI fhe=mesh.cfh_iter(fh) ; fhe.is_valid() ; fhe++) 
        {
            HEH heh = *fhe ; 
            VH vto = mesh.to_vertex_handle( heh );
            vtos.push_back(vto);
            ss <<  (*this)( heh ) << std::endl ; 
        }

        for(unsigned i=0 ; i < vtos.size() ; i++) ss << (*this)(vtos[i]) << std::endl ; 
    }


    return ss.str();
}





template <typename T>
void NOpenMeshDesc<T>::dump_vertices(const char* msg) const 
{
    LOG(info) << msg ; 

    typedef typename T::Point          P ; 
    typedef typename T::VertexIter     VI ; 

    VI beg = mesh.vertices_begin() ;
    VI end = mesh.vertices_end() ;

    for (VI vit=beg ; vit != end ; ++vit) 
    {
        VH vh = *vit ; 
        int idx = vh.idx() ;
        assert( idx == std::distance( beg, vit ) ) ;

        const P& p = mesh.point(vh); 

        const HEH heh = mesh.halfedge_handle(vh); 
        bool heh_valid = mesh.is_valid_handle(heh);

        std::cout 
             << " vh " << std::setw(5) << vh  
             << " p " 
             << "[" 
             << std::setw(15) << std::fixed << std::setprecision(4) << p[0] << ","
             << std::setw(15) << std::fixed << std::setprecision(4) << p[1] << ","
             << std::setw(15) << std::fixed << std::setprecision(4) << p[2] << ","
             << "]"
             << " heh " << std::setw(5) << heh  
             ;

        if(heh_valid)
        {
            const EH eh = mesh.edge_handle(heh); 
            const VH tvh = mesh.to_vertex_handle(heh);
            const VH fvh = mesh.from_vertex_handle(heh);
            const FH fh  = mesh.face_handle(heh);
            bool bnd = mesh.is_boundary(heh);

            std::cout  
                << " fvh->tvh " 
                << std::setw(3) << fvh << "->" 
                << std::setw(3) << tvh   
                << " fh " << std::setw(5) << fh  
                << " eh " << std::setw(5) << eh  
                << " bnd " << std::setw(5) << bnd 
                << std::endl ;
        }
        else
        {
             std::cout << std::endl ; 
        }

    }
}







template <typename T>
void NOpenMeshDesc<T>::dump_faces(const char* msg ) const 
{
    LOG(info) << msg << " nface " << mesh.n_faces() ; 

    typedef typename T::FaceIter            FI ; 
    typedef typename T::ConstFaceVertexIter FVI ; 
    typedef typename T::Point               P ; 

    for( FI f=mesh.faces_begin() ; f != mesh.faces_end(); ++f ) 
    {
        int f_idx = f->idx() ;  
        std::cout << " f " << std::setw(4) << *f 
                  << " i " << std::setw(3) << f_idx 
                  << " v " << std::setw(3) << mesh.valence(*f) 
                  << " : " 
                  ; 

        // over points of the face 
        for(FVI fv=mesh.cfv_iter(*f) ; fv.is_valid() ; fv++) 
             std::cout << std::setw(3) << *fv << " " ;

        for(FVI fv=mesh.cfv_iter(*f) ; fv.is_valid() ; fv++) 
             std::cout 
                       << std::setprecision(3) << std::fixed << std::setw(20) 
                       << mesh.point(*fv) << " "
                       ;

        std::cout << std::endl ; 
    }
}
 
 






template <typename T>
std::string NOpenMeshDesc<T>::desc_point(const typename T::Point& pt, int w, int p) 
{
    std::stringstream ss ; 
    ss 
        << " (" 
        << std::setw(w) << std::setprecision(p) << std::fixed << pt[0]
        << "," 
        << std::setw(w) << std::setprecision(p) << std::fixed << pt[1]
        << "," 
        << std::setw(w) << std::setprecision(p) << std::fixed << pt[2]
        << ") " 
        ;

    return ss.str();
}



template <typename T>
std::string NOpenMeshDesc<T>::operator()(const typename T::Point& pt) const 
{
    return desc_point(pt,10,3);
}


template struct NOpenMeshDesc<NOpenMeshType> ;

