#include "Nuv.hpp"
//#include "NOpenMesh.hpp"
#include "NOpenMeshEnum.hpp"
#include "NOpenMeshProp.hpp"

template <typename T>
const char* NOpenMeshProp<T>::F_INSIDE_OTHER = "f_inside_other" ; 

template <typename T>
const char* NOpenMeshProp<T>::F_GENERATION = "f_generation" ; 


#ifdef WITH_V_GENERATION
template <typename T>
const char* NOpenMeshProp<T>::V_GENERATION = "v_generation" ; 
#endif


template <typename T>
const char* NOpenMeshProp<T>::F_IDENTITY = "f_identity" ; 

template <typename T>
const char* NOpenMeshProp<T>::V_SDF_OTHER = "v_sdf_other" ; 

template <typename T>
const char* NOpenMeshProp<T>::V_PARAMETRIC = "v_parametric" ; 

template <typename T>
const char* NOpenMeshProp<T>::H_BOUNDARY_LOOP = "h_boundary_loop" ; 


template <typename T>
void NOpenMeshProp<T>::init()
{
    mesh.add_property(f_inside_other, F_INSIDE_OTHER);  
    mesh.add_property(f_generation, F_GENERATION);  
    mesh.add_property(f_identity, F_IDENTITY);  

#ifdef WITH_V_GENERATION
    mesh.add_property(v_generation, V_GENERATION);  
#endif
    mesh.add_property(v_sdf_other, V_SDF_OTHER);  
    mesh.add_property(v_parametric, V_PARAMETRIC);  
    mesh.add_property(h_boundary_loop, H_BOUNDARY_LOOP);  

    //init_status();  

   // hmm perhaps too soon, no faces yet ?
    set_fgeneration_all(0); 
#ifdef WITH_V_GENERATION
    set_vgeneration_all(0); 
#endif
}


template <typename T>
void NOpenMeshProp<T>::init_status()
{
    // not needed as specified at compilation, see NOpenMeshType
    // without status get segv on trying to delete a face

    mesh.request_face_status();
    mesh.request_edge_status();
    mesh.request_halfedge_status();
    mesh.request_vertex_status();
}



template <typename T>
void NOpenMeshProp<T>::update_normals()
{
    mesh.request_face_normals();
    mesh.request_vertex_normals();
    mesh.update_normals();
}
 



template <typename T>
NOpenMeshProp<T>::NOpenMeshProp( T& mesh ) 
   : 
   mesh(mesh) 
{
    init();
} 



template <typename T>
bool NOpenMeshProp<T>::is_identity_face(FH fh, const std::vector<int>& identity ) const 
{
    int face_id = get_identity( fh );
    return std::find(identity.begin(), identity.end(), face_id ) != identity.end()  ; 
}

template <typename T>
bool NOpenMeshProp<T>::is_identity_face(FH fh, int identity ) const 
{
    int face_id = get_identity( fh );
    return face_id == identity ; 
}


template <typename T>
std::string NOpenMeshProp<T>::get_identity_b36(FH fh) const 
{
    int face_id = get_identity( fh );
    return b36(face_id) ; 
}

template <typename T>
std::string NOpenMeshProp<T>::get_index_b36(FH fh) const 
{
    return b36(fh.idx()) ; 
}






template <typename T>
int NOpenMeshProp<T>::get_identity( FH fh ) const 
{
    return mesh.property(f_identity, fh) ; 
}

template <typename T>
void NOpenMeshProp<T>::set_identity( FH fh, int identity )
{
    mesh.property(f_identity, fh) = identity ; 
}





template <typename T>
int NOpenMeshProp<T>::get_facemask( FH fh ) const 
{
    return mesh.property(f_inside_other, fh) ; 
}
template <typename T>
void NOpenMeshProp<T>::set_facemask( FH fh, int facemask )
{
    mesh.property(f_inside_other, fh) = facemask ; 
}
template <typename T>
bool NOpenMeshProp<T>::is_facemask_face(FH fh, int fmsk ) const 
{
    int facemask = get_facemask( fh );
    bool pure = facemask == PROP_OUTSIDE_OTHER || facemask == PROP_INSIDE_OTHER ;
    return fmsk == PROP_FRONTIER ? !pure : facemask == fmsk ;
}



template <typename T>
int NOpenMeshProp<T>::get_fgeneration( FH fh ) const 
{
    return mesh.property(f_generation, fh) ; 
}
template <typename T>
void NOpenMeshProp<T>::set_fgeneration( FH fh, int fgen )
{
    mesh.property(f_generation, fh) = fgen ; 
}
template <typename T>
void NOpenMeshProp<T>::increment_fgeneration( FH fh )
{
    mesh.property(f_generation, fh)++ ; 
}
template <typename T>
bool NOpenMeshProp<T>::is_even_fgeneration(const FH fh, int mingen) const 
{
    int fgen = get_fgeneration(fh) ;
    bool even = fgen % 2 == 0 ;
    return fgen >= mingen && even ; 
}
template <typename T>
bool NOpenMeshProp<T>::is_odd_fgeneration(const FH fh, int mingen)  const 
{
    int fgen = get_fgeneration(fh) ;
    bool odd = fgen % 2 == 1 ;
    return fgen >= mingen && odd ; 
}
template <typename T>
void NOpenMeshProp<T>::set_fgeneration_all( int fgen )
{
    typedef typename T::FaceIter    FI ; 
    for( FI fi=mesh.faces_begin() ; fi != mesh.faces_end(); ++fi ) 
    {
        FH fh = *fi ;  
        set_fgeneration(fh, fgen );
    } 
}





#ifdef WITH_V_GENERATION
template <typename T>
int NOpenMeshProp<T>::get_vgeneration( VH vh ) const 
{
    return mesh.property(v_generation, vh) ; 
}
template <typename T>
void NOpenMeshProp<T>::set_vgeneration( VH vh, int vgen )
{
    mesh.property(v_generation, vh) = vgen ; 
}
template <typename T>
void NOpenMeshProp<T>::increment_vgeneration( VH vh )
{
    mesh.property(v_generation, vh)++ ; 
}
template <typename T>
bool NOpenMeshProp<T>::is_even_vgeneration(const VH vh, int mingen) const 
{
    int vgen = get_vgeneration(vh) ;
    bool even = vgen % 2 == 0 ;
    return vgen >= mingen && even ; 
}
template <typename T>
bool NOpenMeshProp<T>::is_odd_vgeneration(const VH vh, int mingen)  const 
{
    int vgen = get_vgeneration(vh) ;
    bool odd = vgen % 2 == 1 ;
    return vgen >= mingen && odd ; 
}
template <typename T>
void NOpenMeshProp<T>::set_vgeneration_all( int vgen )
{
    typedef typename T::VertexIter    VI ; 
    for( VI vi=mesh.vertices_begin() ; vi != mesh.vertices_end(); ++vi ) 
    {
        VH vh = *vi ;  
        set_vgeneration(vh, vgen );
    } 
}
#endif



template <typename T>
int NOpenMeshProp<T>::get_hbloop( HEH heh ) const 
{
    return mesh.property(h_boundary_loop, heh) ; 
}
template <typename T>
void NOpenMeshProp<T>::set_hbloop( HEH heh, int hbl )
{
    mesh.property(h_boundary_loop, heh) = hbl ; 
}
template <typename T>
void NOpenMeshProp<T>::set_hbloop_all( int hbl  )
{
    typedef typename T::EdgeIter EI ; 

    for(EI ei=mesh.edges_begin() ; ei != mesh.edges_end() ; ++ei) 
    {
        EH eh = *ei ; 
        HEH a = mesh.halfedge_handle(eh, 0);
        HEH b = mesh.halfedge_handle(eh, 1);  

       // although technically only one of the heh of an edge 
       // can ever be a boundary, this is just setting a starting point value

        set_hbloop(a, hbl );
        set_hbloop(b, hbl );
    }
}











template <typename T>
nuv NOpenMeshProp<T>::get_uv( VH vh ) const 
{
    nuv uv = mesh.property(v_parametric, vh) ; 
    return uv ; 
}

template <typename T>
void NOpenMeshProp<T>::set_uv( VH vh, nuv uv )
{
    mesh.property(v_parametric, vh) = uv  ; 
}












template struct NOpenMeshProp<NOpenMeshType> ;

