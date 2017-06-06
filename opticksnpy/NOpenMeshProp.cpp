#include "Nuv.hpp"
//#include "NOpenMesh.hpp"
#include "NOpenMeshProp.hpp"

template <typename T>
const char* NOpenMeshProp<T>::F_INSIDE_OTHER = "f_inside_other" ; 

template <typename T>
const char* NOpenMeshProp<T>::F_GENERATION = "f_generation" ; 

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

    mesh.add_property(v_sdf_other, V_SDF_OTHER);  
    mesh.add_property(v_parametric, V_PARAMETRIC);  
    mesh.add_property(h_boundary_loop, H_BOUNDARY_LOOP);  

    // without the below get segv on trying to delete a face
    // unless compile traits into NOpenMeshType
    /*
    mesh.request_face_status();
    mesh.request_edge_status();
    mesh.request_halfedge_status();
    mesh.request_vertex_status();
    */
 
    set_generation_all(0); // hmm perhaps too soon, no faces yet ?
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
    bool pure = facemask == ALL_OUTSIDE_OTHER || facemask == ALL_INSIDE_OTHER ;
    return fmsk > -1 ? facemask == fmsk : !pure ;
}

template <typename T>
int NOpenMeshProp<T>::get_generation( FH fh ) const 
{
    return mesh.property(f_generation, fh) ; 
}

template <typename T>
void NOpenMeshProp<T>::set_generation( FH fh, int fgen )
{
    mesh.property(f_generation, fh) = fgen ; 
}

template <typename T>
void NOpenMeshProp<T>::increment_generation( FH fh )
{
    mesh.property(f_generation, fh)++ ; 
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





template <typename T>
void NOpenMeshProp<T>::set_generation_all( int fgen )
{
    typedef typename T::FaceIter    FI ; 

    for( FI fi=mesh.faces_begin() ; fi != mesh.faces_end(); ++fi ) 
    {
        FH fh = *fi ;  
        set_generation(fh, fgen );
    } 
}








template struct NOpenMeshProp<NOpenMeshType> ;

