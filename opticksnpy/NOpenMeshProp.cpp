#include "Nuv.hpp"
#include "NOpenMeshProp.hpp"

template <typename T>
const char* NOpenMeshProp<T>::F_INSIDE_OTHER = "f_inside_other" ; 

template <typename T>
const char* NOpenMeshProp<T>::F_GENERATION = "f_generation" ; 

template <typename T>
const char* NOpenMeshProp<T>::V_SDF_OTHER = "v_sdf_other" ; 

template <typename T>
const char* NOpenMeshProp<T>::V_PARAMETRIC = "v_parametric" ; 

template <typename T>
const char* NOpenMeshProp<T>::H_BOUNDARY_LOOP = "h_boundary_loop" ; 




template <typename T>
NOpenMeshProp<T>::NOpenMeshProp( T& mesh ) 
   : 
   mesh(mesh) 
{
    init();
} 



template <typename T>
int NOpenMeshProp<T>::get_f_inside_other( typename T::FaceHandle fh ) const 
{
    return mesh.property(f_inside_other, fh) ; 
}

template <typename T>
void NOpenMeshProp<T>::set_f_inside_other( typename T::FaceHandle fh, int facemask )
{
    mesh.property(f_inside_other, fh) = facemask ; 
}

template <typename T>
bool NOpenMeshProp<T>::is_border_face(typename T::FaceHandle fh ) const 
{
    int facemask = get_f_inside_other( fh );
    return !( facemask == ALL_OUTSIDE_OTHER || facemask == ALL_INSIDE_OTHER ) ; 
}


template <typename T>
int NOpenMeshProp<T>::get_generation( typename T::FaceHandle fh ) const 
{
    return mesh.property(f_generation, fh) ; 
}

template <typename T>
void NOpenMeshProp<T>::set_generation( typename T::FaceHandle fh, int fgen )
{
    mesh.property(f_generation, fh) = fgen ; 
}

template <typename T>
void NOpenMeshProp<T>::increment_generation( typename T::FaceHandle fh )
{
    mesh.property(f_generation, fh)++ ; 
}



template <typename T>
void NOpenMeshProp<T>::set_generation_all( int fgen )
{
    typedef typename T::FaceHandle  FH ; 
    typedef typename T::FaceIter    FI ; 

    for( FI fi=mesh.faces_begin() ; fi != mesh.faces_end(); ++fi ) 
    {
        FH fh = *fi ;  
        set_generation(fh, fgen );
    } 
}







template <typename T>
void NOpenMeshProp<T>::init()
{
    mesh.add_property(f_inside_other, F_INSIDE_OTHER);  
    mesh.add_property(f_generation, F_GENERATION);  
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
 
    set_generation_all(0);

}

template struct NOpenMeshProp<NOpenMeshType> ;

