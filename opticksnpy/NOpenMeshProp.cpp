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
void NOpenMeshProp<T>::init()
{

    mesh.add_property(f_inside_other, F_INSIDE_OTHER);  
    mesh.add_property(f_generation, F_GENERATION);  
    mesh.add_property(v_sdf_other, V_SDF_OTHER);  
    mesh.add_property(v_parametric, V_PARAMETRIC);  
    mesh.add_property(h_boundary_loop, H_BOUNDARY_LOOP);  

    // without the below get segv on trying to delete a face
    // unless compile in the traits
    /*
    mesh.request_face_status();
    mesh.request_edge_status();
    mesh.request_halfedge_status();
    mesh.request_vertex_status();
    */
 

}

template struct NOpenMeshProp<NOpenMeshType> ;

