#pragma once

struct nuv ; 

#include "NOpenMeshType.hpp"

template <typename T>
struct NPY_API  NOpenMeshProp
{
    enum
    {
        ALL_OUTSIDE_OTHER = 0,
        ALL_INSIDE_OTHER = 7  
    };


    static const char* F_INSIDE_OTHER ; 
    static const char* F_GENERATION ; 
    static const char* V_SDF_OTHER ; 
    static const char* V_PARAMETRIC ; 
    static const char* H_BOUNDARY_LOOP ; 

    NOpenMeshProp( T& mesh );
    void init();

    bool is_border_face(    typename T::FaceHandle fh ) const ;
    int  get_f_inside_other( typename T::FaceHandle fh ) const ;
    void set_f_inside_other( typename T::FaceHandle fh, int facemask );

    int  get_generation( typename T::FaceHandle fh ) const ;
    void set_generation( typename T::FaceHandle fh, int fgen );
    void increment_generation( typename T::FaceHandle fh );
    void set_generation_all( int fgen );


    T& mesh  ;

    OpenMesh::VPropHandleT<nuv>    v_parametric ;
    OpenMesh::VPropHandleT<float>  v_sdf_other ;
    OpenMesh::FPropHandleT<int>    f_inside_other ;
    OpenMesh::FPropHandleT<int>    f_generation ;
    OpenMesh::HPropHandleT<int>    h_boundary_loop ;

};
 



