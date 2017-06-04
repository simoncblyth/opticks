#pragma once

struct nuv ; 

#include "NOpenMeshType.hpp"

template <typename T>
struct NPY_API  NOpenMeshProp
{

    static const char* F_INSIDE_OTHER ; 
    static const char* F_GENERATION ; 
    static const char* V_SDF_OTHER ; 
    static const char* V_PARAMETRIC ; 
    static const char* H_BOUNDARY_LOOP ; 

    NOpenMeshProp( T& mesh );
    void init();

    T& mesh  ;

    OpenMesh::VPropHandleT<nuv>    v_parametric ;
    OpenMesh::VPropHandleT<float>  v_sdf_other ;
    OpenMesh::FPropHandleT<int>    f_inside_other ;
    OpenMesh::FPropHandleT<int>    f_generation ;
    OpenMesh::HPropHandleT<int>    h_boundary_loop ;

};
 



