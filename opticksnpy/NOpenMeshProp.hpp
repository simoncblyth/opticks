#pragma once

#include <vector>
struct nuv ; 

#include "NOpenMeshType.hpp"

template <typename T>
struct NPY_API  NOpenMeshProp
{
    typedef typename T::FaceHandle FH ; 

    enum
    {
        ALL_OUTSIDE_OTHER = 0,
        ALL_INSIDE_OTHER = 7  
    };


    static const char* F_INSIDE_OTHER ; 
    static const char* F_GENERATION ; 
    static const char* F_IDENTITY ;
 
    static const char* V_SDF_OTHER ; 
    static const char* V_PARAMETRIC ; 
    static const char* H_BOUNDARY_LOOP ; 

    NOpenMeshProp( T& mesh );
    void init();

    bool is_border_face( FH fh ) const ;
    int  get_facemask( FH fh ) const ;
    void set_facemask( FH fh, int facemask );

    bool is_identity_face( FH fh, const std::vector<int>& identity ) const ;
    bool is_identity_face( FH fh, int identity ) const ;
    int  get_identity( FH fh ) const ;
    void set_identity( FH fh, int identity );

    int  get_generation( FH fh ) const ;
    void set_generation( FH fh, int fgen );
    void increment_generation( FH fh );
    void set_generation_all( int fgen );


    T& mesh  ;

    OpenMesh::VPropHandleT<nuv>    v_parametric ;
    OpenMesh::VPropHandleT<float>  v_sdf_other ;

    OpenMesh::FPropHandleT<int>    f_inside_other ;
    OpenMesh::FPropHandleT<int>    f_generation ;
    OpenMesh::FPropHandleT<int>    f_identity ;

    OpenMesh::HPropHandleT<int>    h_boundary_loop ;

};
 



