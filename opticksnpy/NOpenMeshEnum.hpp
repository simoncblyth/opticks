#pragma once


typedef enum {
   FIND_ALL_FACE,
   FIND_IDENTITY_FACE,
   FIND_EVENGEN_FACE,
   FIND_ODDGEN_FACE,
   FIND_FACEMASK_FACE,
   FIND_NONBOUNDARY_FACE,
   FIND_BOUNDARY_FACE,
   FIND_REGULAR_FACE,
   FIND_INTERIOR_FACE
} NOpenMeshFindType ; 


typedef enum {
   ORDER_DEFAULT_FACE,
   ORDER_REVERSE_FACE
} NOpenMeshOrderType ; 



#include "NPY_API_EXPORT.hh"

struct NPY_API NOpenMeshEnum 
{
    static const char* FIND_ALL_FACE_ ; 
    static const char* FIND_IDENTITY_FACE_ ;
    static const char* FIND_EVENGEN_FACE_ ;
    static const char* FIND_ODDGEN_FACE_ ;
    static const char* FIND_FACEMASK_FACE_ ;
    static const char* FIND_NONBOUNDARY_FACE_ ;
    static const char* FIND_BOUNDARY_FACE_ ;
    static const char* FIND_REGULAR_FACE_ ;
    static const char* FIND_INTERIOR_FACE_ ; 

    static const char* FindType( NOpenMeshFindType find );


    static const char* ORDER_DEFAULT_FACE_ ; 
    static const char* ORDER_REVERSE_FACE_ ; 
    static const char* OrderType( NOpenMeshOrderType order );



};


