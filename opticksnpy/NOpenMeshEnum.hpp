#pragma once


typedef enum {
   FIND_ALL_FACE,
   FIND_IDENTITY_FACE,
   FIND_FACEMASK_FACE,
   FIND_NONBOUNDARY_FACE,
   FIND_BOUNDARY_FACE,
   FIND_REGULAR_FACE,
   FIND_INTERIOR_FACE
} NOpenMeshFindType ; 


#include "NPY_API_EXPORT.hh"

struct NPY_API NOpenMeshEnum 
{
    static const char* FIND_ALL_FACE_ ; 
    static const char* FIND_IDENTITY_FACE_ ;
    static const char* FIND_FACEMASK_FACE_ ;
    static const char* FIND_NONBOUNDARY_FACE_ ;
    static const char* FIND_BOUNDARY_FACE_ ;
    static const char* FIND_REGULAR_FACE_ ;
    static const char* FIND_INTERIOR_FACE_ ; 

    static const char* FindType( NOpenMeshFindType find );

};


