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
   FIND_INTERIOR_FACE,
   FIND_CORNER_FACE,
   FIND_SIDE_FACE,
   FIND_SIDECORNER_FACE
} NOpenMeshFindType ; 


typedef enum {
   ORDER_DEFAULT_FACE,
   ORDER_REVERSE_FACE
} NOpenMeshOrderType ; 


typedef enum {
   PROP_OUTSIDE_OTHER = 0,
   PROP_INSIDE_OTHER = 7,
   PROP_FRONTIER = -1  
} NOpenMeshPropType ; 



typedef enum { 
   COMP_COMBINED, 
   COMP_LEFT, 
   COMP_RIGHT 
} NOpenMeshCompType ;


typedef enum {
   COMBINE_HYBRID,
   COMBINE_CSGBSP 
} NOpenMeshCombineType ;
// NOpenMeshMode_t ; 



#include "NPY_API_EXPORT.hh"

struct NPY_API NOpenMeshEnum 
{
    static unsigned OpenMeshVersion();

    static const char* COMBINE_HYBRID_ ; 
    static const char* COMBINE_CSGBSP_ ; 
    static const char* CombineType(NOpenMeshCombineType meshmode) ;
    static NOpenMeshCombineType CombineTypeFromPoly(const char* poly);

    static const char* FIND_ALL_FACE_ ; 
    static const char* FIND_IDENTITY_FACE_ ;
    static const char* FIND_EVENGEN_FACE_ ;
    static const char* FIND_ODDGEN_FACE_ ;
    static const char* FIND_FACEMASK_FACE_ ;
    static const char* FIND_NONBOUNDARY_FACE_ ;
    static const char* FIND_BOUNDARY_FACE_ ;
    static const char* FIND_REGULAR_FACE_ ;
    static const char* FIND_INTERIOR_FACE_ ; 
    static const char* FIND_CORNER_FACE_ ; 
    static const char* FIND_SIDE_FACE_ ; 
    static const char* FIND_SIDECORNER_FACE_ ; 
    static const char* FindType( NOpenMeshFindType find );


    static const char* ORDER_DEFAULT_FACE_ ; 
    static const char* ORDER_REVERSE_FACE_ ; 
    static const char* OrderType( NOpenMeshOrderType order );

    static const char* PROP_OUTSIDE_OTHER_ ; 
    static const char* PROP_INSIDE_OTHER_ ; 
    static const char* PROP_FRONTIER_ ; 
    static const char* PropType( NOpenMeshPropType prop );

    static const char* COMP_COMBINED_ ; 
    static const char* COMP_LEFT_ ; 
    static const char* COMP_RIGHT_ ; 
    static const char* CompType( NOpenMeshCompType comp );


};


