#include <cstddef>
#include "NOpenMeshEnum.hpp"

/*
  .,+10s/\s*\(FIND\w*\).*$/static const char* \1_ = "\1" ;/gc
*/

const char* NOpenMeshEnum::FIND_ALL_FACE_ = "FIND_ALL_FACE" ;
const char* NOpenMeshEnum::FIND_IDENTITY_FACE_ = "FIND_IDENTITY_FACE" ;
const char* NOpenMeshEnum::FIND_FACEMASK_FACE_ = "FIND_FACEMASK_FACE" ;
const char* NOpenMeshEnum::FIND_NONBOUNDARY_FACE_ = "FIND_NONBOUNDARY_FACE" ;
const char* NOpenMeshEnum::FIND_BOUNDARY_FACE_ = "FIND_BOUNDARY_FACE" ;
const char* NOpenMeshEnum::FIND_REGULAR_FACE_ = "FIND_REGULAR_FACE" ;
const char* NOpenMeshEnum::FIND_INTERIOR_FACE_ = "FIND_INTERIOR_FACE" ;

const char* NOpenMeshEnum::FindType( NOpenMeshFindType select )
{
    const char* s = NULL ; 
    switch(select)
    {
        case FIND_ALL_FACE         : s="FIND_ALL_FACE"         ;break; 
        case FIND_IDENTITY_FACE    : s="FIND_IDENTITY_FACE"    ;break; 
        case FIND_FACEMASK_FACE    : s="FIND_FACEMASK_FACE"    ;break; 
        case FIND_NONBOUNDARY_FACE : s="FIND_NONBOUNDARY_FACE" ;break; 
        case FIND_BOUNDARY_FACE    : s="FIND_BOUNDARY_FACE"    ;break; 
        case FIND_REGULAR_FACE     : s="FIND_REGULAR_FACE"     ;break; 
        case FIND_INTERIOR_FACE    : s="FIND_INTERIOR_FACE"    ;break; 
    }
    return s ; 
}







