#include <cstddef>
#include <cstring>
#include "PLOG.hh"

#include <OpenMesh/Core/System/config.hh>
#include "NOpenMeshEnum.hpp"

/*
  .,+10s/\s*\(PROP\w*\).*$/const char* NOpenMeshEnum::\1_ = "\1" ;/gc
*/

const char* NOpenMeshEnum::PROP_OUTSIDE_OTHER_ = "PROP_OUTSIDE_OTHER" ;
const char* NOpenMeshEnum::PROP_INSIDE_OTHER_ = "PROP_INSIDE_OTHER" ;
const char* NOpenMeshEnum::PROP_FRONTIER_ = "PROP_FRONTIER" ;

const char* NOpenMeshEnum::PropType( NOpenMeshPropType prop )
{
    const char* s = NULL ; 
    switch(prop)
    {
        case PROP_OUTSIDE_OTHER : s=PROP_OUTSIDE_OTHER_  ;break; 
        case PROP_INSIDE_OTHER  : s=PROP_INSIDE_OTHER_   ;break; 
        case PROP_FRONTIER      : s=PROP_FRONTIER_       ;break; 
    }
    return s ;
}








const char* NOpenMeshEnum::COMBINE_HYBRID_ = "NOpenMeshEnum::COMBINE_HYBRID" ; 
const char* NOpenMeshEnum::COMBINE_CSGBSP_ = "NOpenMeshEnum::COMBINE_CSGBSP" ; 

const char* NOpenMeshEnum::CombineType(NOpenMeshCombineType meshmode)
{
    const char* s = NULL ; 
    switch( meshmode )
    {
        case COMBINE_HYBRID : s = COMBINE_HYBRID_  ; break ; 
        case COMBINE_CSGBSP : s = COMBINE_CSGBSP_  ; break ; 
    }
    return s ; 
}

NOpenMeshCombineType NOpenMeshEnum::CombineTypeFromPoly(const char* poly)
{
    if(strcmp(poly,"HY") == 0 )  return COMBINE_HYBRID ;
    if(strcmp(poly,"BSP") == 0 ) return COMBINE_CSGBSP ;

    LOG(fatal) << "NOpenMeshEnum::CombineTypeFromPoly"
               << " bad poly " << poly 
               ;

    //assert(0) ; 
    return COMBINE_HYBRID ;
}




//  .,+3s/\s*\(COMP\w*\).*$/const char* NOpenMeshEnum::\1_ = "\1" ;/g 

// .,+3s/\s*\(\S*\).*/case \1 : s="\1" ;break; /g


const char* NOpenMeshEnum::COMP_THIS_ = "COMP_THIS" ;
const char* NOpenMeshEnum::COMP_OTHER_ = "COMP_OTHER" ;
const char* NOpenMeshEnum::COMP_COMBINED_ = "COMP_COMBINED" ;
const char* NOpenMeshEnum::COMP_LEFT_ = "COMP_LEFT" ;
const char* NOpenMeshEnum::COMP_RIGHT_ = "COMP_RIGHT" ;



const char* NOpenMeshEnum::CompType( NOpenMeshCompType comp )
{
    const char* s = NULL ; 
    switch(comp)
    {
        case COMP_THIS     : s=COMP_THIS_     ;break; 
        case COMP_OTHER    : s=COMP_OTHER_    ;break; 
        case COMP_COMBINED : s=COMP_COMBINED_ ;break; 
        case COMP_LEFT     : s=COMP_LEFT_     ;break; 
        case COMP_RIGHT    : s=COMP_RIGHT_    ;break; 
    }
    return s ; 
}



unsigned NOpenMeshEnum::OpenMeshVersion()
{
    return OM_VERSION ;  
}




const char* NOpenMeshEnum::FIND_ALL_FACE_ = "FIND_ALL_FACE" ;
const char* NOpenMeshEnum::FIND_IDENTITY_FACE_ = "FIND_IDENTITY_FACE" ;
const char* NOpenMeshEnum::FIND_EVENGEN_FACE_ = "FIND_EVENGEN_FACE" ;
const char* NOpenMeshEnum::FIND_ODDGEN_FACE_ = "FIND_ODDGEN_FACE" ;
const char* NOpenMeshEnum::FIND_FACEMASK_FACE_ = "FIND_FACEMASK_FACE" ;
const char* NOpenMeshEnum::FIND_NONBOUNDARY_FACE_ = "FIND_NONBOUNDARY_FACE" ;
const char* NOpenMeshEnum::FIND_BOUNDARY_FACE_ = "FIND_BOUNDARY_FACE" ;
const char* NOpenMeshEnum::FIND_REGULAR_FACE_ = "FIND_REGULAR_FACE" ;
const char* NOpenMeshEnum::FIND_INTERIOR_FACE_ = "FIND_INTERIOR_FACE" ;
const char* NOpenMeshEnum::FIND_CORNER_FACE_ = "FIND_CORNER_FACE" ;
const char* NOpenMeshEnum::FIND_SIDE_FACE_ = "FIND_SIDE_FACE" ;
const char* NOpenMeshEnum::FIND_SIDECORNER_FACE_ = "FIND_SIDECORNER_FACE" ;

const char* NOpenMeshEnum::FindType( NOpenMeshFindType select )
{
    const char* s = NULL ; 
    switch(select)
    {
        case FIND_ALL_FACE         : s=FIND_ALL_FACE_         ;break; 
        case FIND_IDENTITY_FACE    : s=FIND_IDENTITY_FACE_    ;break; 
        case FIND_EVENGEN_FACE     : s=FIND_EVENGEN_FACE_     ;break; 
        case FIND_ODDGEN_FACE      : s=FIND_ODDGEN_FACE_      ;break; 
        case FIND_FACEMASK_FACE    : s=FIND_FACEMASK_FACE_    ;break; 
        case FIND_NONBOUNDARY_FACE : s=FIND_NONBOUNDARY_FACE_ ;break; 
        case FIND_BOUNDARY_FACE    : s=FIND_BOUNDARY_FACE_    ;break; 
        case FIND_REGULAR_FACE     : s=FIND_REGULAR_FACE_     ;break; 
        case FIND_INTERIOR_FACE    : s=FIND_INTERIOR_FACE_    ;break; 
        case FIND_CORNER_FACE      : s=FIND_CORNER_FACE_      ;break; 
        case FIND_SIDE_FACE        : s=FIND_SIDE_FACE_        ;break; 
        case FIND_SIDECORNER_FACE  : s=FIND_SIDECORNER_FACE_  ;break; 
    }
    return s ; 
}




const char* NOpenMeshEnum::ORDER_DEFAULT_FACE_ = "ORDER_DEFAULT_FACE" ;
const char* NOpenMeshEnum::ORDER_REVERSE_FACE_ = "ORDER_REVERSE_FACE" ;


const char* NOpenMeshEnum::OrderType( NOpenMeshOrderType order )
{
    const char* s = NULL ; 
    switch(order)
    {
        case ORDER_DEFAULT_FACE         : s=ORDER_DEFAULT_FACE_         ;break; 
        case ORDER_REVERSE_FACE         : s=ORDER_REVERSE_FACE_         ;break; 
    }
    return s ; 
}





