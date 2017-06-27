#include <sstream>
#include "NBBoxEnum.hpp"

const char* nbboxenum::UNCLASSIFIED_ = "UNCLASSIFIED" ; 

const char* nbboxenum::XMIN_INSIDE_ = "XMIN_IN" ; 
const char* nbboxenum::YMIN_INSIDE_ = "YMIN_IN" ; 
const char* nbboxenum::ZMIN_INSIDE_ = "ZMIN_IN" ; 

const char* nbboxenum::XMAX_INSIDE_ = "XMAX_IN" ; 
const char* nbboxenum::YMAX_INSIDE_ = "YMAX_IN" ; 
const char* nbboxenum::ZMAX_INSIDE_ = "ZMAX_IN" ; 

const char* nbboxenum::XMIN_COINCIDENT_ = "XMIN_CO" ; 
const char* nbboxenum::YMIN_COINCIDENT_ = "YMIN_CO" ; 
const char* nbboxenum::ZMIN_COINCIDENT_ = "ZMIN_CO" ; 

const char* nbboxenum::XMAX_COINCIDENT_ = "XMAX_CO" ; 
const char* nbboxenum::YMAX_COINCIDENT_ = "YMAX_CO" ; 
const char* nbboxenum::ZMAX_COINCIDENT_ = "ZMAX_CO" ; 

const char* nbboxenum::XMIN_OUTSIDE_ = "XMIN_OUT" ; 
const char* nbboxenum::YMIN_OUTSIDE_ = "YMIN_OUT" ; 
const char* nbboxenum::ZMIN_OUTSIDE_ = "ZMIN_OUT" ; 

const char* nbboxenum::XMAX_OUTSIDE_ = "XMAX_OUT" ; 
const char* nbboxenum::YMAX_OUTSIDE_ = "YMAX_OUT" ; 
const char* nbboxenum::ZMAX_OUTSIDE_ = "ZMAX_OUT" ; 


std::string nbboxenum::ContainmentMaskString( unsigned mask )
{
    std::stringstream ss ; 
    for(unsigned i=0 ; i < 18 ; i++)
    { 
        NBBoxContainment_t cont = (NBBoxContainment_t)(0x1 << i) ;
        if(mask & cont) 
        {
            ss << ContainmentName(cont) << " " ;   
        }
    }
    return ss.str();
}

const char* nbboxenum::ContainmentName( NBBoxContainment_t cont )
{
    const char* s = NULL ; 
    switch(cont)
    {
        case UNCLASSIFIED: s = UNCLASSIFIED_ ; break ; 

        case XMIN_INSIDE: s = XMIN_INSIDE_ ; break ; 
        case YMIN_INSIDE: s = YMIN_INSIDE_ ; break ; 
        case ZMIN_INSIDE: s = ZMIN_INSIDE_ ; break ; 
        case XMAX_INSIDE: s = XMAX_INSIDE_ ; break ; 
        case YMAX_INSIDE: s = YMAX_INSIDE_ ; break ; 
        case ZMAX_INSIDE: s = ZMAX_INSIDE_ ; break ; 

        case XMIN_COINCIDENT: s = XMIN_COINCIDENT_ ; break ; 
        case YMIN_COINCIDENT: s = YMIN_COINCIDENT_ ; break ; 
        case ZMIN_COINCIDENT: s = ZMIN_COINCIDENT_ ; break ; 
        case XMAX_COINCIDENT: s = XMAX_COINCIDENT_ ; break ; 
        case YMAX_COINCIDENT: s = YMAX_COINCIDENT_ ; break ; 
        case ZMAX_COINCIDENT: s = ZMAX_COINCIDENT_ ; break ; 

        case XMIN_OUTSIDE: s = XMIN_OUTSIDE_ ; break ; 
        case YMIN_OUTSIDE: s = YMIN_OUTSIDE_ ; break ; 
        case ZMIN_OUTSIDE: s = ZMIN_OUTSIDE_ ; break ; 
        case XMAX_OUTSIDE: s = XMAX_OUTSIDE_ ; break ; 
        case YMAX_OUTSIDE: s = YMAX_OUTSIDE_ ; break ; 
        case ZMAX_OUTSIDE: s = ZMAX_OUTSIDE_ ; break ; 
    }
    return s ; 
}


