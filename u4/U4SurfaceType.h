#pragma once
/**
U4SurfaceType.h
=================

g4-cls G4SurfaceProperty

**/

struct U4SurfaceType
{
    static constexpr const char* dielectric_metal_ = "dielectric_metal"  ; 
    static constexpr const char* dielectric_dielectric_ = "dielectric_dielectric" ; 
    static constexpr const char* dielectric_LUT_ = "dielectric_LUT" ;            
    static constexpr const char* dielectric_LUTDAVIS_ = "dielectric_LUTDAVIS" ;      
    static constexpr const char* dielectric_dichroic_ = "dielectric_dichroic" ;     
    static constexpr const char* firsov_ = "firsov" ;                 
    static constexpr const char* x_ray_ = "x_ray" ;                  

    static const char* Name(unsigned type); 
};

inline const char* U4SurfaceType::Name(unsigned type)
{
    const char* s = nullptr ; 
    switch(type)
    {
        case dielectric_metal:      s = dielectric_metal_      ; break ; 
        case dielectric_dielectric: s = dielectric_dielectric_ ; break ; 
        case dielectric_LUT:        s = dielectric_LUT_        ; break ; 
        case dielectric_LUTDAVIS:   s = dielectric_LUTDAVIS_   ; break ; 
        case dielectric_dichroic:   s = dielectric_dichroic_   ; break ; 
        case firsov:                s = firsov_                ; break ;  
        case x_ray:                 s = x_ray_                 ; break ;  
    }
    return s ;   
}
