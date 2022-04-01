#pragma once

#include "X4_API_EXPORT.hh"
#include "G4Version.hh"
#include "G4SurfaceProperty.hh"

/**
X4SurfaceType
==================

The G4SurfaceType enum if defined in G4SurfaceProperty.hh::

   g4-cls G4SurfaceProperty

**/

struct X4_API X4SurfaceType
{
    static bool IsOpticksSupported(G4SurfaceType type); 
    static const char* Name(G4SurfaceType type);
    static G4SurfaceType Type(const char* name);

    static constexpr const char* dielectric_metal_      = "dielectric_metal" ; 
    static constexpr const char* dielectric_dielectric_ = "dielectric_dielectric" ; 
    static constexpr const char* dielectric_LUT_        = "dielectric_LUT" ; 
#if (G4VERSION_NUMBER >= 1042)
    static constexpr const char* dielectric_LUTDAVIS_   = "dielectric_LUTDAVIS" ; 
#endif
    static constexpr const char* dielectric_dichroic_   = "dielectric_dichroic" ; 
    static constexpr const char* firsov_                = "firsov" ; 
    static constexpr const char* x_ray_                 = "x_ray" ; 
};


inline bool X4SurfaceType::IsOpticksSupported(G4SurfaceType type) // static
{
    return type == dielectric_metal || type == dielectric_dielectric ; 
}

inline const char* X4SurfaceType::Name(G4SurfaceType type) // static
{
   const char* t = NULL ; 
   switch(type)
   {
       case dielectric_metal      : t=dielectric_metal_      ; break ; 
       case dielectric_dielectric : t=dielectric_dielectric_ ; break ; 
       case dielectric_LUT        : t=dielectric_LUT_        ; break ; 
#if (G4VERSION_NUMBER >= 1042)
       case dielectric_LUTDAVIS   : t=dielectric_LUTDAVIS_   ; break ; 
#endif
       case dielectric_dichroic   : t=dielectric_dichroic_   ; break ; 
       case firsov                : t=firsov_                ; break ;
       case x_ray                 : t=x_ray_                 ; break ;
   }
   return t ; 
}    

inline G4SurfaceType X4SurfaceType::Type(const char* name) // static
{
    G4SurfaceType type=dielectric_dielectric ; 
    if(strcmp(name,dielectric_metal_)==0 )       type=dielectric_metal ; 
    if(strcmp(name,dielectric_dielectric_)==0 )  type=dielectric_dielectric ; 
    if(strcmp(name,dielectric_LUT_)==0 )         type=dielectric_LUT ; 
#if (G4VERSION_NUMBER >= 1042)
    if(strcmp(name,dielectric_LUTDAVIS_)==0 )    type=dielectric_LUTDAVIS ; 
#endif
    if(strcmp(name,dielectric_dichroic_)==0 )    type=dielectric_dichroic ; 
    if(strcmp(name,firsov_)==0 )                 type=firsov ; 
    if(strcmp(name,x_ray_)==0 )                  type=x_ray ; 
    return type ; 
}

