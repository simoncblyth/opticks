#pragma once 
#include <cstdio>
#include <cassert>
#include <cstring>

enum { 
   UNKNOWN,
   RNG_SEQUENCE,
   WAVELENGTH_S,
   WAVELENGTH_C,
   SCINT_PHOTON_P,
   CERENKOV_PHOTON_K,
   CERENKOV_PHOTON_ENPROP_E,
   CERENKOV_PHOTON_EXPT_X,
   GENERATE_PHOTON_G,
   BOUNDARY_LOOKUP_ALL_A,
   BOUNDARY_LOOKUP_LINE_WATER_W,
   WATER,  
   BOUNDARY_LOOKUP_LINE_LS_L,
   PROP_LOOKUP_Y,
   FILL_STATE_0,
   FILL_STATE_1,
   RAYLEIGH_SCATTER_ALIGN,
   PROPAGATE_TO_BOUNDARY,
   PROPAGATE_AT_BOUNDARY,
   HEMISPHERE_S_POLARIZED,
   HEMISPHERE_P_POLARIZED,
   PROPAGATE_AT_SURFACE,
   PROPAGATE_AT_BOUNDARY_S_POLARIZED,
   PROPAGATE_AT_BOUNDARY_P_POLARIZED
};
 
struct QSimLaunch
{
    static unsigned    Type(const char* name) ;  
    static bool        IsMutate(unsigned type) ;  
    static unsigned    MutateSource(unsigned type); 
    static const char* Name(unsigned type ); 

    static const char* RNG_SEQUENCE_ ; 
    static const char* WATER_ ; 
    static const char* FILL_STATE_0_ ;
    static const char* FILL_STATE_1_ ;
    static const char* RAYLEIGH_SCATTER_ALIGN_ ;
    static const char* PROPAGATE_TO_BOUNDARY_ ; 
    static const char* PROPAGATE_AT_BOUNDARY_ ; 
    static const char* HEMISPHERE_S_POLARIZED_ ; 
    static const char* HEMISPHERE_P_POLARIZED_ ; 
    static const char* PROPAGATE_AT_SURFACE_ ; 
    static const char* PROPAGATE_AT_BOUNDARY_S_POLARIZED_ ; 
    static const char* PROPAGATE_AT_BOUNDARY_P_POLARIZED_ ; 
};

const char* QSimLaunch::RNG_SEQUENCE_ = "rng_sequence" ; 
const char* QSimLaunch::WATER_ = "water" ; 
const char* QSimLaunch::FILL_STATE_0_ = "fill_state_0" ; 
const char* QSimLaunch::FILL_STATE_1_ = "fill_state_1" ; 
const char* QSimLaunch::RAYLEIGH_SCATTER_ALIGN_ = "rayleigh_scatter_align" ; 
const char* QSimLaunch::PROPAGATE_TO_BOUNDARY_ = "propagate_to_boundary" ; 
const char* QSimLaunch::PROPAGATE_AT_BOUNDARY_ = "propagate_at_boundary" ; 
const char* QSimLaunch::HEMISPHERE_S_POLARIZED_ = "hemisphere_s_polarized" ; 
const char* QSimLaunch::HEMISPHERE_P_POLARIZED_ = "hemisphere_p_polarized" ; 
const char* QSimLaunch::PROPAGATE_AT_SURFACE_ = "propagate_at_surface" ; 
const char* QSimLaunch::PROPAGATE_AT_BOUNDARY_S_POLARIZED_ = "propagate_at_boundary_s_polarized" ; 
const char* QSimLaunch::PROPAGATE_AT_BOUNDARY_P_POLARIZED_ = "propagate_at_boundary_p_polarized" ; 


inline unsigned QSimLaunch::Type( const char* name )
{
   unsigned test = UNKNOWN ;  
   if(strcmp(name,"S") == 0 ) test = WAVELENGTH_S ; 
   if(strcmp(name,"C") == 0 ) test = WAVELENGTH_C ;
   if(strcmp(name,"P") == 0 ) test = SCINT_PHOTON_P ;
   if(strcmp(name,"K") == 0 ) test = CERENKOV_PHOTON_K ;
   if(strcmp(name,"E") == 0 ) test = CERENKOV_PHOTON_ENPROP_E ;
   if(strcmp(name,"X") == 0 ) test = CERENKOV_PHOTON_EXPT_X ;
   if(strcmp(name,"G") == 0 ) test = GENERATE_PHOTON_G ;
   if(strcmp(name,"A") == 0 ) test = BOUNDARY_LOOKUP_ALL_A ;
   if(strcmp(name,"L") == 0 ) test = BOUNDARY_LOOKUP_LINE_LS_L ;
   if(strcmp(name,"Y") == 0 ) test = PROP_LOOKUP_Y ;

   if(strcmp(name,RNG_SEQUENCE_) == 0 )          test = RNG_SEQUENCE ; 
   if(strcmp(name,WATER_) == 0    )              test = BOUNDARY_LOOKUP_LINE_WATER_W ;
   if(strcmp(name,FILL_STATE_0_) == 0)           test = FILL_STATE_0 ;
   if(strcmp(name,FILL_STATE_1_) == 0)           test = FILL_STATE_1 ;
   if(strcmp(name,RAYLEIGH_SCATTER_ALIGN_) == 0) test = RAYLEIGH_SCATTER_ALIGN ;
   if(strcmp(name,PROPAGATE_TO_BOUNDARY_) == 0)  test = PROPAGATE_TO_BOUNDARY ;
   if(strcmp(name,PROPAGATE_AT_BOUNDARY_) == 0)  test = PROPAGATE_AT_BOUNDARY ;
   if(strcmp(name,HEMISPHERE_S_POLARIZED_) == 0) test = HEMISPHERE_S_POLARIZED ;
   if(strcmp(name,HEMISPHERE_P_POLARIZED_) == 0) test = HEMISPHERE_P_POLARIZED ;
   if(strcmp(name,PROPAGATE_AT_SURFACE_)  == 0)  test = PROPAGATE_AT_SURFACE ;

   if(strcmp(name,PROPAGATE_AT_BOUNDARY_S_POLARIZED_) == 0)  test = PROPAGATE_AT_BOUNDARY_S_POLARIZED ;
   if(strcmp(name,PROPAGATE_AT_BOUNDARY_P_POLARIZED_) == 0)  test = PROPAGATE_AT_BOUNDARY_P_POLARIZED ;
   
   bool known =  test != UNKNOWN  ;
   if(!known) printf("QSimLaunch::Type name [%s] is unknown \n", name) ; 
   assert(known);  
   return test ; 
}

inline bool QSimLaunch::IsMutate( unsigned type )
{
    return type == PROPAGATE_AT_BOUNDARY_S_POLARIZED || PROPAGATE_AT_BOUNDARY_P_POLARIZED    ; 
}
inline unsigned QSimLaunch::MutateSource( unsigned type )
{
    unsigned src = UNKNOWN ; 
    switch(type)
    {
       case PROPAGATE_AT_BOUNDARY_S_POLARIZED:  src = HEMISPHERE_S_POLARIZED ; break ; 
       case PROPAGATE_AT_BOUNDARY_P_POLARIZED:  src = HEMISPHERE_P_POLARIZED ; break ; 
    } 
    return src ; 
}


inline const char* QSimLaunch::Name( unsigned type )
{
    const char* s = nullptr ; 
    switch(type)
    {
        case RNG_SEQUENCE:           s = RNG_SEQUENCE_           ; break ; 
        case WATER:                  s = WATER_                  ; break ; 
        case FILL_STATE_0:           s = FILL_STATE_0_           ; break ; 
        case FILL_STATE_1:           s = FILL_STATE_1_           ; break ; 
        case RAYLEIGH_SCATTER_ALIGN: s = RAYLEIGH_SCATTER_ALIGN_ ; break ;
        case PROPAGATE_TO_BOUNDARY:  s = PROPAGATE_TO_BOUNDARY_  ; break ;  
        case PROPAGATE_AT_BOUNDARY:  s = PROPAGATE_AT_BOUNDARY_  ; break ;  
        case HEMISPHERE_S_POLARIZED: s = HEMISPHERE_S_POLARIZED_ ; break ; 
        case HEMISPHERE_P_POLARIZED: s = HEMISPHERE_P_POLARIZED_ ; break ; 
        case PROPAGATE_AT_SURFACE:   s = PROPAGATE_AT_SURFACE_   ; break ; 
        case PROPAGATE_AT_BOUNDARY_S_POLARIZED:  s = PROPAGATE_AT_BOUNDARY_S_POLARIZED_  ; break ;  
        case PROPAGATE_AT_BOUNDARY_P_POLARIZED:  s = PROPAGATE_AT_BOUNDARY_P_POLARIZED_  ; break ;  
    }
    return s; 
}

