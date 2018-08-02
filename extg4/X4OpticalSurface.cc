#include <cassert>
#include "BStr.hh"

#include "G4OpticalSurface.hh"
#include "X4.hh"
#include "X4OpticalSurface.hh"
#include "GOpticalSurface.hh"

#include "PLOG.hh"


const char* X4OpticalSurface::Type(G4SurfaceType type)
{
   const char* t = NULL ; 
   switch(type)
   {
       case dielectric_metal      : t="dielectric_metal"      ; break ; 
       case dielectric_dielectric : t="dielectric_dielectric" ; break ; 
       case dielectric_LUT        : t="dielectric_LUT"        ; break ; 
       case dielectric_dichroic   : t="dielectric_dichroic"   ; break ; 
       case firsov                : t="firsov"                ; break ;
       case x_ray                 : t="x_ray"                 ; break ;
   }
   return t ; 
}    


GOpticalSurface* X4OpticalSurface::Convert( const G4OpticalSurface* const surf )
{
   
    const char* name = X4::Name<G4OpticalSurface>(surf);

    G4SurfaceType type = surf->GetType() ; 

    LOG(info) 
          << " name " << name
          << " type " << Type(type)
          ; 


    switch( type )
    {
        case dielectric_metal       :            ; break ;  // dielectric-metal interface
        case dielectric_dielectric  :            ; break ;  // dielectric-dielectric interface
        case dielectric_LUT         :  assert(0) ; break ;  // dielectric-Look-Up-Table interface
        case dielectric_dichroic    :  assert(0) ; break ;  // dichroic filter interface
        case firsov                 :  assert(0) ; break ;  // for Firsov Process
        case x_ray                  :  assert(0) ; break ; 
        default                     :  assert(0) ; break ; 
    }

    G4OpticalSurfaceModel model = surf->GetModel(); 
    switch( model )
    {
        case glisur             : assert(0) ; break ;   // original GEANT3 model
        case unified            :             break ;   // UNIFIED model
        case LUT                : assert(0) ; break ;   // Look-Up-Table model
        case dichroic           : assert(0) ; break ; 
        default                 : assert(0) ; break ; 
    }           


    G4OpticalSurfaceFinish finish = surf->GetFinish(); 

    bool specular = false ;    // HUH: not used, TODO:check cfg4 
    switch(finish)
    {
        case polished              :  specular = true ; break ; // smooth perfectly polished surface
        case polishedfrontpainted  :  specular = true ; break ; // smooth top-layer (front) paint
        case polishedbackpainted   :  assert(0)       ; break ; // same is 'polished' but with a back-paint

        case ground                :  specular = false ; break ; // rough surface
        case groundfrontpainted    :  specular = false ; break ; // rough top-layer (front) paint
        case groundbackpainted     :  assert(0)        ; break ; // same as 'ground' but with a back-paint

        default                    :  assert( 0 && "unexpected finish " ) ;  break ;   
    }

    G4double value = (model==glisur) ? surf->GetPolish() : surf->GetSigmaAlpha();


    const char* osnam = name ; 
    const char* ostyp = BStr::itoa(type);  
    const char* osmod = BStr::itoa(model);  
    const char* osfin = BStr::itoa(finish);  
    int percent = int(value*100.0) ;  
    const char* osval = BStr::itoa(percent);  

    GOpticalSurface* os = osnam && ostyp && osmod && osfin && osval ? new GOpticalSurface(osnam, ostyp, osmod, osfin, osval) : NULL ;
    assert( os ); 
    return os ; 
}





