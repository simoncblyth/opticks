#include <cassert>

#include "G4OpticalSurface.hh"   
#include "G4LogicalSkinSurface.hh"   

#include "X4.hh"
#include "X4LogicalSurface.hh"
#include "X4LogicalSkinSurface.hh"
#include "X4OpticalSurface.hh"

#include "GOpticalSurface.hh"   
#include "GSkinSurface.hh"   

#include "PLOG.hh"


GSkinSurface* X4LogicalSkinSurface::Convert(const G4LogicalSkinSurface* src)
{
    const char* name = X4::Name( src ); 
    size_t index = X4::GetOpticksIndex( src ) ;  

    G4OpticalSurface* os = dynamic_cast<G4OpticalSurface*>(src->GetSurfaceProperty());
    assert( os );
    GOpticalSurface* optical_surface = X4OpticalSurface::Convert(os);   ; 
    assert( optical_surface );

    GSkinSurface* dst = new GSkinSurface( name, index, optical_surface) ;  
    // standard domain is set by GSkinSurface::init

    X4LogicalSurface::Convert( dst, src);

    const G4LogicalVolume* lv = src->GetLogicalVolume();

   
    /*
    LOG(fatal) 
         << " X4::Name(lv)  " << X4::Name(lv)
         << " X4::BaseNameAsis(lv) " << X4::BaseNameAsis(lv)
         ; 
    */

    dst->setSkinSurface(  X4::BaseNameAsis(lv) ) ; 


    return dst ; 
}

int X4LogicalSkinSurface::GetItemIndex( const G4LogicalSkinSurface* src )
{
    const G4LogicalSkinSurfaceTable* vec = G4LogicalSkinSurface::GetSurfaceTable() ; 
    return X4::GetItemIndex<G4LogicalSkinSurface>( vec, src ); 
}



