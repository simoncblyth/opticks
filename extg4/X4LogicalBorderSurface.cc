#include <cassert>

#include "G4OpticalSurface.hh"   
#include "G4LogicalBorderSurface.hh"   

#include "X4.hh"
#include "X4LogicalSurface.hh"
#include "X4LogicalBorderSurface.hh"
#include "X4OpticalSurface.hh"

#include "GOpticalSurface.hh"   
#include "GBorderSurface.hh"   
#include "GDomain.hh"   

#include "PLOG.hh"


GBorderSurface* X4LogicalBorderSurface::Convert(const G4LogicalBorderSurface* src)
{
    const char* name = X4::Name( src ); 
    size_t index = X4::GetOpticksIndex( src ) ;  

    G4OpticalSurface* os = dynamic_cast<G4OpticalSurface*>(src->GetSurfaceProperty());
    assert( os );
    GOpticalSurface* optical_surface = X4OpticalSurface::Convert(os);   ; 
    assert( optical_surface );

    GBorderSurface* dst = new GBorderSurface( name, index, optical_surface) ;  
    // standard domain is set by GBorderSurface::init

    X4LogicalSurface::Convert( dst, src);

    const G4VPhysicalVolume* pv1 = src->GetVolume1(); 
    const G4VPhysicalVolume* pv2 = src->GetVolume2(); 
    assert( pv1 && pv2 ) ; 

    dst->setBorderSurface( X4::Name(pv1), X4::Name(pv2) );   


    return dst ; 
}

int X4LogicalBorderSurface::GetItemIndex( const G4LogicalBorderSurface* src )
{
    const G4LogicalBorderSurfaceTable* vec = G4LogicalBorderSurface::GetSurfaceTable() ; 
    return X4::GetItemIndex<G4LogicalBorderSurface>( vec, src ); 
}


