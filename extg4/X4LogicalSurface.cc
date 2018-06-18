#include <cassert>

#include "G4OpticalSurface.hh"
#include "G4LogicalSurface.hh"
#include "G4MaterialPropertiesTable.hh"

#include "X4LogicalSurface.hh"

#include "X4MaterialPropertiesTable.hh"
#include "GPropertyMap.hh"

void X4LogicalSurface::Convert(GPropertyMap<float>* dst,  const G4LogicalSurface* src)
{
    const G4SurfaceProperty*  psurf = src->GetSurfaceProperty() ;   
    const G4OpticalSurface* opsurf = dynamic_cast<const G4OpticalSurface*>(psurf);
    assert( opsurf );   
    G4MaterialPropertiesTable* mpt = opsurf->GetMaterialPropertiesTable() ;
    X4MaterialPropertiesTable::Convert( dst, mpt );
}



