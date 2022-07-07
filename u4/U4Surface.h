#pragma once

class G4LogicalBorderSurface ;
class G4VPhysicalVolume ; 
class G4LogicalSurface ; 

struct U4Surface
{
    static G4LogicalBorderSurface* MakePerfectAbsorberSurface(const char* name, G4VPhysicalVolume* pv1, G4VPhysicalVolume* pv2 ); 
};


#include "G4String.hh"
#include "G4OpticalSurface.hh"
#include "G4MaterialPropertiesTable.hh"
#include "G4LogicalBorderSurface.hh"
#include "G4LogicalSkinSurface.hh"

#include "U4Material.hh"

/**
U4Surface::MakePerfectAborberSurface
--------------------------------------

From InstrumentedG4OpBoundaryProcess I think it needs a RINDEX property even though that is not 
going to be used for anything.  Also it needs REFLECTIVITY of zero. 

**/

inline G4LogicalBorderSurface* U4Surface::MakePerfectAbsorberSurface(const char* name_, G4VPhysicalVolume* pv1, G4VPhysicalVolume* pv2)
{
    G4String name = name_ ; 
    G4OpticalSurfaceModel model = glisur ; 
    G4OpticalSurfaceFinish finish = polished ; 
    G4SurfaceType type = dielectric_dielectric ; 
    G4double value = 1.0 ; 
    G4OpticalSurface* os = new G4OpticalSurface(name, model, finish, type, value );  
    G4MaterialPropertiesTable* mpt = new G4MaterialPropertiesTable ; 
    G4MaterialPropertyVector* rindex = U4Material::MakeProperty(1.);  
    G4MaterialPropertyVector* reflectivity = U4Material::MakeProperty(0.);  
    mpt->AddProperty("RINDEX", rindex );  
    mpt->AddProperty("REFLECTIVITY",reflectivity );  
    os->SetMaterialPropertiesTable(mpt);  
    G4LogicalBorderSurface* bs = new G4LogicalBorderSurface(name, pv1, pv2, os ); 
    return bs ; 
}


