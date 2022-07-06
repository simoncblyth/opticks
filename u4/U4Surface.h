#pragma once

class G4LogicalBorderSurface ;
class G4VPhysicalVolume ; 
class G4LogicalSurface ; 

struct U4Surface
{
    static G4LogicalBorderSurface* MakePerfectAbsorberSurface(const char* name, G4VPhysicalVolume* pv1, G4VPhysicalVolume* pv2 ); 
    static G4LogicalSurface* GetLogicalSurface(const G4VPhysicalVolume* thePrePV, const G4VPhysicalVolume* thePostPV); 
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

/**
U4Surface::GetLogicalSurface
-----------------------------

::

      +------------------------------------------+
      | thePrePV                                 |
      |                                          |
      |                                          |
      |       +------------------------+         |
      |       | thePostPV              |         |
      |       |                        |         |
      |   +--->                        |         |
      |       |                        |         |
      |       |                        |         |
      |       +------------------------+         |
      |                                          |
      |                                          |
      |                                          |
      +------------------------------------------+


      enteredDaughter = thePostPV->GetMotherLogical() == thePrePV ->GetLogicalVolume() 

      enteredDaughter:True 
          "inwards" photons


      +------------------------------------------+
      | thePostPV                                |
      |                                          |
      |                                          |
      |       +------------------------+         |
      |       | thePrePV               |         |
      |       |                        |         |
      |       <---+                    |         |
      |       |                        |         |
      |       |                        |         |
      |       +------------------------+         |
      |                                          |
      |                                          |
      |                                          |
      +------------------------------------------+


      enteredDaughter = thePostPV->GetMotherLogical() == thePrePV ->GetLogicalVolume() 

      enteredDaughter:False
          "outwards" photons

**/


inline G4LogicalSurface* U4Surface::GetLogicalSurface(const G4VPhysicalVolume* thePrePV, const G4VPhysicalVolume* thePostPV)
{
    G4LogicalSurface* Surface = G4LogicalBorderSurface::GetSurface(thePrePV, thePostPV);
    if (Surface == nullptr)
    {
        G4bool enteredDaughter = thePostPV->GetMotherLogical() == thePrePV ->GetLogicalVolume();

        if(enteredDaughter)
        {
            Surface = G4LogicalSkinSurface::GetSurface(thePostPV->GetLogicalVolume());
            if(Surface == nullptr) Surface = G4LogicalSkinSurface::GetSurface(thePrePV->GetLogicalVolume());
        }    
        else  // "leavingDaughter"
        {
            Surface = G4LogicalSkinSurface::GetSurface(thePrePV->GetLogicalVolume());
            if(Surface == NULL) Surface = G4LogicalSkinSurface::GetSurface(thePostPV->GetLogicalVolume());
        }
    }
    return Surface ; 
}


