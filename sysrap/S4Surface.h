#pragma once
/**
S4Surface.h
============

Extract a few methods from U4Surface.h as needed for both old and new workflows. 

**/
#include <string>
#include <cstring>

class G4LogicalSurface ; 
class G4VPhysicalVolume ; 

struct S4Surface
{
    static G4LogicalSurface* Find( const G4VPhysicalVolume* thePrePV, const G4VPhysicalVolume* thePostPV ) ;  
};


/**
S4Surface::Find
-----------------

Note that U4Surface::Find does the same as this. But duplicated
here as need this whilst trying to get osut implicit boundaries
to work in old workflow.  

Looks for a border or skin surface in the same way 
as G4OpBoundaryProcess::PostStepDoIt which the code
is based on. 

**/

inline G4LogicalSurface* S4Surface::Find( const G4VPhysicalVolume* thePrePV, const G4VPhysicalVolume* thePostPV ) 
{
    if(thePostPV == nullptr || thePrePV == nullptr ) return nullptr ;  // surface on world volume not allowed 
    G4LogicalSurface* Surface = G4LogicalBorderSurface::GetSurface(thePrePV, thePostPV);
    if(Surface == nullptr)
    {
        G4bool enteredDaughter = thePostPV->GetMotherLogical() == thePrePV->GetLogicalVolume();
        if(enteredDaughter)
        {
            Surface = G4LogicalSkinSurface::GetSurface(thePostPV->GetLogicalVolume());
            if(Surface == nullptr)
                Surface = G4LogicalSkinSurface::GetSurface(thePrePV->GetLogicalVolume());
        }    
        else 
        {
            Surface = G4LogicalSkinSurface::GetSurface(thePrePV->GetLogicalVolume());
            if(Surface == nullptr)
                Surface = G4LogicalSkinSurface::GetSurface(thePostPV->GetLogicalVolume());
        }    
    }    
    return Surface ; 
}


