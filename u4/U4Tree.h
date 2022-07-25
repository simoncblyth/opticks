#pragma once

class G4VPhysicalVolume ; 


struct U4Tree
{
    static void Traverse(const G4VPhysicalVolume* const top ); 
    static void Traverse_r(const G4VPhysicalVolume* const pv, int depth ); 
    static void Visit( const G4VPhysicalVolume* const pv ); 
}; 

#include "G4VPhysicalVolume.hh"

void U4Tree::Traverse(const G4VPhysicalVolume* const top )
{
    Traverse_r(top, 0); 
}

void U4Tree::Traverse_r(const G4VPhysicalVolume* const pv, int depth )
{
    const G4LogicalVolume* const lv = pv->GetLogicalVolume();
    Visit(pv); 
    for (size_t i=0 ; i < size_t(lv->GetNoDaughters()) ;i++ )
    {    
        const G4VPhysicalVolume* const child_pv = lv->GetDaughter(i);
        Traverse_r(child_pv, depth+1);
    }        
}

void U4Tree::Visit( const G4VPhysicalVolume* const pv )
{
    const G4PVPlacement* placement = dynamic_cast<const G4PVPlacement*>(pv);
    assert(placement);
    G4int copyNumber = placement->GetCopyNo() ;
}



