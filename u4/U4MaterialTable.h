#pragma once

#include <string>
#include <vector>
#include "G4MaterialTable.hh"

struct U4MaterialTable
{
    static void GetMaterialNames(std::vector<std::string>& names );  
}; 

/**
U4MaterialTable::GetMaterialNames
----------------------------------

see G4OpRayleigh::BuildPhysicsTable

**/

inline void U4MaterialTable::GetMaterialNames(std::vector<std::string>& names )
{
    const G4MaterialTable* theMaterialTable = G4Material::GetMaterialTable();
    const G4int numOfMaterials = G4Material::GetNumberOfMaterials();
    for( G4int iMaterial = 0; iMaterial < numOfMaterials; iMaterial++ )
    {
        G4Material* material = (*theMaterialTable)[iMaterial];
        const G4String& _name = material->GetName() ; 
        const char* name = _name.c_str() ; 
        names.push_back(name) ; 
    }
}

