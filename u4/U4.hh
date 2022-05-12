#pragma once

struct NP ; 

#include "G4MaterialPropertyVector.hh"
class G4MaterialPropertiesTable ; 
class G4Material ; 

#include "U4_API_EXPORT.hh"

struct U4_API U4
{
    static G4MaterialPropertyVector* MakeProperty(const NP* a); 
    static G4MaterialPropertiesTable*  MakeMaterialPropertiesTable(
         const char* a_key=nullptr, const G4MaterialPropertyVector* a_prop=nullptr,
         const char* b_key=nullptr, const G4MaterialPropertyVector* b_prop=nullptr,
         const char* c_key=nullptr, const G4MaterialPropertyVector* c_prop=nullptr,
         const char* d_key=nullptr, const G4MaterialPropertyVector* d_prop=nullptr
     ); 

    static G4Material* MakeMaterial(const G4MaterialPropertyVector* rindex, const char* name="Water") ;


};




