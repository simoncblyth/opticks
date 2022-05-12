#pragma once

struct NP ; 

#include "G4MaterialPropertyVector.hh"
class G4MaterialPropertiesTable ; 
class G4Material ; 

#include "U4_API_EXPORT.hh"

struct U4_API U4
{
    static G4MaterialPropertyVector* MakeProperty(const NP* a); 

    static G4MaterialPropertiesTable*  MakeMaterialPropertiesTable_FromProp(
         const char* a_key=nullptr, const G4MaterialPropertyVector* a_prop=nullptr,
         const char* b_key=nullptr, const G4MaterialPropertyVector* b_prop=nullptr,
         const char* c_key=nullptr, const G4MaterialPropertyVector* c_prop=nullptr,
         const char* d_key=nullptr, const G4MaterialPropertyVector* d_prop=nullptr
     ); 

    static G4MaterialPropertiesTable* MakeMaterialPropertiesTable(const char* reldir, const char* keys, char delim ); 
    static G4MaterialPropertiesTable* MakeMaterialPropertiesTable(const char* reldir); 

    static G4Material* MakeWater(const char* name="Water"); 
    static G4Material* MakeMaterial(const G4MaterialPropertyVector* rindex, const char* name="Water") ;
    static G4Material* MakeMaterial(const char* name, const char* reldir, const char* props ); 
    static G4Material* MakeMaterial(const char* name, const char* reldir ); 
    static G4Material* MakeScintillator(); 

};




