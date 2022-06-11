#pragma once

#include <string>
#include "plog/Severity.h"

struct NP ; 

class G4Material ; 
class G4MaterialPropertiesTable ;
#include "G4MaterialPropertyVector.hh"
#include "U4_API_EXPORT.hh"

struct U4_API U4Material
{
    static const plog::Severity LEVEL ; 
    static G4Material* Get(const char* name);
    static G4Material* Get_(const char* name);
    static G4Material* Vacuum(const char* name);



    static G4Material* MakeMaterial(const char* name, const char* reldir, const char* props ); 
    static G4Material* MakeMaterial(const char* name, const char* reldir); 


    static G4MaterialPropertiesTable* MakeMaterialPropertiesTable(const char* reldir); 
    static G4MaterialPropertiesTable* MakeMaterialPropertiesTable(const char* reldir, const char* keys, char delim ); 

    static G4MaterialPropertyVector* MakeProperty(const NP* a); 
    static char Classify(const NP* a); 
    static std::string Desc(const char* key, const NP* a ); 



    static G4MaterialPropertiesTable*  MakeMaterialPropertiesTable_FromProp(
         const char* a_key=nullptr, const G4MaterialPropertyVector* a_prop=nullptr,
         const char* b_key=nullptr, const G4MaterialPropertyVector* b_prop=nullptr,
         const char* c_key=nullptr, const G4MaterialPropertyVector* c_prop=nullptr,
         const char* d_key=nullptr, const G4MaterialPropertyVector* d_prop=nullptr
     ); 


    static G4Material* MakeWater(const char* name="Water"); 
    static G4Material* MakeMaterial(const G4MaterialPropertyVector* rindex, const char* name="Water") ;

    static G4Material* MakeScintillatorOld(); 
    static G4Material* MakeScintillator(); 
    static G4MaterialPropertyVector* GetProperty(const G4Material* mat, const char* name); 






}; 
