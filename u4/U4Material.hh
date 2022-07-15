#pragma once

#include <vector>
#include <string>
#include "plog/Severity.h"

struct NP ; 
struct NPFold ; 

class G4Material ; 
class G4MaterialPropertiesTable ;
#include "G4MaterialPropertyVector.hh"
#include "U4_API_EXPORT.hh"

struct U4_API U4Material
{
    static const plog::Severity LEVEL ; 

    static std::string DescMaterialTable(); 
    static void GetMaterialNames( std::vector<std::string>& names, bool extras=false); 
    static const char* FindMaterialName(const char* volname); 

    static constexpr const char* SCINTILLATOR = "SCINTILLATOR" ; 
    static constexpr const char* VACUUM = "VACUUM" ; 

    static G4Material* Get(const char* name);
    static G4Material* Get_(const char* name);
    static G4Material* Vacuum(const char* name);
    static G4MaterialPropertyVector* GetProperty(const G4Material* mat, const char* name); 
    static G4MaterialPropertyVector* GetProperty(const char* mname, const char* pname); 

    static void GetMinMaxValue( double& mn , double& mx, const G4MaterialPropertyVector* prop ); 
    static std::string DescProperty(const G4MaterialPropertyVector* prop); 
    static std::string DescProperty( const char* mname, const char* pname);  
    static std::string DescProperty(const char* mname); 
    static std::string DescProperty(); 


    static void RemoveProperty( const char* key, G4Material* mat ); 

    static void GetPropertyNames( std::vector<std::string>& names, const G4Material* mat ) ; 
    static NPFold* GetPropertyFold(const G4Material* mat ); 
    static NPFold* GetPropertyFold(); 
    static std::string DescPropertyNames( const G4Material* mat ); 


    static G4MaterialPropertiesTable* MakeMaterialPropertiesTable(const char* reldir); 
    static G4MaterialPropertiesTable* MakeMaterialPropertiesTable(const char* reldir, const char* keys, char delim ); 

    static G4MaterialPropertyVector* MakeProperty( double value ); 
    static G4MaterialPropertyVector* MakeProperty(const NP* a); 
    static NP* MakePropertyArray( double value ); 


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
    static G4Material* MakeMaterial(const char* name, const char* reldir, const char* props ); 
    static G4Material* MakeMaterial(const char* name, const char* reldir); 

    static G4Material* MakeScintillator(); 


    static constexpr const char* LIBDIR = "$IDPath/GMaterialLib" ;  
    static G4Material* LoadOri(const char* name); 
    static void ListOri(std::vector<std::string>& names); 
    static void LoadOri(); 



    static int GetIndex(const std::vector<G4String>& nn, const char* key ); 
    static int GetPropertyIndex( const G4MaterialPropertiesTable* mpt, const char* key ); 
    static G4MaterialPropertyVector* AddProperty( G4MaterialPropertiesTable* mpt, const char* key, const std::vector<double>& dom, const std::vector<double>& val , bool reverse ); 

    static void LoadBnd(const char* ssimdir=nullptr); 
    static void KludgeRemoveRockRINDEX();  // static


}; 
