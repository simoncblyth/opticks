#pragma once

#include <vector>
#include <string>

struct NP ; 
#include "G4MaterialPropertyVector.hh"
class G4Material ; 
class G4MaterialPropertiesTable ; 

struct OpticksUtil
{
    static void qvals( std::vector<float>& vals, const char* key, const char* fallback, int num_expect ); 

    static NP* LoadArray(const char* kdpath);
    static NP* LoadConcat(const char* concat_path);  // formerly LoadRandom

    static G4MaterialPropertyVector* MakeProperty(const NP* a);
    static G4MaterialPropertiesTable*  MakeMaterialPropertiesTable( 
         const char* a_key=nullptr, const G4MaterialPropertyVector* a_prop=nullptr,
         const char* b_key=nullptr, const G4MaterialPropertyVector* b_prop=nullptr,
         const char* c_key=nullptr, const G4MaterialPropertyVector* c_prop=nullptr,
         const char* d_key=nullptr, const G4MaterialPropertyVector* d_prop=nullptr
     ); 
    static G4Material* MakeMaterial(const G4MaterialPropertyVector* rindex, const char* name="Water") ; 

    static int getenvint(const char* envkey, int fallback);
    static bool ExistsPath(const char* base_, const char* reldir_=nullptr, const char* name_=nullptr );
    static std::string prepare_path(const char* dir_, const char* reldir_, const char* name );
    static void ListDir(std::vector<std::string>& names,  const char* path, const char* ext); 

}; 
