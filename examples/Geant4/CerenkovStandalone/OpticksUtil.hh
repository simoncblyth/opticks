#pragma once

#include <vector>
#include <string>

struct NP ; 
#include "G4MaterialPropertyVector.hh"
class G4Material ; 

struct OpticksUtil
{
    static NP* LoadArray(const char* kdpath);
    static NP* LoadRandom(const char* random_path);

    static G4MaterialPropertyVector* MakeProperty(const NP* a);
    static G4Material* MakeMaterial(const G4MaterialPropertyVector* rindex, const char* name="Water") ; 

    static int getenvint(const char* envkey, int fallback);
    static bool ExistsPath(const char* base_, const char* reldir_=nullptr, const char* name_=nullptr );
    static std::string prepare_path(const char* dir_, const char* reldir_, const char* name );
    static void ListDir(std::vector<std::string>& names,  const char* path, const char* ext); 

}; 
