#pragma once

#include "G4Material.hh"

#include "NPFold.h"
#include "sstr.h"
#include "S4MaterialPropertyVector.h"

struct S4Material
{
   static NPFold* MakePropertyFold( const char* mats, const char* props, char delim=',' ); 
}; 

/**
S4Material::MakePropertyFold
----------------------------

This is for example used from LSExpDetectorConstruction_Opticks::SerializePMT with::

    NPFold* pmt_rindex = S4Material::MakePropertyFold("Pyrex,Vacuum","RINDEX") ; 

**/

inline NPFold* S4Material::MakePropertyFold( const char* _mats, const char* _props, char delim )
{
    std::vector<std::string> mats ; 
    sstr::Split(_mats, delim, mats ); 

    std::vector<std::string> props ; 
    sstr::Split(_props, delim, props ); 

    int num_mats = mats.size(); 
    int num_props = props.size(); 

    NPFold* fold = new NPFold ; 

    for(int i=0 ; i < num_mats ; i++)
    {
        const char* _mat = mats[i].c_str(); 
        G4Material* mat = G4Material::GetMaterial(_mat) ; 
        G4MaterialPropertiesTable* mpt = mat ? mat->GetMaterialPropertiesTable() : nullptr ; 
        if(mpt == nullptr) continue ; 

        for(int j=0 ; j < num_props ; j++)
        {
            const char* _prop = props[j].c_str() ; 
            G4MaterialPropertyVector* mpv = mpt->GetProperty(_prop) ; 
            NP* arr = S4MaterialPropertyVector::ConvertToArray(mpv );          

            std::stringstream ss ; 
            ss << _mat << _prop ; 
            std::string key = ss.str(); 
            fold->add( key.c_str(), arr ); 
        }
    }
    return fold ; 
}

