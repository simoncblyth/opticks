#pragma once

#include <string>
#include <sstream>
#include <cassert>

#include "G4Version.hh"
#include "G4MaterialPropertiesTable.hh"

#include "U4MaterialPropertyVector.h"
#include "NPFold.h"



struct U4MaterialPropertiesTable
{
    static std::string Detail(const G4MaterialPropertiesTable* mpt );  
    static std::string DescMaterialPropertyNames(const G4MaterialPropertiesTable* mpt); 
    static std::string DescPropertyMap(const G4MaterialPropertiesTable* mpt ); 
    static std::string DescConstPropertyMap(const G4MaterialPropertiesTable* mpt ); 

    static std::string Desc(const G4MaterialPropertiesTable* mpt );  
    static void GetProperties(std::vector<std::string>& keys, std::vector<G4MaterialPropertyVector*>& props, const G4MaterialPropertiesTable* mpt ); 
    static std::string DescProperties(const std::vector<std::string>& keys, const std::vector<G4MaterialPropertyVector*>& props ) ; 

    static NPFold* MakeFold( const G4MaterialPropertiesTable* mpt ); 
    static G4MaterialPropertiesTable* Create(std::vector<std::string>& keys, std::vector<double>& vals ); 
    static G4MaterialPropertiesTable* Create(const char* key, double val ); 
}; 

/**
U4MaterialPropertiesTable::DescMaterialPropertyNames
------------------------------------------------------

Returns all names, what are these G4 developers smoking ?

**/
inline std::string U4MaterialPropertiesTable::DescMaterialPropertyNames(const G4MaterialPropertiesTable* mpt )
{
    std::vector<G4String> names = mpt->GetMaterialPropertyNames(); 
    std::stringstream ss ; 
    ss <<  "DescMaterialPropertyNames" ; 
    ss << " names " << names.size() ; 
    for(unsigned i=0 ; i < names.size() ; i++) ss << names[i] << " "  ; 
    std::string s = ss.str(); 
    return s ; 
}

inline std::string U4MaterialPropertiesTable::DescPropertyMap(const G4MaterialPropertiesTable* mpt )
{
    std::stringstream ss ; 
    ss <<  "DescPropertyMap " ; 
#if G4VERSION_NUMBER < 1100
    typedef std::map<G4int, G4MaterialPropertyVector*> MIV ; 
    const MIV* miv = mpt->GetPropertyMap(); 
    ss << " miv.size " << miv->size() ; 

    std::vector<G4String> names = mpt->GetMaterialPropertyNames(); 

    std::vector<int> ii ; 
    std::vector<G4MaterialPropertyVector*> vv ; // HUH: this vector getting overwritten somehow ? 

    const int N = 50 ;  
    G4MaterialPropertyVector* qq[N] ;  
    for(int i=0 ; i < N ; i++) qq[i] = nullptr ; 

    ss << " v0 [ " ; 
    for(MIV::const_iterator iv=miv->begin() ; iv != miv->end() ; iv++) 
    {
        G4int i = iv->first ;  
        G4MaterialPropertyVector* v = iv->second ; 
        ii.push_back(i) ;
        vv.push_back(v) ;
        if( i < N ) qq[i] = iv->second ;  
        ss << v << " " ;  
    }
    ss << "]" ; 

    unsigned nii = ii.size(); 
    ss << " i [" ; 
    for(unsigned i=0 ; i < nii ; i++ ) ss << ii[i] << ( i < nii - 1 ? " " : "" ) ; 
    ss << "]" ; 

    ss << " n [" ; 
    for(unsigned i=0 ; i < nii ; i++ ) ss << names[ii[i]] << ( i < nii - 1 ? " " : "" ) ; 
    ss << "]" ; 

    ss << " v [" ; 
    for(unsigned i=0 ; i < nii ; i++ ) 
    {
        int idx = ii[i] ; 
        G4MaterialPropertyVector* q = idx < N ? qq[idx] : nullptr  ; 
        //G4MaterialPropertyVector* v = vv[idx] ; 
        ss <<  q  << ( i < nii - 1 ? " " : "" ) ;  
    }

    ss << " vl [" ; 
    for(unsigned i=0 ; i < nii ; i++ ) 
    {
        int idx = ii[i] ; 
        //G4MaterialPropertyVector* v = vv[idx] ; 
        G4MaterialPropertyVector* q = idx < N ? qq[idx] : nullptr  ; 
        size_t vl = q ? q->GetVectorLength() : 0 ; 
        ss <<  vl  << ( i < nii - 1 ? " " : "" ) ;  
    }
    ss << "]" ; 
#else
    ss << " NOT IMPLEMENTED YET FOR G4VERSION 1100+ " ; 
#endif
    std::string s = ss.str(); 
    return s ; 
}
inline std::string U4MaterialPropertiesTable::DescConstPropertyMap(const G4MaterialPropertiesTable* mpt )
{
    std::stringstream ss ; 
    ss <<  "DescConstPropertyMap " ; 
#if G4VERSION_NUMBER < 1100
    typedef std::map<G4int, G4double> MIF ; 
    const MIF* mif = mpt->GetConstPropertyMap() ; 
    ss << " mif.size " << mif->size() ; 
#else
    ss << " NOT IMPLEMENTED YET FOR G4VERSION 1100+ " ; 
#endif
    std::string s = ss.str(); 
    return s ; 
}    

inline std::string U4MaterialPropertiesTable::Detail(const G4MaterialPropertiesTable* mpt )
{
    std::stringstream ss ; 
    ss << "U4MaterialPropertiesTable::Desc" << std::endl ; 
    //ss << DescMaterialPropertyNames(mpt) << std::endl ; 
    ss << DescPropertyMap(mpt) << std::endl ; 
    ss << DescConstPropertyMap(mpt) << std::endl ; 
    std::string s = ss.str(); 
    return s ; 
}





inline std::string U4MaterialPropertiesTable::Desc(const G4MaterialPropertiesTable* mpt )
{
    std::vector<std::string> keys ; 
    std::vector<G4MaterialPropertyVector*> props ; 
    GetProperties(keys, props, mpt); 
    return DescProperties(keys, props) ;  
}


/**
U4MaterialPropertiesTable::GetProperties
------------------------------------------

This aims to provide an API that does not change with Geant4 version. 

Geant4 reference:

* https://geant4.kek.jp/lxr/source/materials/include/G4MaterialPropertiesTable.hh
* https://geant4.kek.jp/lxr/source/materials/src/G4MaterialPropertiesTable.cc

**/

inline void U4MaterialPropertiesTable::GetProperties(
      std::vector<std::string>& keys, 
      std::vector<G4MaterialPropertyVector*>& props, 
      const G4MaterialPropertiesTable* mpt )
{
    std::vector<G4String> names = mpt->GetMaterialPropertyNames(); 

#if G4VERSION_NUMBER < 1100
    typedef std::map<G4int, G4MaterialPropertyVector*> MIV ; 
    const MIV* miv = mpt->GetPropertyMap();   // <-- METHOD REMOVED IN G4 1100 

    for(MIV::const_iterator iv=miv->begin() ; iv != miv->end() ; iv++) 
    {
        G4int i = iv->first ;  
        G4MaterialPropertyVector* v = iv->second ; 
        const char* key = names[i].c_str();   

        keys.push_back(key); 
        props.push_back(v) ; 
    }
#else
    for(unsigned i = 0 ; i < names.size() ; i++)
    {
        const char* key = names[i].c_str() ; 
        G4MaterialPropertyVector* v = mpt->GetProperty(key) ; 
        if( v != nullptr ) 
        {
	  keys.push_back(key);
	  props.push_back(v);  
        }
    }
    // My reading of code suggests that the vector obtained from "props = mpt->GetProperties();"  
    // will usually have lots of nullptr and will have a different length to the names vector
    // unless absolutely all the properties are defined. 
    // That is different behaviour to < 1100  above, so the 
    // existance of properties is checked before adding to the vectors.
#endif
    assert( props.size() == keys.size() ); 

}

inline std::string U4MaterialPropertiesTable::DescProperties(
    const std::vector<std::string>& keys, 
    const std::vector<G4MaterialPropertyVector*>& props ) 
{
    assert( keys.size() == props.size() ); 
    unsigned num_prop = keys.size(); 

    std::stringstream ss ; 

    ss << "U4MaterialPropertiesTable::DescProperties" 
       << " num_prop " << num_prop
       << std::endl 
       ; 

    for(unsigned i=0 ; i < num_prop ; i++) 
    {
        const char* k = keys[i].c_str() ; 
        const G4MaterialPropertyVector* v = props[i] ; 
        ss
            << " i " << std::setw(2) << i 
            << " k " << std::setw(20) << ( k ? k : "-" )
            << " v " << std::setw(5) << ( v ? v->GetVectorLength() : -1 ) 
            << std::endl 
            ; 
    }
    std::string str = ss.str(); 
    return str ; 
}

inline NPFold* U4MaterialPropertiesTable::MakeFold( const G4MaterialPropertiesTable* mpt ) // static 
{
    std::vector<std::string> keys ; 
    std::vector<G4MaterialPropertyVector*> props ; 

    GetProperties(keys, props, mpt); 

    assert( keys.size() == props.size() ); 
    unsigned num_prop = props.size() ; 

    NPFold* fold = new NPFold ; 
    for(unsigned i=0 ; i < num_prop ; i++)
    {
        const char* k = keys[i].c_str() ;
        const G4MaterialPropertyVector* v = props[i] ;
        NP* a = U4MaterialPropertyVector::ConvertToArray( v ); 

        fold->add( k, a ); 
    }
    return fold ; 
}

inline G4MaterialPropertiesTable* U4MaterialPropertiesTable::Create( 
    std::vector<std::string>& keys, 
    std::vector<double>& vals )
{
    G4MaterialPropertiesTable* mpt = new G4MaterialPropertiesTable ; 
    assert( keys.size() == vals.size() ) ; 
    int num = int(keys.size());  
    for(int i=0 ; i < num ; i++)
    {
        const char* key = keys[i].c_str(); 
        double val = vals[i] ; 
        G4MaterialPropertyVector* opv = U4MaterialPropertyVector::Make_V(val) ;   
        mpt->AddProperty(key, opv ); 
    }
    return mpt ; 
}

inline G4MaterialPropertiesTable* U4MaterialPropertiesTable::Create(const char* key, double val )
{
    G4MaterialPropertiesTable* mpt = new G4MaterialPropertiesTable ; 
    G4MaterialPropertyVector* opv = U4MaterialPropertyVector::Make_V(val) ;   
    mpt->AddProperty(key, opv ); 
    return mpt ; 
}


