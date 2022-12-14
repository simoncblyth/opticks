#pragma once

#include <string>
#include <sstream>

#include "G4Version.hh"
#include "G4MaterialPropertiesTable.hh"


struct U4MaterialPropertiesTable
{
    static std::string Desc(const G4MaterialPropertiesTable* mpt );  
    static std::string DescMaterialPropertyNames(const G4MaterialPropertiesTable* mpt); 
    static std::string DescPropertyMap(const G4MaterialPropertiesTable* mpt ); 
    static std::string DescConstPropertyMap(const G4MaterialPropertiesTable* mpt ); 
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

inline std::string U4MaterialPropertiesTable::Desc(const G4MaterialPropertiesTable* mpt )
{
    std::stringstream ss ; 
    ss << "U4MaterialPropertiesTable::Desc" << std::endl ; 
    //ss << DescMaterialPropertyNames(mpt) << std::endl ; 
    ss << DescPropertyMap(mpt) << std::endl ; 
    ss << DescConstPropertyMap(mpt) << std::endl ; 
    std::string s = ss.str(); 
    return s ; 
}



