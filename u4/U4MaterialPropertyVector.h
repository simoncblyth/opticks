#pragma once
/**
U4MaterialPropertyVector.h
============================


After X4MaterialPropertyVector.hh

**/



#include "G4MaterialPropertyVector.hh"
#include "G4SystemOfUnits.hh"

#include "NP.hh"
#include "NPFold.h"

struct U4MaterialPropertyVector
{
    static NP* ConvertToArray(const G4MaterialPropertyVector* vec);
    static G4MaterialPropertyVector* FromArray(const NP* prop);
    static G4MaterialPropertyVector* Make_V(double value); 
    static std::string Desc_V(const G4MaterialPropertyVector* v); 

    static void    Import_MSV(          std::map<std::string,G4MaterialPropertyVector*>& msv, const NPFold* sub);
    static NPFold* Serialize_MSV( const std::map<std::string,G4MaterialPropertyVector*>& msv );
    static std::string Desc_MSV(  const std::map<std::string,G4MaterialPropertyVector*>& msv ) ;

    static void Import_MIMSV(            std::map<int,std::map<std::string,G4MaterialPropertyVector*>>& mimsv, const NPFold* f );
    static NPFold* Serialize_MIMSV(const std::map<int,std::map<std::string,G4MaterialPropertyVector*>>& mimsv ) ;
    static std::string Desc_MIMSV( const std::map<int,std::map<std::string,G4MaterialPropertyVector*>>& mimsv ) ;
};


inline NP* U4MaterialPropertyVector::ConvertToArray(const G4MaterialPropertyVector* prop)
{
    size_t num_val = prop->GetVectorLength() ; 
    NP* a = NP::Make<double>( num_val, 2 );  
    double* a_v = a->values<double>(); 
    for(size_t i=0 ; i < num_val ; i++)
    {   
        G4double energy = prop->Energy(i); 
        G4double value = (*prop)[i] ;
        a_v[2*i+0] = energy ; 
        a_v[2*i+1] = value ; 
    }   
    return a ;   
}


G4MaterialPropertyVector* U4MaterialPropertyVector::FromArray(const NP* a ) // static 
{   
    assert( a->uifc == 'f' && a->ebyte == 8 );
    
    size_t ni = a->shape[0] ;
    size_t nj = a->shape[1] ;
    assert( nj == 2 );
    
    G4double* energy = new G4double[ni] ;
    G4double* value = new G4double[ni] ;
    
    for(int i=0 ; i < int(ni) ; i++)
    {   
        energy[i] = a->get<double>(i,0) ;
        value[i] = a->get<double>(i,1) ;
    }
    G4MaterialPropertyVector* vec = new G4MaterialPropertyVector(energy, value, ni);
    return vec ;
}


inline G4MaterialPropertyVector* U4MaterialPropertyVector::Make_V(double value) // static
{
    int n = 2 ;
    G4double* e = new G4double[n] ;
    G4double* v = new G4double[n] ;

    e[0] = 1.55*eV ;
    e[1] = 15.5*eV ;

    v[0] = value ;
    v[1] = value ;

    G4MaterialPropertyVector* mpt = new G4MaterialPropertyVector(e, v, n);
    return mpt ;
}

inline std::string U4MaterialPropertyVector::Desc_V(const G4MaterialPropertyVector* v)
{
    size_t len = v->GetVectorLength() ;  
    std::stringstream ss ; 
    ss << " Desc_V" 
       << " len " << len
       << std::endl 
       ; 
    std::string s = ss.str(); 
    return s ; 
}


inline void U4MaterialPropertyVector::Import_MSV( std::map<std::string, G4MaterialPropertyVector*>& msv, const NPFold* sub)  // static
{
    int num_sub = sub->get_num_subfold();
    unsigned num_items = sub->num_items();
    assert( num_sub == 0 );

    for(unsigned idx=0 ; idx < num_items ; idx++)
    {
        const char* key_ = sub->get_key(idx);
        const char* key = NPFold::BareKey(key_);
        const NP* a = sub->get_array(idx);
        G4MaterialPropertyVector* mpv = FromArray(a) ;
        msv[key] = mpv ;
    }
}

inline NPFold* U4MaterialPropertyVector::Serialize_MSV( const std::map<std::string, G4MaterialPropertyVector*>& msv ) // static
{
    NPFold* f = new NPFold ; 
    typedef std::map<std::string, G4MaterialPropertyVector*> MSV ; 
    MSV::const_iterator it = msv.begin(); 

    for(unsigned i=0 ; i < msv.size() ; i++)
    { 
        const std::string& k = it->first ; 
        const G4MaterialPropertyVector* v = it->second ; 
        NP* a = ConvertToArray( v ); 
        f->add( k.c_str(), a );     
        std::advance(it, 1);  
    }
    return f ; 
} 

inline std::string U4MaterialPropertyVector::Desc_MSV(const std::map<std::string, G4MaterialPropertyVector*>& msv ) 
{
    typedef std::map<std::string, G4MaterialPropertyVector*> MSV ; 
    MSV::const_iterator it = msv.begin(); 
    std::stringstream ss ; 
    ss << "U4MaterialPropertyVector::Desc_MSV" << std::endl ; 

    for(unsigned i=0 ; i < msv.size() ; i++)
    { 
        const std::string& key = it->first ; 
        const G4MaterialPropertyVector* v = it->second ; 
        ss
           << " key " << key 
           << Desc_V(v) 
           << std::endl
           ;  
        std::advance(it, 1);  
    }
    std::string s = ss.str(); 
    return s ; 
}




inline void U4MaterialPropertyVector::Import_MIMSV( std::map<int, std::map<std::string, G4MaterialPropertyVector*>>& mimsv, const NPFold* f )
{
    typedef std::map<std::string, G4MaterialPropertyVector*> MSV ; 
    int num_sub = f->get_num_subfold();    

    for(int idx=0 ; idx < num_sub ; idx++)
    {
        const char* cat = f->get_subfold_key(idx); 
        int icat = U::To<int>(cat); 

        NPFold* sub = f->get_subfold(idx); 

        MSV& msv = mimsv[icat] ; 

        Import_MSV( msv, sub );         
    }
}


inline NPFold* U4MaterialPropertyVector::Serialize_MIMSV( const std::map<int, std::map<std::string, G4MaterialPropertyVector*>>& mimsv)  
{
    NPFold* f = new NPFold ; 

    typedef std::map<std::string, G4MaterialPropertyVector*> MSV ; 
    typedef std::map<int, MSV> MIMSV ; 

    MIMSV::const_iterator it = mimsv.begin(); 

    for(unsigned i=0 ; i < mimsv.size() ; i++)
    {
        int icat = it->first ; 
        const char* cat = U::FormName(icat) ; 

        const MSV& msv = it->second ; 
        NPFold* sub = Serialize_MSV( msv ); 

        f->add_subfold(cat, sub);         

        std::advance(it, 1); 
    }
    return f; 
}

inline std::string U4MaterialPropertyVector::Desc_MIMSV(const std::map<int,std::map<std::string, G4MaterialPropertyVector*>>& mimsv ) 
{
    std::stringstream ss ; 
    ss << "U4MaterialPropertyVector::Desc_MIMSV" << std::endl ; 

    typedef std::map<std::string, G4MaterialPropertyVector*> MSV ; 
    typedef std::map<int, MSV> MIMSV ; 
    MIMSV::const_iterator it = mimsv.begin(); 

    for(unsigned i=0 ; i < mimsv.size() ; i++)
    {
        int cat = it->first ; 
        const MSV& msv = it->second ; 
        ss
            << " cat " << cat 
            << " msv.size " << msv.size()
            << std::endl 
            << Desc_MSV(msv)
            << std::endl 
            ;
          
        std::advance(it, 1); 
    }
    std::string s = ss.str(); 
    return s ; 
}



