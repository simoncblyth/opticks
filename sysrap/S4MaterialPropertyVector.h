#pragma once
/**
S4MaterialPropertyVector.h
============================

This provides serialization of int, string keyed maps of 
G4MaterialPropertyVector into NPFold as well as the 
import of the NPFold back into maps. 

Hence this facilitates the saving and loading of maps of 
material properties to/from directory trees using NPFold persistency. 

It simplifies dependencies for this functionality to be available 
from sysrap rather than from u4, so this struct clones u4/U4MaterialPropertyVector.h 

**/

#include "G4MaterialPropertyVector.hh"
#include "G4SystemOfUnits.hh"

#include "NP.hh"
#include "NPFold.h"

struct S4MaterialPropertyVector
{
    static NP* ConvertToArray(const G4MaterialPropertyVector* vec);
    static G4MaterialPropertyVector* Load(const char* path);

    static G4MaterialPropertyVector* FromArrayData(const double* aa, int ni, int nj  ); 
    static G4MaterialPropertyVector* FromArray(const NP* prop);
    static G4MaterialPropertyVector* Make_V(double value); 
    static std::string Desc_V(const G4MaterialPropertyVector* v); 


    static void    Import_VV(         std::vector<G4MaterialPropertyVector*>& vv, const NPFold* f ) ;
    static NPFold* Serialize_VV(const std::vector<G4MaterialPropertyVector*>& vv ) ;
    static std::string  Desc_VV(const std::vector<G4MaterialPropertyVector*>& vv ) ;

    static void  Import_VV_CombinedArray(          std::vector<G4MaterialPropertyVector*>& vv, const NP* vvcom ); 
    static NP* Serialize_VV_CombinedArray(   const std::vector<G4MaterialPropertyVector*>& vv ) ; 
    static std::string Desc_VV_CombinedArray(const std::vector<G4MaterialPropertyVector*>& vv ); 


    static void    Import_MSV(          std::map<std::string,G4MaterialPropertyVector*>& msv, const NPFold* sub);
    static NPFold* Serialize_MSV( const std::map<std::string,G4MaterialPropertyVector*>& msv );
    static std::string Desc_MSV(  const std::map<std::string,G4MaterialPropertyVector*>& msv ) ;

    static void Import_MIMSV(            std::map<int,std::map<std::string,G4MaterialPropertyVector*>>& mimsv, const NPFold* f );
    static NPFold* Serialize_MIMSV(const std::map<int,std::map<std::string,G4MaterialPropertyVector*>>& mimsv ) ;
    static std::string Desc_MIMSV( const std::map<int,std::map<std::string,G4MaterialPropertyVector*>>& mimsv ) ;

};


inline NP* S4MaterialPropertyVector::ConvertToArray(const G4MaterialPropertyVector* prop)
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

inline G4MaterialPropertyVector* S4MaterialPropertyVector::Load(const char* path ) // static 
{    
    const NP* a = NP::Load(path); 
    return a ? FromArray(a) : nullptr ; 
}


inline G4MaterialPropertyVector* S4MaterialPropertyVector::FromArrayData(const double* aa, int ni, int nj  ) // static 
{   
    assert( ni >= 2 );
    assert( nj == 2 );
    
    G4double* energy = new G4double[ni] ;
    G4double* value = new G4double[ni] ;
    
    for(int i=0 ; i < int(ni) ; i++)
    {   
        energy[i] = aa[i*nj+0] ; 
        value[i] = aa[i*nj+1] ;
    }
    G4MaterialPropertyVector* vec = new G4MaterialPropertyVector(energy, value, ni);
    return vec ;
}




#ifdef OLD
inline G4MaterialPropertyVector* S4MaterialPropertyVector::FromArray(const NP* a ) // static 
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
#else
inline G4MaterialPropertyVector* S4MaterialPropertyVector::FromArray(const NP* a ) // static 
{   
    size_t ni = a->shape[0] ;
    size_t nj = a->shape[1] ;
    assert( nj == 2 );
    assert( a->uifc == 'f' && a->ebyte == 8 );
    const double* aa = a->cvalues<double>(); 
    G4MaterialPropertyVector* vec = FromArrayData(aa, ni, nj );  
    return vec ;
}
#endif






inline G4MaterialPropertyVector* S4MaterialPropertyVector::Make_V(double value) // static
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

inline std::string S4MaterialPropertyVector::Desc_V(const G4MaterialPropertyVector* v)
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






inline void  S4MaterialPropertyVector::Import_VV(         std::vector<G4MaterialPropertyVector*>& vv, const NPFold* f ) // static
{
    [[maybe_unused]] int num_sub = f->get_num_subfold();
    int num_items = f->num_items();
    assert( num_sub == 0 );

    vv.resize(num_items); 

    for(int i=0 ; i < num_items ; i++)
    {
        const char* key_ = f->get_key(i);
        const char* key = NPFold::BareKey(key_);
        int idx = std::atoi(key); 
        assert( idx == i );  
        const NP* a = f->get_array(idx);
        G4MaterialPropertyVector* mpv = FromArray(a) ;
        vv[i] = mpv ;
    }
}

inline NPFold* S4MaterialPropertyVector::Serialize_VV(const std::vector<G4MaterialPropertyVector*>& vv ) 
{
    NPFold* f = new NPFold ; 
    int num_v = vv.size(); 
    for(int i=0 ; i < num_v ; i++)
    { 
        std::string k = std::to_string(i);  
        const G4MaterialPropertyVector* v = vv[i] ; 
        NP* a = ConvertToArray( v ); 
        f->add( k.c_str(), a );     
    }
    return f ; 
}

inline std::string S4MaterialPropertyVector::Desc_VV(     const std::vector<G4MaterialPropertyVector*>& vv )
{
    int num_v = vv.size(); 
    std::stringstream ss ; 
    ss << "S4MaterialPropertyVector::Desc_VV num_v:" << num_v << std::endl ; 
    std::string str = ss.str() ;
    return str ;  
}



inline void  S4MaterialPropertyVector::Import_VV_CombinedArray(         std::vector<G4MaterialPropertyVector*>& vv, const NP* vvcom ) // static
{
    assert( vvcom && vvcom->shape.size() == 3 );
    int ni = vvcom->shape[0] ; 
    [[maybe_unused]] int nj = vvcom->shape[1] ; 
    [[maybe_unused]] int nk = vvcom->shape[2] ;
    assert( nj > 1 ); 
    assert( nk == 2 );  

    vv.resize(ni); 

    const double* aa0 = vvcom->cvalues<double>() ; 

    for(int i=0 ; i < ni ; i++)
    {
        const double* aa = aa0 + nj*nk*i ;  
        G4MaterialPropertyVector* mpv = FromArrayData(aa, nj, nk) ;
        vv[i] = mpv ;
    }
}



/**
S4MaterialPropertyVector::Serialize_VV_CombinedArray
---------------------------------------------------------

This alternative to Serialize_VV provides a more efficient serialization
for vectors with large numbers of entries. 

**/

inline NP* S4MaterialPropertyVector::Serialize_VV_CombinedArray(const std::vector<G4MaterialPropertyVector*>& vv ) 
{
    int num_v = vv.size();
    std::vector<const NP*> aa ;
    aa.resize(num_v);
    std::set<int> u_ni ;

    for(int i=0 ; i < num_v ; i++)
    {
        const G4MaterialPropertyVector* v = vv[i] ;
        const NP* a = ConvertToArray( v );

        assert( a->shape.size() == 2 );
        int ni = a->shape[0] ;
        [[maybe_unused]] int nj = a->shape[1] ;
        assert( nj == 2 ) ;
        u_ni.insert(ni) ;
        aa[i] = a ;
    }
    bool all_same_shape = u_ni.size() == 1 ;
    bool annotate = !all_same_shape ;
    const NP* parasite = nullptr ;

    NP* vvcom = NP::Combine(aa, annotate, parasite);
    return vvcom ;
}

inline std::string S4MaterialPropertyVector::Desc_VV_CombinedArray(const std::vector<G4MaterialPropertyVector*>& vv )
{
    int num_v = vv.size();
    std::set<size_t> ulen ;
    for(int i=0 ; i < num_v ; i++)
    {
        const G4MaterialPropertyVector* v = vv[i] ;
        size_t len = v->GetVectorLength() ;
        ulen.insert(len);
    }

    size_t len = ulen.size() == 1 ? *ulen.begin() : 0 ;
    std::stringstream ss ;
    ss << "S4MaterialPropertyVector::Desc_VV_CombinedArray num_v " << num_v << " len " << len << std::endl;
    std::string str = ss.str();
    return str ;
} 


inline void S4MaterialPropertyVector::Import_MSV( std::map<std::string, G4MaterialPropertyVector*>& msv, const NPFold* sub)  // static
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

inline NPFold* S4MaterialPropertyVector::Serialize_MSV( const std::map<std::string, G4MaterialPropertyVector*>& msv ) // static
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

inline std::string S4MaterialPropertyVector::Desc_MSV(const std::map<std::string, G4MaterialPropertyVector*>& msv )
{
    typedef std::map<std::string, G4MaterialPropertyVector*> MSV ;
    MSV::const_iterator it = msv.begin();
    std::stringstream ss ;
    ss << "S4MaterialPropertyVector::Desc_MSV" << std::endl ;

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
    std::string str = ss.str();
    return str ;
}




inline void S4MaterialPropertyVector::Import_MIMSV( std::map<int, std::map<std::string, G4MaterialPropertyVector*>>& mimsv, const NPFold* f )
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


inline NPFold* S4MaterialPropertyVector::Serialize_MIMSV( const std::map<int, std::map<std::string, G4MaterialPropertyVector*>>& mimsv)
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

inline std::string S4MaterialPropertyVector::Desc_MIMSV(const std::map<int,std::map<std::string, G4MaterialPropertyVector*>>& mimsv )
{
    std::stringstream ss ;
    ss << "S4MaterialPropertyVector::Desc_MIMSV" << std::endl ;

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



