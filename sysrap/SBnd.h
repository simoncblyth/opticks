#pragma once
/**
SBnd.h
========

Principal user QBnd.hh

* SSim has several functions that perhaps could be relocated here 
* Note that PLOG logging machinery doesnt work with header only imps 

**/

#include <cassert>
#include <string>
#include <vector>
#include <array>
#include <sstream>

#include "plog/Severity.h"
#include "NP.hh"
#include "SSim.hh"
#include "SStr.hh"


struct SBndProp
{
    int  group ; 
    int  prop ;  
    char name[16] ; 
    std::string desc() const ; 
}; 

inline std::string SBndProp::desc() const 
{
    std::stringstream ss ; 
    ss << "(" << group << "," << prop << ") " << name ; 
    std::string s = ss.str(); 
    return s ; 
}


struct SBnd
{
    const NP* src ; 

    // static constexpr const plog::Severity LEVEL = PLOG::EnvLevel("SBnd", "DEBUG")  ;  
    //     cannot constexpr LEVEL as not appropriate to define LEVEL as compile time even if could
    //
    // static const plog::Severity LEVEL ;  

    static constexpr const unsigned MISSING = ~0u ;
    const std::vector<std::string>& bnames ; 

    static constexpr std::array<SBndProp, 8> MaterialProp = 
    {{
        { 0,0,"RINDEX" },
        { 0,1,"ABSLENGTH" },
        { 0,2,"RAYLEIGH" },
        { 0,3,"REEMISSIONPROB" },
        { 1,0,"GROUPVEL" },
        { 1,1,"SPARE11"  },
        { 1,2,"SPARE12"  },
        { 1,3,"SPARE13"  },
    }};
    static std::string DescMaterialProp(); 
    static void GetMaterialPropNames(std::vector<std::string>& pnames, const char* skip_prefix="SPARE"); 
    static const SBndProp* FindMaterialProp(const char* pname); 


    SBnd(const NP* src_); 

    std::string getItemDigest( int i, int j, int w=8 ) const ;
    std::string descBoundary() const ;
    unsigned getNumBoundary() const ;
    const char* getBoundarySpec(unsigned idx) const ;
    void        getBoundarySpec(std::vector<std::string>& names, const unsigned* idx , unsigned num_idx ) const ;

    unsigned    getBoundaryIndex(const char* spec) const ;

    void        getBoundaryIndices( std::vector<unsigned>& bnd_idx, const char* bnd_sequence, char delim=',' ) const ;
    std::string descBoundaryIndices( const std::vector<unsigned>& bnd_idx ) const ;

    unsigned    getBoundaryLine(const char* spec, unsigned j) const ;
    static unsigned GetMaterialLine( const char* material, const std::vector<std::string>& specs );
    unsigned    getMaterialLine( const char* material ) const ;

    std::string desc() const ; 

    void getMaterialNames( std::vector<std::string>& names ) const ; 
    static std::string DescNames( std::vector<std::string>& names ) ; 



    bool findName( int& i, int& j, const char* qname ) const ; 
    static bool FindName( int& i, int& j, const char* qname, const std::vector<std::string>& names ) ; 

    NP* getPropertyGroup(const char* qname, int k=-1) const ;  

    template<typename T>
    void getProperty(std::vector<T>& out, const char* qname, const char* propname ) const ; 

};


inline std::string SBnd::DescMaterialProp() // static 
{
    std::stringstream ss ; 
    for(unsigned i=0 ; i < MaterialProp.size() ; i++)  ss << MaterialProp[i].desc() << std::endl; 
    std::string s = ss.str(); 
    return s ; 
}

inline void SBnd::GetMaterialPropNames(std::vector<std::string>& pnames, const char* skip_prefix ) // static
{
    for(unsigned i=0 ; i < MaterialProp.size() ; i++) 
    {
        const char* name = MaterialProp[i].name ; 
        if(SStr::StartsWith(name, skip_prefix) == false ) pnames.push_back(name) ; 
    }
}


inline const SBndProp* SBnd::FindMaterialProp(const char* pname) // static
{
    const SBndProp* prop = nullptr ; 
    for(unsigned i=0 ; i < MaterialProp.size() ; i++) if(strcmp(MaterialProp[i].name, pname)==0) prop = &MaterialProp[i] ; 
    return prop ; 
}




// this gives duplicate symbols headeronly as cannot inline static constants in C++11
// const plog::Severity SBnd::LEVEL = PLOG::EnvLevel("SBnd", "DEBUG"); 

inline SBnd::SBnd(const NP* src_)
    :
    src(src_),
    bnames(src->names)
{
    assert(bnames.size() > 0 ); 
}

inline std::string SBnd::getItemDigest( int i, int j, int w ) const 
{
    return SSim::GetItemDigest(src, i, j, w );  
}
inline std::string SBnd::descBoundary() const
{
    std::stringstream ss ; 
    for(unsigned i=0 ; i < bnames.size() ; i++) 
       ss << std::setw(2) << i << " " << bnames[i] << std::endl ; 
    std::string s = ss.str(); 
    return s ; 
} 


inline unsigned SBnd::getNumBoundary() const
{
    return bnames.size(); 
}
inline const char* SBnd::getBoundarySpec(unsigned idx) const 
{
    assert( idx < bnames.size() );  
    const std::string& s = bnames[idx]; 
    return s.c_str(); 
}
inline void SBnd::getBoundarySpec(std::vector<std::string>& names, const unsigned* idx , unsigned num_idx ) const 
{
    for(unsigned i=0 ; i < num_idx ; i++)
    {   
        unsigned index = idx[i] ;   
        const char* spec = getBoundarySpec(index);   // 0-based 
        names.push_back(spec); 
    }   
} 


inline unsigned SBnd::getBoundaryIndex(const char* spec) const 
{
    unsigned idx = MISSING ; 
    for(unsigned i=0 ; i < bnames.size() ; i++) 
    {
        if(spec && strcmp(bnames[i].c_str(), spec) == 0) 
        {
            idx = i ; 
            break ; 
        }
    }
    return idx ;  
}

inline void SBnd::getBoundaryIndices( std::vector<unsigned>& bnd_idx, const char* bnd_sequence, char delim ) const 
{
    assert( bnd_idx.size() == 0 ); 

    std::vector<std::string> bnd ; 
    SStr::Split(bnd_sequence,delim, bnd ); 

    for(unsigned i=0 ; i < bnd.size() ; i++)
    {
        const char* spec = bnd[i].c_str(); 
        if(strlen(spec) == 0) continue ;  
        unsigned bidx = getBoundaryIndex(spec); 
        if( bidx == MISSING ) std::cerr << " i " << i << " invalid spec [" << spec << "]" << std::endl ;      
        assert( bidx != MISSING ); 

        bnd_idx.push_back(bidx) ; 
    }
}

inline std::string SBnd::descBoundaryIndices( const std::vector<unsigned>& bnd_idx ) const 
{
    std::stringstream ss ; 
    for(unsigned i=0 ; i < bnd_idx.size() ; i++)
    {
        unsigned bidx = bnd_idx[i] ;  
        const char* spec = getBoundarySpec(bidx); 
        ss
            << " i " << std::setw(3) << i 
            << " bidx " << std::setw(3) << bidx
            << " spec " << spec
            << std::endl 
            ;
    }
    std::string s = ss.str(); 
    return s ; 
}



inline unsigned SBnd::getBoundaryLine(const char* spec, unsigned j) const 
{
    unsigned idx = getBoundaryIndex(spec); 
    bool is_missing = idx == MISSING ; 
    bool is_valid = !is_missing && idx < bnames.size() ;

    if(!is_valid) 
    {
        std::cerr
            << " not is_valid " 
            << " spec " << spec
            << " idx " << idx
            << " is_missing " << is_missing 
            << " bnames.size " << bnames.size() 
            << std::endl 
            ;  
    }

    assert( is_valid ); 
    unsigned line = 4*idx + j ;    
    return line ;  
}


inline unsigned SBnd::GetMaterialLine( const char* material, const std::vector<std::string>& specs ) // static
{
    unsigned line = MISSING ; 
    for(unsigned i=0 ; i < specs.size() ; i++) 
    {
        std::vector<std::string> elem ; 
        SStr::Split(specs[i].c_str(), '/', elem );  
        const char* omat = elem[0].c_str(); 
        const char* imat = elem[3].c_str(); 

        if(strcmp( material, omat ) == 0 )
        {
            line = i*4 + 0 ; 
            break ; 
        }
        if(strcmp( material, imat ) == 0 )
        {
            line = i*4 + 3 ; 
            break ; 
        }
    }
    return line ; 
}

/**
SBnd::getMaterialLine
-----------------------

Searches the bname spec for the *material* name in omat or imat slots, 
returning the first found.  

**/

inline unsigned SBnd::getMaterialLine( const char* material ) const
{
    return GetMaterialLine(material, bnames);
}






inline std::string SBnd::desc() const 
{
    return SSim::DescDigest(src,8) ;
    // TODO: seems funny getting this general functionality from SSim, relocate 
}


/**
inline void SBnd::GetSpecsFromString( std::vector<std::string>& specs , const char* specs_, char delim )
{
    std::stringstream ss;
    ss.str(specs_)  ;
    std::string s;
    while (std::getline(ss, s, delim)) if(!SStr::Blank(s.c_str())) specs.push_back(s) ;
    std::cout << " specs_ [" << specs_ << "] specs.size " << specs.size()  ;   
}
**/


inline void SBnd::getMaterialNames( std::vector<std::string>& names ) const 
{
    for(unsigned i=0 ; i < bnames.size() ; i++) 
    {
        std::vector<std::string> elem ; 
        SStr::Split(bnames[i].c_str(), '/', elem );  
        const char* omat = elem[0].c_str(); 
        const char* imat = elem[3].c_str(); 

        if(std::find(names.begin(), names.end(), omat) == names.end()) names.push_back(omat); 
        if(std::find(names.begin(), names.end(), imat) == names.end()) names.push_back(imat); 
    }
}
inline std::string SBnd::DescNames( std::vector<std::string>& names ) 
{
    std::stringstream ss ; 
    for(unsigned i=0 ; i < names.size() ; i++) ss << names[i] << std::endl ; 
    std::string s = ss.str(); 
    return s ; 
}



/**
SSim::findName
----------------

Returns the first (i,j)=(bidx,species) with element name 
matching query name *qname*. 

bidx 
    0-based boundary index 

species
    0,1,2,3 for omat/osur/isur/imat

**/

inline bool SBnd::findName( int& i, int& j, const char* qname ) const 
{
    return FindName(i, j, qname, bnames ) ; 
}

inline bool SBnd::FindName( int& i, int& j, const char* qname, const std::vector<std::string>& names ) 
{
    i = -1 ; 
    j = -1 ; 
    for(int b=0 ; b < int(names.size()) ; b++) 
    {
        std::vector<std::string> elem ; 
        SStr::Split(names[b].c_str(), '/', elem );  

        for(int s=0 ; s < 4 ; s++)
        {
            const char* name = elem[s].c_str(); 
            if(strcmp(name, qname) == 0 )
            {
                i = b ; 
                j = s ; 
                return true ; 
            }
        }
    }
    return false ;  
}

/**
SBnd::getPropertyGroup
------------------------

Returns an array of material or surface properties selected by *qname* eg "Water".
For example with a source bnd array of shape  (44, 4, 2, 761, 4, )
the expected spawned property group  array shape depends on the value of k:

k=-1
     (2, 761, 4,)   eight property vaulues across wavelength domain
k=0
     (761, 4)       four property vaulues across wavelength domain 
k=1
     (761, 4)

**/

inline NP* SBnd::getPropertyGroup(const char* qname, int k) const 
{
    int i, j ; 
    bool found = findName(i, j, qname); 
    assert(found); 
    return src->spawn_item(i,j,k);  
}

/**
SBnd::getProperty
------------------

::

    // <f8(44, 4, 2, 761, 4, )
    int species = 0 ; 
    int group = 1 ; 
    int wavelength = -1 ; 
    int prop = 0 ; 

**/

template<typename T>
inline void SBnd::getProperty(std::vector<T>& out, const char* qname, const char* propname ) const 
{
    assert( sizeof(T) == src->ebyte ); 

    int boundary, species ; 
    bool found_qname = findName(boundary, species, qname); 
    assert(found_qname); 
 
    const SBndProp* bp = FindMaterialProp(propname); 
    assert(bp); 

    int group = bp->group ; 
    int prop = bp->prop  ; 
    int wavelength = -1 ;   // slice dimension 

    src->slice(out, boundary, species, group, wavelength, prop );  
}





