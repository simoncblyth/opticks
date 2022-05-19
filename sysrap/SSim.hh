#pragma once

struct NP ; 
struct NPFold ; 
struct SName ; 

#include <vector>
#include <string>
#include "plog/Severity.h"
#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API SSim
{
    static const plog::Severity LEVEL ; 
    static constexpr const char* BND = "bnd.npy" ; 
    static constexpr const char* OPTICAL = "optical.npy" ;
 
    static const unsigned       MISSING ; 
    static SSim* INSTANCE ; 
    static SSim* Get(); 
    static SSim* Load(const char* base); 
    static SSim* Load(const char* base, const char* rel); 


    static void Add( NP** opticalplus, NP** bndplus, const NP* optical, const NP* bnd,  const std::vector<std::string>& specs ); 
    static NP*  AddOptical( const NP* optical, const std::vector<std::string>& bnames, const std::vector<std::string>& specs ) ; 
    static NP*  AddBoundary( const NP* src, const std::vector<std::string>& specs ); 
    static const NP* NarrowIfWide(const NP* buf ); 
    static bool FindName( unsigned& i, unsigned& j, const char* qname, const std::vector<std::string>& names ); 
    bool   findName( unsigned& i, unsigned& j, const char* qname ) const ; 
    static void GetPerfectValues( std::vector<float>& values, unsigned nk, unsigned nl, unsigned nm, const char* name ); 
    static std::string DescOptical(const NP* optical, const NP* bnd ); 
    static std::string DescDigest(const NP* bnd, int w=16) ; 
    static std::string GetItemDigest( const NP* bnd, int i, int j, int w ); 


    NPFold* fold ; 
    SName*  bd ; 

    SSim(); 

    void add(const char* k, const NP* a ); 
    const NP* get(const char* k) const ; 

    void load(const char* base); 
    void load(const char* base, const char* rel) ; 
    void save(const char* base) const ; 
    void save(const char* base, const char* rel) const ; 

    std::string desc() const ; 

    const char* getBndName(unsigned bidx) const ; 
    int getBndIndex(const char* bname) const ; 

};







