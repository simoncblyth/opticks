#pragma once
/**
SSim.hh : Manages input arrays for QUDARap/QSim
===================================================

Canonically instanciated by CSGFoundry::CSGFoundry 
and populated by GGeo::convertSim which is for example invoked 
during GGeo to CSGFoundry conversion within CSG_GGeo_Convert::convertSim

Currently the SSim instance is persisted within CSGFoundry/SSim 
using NPFold functionality.  

The SSim instance provides the input arrays to QSim
which does the uploading to the device. 

**/

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
    static constexpr const char* ICDF = "icdf.npy" ;
    static constexpr const char* PROPCOM = "propcom.npy" ;
    static constexpr const char* MULTIFILM = "multifilm.npy" ;
 
    static const unsigned       MISSING ; 
    static SSim* INSTANCE ; 
    static SSim* Get(); 
    static SSim* Create(); 
    static SSim* Load(); 
    static SSim* Load(const char* base); 
    static SSim* Load(const char* base, const char* rel); 
    static int Compare( const SSim* a , const SSim* b, bool dump=false ) ; 


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


    template<typename ... Args>
    void addFake( Args ... args ); 

    void addFake_( const std::vector<std::string>& specs ); 


    NPFold* fold ; 
    SName*  bd ; 


    void add(const char* k, const NP* a ); 
    const NP* get(const char* k) const ; 

    void load(const char* base); 
    void load(const char* base, const char* rel) ; 
    void postload(); 

    void save(const char* base) const ; 
    void save(const char* base, const char* rel) const ; 

    std::string desc() const ; 
    std::string descOptical() const ; 
    bool hasOptical() const ; 

    const char* getBndName(unsigned bidx) const ; 
    int getBndIndex(const char* bname) const ; 

private:
    SSim(); 


};







