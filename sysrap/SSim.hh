#pragma once
/**
SSim.hh : Manages input arrays for QUDARap/QSim : Using Single NPFold Member
==================================================================================


SSim must be instanciated with SSim::Create prior to CSGFoundry::CSGFoundry 

Currently that is done from G4CXOpticks::G4CXOpticks 


and populated by GGeo::convertSim which is for example invoked 
during GGeo to CSGFoundry conversion within CSG_GGeo_Convert::convertSim

Currently the SSim instance is persisted within CSGFoundry/SSim 
using NPFold functionality.  

The SSim instance provides the input arrays to QSim
which does the uploading to the device. 

**/

struct NP ; 
struct NPFold ; 
struct SBnd ; 
struct stree ; 

#include <vector>
#include <string>
#include "plog/Severity.h"
#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API SSim
{
    static const plog::Severity LEVEL ; 
    static const int stree_level ; 

    static constexpr const char* RELDIR = "SSim" ; 
    static constexpr const char* BND = "bnd.npy" ; 
    static constexpr const char* OPTICAL = "optical.npy" ;
    static constexpr const char* ICDF = "icdf.npy" ;
    static constexpr const char* PROPCOM = "propcom.npy" ;
    static constexpr const char* MULTIFILM = "multifilm.npy" ;
 
    static const unsigned       MISSING ; 
    static SSim* INSTANCE ; 
    static SSim* Get(); 
    static SSim* CreateOrReuse(); 

    static void  Add(const char* k, const NP* a); 
    static void  AddSubfold(const char* k, NPFold* f); 

    static SSim* Create(); 
    static const char* DEFAULT ; 
    static SSim* Load(); 
    static SSim* Load(const char* base, const char* rel=RELDIR ); 
    static SSim* Load_(const char* dir); 

    static int Compare( const SSim* a , const SSim* b ) ; 
    static std::string DescCompare( const SSim* a , const SSim* b ); 


    static void Add( NP** opticalplus, NP** bndplus, const NP* optical, const NP* bnd,  const std::vector<std::string>& specs ); 
    static NP*  AddOptical( const NP* optical, const std::vector<std::string>& bnames, const std::vector<std::string>& specs ) ; 
    static NP*  AddBoundary( const NP* src, const std::vector<std::string>& specs ); 
    static void GetPerfectValues( std::vector<float>& values, unsigned nk, unsigned nl, unsigned nm, const char* name ); 
    static std::string DescOptical(const NP* optical, const NP* bnd ); 
    static std::string GetItemDigest( const NP* bnd, int i, int j, int w ); 
    bool   findName( int& i, int& j, const char* qname ) const ; 

    template<typename ... Args>
    void addFake( Args ... args ); 

    void addFake_( const std::vector<std::string>& specs ); 

    NPFold* fold ; 
    stree*  tree ;  // instanciated with SSim::SSim


    void add(const char* k, const NP* a ); 
    void add_subfold(const char* k, NPFold* f ); 

    const NP* get(const char* k) const ; 
    const NP* get_bnd() const ; 
    const SBnd* get_sbnd() const ; 
    void import_bnd() ; 
    stree* get_tree() const ; 
    int lookup_mtline( int mtindex ) const ; 

    void save(const char* base, const char* reldir=RELDIR) const ; 
    void load(const char* base, const char* reldir=RELDIR) ; 
    static const bool load_tree_load ;   

    std::string desc() const ; 
    std::string descOptical() const ; 
    bool hasOptical() const ; 

    const char* getBndName(unsigned bidx) const ; 
    int getBndIndex(const char* bname) const ; 

private:
    SSim(); 
    void init(); 
};


