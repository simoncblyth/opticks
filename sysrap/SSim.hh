#pragma once
/**
SSim.hh : Manages input arrays for QUDARap/QSim : Using NPFold plumbing
==================================================================================

The SSim instance provides the input arrays to QSim
which does the uploading to the device.
Currently the SSim instance is persisted within CSGFoundry/SSim
using NPFold functionality.

Initially SSim might seem like an extraneous wrapper on top of stree.h
but on listing features it is clear that the extra layer is worthwhile.

1. SSim.hh singleton, so stree.h/U4Tree.h can stay headeronly
2. collection of extra NPFold (eg jpmt) unrelated to stree.h
3. switching between alternate array sources so QSim need not know those details

SSim must be instanciated with SSim::Create prior to CSGFoundry::CSGFoundry
Currently that is done from G4CXOpticks::G4CXOpticks


**/

struct NP ;
struct NPFold ;
struct SBnd ;
struct SPrd ;
struct stree ;
struct SScene ;
struct SPMT ;

#include <vector>
#include <string>
#include "plog/Severity.h"
#include "SYSRAP_API_EXPORT.hh"
#include "snam.h"


struct SYSRAP_API SSim
{
    const char* relp ;
    NPFold*   top ;
    NPFold*   extra ;
    stree*    tree ;    // instanciated with SSim::SSim
    SScene*   scene ;


    static const plog::Severity LEVEL ;
    static const int stree_level ;
    static constexpr const char* RELDIR = "SSim" ;
    static constexpr const char* EXTRA = "extra" ;
    static constexpr const char* JPMT_RELP = "extra/jpmt" ;
    static constexpr const char* RELP_DEFAULT = "stree/standard" ;

    static SSim* INSTANCE ;
    static SSim* Get();
    static SSim* CreateOrReuse();

    static void  AddExtraSubfold(const char* k, const char* dir );
    static void  AddExtraSubfold(const char* k, NPFold* f);

    static int Compare( const SSim* a , const SSim* b ) ;
    static std::string DescCompare( const SSim* a , const SSim* b );

    static SSim* Create();
    static SSim* Load();
    static SSim* Load_(const char* dir);
    static SSim* Load(const char* base, const char* reldir=RELDIR );

private:
    SSim();
    void init();
public:
    stree* get_tree() const ;
    SScene* get_scene() const ;
    void set_override_scene(SScene* _scene);
    void initSceneFromTree();

public:
    int lookup_mtline( int mtindex ) const ;
    std::string desc_mt() const ;
public:
    // top NPFold must be populated with SSim::serialize
    // prior to these accessors working
    std::string desc() const ;
    std::string brief() const ;

    const NP* get(const char* k) const ;
    void      set(const char* k, const NP* a) ;

    const NP* get_bnd() const ;
    const char* getBndName(unsigned bidx) const ;
    int getBndIndex(const char* bname) const ;
    const SBnd* get_sbnd() const ;
    const SPrd* get_sprd() const ;


    const NPFold* get_jpmt_nocopy() const ;   // raw PMT info
    const NPFold* get_jpmt() const ;   // raw PMT info
    const SPMT*   get_spmt() const ;   // struct that summarizes PMT info
    const NPFold* get_spmt_f() const ; // fold with summarized PMT info
public:
    void add_extra_subfold(const char* k, NPFold* f );

public:
    void save(const char* base, const char* reldir=RELDIR) ;  // not const as may serialize
    void load(const char* base, const char* reldir=RELDIR) ;
    void load_(const char* dir);
    void serialize();
    bool hasTop() const ;


public:

    /**
    TODO: MOST OF THE BELOW ARE DETAILS THAT ARE
    ONLY RELEVANT TO TEST GEOMETRIES HENCE THEY
    SHOULD BE RELOCATED ELSEWHERE, AND THE API
    UP HERE SLIMMED DOWN DRASTICALLY
    **/

    template<typename ... Args> void addFake( Args ... args );
    void addFake_( const std::vector<std::string>& specs );

    static void Add(
        NP** opticalplus,
        NP** bndplus,
        const NP* optical,
        const NP* bnd,
        const std::vector<std::string>& specs
        );

    static NP*  AddOptical(
        const NP* optical,
        const std::vector<std::string>& bnames,
        const std::vector<std::string>& specs
        ) ;

    static NP*  AddBoundary(
        const NP* src,
        const std::vector<std::string>& specs
        );

    static void GetPerfectValues(
        std::vector<float>& values,
        unsigned nk, unsigned nl, unsigned nm, const char* name
        );

    bool hasOptical() const ;
    std::string descOptical() const ;
    static std::string DescOptical(const NP* optical, const NP* bnd );

    static std::string GetItemDigest( const NP* bnd, int i, int j, int w );
    bool   findName( int& i, int& j, const char* qname ) const ;
public:
    void set_extra(const char * k , const NP* f );
    const NP* get_extra(const char * k ) const;
    static void AddMultiFilm(const char* k, const NP* f);
};


