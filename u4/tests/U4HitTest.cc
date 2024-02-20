/**
U4HitTest.cc
==============

::

    ~/o/u4/tests/U4HitTest.sh run_cat


**/

#include "OPTICKS_LOG.hh"
#include "SEvt.hh"
#include "ssys.h"
#include "SSim.hh"
#include "SProf.hh"
#include "spath.h"

#include "CSGFoundry.h"

#include "U4Hit.h"
#include "U4HitGet.h"


struct U4HitTest
{
    const char* METHOD ; 
    const char* rel ; 
    int ins ;   // SEvt::EGPU
    int idx ;   // 1st valid index : A000 
    SEvt* sev ; 
    unsigned num_hit ; 
    const char* cfbase ; 
    const CSGFoundry* fd ;

    sphoton global = {} ; 
    sphit ht = {}  ; 
    sphoton local = {}  ; 
    U4Hit hit = {} ; 

    sphit ht_alt = {}  ; 
    sphoton local_alt = {}  ; 


    int32_t delta_rs ; 
    int32_t range_rs ; 
    unsigned hit_idx ; 

    std::string desc() const ; 

    U4HitTest(); 
    void init(); 

    std::string dump() const ; 
    std::string brief() const ; 
    std::string smry() const ; 

    void convertHit(unsigned hidx, bool is_repeat); 
    void convertHit_ALT(unsigned hidx, bool is_repeat); 
    void convertHit_COMPARE(unsigned hidx, bool is_repeat); 

    void convertHits(); 
    void save() const ; 
}; 


inline std::string U4HitTest::desc() const 
{
    std::stringstream ss ; 
    ss << "U4HitTest::desc" << std::endl
       << "  sev   " << ( sev ? "YES" : "NO " ) << std::endl 
       << " num_hit " << num_hit << std::endl
       << " cfbase " << ( cfbase ? cfbase : "-" ) << std::endl
       << " fd     " << ( fd ? "YES" : "NO " ) << std::endl
       ;
    std::string str = ss.str(); 
    return str ;  
}

inline U4HitTest::U4HitTest()
    :
    METHOD(ssys::getenvvar("METHOD", "convertHit")),
    rel(nullptr),
    ins(0),
    idx(0),  
    sev(SEvt::LoadRelative(rel, ins, idx)),
    num_hit(sev ? sev->getNumHit() : 0),
    cfbase(sev ? sev->getSearchCFBase() : nullptr),
    fd(cfbase ? CSGFoundry::Load(cfbase) : nullptr ),
    delta_rs(0),
    range_rs(0),
    hit_idx(0)
{
    init(); 
}

inline void U4HitTest::init()
{
    sev->setGeo(fd); 

    assert( sev->hasInstance() );  // check that the instance is persisted and retrieved (via domain metadata)
    assert( sev == SEvt::Get(sev->instance) );  // check the loaded SEvt got slotted in expected slot 
}


inline std::string U4HitTest::dump() const 
{
    std::stringstream ss ; 
    ss << "U4HitTest::dump" << std::endl
       << " hit_idx " << hit_idx 
       << " Delta_RS " << delta_rs 
       << std::endl 
       << " global " << global.desc() << std::endl 
       << " local " << local.desc() << std::endl 
       << " hit " << hit.desc() << std::endl 
       << " ht " << ht.desc() << std::endl 
       ; 
    std::string str = ss.str(); 
    return str ;  
}

inline std::string U4HitTest::brief() const 
{
    std::stringstream ss ; 
    ss << "U4HitTest::brief" 
       << " hit_idx " << hit_idx 
       << " Delta_RS " << delta_rs 
       ; 
    std::string str = ss.str(); 
    return str ;  
}

inline std::string U4HitTest::smry() const 
{
    std::stringstream ss ; 
    ss << "U4HitTest::smry" << std::endl
       << " METHOD " << METHOD
       << " num_hit " << num_hit 
       << " SProf::Range_RS " << range_rs 
       << " SProf::Range_RS/num_hit " << std::setw(10) << std::fixed << std::setprecision(4) << double(range_rs)/double(num_hit) 
       << std::endl 
       ;
    std::string str = ss.str(); 
    return str ;  
}


inline void U4HitTest::convertHit(unsigned hidx, bool is_repeat)
{
    SProf::SetTag(hidx); 
    SProf::Add("Head"); 

    sev->getHit(global, hidx); 
    sev->getLocalHit( ht, local,  hidx); 
 
    U4HitGet::ConvertFromPhoton(hit,global,local, ht); 

    SProf::Add("Tail"); 
    delta_rs = SProf::Delta_RS(); 
    range_rs = SProf::Range_RS(); 
    //LOG_IF(info, delta_rs > 0) << dump() ; 
    LOG_IF(info, delta_rs > 0 || is_repeat) << brief() ; 
}

inline void U4HitTest::convertHit_ALT(unsigned hidx, bool is_repeat)
{
    SProf::SetTag(hidx); 
    SProf::Add("Head"); 

    sev->getHit(global, hidx); 
    sev->getLocalHit_ALT( ht_alt, local_alt,  hidx); 
 
    U4HitGet::ConvertFromPhoton(hit,global,local_alt, ht_alt); 

    SProf::Add("Tail"); 
    delta_rs = SProf::Delta_RS(); 
    range_rs = SProf::Range_RS(); 
    //LOG_IF(info, delta_rs > 0) << dump() ; 
    LOG_IF(info, delta_rs > 0 || is_repeat) << brief() ; 
}

inline void U4HitTest::convertHit_COMPARE(unsigned hidx, bool is_repeat)
{
    SProf::SetTag(hidx); 
    SProf::Add("Head"); 

    sev->getHit(global, hidx); 
    sev->getLocalHit( ht, local,  hidx); 
    sev->getLocalHit_ALT( ht_alt, local_alt,  hidx); 

    bool local_equal_flags = sphoton::EqualFlags(local, local_alt) ; 

    LOG_IF(fatal, !local_equal_flags) 
         << "sphoton::EqualFlags FAIL "
         << " hidx : " << hidx 
         << " local_equal_flags " << local_equal_flags
         << std::endl  
         << " local " << std::endl << local << std::endl 
         << " local_alt " << std::endl << local_alt << std::endl 
         ;

    float4 local_delta = sphoton::DeltaMax(local, local_alt ) ; 
    std::cout << " local_delta " << local_delta << std::endl ; 

    bool ht_match = ht == ht_alt ; 
    LOG_IF(fatal, !ht_match )
         << " hidx : " << hidx 
         << " FATAL : NOT ht_match "
         << std::endl 
         << " ht     : " << ht.desc()   
         << std::endl
         << " ht_alt : " << ht_alt.desc() 
         ;   


    U4HitGet::ConvertFromPhoton(hit,global,local, ht); 

    SProf::Add("Tail"); 
    delta_rs = SProf::Delta_RS(); 
    range_rs = SProf::Range_RS(); 
    //LOG_IF(info, delta_rs > 0) << dump() ; 
    LOG_IF(info, delta_rs > 0 || is_repeat) << brief() ; 

}

/**
U4HitTest::convertHits
------------------------

Trying to repeat a leaky hit, but the repeat doesnt leak 
Maybe cached/consolidated free or something. Means cannot 
fully trust RSS changes ? 

**/

inline void U4HitTest::convertHits()
{
    for(hit_idx=0 ; hit_idx < num_hit ; hit_idx++ )
    {
        delta_rs = 0 ; 

        if(strcmp(METHOD, "convertHit") == 0 )
        {
            convertHit(hit_idx, false); 
        }
        else if(strcmp(METHOD, "convertHit_COMPARE") == 0 )
        {
            convertHit_COMPARE(hit_idx, false); 
        }
        else if(strcmp(METHOD, "convertHit_ALT") == 0 )
        {
            convertHit_ALT(hit_idx, false); 
        }

        // if( delta_rs > 0 ) convertHit(hit_idx, true);   
        // for leaky hit, do it again to check reproducibility : it doesnt reproduce the leak 
    }
}

inline void U4HitTest::save() const 
{ 
    bool append = false ; 
    const char* _path = "$TMP/U4HitTest/U4HitTest.txt" ;  
    const char* path = spath::Resolve(_path) ; 
    SProf::Write(path, append); 
}

int main(int argc, char** argv)
{ 
    OPTICKS_LOG(argc, argv); 
    LOG(info) ;  

    // SSim Create/Load needed before CSGFoundry::Load

    //SSim::Create();    // this creates an empty stree
    SSim::Load();   // imports GEOM default persisted stree 

    U4HitTest test ; 

    LOG(info) << test.desc() ;  
    //LOG(info) << "test.sev->descFull" << std::endl << test.sev->descFull() ; 

    if(test.num_hit == 0) return 0 ; 

    test.convertHits(); 
    test.save(); 

    LOG(info) << test.smry() ; 

    return 0 ; 
}

