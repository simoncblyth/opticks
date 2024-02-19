/**
U4HitTest.cc
==============

::

    ~/o/u4/tests/U4HitTest.sh run_cat


**/

#include "OPTICKS_LOG.hh"
#include "SEvt.hh"
#include "SSys.hh"
#include "SSim.hh"
#include "SProf.hh"
#include "spath.h"

#include "CSGFoundry.h"

#include "U4Hit.h"
#include "U4HitGet.h"


struct U4HitTest
{
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
    LOG(info) << desc() ;  
    sev->setGeo(fd); 

    assert( sev->hasInstance() );  // check that the instance is persisted and retrieved (via domain metadata)
    assert( sev == SEvt::Get(sev->instance) );  // check the loaded SEvt got slotted in expected slot 

    LOG(info) << "sev->descFull" << std::endl << sev->descFull() ; 
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
       << " num_hit " << num_hit 
       << " SProf::Range_RS " << range_rs 
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
        convertHit(hit_idx, false); 

        if( delta_rs > 0 ) convertHit(hit_idx, true);   
        // for leaky hit, do it again to check reproducibility
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

    SSim::Create();    // needed before CSGFoundry::Load
    U4HitTest test ; 
    if(test.num_hit == 0) return 0 ; 
    test.convertHits(); 
    test.save(); 

    LOG(info) << test.smry() ; 

    return 0 ; 
}

