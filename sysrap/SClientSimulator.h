#pragma once
/**
SClientSimulator.h
===================

SClientSimulator.h can only be used from inside WITH_CURL guards,
which ensures libcurl of at least 8.12.1 is available as required by NP_CURL.h

**/

#include <cassert>
#include "stree.h"
#include "SSimulator.h"

#ifdef WITH_SEVT_MOCK
#include "SEvtMock.h"
#else
#include "SEvt.hh"
#include "SEventConfig.hh"
#endif

#include "NP.hh"
#include "NP_CURL.h"

struct SClientSimulator : public SSimulator
{
    static constexpr const char* NAME = "SClientSimulator" ;
    virtual ~SClientSimulator() = default ;

    static SClientSimulator* Create(const char* path="$CFBaseFromGEOM/CSGFoundry/SSim");
    static SClientSimulator* Create(const stree* tree);
    SClientSimulator(const stree* tr);

    const char* desc() const ;

    // low level API that enables QSim to control CSGOptiX irrespective of pkg dependency
    double render_launch();
    double simtrace_launch();
    double simulate_launch();
    double launch();


    double simtrace(int eventID);
    double render(const char* stem = nullptr);

    double simulate(int eventID, bool reset = false);
    void reset(int eventID);


    const stree* tree ;
    const char*  tree_digest ;

#ifdef WITH_SEVT_MOCK
    SEvtMock*    sev ;
#else
    SEvt*        sev ;
#endif
    int          placeholder ;

};




SClientSimulator* SClientSimulator::Create(const char* path) // static
{
    const char* ss = spath::Resolve(path) ;
    if(sstr::StartsWith(ss,"CFBaseFromGEOM"))
    {
         std::cerr << "SClientSimulator::Create - FAILED TO RESOLVE CFBaseFromGEOM \n";
         return nullptr ;
    }

    stree* tree = stree::Load(ss);
    if(!tree)
    {
        std::cerr << "SClientSimulator::Create - FAILED TO LOAD TREE FROM " << ( ss ? ss : "-" ) << "\n" ;
        return nullptr ;
    }
    return Create(tree);
}

SClientSimulator* SClientSimulator::Create(const stree* tree) // static
{
    SClientSimulator* client = new SClientSimulator(tree); ;
    return client ;
}




inline SClientSimulator::SClientSimulator(const stree* _tree)
    :
    tree(_tree),
    tree_digest(tree ? tree->get_tree_digest() : nullptr),
#ifdef WITH_SEVT_MOCK
    sev(SEvtMock::Get_EGPU()),
#else
    sev(SEvt::Get_EGPU()),
#endif
    placeholder(0)
{
}




// HMM: these are not relevant for Client
inline double SClientSimulator::render_launch(){ return 0. ; }
inline double SClientSimulator::simtrace_launch(){ return 0. ; }
inline double SClientSimulator::simulate_launch(){ return 0. ; }
inline double SClientSimulator::launch(){ return 0. ; }

inline const char* SClientSimulator::desc() const { return NAME ; }

inline double SClientSimulator::simtrace(int)
{
    return 0 ;
}
inline double SClientSimulator::render(const char*)
{
    return 0 ;
}


/**
SClientSimulator::simulate
---------------------------

In normal running u4 collects into SEvt::gensteps vector and at endOfEvent QSim::simulate
pulls genstep array from the gensteps vector::

    sev->beginOfEvent(eventID);  // set SEvt index and tees up frame gensteps for simtrace and input photon simulate running
    NP* igs = sev->makeGenstepArrayFromVector();

HMM: thinking about client server settings consistency, could just use settings from server
and give warnings on client when inconsistent ?

Methods on the server at the other end of the NP_CURL::TransformRemote call are::

    CSGOptiXService::simulate
    CSGOptiX::simulate  (accepting gs)
    QSim::simulate  (accepting gs)


**/


inline double SClientSimulator::simulate(int eventID, bool reset )
{
    sev->beginOfEvent(eventID);
    NP* gs = sev->makeGenstepArrayFromVector();
    if(gs == nullptr)
    {
        std::cerr
            << "SClientSimulator::simulate"
            << " eventID " << eventID
            << " NO GENSTEPS - NOTHING TO DO "
            << "\n"
            ;
        return 0;
    }

#ifdef WITH_SEVT_MOCK
#else
    std::string client_settings = SEventConfig::Settings();
    std::string client_digest = tree_digest ;
    gs->set_meta<std::string>("Settings",client_settings);
    gs->set_meta<std::string>("TreeDigest",client_digest);
#endif

    NP* hc = NP_CURL::TransformRemote(gs,eventID);  // "hc" hit-component one of : hit/hitlite/hitlitemerged/hitmerged
    sev->setHit(hc);

    double dt = hc->get_meta<double>("QSim__simulate_tot_dt", 0. );


    std::cout
          << "SClientSimulator::simulate "
          << " eventID " << eventID
          << " reset " << reset
          << " hc " << ( hc ? hc->sstr() : "-" )
          << " dt " << dt
          << "\n"
          ;

#ifdef WITH_SEVT_MOCK
#else
    std::string server_settings = hc->get_meta<std::string>("Settings");
    bool match_settings = SEventConfig::SettingsMatch(client_settings,server_settings);

    std::string server_digest = hc->get_meta<std::string>("TreeDigest");
    bool match_digest = 0 == strcmp( server_digest.c_str(), client_digest.c_str() );

    std::cout
          << "SClientSimulator::simulate "
          << " eventID " << eventID
          << " match_settings " << ( match_settings ? "YES" : "NO " )
          << " client_settings [" << client_settings << "]"
          << " server_settings [" << server_settings << "]"
          << " match_digest " << ( match_digest ? "YES" : "NO " )
          << " client_digest [" << client_digest << "]"
          << " server_digest [" << server_digest << "]"
          << "\n"
          ;
#endif

    return dt ;
}
inline void SClientSimulator::reset(int eventID)
{
    assert(eventID > -1);
}


