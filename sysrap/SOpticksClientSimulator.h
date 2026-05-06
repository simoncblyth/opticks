#pragma once
/**
SOpticksClientSimulator.h
==========================

SOpticksClientSimulator.h can only be used from inside WITH_CURL guards,
which ensures libcurl of at least 7.76.1 is available as required by NP_CURL.h

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

struct SOpticksClientSimulator : public SSimulator
{
    static constexpr const char* NAME = "SOpticksClientSimulator" ;
    virtual ~SOpticksClientSimulator() = default ;

    static SOpticksClientSimulator* Create(const char* path="$CFBaseFromGEOM/CSGFoundry/SSim");
    static SOpticksClientSimulator* Create(const stree* tree);
    SOpticksClientSimulator(const stree* tr);

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




inline SOpticksClientSimulator* SOpticksClientSimulator::Create(const char* path) // static
{
    const char* ss = spath::Resolve(path) ;
    if(sstr::StartsWith(ss,"CFBaseFromGEOM"))
    {
         std::cerr << "SOpticksClientSimulator::Create - FAILED TO RESOLVE CFBaseFromGEOM \n";
         return nullptr ;
    }

    stree* tree = stree::Load(ss);
    if(!tree)
    {
        std::cerr << "SOpticksClientSimulator::Create - FAILED TO LOAD TREE FROM " << ( ss ? ss : "-" ) << "\n" ;
        return nullptr ;
    }
    return Create(tree);
}

inline SOpticksClientSimulator* SOpticksClientSimulator::Create(const stree* tree) // static
{
    SOpticksClientSimulator* client = new SOpticksClientSimulator(tree); ;
    return client ;
}




inline SOpticksClientSimulator::SOpticksClientSimulator(const stree* _tree)
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
inline double SOpticksClientSimulator::render_launch(){ return 0. ; }
inline double SOpticksClientSimulator::simtrace_launch(){ return 0. ; }
inline double SOpticksClientSimulator::simulate_launch(){ return 0. ; }
inline double SOpticksClientSimulator::launch(){ return 0. ; }

inline const char* SOpticksClientSimulator::desc() const { return NAME ; }

inline double SOpticksClientSimulator::simtrace(int)
{
    return 0 ;
}
inline double SOpticksClientSimulator::render(const char*)
{
    return 0 ;
}


/**
SOpticksClientSimulator::simulate
-----------------------------------

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


inline double SOpticksClientSimulator::simulate(int eventID, bool reset )
{
    sev->beginOfEvent(eventID);
    NP* gs = sev->makeGenstepArrayFromVector();
    if(gs == nullptr)
    {
        std::cerr
            << "SOpticksClientSimulator::simulate"
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

    if(hc == nullptr)
    {
        std::cerr << "SOpticksClientSimulator::simulate  ERROR NP_CURL::TransformRemote gave hc (NP*)nullptr - IS THE SERVER RUNNING ?\n" ;
        return -1. ;
    }


    sev->setHit(hc);
    double dt = hc ? hc->get_meta<double>("QSim__simulate_tot_dt", 0. ) : -1. ;


    std::cout
          << "SOpticksClientSimulator::simulate "
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
          << "SOpticksClientSimulator::simulate "
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
inline void SOpticksClientSimulator::reset(int eventID)
{
    assert(eventID > -1);
}


