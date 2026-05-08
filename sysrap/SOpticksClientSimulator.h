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
#include "SEvt.hh"
#include "SEventConfig.hh"
#include "SProf.hh"

#include "NP.hh"
#include "NP_CURL.h"

struct SOpticksClientSimulator : public SSimulator
{
    static constexpr const char* NAME = "SOpticksClientSimulator" ;
    static constexpr const char* Settings = "Settings" ;
    static constexpr const char* TreeDigest = "TreeDigest" ;

    virtual ~SOpticksClientSimulator() = default ;

    static SOpticksClientSimulator* Create(const char* path="$CFBaseFromGEOM/CSGFoundry/SSim");
    static SOpticksClientSimulator* Create(const stree* tree);

    static bool            Consistent( const NP* gs, const NP* hc, const char* key );
    static std::string DescConsistent( const NP* gs, const NP* hc, const char* key );

    SOpticksClientSimulator(const stree* tr);

    const char* desc() const ;

    // low level API that enables QSim to control CSGOptiX irrespective of pkg dependency
    double render_launch();
    double simtrace_launch();
    double simulate_launch();
    double launch();


    double simtrace(int eventID);
    double render(const char* stem = nullptr);

    void   annotate_genstep(NP* gs) const;
    double simulate(int eventID, bool reset = false);



    void reset(int eventID);


    const stree* tree ;
    const char*  tree_digest ;

    SEvt*        sev ;

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

inline bool SOpticksClientSimulator::Consistent( const NP* gs, const NP* hc, const char* key )
{
    std::string gs_value = gs->get_meta<std::string>(key);
    std::string hc_value = hc->get_meta<std::string>(key);
    bool match_value = strcmp( gs_value.c_str(), hc_value.c_str() ) == 0 ;
    return match_value ;
}

inline std::string SOpticksClientSimulator::DescConsistent( const NP* gs, const NP* hc, const char* key )
{
    std::string gs_value = gs->get_meta<std::string>(key);
    std::string hc_value = hc->get_meta<std::string>(key);
    bool match_value = strcmp( gs_value.c_str(), hc_value.c_str() ) == 0 ;

    std::stringstream ss ;
    ss << "[SOpticksClientSimulator::DescConsistent "
       << key
       << " gs [" << gs_value << "]"
       << " hc [" << hc_value << "]"
       << " match " << ( match_value ? "YES" : "NO " )
       << "]"
       ;
    std::string str = ss.str() ;
    return str ;
}




inline SOpticksClientSimulator::SOpticksClientSimulator(const stree* _tree)
    :
    tree(_tree),
    tree_digest(tree ? tree->get_tree_digest() : nullptr),
    sev(SEvt::Get_EGPU())
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


inline void SOpticksClientSimulator::annotate_genstep(NP* gs) const
{
    std::string gs_Settings = SEventConfig::Settings();
    std::string gs_TreeDigest = tree_digest ;
    gs->set_meta<std::string>(Settings,  gs_Settings);
    gs->set_meta<std::string>(TreeDigest,gs_TreeDigest);
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
    annotate_genstep(gs);

    NP* hc = NP_CURL::TransformRemote(gs,eventID);  // "hc" hit-component one of : hit/hitlite/hitlitemerged/hitmerged

    if(hc == nullptr)
    {
        std::cerr << "SOpticksClientSimulator::simulate  ERROR NP_CURL::TransformRemote gave hc (NP*)nullptr - IS THE SERVER RUNNING ?\n" ;
        return -1. ;
    }


    sev->setHit(hc);

    double dt = hc ? hc->get_meta<double>("QSim__simulate_tot_dt", 0. ) : -1. ;

    bool consistent_Settings = Consistent(gs, hc, Settings) ;
    bool consistent_TreeDigest = Consistent(gs, hc, TreeDigest) ;
    bool consistent = consistent_Settings && consistent_TreeDigest ;

    std::cout
          << "SOpticksClientSimulator::simulate "
          << " eventID " << eventID
          << " reset " << reset
          << " gs " << ( gs ? gs->sstr() : "-" )
          << " hc " << ( hc ? hc->sstr() : "-" )
          << " All/Settings/TreeDigest: "
          << ( consistent            ? "Y" : "N" )
          << ( consistent_Settings   ? "Y" : "N" )
          << ( consistent_TreeDigest ? "Y" : "N" )
          << " dt " << dt
          << "\n"
          ;

    if(!consistent) std::cerr
        << "SOpticksClientSimulator::simulate "
        << " consistent " << ( consistent ? "YES" : "NO " )
        << " " << DescConsistent(gs, hc, Settings)
        << " " << DescConsistent(gs, hc, TreeDigest)
        << "\n"
        ;

    return dt ;
}







/**
SOpticksClientSimulator::reset
--------------------------------

Reset is vital in client running to clear the gensteps vector
after each event simulation.


Q: When should the client call SEvt::endOfEvent ?
A: Need to follow full opticks pattern, invoke SEvt::endOfEvent via the reset chain of methods
   that are invoked from the highest level. Moving to client Opticks switches
   the CSGOptiX simulator to this SOpticksClientSimulator so this must obey the reset.

**/


inline void SOpticksClientSimulator::reset(int eventID)
{
    SProf::Add("SOpticksClientSimulator__reset_HEAD");
    sev->endOfEvent(eventID);
    SProf::Add("SOpticksClientSimulator__reset_TAIL");
}


