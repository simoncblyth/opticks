#pragma once
/**
SClientSimulator.h
===================

**/

#include <cassert>
#include "stree.h"
#include "SSimulator.h"

#ifdef WITH_SEVT_MOCK
#include "SEvtMock.h"
#else
#include "SEvt.hh"
#endif

#include "NP.hh"
#include "NP_CURL.h"

struct SClientSimulator : public SSimulator
{
    static constexpr const char* NAME = "SClientSimulator" ;
    virtual ~SClientSimulator() = default ;

    static SClientSimulator* Create(const char* path="$CFBaseFromGEOM/CSGFoundry/SSim");
    SClientSimulator(const stree* tr);
    void init();

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

    SClientSimulator* client = new SClientSimulator(tree); ;
    return client ;
}



inline SClientSimulator::SClientSimulator(const stree* _tree)
    :
    tree(_tree),
#ifdef WITH_SEVT_MOCK
    sev(new SEvtMock),
#else
    sev(SEvt::Get_EGPU()),
#endif
    placeholder(0)
{
    init();
}

inline void SClientSimulator::init()
{
#ifdef WITH_SEVT_MOCK
    sev->load_genstep("$FOLD/gs.npy");
#endif
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

TODO: implement this using NP_CURL.h
get gensteps from SEvt, then populate SEvt hits


In normal running u4 collects into SEvt::gensteps vector and at endOfEvent QSim::simulate
pulls genstep array from the gensteps vector::

    sev->beginOfEvent(eventID);  // set SEvt index and tees up frame gensteps for simtrace and input photon simulate running
    NP* igs = sev->makeGenstepArrayFromVector();



**/


inline double SClientSimulator::simulate(int eventID, bool reset )
{
    assert(eventID > -1);
    assert(reset == false);

    sev->beginOfEvent(eventID);
    NP* igs = sev->makeGenstepArrayFromVector();

    NP* hit = NP_CURL::TransformRemote(igs,eventID);

    // TODO: PLACE HIT INTO CLIENT SEvt SO IT LOOKS THE SAME AS MONO-RUNNING

    std::cout << "SClientSimulator::simulate " << eventID << " hit " << ( hit ? hit->sstr() : "-" ) << "\n" ;


    return 0. ;
}
inline void SClientSimulator::reset(int eventID)
{
    assert(eventID > -1);
}


