#pragma once
/**
SClientSimulator.h
===================

**/

#include <cassert>
#include "stree.h"
#include "SSimulator.h"

#include "NP.hh"
#include "NP_CURL.h"

struct SClientSimulator : public SSimulator
{
    static constexpr const char* NAME = "SClientSimulator" ;
    virtual ~SClientSimulator() = default ;

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

};


inline SClientSimulator::SClientSimulator(const stree* _tr)
   :
   tree(_tree)
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

TODO: implement this using NP_CURL.h
get gensteps from SEvt, then populate SEvt hits

**/


inline double SClientSimulator::simulate(int eventID, bool reset )
{
    assert(eventID > -1);
    assert(reset == false);
    return 0. ;
}
inline void SClientSimulator::reset(int eventID)
{
    assert(eventID > -1);
}


