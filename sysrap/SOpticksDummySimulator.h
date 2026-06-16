#pragma once
/**
SOpticksDummySimulator.h
==========================

This placeholder simulator is used from G4CXOpticks::CreateSimulator when no CUDA capable
device is detected. Simulate that on a GPU workstation with::

   export CUDA_VISIBLE_DEVICES=""

**/

#include <cassert>
#include "stree.h"
#include "SSimulator.h"

struct SOpticksDummySimulator : public SSimulator
{
    static constexpr const char* NAME = "SOpticksDummySimulator" ;

    virtual ~SOpticksDummySimulator() = default ;

    static SOpticksDummySimulator* Create(const char* path="$CFBaseFromGEOM/CSGFoundry/SSim");
    static SOpticksDummySimulator* Create(const stree* tree);

    SOpticksDummySimulator(const stree* tr);

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

};




inline SOpticksDummySimulator* SOpticksDummySimulator::Create(const char* path) // static
{
    const char* ss = spath::Resolve(path) ;
    if(sstr::StartsWith(ss,"CFBaseFromGEOM"))
    {
         std::cerr << "SOpticksDummySimulator::Create - FAILED TO RESOLVE CFBaseFromGEOM \n";
         return nullptr ;
    }

    stree* tree = stree::Load(ss);
    if(!tree)
    {
        std::cerr << "SOpticksDummySimulator::Create - FAILED TO LOAD TREE FROM " << ( ss ? ss : "-" ) << "\n" ;
        return nullptr ;
    }
    return Create(tree);
}

inline SOpticksDummySimulator* SOpticksDummySimulator::Create(const stree* tree) // static
{
    SOpticksDummySimulator* dummy = new SOpticksDummySimulator(tree); ;
    return dummy ;
}



inline SOpticksDummySimulator::SOpticksDummySimulator(const stree* _tree)
    :
    tree(_tree),
    tree_digest(tree ? tree->get_tree_digest() : nullptr)
{
}



// HMM: these are not relevant for Dummy
inline double SOpticksDummySimulator::render_launch(){ return 0. ; }
inline double SOpticksDummySimulator::simtrace_launch(){ return 0. ; }
inline double SOpticksDummySimulator::simulate_launch(){ return 0. ; }
inline double SOpticksDummySimulator::launch(){ return 0. ; }

inline const char* SOpticksDummySimulator::desc() const { return NAME ; }

inline double SOpticksDummySimulator::simtrace(int)
{
    assert(0);
    return 0 ;
}
inline double SOpticksDummySimulator::render(const char*)
{
    assert(0);
    return 0 ;
}

inline double SOpticksDummySimulator::simulate(int, bool)
{
    assert(0);
    return 0. ;
}

inline void SOpticksDummySimulator::reset(int)
{
    assert(0);
}


