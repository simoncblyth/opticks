#pragma once
/**
SSimulator.h : pure virtual interface used from QSim, G4CXOpticks, CSGOptiX
=============================================================================

Note:

1. this protocol interface does not currently depend on any Opticks types,
   making it usable from any level of the package heirarchy

2. ctor and convenience statics such as Create are not part of this interface

**/


struct SSimulator
{
    virtual ~SSimulator() = default ;

    // low level API that enables QSim to control CSGOptiX irrespective of pkg dependency
    virtual double render_launch() = 0 ;
    virtual double simtrace_launch() = 0 ;
    virtual double simulate_launch() = 0 ;
    virtual double launch() = 0 ;

    // informational
    virtual const char* desc() const = 0 ;

    // high level API that G4CXOpticks uses to control for example the CSGOptiX local GPU backend
    virtual double simulate(int eventID, bool reset = false) = 0 ;
    virtual double simtrace(int eventID) = 0 ;
    virtual double render(const char* stem = nullptr) = 0 ;
    virtual void reset(int eventID) = 0 ;

};






