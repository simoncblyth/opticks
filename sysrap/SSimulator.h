#pragma once

#include <string>
struct CSGFoundry ;

struct SSimulator
{
    virtual ~SSimulator() = default ;

    virtual double render_launch() = 0 ;
    virtual double simtrace_launch() = 0 ;
    virtual double simulate_launch() = 0 ;
    virtual double launch() = 0 ;

    virtual const char* desc() const = 0 ;

    virtual double simulate(int eventID, bool reset = false) = 0 ; 
    virtual double simtrace(int eventID) = 0 ;
    virtual double render(const char* stem = nullptr) = 0 ;

    virtual void reset(int eventID) = 0 ;

    static SSimulator* Create(CSGFoundry* foundry );
};


inline SSimulator* SSimulator::Create(CSGFoundry* )
{
    return nullptr ;
}





