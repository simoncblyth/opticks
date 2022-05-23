#pragma once 

struct SCSGOptiX 
{
    virtual double render() = 0 ;
    virtual double simtrace() = 0 ;
    virtual double simulate() = 0 ;
    virtual double launch() = 0 ;
};


