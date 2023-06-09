#pragma once 
/**
SCSGOptiX.h
============

Protocol used for example to allow QSim::simulate
to invoke CSGOptiX::simulate without QUDARap package
needing to depend on CSGOptiX package. 

**/

struct SCSGOptiX 
{
    virtual double render() = 0 ;
    virtual double simtrace() = 0 ;
    virtual double simulate() = 0 ;
    virtual double launch() = 0 ;
};


