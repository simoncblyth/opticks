#pragma once
/**
U4RandomMonitor
=================
**/

#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cstdlib>
#include <csignal>
#include <cassert>

#include "Randomize.hh"
#include "G4Types.hh"
#include "CLHEP/Random/RandomEngine.h"
#include "U4_API_EXPORT.hh"

struct U4_API U4RandomMonitor : public CLHEP::HepRandomEngine
{
    static constexpr const char* NAME = "U4RandomMonitor" ; 
    static std::string Desc(); 

    U4RandomMonitor(); 

    CLHEP::HepRandomEngine*  m_engine ;

    // mandatory CLHEP::HepRandomEngine methods
    double flat();
    void flatArray(const int size, double* vect);
    void setSeed(long seed, int);
    void setSeeds(const long * seeds, int); 
    void saveStatus( const char filename[] = "Config.conf") const ;
    void restoreStatus( const char filename[] = "Config.conf" ) ;
    void showStatus() const ;
    std::string name() const ;


}; 

inline U4RandomMonitor::U4RandomMonitor()
    :
    m_engine(CLHEP::HepRandom::getTheEngine())
{
    CLHEP::HepRandom::setTheEngine(this); 
}


/**
U4RandomMonitor::flat
------------------------

This is the engine method that gets invoked by G4UniformRand calls 

**/

inline double U4RandomMonitor::flat()
{
    double d = m_engine->flat(); 
    std::cerr << "U4RandomMonitor::flat " << d << std::endl ;
    return d ; 
}


/**
U4RandomMonitor::flatArray
------------------------------

This method and several others are required as U4RandomMonitor ISA CLHEP::HepRandomEngine

**/

inline void U4RandomMonitor::flatArray(const int size, double* vect)
{
     assert(0); 
}
inline void U4RandomMonitor::setSeed(long seed, int)
{
    assert(0); 
}
inline void U4RandomMonitor::setSeeds(const long * seeds, int)
{
    assert(0); 
}
inline void U4RandomMonitor::saveStatus( const char filename[]) const 
{
    assert(0); 
}
inline void U4RandomMonitor::restoreStatus( const char filename[]) 
{
    assert(0); 
}
inline void U4RandomMonitor::showStatus() const 
{
    assert(0); 
}
inline std::string U4RandomMonitor::name() const 
{
    return NAME ; 
}

