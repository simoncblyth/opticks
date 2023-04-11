#pragma once
/**
S4RandomMonitor
=================

Instanciation holds onto the current engine in m_engine and 
replaces it with itself.  Flat calls to the engine are 
then passed thru to the original engine and monitoring logging 
is provided.  See S4RandomArray for saving all the randoms. 

NB test which requires Geant4 is in u4/tests/S4RandomMonitorTest.cc

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

struct S4RandomMonitor : public CLHEP::HepRandomEngine
{
    static constexpr const char* NAME = "S4RandomMonitor" ; 
    CLHEP::HepRandomEngine*  m_engine ;

    S4RandomMonitor(); 

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

inline S4RandomMonitor::S4RandomMonitor()
    :
    m_engine(CLHEP::HepRandom::getTheEngine())
{
    CLHEP::HepRandom::setTheEngine(this); 
}


/**
S4RandomMonitor::flat
------------------------

This is the engine method that gets invoked by G4UniformRand calls 

**/

inline double S4RandomMonitor::flat()
{
    double d = m_engine->flat(); 
    std::cerr << "S4RandomMonitor::flat " << d << std::endl ;
    return d ; 
}


/**
S4RandomMonitor::flatArray
------------------------------

This method and several others are required as S4RandomMonitor ISA CLHEP::HepRandomEngine

**/

inline void S4RandomMonitor::flatArray(const int size, double* vect)
{
     assert(0); 
}
inline void S4RandomMonitor::setSeed(long seed, int)
{
    assert(0); 
}
inline void S4RandomMonitor::setSeeds(const long * seeds, int)
{
    assert(0); 
}
inline void S4RandomMonitor::saveStatus( const char filename[]) const 
{
    assert(0); 
}
inline void S4RandomMonitor::restoreStatus( const char filename[]) 
{
    assert(0); 
}
inline void S4RandomMonitor::showStatus() const 
{
    assert(0); 
}
inline std::string S4RandomMonitor::name() const 
{
    return NAME ; 
}

