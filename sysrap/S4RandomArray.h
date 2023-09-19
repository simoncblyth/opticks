#pragma once
/**
S4RandomArray : Uses Current Engine, but collects the randoms
===============================================================

Instanciation holds onto the current engine in m_engine and 
replaces it with itself.  Flat calls to the engine are 
then passed thru to the original engine and each random 
is added to a vector.  See S4RandomMonitor.h 
for simply logging the randoms. 

NB tests which require Geant4 in u4/tests/U4RandomArrayTest.cc

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
#include "NPX.h"

struct S4RandomArray : public CLHEP::HepRandomEngine
{
    static constexpr const char* NAME = "S4RandomArray.npy" ; 

    std::vector<double>      m_array ;  
    CLHEP::HepRandomEngine*  m_engine ;

    S4RandomArray(); 
    void clear(); 
    NP* serialize() const ;  

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

inline S4RandomArray::S4RandomArray()
    :
    m_engine(CLHEP::HepRandom::getTheEngine())
{
    CLHEP::HepRandom::setTheEngine(this); 
}

inline void S4RandomArray::clear()
{
    m_array.clear(); 
}
inline NP* S4RandomArray::serialize() const 
{
    return NPX::ArrayFromVec<double, double>(m_array) ; 
}


/**
S4RandomArray::flat
------------------------

This is the engine method that gets invoked by G4UniformRand calls 

**/

inline double S4RandomArray::flat()
{
    double d = m_engine->flat(); 
    m_array.push_back(d) ; 
    return d ; 
}

/**
S4RandomArray::flatArray
-------------------------

G4VEnergyLossProcess::AlongStepDoIt/G4UniversalFluctuation::SampleFluctuations needs this

**/

inline void S4RandomArray::flatArray(const int size, double* vect)
{
    m_engine->flatArray(size, vect); 
    for(int i=0 ; i < size ; i++) m_array.push_back(vect[i]) ; 
}


/**
S4RandomArray::flatArray
------------------------------

This method and several others are required as S4RandomArray ISA CLHEP::HepRandomEngine

**/


inline void S4RandomArray::setSeed(long seed, int)
{
    assert(0); 
}
inline void S4RandomArray::setSeeds(const long * seeds, int)
{
    assert(0); 
}
inline void S4RandomArray::saveStatus( const char filename[]) const 
{
    assert(0); 
}
inline void S4RandomArray::restoreStatus( const char filename[]) 
{
    assert(0); 
}
inline void S4RandomArray::showStatus() const 
{
    assert(0); 
}
inline std::string S4RandomArray::name() const 
{
    return NAME ; 
}

