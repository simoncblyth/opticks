#pragma once
/**
U4RandomArray
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
#include "NPX.h"

struct U4RandomArray : public CLHEP::HepRandomEngine
{
    static constexpr const char* NAME = "U4RandomArray" ; 

    std::vector<double>      m_array ;  
    CLHEP::HepRandomEngine*  m_engine ;

    U4RandomArray(); 
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

inline U4RandomArray::U4RandomArray()
    :
    m_engine(CLHEP::HepRandom::getTheEngine())
{
    CLHEP::HepRandom::setTheEngine(this); 
}

inline void U4RandomArray::clear()
{
    m_array.clear(); 
}
inline NP* U4RandomArray::serialize() const 
{
    return NPX::ArrayFromVec<double, double>(m_array) ; 
}


/**
U4RandomArray::flat
------------------------

This is the engine method that gets invoked by G4UniformRand calls 

**/

inline double U4RandomArray::flat()
{
    double d = m_engine->flat(); 
    m_array.push_back(d) ; 
    return d ; 
}


/**
U4RandomArray::flatArray
------------------------------

This method and several others are required as U4RandomArray ISA CLHEP::HepRandomEngine

**/

inline void U4RandomArray::flatArray(const int size, double* vect)
{
     assert(0); 
}
inline void U4RandomArray::setSeed(long seed, int)
{
    assert(0); 
}
inline void U4RandomArray::setSeeds(const long * seeds, int)
{
    assert(0); 
}
inline void U4RandomArray::saveStatus( const char filename[]) const 
{
    assert(0); 
}
inline void U4RandomArray::restoreStatus( const char filename[]) 
{
    assert(0); 
}
inline void U4RandomArray::showStatus() const 
{
    assert(0); 
}
inline std::string U4RandomArray::name() const 
{
    return NAME ; 
}

