#pragma once
/**
S4Random.h
============

For all the bells and whistles use U4Random.hh

S4Random.h aims to be a minimal headeronly alternative 
intended for use from standalone tests.

**/

#include "CLHEP/Random/RandomEngine.h"
#include "Randomize.hh"
#include "s_seq.h"

struct S4Random : public CLHEP::HepRandomEngine
{
    static constexpr const char* NAME = "S4Random" ; 

    // assert-ing methods : but mandatory for CLHEP::HepRandomEngine
    void flatArray(const int size, double* vect){ assert(0); }
    void setSeed(long seed, int){ assert(0); }
    void setSeeds(const long * seeds, int){ assert(0); }
    void saveStatus( const char filename[]) const{ assert(0); }
    void restoreStatus( const char filename[]){ assert(0); }
    void showStatus() const { assert(0); }
    std::string name() const { return NAME ; }

    S4Random(); 

    std::string desc() const ; 
    double      flat();
    std::string demo(int n) ; 
    void        setSequenceIndex(int index_);   // -ve to disable, must be less than ni  

private:
    void enable();
    void disable();

    s_seq*                   m_seq ; 
    CLHEP::HepRandomEngine*  m_default ;

};

inline S4Random::S4Random()
    :
    m_seq(new s_seq),
    m_default(CLHEP::HepRandom::getTheEngine())
{
}

inline std::string S4Random::desc() const { return m_seq->desc() ; }
inline double      S4Random::flat(){        return m_seq->flat() ; }
inline std::string S4Random::demo(int n) {  return m_seq->demo(n) ; }


/**
S4Random::setSequenceIndex
--------------------------------

Switches random stream when index is not negative.
This is used for example to switch between the separate streams 
used for each photon.

A negative index disables the control of the Geant4 random engine.  

**/

inline void S4Random::setSequenceIndex(int index_)
{
    m_seq->setSequenceIndex(index_); 

    if(index_ < 0 ) 
    {
        disable() ; 
    }
    else 
    {
        enable() ; 
    }
}

/**
S4Random::enable 
-----------------

Invokes CLHEP::HepRandom::setTheEngine to *this* U4Random instance 
which means that all subsequent calls to G4UniformRand will provide pre-cooked 
randoms from the stream controlled by *U4Random::setSequenceIndex*

**/
inline void S4Random::enable(){  CLHEP::HepRandom::setTheEngine(this); }

/**
S4Random::disable
------------------

Returns to the engine active at instanciation, typically the default engine.

**/

inline void S4Random::disable(){ CLHEP::HepRandom::setTheEngine(m_default); }



