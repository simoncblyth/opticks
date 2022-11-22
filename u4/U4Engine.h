#pragma once
/**
U4Engine.h
============

Hmm not convenient to persist state into files when have lots of them.  
put/get interface looks better::

    108   virtual std::vector<unsigned long> put () const;
    109   virtual bool get (const std::vector<unsigned long> & v);

Note the inverted API naming, "get" is restoring and "put" is saving. 
For MixMaxRng the vector length is 17*2+4 = 38 and 32 bit unsigned long is OK

**/
#include "CLHEP/Random/Randomize.h"
#include "CLHEP/Random/RandomEngine.h"
#include "U4_API_EXPORT.hh"
#include "spath.h"
#include "NP.hh"

struct U4_API U4Engine
{
    static std::string Desc(); 
    static void ShowStatus(); 
    static std::string ConfPath(const char* fold, const char* name=nullptr); 
    static void SaveStatus(const char* fold, const char* name=nullptr) ; 
    static void RestoreStatus(const char* fold, const char* name=nullptr) ; 

    static void SaveStatus(          NP* states, int idx ); 
    static void RestoreStatus( const NP* states, int idx ); 

}; 

/**
U4Engine::Desc
-----------------

Trying to introspect the engine::

    g4-cls Randomize
    g4-cls MixMaxRng   ## by inspection the default engine in 10.4.2
    g4-cls RandomEngine

**/

inline std::string U4Engine::Desc() // static
{
    CLHEP::HepRandomEngine* engine = CLHEP::HepRandom::getTheEngine() ; 
    std::stringstream ss ; 
    ss << "U4Engine::Desc" 
       << " engine " << ( engine ? "Y" : "N" )
       << " engine.name " << ( engine ? engine->name() : "-" )
       ; 
    std::string s = ss.str() ; 
    return s ; 
}
inline void U4Engine::ShowStatus() // static
{
    CLHEP::HepRandomEngine* engine = CLHEP::HepRandom::getTheEngine() ; 
    engine->showStatus(); 
}

inline std::string U4Engine::ConfPath(const char* fold, const char* name) // static
{
    CLHEP::HepRandomEngine* engine = CLHEP::HepRandom::getTheEngine() ; 
    std::string defconf = engine->name() + "State.conf" ;
    if(name == nullptr) name = defconf.c_str() ; 
    std::string path = spath::Join(fold, name) ;  
    return path ;    
}

inline void U4Engine::SaveStatus(const char* fold, const char* name) // status
{
    std::string conf = ConfPath(fold, name) ;  
    std::cout << "U4Engine::SaveStatus " << conf << std::endl ;  
    CLHEP::HepRandomEngine* engine = CLHEP::HepRandom::getTheEngine() ; 
    engine->saveStatus(conf.c_str()) ; 
}

inline void U4Engine::RestoreStatus(const char* fold, const char* name) // status
{
    std::string conf = ConfPath(fold, name) ;  
    std::cout << "U4Engine::RestoreStatus " << conf << std::endl ;  
    CLHEP::HepRandomEngine* engine = CLHEP::HepRandom::getTheEngine() ; 
    engine->restoreStatus(conf.c_str()) ; 
}

inline void U4Engine::SaveStatus( NP* states, int idx ) // static
{
    assert( states && states->shape.size() > 0 && idx < states->shape[0] ); 

    CLHEP::HepRandomEngine* engine = CLHEP::HepRandom::getTheEngine() ; 
    std::vector<unsigned long> state = engine->put() ; // NB: back-to-front API name 
    unsigned niv = states->num_itemvalues() ; 
    unsigned size = states->item_bytes();
    assert( niv*sizeof(unsigned long) == size ); 

    assert( state.size() == niv ); 
    memcpy( states->bytes() + size*idx,  state.data(), size  ); 
}

inline void U4Engine::RestoreStatus( const NP* states, int idx ) // static
{
    assert( states && states->shape.size() > 0 && idx < states->shape[0] ); 

    unsigned niv = states->num_itemvalues() ; 
    std::vector<unsigned long> state ; 
    states->slice(state, idx, -1 );  // -1: terminates the dimensions to select on  
    assert( state.size() == niv ); 

    CLHEP::HepRandomEngine* engine = CLHEP::HepRandom::getTheEngine() ; 
    
    bool rc = engine->get(state);  // HUH: back-to-front API name ?
    assert( rc ); 
}

