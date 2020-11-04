#pragma once
#include <string>
#include <glm/fwd.hpp>
#include "plog/Severity.h"
#include "OKCORE_API_EXPORT.hh"

/**
OpticksPhotonFlags 
===================

Encapsulate the quad of photon flags written into GPU photon buffer 
by optixrap/cu/generate.cu

The whole flags Quad is passed to the static functions to 
ease future rearrangments of flags 

NB sensorIndex being int wastes half the bits just for -1 to mean not-a-hit. 
For a detector with more that 0x7fff (32767) sensors should move to 1-based 
sensor index and 0 means not-a-hit to give up to 0xffff (65535) sensors.  


**/

struct OKCORE_API OpticksPhotonFlags 
{
    static const plog::Severity LEVEL ; 
    union uif_t {
        unsigned u ; 
        int      i ; 
        float    f ; 
    };  
     
    static int      Boundary(   const float& x , const float&   , const float&   , const float&   );
    static int      SensorIndex(const float& x , const float&   , const float&   , const float&   );
    static unsigned NodeIndex(  const float&,    const float& y , const float&   , const float&   );
    static unsigned PhotonIndex(const float&,    const float&   , const float& z , const float&   );
    static unsigned FlagMask(   const float&,    const float&   , const float&   , const float& w );

    static int      Boundary(   const glm::vec4& f ); 
    static int      SensorIndex(const glm::vec4& f );
    static unsigned NodeIndex(  const glm::vec4& f );
    static unsigned PhotonIndex(const glm::vec4& f );
    static unsigned FlagMask(   const glm::vec4& f );

    int      boundary ; 
    int      sensorIndex ; 
    unsigned nodeIndex ; 
    unsigned photonIndex ; 
    unsigned flagMask ; 

    OpticksPhotonFlags( const glm::vec4& f );
    OpticksPhotonFlags( int boundary, int sensorIndex, unsigned nodeIndex, unsigned photonIndex, unsigned flagMask ); 
    std::string desc() const ;
    std::string brief() const ;

    bool operator==(const OpticksPhotonFlags& rhs) const ;
};


