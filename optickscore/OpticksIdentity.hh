#pragma once

#include <string>
#include "plog/Severity.h"
#include "NGLM.hpp"
#include "OKCORE_API_EXPORT.hh"

/**
OpticksIdentity
=================

Triplet volume/node "RPO" identity comprising three 0-based indices:

R
   repeat_index 
P
   placement_index of the instance
O
   offset_index within the instance

This is the "native" identifier of volumes within the Opticks instance-centric 
geometry model. Contrast this with the node_index which is just a traversal index
over the entire tree. 

The full node tree is labelled with encoded identifiers by GInstancer 


All three indices are encoded into a 32 bit identifier leading to range constraints. 
For repeat_index > 0::

    -- -- -- --
    rr pp pp oo 

    r < 0xff + 1    (256)      256 different repeats 
    p < 0xffff + 1  (65536)    65k placements 
    o < 0xff + 1    (256)      256 volumes within the repeated instance : probably the "tightest" limit (but easily enough for JUNO)

For repeat_index 0 the placement_index is always 0, hence::

    -- -- -- --
    rr oo oo oo 

    r == 0 
    p == 0 
    o < 0xffffff + 1   (16777216)      16.7M global volumes limit  

**/

class OKCORE_API OpticksIdentity {
    public:
        static const plog::Severity LEVEL ;   
        static unsigned    Encode(unsigned repeat_index, unsigned placement_index, unsigned offset_index);
        static bool        Decode(unsigned identifier, unsigned& repeat_index, unsigned& placement_index, unsigned& offset_index );
        static std::string Desc(unsigned identifier);
        static std::string Desc(const char* label, const glm::uvec4& id );

        static unsigned    RepeatIndex(unsigned identifier); 
        static unsigned    PlacementIndex(unsigned identifier); 
        static unsigned    OffsetIndex(unsigned identifier); 
    public:
        OpticksIdentity(unsigned repeat_index, unsigned placement_index, unsigned offset_index); 
        OpticksIdentity(unsigned identifier);
    public:
        std::string desc() const ; 
        unsigned getRepeatIndex() const ; 
        unsigned getPlacementIndex() const ; 
        unsigned getOffsetIndex() const ; 
        unsigned getEncodedIdentifier() const ; 
    private:
        unsigned m_repeat_index ; 
        unsigned m_placement_index ; 
        unsigned m_offset_index ; 
    private:
        unsigned m_encoded_identifier ; 
        bool     m_decoded ; 


};


