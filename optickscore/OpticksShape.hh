#pragma once

#include <string>
#include "NGLM.hpp"
#include "plog/Severity.h"
#include "OKCORE_API_EXPORT.hh"

class OKCORE_API OpticksShape {
    public:
        static const plog::Severity LEVEL ;   
    public:
        static unsigned Encode( unsigned meshIndex, unsigned boundaryIndex ); 
    public:
        static unsigned MeshIndex(     unsigned shape ); 
        static unsigned BoundaryIndex( unsigned shape ); 
    public:
        static unsigned MeshIndex(     const glm::uvec4& identity); 
        static unsigned BoundaryIndex( const glm::uvec4& identity); 
};


 
