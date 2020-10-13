#include <cassert>
#include <sstream>
#include "PLOG.hh"
#include "SPack.hh"
#include "OpticksShape.hh"

const plog::Severity OpticksShape::LEVEL = PLOG::EnvLevel("OpticksShape", "DEBUG"); 


unsigned OpticksShape::Encode( unsigned meshIndex, unsigned boundaryIndex )
{
    return SPack::Encode22( meshIndex, boundaryIndex );
}

unsigned OpticksShape::MeshIndex(const glm::uvec4& identity)
{
    return MeshIndex(identity.z);  
}
unsigned OpticksShape::BoundaryIndex(const glm::uvec4& identity)
{
    return BoundaryIndex(identity.z);  
}

unsigned OpticksShape::MeshIndex(unsigned shape )
{
    return SPack::Decode22a(shape);  
} 
unsigned OpticksShape::BoundaryIndex(unsigned shape )
{
    return SPack::Decode22b(shape);  
} 






