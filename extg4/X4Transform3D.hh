#pragma once

#include <string>
#include <array>
#include "X4_API_EXPORT.hh"
#include "G4Transform3D.hh"
class G4VPhysicalVolume ; 

#include "NGLM.hpp"

/**
X4Transform3D
=================

**/

//#define X4_TRANSFORM_43 1

struct X4_API X4Transform3D
{
    static std::string Digest(const G4Transform3D&  transform); 
    static glm::mat4 Convert(const G4Transform3D&  transform);
    static glm::mat4* GetLocalTransform(const G4VPhysicalVolume* const pv);

    std::array<float, 16>   ar ; 
    glm::mat4               tr ; 

    X4Transform3D(const G4Transform3D&  transform); 
    std::string digest() const ; 

}; 


