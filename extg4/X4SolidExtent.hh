#pragma once

#include <glm/fwd.hpp>

class G4VSolid ; 
struct nbbox ; 

#include "G4Transform3D.hh"
#include "X4_API_EXPORT.hh"

/**
X4SolidExtent
===============

Duplicate of CFG4.CSolid as 10.4.2 is crashing in G4VisExtent

**/

class X4_API X4SolidExtent {

   public:
       static nbbox* Extent( const G4VSolid* solid ); 
   public:
       X4SolidExtent(const G4VSolid* solid);
       void extent(const G4Transform3D& tran, glm::vec3& low, glm::vec3& high, glm::vec4& center_extent);
   private:
       const G4VSolid* m_solid ; 
};


