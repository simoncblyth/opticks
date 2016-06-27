#pragma once

#include <glm/fwd.hpp>

class G4VSolid ; 
#include "G4Transform3D.hh"
#include "CFG4_API_EXPORT.hh"
class CFG4_API CSolid {
   public:
       CSolid(const G4VSolid* solid);
       void extent(const G4Transform3D& tran, glm::vec3& low, glm::vec3& high, glm::vec4& center_extent);
   private:
       const G4VSolid* m_solid ; 
};


