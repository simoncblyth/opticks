#pragma once

#include <glm/glm.hpp>
#include "G4Transform3D.hh"
class G4VSolid ; 

class CSolid {
   public:
       CSolid(const G4VSolid* solid);
       void extent(const G4Transform3D& tran, glm::vec3& low, glm::vec3& high, glm::vec4& center_extent);
   private:
       const G4VSolid* m_solid ; 
};

inline CSolid::CSolid(const G4VSolid* solid) 
   :
      m_solid(solid)
{
}


