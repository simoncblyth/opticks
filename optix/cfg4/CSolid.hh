#pragma once
#include "G4Transform3D.hh"

class G4VSolid ; 

class CSolid {
   public:
       CSolid(G4VSolid* solid);
       void extent(const G4Transform3D& tran);
   private:
       G4VSolid* m_solid ; 
};

inline CSolid::CSolid(G4VSolid* solid) 
   :
      m_solid(solid)
{
}


