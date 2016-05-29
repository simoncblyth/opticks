#pragma once

#include "G4Transform3D.hh"
#include "G4AffineTransform.hh"

class CMath {
   public:
      static G4AffineTransform make_affineTransform(const G4Transform3D& T );

};


