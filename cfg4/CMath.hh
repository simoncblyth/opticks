#pragma once

#include "G4Transform3D.hh"
#include "G4AffineTransform.hh"
#include "CFG4_API_EXPORT.hh"

class CFG4_API CMath {
   public:
      static G4AffineTransform make_affineTransform(const G4Transform3D& T );

};


