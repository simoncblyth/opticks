#include "CFG4_BODY.hh"
#include "CMath.hh"

#include "G4RotationMatrix.hh"
#include "G4ThreeVector.hh"

G4AffineTransform CMath::make_affineTransform(const G4Transform3D& T )
{
    G4ThreeVector colX(T.xx(), T.xy(), T.xz());
    G4ThreeVector colY(T.yx(), T.yy(), T.yz());
    G4ThreeVector colZ(T.zx(), T.zy(), T.zz());

    G4RotationMatrix rot(colX,colY,colZ) ;
    G4ThreeVector tlate(T.dx(), T.dy(), T.dz());

    return G4AffineTransform( rot, tlate) ; 
}

