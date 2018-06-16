#pragma once

#include <vector>
#include "X4_API_EXPORT.hh"

#include "G4ThreeVector.hh"
#include "G4RotationMatrix.hh"
#include "X4SolidBase.hh"
class G4VSolid ; 
struct nnode ; 

/**
X4Solid
==========

Converts G4VSolid into OpticksCSG nnode trees, the number
of nodes in the tree depends on G4VSolid parameter values, 
eg whether an inner radius greater than zero is set, or phi 
segments are set.

No tree balancing is implemented yet (see ../analytic/csg.py), 
however polycone primitives are hung on a UnionTree and 
the tree is pruned a bit using NTreeBuilder.

TODO
-----

* provide digest methods for each of the ~11 converted G4VSolid, 
  so the geometry digest will notice changes to the solids

**/

class X4_API X4Solid : public X4SolidBase 
{
    struct zplane 
    {
        double rmin ;  
        double rmax ;  
        double z ; 
    };
    public:
        static nnode* Convert(const G4VSolid* solid);
    public:
        X4Solid(const G4VSolid* solid); 
    private:
        void init();
    private:
        void convertBooleanSolid();
        void convertSphere();
        void convertOrb();
        void convertBox();
        void convertTubs();
        void convertTrd();
        void convertCons();
        void convertTorus();
        void convertEllipsoid();
        void convertPolycone();
        void convertHype();
    private:
        nnode* intersectWithPhiSegment(nnode* whole, float startPhi, float deltaPhi, float segZ, float segR );
        void booleanDisplacement( G4VSolid** pp, G4ThreeVector& pos, G4ThreeVector& rot );
        G4ThreeVector GetAngles(const G4RotationMatrix& mtx);
        nnode* convertSphere_(bool only_inner);
        nnode* convertCons_(bool only_inner);
        nnode* convertHype_(bool only_inner);
        void   convertPolyconePrimitives( const std::vector<zplane>& zp,  std::vector<nnode*>& prims );

};

