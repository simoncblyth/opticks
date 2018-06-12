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


/usr/local/opticks/externals/g4/geant4_10_02_p01/source/persistency/gdml/src/G4GDMLWriteSolids.cc


Converting to nnode OR NCSG ?
------------------------------

* NCSG is higher level focusing on tree transport/import/export 
* nnode focuses on shape param, bbox, sdf : and is the base class for 
  eg nbox, ncone, nsphere, nconvexpolyhedron, nunion, ...

Thus nnode is the appropriate target, and NCSG has an nnode ctor : so can be 
obtained later.


Should names has pointer suffixes like GDML ? 
-----------------------------------------------

Probably NO as its a live conversion 
(not via file) from one pointer to another : are not using 
the name as a "pointer" reference. 


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
    private:
        nnode* intersectWithPhiSegment(nnode* whole, float startPhi, float deltaPhi, float segZ, float segR );
        void booleanDisplacement( G4VSolid** pp, G4ThreeVector& pos, G4ThreeVector& rot );
        G4ThreeVector GetAngles(const G4RotationMatrix& mtx);
        nnode* convertSphere_(bool only_inner);
        nnode* convertCons_(bool only_inner);
        void   convertPolyconePrimitives( const std::vector<zplane>& zp,  std::vector<nnode*>& prims );

};

