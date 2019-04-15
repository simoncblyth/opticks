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

Whilst converting G4Solid to nnode the g4code to instanciate
the solids are collected allowing generation of C++ source 
code to create the G4 solids. This g4code is tacked onto the nnode.  
TODO: see if the g4code survives tree balancing, probably cases 
without one-to-one model match will not work. 

Note that cfg4/CMaker can do the opposite conversion, 
creating a G4VSolid from nnode.

NB the results of X4Solid conversions are **not visible in the glTF**
renders, as those are based on the G4Polyhedron polgonization 
of the solids.  Thus the skipping of G4DisplacedSolid displacement
info will impact the ray trace (and the simulation) but not the glTF.

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
        static void SetVerbosity(unsigned verbosity);
        static nnode* Convert(const G4VSolid* solid, const char* boundary=NULL);
        static nnode* Balance(nnode* raw, unsigned soIdx=0 , unsigned lvIdx=0 );  // cannot be const due to inplace positivization
    public:
        X4Solid(const G4VSolid* solid, bool top); 

        bool hasDisplaced() const ; 
        X4Solid* getDisplaced() const ; 
        void setDisplaced(X4Solid* displaced);
    private:
        void init();
        static unsigned fVerbosity ; 
    private:
        void convertDisplacedSolid();
        void convertUnionSolid();
        void convertIntersectionSolid();
        void convertSubtractionSolid();
    private:
        void convertBooleanSolid();
        void convertSphere();
        static const bool convertSphere_duplicate_py_segment_omission ; 
        void convertOrb();
        void convertBox();
        void convertTubs();
        void convertTrd();
        void convertCons();
        void convertTorus();
        void convertEllipsoid();
        void convertPolycone();
        static const bool convertPolycone_duplicate_py_inner_omission ; 
        void convertHype();
    private:
        static const float hz_disc_cylinder_cut ; 
        nnode* convertTubs_disc();
        nnode* convertTubs_cylinder();
    private:
        nnode* intersectWithPhiSegment(nnode* whole, float startPhi, float deltaPhi, float segZ, float segR );
        void booleanDisplacement( G4VSolid** pp, G4ThreeVector& pos, G4ThreeVector& rot );
        G4ThreeVector GetAngles(const G4RotationMatrix& mtx);
        nnode* convertSphere_(bool only_inner);
        nnode* convertCons_(bool only_inner);
        nnode* convertHype_(bool only_inner);
        void   convertPolyconePrimitives( const std::vector<zplane>& zp,  std::vector<nnode*>& prims );
        void   convertPolycone_g4code();
    private:
        X4Solid* m_displaced ; 

};

