CX4GDMLTest_soIdx_126_gltf
=============================

Attempt to phi segment a sphere with deltaPhi 180, fails : because 
the segment degenerates to a plane for that angle.

Approaches to fix:

1. identify degenerate case  : gives non-sensical -ve plane distances from origin
2. swap to zsphere ? 

   * not a solution, sometimes need zslice and phi segm, also 
     same problem for other shapes
 
3. OR : intersect instead with a box dimensioned appropriately based on slightly enlarged 
   bbox of what are chopping up 



CX4GDMLTest
-------------

::


    2018-06-21 17:16:25.034 INFO  [24225540] [*X4PhysicalVolume::convertNode@404] convertNode  ndIdx 4567 soIdx 126
    2018-06-21 17:16:25.034 INFO  [24225540] [X4Solid::init@57] X4SolidBase name         AmCCo60AcrylicContainer0xc0b23b8 entityType 1 entityName G4UnionSolid root 0x0
    2018-06-21 17:16:25.034 INFO  [24225540] [X4Solid::init@57] X4SolidBase name AcrylicCylinder+ChildForAmCCo60AcrylicContainer0xc0b1f38 entityType 1 entityName G4UnionSolid root 0x0
    2018-06-21 17:16:25.034 INFO  [24225540] [X4Solid::init@57] X4SolidBase name                 AcrylicCylinder0xc0b22c0 entityType 25 entityName G4Tubs root 0x0
    2018-06-21 17:16:25.034 INFO  [24225540] [X4Solid::convertTubs@359] 
    -----------------------------------------------------------
        *** Dump for solid - AcrylicCylinder0xc0b22c0 ***
        ===================================================
     Solid type: G4Tubs
     Parameters: 
        inner radius : 0 mm 
        outer radius : 10.035 mm 
        half length Z: 14.865 mm 
        starting phi : 0 degrees 
        delta phi    : 360 degrees 
    -----------------------------------------------------------

    2018-06-21 17:16:25.034 INFO  [24225540] [X4Solid::init@57] X4SolidBase name                                  placedB entityType 0 entityName G4DisplacedSolid root 0x0
    2018-06-21 17:16:25.034 INFO  [24225540] [X4Solid::init@57] X4SolidBase name          UpperAcrylicHemisphere0xc0b2ac0 entityType 18 entityName G4Sphere root 0x0
    2018-06-21 17:16:25.034 INFO  [24225540] [X4Solid::convertSphere@308] 
    -----------------------------------------------------------
        *** Dump for solid - UpperAcrylicHemisphere0xc0b2ac0 ***
        ===================================================
     Solid type: G4Sphere
     Parameters: 
        inner radius: 0 mm 
        outer radius: 10.035 mm 
        starting phi of segment  : 0 degrees 
        delta phi of segment     : 180 degrees 
        starting theta of segment: 0 degrees 
        delta theta of segment   : 180 degrees 
    -----------------------------------------------------------

    2018-06-21 17:16:25.034 INFO  [24225540] [*X4Solid::convertSphere_@218]  radius : 10.035 only_inner : 0 has_inner : 0
    2018-06-21 17:16:25.034 INFO  [24225540] [*X4Solid::convertSphere_@234]  rTheta : 0 lTheta : 180 zslice : 0
    2018-06-21 17:16:25.034 INFO  [24225540] [X4Solid::init@57] X4SolidBase name                                  placedB entityType 0 entityName G4DisplacedSolid root 0x0
    2018-06-21 17:16:25.034 INFO  [24225540] [X4Solid::init@57] X4SolidBase name          LowerAcrylicHemisphere0xc0b2be8 entityType 18 entityName G4Sphere root 0x0
    2018-06-21 17:16:25.034 INFO  [24225540] [X4Solid::convertSphere@308] 
    -----------------------------------------------------------
        *** Dump for solid - LowerAcrylicHemisphere0xc0b2be8 ***
        ===================================================
     Solid type: G4Sphere
     Parameters: 
        inner radius: 0 mm 
        outer radius: 10.035 mm 
        starting phi of segment  : 0 degrees 
        delta phi of segment     : 180 degrees 
        starting theta of segment: 0 degrees 
        delta theta of segment   : 180 degrees 
    -----------------------------------------------------------

    2018-06-21 17:16:25.034 INFO  [24225540] [*X4Solid::convertSphere_@218]  radius : 10.035 only_inner : 0 has_inner : 0
    2018-06-21 17:16:25.034 INFO  [24225540] [*X4Solid::convertSphere_@234]  rTheta : 0 lTheta : 180 zslice : 0
    2018-06-21 17:16:25.035 INFO  [24225540] [X4Mesh::polygonize@125] v 342 f 408 cout 0 cerr 0 
    2018-06-21 17:16:25.035 INFO  [24225540] [NNodeNudger::update_prim_bb@37] NNodeNudger::update_prim_bb nprim 5
    Assertion failed: (plane.w >= 0.f), function operator(), file /Users/blyth/opticks-cmake-overhaul/npy/NConvexPolyhedron.cpp, line 34.
    Abort trap: 6
    epsilon:tests blyth$ 
