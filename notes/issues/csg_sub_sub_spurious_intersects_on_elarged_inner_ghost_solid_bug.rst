csg_sub_sub_spurious_intersects_on_elarged_inner_ghost_solid_bug
===================================================================

* https://simoncblyth.bitbucket.io/env/presentation/opticks_autumn_20211019.html

* **A-(B-C)**


Search for experience
----------------------

* :google:`CSG coincident faces`


Old Paper 
------------


* https://www.cc.gatech.edu/~jarek/papers/DIBs.pdf  

  Rossignac and Wu
  Correct Shading of Regularized CSG Solids Using a Depth-Interval Buffer 



Pay wall paper abstract : makes me think about simplifying cylinder intersects
----------------------------------------------------------------------------------

::

    L. A. Whitehead, "Simplified ray tracing in cylindrical systems," Appl. Opt. 21, 3536-3538 (1982) 
    https://www.osapublishing.org/ao/abstract.cfm?URI=ao-21-19-3536

    Abstract

    A simplified method of ray tracing in cylindrical optical systems of arbitrary
    cross section is described. The technique involves the projection of a ray’s
    path onto a cross-sectional plane perpendicular to the axis of translational
    symmetry. It is shown that this projected path obeys a generalized form of
    Snell’s law, enabling application of conventional 2-D ray tracing methods. The
    approach is illustrated by demonstrating the optical characteristics of the
    recently patented prism light guide.



Can cylinder features be used to optimize intersects
-------------------------------------------------------

* difference between an infinite cylinder and a finite one is very simple

  * treat it as infinite and then z-cut candidate intersects


Seems nothing more than are doing already.

* https://gamedev.stackexchange.com/questions/191328/how-to-perform-a-fast-ray-cylinder-intersection-test


CSG/csg_intersect_node.h:intersect_node_cylinder
--------------------------------------------------

Already unwieldy. Trying to do cylinder rmin within the primitive certaintly possible, but significant effort. 



Godot Engine Has CSG Imp, see CSGCombiner : but its for meshes
----------------------------------------------------------------------

* https://github.com/godotengine/godot/issues/21125
* https://github.com/godotengine/godot/search?q=CSG&type=

* https://docs.godotengine.org/en/stable/index.html
* https://docs.godotengine.org/en/stable/tutorials/3d/csg_tools.html
* https://docs.godotengine.org/en/stable/classes/class_csgtorus.html#class-csgtorus


CSG Searching : see ~/env/graphics/csg/csg.bash
--------------------------------------------------

::

   csg-vi 




Geant4 Documentation
-----------------------

* https://geant4-userdoc.web.cern.ch/UsersGuides/ForApplicationDeveloper/html/GettingStarted/geometryDef.html

Each volume is created by describing its shape and its physical
characteristics, and then placing it inside a containing volume.

* https://geant4-userdoc.web.cern.ch/UsersGuides/ForApplicationDeveloper/html/Detector/Geometry/geomSolids.html#geom-solids-boolop

The constituent solids of a Boolean operation should possibly avoid be composed
by sharing all or part of their surfaces. This precaution is necessary in order
to avoid the generation of ‘fake’ surfaces due to precision loss, or errors in
the final visualization of the Boolean shape. In particular, if any one of the
subtractor surfaces is coincident with a surface of the subtractee, the result
is undefined. Moreover, the final Boolean solid should represent a single
‘closed’ solid, i.e. a Boolean operation between two solids which are disjoint
or far apart each other, is not a valid Boolean composition.


G4Tubs
--------

::

    g4-cls G4Tubs
    g4-cls G4CSGSolid   # does little
    g4-cls G4VSolid


    class G4Tubs : public G4CSGSolid
    class G4CSGSolid : public G4VSolid


How to test as change G4 geometry modelling 
---------------------------------------------

* CSG_GGeo can be used to easily do partial geocache to CX-CSG geometry conversions : but that is starting from geocache GParts 
  which is hardly an editing format  

* "--g4codegen" option uses x4gen to generate executables with G4 solid construction code for every solid, see X4CSG 

  * but then how to use this 
  * access to all solids used in the geometry and variations thereof needs to be just a static call away "X4Gen::GetSolid(const char*)"
  * need some interface to enable doing G4VSolid implementation swaps 

    * editing GDML is too manual  
    * look into the X4 geometry conversion workflow and see how could do the swap in the conversion

* the old geotest machinery diverged too much from mainline geometry workflow, so its too much work to keep it going  


Swapping G4VSolid in conversion ? 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* objective is to easily test variations in a solid implementation 
* hmm could effect a solid swap in X4PhysicalVolume::convertSolids_r
* but thats not so useful as needs to change volumes too.. : so have to go back to offline and a python switch 

::

     956 void X4PhysicalVolume::convertSolids_r(const G4VPhysicalVolume* const pv, int depth)
     957 {
     958     const G4LogicalVolume* lv = pv->GetLogicalVolume() ;
     959 
     960     // G4LogicalVolume::GetNoDaughters returns 1042:G4int, 1062:size_t
     961     for (size_t i=0 ; i < size_t(lv->GetNoDaughters()) ;i++ )
     962     {
     963         const G4VPhysicalVolume* const daughter_pv = lv->GetDaughter(i);
     964         convertSolids_r( daughter_pv , depth + 1 );
     965     }
     966 
     967     // for newly encountered lv record the tail/postorder idx for the lv
     968     if(std::find(m_lvlist.begin(), m_lvlist.end(), lv) == m_lvlist.end())
     969     {
     970         int lvIdx = m_lvlist.size();
     971         int soIdx = lvIdx ; // when converting in postorder soIdx is the same as lvIdx
     972         m_lvidx[lv] = lvIdx ;
     973         m_lvlist.push_back(lv);
     974 
     975         const G4VSolid* const solid = lv->GetSolid();
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ swap in alternative here 

     976         const std::string& lvname = lv->GetName() ;
     977         const std::string& soname = solid->GetName() ;
     978 



Questions 
----------

1. does the problem manifest in a simpler case, eg box - pipeCylinder ?
2. what happens to the spurious intersects when skip enlarge nudging the inner cylinder ?

These can be answered without returning all the way to Geant4 geometry, so do not 
need to start from CSG_GGeo can just operate in CSG. 


CSGDemo Workflow with *dcyl* and *bssc* demo solid
-------------------------------------------------------

::

   cd ~/opticks/CSG       ## create foundry geometry and persist 
   ./CSGDemoTest.sh    

   cd ~/opticks/CSGOptiX  ## create jpg image ray trace render of geometry 
   ./cxr_demo.sh 


* surprised to find bssc:"box-(cyl-cyl)" not showing spurious intersects

  * maybe that points the finger of suspicion at tree balancing ?


::

    092 AdditionAcrylicConstruction::makeAdditionLogical(){
    108         double ZNodes3[3];
    109         double RminNodes3[3];
    110         double RmaxNodes3[3];
    111         ZNodes3[0] = 5.7*mm; RminNodes3[0] = 0*mm; RmaxNodes3[0] = 450.*mm;
    112         ZNodes3[1] = 0.0*mm; RminNodes3[1] = 0*mm; RmaxNodes3[1] = 450.*mm;
    113         ZNodes3[2] = -140.0*mm; RminNodes3[2] = 0*mm; RmaxNodes3[2] = 200.*mm;
    115         solidAddition_down = new G4Polycone("solidAddition_down",0.0*deg,360.0*deg,3,ZNodes3,RminNodes3,RmaxNodes3);
    ...
    122     solidAddition_up = new G4Sphere("solidAddition_up",0*mm,17820*mm,0.0*deg,360.0*deg,0.0*deg,180.*deg); 
    123
    124     uni_acrylic1 = new G4SubtractionSolid("uni_acrylic1",solidAddition_down,solidAddition_up,0,G4ThreeVector(0*mm,0*mm,+17820.0*mm));
    125
    126     solidAddition_up1 = new G4Tubs("solidAddition_up1",120*mm,208*mm,15.2*mm,0.0*deg,360.0*deg);
    127     uni_acrylic2 = new G4SubtractionSolid("uni_acrylic2",uni_acrylic1,solidAddition_up1,0,G4ThreeVector(0.*mm,0.*mm,-20*mm));
    128     solidAddition_up2 = new G4Tubs("solidAddition_up2",0,14*mm,52.5*mm,0.0*deg,360.0*deg);
    130     for(int i=0;i<8;i++)
    131     {
    132     uni_acrylic3 = new G4SubtractionSolid("uni_acrylic3",uni_acrylic2,solidAddition_up2,0,G4ThreeVector(164.*cos(i*pi/4)*mm,164.*sin(i*pi/4)*mm,-87.5));
    133     uni_acrylic2 = uni_acrylic3;
    135     }




::

    solidAddition_down :   union( smallCyl, bigCone ) 

    solidAddition_up   :   bigSphere 

    uni_acrylic1       :   difference( solidAddition_down,  solidAddition_up  )

    solidAddition_up1  :   difference( outerCyl, innerCyl )

    uni_acrylic2       :   difference( uni_acrylic1 , solidAddition_up1   )

    solidAddition_up2  :   cylCavity

    uni_acrylic3       :   difference(  uni_acrylic2, cylCavity ) 



How to proceed?
-----------------

* need to examine the tree structure of the actual uni_acrylic3 eg render all the primitives


::

    geocache-29aug2021

    cg 
    ./run.sh  


    2021-10-22 20:56:13.510 INFO  [4536280] [*CSG_GGeo_Convert::convertSolid@220]  repeatIdx 8 nmm 10 numPrim(GParts.getNumPrim) 1 rlabel r8 num_inst 590 dump_ridx 8 dump 1
    CSG_GGeo_Convert::convertPrim primIdx 0 numPrim 1 numParts 31 meshIdx 96 last_ridx 8 dump 1
      0 CSGNode     0  in aabb:    -0.0    -0.0    -0.0     0.0     0.0     0.0  trIdx:     0 atm     0 IsOnlyIntersectionMask 0 is_complemented_leaf 0 bbskip 0
      1 CSGNode     1  in aabb:    -0.0    -0.0    -0.0     0.0     0.0     0.0  trIdx:     0 atm     4 IsOnlyIntersectionMask 1 is_complemented_leaf 0 bbskip 0
      2 CSGNode     2  in aabb:    -0.0    -0.0    -0.0     0.0     0.0     0.0  trIdx:     0 atm     4 IsOnlyIntersectionMask 1 is_complemented_leaf 0 bbskip 0
      3 CSGNode     3  in aabb:    -0.0    -0.0    -0.0     0.0     0.0     0.0  trIdx:     0 atm     4 IsOnlyIntersectionMask 1 is_complemented_leaf 0 bbskip 0
      4 CSGNode     4  in aabb:    -0.0    -0.0    -0.0     0.0     0.0     0.0  trIdx:     0 atm     4 IsOnlyIntersectionMask 1 is_complemented_leaf 0 bbskip 0
      5 CSGNode     5  in aabb:    -0.0    -0.0    -0.0     0.0     0.0     0.0  trIdx:     0 atm     4 IsOnlyIntersectionMask 1 is_complemented_leaf 0 bbskip 0
      6 CSGNode     6  un aabb:    -0.0    -0.0    -0.0     0.0     0.0     0.0  trIdx:     0 atm     4 IsOnlyIntersectionMask 1 is_complemented_leaf 0 bbskip 0
      7 CSGNode     7  in aabb:    -0.0    -0.0    -0.0     0.0     0.0     0.0  trIdx:     0 atm     4 IsOnlyIntersectionMask 1 is_complemented_leaf 0 bbskip 0
      8 CSGNode     8  in aabb:    -0.0    -0.0    -0.0     0.0     0.0     0.0  trIdx:     0 atm     4 IsOnlyIntersectionMask 1 is_complemented_leaf 0 bbskip 0
      9 CSGNode     9  in aabb:    -0.0    -0.0    -0.0     0.0     0.0     0.0  trIdx:     0 atm     4 IsOnlyIntersectionMask 1 is_complemented_leaf 0 bbskip 0
     10 CSGNode    10  in aabb:    -0.0    -0.0    -0.0     0.0     0.0     0.0  trIdx:     0 atm     4 IsOnlyIntersectionMask 1 is_complemented_leaf 0 bbskip 0
     11 CSGNode    11 !cy aabb:   102.0  -130.0  -140.0   130.0  -102.0   -35.0  trIdx:  8063 atm     4 IsOnlyIntersectionMask 1 is_complemented_leaf 1 bbskip 1
     12 CSGNode    12  un aabb:    -0.0    -0.0    -0.0     0.0     0.0     0.0  trIdx:     0 atm     4 IsOnlyIntersectionMask 1 is_complemented_leaf 0 bbskip 0
     13 CSGNode    13 !cy aabb:  -208.0  -208.0   -35.2   208.0   208.0    -4.8  trIdx:  8064 atm     6 IsOnlyIntersectionMask 0 is_complemented_leaf 1 bbskip 0
     14 CSGNode    14  cy aabb:  -120.0  -120.0   -35.4   120.0   120.0    -4.6  trIdx:  8065 atm     6 IsOnlyIntersectionMask 0 is_complemented_leaf 0 bbskip 0
     15 CSGNode    15 !sp aabb: -17820.0 -17820.0     0.0 17820.0 17820.0 35640.0  trIdx:  8066 atm     4 IsOnlyIntersectionMask 1 is_complemented_leaf 1 bbskip 1
     16 CSGNode    16 !cy aabb:   150.0   -14.0  -140.0   178.0    14.0   -35.0  trIdx:  8067 atm     4 IsOnlyIntersectionMask 1 is_complemented_leaf 1 bbskip 1
     17 CSGNode    17 !cy aabb:   102.0   102.0  -140.0   130.0   130.0   -35.0  trIdx:  8068 atm     4 IsOnlyIntersectionMask 1 is_complemented_leaf 1 bbskip 1
     18 CSGNode    18 !cy aabb:   -14.0   150.0  -140.0    14.0   178.0   -35.0  trIdx:  8069 atm     4 IsOnlyIntersectionMask 1 is_complemented_leaf 1 bbskip 1
     19 CSGNode    19 !cy aabb:  -130.0   102.0  -140.0  -102.0   130.0   -35.0  trIdx:  8070 atm     4 IsOnlyIntersectionMask 1 is_complemented_leaf 1 bbskip 1
     20 CSGNode    20 !cy aabb:  -178.0   -14.0  -140.0  -150.0    14.0   -35.0  trIdx:  8071 atm     4 IsOnlyIntersectionMask 1 is_complemented_leaf 1 bbskip 1
     21 CSGNode    21 !cy aabb:  -130.0  -130.0  -140.0  -102.0  -102.0   -35.0  trIdx:  8072 atm     4 IsOnlyIntersectionMask 1 is_complemented_leaf 1 bbskip 1
     22 CSGNode    22 !cy aabb:   -14.0  -178.0  -140.0    14.0  -150.0   -35.0  trIdx:  8073 atm     4 IsOnlyIntersectionMask 1 is_complemented_leaf 1 bbskip 1
     25 CSGNode    25  co aabb:  -450.0  -450.0  -140.0   450.0   450.0     1.0  trIdx:  8074 atm     6 IsOnlyIntersectionMask 0 is_complemented_leaf 0 bbskip 0
     26 CSGNode    26  cy aabb:  -450.0  -450.0     0.0   450.0   450.0     5.7  trIdx:  8075 atm     6 IsOnlyIntersectionMask 0 is_complemented_leaf 0 bbskip 0
    CSG_GGeo_Convert::convertPrim  ridx  8 primIdx   0 AABB    -450.00    -450.00    -140.00     450.00     450.00       5.70 
    2021-10-22 20:56:13.511 INFO  [4536280] [CSG_GGeo_Convert::addInstances@174]  reapeatIdx 8 iid 590,1,4



Using GeoChain/GeoChainTest.cc to simplify to see where the spurious intersects start happening
----------------------------------------------------------------------------------------------------

Simple pipe cylinder has no problem::

   .   di 

   cy      cy



Subtracting from a cy makes the problem appear::

    2021-10-26 03:44:28.354 INFO  [341315] [NTreeProcess<T>::Process@75] before
    NTreeAnalyse height 2 count 5
          di            

      cy          di    

              cy      cy


    2021-10-26 03:44:28.355 INFO  [341315] [NTreeProcess<T>::Process@90] after
    NTreeAnalyse height 2 count 5
          in            

      cy          un    

             !cy      cy


Eliminating the zshift, problem still there. 





Try BoxMinusTubs
------------------

::


    2021-10-26 16:37:33.078 INFO  [7428652] [*NTreeProcess<nnode>::Process@75] before
    NTreeAnalyse height 2 count 5
          di            

      bo          di    

              cy      cy


    inorder (left-to-right) 
     [ 0:bo] P box_box3 
     [ 0:di] C di 
     [ 0:cy] P tubs_outer 
     [ 0:di] C tubs_difference 
     [ 0:cy] P tubs_inner 


    2021-10-26 16:37:33.079 INFO  [7428652] [*NTreeProcess<nnode>::Process@90] after
    NTreeAnalyse height 2 count 5
          in            

      bo          un    

             !cy      cy


    inorder (left-to-right) 
     [ 0:bo] P box_box3 
     [ 0:in] C di 
    [ 0:!cy] P tubs_outer 
     [ 0:un] C tubs_difference 
     [ 0:cy] P tubs_inner 


