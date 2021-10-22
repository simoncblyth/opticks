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


CSGDemo Workflow
----------------------

::

   cd ~/opticks/CSG       ## create foundry geometry and persist 
   ./CSGDemoTest.sh    

   cd ~/opticks/CSGOptiX  ## create jpg image ray trace render of geometry 
   ./cxr_demo.sh 







