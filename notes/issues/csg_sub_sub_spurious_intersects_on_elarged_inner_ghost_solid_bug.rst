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


Godot Engine Has CSG Imp, see CSGCombiner 
----------------------------------------------

* https://github.com/godotengine/godot/issues/21125
* https://github.com/godotengine/godot/search?q=CSG&type=

* https://docs.godotengine.org/en/stable/index.html
* https://docs.godotengine.org/en/stable/tutorials/3d/csg_tools.html
* https://docs.godotengine.org/en/stable/classes/class_csgtorus.html#class-csgtorus


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



