spurious-internal-boundary-intersects-in-high-node-count-solids
================================================================


Issue was first noticed with JUNO AltXJfixtureConstruction_XY
The Alt fixed the coincident faces but was still left with 
some spurios intersects on parts of the cylinder within 
some of the boxes visible in XY cross section.

Actually looking back the first hint was with the 
unexpected vertical (+Z) cylinder edge spurious 
intersects from XJfixtureConstruction.

To provide a more systematic way of checking all intersects I started
filling out the distance_node functions to csg_intersect_node.h
and implemented the postorder traverse combined CSG calculation
in csg_intersect_tree.h::distance_tree

Using AnnulusFourBoxUnion AnnulusThreeBoxUnion noted that 
only high node count solids are afflicted putting 
suspicion on balancing. 

Extensive CSG debugging using DEBUG_RECORD and CSGRecord to 
follow the micro-steps of the CSG intersection implementation 
revealed the cause to be disjoint unions. Working through the
CSG algorithm manually for various CSG trees demonstrates how.

Cause has been confirmed to be due to tree balancing 
managing to sometimes yield disjoint unions which 
mess up the CSG algorithm preventing it from getting 
all the way to the last exit. 

::

    c
    ./csg_geochain.sh 

    SPURIOUS=1 IXIYIZ=-6,0,0 IW=17 GEOM=AnnulusFourBoxUnion_YX  ./csg_geochain.sh
    SPURIOUS=1 IXIYIZ=-6,0,0 IW=17 GEOM=BoxFourBoxUnion_YX      ./csg_geochain.sh


Switching off balancing avoids the problem but causes
a performance issue for solids with many primitives.


TODO
------

1. check how balancing changes the ordering, expt to see 
   if disjointness at the end of the postorder traverse 
   is also precluded 


2. experiment with a new compound "CSG_BLOBUNION" that 
   does something a bit similar to G4MultiUnion with restrictions
   on the topology to make finding compound enter and exit points 
   performant   

   * much of the demand for handling high node count solids could 
     be mopped up into blobs 

3. improve presentation of CSG/tests/CSGIntersectSolidTest.py so can explain the issue 







