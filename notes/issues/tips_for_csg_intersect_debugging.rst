tips_for_csg_intersect_debugging
==================================


Overview
-----------

A primitive raytrace that looks correct does not mean 
that the primitive will successfully work as a node/solid 
in a CSG tree.

There are additional requirements:

* tmin cutting into the solid MUST yield the otherside intersect 

Note that some primitives have special case handling of 
axial rays, this means that simple tmin near scanning 
for the solid will typically not show problems unless take
action to get a precisely axial viewpoint.

Axial special cases provides a place for bugs to hide, see
:doc:`tlens-concave-ignored-due-to-cylinder-axial-photon-intersect-failure`



Problem Isolation steps
------------------------

* change type of primitive
* try axial rays ONLY, otherwise off-axial intersects can hide issue
* try off-axial rays 


From smoking gun to fix
-------------------------

Configure torch with a small number of photons (~10) that 
all exhibit the issue, then add debugging within oxrap-/cu/









