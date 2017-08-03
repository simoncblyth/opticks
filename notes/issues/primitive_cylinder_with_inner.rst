Primitive Cylinder With Inner Radius
======================================

Currently cy with rmin is handled by 
CSG subtraction. Because the cylinder intersect
is already complicated enough.

Where primitive Cy with Inner would Help
-----------------------------------------------------

* quick torus bounding test to skip expensive (and artifact prone)
  quartic root finding when no intersect with a bounding in(cy,!cy)

* tree simplification

How to imp ?
---------------

Directly in the cy prim is a nono : too many cases.
Perhaps some intermediate, single-operation primitive "bileaf" 
could be implemented.



  



 
