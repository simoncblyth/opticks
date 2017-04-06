CSG Transform Support
=========================


CSG Tree Objective
----------------------

* recall the CSG trees are intended to be small per-solid trees
  corresponding to shape definitions (ie not definitions of full scene geometry)



Transforms with Ray-trace and SDF
------------------------------------

To translate or rotate a surface modeled as an CSG tree, 
apply the inverse transformation to the point for SDFs or the ray for 
raytracing before doing the SDF distance calc or ray tracing intersection
calc.


SDF
------

* Where to hold the transform in nnode trees and CSG trees ?

 * G4 allows the RHS of a boolean combination to be transformed using 
   a transform that lives with the combination



* use glm::mat4 ?


local/global transforms ?
~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    111 double nunion::operator()(double px, double py, double pz)
    112 {

    ///    just transform px,py,pz here only  ?

    113     assert( left && right );
    114     double l = (*left)(px, py, pz) ;
    115     double r = (*right)(px, py, pz) ;
    116     return fmin(l, r);
    117 }


Perhaps can just locally apply the transform ? to the coordinates
passed down the tree ? Relying on subsequent transforms transforming 
again the transformed coordinates... this would be simplest.

The alternative would be to traverse up the tree thru parent 
links collecting and multiplying transforms and store that 
as a global transfrom within each node to apply to global coordinates.

Actually its not clear how to use global transforms as the evaluation is done
treewise ... with each node not knowing where it is in the tree ?

BUT: for internal nodes the coordinates are not actually used, they are 
just being passed down the tree until reach the leaves/primitives ... so this 
means can collect ancestor transforms into the primitives : this is 
what will need to do on GPU, so actually its better to take same approach on CPU 


* adopted globaltransform held in primitive, which is obtained at deserialization (in NCSG)
  from product of ancestor node transforms



Transforming BBox ?
---------------------

* http://dev.theomader.com/transform-bounding-boxes/
* http://www.cs.unc.edu/~zhangh/technotes/bbox.pdf

* https://www.geometrictools.com/Documentation/AABBForTransformedAABB.pdf
* https://github.com/erich666/GraphicsGems/blob/master/gems/TransBox.c
* http://www.akshayloke.com/2012/10/22/optimized-transformations-for-aabbs/



Models
-------

* input python model opticks.dev.csg.csg.CSG
* numpy array serialization
* NCSG created nnode model  


Where to hang the transform ?
--------------------------------

parent.rtransform OR node.transform ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* transform reference on CSG operation node is advantageous, as no space pressure there

  * actually above "advantage" is conflating the serialization with the in memory nnode model, 
    the in nnode model does not have any space issues, and it does not need to 
    precisely follow what the serialization does

* so can define and serialize using rtransform and then deserialize onto transforms 
  directly on nodes as that is easier in usage 

* not so clear that node.transform is easier in usage... as 
  would mean that every primitive needs to implement coordinate transformations 
  handling as opposed to just the 3 CSG operation nodes



