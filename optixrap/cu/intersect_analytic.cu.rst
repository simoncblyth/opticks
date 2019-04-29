intersect_analytic.cu
===========================

*intersect_analytic.cu* provides intersection of rays with arbitrarily complex 
solids represented using a constructive solid geometry model.
The model uses a complete binary tree, with primitives at the leaves 
and boolean operations (intersection, union, difference) at the other nodes.

The principal of the approach was described in a note by Andrew Kensler
"Ray Tracing CSG Objects Using Single Hit Intersections", that 
is available from: http://xrt.wikidot.com/doc:csg together with 
discussion of the technique as used in the XRT raytracer.


OptiX intersection interface 
-------------------------------

* https://bitbucket.org/simoncblyth/opticks/src/default/optixrap/cu/intersect_analytic.cu

* OptiX bounds and intersect programs, the relevant branch is CSG_FLAGNODETREE

Buffers::

    primBuffer : offsets into part/tran/plan (1 quad)
    partBuffer : nodes of the CSG tree (both operators and primitives) (4 quads) 
    tranBuffer : transforms
    planBuffer : planes

    quad is a union of float4, int4 and uint4 



csg_intersect_part.h : primitive intersection
------------------------------------------------

* https://bitbucket.org/simoncblyth/opticks/src/default/optixrap/cu/csg_intersect_part.h

Typecode switch statement invoking bounds and intersect functions for each shape.


csg_intersect_primitive.h : Primitive Intersections with ~10 different solids
---------------------------------------------------------------------------------
        
* https://bitbucket.org/simoncblyth/opticks/src/default/optixrap/cu/csg_intersect_primitive.h


csg_intersect_boolean.h
----------------------------

evaluative_csg
~~~~~~~~~~~~~~~~

General approach is to emulate recursion using a stack of slices
into the postorder traversal sequence of the complete binary tree.
Uses the postorder traversal feature that left and right children of a 
node are always visted before the node itself. 

* https://bitbucket.org/simoncblyth/opticks/src/default/optixrap/cu/csg_intersect_boolean.h#lines-545

  evaluative_csg (relevant branch is USE_TWIDDLE_POSTORDER)



Postorder Tranch and CSG intersect stacks used by evaluative_csg::


     186 #define CSG_STACK_SIZE 15
     187 #define TRANCHE_STACK_SIZE 4
     ...
     320 struct Tranche   // slice is a pair of begin and end indices of the postorder traversal
     321 {
     322     float tmin[TRANCHE_STACK_SIZE] ;     // TRANCHE_STACK_SIZE is 4 
     323     unsigned  slice[TRANCHE_STACK_SIZE] ;
     324     int curr ;
     325 };
     ...
     397 struct CSG      // holds isect(normal and t) and nodeIdx
     398 {
     399    float4 data[CSG_STACK_SIZE] ;
     400    unsigned idx[CSG_STACK_SIZE] ;
     401    int curr ;
     402 };




