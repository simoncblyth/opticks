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


Want torus primitive to have an associated bounding "prim" triplet: cy-cy
----------------------------------------------------------------------------

* torus bounding cy-cy is known at initialization so should do it there 

* need way to associate a CSG tree (no need to restrict to torus/prim) 
  with a cheaper other CSG tree which can provide 
  better bounding than aabb 

* then 


csg_intersect_bileaf
--------------------------

* no need to restrict to cy-cy implement csg_intersect_bileaf
  for geometry of form operator(left,right) where left/right are primitives

* hmm no need for separate typecode when tree height is 1 its a bileaf

* interface needs to fit match csg_intersect_part 

::

     667             if(primitive)
     668             {
     669                 float4 isect = make_float4(0.f, 0.f, 0.f, 0.f) ;
     670 
     671                 csg_intersect_part( prim, partOffset+nodeIdx-1, tmin, isect );
      



intersect_analytic.cu

* needs to check if a tree has an associated bounding tree, and check intersect with that 
  first ... no cannot do up here torus is usually deep in the tree

::

    080 
     81 #include "csg_intersect_primitive.h"
     82 #include "csg_intersect_part.h"
     83 #include "csg_intersect_boolean.h"
     84 
     85 #include "intersect_ztubs.h"
     86 #include "intersect_zsphere.h"
     87 #include "intersect_box.h"
     88 #include "intersect_prism.h"
     89 
     90 
     91 RT_PROGRAM void bounds (int primIdx, float result[6])
     92 {
     93     //if(primIdx == 0) transform_test();
    ...

    172 RT_PROGRAM void intersect(int primIdx)
    173 {
    174     const Prim& prim    = primBuffer[primIdx];
    175 
    176     unsigned partOffset  = prim.partOffset() ;
    177     unsigned numParts    = prim.numParts() ;   // <-- nodes in tree for CSG_FLAGNODETREE
    178     unsigned primFlag    = prim.primFlag() ;
    179 
    180     uint4 identity = identityBuffer[instance_index] ;
    181 
    182 
    183     if(primFlag == CSG_FLAGNODETREE)
    184     {        
    185         Part pt0 = partBuffer[partOffset + 0] ;  
    186 
    187         identity.z = pt0.boundary() ;        // replace placeholder zero with test analytic geometry root node boundary
    188 
    189         evaluative_csg( prim, identity );
    190         //intersect_csg( prim, identity );
    191 
    192     }            


csg_intersect_part.h
-------------------------

::

    void csg_bounds_prim(int primIdx, const Prim& prim, optix::Aabb* aabb )
    void csg_intersect_part(const Prim& prim, const unsigned partIdx, const float& tt_min, float4& tt  )
     

Fans out to the csg_bounds_sphere etc.. for all primitives::

    105 static __device__
    106 void csg_intersect_part(const Prim& prim, const unsigned partIdx, const float& tt_min, float4& tt  )
    107 {
    108     unsigned tranOffset = prim.tranOffset();
    109     unsigned planOffset = prim.planOffset();
    110     Part pt = partBuffer[partIdx] ;
    111 
    112     unsigned typecode = pt.typecode() ;
    113     unsigned gtransformIdx = pt.gtransformIdx() ;  //  gtransformIdx is 1-based, 0 meaning None
    114     bool complement = pt.complement();
    115 
    116     bool valid_intersect = false ;
    117 
    118     if(gtransformIdx == 0)
    119     {
    120         switch(typecode)
    121         {
    122             case CSG_SPHERE:    valid_intersect = csg_intersect_sphere(   pt.q0,               tt_min, tt, ray.origin, ray.direction ) ; break ;
    123             case CSG_ZSPHERE:   valid_intersect = csg_intersect_zsphere(  pt.q0, pt.q1, pt.q2, tt_min, tt, ray.origin, ray.direction ) ; break ;
    124             case CSG_BOX:       valid_intersect = csg_intersect_box(      pt.q0,               tt_min, tt, ray.origin, ray.direction ) ; break ;
    125             case CSG_BOX3:      valid_intersect = csg_intersect_box3(     pt.q0,               tt_min, tt, ray.origin, ray.direction ) ; break ;




OptiX selector : can select between OptiX prim children
-----------------------------------------------------------

::

    A selector is similar to a group in that it is a collection of higher
    level graph nodes. The number of nodes in the collection is set by
    rtSelectorSetChildCount, and the individual children are assigned with
    rtSelectorSetChild. Valid child types are rtGroup, rtGeometryGroup,
    rtTransform, and rtSelector.  The main difference between selectors and groups
    is that selectors do not have an acceleration structure associated with them.
    Instead, a visit program is specified with rtSelectorSetVisitProgram. This
    program is executed every time a ray encounters the selector node during graph
    traversal. The program specifies which children the ray should continue
    traversal through by calling rtIntersectChild.  A typical use case for a
    selector is dynamic (i.e. per-ray) level of detail: an object in the scene may
    be represented by a number of geometry nodes, each containing a different level
    of detail version of the object. The geometry groups containing these different
    representations can be assigned as children of a selector. 

    The visit program
    can select which child to intersect using any criterion (e.g. based on the
    footprint or length of the current ray), and ignore the others.  As for groups
    and other graph nodes, child nodes of a selector can be shared with other graph
    nodes to allow flexible instancing.



Maybe rays could have a lod flag used in selector to first 
intersect with cheap tree ? 

* seems too awkward having to do something for all... rays
  just to handle some expensive geometry 


/Developer/OptiX/SDK/optixSelector/selector_example.cu::

     29 #include <optix.h>
     30 #include <optixu/optixu_math_namespace.h>
     31 
     32 using namespace optix;
     33 
     34 rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
     35 
     36 RT_PROGRAM void visit()
     37 {
     38   unsigned int index = (unsigned int)( ray.direction.y < 0.0f );
     39   rtIntersectChild( index );
     40 }



::

    251     // Geometry group nodes
    252     RTgeometrygroup group[2];
    253     group[0] = makeGeometryGroup( context, instance[0], acceleration[0] );
    254     group[1] = makeGeometryGroup( context, instance[1], acceleration[1] );
    255 
    256     /* Setup selector as top objects */
    257 
    258     // Init selector node
    259     RTselector selector;
    260     RTprogram  stor_visit_program;
    261     RT_CHECK_ERROR( rtSelectorCreate(context,&selector) );
    262     RT_CHECK_ERROR( rtProgramCreateFromPTXFile(context,ptxpath("selector_example.cu").c_str(),"visit",&stor_visit_program) );
    263     RT_CHECK_ERROR( rtSelectorSetVisitProgram(selector,stor_visit_program) );
    264     RT_CHECK_ERROR( rtSelectorSetChildCount(selector,2) );
    265     RT_CHECK_ERROR( rtSelectorSetChild(selector, 0, group[0]) );
    266     RT_CHECK_ERROR( rtSelectorSetChild(selector, 1, group[1]) );





Hmm could     
