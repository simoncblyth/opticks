csg_contiguous_discontiguos_multiunion
=========================================


Requirements
---------------

1. act like a CSG primitive, but at the same time be composed of multiple internal 
   hidden CSGNode primitive nodes (not CSG trees or other multiunion) 
   each of which can have different transforms (like G4MultiUnion) 

2. able to take part in CSG combinations with other CSG primitives or trees

3. CSG_CONTIGUOUS has additional requirement that the combined shape has the 
   same topology as a ball : ie no holes, this makes the implementation simple 
   and means it can work with very little storage requirements  

4. CSG_DISCONTIGUOUS removes the contiguous restriction, but the implementation will be 
   considerably slower : as is forced to store distances for all internal nodes


How to implement
-------------------

* CSGPrim currently just (numNode, nodeOffset) as it assumes all Prim are CSG trees
* hmm could fork with different types of CSGPrim but then the compound would not be 
  able to participate as just another node in a CSG tree

   
Splitting at CSGPrim level could simply be done in intersect_prim : but then cannot be part of ordinary tree
---------------------------------------------------------------------------------------------------------------  

::

    1522 INTERSECT_FUNC
    1523 bool intersect_prim( float4& isect, int numNode, const CSGNode* node, const float4* plan, const qat4* itra, const float t_min , const float3& ray_origin, const float3& ray     _direction )
    1524 {
    1525     return numNode == 1
    1526                ?
    1527                   intersect_node(isect,          node, plan, itra, t_min, ray_origin, ray_direction )
    1528                :
    1529                   intersect_tree(isect, numNode, node, plan, itra, t_min, ray_origin, ray_direction )
    1530                ;
    1531 }

::

    349 /**
    350 __intersection__is
    351 ----------------------
    352 
    353 HitGroupData provides the numNode and nodeOffset of the intersected CSGPrim.
    354 Which Prim gets intersected relies on the CSGPrim::setSbtIndexOffset
    355 
    356 **/
    357 extern "C" __global__ void __intersection__is()
    358 {
    359     HitGroupData* hg  = (HitGroupData*)optixGetSbtDataPointer();
    360     int numNode = hg->numNode ;        // equivalent to CSGPrim, as same info : specify complete binary tree sequence of CSGNode 
    361     int nodeOffset = hg->nodeOffset ;
    362 
    363     const CSGNode* node = params.node + nodeOffset ;  // root of tree
    364     const float4* plan = params.plan ;
    365     const qat4*   itra = params.itra ;
    366 
    367     const float  t_min = optixGetRayTmin() ;
    368     const float3 ray_origin = optixGetObjectRayOrigin();
    369     const float3 ray_direction = optixGetObjectRayDirection();
    370 
    371     float4 isect ; // .xyz normal .w distance 
    372     if(intersect_prim(isect, numNode, node, plan, itra, t_min , ray_origin, ray_direction ))
    373     {



How could a split at CSGNode level be done ?
-----------------------------------------------

The compound CSGNode needs to be able to refer to other CSGNode, 

* can impose requirement that the internal CSGNode are stored sequentially so just needs (nodeOffset, numNode) just like CSGPrim    
  but in this case numNode will not be complete binary tree length with placeholder CSG_ZERO it will be whatever is needed  

* follow convention of laying down the constituent nodes immediately after their compound "container" parent so the 
  compound then just needs numConstituent (this avoids having to pass in node0) 

* hmm its going to need to call intersect_node for each of its constituents 

  * probably OptiX will not allow such a recursive call 



::

    1455 INTERSECT_FUNC
    1456 bool intersect_node( float4& isect, const CSGNode* node, const float4* plan, const qat4* itra, const float t_min , const float3& ray_origin , const float3& ray_direction )
    1457 {
    1458     const unsigned typecode = node->typecode() ; 
    1459     const unsigned gtransformIdx = node->gtransformIdx() ;
    1460     const bool complement = node->is_complement();
    1461 
    1462     const qat4* q = gtransformIdx > 0 ? itra + gtransformIdx - 1 : nullptr ;  // gtransformIdx is 1-based, 0 meaning None
    1463 
    1464     float3 origin    = q ? q->right_multiply(ray_origin,    1.f) : ray_origin ; 
    1465     float3 direction = q ? q->right_multiply(ray_direction, 0.f) : ray_direction ;  
    1466 
    1483     bool valid_isect = false ;
    1484     switch(typecode)
    1485     {
    1486         case CSG_SPHERE:           valid_isect = intersect_node_sphere(           isect, node->q0,               t_min, origin, direction ) ; break ;
    1487         case CSG_ZSPHERE:          valid_isect = intersect_node_zsphere(          isect, node->q0, node->q1,     t_min, origin, direction ) ; break ;
    1488         case CSG_CONVEXPOLYHEDRON: valid_isect = intersect_node_convexpolyhedron( isect, node, plan,             t_min, origin, direction ) ; break ;
    1489         case CSG_CONTIGUOUS:       valid_isect = intersect_node_contiguous(       isect, node, itra,             t_min, origin, direction ) ; break ;



