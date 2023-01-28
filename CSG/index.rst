CSG
=====

Primary Geometry Model
-------------------------

CSGFoundry.h .cc
    geometry vector holder, nexus of control 

CSGSolid.h .cc
    references one or more CSGPrim, corresponds to compound solids 
    such as the set of G4VSolid and PV/LV making up JUNO PMTs

CSGPrim.h .cc
    references one or more CSGNode, corresponds to "root" G4VSolid like G4UnionSolid 

CSGNode.h .cc
    constituent CSG nodes, typically low level G4VSolid like G4Sphere G4Tubs etc..

CSGPrimSpec.h .cc
    specification to access the AABB and sbtIndexOffset of all CSGPrim of a CSGSolid

    * HMM: MAYBE CSGSolidSpec would be a better name


CSGFoundry Helpers
-------------------

CSGTarget.h .cc
    const CSGFoundry ctor argument, sframe/CE:center_extent/transform access

    * CSGFoundry::target instance : with transform related access
    
CSGMaker.h .cc
    non-const CSGFoundry ctor argument, many "make" methods adding CSGSolid/CSGPrim/CSGNode to foundry

    * CSGFoundry::maker instance  

CSGCopy.h .cc
    CSGFoundry cloning and copying with selection  

CSGQuery.h .cc
    const CSGFoundry ctor argument, CSGPrim and CSGNode selection API  

CU.h CU.cc
    upload/download array/vec used for CSGFoundry upload 


Other CSGFoundry Helpers
-------------------------

CSGDraw.h .cc
    CSGQuery ctor argument providing CSGNode access, 
    ascii drawing of CSG node trees

CSGGeometry.h .cc
    higher level wrapper for CSGFoundry which avoids repetition of geometry setup, 
    loading and querying mechanics used from some older tests

    * TODO: review usage and see if still useful or can be consolidated

CSGGrid.h
    signed distance field used from CSGQuery, CSGGeometry


Dead? OR should be rearranged
-------------------------------

InstanceId.h
    simple bit packing of ins_idx and gas_idx
 
    * PROBABLY DEAD

Sys.h
    simple union conversions : unsigned_as_float , float_as_unsigned 

CSGGenstep.h .cc
     const CSGFoundry ctor argumnent

     * TODO: suspect this has been superceeded by sysrap equiv 

CSGEnum.h
    solid type enum

    * TODO: eliminate or consolidate with CSGSolid.h

old_sutil_vec_math.h
   HUH: now comes from scuda.h, DEAD? 


CSGView.h .cc
    glm based eye, look, up projection transform maths

    * HMM: no CSG dependency, this can and should be done at lower sysrap level
    * TODO: review users and check for duplicated functionality  



Testing 
---------

CSGSimtrace.hh .cc
    2D cross sectioning, loads CSGFoundry, used from main CSGSimtraceTest.cc

CSGSimtraceRerun.h .cc
    on CPU csg_intersect debugging, used from main CSGSimtraceRerunTest.cc

CSGSimtraceSample.h
    on CPU csg_intersect debugging, used from main CSGSimtraceSampleTest.cc

    * TODO: contrast with Rerun, maybe consolidate 

CSGScan.h .cc
    CPU testing of GPU csg_intersect impl, used from main CSGScanTest.cc


Debug Machinery
-----------------

CSGRecord.h .cc
    behind DEBUG_RECORD macro, used for deep debugging of CSG intersect impl on CPU 

CSGDebug_Cylinder.hh .cc
    recording deep details of cylinder intersection

    
Plumbing
----------

CUDA_CHECK.h
    hostside CUDA_CHECK macro 

CSG_LOG.hh
    logging setup

CSG_API_EXPORT.hh
    symbol visibility  


 

Primary csg_intersect headers : functions take CSGNode arguments
------------------------------------------------------------------------

csg_intersect_tree.h
   distance_tree, distance_list, intersect_tree, intersect_prim, distance_prim 

csg_intersect_node.h
   distance_node_list, intersect_node_contiguous, intersect_node_discontiguous, intersect_node_overlap, 
   intersect_node, distance_node

csg_intersect_leaf.h
   distance_leaf_sphere, intersect_leaf_sphere, ... , intersect_leaf

csg_intersect_leaf_newcone.h
   intersect_leaf_newcone

csg_intersect_leaf_oldcone.h
   intersect_leaf_oldcone

csg_intersect_leaf_oldcylinder.h
   intersect_leaf_oldcylinder

csg_intersect_leaf_phicut.h
   distance_leaf_phicut, intersect_leaf_phicut, intersect_leaf_phicut_dev, intersect_leaf_phicut_lucas

csg_intersect_leaf_thetacut.h
   intersect_leaf_thetacut, intersect_leaf_thetacut_lucas


Helpers for csg_intersect
---------------------------

csg_classify.h
   enum : CTRL_RETURN_MISS/../LOOP_B 
   enum : State_Enter/Exit/Miss
   struct LUT 

csg_error.h
   enum : ERROR_LHS_POP_EMPTY ...

csg_tranche.h
   struct Tranche
   Postorder Tranch storing a stack of slices into the postorder sequence

csg_pack.h
   PACK4/UNPACK4 macros 

csg_postorder.h
   complete binary tree traversal in bit-twiddling macros 

csg_robust_quadratic_roots.h
   Numerically Stable Method for Solving Quadratic Equations 

csg_stack.h
   CSG_Stack struct, csg_push, csg_pop

f4_stack.h
   struct F4_Stack using float4 as micro stack



