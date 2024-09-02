CSGScanTest_with_CSGMaker_solids_not_intersecting_booleans
============================================================


NOW FIXED : Issue : unexpected zero hits with CSGMaker booleans
-------------------------------------------------------------------

::

    [blyth@localhost ~]$ GEOM=UnionBoxSphere ~/o/CSG/tests/CSGScanTest.sh run
    /home/blyth/o/CSG/tests/CSGScanTest.sh - GEOM UnionBoxSphere run
    CSGScan::add_rectangle_scan extent 100 halfside 200
    CSGScan::add_circle_scan extent 100 radius 200
     h  num 17890 n_hit 0 (num-n_hit) 17890
     d  num 17890 n_hit 0 (num-n_hit) 17890
    [blyth@localhost ~]$ 


Fixed with setSubNum in CSGMaker::makeBooleanTriplet
----------------------------------------------------

::

     528 CSGSolid* CSGMaker::makeBooleanTriplet( const char* label, unsigned op_, const CSGNode& left, const CSGNode& right, int meshIdx )
     529 {
     530     unsigned numPrim = 1 ;
     531     CSGSolid* so = fd->addSolid(numPrim, label);
     532 
     533     unsigned numNode = 3 ;
     534     int nodeOffset_ = -1 ;
     535     CSGPrim* p = fd->addPrim(numNode, nodeOffset_ );
     536     if(meshIdx > -1) p->setMeshIdx(meshIdx);
     537 
     538     CSGNode op = CSGNode::BooleanOperator(op_, -1);  // CHANGED 3 to -1 as this is standard boolean ?
     539     CSGNode* n = fd->addNode(op);
     540 
     541     CSGNode* root = n ;
     542     // cf CSGImport::importPrim 
     543     root->setSubNum(numNode); // avoids notes/issues/CSGScanTest_with_CSGMaker_solids_not_intersecting_booleans.rst
     544     root->setSubOffset(0);
     545 
     546 
     547     fd->addNode(left);
     548     fd->addNode(right);
     549 




Debug : shows subNum in intersect_tree is zero : so missing some geometry setup with CSGMaker solid ? 
--------------------------------------------------------------------------------------------------------

::

    BP=intersect_prim ~/o/CSG/tests/CSGScanTest.sh

::

    d 1 "CSGScanTest" hit Breakpoint 3, intersect_tree (isect=..., node=0x7ffff43c8010, plan0=0x0, itra0=0x7ffff3dac010, t_min=0, ray_origin=..., ray_direction=...) at /home/blyth/opticks/CSG/csg_intersect_tree.h:269
    269     const int numNode=node->subNum() ;   // SO THIS SHOULD NO LONGER EVER BE 1 
    (gdb) list 
    264 **/
    265 
    266 TREE_FUNC
    267 bool intersect_tree( float4& isect, const CSGNode* node, const float4* plan0, const qat4* itra0, const float t_min , const float3& ray_origin, const float3& ray_direction )
    268 {
    269     const int numNode=node->subNum() ;   // SO THIS SHOULD NO LONGER EVER BE 1 
    270     unsigned height = TREE_HEIGHT(numNode) ; // 1->0, 3->1, 7->2, 15->3, 31->4 
    271     float propagate_epsilon = 0.0001f ;  // ? 
    272     int ierr = 0 ;  
    273 
    (gdb) b 272
    Breakpoint 4 at 0x7ffff7c8db08: file /home/blyth/opticks/CSG/csg_intersect_tree.h, line 272.
    (gdb) c
    Continuing.

    Thread 1 "CSGScanTest" hit Breakpoint 4, intersect_tree (isect=..., node=0x7ffff43c8010, plan0=0x0, itra0=0x7ffff3dac010, t_min=0, ray_origin=..., ray_direction=...) at /home/blyth/opticks/CSG/csg_intersect_tree.h:272
    272     int ierr = 0 ;  
    (gdb) p numNode
    $7 = 0
    (gdb) 

