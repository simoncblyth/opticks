CX4GDMLTest_soIdx_33_gltf
=========================================

Fixed several issues with NTreeBuilder created nnode trees, 
seen first with polycone UnionTree.


FIXED : SDF : bare nnode in operator ?
----------------------------------------

Reproduce with UnionTree of more than 2 prim::

   om-mk "make NTreeBuilderTest && ./NTreeBuilderTest 3 "


::

    (lldb) f 5
    frame #5: 0x0000000105070163 libNPY.dylib`nunion::operator(this=0x000000010db8e048, x=2000, y=0, z=-1968.5)(float, float, float) const at NNode.cpp:690
       687 	{
       688 	    assert( left && right );
       689 	    float l = (*left)(x, y, z) ;
    -> 690 	    float r = (*right)(x, y, z) ;
       691 	    return fminf( l, r );
       692 	}
       693 	float nintersection::operator()(float x, float y, float z) const 
    (lldb) p left
    (ncylinder *) $1 = 0x000000010db8afc0
    (lldb) p right
    (nnode *) $2 = 0x000000010db8b6b0
    (lldb) 

    (lldb) f 4
    frame #4: 0x000000010507472d libNPY.dylib`nnode::operator(this=0x000000010db8b6b0, (null)=2000, (null)=0, (null)=-1968.5)(float, float, float) const at NNode.cpp:903
       900 	
       901 	float nnode::operator()(float , float , float ) const 
       902 	{
    -> 903 	    assert(0 && "nnode::operator() needs override ");
       904 	    return 0.f ; 
       905 	}
       906 	
    (lldb) 



FIXED : infinite recursion from bbox
--------------------------------------

::

    2018-06-20 23:39:59.054 INFO  [23910386] [*X4PhysicalVolume::convertNode@404] convertNode  ndIdx 3156 soIdx 33
    2018-06-20 23:39:59.054 INFO  [23910386] [X4Solid::init@56] X4SolidBase name                             oav0xc2ed7c8 entityType 1 entityName G4UnionSolid root 0x0
    2018-06-20 23:39:59.054 INFO  [23910386] [X4Solid::init@56] X4SolidBase name                         oav_cyl0xc234858 entityType 25 entityName G4Tubs root 0x0
    2018-06-20 23:39:59.055 INFO  [23910386] [X4Solid::convertTubs@356] 
    -----------------------------------------------------------
        *** Dump for solid - oav_cyl0xc234858 ***
        ===================================================
     Solid type: G4Tubs
     Parameters: 
        inner radius : 0 mm 
        outer radius : 2000 mm 
        half length Z: 1968.5 mm 
        starting phi : 0 degrees 
        delta phi    : 360 degrees 
    -----------------------------------------------------------

    2018-06-20 23:39:59.055 INFO  [23910386] [X4Solid::init@56] X4SolidBase name                                  placedB entityType 0 entityName G4DisplacedSolid root 0x0
    2018-06-20 23:39:59.055 INFO  [23910386] [X4Solid::init@56] X4SolidBase name                    oav_polycone0xbf1c840 entityType 15 entityName G4Polycone root 0x0
    2018-06-20 23:39:59.055 WARN  [23910386] [X4Solid::convertPolyconePrimitives@605]  skipping z2 == z1 zp 
    [ 0:cy oav_polycone0xbf1c8400_zp_cylinder]
    [ 0:co oav_polycone0xbf1c8402_zp_cone]
    2018-06-20 23:39:59.055 INFO  [23910386] [NTreeBuilder<nnode>::analyse@93]  NNodeAnalyse 
    NNodeAnalyse height 1 count 3
          un    

      ze      ze


    2018-06-20 23:39:59.055 INFO  [23910386] [NTreeBuilder<nnode>::analyse@93]  NNodeAnalyse 
    NNodeAnalyse height 1 count 3
          un    

    oav_polycone0xbf1c8400_zp_cylinder    oav_polycone0xbf1c8402_zp_cone


    2018-06-20 23:39:59.055 INFO  [23910386] [NTreeBuilder<nnode>::analyse@93]  NNodeAnalyse 
    NNodeAnalyse height 1 count 3
          un    

    oav_polycone0xbf1c8400_zp_cylinder    oav_polycone0xbf1c8402_zp_cone


    2018-06-20 23:39:59.055 INFO  [23910386] [*NTreeBuilder<nnode>::CommonTree@19]  num_prims 2 height 1 operator union
    2018-06-20 23:39:59.055 INFO  [23910386] [*X4Mesh::Placeholder@29]  visExtent G4VisExtent (bounding box):
      X limits: -2040 2170.92
      Y limits: -2137.94 2137.94
      Z limits: -1968.5 2126.12
    Process 67572 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = EXC_BAD_ACCESS (code=2, address=0x7ffeef3fffd0)
        frame #0: 0x000000010506e33a libNPY.dylib`nnode::get_primitive_bbox(this=0x000000010e62cfc0, bb=0x00007ffeef400880) const at NNode.cpp:542
       539 	
       540 	void nnode::get_primitive_bbox(nbbox& bb) const 
       541 	{
    -> 542 	    assert(is_primitive());
       543 	
       544 	    const nnode* node = this ;  
       545 	
    Target 0: (CX4GDMLTest) stopped.
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = EXC_BAD_ACCESS (code=2, address=0x7ffeef3fffd0)
      * frame #0: 0x000000010506e33a libNPY.dylib`nnode::get_primitive_bbox(this=0x000000010e62cfc0, bb=0x00007ffeef400880) const at NNode.cpp:542
        frame #1: 0x000000010506eb7a libNPY.dylib`nnode::bbox(this=0x000000010e62cfc0) const at NNode.cpp:625
        frame #2: 0x000000010506e5f0 libNPY.dylib`nnode::get_primitive_bbox(this=0x000000010e62cfc0, bb=0x00007ffeef400f30) const at NNode.cpp:571
        frame #3: 0x000000010506eb7a libNPY.dylib`nnode::bbox(this=0x000000010e62cfc0) const at NNode.cpp:625
        frame #4: 0x000000010506e5f0 libNPY.dylib`nnode::get_primitive_bbox(this=0x000000010e62cfc0, bb=0x00007ffeef4015e0) const at NNode.cpp:571
        frame #5: 0x000000010506eb7a libNPY.dylib`nnode::bbox(this=0x000000010e62cfc0) const at NNode.cpp:625
        frame #6: 0x000000010506e5f0 libNPY.dylib`nnode::get_primitive_bbox(this=0x000000010e62cfc0, bb=0x00007ffeef401c90) const at NNode.cpp:571
        frame #7: 0x000000010506eb7a libNPY.dylib`nnode::bbox(this=0x000000010e62cfc0) const at NNode.cpp:625
        frame #8: 0x000000010506e5f0 libNPY.dylib`nnode::get_primitive_bbox(this=0x000000010e62cfc0, bb=0x00007ffeef402340) const at NNode.cpp:571
        frame #9: 0x000000010506eb7a libNPY.dylib`nnode::bbox(this=0x000000010e62cfc0) const at NNode.cpp:625
        frame #10: 0x000000010506e5f0 libNPY.dylib`nnode::get_primitive_bbox(this=0x000000010e62cfc0, bb=0x00007ffeef4029f0) const at NNode.cpp:571
        frame #11: 0x000000010506eb7a libNPY.dylib`nnode::bbox(this=0x000000010e62cfc0) const at NNode.cpp:625
        frame #12: 0x000000010506e5f0 libNPY.dylib`nnode::get_primitive_bbox(this=0x000000010e62cfc0, bb=0x00007ffeef4030a0) const at NNode.cpp:571
        frame #13: 0x000000010506eb7a libNPY.dylib`nnode::bbox(this=0x000000010e62cfc0) const at NNode.cpp:625
        frame #14: 0x000000010506e5f0 libNPY.dylib`nnode::get_primitive_bbox(this=0x000000010e62cfc0, bb=0x00007ffeef403750) const at NNode.cpp:

        ...

        frame #9774: 0x000000010506e5f0 libNPY.dylib`nnode::get_primitive_bbox(this=0x000000010e62cfc0, bb=0x00007ffeefbfb250) const at NNode.cpp:571
        frame #9775: 0x000000010506eb7a libNPY.dylib`nnode::bbox(this=0x000000010e62cfc0) const at NNode.cpp:625
        frame #9776: 0x000000010506e5f0 libNPY.dylib`nnode::get_primitive_bbox(this=0x000000010e62cfc0, bb=0x00007ffeefbfb900) const at NNode.cpp:571
        frame #9777: 0x000000010506eb7a libNPY.dylib`nnode::bbox(this=0x000000010e62cfc0) const at NNode.cpp:625
        frame #9778: 0x000000010506e5f0 libNPY.dylib`nnode::get_primitive_bbox(this=0x000000010e62cfc0, bb=0x00007ffeefbfbe40) const at NNode.cpp:571
        frame #9779: 0x000000010506eb7a libNPY.dylib`nnode::bbox(this=0x000000010e62cfc0) const at NNode.cpp:625
        frame #9780: 0x00000001050afbed libNPY.dylib`NNodeNudger::update_prim_bb(this=0x000000010e62eb30) at NNodeNudger.cpp:41
        frame #9781: 0x00000001050af5b2 libNPY.dylib`NNodeNudger::init(this=0x000000010e62eb30) at NNodeNudger.cpp:29
        frame #9782: 0x00000001050af51a libNPY.dylib`NNodeNudger::NNodeNudger(this=0x000000010e62eb30, root_=0x000000010e62ceb0, epsilon_=0.00000999999974, (null)=0) at NNodeNudger.cpp:23
        frame #9783: 0x00000001050af5fd libNPY.dylib`NNodeNudger::NNodeNudger(this=0x000000010e62eb30, root_=0x000000010e62ceb0, epsilon_=0.00000999999974, (null)=0) at NNodeNudger.cpp:22
        frame #9784: 0x000000010512bba9 libNPY.dylib`NCSG::make_nudger(this=0x000000010e62ea40) const at NCSG.cpp:150
        frame #9785: 0x000000010512b723 libNPY.dylib`NCSG::NCSG(this=0x000000010e62ea40, root=0x000000010e62ceb0) at NCSG.cpp:100
        frame #9786: 0x000000010512bc7d libNPY.dylib`NCSG::NCSG(this=0x000000010e62ea40, root=0x000000010e62ceb0) at NCSG.cpp:119
        frame #9787: 0x000000010513a1fc libNPY.dylib`NCSG::FromNode(root=0x000000010e62ceb0, config=0x0000000000000000) at NCSG.cpp:1466
        frame #9788: 0x000000010031e2ad libExtG4.dylib`X4PhysicalVolume::convertNode(this=0x00007ffeefbfdc98, pv=0x000000010f48dd50, parent=0x000000010e62bcf0, depth=11, pv_p=0x0000000107b792a0) at X4PhysicalVolume.cc:436
        frame #9789: 0x000000010031d395 libExtG4.dylib`X4PhysicalVolume::convertTree_r(this=0x00007ffeefbfdc98, pv=0x000000010f48dd50, parent=0x000000010e62bcf0, depth=11, parent_pv=0x0000000107b792a0) at X4PhysicalVolume.cc:336
        frame #9790: 0x000000010031d3f2 libExtG4.dylib`X4PhysicalVolume::convertTree_r(this=0x00007ffeefbfdc98, pv=0x0000000107b792a0, parent=0x000000010e628c70, depth=10, parent_pv=0x0000000107b85c00) at X4PhysicalVolume.cc:343
        frame #9791: 0x000000010031d3f2 libExtG4.dylib`X4PhysicalVolume::convertTree_r(this=0x00007ffeefbfdc98, pv=0x0000000107b85c00, parent=0x000000010e625de0, depth=9, parent_pv=0x0000000107b88310) at X4PhysicalVolume.cc:343
        frame #9792: 0x000000010031d3f2 libExtG4.dylib`X4PhysicalVolume::convertTree_r(this=0x00007ffeefbfdc98, pv=0x0000000107b88310, parent=0x000000010e6230e0, depth=8, parent_pv=0x000000010be9bd70) at X4PhysicalVolume.cc:343
        frame #9793: 0x000000010031d3f2 libExtG4.dylib`X4PhysicalVolume::convertTree_r(this=0x00007ffeefbfdc98, pv=0x000000010be9bd70, parent=0x000000010e3c4370, depth=7, parent_pv=0x000000010be9de50) at X4PhysicalVolume.cc:343
        frame #9794: 0x000000010031d3f2 libExtG4.dylib`X4PhysicalVolume::convertTree_r(this=0x00007ffeefbfdc98, pv=0x000000010be9de50, parent=0x000000010e1ba200, depth=6, parent_pv=0x000000010bebe8e0) at X4PhysicalVolume.cc:343
        frame #9795: 0x000000010031d3f2 libExtG4.dylib`X4PhysicalVolume::convertTree_r(this=0x00007ffeefbfdc98, pv=0x000000010bebe8e0, parent=0x000000010e54ecb0, depth=5, parent_pv=0x000000010bebf6b0) at X4PhysicalVolume.cc:343
        frame #9796: 0x000000010031d3f2 libExtG4.dylib`X4PhysicalVolume::convertTree_r(this=0x00007ffeefbfdc98, pv=0x000000010bebf6b0, parent=0x000000010e25dd50, depth=4, parent_pv=0x000000010bec0ba0) at X4PhysicalVolume.cc:343
        frame #9797: 0x000000010031d3f2 libExtG4.dylib`X4PhysicalVolume::convertTree_r(this=0x00007ffeefbfdc98, pv=0x000000010bec0ba0, parent=0x000000010e155040, depth=3, parent_pv=0x000000010bec16f0) at X4PhysicalVolume.cc:343
        frame #9798: 0x000000010031d3f2 libExtG4.dylib`X4PhysicalVolume::convertTree_r(this=0x00007ffeefbfdc98, pv=0x000000010bec16f0, parent=0x000000010f6d60c0, depth=2, parent_pv=0x000000010bec1b10) at X4PhysicalVolume.cc:343
        frame #9799: 0x000000010031d3f2 libExtG4.dylib`X4PhysicalVolume::convertTree_r(this=0x00007ffeefbfdc98, pv=0x000000010bec1b10, parent=0x000000010bec7980, depth=1, parent_pv=0x000000010bec1cb0) at X4PhysicalVolume.cc:343
        frame #9800: 0x000000010031d3f2 libExtG4.dylib`X4PhysicalVolume::convertTree_r(this=0x00007ffeefbfdc98, pv=0x000000010bec1cb0, parent=0x0000000000000000, depth=0, parent_pv=0x0000000000000000) at X4PhysicalVolume.cc:343
        frame #9801: 0x000000010031b414 libExtG4.dylib`X4PhysicalVolume::convertStructure(this=0x00007ffeefbfdc98) at X4PhysicalVolume.cc:328
        frame #9802: 0x000000010031ad3b libExtG4.dylib`X4PhysicalVolume::init(this=0x00007ffeefbfdc98) at X4PhysicalVolume.cc:114
        frame #9803: 0x000000010031ac65 libExtG4.dylib`X4PhysicalVolume::X4PhysicalVolume(this=0x00007ffeefbfdc98, ggeo=0x000000010f686970, top=0x000000010bec1cb0) at X4PhysicalVolume.cc:102
        frame #9804: 0x000000010031a915 libExtG4.dylib`X4PhysicalVolume::X4PhysicalVolume(this=0x00007ffeefbfdc98, ggeo=0x000000010f686970, top=0x000000010bec1cb0) at X4PhysicalVolume.cc:101
        frame #9805: 0x0000000100319f71 libExtG4.dylib`X4PhysicalVolume::Convert(top=0x000000010bec1cb0) at X4PhysicalVolume.cc:77
        frame #9806: 0x000000010000f990 CX4GDMLTest`main(argc=1, argv=0x00007ffeefbfe9b0) at CX4GDMLTest.cc:72
        frame #9807: 0x00007fff55eb1015 libdyld.dylib`start + 1
    (lldb) exit
    Quitting LLDB will kill one or more processes. Do you really want to proceed: [Y/n] 
    epsilon:yoctoglrap blyth$ 
    epsilon:yoctoglrap blyth$ 
    epsilon:yoctoglrap blyth$ 
    epsilon:yoctoglrap blyth$ 
    epsilon:yoctoglrap blyth$ 
    epsilon:yoctoglrap blyth$ lldb CX4GDMLTest 

