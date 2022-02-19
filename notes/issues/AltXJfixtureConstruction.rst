AltXJfixtureConstruction
===========================

Looks like Z-shift transforms present in G4VSolid are not getting thru the GeoChain:: 

     geom  ## check that only one entry is uncommented in ~/.opticks/GEOM.txt eg AltXJfixtureConstruction_YZ

 
All the below scripts access the GEOM.txt file to set the geometry to create, visualize or shoot single rays at:: 
  
     gc ; ./run.sh 

     x4 ; ./X4MeshTest.sh    ## CPU : Geant4 polygons visualized with pyvista

     x4 ; ./xxs.sh           ## CPU : 2D Geant4 intersects visualized with matplotlib and/or pyvista

     c ; ./sdf_geochain.sh   ## CPU : 3D Opticks distance field visualised with pyvista iso-surface finding 

     c ; ./csg_geochain.sh   ## CPU : 2D(or 3D) pyvista visualization of Opticks intersects (CPU test run of CUDA comparible intersect code)

     cx ; ./cxr_geochain.sh  ## GPU : 3D OptiX/Opticks render of geometry      



     c ; ./CSGQueryTest.sh   ## CPU : test mostly used for shooting single rays at geometry, useful after compiling with DEBUG flag enabled   




* Added CrossHairs to X4MeshTest.sh and sdf_geochain.sh : clearly zero level in different place



XJFixtureConstruction assert : not expecting more than one level of translation
----------------------------------------------------------------------------------

::

    2022-02-19 15:40:20.119 FATAL [5281713] [GeoChain::convertSolid@65] [
    2022-02-19 15:40:20.119 INFO  [5281713] [GeoChain::convertSolid@67] meta.empty
    X4SolidTree::BooleanClone expect_tla ERROR (not expecting more than one level of translation) 
    X4SolidTree::BooleanClone tla( 0 0 -25) 
    Assertion failed: (expect_tla), function BooleanClone, file /Users/blyth/opticks/extg4/X4SolidTree.cc, line 1943.
    Process 52331 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff72958b66 libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill:
    ->  0x7fff72958b66 <+10>: jae    0x7fff72958b70            ; <+20>
        0x7fff72958b68 <+12>: movq   %rax, %rdi
        0x7fff72958b6b <+15>: jmp    0x7fff7294fae9            ; cerror_nocancel
        0x7fff72958b70 <+20>: retq   
    Target 0: (GeoChainSolidTest) stopped.

    Process 52331 launched: '/usr/local/opticks/lib/GeoChainSolidTest' (x86_64)
    (lldb) bt
        frame #3: 0x00007fff7287c1ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x00000001001b19ed libExtG4.dylib`X4SolidTree::BooleanClone(solid=0x000000010852efe0, depth=1, rot=0x00007ffeefbfcb50, tla=0x00007ffeefbfcb20) at X4SolidTree.cc:1943
        frame #5: 0x00000001001b14ab libExtG4.dylib`X4SolidTree::DeepClone_r(node_=0x000000010852f290, depth=1, rot=0x00007ffeefbfcb50, tla=0x00007ffeefbfcb20) at X4SolidTree.cc:1889
        frame #6: 0x00000001001b1be0 libExtG4.dylib`X4SolidTree::BooleanClone(solid=0x000000010852f1c0, depth=0, rot=0x0000000000000000, tla=0x0000000000000000) at X4SolidTree.cc:1952
        frame #7: 0x00000001001b14ab libExtG4.dylib`X4SolidTree::DeepClone_r(node_=0x000000010852f1c0, depth=0, rot=0x0000000000000000, tla=0x0000000000000000) at X4SolidTree.cc:1889
        frame #8: 0x000000010019fa37 libExtG4.dylib`X4SolidTree::DeepClone(solid=0x000000010852f1c0) at X4SolidTree.cc:1845
        frame #9: 0x000000010019ee2d libExtG4.dylib`X4SolidTree::X4SolidTree(this=0x000000010852fa20, original_=0x000000010852f1c0) at X4SolidTree.cc:59
        frame #10: 0x000000010019dc5d libExtG4.dylib`X4SolidTree::X4SolidTree(this=0x000000010852fa20, original_=0x000000010852f1c0) at X4SolidTree.cc:88
        frame #11: 0x000000010019de6f libExtG4.dylib`X4SolidTree::Draw(original=0x000000010852f1c0, msg="GeoChain::convertSolid original G4VSolid tree") at X4SolidTree.cc:50
        frame #12: 0x00000001000dc54f libGeoChain.dylib`GeoChain::convertSolid(this=0x00007ffeefbfe1a0, solid=0x000000010852f1c0, meta="") at GeoChain.cc:70
        frame #13: 0x000000010000dc22 GeoChainSolidTest`main(argc=1, argv=0x00007ffeefbfe790) at GeoChainSolidTest.cc:83
        frame #14: 0x00007fff72808015 libdyld.dylib`start + 1
    (lldb) 




