Geant4_update_to_10_4_2
=========================


* theParticleIterator
* needed to const_cast G4MaterialPropertiesTable
* getting NULL dynamicparticle

* X4PhysicsVector::Digest reproducibly SIGABRT on macOS, workaround get digest from converted GProperty 


ckm- polygonize crash, and dont have debug symbols ?? for G4 

::

    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = EXC_BAD_ACCESS (code=1, address=0x48)
      * frame #0: 0x00000001029d02b9 libG4geometry.dylib`G4Box::GetExtent() const + 9
        frame #1: 0x000000010066ad2d libExtG4.dylib`X4Mesh::polygonize(this=0x00007ffeefbfc7c0) at X4Mesh.cc:128
        frame #2: 0x0000000100669fbf libExtG4.dylib`X4Mesh::init(this=0x00007ffeefbfc7c0) at X4Mesh.cc:93
        frame #3: 0x0000000100669f92 libExtG4.dylib`X4Mesh::X4Mesh(this=0x00007ffeefbfc7c0, solid=0x000000010e297bd0) at X4Mesh.cc:83
        frame #4: 0x0000000100669f0d libExtG4.dylib`X4Mesh::X4Mesh(this=0x00007ffeefbfc7c0, solid=0x000000010e297bd0) at X4Mesh.cc:82
        frame #5: 0x0000000100669ebc libExtG4.dylib`X4Mesh::Convert(solid=0x000000010e297bd0) at X4Mesh.cc:67

X4MeshTest has the same issue with GetExtent()::

    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = EXC_BAD_ACCESS (code=1, address=0x78)
      * frame #0: 0x0000000102491f09 libG4geometry.dylib`G4Sphere::GetExtent() const + 9
        frame #1: 0x000000010010dd2d libExtG4.dylib`X4Mesh::polygonize(this=0x0000000106814740) at X4Mesh.cc:128
        frame #2: 0x000000010010cfbf libExtG4.dylib`X4Mesh::init(this=0x0000000106814740) at X4Mesh.cc:93
        frame #3: 0x000000010010cf92 libExtG4.dylib`X4Mesh::X4Mesh(this=0x0000000106814740, solid=0x00000001068147d0) at X4Mesh.cc:83
        frame #4: 0x000000010010cf0d libExtG4.dylib`X4Mesh::X4Mesh(this=0x0000000106814740, solid=0x00000001068147d0) at X4Mesh.cc:82
        frame #5: 0x000000010000da58 X4MeshTest`main(argc=2, argv=0x00007ffeefbfea88) at X4MeshTest.cc:16
        frame #6: 0x00007fff533b2015 libdyld.dylib`start + 1
        frame #7: 0x00007fff533b2015 libdyld.dylib`start + 1
    (lldb) 


Nuclear option as g4-configure omitted to setup Debug build.::

   g4-wipe
   g4--






