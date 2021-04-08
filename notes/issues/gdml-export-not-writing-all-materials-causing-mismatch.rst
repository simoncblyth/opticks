gdml-export-not-writing-all-materials-causing-mismatch
=========================================================

Issue
--------

Geant4 gdml export is only writing out used materials whereas the 
Opticks conversion does all materials of the geometry.

* YES : so what : where does the error happen ?


History of the issue
------------------------

* prior :doc:`cluster-opticks-t-shakedown`


1. opticks-t test fails on lxslc7 when using the live OPTICKS_KEY geocache 
   due to a lack of gdmlpath on the commanline read from geocache metadata


::

      3  /38  Test #3  : CFG4Test.CTestDetectorTest                    ***Exception: SegFault         2.86   
      5  /38  Test #5  : CFG4Test.CGDMLDetectorTest                    Subprocess aborted***Exception:   2.71   
      7  /38  Test #7  : CFG4Test.CGeometryTest                        Subprocess aborted***Exception:   2.74   
      8  /38  Test #8  : CFG4Test.CG4Test                              ***Exception: SegFault         2.79   
      26 /38  Test #26 : CFG4Test.CInterpolationTest                   ***Exception: SegFault         2.79   
      32 /38  Test #32 : CFG4Test.CRandomEngineTest                    ***Exception: SegFault         2.82   
      1  /1   Test #1  : OKG4Test.OKG4Test                             ***Exception: SegFault         2.91   


2. attempt at solution by always saving an "origin.gdml" into the geocache.  
3. but having done that now find mismatch between geocache materials and materials from the "origin.gdml"


Action : making Opticks material conversion handle only the materials that GDML export does
-----------------------------------------------------------------------------------------------







Test : geocache-dbg : hmm looks like have not completed the change to X4PhysicalVolume::convertMaterials
-----------------------------------------------------------------------------------------------------------

::

    epsilon:opticks blyth$ geocache-
    epsilon:opticks blyth$ geocache-dbg 
    === o-cmdline-parse 1 : START
    === o-cmdline-specials 1 :
    === o-cmdline-specials 1 :
    === o-cmdline-binary-match 1 : finding 1st argument with associated binary
    === o-cmdline-binary-match 1 : --okx4test
    === o-cmdline-parse 1 : DONE

         OPTICKS_CMD    : --okx4test 
         OPTICKS_BINARY : /usr/local/opticks/lib/OKX4Test
         OPTICKS_ARGS   : --okx4test --g4codegen --deletegeocache --gdmlpath /Users/blyth/origin.gdml -D

    2021-04-08 14:33:17.301 INFO  [15122955] [X4PhysicalVolume::convertStructure@839] [ creating large tree of GVolume instances
    Assertion failed: (isClosed() && " must close the lib before the indices can be used, as preference sort order may be applied at the close"), function getIndex, file /Users/blyth/opticks/ggeo/GPropertyLib.cc, line 399.
    Process 84135 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff53644b66 libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill:
    ->  0x7fff53644b66 <+10>: jae    0x7fff53644b70            ; <+20>
        0x7fff53644b68 <+12>: movq   %rax, %rdi
        0x7fff53644b6b <+15>: jmp    0x7fff5363bae9            ; cerror_nocancel
        0x7fff53644b70 <+20>: retq   
    Target 0: (OKX4Test) stopped.

    Process 84135 launched: '/usr/local/opticks/lib/OKX4Test' (x86_64)
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff53644b66 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff5380f080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff535a01ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff535681ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x000000010999b2ef libGGeo.dylib`GPropertyLib::getIndex(this=0x0000000114d85e80, shortname="Galactic") const at GPropertyLib.cc:399
        frame #5: 0x00000001099dd574 libGGeo.dylib`GBndLib::add(this=0x0000000114d85ab0, omat_="Galactic", osur_=0x0000000000000000, isur_=0x0000000000000000, imat_="Galactic") at GBndLib.cc:465
        frame #6: 0x00000001099dd517 libGGeo.dylib`GBndLib::addBoundary(this=0x0000000114d85ab0, omat="Galactic", osur=0x0000000000000000, isur=0x0000000000000000, imat="Galactic") at GBndLib.cc:453
        frame #7: 0x00000001038d9c66 libExtG4.dylib`X4PhysicalVolume::addBoundary(this=0x00007ffeefbfc7e0, pv=0x0000000124708a80, pv_p=0x0000000000000000) at X4PhysicalVolume.cc:1059
        frame #8: 0x00000001038d7c9a libExtG4.dylib`X4PhysicalVolume::convertNode(this=0x00007ffeefbfc7e0, pv=0x0000000124708a80, parent=0x0000000000000000, depth=0, pv_p=0x0000000000000000, recursive_select=0x00007ffeefbfb9c3) at X4PhysicalVolume.cc:1135
        frame #9: 0x00000001038d7a4d libExtG4.dylib`X4PhysicalVolume::convertStructure_r(this=0x00007ffeefbfc7e0, pv=0x0000000124708a80, parent=0x0000000000000000, depth=0, parent_pv=0x0000000000000000, recursive_select=0x00007ffeefbfb9c3) at X4PhysicalVolume.cc:919
        frame #10: 0x00000001038d16fc libExtG4.dylib`X4PhysicalVolume::convertStructure(this=0x00007ffeefbfc7e0) at X4PhysicalVolume.cc:850
        frame #11: 0x00000001038d0773 libExtG4.dylib`X4PhysicalVolume::init(this=0x00007ffeefbfc7e0) at X4PhysicalVolume.cc:189
        frame #12: 0x00000001038d043c libExtG4.dylib`X4PhysicalVolume::X4PhysicalVolume(this=0x00007ffeefbfc7e0, ggeo=0x0000000114d857d0, top=0x0000000124708a80) at X4PhysicalVolume.cc:171
        frame #13: 0x00000001038cf595 libExtG4.dylib`X4PhysicalVolume::X4PhysicalVolume(this=0x00007ffeefbfc7e0, ggeo=0x0000000114d857d0, top=0x0000000124708a80) at X4PhysicalVolume.cc:162
        frame #14: 0x0000000100015717 OKX4Test`main(argc=7, argv=0x00007ffeefbfcff0) at OKX4Test.cc:108
        frame #15: 0x00007fff534f4015 libdyld.dylib`start + 1
    (lldb) 







Check G4 GDML Export of Materials
------------------------------------

::

    epsilon:src blyth$ g4-cls G4GDMLWriteMaterials
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02
    vi -R source/persistency/gdml/include/G4GDMLWriteMaterials.hh source/persistency/gdml/src/G4GDMLWriteMaterials.cc
    2 files to edit


    318 void G4GDMLWriteMaterials::AddMaterial(const G4Material* const materialPtr)
    319 {
    320    for (size_t i=0;i<materialList.size();i++)    // Check if material is
    321    {                                             // already in the list!
    322       if (materialList[i] == materialPtr)  { return; }
    323    }
    324    materialList.push_back(materialPtr);
    325    MaterialWrite(materialPtr);
    326 }


    epsilon:src blyth$ grep AddMaterial *.*
    G4GDMLReadMaterials.cc:         if (materialPtr != 0) { material->AddMaterial(materialPtr,n); }
    G4GDMLWriteMaterials.cc:void G4GDMLWriteMaterials::AddMaterial(const G4Material* const materialPtr)
    G4GDMLWriteStructure.cc:   AddMaterial(volumePtr->GetMaterial());
    epsilon:src blyth$ 


AddMaterial invoked in the recursive tail of TraverseVolumeTree::

    388 G4Transform3D G4GDMLWriteStructure::
    389 TraverseVolumeTree(const G4LogicalVolume* const volumePtr, const G4int depth)
    390 {
    391    if (VolumeMap().find(volumePtr) != VolumeMap().end())
    392    { 
    393      return VolumeMap()[volumePtr]; // Volume is already processed
    394    }


    395   
    396    G4VSolid* solidPtr = volumePtr->GetSolid();
    397    G4Transform3D R,invR;
    398    G4int trans=0;
    399 
    400    std::map<const G4LogicalVolume*, G4GDMLAuxListType>::iterator auxiter;
    401 
    402    levelNo++;
    481    for (G4int i=0;i<daughterCount;i++)   // Traverse all the children!
    482      {
    483        const G4VPhysicalVolume* const physvol = volumePtr->GetDaughter(i);
    484        const G4String ModuleName = Modularize(physvol,depth);
    485 
    486        G4Transform3D daughterR;
    487 
    488        if (ModuleName.empty())   // Check if subtree requested to be 
    489      {                         // a separate module!
    490        daughterR = TraverseVolumeTree(physvol->GetLogicalVolume(),depth+1);
    491      }
    492        else
    493      {
    494        G4GDMLWriteStructure writer;
    495        daughterR = writer.Write(ModuleName,physvol->GetLogicalVolume(),
    496                     SchemaLocation,depth+1);
    497      }

    ...
    567    AddExtension(volumeElement, volumePtr);
    568    // Add any possible user defined extension attached to a volume
    569      
    570    AddMaterial(volumePtr->GetMaterial());
    571    // Add the involved materials and solids!
    572      
    573    AddSolid(solidPtr);
    574      
    575    SkinSurfaceCache(GetSkinSurface(volumePtr));
    576      
    577    return R;
    578 }




