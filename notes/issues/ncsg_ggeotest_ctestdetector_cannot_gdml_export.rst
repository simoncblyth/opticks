ncsg_ggeotest_ctestdetector_cannot_gdml_export
=================================================

FIXED : Issue
---------------

Export of test geometry to GDML fails, due to malformed bordersurface with 
missing pv reference.  


Fix required:

* avoid auto-containment motivated tree reversal, with a std::reverse in NCSG::Deserialize
* remove the reversal knockon code
* fixup parent/child hookup in GGeoTest  



::

    simon:opticks blyth$ tboolean-;tboolean-sphere-g
    (lldb) target create "CTestDetectorTest"
    2017-11-04 12:56:40.293 INFO  [2756750] [main@46] CTestDetectorTest
    2017-11-04 12:56:40.295 INFO  [2756750] [Opticks::dumpArgs@806] Opticks::configure argc 6
      0 : CTestDetectorTest
      1 : --test
      2 : --testconfig
      3 : analytic=1_csgpath=/tmp/blyth/opticks/tboolean-sphere--_name=tboolean-sphere--_mode=PyCsgInBox
      4 : --export
      5 : --dbgsurf
    2017-11-04 12:56:40.295 INFO  [2756750] [OpticksHub::configure@169] OpticksHub::configure m_gltf 0



Observations
--------------

GGeoTest created geometry
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* bb0 is being overwritten by the overall bbox 

  * have vague recollection of that
  * only now visible because moved to reversed containment order for test geometry 

* parent node id are not being set 

  * this is messing up bordersurface hookup and causing GDML export to fail
  * maybe due to adhoc solid combination, not using standard GGeoLib way of making merged mesh 
    NOPE the parent hookup needs to happen at tree level, prior to mesh creation

  * TODO: find the standard parent hookup done by GGeo and take same approach for GGeoTest 

::

    2017-11-04 12:56:40.788 INFO  [2756750] [*GMaker::makeFromCSG@188] GMaker::makeFromCSG verbosity 0 index 2 boundary-spec Vacuum///Pyrex numTris 15164 trisMsg 
    2017-11-04 12:56:40.868 INFO  [2756750] [*GMaker::makeFromCSG@188] GMaker::makeFromCSG verbosity 0 index 1 boundary-spec Rock//perfectAbsorbSurface/Vacuum numTris 19196 trisMsg 
    2017-11-04 12:56:40.947 INFO  [2756750] [*GMaker::makeFromCSG@188] GMaker::makeFromCSG verbosity 0 index 0 boundary-spec Vacuum///Rock numTris 19196 trisMsg 
    2017-11-04 12:56:40.948 INFO  [2756750] [GGeoTest::loadCSG@254] GGeoTest::loadCSG DONE 
    2017-11-04 12:56:40.948 INFO  [2756750] [*GGeoTest::combineSolids@404] GGeoTest::combineSolids START 
    2017-11-04 12:56:40.948 INFO  [2756750] [*GMergedMesh::combine@138] GMergedMesh::combine making new mesh  index 0 solids 3 verbosity 3
    2017-11-04 12:56:40.948 INFO  [2756750] [GSolid::Dump@204] GMergedMesh::combine (source solids) numSolid 3
    2017-11-04 12:56:40.948 INFO  [2756750] [GNode::dump@225] mesh.numSolids 0 mesh.ce.0 gfloat4      0.000      0.000      0.000      9.994 
    2017-11-04 12:56:40.948 INFO  [2756750] [GNode::dump@225] mesh.numSolids 0 mesh.ce.0 gfloat4      0.000      0.000      0.000     11.000 
    2017-11-04 12:56:40.948 INFO  [2756750] [GNode::dump@225] mesh.numSolids 0 mesh.ce.0 gfloat4      0.000      0.000      0.000     12.000 
    2017-11-04 12:56:40.959 FATAL [2756750] [GMergedMesh::mergeSolidIdentity@564] GMergedMesh::mergeSolidIdentity mismatch  nodeIndex 2 m_cur_solid 0
    2017-11-04 12:56:40.960 INFO  [2756750] [GParts::add@736]  n0   0 n1   1 num_part_add   1 num_tran_add   1 num_plan_add   0 other_part_buffer  1,4,4 other_tran_buffer  1,3,4,4 other_plan_buffer  0,4
    2017-11-04 12:56:40.969 INFO  [2756750] [GParts::add@736]  n0   1 n1   2 num_part_add   1 num_tran_add   1 num_plan_add   0 other_part_buffer  1,4,4 other_tran_buffer  1,3,4,4 other_plan_buffer  0,4
    2017-11-04 12:56:40.978 FATAL [2756750] [GMergedMesh::mergeSolidIdentity@564] GMergedMesh::mergeSolidIdentity mismatch  nodeIndex 0 m_cur_solid 2
    2017-11-04 12:56:40.979 INFO  [2756750] [GParts::add@736]  n0   2 n1   3 num_part_add   1 num_tran_add   1 num_plan_add   0 other_part_buffer  1,4,4 other_tran_buffer  1,3,4,4 other_plan_buffer  0,4
    2017-11-04 12:56:41.000 INFO  [2756750] [GMergedMesh::dumpSolids@706] GMergedMesh::combine (combined result)  ce0 gfloat4      0.000      0.000      0.000     12.000 
        0 ce             gfloat4      0.000      0.000      0.000     12.000  bb  mn (   -12.000    -12.000    -12.000) mx (    12.000     12.000     12.000)
        1 ce             gfloat4      0.000      0.000      0.000     11.000  bb  mn (   -11.000    -11.000    -11.000) mx (    11.000     11.000     11.000)
        2 ce             gfloat4      0.000      0.000      0.000     12.000  bb  mn (   -12.000    -12.000    -12.000) mx (    12.000     12.000     12.000)
        0 ni[nf/nv/nidx/pidx] (15164,45492,  2,4294967295)  id[nidx,midx,bidx,sidx]  (  2,  2, 33,  0) 
        1 ni[nf/nv/nidx/pidx] (19196,57588,  1,4294967295)  id[nidx,midx,bidx,sidx]  (  1,  1,123,  0) 
        2 ni[nf/nv/nidx/pidx] (19196,57588,  0,4294967295)  id[nidx,midx,bidx,sidx]  (  0,  0,  1,  0) 
    2017-11-04 12:56:41.000 INFO  [2756750] [*GGeoTest::combineSolids@437] GGeoTest::combineSolids DONE 


relationship between NCSG trees
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Full geometry uses GLTF file to represent structural relationship.  For test
geometry simple containment assumptions used.

* 

::

    delta:opticksnpy blyth$ cd /tmp/blyth/opticks/tboolean-sphere--/
    delta:tboolean-sphere-- blyth$ l
    total 8
    -rw-r--r--  1 blyth  wheel   62 Nov  4 12:56 csg.txt
    drwxr-xr-x  8 blyth  wheel  272 Nov  2 18:55 2
    drwxr-xr-x  8 blyth  wheel  272 Nov  2 12:46 0
    drwxr-xr-x  8 blyth  wheel  272 Nov  2 12:46 1
    delta:tboolean-sphere-- blyth$ 
    delta:tboolean-sphere-- blyth$ cat csg.txt 
    Vacuum///Rock
    Rock//perfectAbsorbSurface/Vacuum
    Vacuum///Pyrex





setParent
~~~~~~~~~~~

::

    delta:opticks blyth$ opticks-find ">setParent" 
    ./assimprap/AssimpGGeo.cc:        solid->setParent(parent);
    ./assimprap/AssimpTree.cc:       wrap->setParent(parent);
    ./ggeo/GScene.cc:    node->setParent(parent) ;   // tree hookup 
    ./opticksnpy/MultiViewNPY.cpp:    vec->setParent(this);


Recursive method passes parent GSolid along to allow hookup

::

     818 void AssimpGGeo::convertStructure(GGeo* gg, AssimpNode* node, unsigned int depth, GSolid* parent)
     819 {
     820     // recursive traversal of the AssimpNode tree
     821     // note that full tree is traversed even when a partial selection is applied 
     822 
     823 
     824     GSolid* solid = convertStructureVisit( gg, node, depth, parent);
     825 
     826     bool selected = m_nosel ? true : m_selection && m_selection->contains(node) ;   // twas hotspot for geocache creation before nosel special case
     827 
     828     solid->setSelected(selected);
     829 
     830     gg->add(solid);
     831 
     832     if(parent) // GNode hookup
     833     {
     834         parent->addChild(solid);
     835         solid->setParent(parent);
     836     }
     837     else
     838     {
     839         assert(node->getIndex() == 0);   // only root node has no parent 
     840     }
     841 
     842     for(unsigned int i = 0; i < node->getNumChildren(); i++) convertStructure(gg, node->getChild(i), depth + 1, solid);
     843 }




CDetector::attachSurfaces fail
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    2017-11-04 12:56:41.847 INFO  [2756750] [CTraverser::Summary@106] CDetector::traverse numMaterials 3 numMaterialsWithoutMPT 0
    2017-11-04 12:56:41.847 INFO  [2756750] [CDetector::traverse@101] [--dbgsurf] CDetector::traverse DONE 
    2017-11-04 12:56:41.847 INFO  [2756750] [CDetector::attachSurfaces@266] [--dbgsurf] CDetector::attachSurfaces START closing gsurlib, creating csurlib  
    2017-11-04 12:56:41.847 INFO  [2756750] [GSurLib::close@134] [--dbgsurf] GSurLib::close START 
    2017-11-04 12:56:41.847 INFO  [2756750] [GSurLib::examineSolidBndSurfaces@189] [--dbgsurf] GSurLib::examineSolidBndSurfaces numSolids 3 mm 0x10bb124f0
    2017-11-04 12:56:41.847 INFO  [2756750] [GSurLib::examineSolidBndSurfaces@213] GSurLib::examineSolidBndSurfaces [--dbgsurf]  numSolids 3
     j      0 i(so-idx)      2 lv box_log0
     j      1 i(so-idx)      1 lv box_log1
     j      2 i(so-idx)      0 lv sphere_log2
     j      0 i(so-idx)      2 node(ni.z)      0 node2(id.x)      0 boundary(id.z)      1 parent(ni.w) 4294967295 nodeinfo  (19196,57588,  0,4294967295)  bname Vacuum///Rock
     j      1 i(so-idx)      1 node(ni.z)      1 node2(id.x)      1 boundary(id.z)    123 parent(ni.w) 4294967295 nodeinfo  (19196,57588,  1,4294967295)  bname Rock//perfectAbsorbSurface/Vacuum isur
     j      2 i(so-idx)      0 node(ni.z)      2 node2(id.x)      2 boundary(id.z)     33 parent(ni.w) 4294967295 nodeinfo  (15164,45492,  2,4294967295)  bname Vacuum///Pyrex
    2017-11-04 12:56:41.848 INFO  [2756750] [GSurLib::examineSolidBndSurfaces@286]  node_mismatch 0 node2_mismatch 0
    2017-11-04 12:56:41.848 INFO  [2756750] [GSurLib::close@141] [--dbgsurf] GSurLib::close DONE 
    2017-11-04 12:56:41.848 INFO  [2756750] [CSurLib::convert@136] [--dbgsurf] CSurLib::convert  numSur 48
    2017-11-04 12:56:41.848 INFO  [2756750] [*CSurLib::makeBorderSurface@225] CSurLib::makeBorderSurface name perfectAbsorbSurface ipv1 1 ipv2 4294967295
    Assertion failed: (ipv2 != GSurLib::UNSET && "CSurLib::makeBorderSurface ipv2 UNSET"), function makeBorderSurface, file /Users/blyth/opticks/cfg4/CSurLib.cc, line 234.
    Process 68667 stopped
    * thread #1: tid = 0x2a108e, 0x00007fff8cc60866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff8cc60866 libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill + 10:
    -> 0x7fff8cc60866:  jae    0x7fff8cc60870            ; __pthread_kill + 20
       0x7fff8cc60868:  movq   %rax, %rdi
       0x7fff8cc6086b:  jmp    0x7fff8cc5d175            ; cerror_nocancel
       0x7fff8cc60870:  retq   
    (lldb) 






try checking the G4 geometry by exporting it 
-----------------------------------------------

* this motivated adding some asserts for earlier warning of bordersurface issues

::


    simon:optickscore blyth$ tboolean-;tboolean-sphere-g --export 
    (lldb) target create "CTestDetectorTest"
    Current executable set to 'CTestDetectorTest' (x86_64).
    (lldb) settings set -- target.run-args  "--test" "--testconfig" "analytic=1_csgpath=/tmp/blyth/opticks/tboolean-sphere--_name=tboolean-sphere--_mode=PyCsgInBox" "--export"
    (lldb) r
    Process 64968 launched: '/usr/local/opticks/lib/CTestDetectorTest' (x86_64)
    2017-11-02 18:44:35.529 INFO  [2406779] [main@42] CTestDetectorTest
      0 : CTestDetectorTest
      1 : --test
      2 : --testconfig
      3 : analytic=1_csgpath=/tmp/blyth/opticks/tboolean-sphere--_name=tboolean-sphere--_mode=PyCsgInBox
      4 : --export
    2017-11-02 18:44:35.705 INFO  [2406779] [NSensorList::read@186] NSensorList::read  found 6888 sensors. 


    (lldb) f 4
    frame #4: 0x0000000101c776d3 libG4persistency.dylib`G4GDMLWriteStructure::BorderSurfaceCache(this=0x000000010da00800, bsurf=<unavailable>) + 291 at G4GDMLWriteStructure.cc:245
       242  
       243     const G4String volumeref1 = GenerateName(bsurf->GetVolume1()->GetName(),
       244                                              bsurf->GetVolume1());
    -> 245     const G4String volumeref2 = GenerateName(bsurf->GetVolume2()->GetName(),
       246                                              bsurf->GetVolume2());
       247     xercesc::DOMElement* volumerefElement1 = NewElement("physvolref");
       248     xercesc::DOMElement* volumerefElement2 = NewElement("physvolref");
    (lldb) f 5
    frame #5: 0x0000000101c791af libG4persistency.dylib`G4GDMLWriteStructure::TraverseVolumeTree(this=0x000000010da00800, volumePtr=0x0000000112f43770, depth=0) + 4367 at G4GDMLWriteStructure.cc:525
       522                     
       523             PhysvolWrite(volumeElement,physvol,invR*P*daughterR,ModuleName);
       524           }
    -> 525         BorderSurfaceCache(GetBorderSurface(physvol));
       526       }
       527  
       528     if (cexport)  { ExportEnergyCuts(volumePtr); }
    (lldb) bt
    * thread #1: tid = 0x24b97b, 0x0000000101c606cb libG4persistency.dylib`G4GDMLWrite::GenerateName(G4String const&, void const*) [inlined] std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >::__get_short_size() const at string:1683, queue = 'com.apple.main-thread', stop reason = EXC_BAD_ACCESS (code=1, address=0x18)
        frame #0: 0x0000000101c606cb libG4persistency.dylib`G4GDMLWrite::GenerateName(G4String const&, void const*) [inlined] std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >::__get_short_size() const at string:1683
        frame #1: 0x0000000101c606cb libG4persistency.dylib`G4GDMLWrite::GenerateName(G4String const&, void const*) [inlined] std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >::size() const at string:1398
        frame #2: 0x0000000101c606cb libG4persistency.dylib`G4GDMLWrite::GenerateName(G4String const&, void const*) [inlined] std::__1::basic_stringstream<char, std::__1::char_traits<char>, std::__1::allocator<char> >::basic_stringstream(this=0x0000000101cb31a8, __wch=<unavailable>) at ostream:1068
        frame #3: 0x0000000101c606cb libG4persistency.dylib`G4GDMLWrite::GenerateName(this=0x0000000000000000, name=0x0000000000000018, ptr=0x0000000000000000) + 331 at G4GDMLWrite.cc:126
        frame #4: 0x0000000101c776d3 libG4persistency.dylib`G4GDMLWriteStructure::BorderSurfaceCache(this=0x000000010da00800, bsurf=<unavailable>) + 291 at G4GDMLWriteStructure.cc:245
      * frame #5: 0x0000000101c791af libG4persistency.dylib`G4GDMLWriteStructure::TraverseVolumeTree(this=0x000000010da00800, volumePtr=0x0000000112f43770, depth=0) + 4367 at G4GDMLWriteStructure.cc:525
        frame #6: 0x0000000101c612d3 libG4persistency.dylib`G4GDMLWrite::Write(this=0x000000010da00800, fname=0x00007fff5fbfdad8, logvol=0x0000000112f43770, setSchemaLocation=<unavailable>, depth=0, refs=<unavailable>) + 1587 at G4GDMLWrite.cc:228
        frame #7: 0x000000010171176c libcfg4.dylib`G4GDMLParser::Write(this=0x0000000112f8d880, filename=0x00007fff5fbfdad8, pvol=0x0000000112f42200, refs=true, schemaLocation=0x00007fff5fbfd950) + 236 at G4GDMLParser.icc:68
        frame #8: 0x00000001017109a7 libcfg4.dylib`CDetector::export_gdml(this=0x0000000112f3c600, path_=0x0000000112f9a0a0) + 599 at CDetector.cc:309
        frame #9: 0x000000010168a436 libcfg4.dylib`CGeometry::export_(this=0x0000000112f3c590) + 1558 at CGeometry.cc:155
        frame #10: 0x0000000101689e06 libcfg4.dylib`CGeometry::postinitialize(this=0x0000000112f3c590) + 438 at CGeometry.cc:123
        frame #11: 0x0000000101736d0b libcfg4.dylib`CG4::postinitialize(this=0x00007fff5fbfe840) + 683 at CG4.cc:221
        frame #12: 0x00000001017369fc libcfg4.dylib`CG4::initialize(this=0x00007fff5fbfe840) + 540 at CG4.cc:176
        frame #13: 0x00000001017367a5 libcfg4.dylib`CG4::init(this=0x00007fff5fbfe840) + 21 at CG4.cc:150
        frame #14: 0x000000010173677c libcfg4.dylib`CG4::CG4(this=0x00007fff5fbfe840, hub=0x00007fff5fbfe8f8) + 1564 at CG4.cc:143
        frame #15: 0x00000001017367cd libcfg4.dylib`CG4::CG4(this=0x00007fff5fbfe840, hub=0x00007fff5fbfe8f8) + 29 at CG4.cc:144
        frame #16: 0x000000010000ca29 CTestDetectorTest`main(argc=5, argv=0x00007fff5fbfecb8) + 969 at CTestDetectorTest.cc:53
        frame #17: 0x00007fff880d35fd libdyld.dylib`start + 1
    (lldb) 


::

    g4-cls G4GDMLWriteStructure


Hmm probably because have a border surface on the world.

Argh ... nope need to rejig GSurLib to work with analytic geometry.





