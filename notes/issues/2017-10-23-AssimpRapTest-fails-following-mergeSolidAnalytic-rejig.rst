2017-10-23-AssimpRapTest-fails-following-mergeSolidAnalytic-rejig
====================================================================


FIXED : GParts m_parts companion to GSolid is usually NULL
------------------------------------------------------------





assimprap-t
-------------

::

    simon:opticks blyth$ assimprap-t
    Test project /usr/local/opticks/build/assimprap
        Start 1: AssimpRapTest.AssimpRapTest
    1/3 Test #1: AssimpRapTest.AssimpRapTest ........***Exception: Other  2.71 sec
        Start 2: AssimpRapTest.AssimpImporterTest
    2/3 Test #2: AssimpRapTest.AssimpImporterTest ...   Passed    0.73 sec
        Start 3: AssimpRapTest.AssimpGGeoTest
    3/3 Test #3: AssimpRapTest.AssimpGGeoTest .......   Passed    1.84 sec

    67% tests passed, 1 tests failed out of 3

    Total Test time (real) =   5.28 sec

    The following tests FAILED:
          1 - AssimpRapTest.AssimpRapTest (OTHER_FAULT)
    Errors while running CTest
    opticks-t- : use -V to show output
    simon:opticks blyth$ 



AssimpRapTest
--------------

GParts pts coming up NULL for a solid deep in tree.

::

    simon:opticks blyth$ lldb AssimpRapTest
    (lldb) target create "AssimpRapTest"
    Current executable set to 'AssimpRapTest' (x86_64).
    (lldb) r
    Process 47829 launched: '/usr/local/opticks/lib/AssimpRapTest' (x86_64)
    SAr _argc 1 (  AssimpRapTest ) 
    2017-10-23 11:27:42.033 INFO  [171757] [OpticksQuery::dump@79] OpticksQuery::init queryType range query_string range:3153:12221 query_name NULL query_index 0 query_depth 0 no_selection 0 nrange 2 : 3153 : 12221
    2017-10-23 11:27:42.033 INFO  [171757] [Opticks::init@327] Opticks::init DONE OpticksResource::desc digest 96ff965744a2f6b78c24e33c80d3a4cd age.tot_seconds 4754479 age.tot_minutes 79241.320 age.tot_hours 1320.689 age.tot_days     55.029
    2017-10-23 11:27:42.033 INFO  [171757] [main@69] ok
    2017-10-23 11:27:42.033 INFO  [171757] [Opticks::dumpArgs@768] Opticks::configure argc 1
      0 : AssimpRapTest
    2017-10-23 11:27:42.034 INFO  [171757] [Opticks::configure@836] Opticks::configure  m_size 2880,1704,2,0 m_position 200,200,0,0 prefdir $HOME/.opticks/dayabay/State
    2017-10-23 11:27:42.034 INFO  [171757] [Opticks::configure@857] Opticks::configure DONE  verbosity 0
    2017-10-23 11:27:42.200 INFO  [171757] [NSensorList::read@186] NSensorList::read  found 6888 sensors. 
    2017-10-23 11:27:42.203 INFO  [171757] [NLODConfig::NLODConfig@16] NLODConfig::NLODConfig cfg [levels=3,verbosity=3]
    2017-10-23 11:27:42.203 INFO  [171757] [NSceneConfig::NSceneConfig@50] NSceneConfig::NSceneConfig cfg [check_surf_containment=0,check_aabb_containment=0,instance_repeat_min=400,instance_vertex_min=0]
    after gg
    2017-10-23 11:27:42.206 INFO  [171757] [AssimpGGeo::load@133] AssimpGGeo::load  path /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.dae query range:3153:12221 ctrl volnames verbosity 0
    2017-10-23 11:27:42.206 INFO  [171757] [AssimpImporter::import@195] AssimpImporter::import path /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.dae flags 32779
    2017-10-23 11:27:42.849 INFO  [171757] [AssimpImporter::Summary@112] AssimpImporter::import DONE
    2017-10-23 11:27:42.849 INFO  [171757] [AssimpImporter::Summary@113] AssimpImporter::info m_aiscene  NumMaterials 78 NumMeshes 249
    2017-10-23 11:27:42.902 INFO  [171757] [AssimpGGeo::load@148] AssimpGGeo::load select START 
    ...
    2017-10-23 11:27:44.461 INFO  [171757] [GMergedMesh::mergeSolidAnalytic@552] GMergedMesh::mergeSolidAnalytic pts -
    2017-10-23 11:27:44.461 FATAL [171757] [GMergedMesh::mergeSolidAnalytic@559] GMergedMesh::mergeSolidAnalytic pts NULL 
    Assertion failed: (pts), function mergeSolidAnalytic, file /Users/blyth/opticks/ggeo/GMergedMesh.cc, line 561.
    Process 47829 stopped
    * thread #1: tid = 0x29eed, 0x00007fff8cc60866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff8cc60866 libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill + 10:
    -> 0x7fff8cc60866:  jae    0x7fff8cc60870            ; __pthread_kill + 20
       0x7fff8cc60868:  movq   %rax, %rdi
       0x7fff8cc6086b:  jmp    0x7fff8cc5d175            ; cerror_nocancel
       0x7fff8cc60870:  retq   

    (lldb) bt
    * thread #1: tid = 0x29eed, 0x00007fff8cc60866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff8cc60866 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff842fd35c libsystem_pthread.dylib`pthread_kill + 92
        frame #2: 0x00007fff8b04db1a libsystem_c.dylib`abort + 125
        frame #3: 0x00007fff8b0179bf libsystem_c.dylib`__assert_rtn + 321
        frame #4: 0x00000001016da158 libGGeo.dylib`GMergedMesh::mergeSolidAnalytic(this=0x000000010dc024e0, pts=0x0000000000000000, transform=0x000000010aae9960, verbosity=0) + 808 at GMergedMesh.cc:561
        frame #5: 0x00000001016d791f libGGeo.dylib`GMergedMesh::mergeSolid(this=0x000000010dc024e0, solid=0x000000010aae9da0, selected=true, verbosity=0) + 1199 at GMergedMesh.cc:425
        frame #6: 0x00000001016d8cad libGGeo.dylib`GMergedMesh::traverse_r(this=0x000000010dc024e0, node=0x000000010aae9da0, depth=8, pass=1, verbosity=0) + 1485 at GMergedMesh.cc:617
        frame #7: 0x00000001016d8d38 libGGeo.dylib`GMergedMesh::traverse_r(this=0x000000010dc024e0, node=0x000000010aae9100, depth=7, pass=1, verbosity=0) + 1624 at GMergedMesh.cc:621
        frame #8: 0x00000001016d8d38 libGGeo.dylib`GMergedMesh::traverse_r(this=0x000000010dc024e0, node=0x000000010aae85d0, depth=6, pass=1, verbosity=0) + 1624 at GMergedMesh.cc:621
        frame #9: 0x00000001016d8d38 libGGeo.dylib`GMergedMesh::traverse_r(this=0x000000010dc024e0, node=0x000000010aae7aa0, depth=5, pass=1, verbosity=0) + 1624 at GMergedMesh.cc:621
        frame #10: 0x00000001016d8d38 libGGeo.dylib`GMergedMesh::traverse_r(this=0x000000010dc024e0, node=0x000000010aae6ff0, depth=4, pass=1, verbosity=0) + 1624 at GMergedMesh.cc:621
        frame #11: 0x00000001016d8d38 libGGeo.dylib`GMergedMesh::traverse_r(this=0x000000010dc024e0, node=0x000000010aae66a0, depth=3, pass=1, verbosity=0) + 1624 at GMergedMesh.cc:621
        frame #12: 0x00000001016d8d38 libGGeo.dylib`GMergedMesh::traverse_r(this=0x000000010dc024e0, node=0x000000010aae6060, depth=2, pass=1, verbosity=0) + 1624 at GMergedMesh.cc:621
        frame #13: 0x00000001016d8d38 libGGeo.dylib`GMergedMesh::traverse_r(this=0x000000010dc024e0, node=0x0000000109d6a8d0, depth=1, pass=1, verbosity=0) + 1624 at GMergedMesh.cc:621
        frame #14: 0x00000001016d8d38 libGGeo.dylib`GMergedMesh::traverse_r(this=0x000000010dc024e0, node=0x0000000109d62030, depth=0, pass=1, verbosity=0) + 1624 at GMergedMesh.cc:621
        frame #15: 0x00000001016d8655 libGGeo.dylib`GMergedMesh::create(ridx=0, base=0x0000000000000000, root=0x0000000109d62030, verbosity=0) + 1221 at GMergedMesh.cc:206
        frame #16: 0x00000001016b4488 libGGeo.dylib`GGeoLib::makeMergedMesh(this=0x0000000104465d90, index=0, base=0x0000000000000000, root=0x0000000109d62030, verbosity=0) + 504 at GGeoLib.cc:239
        frame #17: 0x00000001016cc132 libGGeo.dylib`GTreeCheck::makeMergedMeshAndInstancedBuffers(this=0x0000000104466080, verbosity=0) + 162 at GTreeCheck.cc:444
        frame #18: 0x00000001016cbb62 libGGeo.dylib`GTreeCheck::createInstancedMergedMeshes(this=0x0000000104466080, delta=true, verbosity=0) + 242 at GTreeCheck.cc:80
        frame #19: 0x00000001016e43e0 libGGeo.dylib`GGeo::prepareMeshes(this=0x0000000104300de0) + 368 at GGeo.cc:1299
        frame #20: 0x00000001016e2b82 libGGeo.dylib`GGeo::loadFromG4DAE(this=0x0000000104300de0) + 354 at GGeo.cc:607
        frame #21: 0x0000000100004e80 AssimpRapTest`main(argc=1, argv=0x00007fff5fbfee10) + 1584 at AssimpRapTest.cc:85
        frame #22: 0x00007fff880d35fd libdyld.dylib`start + 1
    (lldb) 
    (lldb) f 17
    frame #17: 0x00000001016cc132 libGGeo.dylib`GTreeCheck::makeMergedMeshAndInstancedBuffers(this=0x0000000104466080, verbosity=0) + 162 at GTreeCheck.cc:444
       441  
       442  
       443      // passes thru to GMergedMesh::create with management of the mm in GGeoLib
    -> 444      GMergedMesh* mm0 = m_geolib->makeMergedMesh(0, base, root, verbosity );
       445  
       446  
       447      std::vector<GNode*> placements = getPlacements(0);  // just m_root
    (lldb) 


