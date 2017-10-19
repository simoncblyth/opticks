2017-10-19-almost-all-integration-tests-failing
==================================================

tests-t fails
----------------

::

    simon:tests blyth$ tests-
    simon:tests blyth$ tests-t
    ==                     treflect-t ==  ->    1 
    WARNING : FAILURE OF treflect.bash : RC 1
    ==                         tbox-t ==  ->    1 
    WARNING : FAILURE OF tbox.bash : RC 1
    ==                       tprism-t ==  ->    1 
    WARNING : FAILURE OF tprism.bash : RC 1
    ==                      tnewton-t ==  ->    1 
    WARNING : FAILURE OF tnewton.bash : RC 1
    ==                       twhite-t ==  ->  134 
    WARNING : FAILURE OF twhite.bash : RC 134
    ==                         tpmt-t ==  ->  139 
    WARNING : FAILURE OF tpmt.bash : RC 139
    ==                     trainbow-t ==  ->    1 
    WARNING : FAILURE OF trainbow.bash : RC 1
    ==                        tlens-t ==  ->  134 
    WARNING : FAILURE OF tlens.bash : RC 134
    ==                       tg4gun-t ==  ->    0 
    simon:tests blyth$ 


Overview
----------

All the tests are using the old geometry on the commandline approach.
Which has fallen into disuse following introduction of 
the python serialized CSG geometry approach. 

Have vague recollection that GGeoTestConfig was 
changed substantially whilst working in tboolean- 
prior to introduction of python geometry.

So revive some old style tboolean funcs before tackling 
integration tests, which are a few generations behind.

* :doc:`revive-old-style-tboolean` 


tlens-t
------------

::

    simon:tests blyth$ tlens-- --debugger

    2017-10-19 17:16:49.832 INFO  [369395] [GGeoLib::loadConstituents@122] GGeoLib::loadConstituents mm.reldir GMergedMesh gp.reldir GParts MAX_MERGED_MESH  10
    2017-10-19 17:16:49.832 INFO  [369395] [GGeoLib::loadConstituents@129] /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae
    2017-10-19 17:16:49.960 INFO  [369395] [GGeoLib::loadConstituents@178] GGeoLib::loadConstituents loaded 6 ridx (  0,  1,  2,  3,  4,  5,)
    2017-10-19 17:16:50.036 INFO  [369395] [GMeshLib::loadMeshes@214] idpath /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae
    2017-10-19 17:16:50.069 INFO  [369395] [GGeo::loadFromCache@672] GGeo::loadFromCache DONE
    2017-10-19 17:16:50.069 INFO  [369395] [GGeo::loadAnalyticPmt@788] GGeo::loadAnalyticPmt AnalyticPMTIndex 0 AnalyticPMTSlice ALL Path /usr/local/opticks/opticksdata/export/DayaBay/GPmt/0
    2017-10-19 17:16:50.078 INFO  [369395] [GGeo::loadGeometry@594] GGeo::loadGeometry DONE
    2017-10-19 17:16:50.078 INFO  [369395] [OpticksGeometry::loadGeometryBase@283] OpticksGeometry::loadGeometryBase DONE 
    2017-10-19 17:16:50.078 WARN  [369395] [GGeoTestConfig::getArg@171] GGeoTestConfig::getArg UNRECOGNIZED arg shape
    2017-10-19 17:16:50.079 WARN  [369395] [GGeoTestConfig::set@194] GGeoTestConfig::set WARNING ignoring unrecognized parameter box
    2017-10-19 17:16:50.079 WARN  [369395] [GGeoTestConfig::getArg@171] GGeoTestConfig::getArg UNRECOGNIZED arg shape
    2017-10-19 17:16:50.079 WARN  [369395] [GGeoTestConfig::set@194] GGeoTestConfig::set WARNING ignoring unrecognized parameter lens
    2017-10-19 17:16:50.079 WARN  [369395] [GGeoTest::init@57] GGeoTest::init booting from m_ggeo 
    2017-10-19 17:16:50.079 WARN  [369395] [GMaker::init@178] GMaker::init booting from cache
    2017-10-19 17:16:50.081 INFO  [369395] [GMaterialLib::postLoadFromCache@67] GMaterialLib::postLoadFromCache  nore 0 noab 0 nosc 0 xxre 0 xxab 0 xxsc 0 fxre 0 fxab 0 fxsc 0 groupvel 1
    2017-10-19 17:16:50.081 INFO  [369395] [GMaterialLib::replaceGROUPVEL@552] GMaterialLib::replaceGROUPVEL  ni 38
    2017-10-19 17:16:50.082 INFO  [369395] [GPropertyLib::getIndex@338] GPropertyLib::getIndex type GMaterialLib TRIGGERED A CLOSE  shortname [GdDopedLS]
    2017-10-19 17:16:50.082 INFO  [369395] [GPropertyLib::close@384] GPropertyLib::close type GMaterialLib buf 38,2,39,4
    2017-10-19 17:16:50.085 INFO  [369395] [GGeoLib::loadConstituents@122] GGeoLib::loadConstituents mm.reldir GMergedMesh gp.reldir GParts MAX_MERGED_MESH  10
    2017-10-19 17:16:50.085 INFO  [369395] [GGeoLib::loadConstituents@129] /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae
    2017-10-19 17:16:50.216 INFO  [369395] [GGeoLib::loadConstituents@178] GGeoLib::loadConstituents loaded 6 ridx (  0,  1,  2,  3,  4,  5,)
    2017-10-19 17:16:50.216 FATAL [369395] [GGeoTestConfig::getNumElements@210] GGeoTestConfig::getNumElements ELEMENT MISMATCH IN TEST GEOMETRY CONFIGURATION  nbnd (boundaries) 2 nnod (nodes) 0 npar (parameters) 2 ntra (transforms) 0
    Assertion failed: (equal && "need equal number of boundaries, parameters, transforms and nodes"), function getNumElements, file /Users/blyth/opticks/ggeo/GGeoTestConfig.cc, line 218.
    Process 92690 stopped
    * thread #1: tid = 0x5a2f3, 0x00007fff8b576866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff8b576866 libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill + 10:
    -> 0x7fff8b576866:  jae    0x7fff8b576870            ; __pthread_kill + 20
       0x7fff8b576868:  movq   %rax, %rdi
       0x7fff8b57686b:  jmp    0x7fff8b573175            ; cerror_nocancel
       0x7fff8b576870:  retq   
    (lldb) 
    (lldb) bt
    * thread #1: tid = 0x5a2f3, 0x00007fff8b576866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff8b576866 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff82c1335c libsystem_pthread.dylib`pthread_kill + 92
        frame #2: 0x00007fff89963b1a libsystem_c.dylib`abort + 125
        frame #3: 0x00007fff8992d9bf libsystem_c.dylib`__assert_rtn + 321
        frame #4: 0x000000010211b6d2 libGGeo.dylib`GGeoTestConfig::getNumElements(this=0x000000010841aeb0) + 722 at GGeoTestConfig.cc:218
        frame #5: 0x0000000102114ee0 libGGeo.dylib`GGeoTest::create(this=0x000000010841c0c0) + 496 at GGeoTest.cc:128
        frame #6: 0x0000000102114c0d libGGeo.dylib`GGeoTest::modifyGeometry(this=0x000000010841c0c0) + 157 at GGeoTest.cc:85
        frame #7: 0x000000010214073c libGGeo.dylib`GGeo::modifyGeometry(this=0x0000000105c38b40, config=0x000000010841ae00) + 668 at GGeo.cc:818
        frame #8: 0x00000001022a4844 libOpticksGeometry.dylib`OpticksGeometry::modifyGeometry(this=0x0000000105c36ae0) + 868 at OpticksGeometry.cc:294
        frame #9: 0x00000001022a3aec libOpticksGeometry.dylib`OpticksGeometry::loadGeometry(this=0x0000000105c36ae0) + 572 at OpticksGeometry.cc:224
        frame #10: 0x00000001022a81b9 libOpticksGeometry.dylib`OpticksHub::loadGeometry(this=0x0000000105c2fe20) + 409 at OpticksHub.cc:282
        frame #11: 0x00000001022a720d libOpticksGeometry.dylib`OpticksHub::init(this=0x0000000105c2fe20) + 77 at OpticksHub.cc:102
        frame #12: 0x00000001022a7110 libOpticksGeometry.dylib`OpticksHub::OpticksHub(this=0x0000000105c2fe20, ok=0x0000000105c21cf0) + 432 at OpticksHub.cc:88
        frame #13: 0x00000001022a72fd libOpticksGeometry.dylib`OpticksHub::OpticksHub(this=0x0000000105c2fe20, ok=0x0000000105c21cf0) + 29 at OpticksHub.cc:90
        frame #14: 0x0000000103c471e6 libOK.dylib`OKMgr::OKMgr(this=0x00007fff5fbfe558, argc=23, argv=0x00007fff5fbfe638, argforced=0x0000000000000000) + 262 at OKMgr.cc:46
        frame #15: 0x0000000103c4764b libOK.dylib`OKMgr::OKMgr(this=0x00007fff5fbfe558, argc=23, argv=0x00007fff5fbfe638, argforced=0x0000000000000000) + 43 at OKMgr.cc:49
        frame #16: 0x000000010000adad OKTest`main(argc=23, argv=0x00007fff5fbfe638) + 1373 at OKTest.cc:58
        frame #17: 0x00007fff869e95fd libdyld.dylib`start + 1
        frame #18: 0x00007fff869e95fd libdyld.dylib`start + 1
    (lldb) 






test geometry review
------------------------

::

    805 void GGeo::modifyGeometry(const char* config)
    806 {
    807     // NB only invoked with test option : "op --test" 
    808     //   controlled from OpticksGeometry::loadGeometry 
    809 
    810     GGeoTestConfig* gtc = new GGeoTestConfig(config);
    811 
    812     LOG(trace) << "GGeo::modifyGeometry"
    813               << " config [" << ( config ? config : "" ) << "]" ;
    814 
    815     assert(m_geotest == NULL);
    816 
    817     m_geotest = new GGeoTest(m_ok, gtc, this);
    818     m_geotest->modifyGeometry();
    819 
    820 }

    078 void GGeoTest::modifyGeometry()
     79 {
     80     const char* csgpath = m_config->getCsgPath();
     81     bool analytic = m_config->getAnalytic();
     82 
     83     if(csgpath) assert(analytic == true);
     84 
     85     GMergedMesh* tmm_ = create();
     86 
     87     GMergedMesh* tmm = m_lod > 0 ? GMergedMesh::MakeLODComposite(tmm_, m_lodconfig->levels ) : tmm_ ;
     88 
     89 
     90     char geocode =  analytic ? OpticksConst::GEOCODE_ANALYTIC : OpticksConst::GEOCODE_TRIANGULATED ;  // message to OGeo
     91     tmm->setGeoCode( geocode );
     92 
     93     if(tmm->isTriangulated())
     94     {
     95         tmm->setITransformsBuffer(NULL); // avoiding FaceRepeated complications 
     96     }
     97 
     98     //tmm->dump("GGeoTest::modifyGeometry tmm ");
     99     m_geolib->clear();
    100     m_geolib->setMergedMesh( 0, tmm );
    101 }


    104 GMergedMesh* GGeoTest::create()
    105 {
    106     //TODO: unify all these modes into CSG 
    107     //      whilst still supporting the old partlist approach 
    108 
    109     const char* csgpath = m_config->getCsgPath();
    110     const char* mode = m_config->getMode();
    111 
    112     GMergedMesh* tmm = NULL ;
    113 
    114     if( mode != NULL && strcmp(mode, "PmtInBox") == 0)
    115     {
    116         tmm = createPmtInBox();
    117     }
    118     else
    119     {
    120         std::vector<GSolid*> solids ;
    121         if(csgpath != NULL)
    122         {
    123             assert( strlen(csgpath) > 3 && "unreasonable csgpath strlen");
    124             loadCSG(csgpath, solids);
    125         }
    126         else
    127         {
    128             unsigned int nelem = m_config->getNumElements();
    129             assert(nelem > 0);
    130             if(     strcmp(mode, "BoxInBox") == 0) createBoxInBox(solids);
    131             else  LOG(warning) << "GGeoTest::create mode not recognized " << mode ;
    132         }
    133         tmm = combineSolids(solids);
    134     }
    135     assert(tmm);
    136     return tmm ;
    137 }





