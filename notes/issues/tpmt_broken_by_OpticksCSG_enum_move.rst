tpmt broken by OpticksCSG enum move
======================================

* shape/operator enum unification to use sysrap/OpticksCSG.{h,py} is incomplete
* tpmt broken due to mis-interpretation of part buffer

Vague recollections
---------------------

* GCSG not used for raytrace ? It exists just to allow Geant4 CSG re-creation of geometry ?


2017-10-20 tpmt-- broken (again)
--------------------------------------

1. Following some effort to clean up GGeoTest::makePmtInBox
   still finding failure symptom that the raytrace of the PMT is missing.

2. Also wierd container box distortion, possible bad bbox ?  This turns out to 
   be too tight a far causing clipping. 

3. Found the default apmtidx of 0 loads analytic PMT with old enum codes, 
   created a new one in slot 2.  

   TODO: commit fixed analytic PMT into opticksdata slot 0 once this issue is resolved 

4. Found that the GParts from the loaded PMT was not getting combined with the
   container. After fixing that get OptiX hard crash at launch, forcing 
   reboot. To just do init and avoid crash do::

      tpmt-;tpmt-- -PV   ## --nopropagate --noviz

   Cause of crash was omitting GParts::setPartList for the combi GParts,
   so a PartList was being interpreted as a NodeTree.

5. Now get a raytrace, but with a z-slipped dynode : several relative positions
   seem wrong.



old overview
--------------

* DONE: old PMT serialization needs to be rebuilt with new unified enum   
* rebuilt analytic PMT and stored into opticksdata with non-default apmtidx slot 1 (not committed)


2017-10-20 issue : boundaries not getting into the GParts ?
------------------------------------------------------------

* seems the bndspec is OK, but this is not being treated as
  the input ? Instead the bnd in the .npy which are all zero
  is the input.

  * just need to GParts::close in order to registerBoundaries


::

    simon:opticks blyth$ cat /usr/local/opticks/opticksdata/export/DayaBay/GPmt/0/GPmt_boundaries.txt
    CONTAINING_MATERIAL///Pyrex
    CONTAINING_MATERIAL///Pyrex
    CONTAINING_MATERIAL///Pyrex
    CONTAINING_MATERIAL///Pyrex
    Pyrex///OpaqueVacuum
    Pyrex/SENSOR_SURFACE//Bialkali
    Pyrex/SENSOR_SURFACE//Bialkali
    Pyrex///Vacuum
    Bialkali///Vacuum
    Bialkali///Vacuum
    OpaqueVacuum///Vacuum
    Vacuum///OpaqueVacuum


::

    tpmt-;tpmt-- -PV    # just init for debug

    2017-10-20 18:22:21.843 INFO  [627696] [GGeoTest::createPmtInBox@293] GGeoTest::createPmtInBox  spec Rock/NONE/perfectAbsorbSurface/MineralOil container_inner_material MineralOil
    2017-10-20 18:22:21.845 INFO  [627696] [GPmt::dump@167] GGeoTest::loadPmt (GPmt)pmt --dbganalytic  m_index 0 m_path /usr/local/opticks/opticksdata/export/DayaBay/GPmt/0 m_parts 0x7f98c5ccc180 m_csg 0x7f98c5ccb990 m_bndlib 0x7f98c3e049d0
    2017-10-20 18:22:21.845 INFO  [627696] [GParts::Summary@1120] GGeoTest::loadPmt (GParts)pts --dbganalytic  num_parts 12 num_prim 0
     part  0 : node  0 type  1 boundary [  0] Vacuum///Vacuum  
     part  1 : node  0 type  1 boundary [  0] Vacuum///Vacuum  
     part  2 : node  0 type  1 boundary [  0] Vacuum///Vacuum  
     part  3 : node  0 type  2 boundary [  0] Vacuum///Vacuum  
     part  4 : node  1 type  1 boundary [  0] Vacuum///Vacuum  
     part  5 : node  1 type  1 boundary [  0] Vacuum///Vacuum  
     part  6 : node  1 type  1 boundary [  0] Vacuum///Vacuum  
     part  7 : node  1 type  2 boundary [  0] Vacuum///Vacuum  
     part  8 : node  2 type  1 boundary [  0] Vacuum///Vacuum  
     part  9 : node  2 type  1 boundary [  0] Vacuum///Vacuum  
     part 10 : node  3 type  1 boundary [  0] Vacuum///Vacuum  
     part 11 : node  4 type  2 boundary [  0] Vacuum///Vacuum  
    2017-10-20 18:22:21.845 INFO  [627696] [*GMergedMesh::combine@138] GMergedMesh::combine making new mesh  index 0 solids 1 verbosity 3
    2017-10-20 18:22:21.845 INFO  [627696] [GSolid::Dump@204] GMergedMesh::combine (source solids) numSolid 1


    GPmtTest   # shows same issue ... 





Review NCSG::Deserialize boundary handling
---------------------------------------------

* In tboolean- the boundary strings are
  planted in the python, which get serialized into
  the csg.txt

::

    cat /tmp/blyth/opticks/tboolean-torus--/csg.txt 
    Rock//perfectAbsorbSurface/Vacuum
    Vacuum///GlassSchottF2


* each NCSG tree has only a single boundary spec string
  which gets set in NCSG::Deserialize

::

    1153 int NCSG::Deserialize(const char* basedir, std::vector<NCSG*>& trees, int verbosity )
    1154 {
    ....
    1157     std::string txtpath = BFile::FormPath(basedir, FILENAME) ;
    ....
    1166     NTxt bnd(txtpath.c_str());
    1167     bnd.read();
    1169 
    1170     unsigned nbnd = bnd.getNumLines();
    ....
    1181     // order is reversed so that a tree with the "container" meta data tag at tree slot 0
    1182     // is handled last, so container_bb will then have been adjusted to hold all the others...
    1183     // allowing the auto-bbox setting of the container
    1184 
    1185     for(unsigned j=0 ; j < nbnd ; j++)
    1186     {
    1187         unsigned i = nbnd - 1 - j ;
    1188         std::string treedir = BFile::FormPath(basedir, BStr::itoa(i));
    1189 
    1190         NCSG* tree = new NCSG(treedir.c_str());
    1191         tree->setIndex(i);
    1192         tree->setVerbosity( verbosity );
    1193         tree->setBoundary( bnd.getLine(i) );



::

     165 GParts* GParts::make( NCSG* tree, const char* spec, unsigned verbosity )
     166 {
     167     assert(spec);
     168 
     ...
     238     // GParts originally intended to handle lists of parts each of which 
     239     // must have an associated boundary spec. When holding CSG trees there 
     240     // is really only a need for a single common boundary, but for
     241     // now enable reuse of the old GParts by duplicating the spec 
     242     // for every node of the tree
     243 
     244     const char* reldir = "" ;  // empty reldir avoids defaulting to GItemList  
     245 
     246     GItemList* lspec = GItemList::Repeat("GParts", spec, ni, reldir) ;
     247 
     248     GParts* pts = new GParts(nodebuf, tranbuf, planbuf, lspec) ;
     249 
     250     //pts->setTypeCode(0u, root->type);   //no need, slot 0 is the root node where the type came from
     251     return pts ;
     252 }


* hmm does GParts::close translate the spec into boundary int and write into partBuffer ?
  YEP : void GParts::registerBoundaries() // convert boundary spec names into integer codes using bndlib

::

    200 RT_PROGRAM void intersect(int primIdx)
    201 {
    202     const Prim& prim    = primBuffer[primIdx];
    203 
    204     unsigned partOffset  = prim.partOffset() ;
    205     unsigned numParts    = prim.numParts() ;
    206     unsigned primFlag    = prim.primFlag() ;
    207 
    208     uint4 identity = identityBuffer[instance_index] ;
    209 
    210 
    211     if(primFlag == CSG_FLAGNODETREE)
    212     {
    213         Part pt0 = partBuffer[partOffset + 0] ;
    214 
    215         identity.z = pt0.boundary() ;        // replace placeholder zero with test analytic geometry root node boundary
    216 
    217         evaluative_csg( prim, identity );
    218         //intersect_csg( prim, identity );
    219 
    220     }
    221     else if(primFlag == CSG_FLAGINVISIBLE)
    222     {
    223         // do nothing : report no intersections for primitives marked with primFlag CSG_FLAGINVISIBLE 
    224     }
    225 #ifdef WITH_PARTLIST
    226     else if(primFlag == CSG_FLAGPARTLIST)
    227     {
    228         for(unsigned int p=0 ; p < numParts ; p++)
    229         {
    230             Part pt = partBuffer[partOffset + p] ;
    231 
    232             identity.z = pt.boundary() ;
    233 






revisit tpmt--
----------------

The --apmtidx 1 option results in loading::

    2017-04-10 15:02:46.231 FATAL [50057] [GGeo::loadAnalyticPmt@733] GGeo::loadAnalyticPmt AnalyticPMTIndex 1 AnalyticPMTSlice ALL Path /usr/local/opticks/opticksdata/export/DayaBay/GPmt/1

::

    155 tpmt--(){
    ...
    176 
    177     local apmtidx=1
    178     # non-default AnalyticPMTIndex currently required for updated enums
    ...
    181    op.sh \
    182        --anakey $anakey \
    183        --save \
    184        --test --testconfig "$(tpmt-testconfig)" \
    185        --torch --torchconfig "$(tpmt-torchconfig)" \
    186        --cat $(tpmt-det) \
    187        --tag $tag \
    188        --timemax 10 \
    189        --animtimemax 10 \
    190        --eye 0.0,-0.5,0.0 \
    191        --geocenter \
    192        --apmtidx $apmtidx \
    193        $*
    194 
    195 }


root cause of difficulty
--------------------------

* kludgy association of an old triangulated PMT with the analytic CSG one, 
  actually it looks like there is one extra node in the triangulated ?

* best solution would be to find a way to triangulate the CSG, so there 
  would then be no solid/node matching problem 

* developing CSG to triangulation will take a while, so meanwhile just 
  construct meshes using CSG bboxen ?  See ggeo/test/GPmtTest.cc for start of this


symptom3 : surface attachement failure
------------------------------------------

* see :doc:`geant4_opticks_integration/surlib_with_test_geometry` 

::

    2017-03-16 17:49:08.898 INFO  [980504] [CTraverser::Traverse@128] CTraverser::Traverse DONE
    2017-03-16 17:49:08.898 INFO  [980504] [CTraverser::Summary@104] CDetector::traverse numMaterials 5 numMaterialsWithoutMPT 0
    2017-03-16 17:49:08.898 INFO  [980504] [CDetector::attachSurfaces@240] CDetector::attachSurfaces
    2017-03-16 17:49:08.898 INFO  [980504] [GSurLib::examineSolidBndSurfaces@115] GSurLib::examineSolidBndSurfaces numSolids 7
    Assertion failed: (node == i), function examineSolidBndSurfaces, file /Users/blyth/opticks/ggeo/GSurLib.cc, line 124.
    Process 79145 stopped
    * thread #1: tid = 0xef618, 0x00007fff96f1a866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff96f1a866 libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill + 10:
    -> 0x7fff96f1a866:  jae    0x7fff96f1a870            ; __pthread_kill + 20
       0x7fff96f1a868:  movq   %rax, %rdi
       0x7fff96f1a86b:  jmp    0x7fff96f17175            ; cerror_nocancel
       0x7fff96f1a870:  retq   
    (lldb) bt
    * thread #1: tid = 0xef618, 0x00007fff96f1a866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff96f1a866 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff8e5b735c libsystem_pthread.dylib`pthread_kill + 92
        frame #2: 0x00007fff95307b1a libsystem_c.dylib`abort + 125
        frame #3: 0x00007fff952d19bf libsystem_c.dylib`__assert_rtn + 321
        frame #4: 0x0000000101ce0ac9 libGGeo.dylib`GSurLib::examineSolidBndSurfaces(this=0x000000010e21e4a0) + 521 at GSurLib.cc:124
        frame #5: 0x0000000101ce08ad libGGeo.dylib`GSurLib::close(this=0x000000010e21e4a0) + 29 at GSurLib.cc:93
        frame #6: 0x0000000103ee0497 libcfg4.dylib`CDetector::attachSurfaces(this=0x000000010e21e1c0) + 247 at CDetector.cc:244
        frame #7: 0x0000000103e5ad26 libcfg4.dylib`CGeometry::init(this=0x000000010e21dc30) + 1446 at CGeometry.cc:73
        frame #8: 0x0000000103e5a770 libcfg4.dylib`CGeometry::CGeometry(this=0x000000010e21dc30, hub=0x000000010980c7a0) + 112 at CGeometry.cc:39
        frame #9: 0x0000000103e5ad8d libcfg4.dylib`CGeometry::CGeometry(this=0x000000010e21dc30, hub=0x000000010980c7a0) + 29 at CGeometry.cc:40
        frame #10: 0x0000000103f01286 libcfg4.dylib`CG4::CG4(this=0x000000010cadeab0, hub=0x000000010980c7a0) + 214 at CG4.cc:122
        frame #11: 0x0000000103f017bd libcfg4.dylib`CG4::CG4(this=0x000000010cadeab0, hub=0x000000010980c7a0) + 29 at CG4.cc:144
        frame #12: 0x0000000103ff1da3 libokg4.dylib`OKG4Mgr::OKG4Mgr(this=0x00007fff5fbfe6b0, argc=23, argv=0x00007fff5fbfe790) + 547 at OKG4Mgr.cc:35
        frame #13: 0x0000000103ff1ff3 libokg4.dylib`OKG4Mgr::OKG4Mgr(this=0x00007fff5fbfe6b0, argc=23, argv=0x00007fff5fbfe790) + 35 at OKG4Mgr.cc:41
        frame #14: 0x00000001000139be OKG4Test`main(argc=23, argv=0x00007fff5fbfe790) + 1486 at OKG4Test.cc:56
        frame #15: 0x00007fff9238d5fd libdyld.dylib`start + 1
    (lldb) 

::

    (lldb) f 7
    frame #7: 0x0000000103e5ad26 libcfg4.dylib`CGeometry::init(this=0x000000010e21dc30) + 1446 at CGeometry.cc:73
       70           detector  = static_cast<CDetector*>(new CGDMLDetector(m_hub, query)) ; 
       71       }
       72   
    -> 73       detector->attachSurfaces();
       74       //m_csurlib->convert(detector);
       75   
       76       m_detector = detector ; 
    (lldb) 




symptom 2 : CPU/G4 cfg4/CTestDetector misunderstanding primordial CSG buffer ?
-----------------------------------------------------------------------------------

* actually the PmtInBox code appears to be unaware of GCSG 

::

    tpmt-- --okg4

    2017-03-16 13:51:10.046 INFO  [889146] [OpticksGen::targetGenstep@125] OpticksGen::targetGenstep setting frame 1 1.0000,0.0000,0.0000,0.0000 0.0000,1.0000,0.0000,0.0000 0.0000,0.0000,1.0000,0.0000 0.0000,0.0000,0.0000,1.0000
    2017-03-16 13:51:10.047 FATAL [889146] [GenstepNPY::setPolarization@212] GenstepNPY::setPolarization pol 0.0000,0.0000,0.0000,0.0000 npol nan,nan,nan,nan m_polw nan,nan,nan,380.0000
    2017-03-16 13:51:10.047 INFO  [889146] [SLog::operator@15] OpticksHub::OpticksHub DONE

    *************************************************************
     Geant4 version Name: geant4-10-02-patch-01    (26-February-2016)
                          Copyright : Geant4 Collaboration
                          Reference : NIM A 506 (2003), 250-303
                                WWW : http://cern.ch/geant4
    *************************************************************

    2017-03-16 13:51:10.122 FATAL [889146] [CGeometry::init@59] CGeometry::init G4 simple test geometry 
    2017-03-16 13:51:10.122 INFO  [889146] [GGeo::createSurLib@656] deferred creation of GSurLib 
    2017-03-16 13:51:10.122 INFO  [889146] [GSurLib::collectSur@79]  nsur 48
    2017-03-16 13:51:10.122 INFO  [889146] [CPropLib::init@68] CPropLib::init
    2017-03-16 13:51:10.122 INFO  [889146] [CPropLib::initCheckConstants@120] CPropLib::initCheckConstants mm 1 MeV 1 nanosecond 1 ns 1 nm 1e-06 GC::nanometer 1e-06 h_Planck 4.13567e-12 GC::h_Planck 4.13567e-12 c_light 299.792 GC::c_light 299.792 dscale 0.00123984
    2017-03-16 13:51:10.122 INFO  [889146] [*CTestDetector::makeDetector@118] CTestDetector::makeDetector PmtInBox 1 BoxInBox 0 numSolids (from mesh0) 7 numSolids (from config) 1
    Assertion failed: (numSolids == numSolidsConfig), function makeDetector, file /Users/blyth/opticks/cfg4/CTestDetector.cc, line 127.
    /Users/blyth/opticks/bin/op.sh: line 580: 41465 Abort trap: 6           /usr/local/opticks/lib/OKG4Test --anakey tpmt --save --test --testconfig mode=PmtInBox_pmtpath=/usr/local/opticks/opticksdata/export/dpib/GMergedMesh/0_control=1,0,0,0_analytic=1_apmtidx=1_node=box_parameters=0,0,0,300_boundary=Rock/NONE/perfectAbsorbSurface/MineralOil --torch --torchconfig type=disc_photons=500000_wavelength=380_frame=1_source=0,0,300_target=0,0,0_radius=100_zenithazimuth=0,1,0,1_material=Vacuum_mode=_polarization= --cat PmtInBox --tag 10 --timemax 10 --animtimemax 10 --eye 0.0,-0.5,0.0 --geocenter --okg4
    /Users/blyth/opticks/bin/op.sh RC 134
    simon:opticks blyth$ 


    2017-03-16 14:17:21.209 INFO  [901864] [CPropLib::initCheckConstants@120] CPropLib::initCheckConstants mm 1 MeV 1 nanosecond 1 ns 1 nm 1e-06 GC::nanometer 1e-06 h_Planck 4.13567e-12 GC::h_Planck 4.13567e-12 c_light 299.792 GC::c_light 299.792 dscale 0.00123984
    2017-03-16 14:17:21.209 INFO  [901864] [*CTestDetector::makeDetector@118] CTestDetector::makeDetector PmtInBox 1 BoxInBox 0 numSolidsMesh 7 numSolidsConfig 1
    2017-03-16 14:17:21.209 INFO  [901864] [GMergedMesh::dumpSolids@617] CTestDetector::makeDetector (solid count inconsistent)
        0 ce             gfloat4      0.000      0.000      0.000    300.000  bb bb min   -300.000   -300.000   -300.000  max    300.000    300.000    300.000  ni(         0,         0,         0,4294967295) id(         0,         5,         0,         0)
        1 ce             gfloat4      0.000      0.000    -18.997    149.997  bb bb min   -100.288   -100.288   -168.995  max    100.288    100.288    131.000  ni(       720,       362,         1,         0) id(         1,         4,         1,         0)
        2 ce             gfloat4      0.000      0.000    -18.247    146.247  bb bb min    -97.288    -97.288   -164.495  max     97.288     97.288    128.000  ni(       720,       362,         2,         1) id(         2,         3,         2,         0)
        3 ce             gfloat4      0.005      0.004     91.998     98.143  bb bb min    -98.138    -98.139     55.996  max     98.148     98.147    128.000  ni(       960,       482,         3,         2) id(         3,         0,         3,         0)
        4 ce             gfloat4      0.000      0.000     13.066     98.143  bb bb min    -98.143    -98.143    -30.000  max     98.143     98.143     56.131  ni(       576,       288,         4,         2) id(         4,         1,         4,         0)
        5 ce             gfloat4      0.000      0.000    -81.500     83.000  bb bb min    -27.500    -27.500   -164.500  max     27.500     27.500      1.500  ni(        96,        50,         5,         2) id(         5,         2,         4,         0)
        6 ce             gfloat4      0.000      0.000      0.000    300.000  bb bb min   -300.000   -300.000   -300.000  max    300.000    300.000    300.000  ni(        12,        24,         0,4294967295) id(         0,      1000,         0,         0)
    Assertion failed: (numSolidsMesh == numSolidsConfig), function makeDetector, file /Users/blyth/opticks/cfg4/CTestDetector.cc, line 133.


looks like okg4 not updated since primordial GCSG 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Approach 

* make connection between the analytic GCSG volumes that CTestDetector::makePMT 
  is going to use and the triangulated GMergedMesh solid count, 
  then can update the assert

* avoid duplicity regards the analytic PMT and honour the apmtidx version, by 
  eliminating CPropLib::getPmtCSG

::

    simon:opticks blyth$ opticks-find getPmtCSG
    ./cfg4/CPropLib.cc:GCSG* CPropLib::getPmtCSG(NSlice* slice)
    ./cfg4/CPropLib.cc:        LOG(error) << "CPropLib::getPmtCSG failed to load PMT" ;
    ./cfg4/CPropLib.cc:        LOG(error) << "CPropLib::getPmtCSG failed to getCSG from GPmt" ;
    ./cfg4/CTestDetector.cc:    GCSG* csg = m_mlib->getPmtCSG(slice);
    ./cfg4/CPropLib.hh:       GCSG*       getPmtCSG(NSlice* slice);


    162 GCSG* CPropLib::getPmtCSG(NSlice* slice)
    163 {
    164    // hmm this is probably already loaded ???
    165    
    166     GPmt* pmt = GPmt::load( m_ok, m_bndlib, 0, slice );    // pmtIndex:0
    167     
    168     if(pmt == NULL)
    169     {
    170         LOG(error) << "CPropLib::getPmtCSG failed to load PMT" ;
    171         return NULL ; 
    172     }   
    173     
    174     GCSG* csg = pmt->getCSG();
    175     
    176     if(csg == NULL)
    177     {
    178         LOG(error) << "CPropLib::getPmtCSG failed to getCSG from GPmt" ;
    179         return NULL ; 
    180     }   
    181     return csg ;
    182 }   





FIXED : symptom 1, GPU side mis-interpreting parts buffer after enum change
-----------------------------------------------------------------------------

::

    tpmt--   

    2017-03-15 20:48:44.712 INFO  [829428] [OContext::close@219] OContext::close numEntryPoint 2
    ##hemi-pmt.cu:bounds primIdx 0 is_partlist:0 min  -101.1682  -101.1682   -23.8382 max   101.1682   101.1682    56.0000 
    ##hemi-pmt.cu:bounds primIdx 1 is_partlist:0 min   -98.1428   -98.1428    56.0000 max    98.1428    98.1428    98.0465 
    ##hemi-pmt.cu:bounds primIdx 2 is_partlist:0 min   -98.0932   -98.0932    55.9934 max    98.0932    98.0932    98.0128 
    ##hemi-pmt.cu:bounds primIdx 3 is_partlist:0 min   -27.5000   -27.5000  -164.5000 max    27.5000    27.5000     1.5000 
    ##hemi-pmt.cu:bounds primIdx 4 is_partlist:0 min  -300.0100  -300.0100  -300.0100 max   300.0100   300.0100   300.0100 
    2017-03-15 20:48:45.342 INFO  [829428] [OPropagator::prelaunch@149] 1 : (0;500000,1) prelaunch_times vali,comp,prel,lnch  0.0000 0.2694 0.2364 0.0000
    evaluative_csg primIdx_ 1 numParts 4 perfect tree fullHeight 4294967295 exceeds current limit
    evaluative_csg primIdx_ 1 numParts 4 perfect tree fullHeight 4294967295 exceeds current limit
    evaluative_csg primIdx_ 1 numParts 4 perfect tree fullHeight 4294967295 exceeds current limit
    evaluative_csg primIdx_ 1 numParts 4 perfect tree fullHeight 4294967295 exceeds current limit


review of analytic PMT serialization
--------------------------------------

* ana/pmt/analytic.py 

Recreate the analytic PMT from detdecs parse with

::

   pmt-analytic-tmp   # writing to $TMP/GPmt/0/GPmt.npy
   pmt-analytic       # writing to $IDPATH/GPmt/0/GPmt.npy

Actual one in use is from opticksdata repo $OPTICKS_DATA/export/DayaBay/GPmt/0/  


Comparing existing serializations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All three look effectively the same, with no influence from new enum so far::

    simon:pmt blyth$ l /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GPmt/0/
    total 48
    -rw-r--r--  1 blyth  staff   848 Mar 15 16:27 GPmt.npy
    -rw-r--r--  1 blyth  staff   289 Mar 15 16:27 GPmt_boundaries.txt
    -rw-r--r--  1 blyth  staff  1168 Mar 15 16:27 GPmt_csg.npy
    -rw-r--r--  1 blyth  staff    74 Mar 15 16:27 GPmt_lvnames.txt
    -rw-r--r--  1 blyth  staff    47 Mar 15 16:27 GPmt_materials.txt
    -rw-r--r--  1 blyth  staff    74 Mar 15 16:27 GPmt_pvnames.txt
    simon:pmt blyth$ 
    simon:pmt blyth$ 
    simon:pmt blyth$ l $TMP/GPmt/0/
    total 48
    -rw-r--r--  1 blyth  wheel   848 Mar 15 17:31 GPmt.npy
    -rw-r--r--  1 blyth  wheel   289 Mar 15 17:31 GPmt_boundaries.txt
    -rw-r--r--  1 blyth  wheel  1168 Mar 15 17:31 GPmt_csg.npy
    -rw-r--r--  1 blyth  wheel    74 Mar 15 17:31 GPmt_lvnames.txt
    -rw-r--r--  1 blyth  wheel    47 Mar 15 17:31 GPmt_materials.txt
    -rw-r--r--  1 blyth  wheel    74 Mar 15 17:31 GPmt_pvnames.txt
    simon:pmt blyth$ diff -r --brief $IDPATH/GPmt/0 $TMP/GPmt/0
    simon:pmt blyth$ 
    simon:pmt blyth$ 
    simon:pmt blyth$ l /usr/local/opticks/opticksdata/export/DayaBay/GPmt/0/
    total 80
    -rw-r--r--  1 blyth  staff   848 Jul  5  2016 GPmt.npy
    -rw-r--r--  1 blyth  staff   289 Jul  5  2016 GPmt.txt
    -rw-r--r--  1 blyth  staff   289 Jul  5  2016 GPmt_boundaries.txt
    -rw-r--r--  1 blyth  staff   848 Jul  5  2016 GPmt_check.npy
    -rw-r--r--  1 blyth  staff   289 Jul  5  2016 GPmt_check.txt
    -rw-r--r--  1 blyth  staff  1168 Jul  5  2016 GPmt_csg.npy
    -rw-r--r--  1 blyth  staff    47 Jul  5  2016 GPmt_csg.txt
    -rw-r--r--  1 blyth  staff    74 Jul  5  2016 GPmt_lvnames.txt
    -rw-r--r--  1 blyth  staff    47 Jul  5  2016 GPmt_materials.txt
    -rw-r--r--  1 blyth  staff    74 Jul  5  2016 GPmt_pvnames.txt

    simon:pmt blyth$ echo $OPTICKS_DATA
    /usr/local/opticks/opticksdata
    simon:pmt blyth$ 
    simon:pmt blyth$ diff -r --brief $OPTICKS_DATA/export/DayaBay/GPmt/0/ $TMP/GPmt/0/
    Only in /usr/local/opticks/opticksdata/export/DayaBay/GPmt/0/: GPmt.txt
    Only in /usr/local/opticks/opticksdata/export/DayaBay/GPmt/0/: GPmt_check.npy
    Only in /usr/local/opticks/opticksdata/export/DayaBay/GPmt/0/: GPmt_check.txt
    Only in /usr/local/opticks/opticksdata/export/DayaBay/GPmt/0/: GPmt_csg.txt



