ncsg_support_in_cfg4 : Push CSG node tree support thru to cfg4
=================================================================

Objective
----------

Direct CSG intersect comparisons between Opticks and G4 using Opticks 
defined test geometries, eg invoked by the tboolean- tests 
using --okg4 to switch on bi-simulation.

For example, get the below to work and pass validation comparisons::

    tboolean-;

    tboolean-torus --okg4 -D --dbgsurf
       ## bi-simulation

    tboolean-torus --okg4 --load --vizg4
       ## visualize the G4 evt 

    tboolean-torus-a
       ## OpticksEventCompareTest OR other such exe

    tboolean-torus-a --vizg4 
       ## load the G4 event, for dumping etc..



STATUS
---------

* bi-simulation and analysis runs
* torch source translation into G4 fails to reproduce expected source photons 



Torch Config Convert
------------------------
::

   cfg4/CGenerator.cc
   cfg4/CTorchSource.cc




::

    115 CG4::CG4(OpticksHub* hub)
    116    :
    117      m_hub(hub),
    118      m_ok(m_hub->getOpticks()),
    119      m_run(m_ok->getRun()),
    120      m_cfg(m_ok->getCfg()),
    121      m_physics(new CPhysics(this)),
    122      m_runManager(m_physics->getRunManager()),
    123      m_geometry(new CGeometry(m_hub)),
    124      m_hookup(m_geometry->hookup(this)),
    125      m_lib(m_geometry->getPropLib()),
    126      m_detector(m_geometry->getDetector()),
    127      m_generator(new CGenerator(m_hub, this)),
    128      m_collector(NULL),   // deferred instanciation until CG4::postinitialize after G4 materials have overridden lookupA
    129      m_recorder(new CRecorder(m_ok, m_geometry, m_generator->isDynamic())),
    130      //m_rec(new Rec(m_ok, m_geometry, m_generator->isDynamic())), 
    131      m_steprec(new CStepRec(m_ok, m_generator->isDynamic())),
    132      m_visManager(NULL),
    133      m_uiManager(NULL),
    134      m_ui(NULL),
    135      m_pga(new CPrimaryGeneratorAction(m_generator->getSource())),
    136      m_sa(new CSteppingAction(this, m_generator->isDynamic())),
    137      m_ta(new CTrackingAction(this)),
    138      m_ra(new CRunAction(m_hub)),
    139      m_ea(new CEventAction(this)),
    140      m_initialized(false)
    141 {
    142      OK_PROFILE("CG4::CG4");
    143      init();
    144 }




Overview
----------

* creation of Geant4 geometries from the NCSG/GParts node tree description
* comparisons of GPU and CPU propagations using CSG node tree geometries

* tpmt-t tconcentric-t were primary users of cfg4 comparison funcs
  using GCSG translation : but GCSG translation to G4 geometry was
  very limited ... OpticksCSG supports many more primitives  



Approach
-------------------------------------------------------

* review GCSG usage in cfg4 
* decide what level to operate (NCSG/GParts/..) ? 
* start with test geometry scope only, not full structure
* implement the conversion
* new versions of tpmt-t tconcentric-t 



OKG4Mgr vs OKMgr : principal difference is instanciation of m_g4 (CG4) with m_hub argument in OKG4Mgr
---------------------------------------------------------------------------------------------------------

* okg4 option uses OKG4Test executable (OKG4Mgr) rather than default OKTest (OKMgr) executable

::

     34 OKMgr::OKMgr(int argc, char** argv, const char* argforced )
     35     :
     36     m_log(new SLog("OKMgr::OKMgr")),
     37     m_ok(new Opticks(argc, argv, argforced)),
     38     m_hub(new OpticksHub(m_ok)),            // immediate configure and loadGeometry 
     39     m_idx(new OpticksIdx(m_hub)),
     40     m_num_event(m_ok->getMultiEvent()),     // after hub instanciation, as that configures Opticks
     41     m_gen(m_hub->getGen()),
     42     m_run(m_hub->getRun()),
     43     m_viz(m_ok->isCompute() ? NULL : new OpticksViz(m_hub, m_idx, true)),
     44     m_propagator(new OKPropagator(m_hub, m_idx, m_viz)),
     45     m_count(0)
     46 {
     47     init();
     48     (*m_log)("DONE");
     49 }


     26 OKG4Mgr::OKG4Mgr(int argc, char** argv)
     27     :
     28     m_log(new SLog("OKG4Mgr::OKG4Mgr")),
     29     m_ok(new Opticks(argc, argv)),
     30     m_run(m_ok->getRun()),
     31     m_hub(new OpticksHub(m_ok)),                       // configure, loadGeometry and setupInputGensteps immediately
     32     m_load(m_ok->isLoad()),
     33     m_idx(new OpticksIdx(m_hub)),
     34     m_num_event(m_ok->getMultiEvent()),                    // after hub instanciation, as that configures Opticks
     35     m_gen(m_hub->getGen()),

     36     m_g4(m_load ? NULL : new CG4(m_hub)),                        // configure and initialize immediately 
     ..     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

     37     m_viz(m_ok->isCompute() ? NULL : new OpticksViz(m_hub, m_idx, true)),    // true: load/create Bookmarks, setup shaders, upload geometry immediately 
     38     m_propagator(new OKPropagator(m_hub, m_idx, m_viz))
     39 {
     40     (*m_log)("DONE");
     41 }
     42 
     43 OKG4Mgr::~OKG4Mgr()
     44 {
     45     cleanup();
     46 }


CG4 : instanciates CGeometry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    115 CG4::CG4(OpticksHub* hub)
    116    :
    117      m_hub(hub),
    118      m_ok(m_hub->getOpticks()),
    119      m_run(m_ok->getRun()),
    120      m_cfg(m_ok->getCfg()),
    121      m_physics(new CPhysics(this)),
    122      m_runManager(m_physics->getRunManager()),
    123      m_geometry(new CGeometry(m_hub)),
    124      m_hookup(m_geometry->hookup(this)),
    125      m_lib(m_geometry->getPropLib()),
    126      m_detector(m_geometry->getDetector()),
    127      m_generator(new CGenerator(m_hub, this)),
    128      m_collector(NULL),   // deferred instanciation until CG4::postinitialize after G4 materials have overridden lookupA
    129      m_recorder(new CRecorder(m_ok, m_geometry, m_generator->isDynamic())),
    130      //m_rec(new Rec(m_ok, m_geometry, m_generator->isDynamic())), 
    131      m_steprec(new CStepRec(m_ok, m_generator->isDynamic())),
    132      m_visManager(NULL),
    133      m_uiManager(NULL),
    134      m_ui(NULL),
    135      m_pga(new CPrimaryGeneratorAction(m_generator->getSource())),
    136      m_sa(new CSteppingAction(this, m_generator->isDynamic())),
    137      m_ta(new CTrackingAction(this)),
    138      m_ra(new CRunAction(m_hub)),
    139      m_ea(new CEventAction(this)),
    140      m_initialized(false)
    141 {
    142      OK_PROFILE("CG4::CG4");
    143      init();
    144 }


CGeometry : instanciates CDetector  (either CTestDetector or GGDMLDetector)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

     39 CGeometry::CGeometry(OpticksHub* hub)
     40    :
     41    m_hub(hub),
     42    m_ok(m_hub->getOpticks()),
     43    m_cfg(m_ok->getCfg()),
     44    m_detector(NULL),
     45    m_lib(NULL),
     46    m_material_table(NULL),
     47    m_material_bridge(NULL),
     48    m_surface_bridge(NULL)
     49 {  
     50    init();
     51 }  
     52 
     53 void CGeometry::init()
     54 {
     55     CDetector* detector = NULL ;
     56     if(m_ok->hasOpt("test"))
     57     {
     58         LOG(fatal) << "CGeometry::init G4 simple test geometry " ;
     59         std::string testconfig = m_cfg->getTestConfig();
     60         GGeoTestConfig* ggtc = new GGeoTestConfig( testconfig.empty() ? NULL : testconfig.c_str() );
     61         OpticksQuery* query = NULL ;  // normally no OPTICKS_QUERY geometry subselection with test geometries
     62         detector  = static_cast<CDetector*>(new CTestDetector(m_hub, ggtc, query)) ; 
     63     }   
     64     else
     65     {
     66         // no options here: will load the .gdml sidecar of the geocache .dae 
     67         LOG(fatal) << "CGeometry::init G4 GDML geometry " ; 
     68         OpticksQuery* query = m_ok->getQuery();
     69         detector  = static_cast<CDetector*>(new CGDMLDetector(m_hub, query)) ;
     70     }   
     71     
     72     detector->attachSurfaces();
     73     //m_csurlib->convert(detector);
     74     
     75     m_detector = detector ;
     76     m_lib = detector->getPropLib();
     77 }   



CTestDetector
~~~~~~~~~~~~~~~

* note that this is starting from scratch with the GGeoTestConfig, 
  whereas now that GGeoTest lives in OpticksHub it can now use the existing GGeoTest instance 


::

     60 CTestDetector::CTestDetector(OpticksHub* hub, GGeoTestConfig* config, OpticksQuery* query)
     61   :
     62   CDetector(hub, query),
     63   m_config(config),
     64   m_maker(NULL)
     65 {
     66     init();
     67 }
     68 
     69 
     70 
     71 void CTestDetector::init()
     72 {
     73     LOG(trace) << "CTestDetector::init" ;
     74 
     75     if(m_ok->hasOpt("dbgtestgeo"))
     76     {
     77         LOG(info) << "CTestDetector::init --dbgtestgeo upping verbosity" ;
     78         setVerbosity(1);
     79     }
     80 
     81 
     82     m_maker = new CMaker(m_ok);
     83 
     84     LOG(trace) << "CTestDetector::init CMaker created" ;
     85 
     86     G4VPhysicalVolume* top = makeDetector();
     87 
     88     LOG(trace) << "CTestDetector::init makeDetector DONE" ;
     89 
     90     setTop(top) ;




Here is the terminator line
----------------------------

::

    tboolean-;tboolean-torus --okg4 -D
    tboolean-torus --okg4 

    ...
    2017-10-27 16:20:23.855 INFO  [1204353] [SLog::operator@15] OpticksHub::OpticksHub DONE

    *************************************************************
     Geant4 version Name: geant4-10-02-patch-01    (26-February-2016)
                          Copyright : Geant4 Collaboration
                          Reference : NIM A 506 (2003), 250-303
                                WWW : http://cern.ch/geant4
    *************************************************************

    2017-10-27 16:20:23.918 FATAL [1204353] [CGeometry::init@59] CGeometry::init G4 simple test geometry 
    2017-10-27 16:20:23.918 INFO  [1204353] [GGeo::createSurLib@725] deferred creation of GSurLib 
    2017-10-27 16:20:23.918 INFO  [1204353] [GSurLib::collectSur@79]  nsur 48
    2017-10-27 16:20:23.919 INFO  [1204353] [CPropLib::init@66] CPropLib::init
    2017-10-27 16:20:23.920 INFO  [1204353] [CPropLib::initCheckConstants@118] CPropLib::initCheckConstants mm 1 MeV 1 nanosecond 1 ns 1 nm 1e-06 GC::nanometer 1e-06 h_Planck 4.13567e-12 GC::h_Planck 4.13567e-12 c_light 299.792 GC::c_light 299.792 dscale 0.00123984
    2017-10-27 16:20:23.921 INFO  [1204353] [*CTestDetector::makeDetector@121] CTestDetector::makeDetector PmtInBox 0 BoxInBox 0 numSolidsMesh 2 numSolidsConfig 0
    Assertion failed: (( is_pib || is_bib ) && "CTestDetector::makeDetector mode not recognized"), function makeDetector, file /Users/blyth/opticks/cfg4/CTestDetector.cc, line 128.
    /Users/blyth/opticks/bin/op.sh: line 754: 70618 Abort trap: 6           /usr/local/opticks/lib/OKG4Test --okg4 --rendermode +global,+axis --animtimemax 20 --timemax 20 --geocenter --stack 2180 --eye 1,0,0 --dbganalytic --test --testconfig analytic=1_csgpath=/tmp/blyth/opticks/tboolean-torus--_name=tboolean-torus--_mode=PyCsgInBox --torch --torchconfig type=discaxial_photons=100000_frame=-1_transform=1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000_source=0,0,0_target=0,0,0_time=0.1_radius=100_distance=400_zenithazimuth=0,1,0,1_material=Vacuum_wavelength=500 --torchdbg --tag 1 --cat tboolean-torus --save
    /Users/blyth/opticks/bin/op.sh RC 134
    simon:opticks blyth$ 




Q & A
------

What kicks off geo conversion ? 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Loosely the instanciation chain:

* --okg4 -> OKG4Test -> OKG4Mgr -> CG4 -> CGeometry -> CTestDetector/GGDMLDetector 

   


GSurLib close issue
---------------------

::

    2017-10-27 19:22:45.382 INFO  [1267219] [CDetector::attachSurfaces@240] CDetector::attachSurfaces
    2017-10-27 19:22:45.382 INFO  [1267219] [GSurLib::examineSolidBndSurfaces@115] GSurLib::examineSolidBndSurfaces numSolids 2
    2017-10-27 19:22:45.382 FATAL [1267219] [GSurLib::examineSolidBndSurfaces@137] GSurLib::examineSolidBndSurfaces i(mm-idx)      0 node(ni.z)      1 node2(id.x)      1 boundary(id.z)    123 parent(ni.w) 4294967295 bname Rock//perfectAbsorbSurface/Vacuum lv World0xc15cfc0
    Assertion failed: (node == i), function examineSolidBndSurfaces, file /Users/blyth/opticks/ggeo/GSurLib.cc, line 147.
    Process 86354 stopped
    * thread #1: tid = 0x135613, 0x00007fff8cc60866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff8cc60866 libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill + 10:
    -> 0x7fff8cc60866:  jae    0x7fff8cc60870            ; __pthread_kill + 20
       0x7fff8cc60868:  movq   %rax, %rdi
       0x7fff8cc6086b:  jmp    0x7fff8cc5d175            ; cerror_nocancel
       0x7fff8cc60870:  retq   
    (lldb) bt
    * thread #1: tid = 0x135613, 0x00007fff8cc60866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff8cc60866 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff842fd35c libsystem_pthread.dylib`pthread_kill + 92
        frame #2: 0x00007fff8b04db1a libsystem_c.dylib`abort + 125
        frame #3: 0x00007fff8b0179bf libsystem_c.dylib`__assert_rtn + 321
        frame #4: 0x00000001020edf0e libGGeo.dylib`GSurLib::examineSolidBndSurfaces(this=0x000000010de3c5d0) + 2110 at GSurLib.cc:147
        frame #5: 0x00000001020ed6bd libGGeo.dylib`GSurLib::close(this=0x000000010de3c5d0) + 29 at GSurLib.cc:93
        frame #6: 0x000000010411a697 libcfg4.dylib`CDetector::attachSurfaces(this=0x000000010de3c4e0) + 247 at CDetector.cc:244
        frame #7: 0x0000000104094c63 libcfg4.dylib`CGeometry::init(this=0x000000010de3c470) + 867 at CGeometry.cc:77
        frame #8: 0x00000001040948f0 libcfg4.dylib`CGeometry::CGeometry(this=0x000000010de3c470, hub=0x000000010950e770) + 112 at CGeometry.cc:50
        frame #9: 0x0000000104094cbd libcfg4.dylib`CGeometry::CGeometry(this=0x000000010de3c470, hub=0x000000010950e770) + 29 at CGeometry.cc:51
        frame #10: 0x000000010413e176 libcfg4.dylib`CG4::CG4(this=0x000000010dd008f0, hub=0x000000010950e770) + 214 at CG4.cc:122
        frame #11: 0x000000010413e70d libcfg4.dylib`CG4::CG4(this=0x000000010dd008f0, hub=0x000000010950e770) + 29 at CG4.cc:144
        frame #12: 0x0000000104231cc3 libokg4.dylib`OKG4Mgr::OKG4Mgr(this=0x00007fff5fbfe500, argc=27, argv=0x00007fff5fbfe5e8) + 547 at OKG4Mgr.cc:35
        frame #13: 0x0000000104231f53 libokg4.dylib`OKG4Mgr::OKG4Mgr(this=0x00007fff5fbfe500, argc=27, argv=0x00007fff5fbfe5e8) + 35 at OKG4Mgr.cc:41
        frame #14: 0x00000001000132ee OKG4Test`main(argc=27, argv=0x00007fff5fbfe5e8) + 1486 at OKG4Test.cc:56
        frame #15: 0x00007fff880d35fd libdyld.dylib`start + 1
        frame #16: 0x00007fff880d35fd libdyld.dylib`start + 1
    (lldb) f 6
    frame #6: 0x000000010411a697 libcfg4.dylib`CDetector::attachSurfaces(this=0x000000010de3c4e0) + 247 at CDetector.cc:244
       241  
       242  
       243      
    -> 244      m_gsurlib->close();
       245   
       246      m_csurlib = new CSurLib(m_gsurlib);
       247  
    (lldb) 


::

    104 void GSurLib::examineSolidBndSurfaces()
    105 {
    106     // this is deferred to CDetector::attachSurfaces 
    107     // to allow CTestDetector to fixup mesh0 info 
    108 
    109     GGeo* gg = m_ggeo ;
    110 
    111     GMergedMesh* mm = gg->getMergedMesh(0) ;
    112 
    113     unsigned numSolids = mm->getNumSolids();
    114 
    115     LOG(info) << "GSurLib::examineSolidBndSurfaces"
    116               << " numSolids " << numSolids
    117               ;
    118 
    119     for(unsigned i=0 ; i < numSolids ; i++)
    120     {
    121         guint4 id = mm->getIdentity(i);
    122         guint4 ni = mm->getNodeInfo(i);
    123         const char* lv = gg->getLVName(i) ;
    124 
    125         // hmm for test geometry the lv returned are the global ones, not the test geometry ones
    126         // and the boundary names look wrong too
    127 
    128         unsigned node = ni.z ;
    129         unsigned parent = ni.w ;
    130 
    131         unsigned node2 = id.x ;
    132         unsigned boundary = id.z ;
    133 
    134         std::string bname = m_blib->shortname(boundary);
    135 
    136         if(node != i)
    137            LOG(fatal) << "GSurLib::examineSolidBndSurfaces"
    138                       << " i(mm-idx) " << std::setw(6) << i
    139                       << " node(ni.z) " << std::setw(6) << node
    140                       << " node2(id.x) " << std::setw(6) << node2
    141                       << " boundary(id.z) " << std::setw(6) << boundary
    142                       << " parent(ni.w) " << std::setw(6) << parent
    143                       << " bname " << bname
    144                       << " lv " << ( lv ? lv : "NULL" )
    145                       ;
    146 
    147         assert( node == i );
    148 
    149 
    150         //unsigned mesh = id.y ;
    151         //unsigned sensor = id.w ;
    152         assert( node2 == i );
    153 
    154         guint4 bnd = m_blib->getBnd(boundary);





review GCSG, ggeo created, used in cfg4
------------------------------------------

GCSG:

* primordial CSG approach, used to describe manual/detdesc analytic PMT
* is referred to in past tense, as regarded as almost dead code, new dev should not use it.
* keeping alive to enable comparisons with new approaches only, until the new approaches can take over
* very limited, sphere/tubs/boolean, to what was needed for DYB PMT


::

    simon:cfg4 blyth$ grep GCSG *.*
    CMaker.cc:#include "GCSG.hh"
    CMaker.cc:G4VSolid* CMaker::makeSolid(GCSG* csg, unsigned int index)
    CMaker.cc:           << "CMaker::makeSolid (GCSG)  "
    CMaker.hh:class GCSG ; 
    CMaker.hh:to convert GCSG geometry into G4 geometry in 
    CMaker.hh:        G4VSolid* makeSolid(GCSG* csg, unsigned int i);  // ancient CSG 
    CTestDetector.cc:#include "GCSG.hh"
    CTestDetector.cc:    GCSG* csg = pmt ? pmt->getCSG() : NULL ;
    CTestDetector.cc:G4LogicalVolume* CTestDetector::makeLV(GCSG* csg, unsigned int i)
    CTestDetector.hh:class GCSG ; 
    CTestDetector.hh:    G4LogicalVolume* makeLV(GCSG* csg, unsigned int i);
    cfg4.bash:     Constitent of CTestDetector used to convert GCSG geometry 
    simon:cfg4 blyth$ 


::

     78 G4VSolid* CMaker::makeSolid(GCSG* csg, unsigned int index)
     79 {
     80    // hmm this is somewhat specialized to known structure of DYB PMT
     81    //  eg intersections are limited to 3 ?
     82 
     83     unsigned int nc = csg->getNumChildren(index);
     84     unsigned int fc = csg->getFirstChildIndex(index);
     85     unsigned int lc = csg->getLastChildIndex(index);
     86     unsigned int tc = csg->getTypeCode(index);
     87     const char* tn = csg->getTypeName(index);
     88 



::

    105 G4VPhysicalVolume* CTestDetector::makeDetector()
    106 {
    107    // analagous to ggeo-/GGeoTest::CreateBoxInBox
    108    // but need to translate from a surface based geometry spec into a volume based one
    109    //
    110    // creates Russian doll geometry layer by layer, starting from the outermost 
    111    // hooking up mother volume to prior 
    112    //
    113     GMergedMesh* mm = m_ggeo->getMergedMesh(0);
    114     unsigned numSolidsMesh = mm->getNumSolids();
    115     unsigned int numSolidsConfig = m_config->getNumElements();
    116 
    117     bool is_pib = isPmtInBox() ;
    118     bool is_bib = isBoxInBox() ;
    119     // CsgInBox not yet handled
    120 
    121     LOG(info)  << "CTestDetector::makeDetector"
    122                << " PmtInBox " << is_pib
    123                << " BoxInBox " << is_bib
    124                << " numSolidsMesh " << numSolidsMesh
    125                << " numSolidsConfig " << numSolidsConfig
    126               ;
    127 
    128     assert( ( is_pib || is_bib ) && "CTestDetector::makeDetector mode not recognized");
    129 





NCSG
------

Huh, made start already.

::

    294 G4VSolid* CMaker::makeSolid(NCSG* csg)
    295 {
    296     nnode* root_ = csg->getRoot();
    297 
    298     G4VSolid* root = makeSolid_r(root_);
    299 
    300     return root  ;
    301 }
    302 
    303 G4VSolid* CMaker::makeSolid_r(const nnode* node)
    304 {
    305     // hmm rmin/rmax is handled as a CSG subtraction
    306     // so could collapse some operators into primitives





