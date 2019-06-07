review-test-geometry
=======================


Objective
---------------

Want to be able to grab solids from the basis geomerty (by lvidx/soidx) 
and use them within the test geometry.

How to do this ?

* means that the CSG nodetree coming in from the python must act as a proxy for the basis nodetree
* actually just a single node is needed, hmm could use a box that is used to contain the proxied tree
* am not changing the russian doll assumption, just need to insert a volume or volumes into the list 
 

isTest from --test option
------------------------------

::

    [blyth@localhost opticks]$ opticks-f ">isTest()"
    ./cfg4/CGeometry.cc:    if(m_ok->isTest())  // --test
    ./ggeo/GGeo.cc:    if( m_ok->isTest() )
    ./opticksgeo/OpticksHub.cc:    if(m_ok->isTest())
    ./opticksgeo/OpticksHub.cc:    assert(m_ok->isTest());
    ./opticksgeo/OpticksHub.cc:    if(m_ok->isTest())
    ./opticksgeo/OpticksHub.cc:    bool test = m_ok->isTest() ; 
    ./optickscore/OpticksAna.cc:    if(m_ok->isTest())


CGeometry : --test branches between  CGDMLDetector and CTestDetector
~~~~~~~~~~~~~~~~~~~~~~~~~~~-------~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

     63 void CGeometry::init()
     64 {
     65     CDetector* detector = NULL ;
     66     if(m_ok->isTest())  // --test
     67     {
     68         LOG(fatal) << "G4 simple test geometry " ;
     69         OpticksQuery* query = NULL ;  // normally no OPTICKS_QUERY geometry subselection with test geometries
     70         detector  = static_cast<CDetector*>(new CTestDetector(m_hub, query, m_sd)) ;
     71     }
     72     else
     73     {
     74         // no options here: will load the .gdml sidecar of the geocache .dae 
     75         LOG(fatal) << "G4 GDML geometry " ;
     76         OpticksQuery* query = m_ok->getQuery();
     77         detector  = static_cast<CDetector*>(new CGDMLDetector(m_hub, query, m_sd)) ;
     78     }
     79 
     80     // detector->attachSurfaces();  moved into the ::init of CTestDetector and CGDMLDetector to avoid omission
     81 
     82     m_detector = detector ;
     83     m_mlib = detector->getMaterialLib();
     84 }
     85 


GGeo : switch off using lv2sd association for test geometry, as the LV will not be present
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

     755 void GGeo::loadCacheMeta() // loads metadata that the process that created the geocache persisted into the geocache
     756 {
     ...
     779 
     780     if( m_ok->isTest() )
     781     {
     782          LOG(error) << "NOT USING the lv2sd association as --test is active " ;
     783     }
     784     else
     785     {
     786          m_lv2sd = lv2sd ;
     787     }
     788 }


OpticksAna : commented out
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

     63 void OpticksAna::setEnv()
     64 {
     65     if(m_ok->isTest())
     66     {
     67 
     68         /*
     69         const char* key = "OPTICKS_EVENT_BASE" ;  
     70         const char* evtbase = BResource::GetDir("evtbase"); 
     71         LOG(info) << " setting envvar key " << key << " evtbase " << evtbase ; 
     72         SSys::setenvvar(key, evtbase ); 
     73 
     74         formerly thought should be example specific /tmp/tboolean-box
     75         but now think that is a mistake, much better for OPTICKS_EVENT_BASE 
     76         to be more stable than that and not include specifics, 
     77         eg /tmp OR /tmp/$USER/opticks
     78 
     79         */
     80 
     81     }
     82 }



OpticksHub::loadGeometry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

     486 void OpticksHub::loadGeometry()
     487 {
     488     assert(m_geometry == NULL && "OpticksHub::loadGeometry should only be called once");
     489 
     490     LOG(info) << "[ " << m_ok->getIdPath()  ;
     491 
     492     m_geometry = new OpticksGeometry(this);   // m_lookup is set into m_ggeo here 
     493 
     494     m_geometry->loadGeometry();
     495 
     496     m_ggeo = m_geometry->getGGeo();
     497 
     498     m_gscene = m_ggeo->getScene();
     499 
     500 
     501     //   Lookup A and B are now set ...
     502     //      A : by OpticksHub::configureLookupA (ChromaMaterialMap.json)
     503     //      B : on GGeo loading in GGeo::setupLookup
     504 
     505 
     506     if(m_ok->isTest())  // --test 
     507     {
     508         LOG(info) << "--test modifying geometry" ;
     509 
     510         assert(m_geotest == NULL);
     511 
     512         GGeoBase* basis = getGGeoBasePrimary(); // ana OR tri depending on --gltf
     513 
     514         m_geotest = createTestGeometry(basis);
     515 
     516         int err = m_geotest->getErr() ;
     517         if(err)
     518         {
     519             setErr(err);
     520             return ;
     521         }
     522     }
     523     else
     524     {
     525         LOG(LEVEL) << "NOT modifying geometry" ;
     526     }
     527 
     528     registerGeometry();
     529 
     530     m_ggeo->setComposition(m_composition);
     531 
     532     m_ggeo->close();  // mlib and slib  (June 2018, following remove the auto-trigger-close on getIndex in the proplib )
     533 
     534     LOG(info) << "]" ;
     535 }


     556 GGeoTest* OpticksHub::createTestGeometry(GGeoBase* basis)
     557 {
     558     assert(m_ok->isTest());  // --test
     559 
     560     LOG(info) << "[" ;
     561 
     562     GGeoTest* testgeo = new GGeoTest(m_ok, basis);
     563 
     564     LOG(info) << "]" ;
     565 
     566     return testgeo ;
     567 }

     653 void OpticksHub::configureGeometry()
     654 {
     655     if(m_ok->isTest()) // --test
     656     {
     657         configureGeometryTest();
     658     }
     659     else if(m_gltf==0)
     660     {
     661         configureGeometryTri();
     662     }
     663     else
     664     {
     665         configureGeometryTriAna();
     666     }
     667 }



GGeoTest
------------

* has its own instances of the material and surface libs, but based apon those from the basis geometry
* see comments added to ggeo/GGeoTest.cc

::

    096 GGeoTest::GGeoTest(Opticks* ok, GGeoBase* basis)
     97     :
     98     m_ok(ok),
     99     m_config_(ok->getTestConfig()),
    100     m_config(new NGeoTestConfig(m_config_)),
    101     m_verbosity(m_config->getVerbosity()),
    102     m_resource(ok->getResource()),
    103     m_dbgbnd(m_ok->isDbgBnd()),
    104     m_dbganalytic(m_ok->isDbgAnalytic()),
    105     m_lodconfig(ok->getLODConfig()),
    106     m_lod(ok->getLOD()),
    107     m_analytic(m_config->getAnalytic()),
    108     m_csgpath(m_config->getCSGPath()),
    109     m_test(true),
    110     m_basis(basis),
    111     m_pmtlib(basis->getPmtLib()),
    112     m_mlib(new GMaterialLib(m_ok, basis->getMaterialLib())),
    113     m_slib(new GSurfaceLib(m_ok, basis->getSurfaceLib())),
    114     m_bndlib(new GBndLib(m_ok, m_mlib, m_slib)),
    115     m_geolib(new GGeoLib(m_ok,m_analytic,m_bndlib)),
    116     m_nodelib(new GNodeLib(m_ok, m_analytic, m_test)),
    117     m_maker(new GMaker(m_ok, m_bndlib)),
    118     m_csglist(m_csgpath ? NCSGList::Load(m_csgpath, m_verbosity ) : NULL),
    119     m_solist(new GVolumeList()),
    120     m_err(0)
    121 {
    122     assert(m_basis);
    123 
    124     init();
    125 }
    126 



CTestDetector
-----------------

::

     53 CTestDetector::CTestDetector(OpticksHub* hub, OpticksQuery* query, CSensitiveDetector* sd)
     54     :
     55     CDetector(hub, query, sd),
     56     m_geotest(hub->getGGeoTest()),
     57     m_config(m_geotest->getConfig())
     58 {
     59     init();
     60 }A


CTestDetector::makeDetector_NCSG
---------------------------------

Converts the list of GVolumes obtained from GNodeLib, 
which are assumed to have a simple Russian-doll geometry into a Geant4
volume "tree" structure. 


