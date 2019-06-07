review-test-geometry
=======================


Objective
---------------

Want to be able to grab solids from the basis geomerty (by lvidx/soidx) 
and use them within the test geometry.

How to do this ?

* means that the CSG nodetree coming in from the python must act as a proxy for the basis nodetree
* actually just a single node is needed, hmm could use a box that is used to contain the proxied tree

  * can just look for "proxylvid" on root nodes of CSG trees

* am not changing the russian doll assumption, just need to insert a volume or volumes into the list 
 
  * but will I need to create corresponding NCSG ?



Thoughts
------------

* it aint easy with what legacy persists, so dont complicate the 
  situation by trying to do it there 

* instead as a 1st step bring test running into direct workflow  : DONE 

* with direct workflow have advantage that can cycle on creating the
  base geocache much more easily : so all stages of processing are within easy reach

* but still now to do it ?  start by checking X4PhysicalVolume::convertSolids

And the answer looks like::

    488 GMesh* X4PhysicalVolume::convertSolid( int lvIdx, int soIdx, const G4VSolid* const solid, const std::string& lvname) const
    489 {
    490      assert( lvIdx == soIdx );
    491 
    492      bool dbglv = lvIdx == m_ok->getDbgLV() ;
    493      LOG(info) << " [ "
    494                << ( dbglv ? " --dbglv " : "" )
    495                 << lvIdx << " " << lvname ;
    496 
    497      nnode* raw = X4Solid::Convert(solid, m_ok)  ;
    ... 
    518      nnode* root = NTreeProcess<nnode>::Process(raw, soIdx, lvIdx);  // balances deep trees
    519      root->other = raw ;
    520 
    521 
    522      const NSceneConfig* config = NULL ;
    523      NCSG* csg = NCSG::Adopt( root, config, soIdx, lvIdx );   // Adopt exports nnode tree to m_nodes buffer in NCSG instance
    524      assert( csg ) ;
    525      assert( csg->isUsedGlobally() );
    526 
    527      const std::string& soname = solid->GetName() ;
    528      csg->set_soname( soname.c_str() ) ;
    529      csg->set_lvname( lvname.c_str() ) ;
    530 
    531      bool is_x4polyskip = m_ok->isX4PolySkip(lvIdx);   // --x4polyskip 211,232
    532      if( is_x4polyskip ) LOG(fatal) << " is_x4polyskip " << " soIdx " << soIdx  << " lvIdx " << lvIdx ;
    533 
    534      GMesh* mesh =  is_x4polyskip ? X4Mesh::Placeholder(solid ) : X4Mesh::Convert(solid ) ;
    535      mesh->setCSG( csg );
    536 
    537      LOG(info) << " ] " << lvIdx ;
    538      return mesh ;
    539 }


* the GMesh have associated NCSG which get added to GGeo and have lvIdx set as their index
* so looks like just need to persist the GMesh+NCSG separately as well as collectively into the geocache 
  in addition to the mainline GMergedMesh 

  * the GMesh are already persisted in the GMeshLib, but associated NCSG are not 
  * need to get the NCSGData src vs transport buffer distinction clear  

* the GMesh are a solid level thing (not node level) so not so many of them (less than 40 for JUNO) 




  

Bringing tboolean-box into direct workflow
----------------------------------------------

* this means basing the test geometry off of the direct geocache

Hmm still picking up legacy geomety after unset IDPATH ?::

    [blyth@localhost tmp]$ js.py tboolean-box/evt/tboolean-box/torch/-1/parameters.json  | egrep 'Detector|GEOCACHE|KEY' 
     u'Detector': u'dayabay',
     u'GEOCACHE': u'/home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae',
     u'KEY': u'no-key-spec',


* Ahha, its because are using op.sh that diddles the environment
* created a simpler o.sh to replace op.sh that gets this working in direct workflow

::

    blyth@localhost tmp]$ js.py tboolean-box/evt/tboolean-box/torch/-1/parameters.json  | egrep 'Detector|GEOCACHE|KEY' 
     u'Detector': u'g4live',
     u'GEOCACHE': u'/home/blyth/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/f6cc352e44243f8fa536ab483ad390ce/1',
     u'KEY': u'OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.f6cc352e44243f8fa536ab483ad390ce',




Resolve proxy in GGeoTest::importCSG ?
--------------------------------------------



GNodeLib persists only names
-----------------------------------

* GNodeLib in memory stores GVolume (which have associated GParts), 
  but when persisted to file is just name lists 

::

    [blyth@localhost 1]$ l GNodeLib/
    total 24428
    -rw-rw-r--. 1 blyth blyth 6052728 May 25 10:54 GTreePresent.txt
    -rw-rw-r--. 1 blyth blyth 9152343 May 25 10:54 LVNames.txt
    -rw-rw-r--. 1 blyth blyth 9804263 May 25 10:54 PVNames.txt

* GMergedMesh combines volumes and GParts and persists 

::

     23 GVolume::GVolume( unsigned int index, GMatrix<float>* transform, const GMesh* mesh, unsigned int boundary, NSensor* sensor)
     24     :
     25     GNode(index, transform, mesh ),
     26     m_boundary(boundary),
     27     m_csgflag(CSG_PARTLIST),
     28     m_csgskip(false),
     29     m_sensor(sensor),
     30     m_pvname(NULL),
     31     m_lvname(NULL),
     32     m_sensor_surface_index(0),
     33     m_parts(NULL),
     34     m_parallel_node(NULL)
     35 {
     36 }



Hmm have to reconstitue the GVolume from GMergedMesh ?
----------------------------------------------------------

* GMergedMesh m_nodeinfo has volume level information 


::

     551 void GMergedMesh::mergeVolumeIdentity( GVolume* volume, bool selected )
     552 {
     553     const GMesh* mesh = volume->getMesh();
     554 
     555     unsigned nvert = mesh->getNumVertices();
     556     unsigned nface = mesh->getNumFaces();
     557 
     558     guint4 _identity = volume->getIdentity();
     559 
     560     unsigned nodeIndex = volume->getIndex();
     561     unsigned meshIndex = mesh->getIndex();
     562     unsigned boundary = volume->getBoundary();
     563 
     564     NSensor* sensor = volume->getSensor();
     565     unsigned sensorIndex = NSensor::RefIndex(sensor) ;
     566 
     567     assert(_identity.x == nodeIndex);
     568     assert(_identity.y == meshIndex);
     569     assert(_identity.z == boundary);
     570     //assert(_identity.w == sensorIndex);   this is no longer the case, now require SensorSurface in the identity
     571 
     572     LOG(debug) << "GMergedMesh::mergeVolumeIdentity"
     573               << " m_cur_volume " << m_cur_volume
     574               << " nodeIndex " << nodeIndex
     575               << " boundaryIndex " << boundary
     576               << " sensorIndex " << sensorIndex
     577               << " sensor " << ( sensor ? sensor->description() : "NULL" )
     578               ;
     579 
     580 
     581     GNode* parent = volume->getParent();
     582     unsigned int parentIndex = parent ? parent->getIndex() : UINT_MAX ;
     583 

     584     m_meshes[m_cur_volume] = meshIndex ;
     585 
     586     // face and vertex counts must use same selection as above to be usable 
     587     // with the above filled vertices and indices 
     588 
     589     m_nodeinfo[m_cur_volume].x = selected ? nface : 0 ;
     590     m_nodeinfo[m_cur_volume].y = selected ? nvert : 0 ;
     591     m_nodeinfo[m_cur_volume].z = nodeIndex ;
     592     m_nodeinfo[m_cur_volume].w = parentIndex ;
     593 


For global mm0 in juno directly converted geometry (kcd)::

    [blyth@localhost 0]$ np.py nodeinfo.npy -viF -s 0:20
    a :                                                 nodeinfo.npy :          (366697, 4) : 0df666cebed04081b722d1fb60c54b1c : 20190525-1054 
    (366697, 4)
    i32
    [[ 12   8   0  -1]
     [ 12   8   1   0]
     [ 12   8   2   1]
     [ 96  50   3   2]
     [ 96  50   4   3]
     [192  96   5   3]
     [192  96   6   3]
     [108  58   7   2]
     [ 12   8   8   7]
     [ 12   8   9   8]
     [  0   0  10   9]
     [  0   0  11  10]
     [  0   0  12  11]
     [  0   0  13  12]
     [  0   0  14  11]
     [  0   0  15  14]
     [  0   0  16  11]
     [  0   0  17  16]
     [  0   0  18  11]
     [  0   0  19  18]]


    [blyth@localhost 1]$ np.py GMergedMesh/0/nodeinfo.npy -viF -s _20:_1
    a :                                   GMergedMesh/0/nodeinfo.npy :          (366697, 4) : 0df666cebed04081b722d1fb60c54b1c : 20190525-1054 
    (366697, 4)
    i32
    [[     0      0 366677 366676]
     [     0      0 366678 366676]
     [     0      0 366679  62590]
     [     0      0 366680 366679]
     [     0      0 366681 366679]
     [     0      0 366682 366681]
     [     0      0 366683 366682]
     [     0      0 366684 366682]
     [     0      0 366685  62590]
     [     0      0 366686 366685]
     [     0      0 366687 366685]
     [     0      0 366688 366687]
     [     0      0 366689 366688]
     [     0      0 366690 366688]
     [     0      0 366691  62590]
     [     0      0 366692 366691]
     [     0      0 366693 366691]
     [     0      0 366694 366693]
     [     0      0 366695 366694]]



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


