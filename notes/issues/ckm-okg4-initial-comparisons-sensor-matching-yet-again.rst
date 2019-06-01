ckm-okg4-initial-comparisons-sensor-matching-yet-again
=======================================================

Context
----------

* :doc:`ckm-okg4-initial-comparisons`


G4-G4 comparison, 1st:CerenkovMinimal 2nd:OKG4Test
-------------------------------------------------------

Very nearly same generated photons, but no hits with 2nd executable::

    [blyth@localhost 1]$ np.py {source,OKG4Test}/evt/g4live/natural/-1/so.npy -T
    a :                          source/evt/g4live/natural/-1/so.npy :          (221, 4, 4) : d1b4242225f7ffc7f0ad38a9669562a4 : 20190531-1723 
    b :                        OKG4Test/evt/g4live/natural/-1/so.npy :          (221, 4, 4) : 882f44b7864bfcde55fe2ebe922895e5 : 20190531-1839 
     max(a-b)   5.96e-08  min(a-b)  -5.96e-08 


    [blyth@localhost 1]$ np.py {source,OKG4Test}/evt/g4live/natural/-1 -T
    /home/blyth/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/27d088654714cda61096045ff5eacc02/1/source/evt/g4live/natural/-1
    . :                          source/evt/g4live/natural/-1/ht.npy :          (108, 4, 4) : f151301a12d1874e9447fd916e7f8719 : 20190531-1723 
    . :                          source/evt/g4live/natural/-1/so.npy :          (221, 4, 4) : d1b4242225f7ffc7f0ad38a9669562a4 : 20190531-1723 
    /home/blyth/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/27d088654714cda61096045ff5eacc02/1/OKG4Test/evt/g4live/natural/-1
    . :                      OKG4Test/evt/g4live/natural/-1/fdom.npy :            (3, 1, 4) : f8c7c318e47b0ccb2c29567e87d95e67 : 20190531-1839 
    . :                        OKG4Test/evt/g4live/natural/-1/gs.npy :            (1, 6, 4) : f21b4f0138c122a64319243596bb2228 : 20190531-1839 
    . :                        OKG4Test/evt/g4live/natural/-1/ht.npy :            (0, 4, 4) : 40d1000a029fc713333b79245d7141c1 : 20190531-1839 
    . :                      OKG4Test/evt/g4live/natural/-1/idom.npy :            (1, 1, 4) : a910ad1008e847548261491f9ca73f9c : 20190531-1839 
    . :                        OKG4Test/evt/g4live/natural/-1/ox.npy :          (221, 4, 4) : 0c933fd9fdab9d2975af9e6871351e46 : 20190531-1839 
    . :                        OKG4Test/evt/g4live/natural/-1/ph.npy :          (221, 1, 2) : 0a50e4992b98714e0391cd6d8deadc9e : 20190531-1839 
    . :                        OKG4Test/evt/g4live/natural/-1/ps.npy :          (221, 1, 4) : 2f17ee76054cc1040f30bee0a8a0153e : 20190531-1839 
    . :                        OKG4Test/evt/g4live/natural/-1/rs.npy :      (221, 10, 1, 4) : 629500c344dc05dbc6777ccf6f386fe5 : 20190531-1839 
    . :                        OKG4Test/evt/g4live/natural/-1/rx.npy :      (221, 10, 2, 4) : 2ce8d2aafab81f6d6f0e6a1cc1877646 : 20190531-1839 
    . :                        OKG4Test/evt/g4live/natural/-1/so.npy :          (221, 4, 4) : 882f44b7864bfcde55fe2ebe922895e5 : 20190531-1839 


* GDML is dropping the ball wrt sensitive detectors




ckm-- CerenkovMinimal SensitiveDetector::ProcessHits calls G4Opticks::collectHit 
----------------------------------------------------------------------------------------------

::

     34 G4bool SensitiveDetector::ProcessHits(G4Step* step,G4TouchableHistory* )
     35 {
     36     G4Track* track = step->GetTrack();
     37     if (track->GetDefinition() != G4OpticalPhoton::Definition()) return false ;
     38 
     39     G4double ene = step->GetTotalEnergyDeposit();
     40     G4StepPoint* point = step->GetPreStepPoint();
     41     G4double time = point->GetGlobalTime();
     42     const G4ThreeVector& pos = point->GetPosition();
     43     const G4ThreeVector& dir = point->GetMomentumDirection();
     44     const G4ThreeVector& pol = point->GetPolarization();
     45 
     46     m_hit_count += 1 ;
     47 
     48 #ifdef WITH_OPTICKS
     49     {
     50         G4double energy = point->GetKineticEnergy();
     51         G4double wavelength = h_Planck*c_light/energy ;
     52         G4double weight = 1.0 ;
     53         G4int flags_x = 0 ;
     54         G4int flags_y = 0 ;
     55         G4int flags_z = 0 ;
     56         G4int flags_w = 0 ;
     57  
     58         G4Opticks::GetOpticks()->collectHit(
     59              pos.x()/mm,
     60              pos.y()/mm,
     61              pos.z()/mm,
     62              time/ns,
     63 
     64              dir.x(),


Uses::

     m_g4hit_collector = new CPhotonCollector ;



The below are from the GPU propagation::

     30 void EventAction::EndOfEventAction(const G4Event* event)
     31 {
     32     G4HCofThisEvent* HCE = event->GetHCofThisEvent() ;
     33     assert(HCE);
     34 
     35 #ifdef WITH_OPTICKS
     36     G4cout << "\n###[ EventAction::EndOfEventAction G4Opticks.propagateOpticalPhotons\n" << G4endl ;
     37 
     38     G4Opticks* ok = G4Opticks::GetOpticks() ;
     39     int num_hits = ok->propagateOpticalPhotons() ;
     40     NPY<float>* hits = ok->getHits();
     41 
     42     assert( hits == NULL || hits->getNumItems() == unsigned(num_hits) ) ;
     43     G4cout
     44            << "EventAction::EndOfEventAction"
     45            << " num_hits " << num_hits
     46            << " hits " << hits
     47            << G4endl
     48            ;
     49 
     50     // TODO: feed the hits into the Hit collection 
     51 
     52     G4cout << "\n###] EventAction::EndOfEventAction G4Opticks.propagateOpticalPhotons\n" << G4endl ;
     53 #endif
     54 
     55     //addDummyHits(HCE);
     56     G4cout
     57          << "EventAction::EndOfEventAction"
     58          << " DumpHitCollections "
     59          << G4endl
     60          ;
     61     SensitiveDetector::DumpHitCollections(HCE);
     62 
     63     // A possible alternative location to invoke the GPU propagation
     64     // and add hits in bulk to hit collections would be SensitiveDetector::EndOfEvent  
     65 }




This is a model matching problem that I've jousted with before : review notes
---------------------------------------------------------------------------------

* :doc:`G4OK_SD_Matching`

   Big picture view of hit formation in G4 and OK, 

   * G4 forms hits only on a material with EFFICIENCY 
   * OK yields SA/SD with propagate_at_surface

* :doc:`stepping_process_review`

   Low level look at G4SteppingManager and CSteppingAction tricks to align RNG consumption, 
   including lldb python scripted breakpointing 

* :doc:`cfg4-bouncemax-not-working`

   Getting hall-of-mirrors tboolean-truncate to agree 

* :doc:`geant4_opticks_integration/tconcentric_post_recording_has_seqmat_zeros`

   Fixing bugs with the switch from live to canned mode in CRecorder


* :doc:`direct_route_needs_AssimpGGeo_convertSensors_equivalent` 2018-08-03  (during direct dev summer)

  Discusses the pseudo SensorSurface and model matching  

  ... With direct geomerty : I have access to original in memory Geant4 geometry model ... so can persist my way 

  * TRUE, but want to support running from bare GDML without sidecar, so need a commandline way 


* :doc:`g4ok_investigate_zero_hits` 20190313

   No hits from OK, due to "--bouncemax 0" in embedded commandline

* :doc:`g4ok_hit_matching` 20190313

   Concluded : For step by step debugging need the instrumented executable to work from gensteps. Have this now.

* :doc:`g4ok_direct_conversion_of_sensors_review` 2019-03-13

   

Searching for Cathode::

    31 /**
    232 CGDMLDetector::kludge_cathode_efficiency
    233 -----------------------------------------
    234 
    235 NOT NEEDED Cathode Efficiency fixup is done by CPropLib AFTER FIXING A KEY BUG 
    236 
    237 See :doc:`notes/issues/direct_route_needs_AssimpGGeo_convertSensors_equivalent`
    238 ...


lvsdname
------------

OpticksCfg.cc --lvsdname:: 

     803    char lvsdname[512];
     804    snprintf(lvsdname,512,
     805 "When lvsdname is blank logical volumes with an SD associated are used. "
     806 "As workaround for GDML not persisting the SD it is also possible to identify SD by their LV names, using this option. "
     807 "Provide a comma delimited string with substrings to search for in the logical volume names "
     808 "when found the volumes will be treated as sensitive detectors, see X4PhysicalVolume::convertSensors "
     809 "Default %s ",  m_lvsdname.c_str() );
     810 
     811    m_desc.add_options()
     812        ("lvsdname",   boost::program_options::value<std::string>(&m_lvsdname), lvsdname ) ;
     813 


* lvsdname is a crutch to get sensitivity into GGeo when running from GDML, when running from 
  a DetectorConstruction eg with CerenkovMinimal the lvsdname is not needed 


lvsdname is not the answer, not yet anyhow
------------------------------------------------

* it does not yet help with 2nd executable bi-simulation running (eg OKG4Test) where
  
  1. Opticks model is booted from geocache (including sensitivity) 
  2. Geant4 model is booted from GDML (with material properties grabbed from Opticks for consistency)


* THIS SUGGESTS THE THING TO TRY : Geant4 model needs some sensor fixup, borrowing from Opticks again
  (maybe CPropLib ?)

  * but did I not do this before ?
  * 2nd executable bi-simulation is a new thing ? new for gensteps, not for input photons ? 
  * need a back translation from Opticks SensorSurface into Geant4 SD 

* perhaps it would have been simpler to fix sensitivity at Geant4 level just after loading GDML,
  rather than having to fix it within both models 


CerenkovMinimal DetectorConstruction
----------------------------------------

* the below association of the SD with the volume is missed in the 2nd executable OKG4Test

::

    187     G4Material* glass = MakeGlass();    // slab of sensitive glass in the water 
    188     AddProperty(glass, "EFFICIENCY", MakeConstantProperty(0.5));
    189     
    190     G4Box* so_2 = new G4Box("Det",400.,400.,10.);  // half sizes 
    191     G4LogicalVolume* lv_2 = new G4LogicalVolume(so_2,glass,"Det",0,0,0);
    192     G4VPhysicalVolume* pv_2 = new G4PVPlacement(0,G4ThreeVector(0,0,100.),lv_2 ,"Det",lv_1,false,0);
    193     assert( pv_2 );
    194     
    195     G4SDManager* SDMan = G4SDManager::GetSDMpointerIfExist();        assert( SDMan && " SDMan should have been created before now " );
    196     G4VSensitiveDetector* sd = SDMan->FindSensitiveDetector(sdname); assert( sd && " failed for find sd with sdname " );
    197     lv_2->SetSensitiveDetector(sd);
    198     
    199     
    200     const std::string& lv_1_name = lv_1->GetName() ; 
    201     //std::cout << " lv_1_name " << lv_1_name << std::endl ; 
    202     assert( strcmp( lv_1_name.c_str(), "Obj" ) == 0 );
    203 
    204     G4cout << "] DetectorConstruction::Construct " << G4endl ;
    205     
    206     return pv_0 ;
    207 }


TO TRY::

   do this lv2sd association in OKG4Test with a kludge, and then workout how to do it more generally 

::

    [blyth@localhost 1]$ opticks-f SetSensitiveDetector
    ./cfg4/CDetector.cc:             << "SetSensitiveDetector"
    ./cfg4/CDetector.cc:        const_cast<G4LogicalVolume*>(lv)->SetSensitiveDetector(m_sd) ; 
    ./examples/Geant4/CerenkovMinimal/DetectorConstruction.cc:    lv_2->SetSensitiveDetector(sd); 
    ./examples/Geant4/GDMLMangledLVNames/DetectorConstruction.cc:        lv_2->SetSensitiveDetector(sd); 



GGeo GSurfaceLib in geocache : the information is there pointing at the right volume
---------------------------------------------------------------------------------------

::

    [blyth@localhost 1]$ jsn.py GSurfaceLib/GPropertyLibMetadata.json
    {u'DetSensorSurface': {u'index': 3,
                           u'name': u'DetSensorSurface',
                           u'shortname': u'DetSensorSurface',
                           u'sslv': u'Det0x169a290',
                           u'type': u'skinsurface'},
     u'perfectAbsorbSurface': {u'index': 1000,
                               u'name': u'perfectAbsorbSurface',
                               u'shortname': u'perfectAbsorbSurface',
                               u'type': u'testsurface'},
     u'perfectDetectSurface': {u'index': 1000,
                               u'name': u'perfectDetectSurface',
                               u'shortname': u'perfectDetectSurface',
                               u'type': u'testsurface'},
     u'perfectDiffuseSurface': {u'index': 1000,
                                u'name': u'perfectDiffuseSurface',
                                u'shortname': u'perfectDiffuseSurface',
                                u'type': u'testsurface'},
     u'perfectSpecularSurface': {u'index': 1000,
                                 u'name': u'perfectSpecularSurface',
                                 u'shortname': u'perfectSpecularSurface',
                                 u'type': u'testsurface'}}


::

    119   <structure>
    120     <volume name="Det0x169a290">
    121       <materialref ref="Glass0x1698560"/>
    122       <solidref ref="Det0x169a230"/>
    123     </volume>




CDetector::hookupSD now succeeds after fixing GGeo persisting of m_lv2sd 
-----------------------------------------------------------------------------

ckm-okg4::

    2019-05-31 20:12:06.517 ERROR [99663] [CDetector::hookupSD@129]  NOT INVOKING SetSensitiveDetector ON ANY VOLUMES AS nlvsd is zero or m_sd NULL  nlvsd 0 m_sd 0x1bb8770 sdname SD0
    2019-05-31 20:12:06.517 INFO  [99663] [CGDMLDetector::CGDMLDetector@44] ]

After fixing the GGeo persisting of m_lv2sd metadata via geocache::

    2019-05-31 20:53:17.620 ERROR [169415] [CDetector::hookupSD@151] SetSensitiveDetector lvn Det0x20a8260 sdn SD0 lv 0x2469c20
    2019-05-31 20:53:17.620 INFO  [169415] [CGDMLDetector::CGDMLDetector@44] ]


::

    123 void CDetector::hookupSD()
    124 {
    125     unsigned nlvsd = m_ggeo->getNumLVSD() ;
    126     const std::string sdname = m_sd ? m_sd->GetName() : "noSD" ;
    127     if(nlvsd == 0 || m_sd == NULL )
    128     {
    129         LOG(error)
    130             << " NOT INVOKING SetSensitiveDetector ON ANY VOLUMES AS nlvsd is zero or m_sd NULL "
    131             << " nlvsd " << nlvsd
    132             << " m_sd " << m_sd
    133             << " sdname " << sdname
    134             ;
    135     }
    136 
    137 
    138     if(!m_sd) return ;
    139     for( unsigned i = 0 ; i < nlvsd ; i++)
    140     {
    141         std::pair<std::string,std::string> lvsd = m_ggeo->getLVSD(i) ;
    142         const char* lvn = lvsd.first.c_str();
    143         const char* sdn = lvsd.second.c_str();
    144 
    145         //assert( strcmp( sdname.c_str(), sdn ) == 0 ) ;  
    146 
    147         //const char* lvn = m_ggeo->getCathodeLV(i); 
    148 
    149         const G4LogicalVolume* lv = m_traverser->getLV(lvn);
    150 
    151         LOG(error)
    152              << "SetSensitiveDetector"
    153              << " lvn " << lvn
    154              << " sdn " << sdn
    155              << " lv " << lv
    156              ;
    157 
    158         if(!lv) LOG(fatal) << " no lv " << lvn ;
    159         assert(lv);
    160 
    161         const_cast<G4LogicalVolume*>(lv)->SetSensitiveDetector(m_sd) ;
    162     }
    163 }



GGeoTest
-------------

::

    ckm-ggeotest(){  OPTICKS_KEY=$(ckm-key) $(ckm-dbg) GGeoTest --envkey ; }

    158 void test_GGeo_sd(const GGeo* m_ggeo)
    159 {
    160     unsigned nlvsd = m_ggeo->getNumLVSD() ;
    161     LOG(info) << " nlvsd " << nlvsd ;
    162 }


ckm-ggeotest::

    2019-05-31 20:17:12.825 INFO  [107719] [test_GGeo_sd@161]  nlvsd 0


GGeo
-----------

::

     301 /**
     302 GGeo::addLVSD
     303 -------------------
     304 
     305 From  
     306 
     307 1. AssimpGGeo::convertSensorsVisit
     308 2. X4PhysicalVolume::convertSensors_r
     309 
     310 **/
     311 
     312 void GGeo::addLVSD(const char* lv, const char* sd)
     313 {
     314    assert( lv ) ;
     315    m_cathode_lv.insert(lv);
     316 
     317    if(sd)
     318    {
     319        if(m_lv2sd == NULL ) m_lv2sd = new NMeta ;
     320        m_lv2sd->set<std::string>(lv, sd) ;
     321    }
     322 }
     323 unsigned GGeo::getNumLVSD() const
     324 {
     325    return m_lv2sd ? m_lv2sd->getNumKeys() : 0 ;
     326 }
     327 std::pair<std::string,std::string> GGeo::getLVSD(unsigned idx) const
     328 {
     329     const char* lv = m_lv2sd->getKey(idx) ;
     330     std::string sd = m_lv2sd->get<std::string>(lv);
     331     return std::pair<std::string,std::string>( lv, sd );
     332 }






CFG4 : CG4, CSensitiveDetector, has hardcoded sd and collection names ?
---------------------------------------------------------------------------

::

    108 CG4::CG4(OpticksHub* hub) 
    109     :
    110     m_log(new SLog("CG4::CG4", "", fatal)),
    111     m_hub(hub),
    112     m_ok(m_hub->getOpticks()),
    113     m_run(m_ok->getRun()),
    114     m_cfg(m_ok->getCfg()),
    115     m_ctx(m_ok),
    116     //m_engine(m_ok->isAlign() ? (CRandomListener*)new CRandomEngine(this) : (CRandomListener*)new CMixMaxRng ),
    117     m_engine(m_ok->isAlign() ? (CRandomListener*)new CRandomEngine(this) : NULL  ),
    118     m_physics(new CPhysics(this)),
    119     m_runManager(m_physics->getRunManager()),
    120     m_sd(new CSensitiveDetector("SD0")),
    121     m_geometry(new CGeometry(m_hub, m_sd)),
    122     m_hookup(m_geometry->hookup(this)),
    123     m_mlib(m_geometry->getMaterialLib()),
    124     m_detector(m_geometry->getDetector()),
    125     m_generator(new CGenerator(m_hub->getGen(), this)),


::

     09 const char* CSensitiveDetector::SDName = NULL ; 
     10 const char* CSensitiveDetector::collectionNameA = "OpHitCollectionA" ;
     11 const char* CSensitiveDetector::collectionNameB = "OpHitCollectionB" ;
     12         
     13 CSensitiveDetector::CSensitiveDetector(const char* name)
     14     :
     15     G4VSensitiveDetector(name)
     16 {
     17     SDName = strdup(name) ; 
     18     collectionName.insert(collectionNameA);
     19     collectionName.insert(collectionNameB); 
     20         
     21     G4SDManager* SDMan = G4SDManager::GetSDMpointer() ;
     22     SDMan->AddNewDetector(this); 
     23 }


X4
----

* OKX4Test loads geometry from GDML, and creates GGeo geocache using X4PhysicalVolume
* G4Opticks::setGeometry as used by CerenkovMinimal does the same, also using X4PhysicalVolume

  * TODO: find a good (and short) name for a top level X4 interface class that uses X4PhysicalVolume
    eg X4Geo/X4Top/X4World/... 


::

    182 /**
    183 X4PhysicalVolume::convertSensors_r
    184 -----------------------------------
    185 
    186 Sensors are identified by two approaches:
    187 
    188 1. logical volume having an associated sensitive detector G4VSensitiveDetector
    189 2. name of logical volume matching one of a comma delimited list 
    190    of strings provided by the "LV sensitive detector name" option
    191    eg  "--lvsdname Cathode,cathode,Sensor,SD" 
    192 
    193 The second approach is useful as a workaround when operating 
    194 with a GDML loaded geometry, as GDML does not yet(?) persist 
    195 the SD LV association.
    196 
    197 Names of sensitive LV are inserted into a set datastructure in GGeo. 
    198 
    199 **/
    200 
    201 void X4PhysicalVolume::convertSensors_r(const G4VPhysicalVolume* const pv, int depth)
    202 {
    203     const G4LogicalVolume* const lv = pv->GetLogicalVolume();
    204     const char* lvname = lv->GetName().c_str();
    205     G4VSensitiveDetector* sd = lv->GetSensitiveDetector() ;
    206 
    207     bool is_lvsdname = m_lvsdname && BStr::Contains(lvname, m_lvsdname, ',' ) ;
    208     bool is_sd = sd != NULL ;
    209 
    210     const std::string sdn = sd ? sd->GetName() : "SD?" ;   // perhaps GetFullPathName() 
    211 
    212     if( is_lvsdname || is_sd )
    213     {
    214         std::string name = BFile::Name(lvname);
    215         std::string nameref = SGDML::GenerateName( name.c_str() , lv , true );
    216         LOG(info)
    217             << " is_lvsdname " << is_lvsdname
    218             << " is_sd " << is_sd
    219             << " name " << name
    220             << " nameref " << nameref
    221             ;
    222 
    223         m_ggeo->addLVSD(nameref.c_str(), sdn.c_str()) ;
    224     }
    225 
    226     for (int i=0 ; i < lv->GetNoDaughters() ;i++ )
    227     {
    228         const G4VPhysicalVolume* const child_pv = lv->GetDaughter(i);
    229         convertSensors_r(child_pv, depth+1 );
    230     }
    231 }




This Looks Like a Smoky inconsistency on num_lvsd 
-----------------------------------------------------

ckm--::

    // translating live G4 into GGeo
    2019-05-31 20:27:21.541 INFO  [124216] [X4PhysicalVolume::convertSurfaces@284]  num_lbs 0 num_sks 0
    2019-05-31 20:27:21.541 INFO  [124216] [X4PhysicalVolume::convertSensors_r@221]  is_lvsdname 0 is_sd 1 name Det nameref Det0x1690260
    2019-05-31 20:27:21.541 INFO  [124216] [X4PhysicalVolume::convertSensors@172]  m_lvsdname (null) num_lvsd 1 num_clv 1 num_bds 0 num_sks0 0 num_sks1 1
    2019-05-31 20:27:21.541 INFO  [124216] [X4PhysicalVolume::convertSolids@436] [

ckm-ggeotest::

    // loading GGeo and dumping 
    2019-05-31 20:29:24.768 INFO  [127600] [test_GGeo_sd@161]  nlvsd 0









:
