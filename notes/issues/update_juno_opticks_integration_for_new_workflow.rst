update_juno_opticks_integration_for_new_workflow
==================================================

* previous : :doc:`ellipsoid_transform_compare_two_geometries`
* in parallel with this : :doc:`joined_up_thinking_geometry_translation`


HMM : Integration requires lots of small changes : how to proceed ?
----------------------------------------------------------------------

* switch off WITH_G4OPTICKS 
* switch on WITH_G4CXOPTICKS : bring over the blocks of code one at a time

Notice how this means can switch the entire integration with a single flip. 

GUIDING PRINCIPALS FOR INTEGRATION 

* IF IT CAN BE DONE WITHIN OPTICKS CODE : DO IT THERE (AS DEVELOPMENT IS SO MUCH EASIER THERE)
* MINIMIZE INTEGRATION CODE IN THE FRAMEWORK  : BECAUSE ITS SO PAINFUL TO DEVELOP WITHIN THE DETECTOR FRAMEWORK 
* AVOID BEING CHATTY IN INTERFACE


WITH_G4OPTICKS -> WITH_G4CXOPTICKS
---------------------------------------

As need to review the entire integration, change the preprocessor macro.

cmake/Modules/FindOpticks.cmake::

     48 #find_package(G4OK CONFIG QUIET)
     49 find_package(G4CX CONFIG QUIET)
     50 
     51 if(G4CX_FOUND)
     52     #add_compile_definitions(WITH_G4OPTICKS)
     53     add_compile_definitions(WITH_G4CXOPTICKS)
     54 


Issue 1 : Lack of Opticks : FIXED BY MOVING Opticks::Configure within G4CXOpticks::setGeometry
-------------------------------------------------------------------------------------------------

::

    #0  0x00007fffd297ec4a in Opticks::getIdPath (this=0x0) at /data/blyth/junotop/opticks/optickscore/Opticks.cc:4644
    #1  0x00007fffd34cf346 in GGeo::init (this=0x9330a40) at /data/blyth/junotop/opticks/ggeo/GGeo.cc:361
    #2  0x00007fffd34ced3f in GGeo::GGeo (this=0x9330a40, ok=0x0, live=true) at /data/blyth/junotop/opticks/ggeo/GGeo.cc:188
    #3  0x00007fffd3e1e711 in X4Geo::Translate (top=0x5725de0) at /data/blyth/junotop/opticks/extg4/X4Geo.cc:25
    #4  0x00007fffd45c1352 in G4CXOpticks::setGeometry (this=0x6dde0f0, world=0x5725de0) at /data/blyth/junotop/opticks/g4cx/G4CXOpticks.cc:175
    #5  0x00007fffd45c05cf in G4CXOpticks::SetGeometry (world=0x5725de0) at /data/blyth/junotop/opticks/g4cx/G4CXOpticks.cc:49
    #6  0x00007fffcfae3f35 in LSExpDetectorConstruction_Opticks::Setup (world=0x5725de0, opticksMode=3)

::

     40 int main(int argc, char** argv)
     41 {
     42     OPTICKS_LOG(argc, argv);
     43     //Opticks::Configure(argc, argv, "--gparts_transform_offset --allownokey" );
     44 
     45     SEventConfig::SetRGModeSimulate();
     46     SEventConfig::SetStandardFullDebug(); // controls which and dimensions of SEvt arrays 
     47 
     48     G4CXOpticks gx ;
     49     gx.setGeometry();



Issue 2 : Lack of idpath prevents GGeo::save : Try living without persisted GGeo
-----------------------------------------------------------------------------------

::

    2022-08-05 18:53:36.861 INFO  [137375] [GInstancer::dumpRepeatCandidates@464]  num_repcan 9 dmax 20
     pdig 159961bde1896fe286c02b4c3f05c8c9 ndig  25600 nprog      4 placements  25600 n PMT_3inch_log_phys
     pdig b82765dbe93381d08867b5bc550ceed3 ndig  12615 nprog      6 placements  12615 n pLPMT_NNVT_MCPPMT
     pdig 838cd73cc9dd9d9add66efd658630c12 ndig   4997 nprog      6 placements   4997 n pLPMT_Hamamatsu_R12860
     pdig 29c21c0b8afac0824902c82e6fbe3146 ndig   2400 nprog      5 placements   2400 n mask_PMT_20inch_vetolMaskVirtual_phys
     pdig ed3d2c21991e3bef5e069713af9fa6ca ndig    590 nprog      0 placements    590 n lSteel_phys
     pdig ac627ab1ccbdb62ec96e702f07f6425b ndig    590 nprog      0 placements    590 n lFasteners_phys
     pdig f899139df5e1059396431415e770c6dd ndig    590 nprog      0 placements    590 n lUpper_phys
     pdig 38b3eff8baf56627478ec76a704e9b52 ndig    590 nprog      0 placements    590 n lAddition_phys
     pdig 4c29bcd2a52a397de5036b415af92efe ndig    504 nprog    129 placements    504 n pPanel_0_f_
    2022-08-05 18:53:55.585 ERROR [137375] [GGeo::save@719] cannot save as no idpath set

    #1  0x00007fffd34d1ac9 in GGeo::save (this=0x938c1d0) at /data/blyth/junotop/opticks/ggeo/GGeo.cc:720
    #2  0x00007fffd34d0bbc in GGeo::postDirectTranslation (this=0x938c1d0) at /data/blyth/junotop/opticks/ggeo/GGeo.cc:607
    #3  0x00007fffd3e1e73e in X4Geo::Translate (top=0x5752710) at /data/blyth/junotop/opticks/extg4/X4Geo.cc:29
    #4  0x00007fffd45c13be in G4CXOpticks::setGeometry (this=0x6e0a910, world=0x5752710) at /data/blyth/junotop/opticks/g4cx/G4CXOpticks.cc:187
    #5  0x00007fffd45c062f in G4CXOpticks::SetGeometry (world=0x5752710) at /data/blyth/junotop/opticks/g4cx/G4CXOpticks.cc:56
    #6  0x00007fffcfae3f35 in LSExpDetectorConstruction_Opticks::Setup (world=0x5752710, opticksMode=3)

    2022-08-05 18:53:56.360 ERROR [137375] [GGeo::convertSim_Prop@2434]  SSim cannot add ri_prop as no idpath $IDPath/GScintillatorLib/LS_ori/RINDEX.npy
    Missing separate debuginfo for /lib64/libcuda.so.1
    Try: yum --enablerepo='*debug*' install /usr/lib/debug/.build-id/3e/1e7dd516361182d263c7713bd47eaa498bf0cd.debug
    [New Thread 0x7fffa63d0700 (LWP 137456)]
    [New Thread 0x7fffa5bcf700 (LWP 137457)]
    [New Thread 0x7fffa53ce700 (LWP 137458)]
    2022-08-05 18:53:58.667 ERROR [137375] [QSim::UploadComponents@116]   propcom null, SSim::PROPCOM propcom.npy
    2022-08-05 18:54:06.785 INFO  [137375] [LSExpDetectorConstruction_Opticks::Setup@31] ] WITH_G4CXOPTICKS 
    /data/blyth/junotop/offline/Simulation/DetSimV2/DetSimOptions/src/LSExpDetectorConstruction.cc:361 completed construction of physiWorld  m_opticksMode 3
    /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/share/Geant4-10.4.2/data/G4NDL4.5


Issue 3 : DsPhysConsOptical : needs code to avoid assert : FIXED
-------------------------------------------------------------------

::

    #3  0x00007ffff6967252 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007fffcfdae30c in DsPhysConsOptical::ConstructProcess (this=0xb603c0)
        at /data/blyth/junotop/offline/Simulation/DetSimV2/PhysiSim/src/DsPhysConsOptical.cc:162
    #5  0x00007fffcfae7048 in LSExpPhysicsList::ConstructProcess (this=0x556dbe0)
        at /data/blyth/junotop/offline/Simulation/DetSimV2/DetSimOptions/src/LSExpPhysicsList.cc:262
    #6  0x00007fffdf9f0185 in G4RunManagerKernel::InitializePhysics() () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4run.so
    #7  0x00007fffdf9dfb73 in G4RunManager::Initialize() () from /data/blyth/junotop/ExternalLibs/Geant4/10.


::

    jcv DsPhysConsOptical

    147 #ifdef WITH_G4CXOPTICKS
    148             LocalG4Cerenkov1042* cerenkov = new LocalG4Cerenkov1042(m_opticksMode) ;
    149             cerenkov->SetMaxNumPhotonsPerStep(m_cerenMaxPhotonPerStep);
    150             cerenkov->SetTrackSecondariesFirst(m_doTrackSecondariesFirst);
    151             cerenkov_ = cerenkov ;
    152 #elif WITH_G4OPTICKS
    153             LocalG4Cerenkov1042* cerenkov = new LocalG4Cerenkov1042(m_opticksMode) ;
    154             cerenkov->SetMaxNumPhotonsPerStep(m_cerenMaxPhotonPerStep);
    155             cerenkov->SetTrackSecondariesFirst(m_doTrackSecondariesFirst);
    156             cerenkov_ = cerenkov ;
    157 #else
    158             G4cout
    159                << __FILE__ << ":" << __LINE__
    160                << " DsPhysConsOptical::ConstructProcess "
    161                << " FATAL "
    162                << " non-zero opticksMode requires compilation -DWITH_G4OPTICKS or -DWITH_G4CXOPTICKS "
    163                << " m_useCerenkov " << m_useCerenkov
    164                << " m_opticksMode " << m_opticksMode
    165                << G4endl
    166                ;
    167             assert(0) ;


Issue 4 : another assert : from lack of merger_opticks : Added to PMTSDMgr
-----------------------------------------------------------------------------

::

    jcv PMTSDMgr


::

    epsilon:offline blyth$ jgr setMergerOpticks
    ./Simulation/DetSimV2/PMTSim/include/junoSD_PMT_v2.hh:        void setMergerOpticks(PMTHitMerger* phm) { m_pmthitmerger_opticks=phm; }
    ./Simulation/DetSimV2/PMTSim/src/PMTSDMgr.cc:        sd->setMergerOpticks(pmthitmerger_opticks);
    epsilon:offline blyth$ 

::

    170	    {
    171	        hitCollection_opticks = new junoHit_PMT_Collection(SensitiveDetectorName,collectionName[2]);
    172	        HCID = -1;
    173	        if(HCID<0) HCID = G4SDManager::GetSDMpointer()->GetCollectionID(hitCollection_opticks);
    174	        HCE->AddHitsCollection( HCID, hitCollection_opticks );
    175	        assert(m_pmthitmerger_opticks); 
    176	        if (m_hit_type == 1) {
    177	            m_pmthitmerger_opticks->init(hitCollection_opticks);
    178	        } else {
    179	            G4cout << "FATAL : unknown hit type [" << m_hit_type << "]" << G4endl;
    (gdb) 


    (gdb) bt
    #0  0x00007ffff696e387 in raise () from /lib64/libc.so.6
    #1  0x00007ffff696fa78 in abort () from /lib64/libc.so.6
    #2  0x00007ffff69671a6 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007ffff6967252 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007fffd3b01d17 in junoSD_PMT_v2::Initialize (this=0x5940600, HCE=0x2b8bb00)
        at /data/blyth/junotop/offline/Simulation/DetSimV2/PMTSim/src/junoSD_PMT_v2.cc:175
    #5  0x00007fffdd63bc25 in G4SDStructure::Initialize(G4HCofThisEvent*) [clone .localalias.79] ()
       from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4digits_hits.so
    #6  0x00007fffdd639b5d in G4SDManager::PrepareNewEvent() () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4digits_hits.so
    #7  0x00007fffdf7460a6 in G4EventManager::DoProcessing(G4Event*) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4event.so
    #8  0x00007fffd04a04a1 in G4SvcRunManager::SimulateEvent (this=0x910900, i_event=0)




Overview of the Integration WITH_G4OPTICKS
---------------------------------------------------------

::

    epsilon:~ blyth$ jgl WITH_G4OPTICKS

    ./Simulation/GenTools/GenTools/GtOpticksTool.h
    ./Simulation/GenTools/src/GtOpticksTool.cc

    ## Does input photons, using NPY.hpp NPho.hpp glm::vec4 getPositionTime 
    ## Opticks now has its own way of doing input photons. 

    ## DONE: added sphoton::Get to load em from NP arrays 
    ## DONE: U4Hit.h copied from G4OpticksHit.hh

    ## HMM: old one had G4OpticksRecorder : but now think 
    ##      that B-side running can be done Opticks side only 
    ##

    ./Simulation/DetSimV2/PhysiSim/include/LocalG4Cerenkov1042.hh
    ./Simulation/DetSimV2/PhysiSim/src/LocalG4Cerenkov1042.cc

    ./Simulation/DetSimV2/PhysiSim/include/DsG4Scintillation.h
    ./Simulation/DetSimV2/PhysiSim/src/DsG4Scintillation.cc

    ./Simulation/DetSimV2/PhysiSim/src/DsPhysConsOptical.cc

    ./Simulation/DetSimV2/PMTSim/include/junoSD_PMT_v2.hh
    ./Simulation/DetSimV2/PMTSim/src/junoSD_PMT_v2.cc

    ./Simulation/DetSimV2/PMTSim/include/junoSD_PMT_v2_Opticks.hh
    ./Simulation/DetSimV2/PMTSim/src/junoSD_PMT_v2_Opticks.cc

    ## TODO: G4Opticks::getHit needs updating for new workflow  
        

    ./Simulation/DetSimV2/PMTSim/include/PMTEfficiencyCheck.hh
    ./Simulation/DetSimV2/PMTSim/src/PMTEfficiencyCheck.cc

    ./Simulation/DetSimV2/PMTSim/src/PMTSDMgr.cc

    ./Simulation/DetSimV2/DetSimMTUtil/src/DetFactorySvc.cc

    ./Simulation/DetSimV2/DetSimOptions/src/DetSim0Svc.cc

    ./Simulation/DetSimV2/DetSimOptions/src/LSExpDetectorConstruction_Opticks.cc

    ## passing over the geometry, G4Opticks -> G4CXOpticks

    ./Simulation/DetSimV2/AnalysisCode/include/G4OpticksAnaMgr.hh
    ./Simulation/DetSimV2/AnalysisCode/src/G4OpticksAnaMgr.cc

    ## HMM : this is using G4OpticksRecorder, could be updated for U4Recorder 
    ## but Opticks alone can do this a bit doubtful of the need

    ./Examples/Tutorial/python/Tutorial/JUNODetSimModule.py



Passing over the geometry in new workflow
---------------------------------------------

::

   jcv  LSExpDetectorConstruction_Opticks
   jcv  LSExpDetectorConstruction_Opticks_OLD


Old way used a chatty interface of communicating sensor data::

    107     const std::vector<G4PVPlacement*>& sensor_placements = g4ok->getSensorPlacements() ;
    108     unsigned num_sensor = sensor_placements.size();
    109 
    110     // 2. use the placements to pass sensor data : efficiencies, categories, identifiers  
    111 
    112     const junoSD_PMT_v2* sd = dynamic_cast<const junoSD_PMT_v2*>(sd_) ;
    113     assert(sd) ;
    114 
    115     LOG(info) << "[ setSensorData num_sensor " << num_sensor ;
    116     for(unsigned i=0 ; i < num_sensor ; i++)
    117     {
    118         const G4PVPlacement* pv = sensor_placements[i] ; // i is 0-based unlike sensor_index
    119         unsigned sensor_index = 1 + i ; // 1-based 
    120         assert(pv);
    121         G4int copyNo = pv->GetCopyNo();
    122         int pmtid = copyNo ;
    123         int pmtcat = 0 ; // sd->getPMTCategory(pmtid); 
    124         float efficiency_1 = sd->getQuantumEfficiency(pmtid);
    125         float efficiency_2 = sd->getEfficiencyScale() ;
    126 
    127         g4ok->setSensorData( sensor_index, efficiency_1, efficiency_2, pmtcat, pmtid );
    128     }

Had idea to avoid the chat...

* :doc:`instanceIdentity-into-new-workflow`


Requires some object of the detector framework to inherit from 
the U4InstanceIdentifier protocol base and implement the method::

     71 class G4PVPlacement ;
     72 
     73 struct U4InstanceIdentifier
     74 {
     75     virtual unsigned getInstanceId(const G4PVPlacement* pv) const = 0 ;
     76 };


This can allow Opticks to provide detector specific identifiers on itersect.
BUT: it does not communicate the Opticks ordering of the sensors which 
is needed to communicate efficiencies.

Can add::

         virtual float getEfficiency(const G4PVPlacement* pv) const = 0 

Actually can add methods for that info. Then the Opticks ordering does
not matter for users, to first order.  












Hit Handling in new workflow
-------------------------------


u4/tests/U4HitTest.cc::

     14     SEvt* sev = SEvt::Load() ;
     15     const char* cfbase = sev->getSearchCFBase() ; // search up dir tree starting from loaddir for dir with CSGFoundry/solid.npy
     16     const CSGFoundry* fd = CSGFoundry::Load(cfbase);
     17     sev->setGeo(fd);
     18 
     19     std::cout << sev->descFull() ;
     20 
     21     unsigned num_hit = sev->getNumHit();
     22     if(num_hit == 0) return 0 ;
     23 
     24     unsigned idx = 0 ;
     25     sphoton global, local  ;
     26     sev->getHit(global, idx);
     27     sev->getLocalHit( local,  idx);
     28 
     29     U4Hit hit ;
     30     U4HitConvert::FromPhoton(hit,global,local);
     31 
     32     std::cout << " global " << global.desc() << std::endl ;
     33     std::cout << " local " << local.desc() << std::endl ;
     34     std::cout << " hit " << hit.desc() << std::endl ;


::

    1579 /**
    1580 SEvt::getLocalPhoton SEvt::getLocalHit
    1581 -----------------------------------------
    1582 
    1583 sphoton::iindex instance index used to get instance frame
    1584 from (SGeo*)cf which is used to transform the photon  
    1585 
    1586 **/
    1587 
    1588 void SEvt::getLocalPhoton(sphoton& lp, unsigned idx) const
    1589 {
    1590     getPhoton(lp, idx);
    1591     applyLocalTransform_w2m(lp);
    1592 }
    1593 void SEvt::getLocalHit(sphoton& lp, unsigned idx) const
    1594 {
    1595     getHit(lp, idx);
    1596     applyLocalTransform_w2m(lp);
    1597 }


The improved precision will come in with the sframe::

    1598 void SEvt::applyLocalTransform_w2m( sphoton& lp) const
    1599 {
    1600     sframe fr ;
    1601     getPhotonFrame(fr, lp);
    1602     fr.transform_w2m(lp);
    1603 }
    1604 void SEvt::getPhotonFrame( sframe& fr, const sphoton& p ) const
    1605 {
    1606     assert(cf);
    1607     cf->getFrame(fr, p.iindex);
    1608     fr.prepare();
    1609 }

::

    2842 int CSGFoundry::getFrame(sframe& fr, int inst_idx) const
    2843 {
    2844     return target->getFrame( fr, inst_idx );
    2845 }


    122 /**
    123 CSGTarget::getFrame
    124 ---------------------
    125 
    126 Note that there are typically multiple CSGPrim within the compound CSGSolid
    127 and that the inst_idx corresponds to the entire compound CSGSolid (aka GMergedMesh).
    128 Hence the ce included with the frame is the one from the full compound CSGSolid. 
    129 
    130 * TODO: avoid the Tran::Invert by keeping paired double precision transforms throughout  
    131 
    132 * DONE: new minimal stree.h geo translation collects paired m2w and w2m transforms
    133   and uses those to give both inst and iinst in double precision 
    134 
    135 * TODO: use that to improve frame precision and avoid the Invert
    136 
    137   * hmm : can I use somehow use stree.h transforms to CSG_GGeo to give access to 
    138     the improved transforms before fully switching to new translation ?
    139 
    140   * would have to add stree persisting to GGeo to so this, 
    141     that just adds complication for a very shortlived benefit 
    142 
    143 **/
    144 
    145 int CSGTarget::getFrame(sframe& fr, int inst_idx ) const
    146 {
    147     const qat4* _t = foundry->getInst(inst_idx);
    148     
    149     unsigned ins_idx, gas_idx, ias_idx ;
    150     _t->getIdentity(ins_idx, gas_idx, ias_idx )  ;
    151     
    152     assert( int(ins_idx) == inst_idx );
    153     fr.set_inst(inst_idx);  



How to simplify integration ?
-----------------------------

* Do not return G4(CX)Opticks instance, so can just change impl not header 
* Keep it totally minimal : ie do everything on Opticks side and the 
  absolute minimum on the Detector Framework side


DONE : moved SEvt into G4CXOpticks, added INSTANCE  

TODO : mimic some of the G4Opticks API to simplify the update 



G4Opticks::getHit : getting local photons
--------------------------------------------

Old way, using GPho hits wrapper::
    
    1322 void G4Opticks::getHit(unsigned i, G4OpticksHit* hit, G4OpticksHitExtra* hit_extra ) const
    1323 {
    1324     assert( i < m_num_hits && hit );
    1325 
    1326     glm::vec4 post = m_hits_wrapper->getPositionTime(i);
    1327     glm::vec4 dirw = m_hits_wrapper->getDirectionWeight(i);
    1328     glm::vec4 polw = m_hits_wrapper->getPolarizationWavelength(i);
    1329 
    1330     // local getters rely on GPho::getLastIntersectNodeIndex/OpticksPhotonFlags::NodeIndex to get the frame
    1331     glm::vec4 local_post = m_hits_wrapper->getLocalPositionTime(i);
    1332     glm::vec4 local_dirw = m_hits_wrapper->getLocalDirectionWeight(i);
    1333     glm::vec4 local_polw = m_hits_wrapper->getLocalPolarizationWavelength(i);
    1334 
    1337     hit->global_position.set(double(post.x), double(post.y), double(post.z));
    1338     hit->time = double(post.w) ;
    1339     hit->global_direction.set(double(dirw.x), double(dirw.y), double(dirw.z));
    1340     hit->weight = double(dirw.w) ;
    1341     hit->global_polarization.set(double(polw.x), double(polw.y), double(polw.z));
    1342     hit->wavelength = double(polw.w);
    1343 
    1344     hit->local_position.set(double(local_post.x), double(local_post.y), double(local_post.z));
    1345     hit->local_direction.set(double(local_dirw.x), double(local_dirw.y), double(local_dirw.z));
    1346     hit->local_polarization.set(double(local_polw.x), double(local_polw.y), double(local_polw.z));
    1347 
    1348     hit->boundary      = pflag.boundary ;
    1349     hit->sensorIndex   = pflag.sensorIndex ;
    1350     hit->nodeIndex     = pflag.nodeIndex ;
    1351     hit->photonIndex   = pflag.photonIndex ;
    1352     hit->flag_mask     = pflag.flagMask ;


This feels like a lot of shuffling...

GPho::get* 
    shuffles values from NPY<float> into glm::vec4 

G4Opticks::getHit
    shuffles values from glm::vec4 into G4OpticksHit(aka U4Hit)/G4ThreeVector etc.. 

junoSD_PMT_v2_Opticks::convertHit
    shuffles from G4OpticksHit(aka U4Hit) into junoHit_PMT 


Is the G4OpticksHit/U4Hit intermediary actually needed ? 

* could go from sphoton -> sphotond -> junoHit_PMT. 


GPho used nodeIndex to access the transform. 

* using nodeIndex is extravagant : no need to use a 0-300k number ( > 0xffff ) 
  when there are only 50k instance transforms (fits in 0xffff 65535 )

* also nodeIdx potentially problematic when the are structural transforms 
  within the compound solid : what you want is to use one instance transform 
  for all coords relevant to an instance not having to worry about shifts between 
  different elements of the compound
  
* how does python find which transform to use ? thats using the sframe thats kinda an input, 
  but that matches with the inst transforms : but only in float precision 


gxs.sh Live dumping gives expected close to origin local coords
--------------------------------------------------------------------

DONE: get a grabed and loaded SEvt on laptop to reproduce the below, see CSG/tests/CSGFoundry_SGeo_SEvt_Test.sh 


::

    2022-07-27 03:48:54.866 INFO  [344673] [SEvt::saveLabels@1359]  a0 -
    2022-07-27 03:48:54.866 INFO  [344673] [SEvt::saveLabels@1363]  a -
    2022-07-27 03:48:54.866 INFO  [344673] [SEvt::saveLabels@1367]  g -
    2022-07-27 03:48:54.866 INFO  [344673] [G4CXOpticks::save@222] SEvt::descPhoton num_fold_photon 1000 max_print 10 num_print 10
     pos (-11951.935,9430.896,11779.457)  t     3.867  mom (-0.624, 0.492, 0.607)  iindex 39216  pol (-0.619,-0.785, 0.000)  wl  440.000   bn 32 fl 40 id 203462960 or -1 ix 0 fm 1840 ab SD
     pos (-11926.811,9411.070,11838.502)  t     3.917  mom (-0.632, 0.498, 0.593)  iindex 39216  pol (-0.619,-0.785,-0.000)  wl  440.000   bn 32 fl 40 id 203462960 or -1 ix 0 fm 1840 ab SD
     pos (-11942.835,9423.715,11797.671)  t     3.876  mom (-0.626, 0.494, 0.603)  iindex 39216  pol (-0.619,-0.785,-0.000)  wl  440.000   bn 32 fl 40 id 203462960 or -1 ix 0 fm 1840 ab SD
     pos (-11952.632,9431.445,11778.164)  t     3.867  mom (-0.624, 0.492, 0.608)  iindex 39216  pol (-0.619,-0.785,-0.000)  wl  440.000   bn 32 fl 40 id 203462960 or -1 ix 0 fm 1840 ab SD
     pos (-11976.379,9450.185,11740.475)  t     3.871  mom (-0.618, 0.487, 0.617)  iindex 39216  pol (-0.619,-0.785,-0.000)  wl  440.000   bn 32 fl 40 id 203462960 or -1 ix 0 fm 1840 ab SD
     pos (-11830.135,9334.786,11708.812)  t     3.094  mom (-0.621, 0.490, 0.611)  iindex 39216  pol (-0.619,-0.785,-0.000)  wl  440.000   bn 28 fl 8 id 203200816 or -1 ix 0 fm 1008 ab AB
     pos (-11973.427,9447.856,11744.587)  t     3.869  mom (-0.618, 0.488, 0.616)  iindex 39216  pol (-0.619,-0.785,-0.000)  wl  440.000   bn 32 fl 40 id 203462960 or -1 ix 0 fm 1840 ab SD
     pos (-11911.435,9398.938,11912.104)  t     4.054  mom (-0.641, 0.506, 0.572)  iindex 39216  pol (-0.619,-0.785,-0.000)  wl  440.000   bn 32 fl 40 id 203462960 or -1 ix 0 fm 1840 ab SD
     pos (-11910.158,9397.930,11946.814)  t     4.146  mom (-0.645, 0.509, 0.559)  iindex 39216  pol (-0.619,-0.785,-0.000)  wl  440.000   bn 32 fl 40 id 203462960 or -1 ix 0 fm 1840 ab SD
     pos (-11985.225,9457.163,11728.927)  t     3.879  mom (-0.616, 0.486, 0.620)  iindex 39216  pol (-0.619,-0.785, 0.000)  wl  440.000   bn 32 fl 40 id 203462960 or -1 ix 0 fm 1840 ab SD

    2022-07-27 03:48:54.867 INFO  [344673] [G4CXOpticks::save@223] SEvt::descLocalPhoton num_fold_photon 1000 max_print 10 num_print 10
     pos (19.456,-0.000,184.434)  t     3.867  mom (-0.005, 0.000,-1.000)  iindex 39216  pol (-0.000, 1.000,-0.000)  wl  440.000   bn 32 fl 40 id 203462960 or -1 ix 0 fm 1840 ab SD
     pos (85.750, 0.000,173.682)  t     3.917  mom (-0.023,-0.000,-1.000)  iindex 39216  pol (-0.000, 1.000,-0.000)  wl  440.000   bn 32 fl 40 id 203462960 or -1 ix 0 fm 1840 ab SD
     pos (40.957, 0.000,182.478)  t     3.876  mom (-0.010, 0.000,-1.000)  iindex 39216  pol (-0.000, 1.000, 0.000)  wl  440.000   bn 32 fl 40 id 203462960 or -1 ix 0 fm 1840 ab SD
     pos (17.890, 0.000,184.522)  t     3.867  mom (-0.005,-0.000,-1.000)  iindex 39216  pol (-0.000, 1.000,-0.000)  wl  440.000   bn 32 fl 40 id 203462960 or -1 ix 0 fm 1840 ab SD
     pos (-30.429,-0.001,183.611)  t     3.871  mom ( 0.008, 0.000,-1.000)  iindex 39216  pol ( 0.000, 1.000, 0.000)  wl  440.000   bn 32 fl 40 id 203462960 or -1 ix 0 fm 1840 ab SD
     pos (58.357, 0.001,350.415)  t     3.094  mom (-0.000, 0.000,-1.000)  iindex 39216  pol (-0.000, 1.000,-0.000)  wl  440.000   bn 28 fl 8 id 203200816 or -1 ix 0 fm 1008 ab AB
     pos (-24.877,-0.002,184.074)  t     3.869  mom ( 0.006, 0.000,-1.000)  iindex 39216  pol (-0.000, 1.000, 0.000)  wl  440.000   bn 32 fl 40 id 203462960 or -1 ix 0 fm 1840 ab SD
     pos (155.978,-0.000,144.204)  t     4.054  mom (-0.047, 0.000,-0.999)  iindex 39216  pol (-0.000, 1.000,-0.000)  wl  440.000   bn 32 fl 40 id 203462960 or -1 ix 0 fm 1840 ab SD
     pos (184.446, 0.001,124.279)  t     4.146  mom (-0.060,-0.000,-0.998)  iindex 39216  pol (-0.000, 1.000,-0.000)  wl  440.000   bn 32 fl 40 id 203462960 or -1 ix 0 fm 1840 ab SD
     pos (-46.456, 0.000,181.750)  t     3.879  mom ( 0.012,-0.000,-1.000)  iindex 39216  pol ( 0.000, 1.000, 0.000)  wl  440.000   bn 32 fl 40 id 203462960 or -1 ix 0 fm 1840 ab SD

    2022-07-27 03:48:54.867 INFO  [344673] [G4CXOpticks::save@224] SEvt::descFramePhoton num_fold_photon 1000 max_print 10 num_print 10
     pos (19.456,-0.000,184.434)  t     3.867  mom (-0.005, 0.000,-1.000)  iindex 39216  pol (-0.000, 1.000,-0.000)  wl  440.000   bn 32 fl 40 id 203462960 or -1 ix 0 fm 1840 ab SD
     pos (85.750, 0.000,173.682)  t     3.917  mom (-0.023,-0.000,-1.000)  iindex 39216  pol (-0.000, 1.000,-0.000)  wl  440.000   bn 32 fl 40 id 203462960 or -1 ix 0 fm 1840 ab SD
     pos (40.957, 0.000,182.478)  t     3.876  mom (-0.010, 0.000,-1.000)  iindex 39216  pol (-0.000, 1.000, 0.000)  wl  440.000   bn 32 fl 40 id 203462960 or -1 ix 0 fm 1840 ab SD
     pos (17.890, 0.000,184.522)  t     3.867  mom (-0.005,-0.000,-1.000)  iindex 39216  pol (-0.000, 1.000,-0.000)  wl  440.000   bn 32 fl 40 id 203462960 or -1 ix 0 fm 1840 ab SD
     pos (-30.429,-0.001,183.611)  t     3.871  mom ( 0.008, 0.000,-1.000)  iindex 39216  pol ( 0.000, 1.000, 0.000)  wl  440.000   bn 32 fl 40 id 203462960 or -1 ix 0 fm 1840 ab SD
     pos (58.357, 0.001,350.415)  t     3.094  mom (-0.000, 0.000,-1.000)  iindex 39216  pol (-0.000, 1.000,-0.000)  wl  440.000   bn 28 fl 8 id 203200816 or -1 ix 0 fm 1008 ab AB
     pos (-24.877,-0.002,184.074)  t     3.869  mom ( 0.006, 0.000,-1.000)  iindex 39216  pol (-0.000, 1.000, 0.000)  wl  440.000   bn 32 fl 40 id 203462960 or -1 ix 0 fm 1840 ab SD
     pos (155.978,-0.000,144.204)  t     4.054  mom (-0.047, 0.000,-0.999)  iindex 39216  pol (-0.000, 1.000,-0.000)  wl  440.000   bn 32 fl 40 id 203462960 or -1 ix 0 fm 1840 ab SD
     pos (184.446, 0.001,124.279)  t     4.146  mom (-0.060,-0.000,-0.998)  iindex 39216  pol (-0.000, 1.000,-0.000)  wl  440.000   bn 32 fl 40 id 203462960 or -1 ix 0 fm 1840 ab SD
     pos (-46.456, 0.000,181.750)  t     3.879  mom ( 0.012,-0.000,-1.000)  iindex 39216  pol ( 0.000, 1.000, 0.000)  wl  440.000   bn 32 fl 40 id 203462960 or -1 ix 0 fm 1840 ab SD

    N[blyth@localhost g4cx]$ 



After using SOpticksResource::SearchCFBase can load the appropriate CFBase and get match::


    ins_idx 39216 num_fold_photon 1000 num_fold_hit    946 num_print 100
    SEvt::descPhoton num_fold_photon 1000 max_print 10 num_print 10
     pos (-11951.935,9430.896,11779.457)  t     3.867  mom (-0.624, 0.492, 0.607)  iindex 39216  pol (-0.619,-0.785, 0.000)  wl  440.000   bn 32 fl 40 id 203462960 or -1 ix 0 fm 1840 ab SD
     pos (-11926.811,9411.070,11838.502)  t     3.917  mom (-0.632, 0.498, 0.593)  iindex 39216  pol (-0.619,-0.785,-0.000)  wl  440.000   bn 32 fl 40 id 203462960 or -1 ix 0 fm 1840 ab SD
     pos (-11942.835,9423.715,11797.671)  t     3.876  mom (-0.626, 0.494, 0.603)  iindex 39216  pol (-0.619,-0.785,-0.000)  wl  440.000   bn 32 fl 40 id 203462960 or -1 ix 0 fm 1840 ab SD
     pos (-11952.632,9431.445,11778.164)  t     3.867  mom (-0.624, 0.492, 0.608)  iindex 39216  pol (-0.619,-0.785,-0.000)  wl  440.000   bn 32 fl 40 id 203462960 or -1 ix 0 fm 1840 ab SD
     pos (-11976.379,9450.185,11740.475)  t     3.871  mom (-0.618, 0.487, 0.617)  iindex 39216  pol (-0.619,-0.785,-0.000)  wl  440.000   bn 32 fl 40 id 203462960 or -1 ix 0 fm 1840 ab SD
     pos (-11830.135,9334.786,11708.812)  t     3.094  mom (-0.621, 0.490, 0.611)  iindex 39216  pol (-0.619,-0.785,-0.000)  wl  440.000   bn 28 fl 8 id 203200816 or -1 ix 0 fm 1008 ab AB
     pos (-11973.427,9447.856,11744.587)  t     3.869  mom (-0.618, 0.488, 0.616)  iindex 39216  pol (-0.619,-0.785,-0.000)  wl  440.000   bn 32 fl 40 id 203462960 or -1 ix 0 fm 1840 ab SD
     pos (-11911.435,9398.938,11912.104)  t     4.054  mom (-0.641, 0.506, 0.572)  iindex 39216  pol (-0.619,-0.785,-0.000)  wl  440.000   bn 32 fl 40 id 203462960 or -1 ix 0 fm 1840 ab SD
     pos (-11910.158,9397.930,11946.814)  t     4.146  mom (-0.645, 0.509, 0.559)  iindex 39216  pol (-0.619,-0.785,-0.000)  wl  440.000   bn 32 fl 40 id 203462960 or -1 ix 0 fm 1840 ab SD
     pos (-11985.225,9457.163,11728.927)  t     3.879  mom (-0.616, 0.486, 0.620)  iindex 39216  pol (-0.619,-0.785, 0.000)  wl  440.000   bn 32 fl 40 id 203462960 or -1 ix 0 fm 1840 ab SD

    SEvt::descLocalPhoton num_fold_photon 1000 max_print 10 num_print 10
     pos (19.456,-0.000,184.434)  t     3.867  mom (-0.005, 0.000,-1.000)  iindex 39216  pol (-0.000, 1.000,-0.000)  wl  440.000   bn 32 fl 40 id 203462960 or -1 ix 0 fm 1840 ab SD
     pos (85.750, 0.000,173.682)  t     3.917  mom (-0.023,-0.000,-1.000)  iindex 39216  pol (-0.000, 1.000,-0.000)  wl  440.000   bn 32 fl 40 id 203462960 or -1 ix 0 fm 1840 ab SD
     pos (40.957, 0.000,182.478)  t     3.876  mom (-0.010, 0.000,-1.000)  iindex 39216  pol (-0.000, 1.000, 0.000)  wl  440.000   bn 32 fl 40 id 203462960 or -1 ix 0 fm 1840 ab SD
     pos (17.890, 0.000,184.522)  t     3.867  mom (-0.005,-0.000,-1.000)  iindex 39216  pol (-0.000, 1.000,-0.000)  wl  440.000   bn 32 fl 40 id 203462960 or -1 ix 0 fm 1840 ab SD
     pos (-30.429,-0.001,183.611)  t     3.871  mom ( 0.008, 0.000,-1.000)  iindex 39216  pol ( 0.000, 1.000, 0.000)  wl  440.000   bn 32 fl 40 id 203462960 or -1 ix 0 fm 1840 ab SD
     pos (58.357, 0.001,350.415)  t     3.094  mom (-0.000, 0.000,-1.000)  iindex 39216  pol (-0.000, 1.000,-0.000)  wl  440.000   bn 28 fl 8 id 203200816 or -1 ix 0 fm 1008 ab AB
     pos (-24.877,-0.002,184.074)  t     3.869  mom ( 0.006, 0.000,-1.000)  iindex 39216  pol (-0.000, 1.000, 0.000)  wl  440.000   bn 32 fl 40 id 203462960 or -1 ix 0 fm 1840 ab SD
     pos (155.978,-0.000,144.204)  t     4.054  mom (-0.047, 0.000,-0.999)  iindex 39216  pol (-0.000, 1.000,-0.000)  wl  440.000   bn 32 fl 40 id 203462960 or -1 ix 0 fm 1840 ab SD
     pos (184.446, 0.001,124.279)  t     4.146  mom (-0.060,-0.000,-0.998)  iindex 39216  pol (-0.000, 1.000,-0.000)  wl  440.000   bn 32 fl 40 id 203462960 or -1 ix 0 fm 1840 ab SD
     pos (-46.456, 0.000,181.750)  t     3.879  mom ( 0.012,-0.000,-1.000)  iindex 39216  pol ( 0.000, 1.000, 0.000)  wl  440.000   bn 32 fl 40 id 203462960 or -1 ix 0 fm 1840 ab SD

    SEvt::descFramePhoton num_fold_photon 1000 max_print 10 num_print 10
     pos (19.456,-0.000,184.434)  t     3.867  mom (-0.005, 0.000,-1.000)  iindex 39216  pol (-0.000, 1.000,-0.000)  wl  440.000   bn 32 fl 40 id 203462960 or -1 ix 0 fm 1840 ab SD
     pos (85.750, 0.000,173.682)  t     3.917  mom (-0.023,-0.000,-1.000)  iindex 39216  pol (-0.000, 1.000,-0.000)  wl  440.000   bn 32 fl 40 id 203462960 or -1 ix 0 fm 1840 ab SD
     pos (40.957, 0.000,182.478)  t     3.876  mom (-0.010, 0.000,-1.000)  iindex 39216  pol (-0.000, 1.000, 0.000)  wl  440.000   bn 32 fl 40 id 203462960 or -1 ix 0 fm 1840 ab SD
     pos (17.890, 0.000,184.522)  t     3.867  mom (-0.005,-0.000,-1.000)  iindex 39216  pol (-0.000, 1.000,-0.000)  wl  440.000   bn 32 fl 40 id 203462960 or -1 ix 0 fm 1840 ab SD
     pos (-30.429,-0.001,183.611)  t     3.871  mom ( 0.008, 0.000,-1.000)  iindex 39216  pol ( 0.000, 1.000, 0.000)  wl  440.000   bn 32 fl 40 id 203462960 or -1 ix 0 fm 1840 ab SD
     pos (58.357, 0.001,350.415)  t     3.094  mom (-0.000, 0.000,-1.000)  iindex 39216  pol (-0.000, 1.000,-0.000)  wl  440.000   bn 28 fl 8 id 203200816 or -1 ix 0 fm 1008 ab AB
     pos (-24.877,-0.002,184.074)  t     3.869  mom ( 0.006, 0.000,-1.000)  iindex 39216  pol (-0.000, 1.000, 0.000)  wl  440.000   bn 32 fl 40 id 203462960 or -1 ix 0 fm 1840 ab SD
     pos (155.978,-0.000,144.204)  t     4.054  mom (-0.047, 0.000,-0.999)  iindex 39216  pol (-0.000, 1.000,-0.000)  wl  440.000   bn 32 fl 40 id 203462960 or -1 ix 0 fm 1840 ab SD
     pos (184.446, 0.001,124.279)  t     4.146  mom (-0.060,-0.000,-0.998)  iindex 39216  pol (-0.000, 1.000,-0.000)  wl  440.000   bn 32 fl 40 id 203462960 or -1 ix 0 fm 1840 ab SD
     pos (-46.456, 0.000,181.750)  t     3.879  mom ( 0.012,-0.000,-1.000)  iindex 39216  pol ( 0.000, 1.000, 0.000)  wl  440.000   bn 32 fl 40 id 203462960 or -1 ix 0 fm 1840 ab SD






New flatter way of accessing local photons, where to consult CF to get the transform ?
-----------------------------------------------------------------------------------------

New way, treats pos,mom,pol together with::

    sphoton::Get 
    p.iindex -> transform

    sphoton::transform -> local photons 
    sphoton::transform_float 
    sphoton::iindex in former weight slot (1,3)

Where to consult CF to get the transform ? 

Obviously not up in gx(or cx) as all that is needed is access to transforms
and SEvt NP/sphoton. 

* access to transforms seems like an approriate thing for SGeo protocol base 

  * CSGFoundry can follow SGeo protocol base, so SEvt can hold onto SGeo* cf, 
    thence SEvt can coordinate access to transforms after "void SEvt::setGeo(const SGeo* cf)" 
    has been called. Which can happen immediately after translation or loading of CF geometry 
    as SEvt should always be instanciated then.    

* so G4CXOpticks::getHit can use sphoton from SEvt::getLocalPhoton SEvt::getPhoton
  replacing GPho in a flatter way with no use of GGeo  


* notice that the python access to local positions eg ana/simtrace_positions.py uses
  frame.w2m that is obtained by Invert in CSGTarget::getFrame::

    103         lpos = np.dot( gpos, frame.w2m )   # local frame intersect positions



::

    In [4]: uii, uiic = np.unique( a.photon.view(np.uint32)[:,1,3], return_counts=True ) ; uii, uiic
    Out[4]: 
    (array([    0, 17819, 27699, 27864, 28212, 29412, 31871, 38549, 39124, 39216, 40935, 41613], dtype=uint32),
     array([  9,   1,   1,   1,   2,   1,   1,   1,   1, 980,   1,   1]))


    In [9]: cf.inst[uii]
    Out[9]: 
    array([[[     1.   ,      0.   ,      0.   ,      0.   ],
            [     0.   ,      1.   ,      0.   ,      0.   ],
            [     0.   ,      0.   ,      1.   ,      0.   ],
            [     0.   ,      0.   ,      0.   ,      1.   ]],

           [[     0.461,     -0.364,      0.809,      0.   ],
            [    -0.619,     -0.785,     -0.   ,      0.   ],
            [     0.635,     -0.501,     -0.587,      0.   ],
            [-12314.685,   9717.144,  11387.06 ,      1.   ]],

           [[     0.523,     -0.383,      0.762,      0.   ],
            [    -0.591,     -0.807,     -0.   ,      0.   ],
            [     0.615,     -0.45 ,     -0.648,      0.   ],
            [-11946.645,   8745.829,  12588.428,      1.   ]],

           [[     0.501,     -0.381,      0.777,      0.   ],
            [    -0.605,     -0.796,     -0.   ,      0.   ],
            [     0.619,     -0.47 ,     -0.63 ,      0.   ],
            [-12020.483,   9137.731,  12234.794,      1.   ]],


::

    In [15]: cf.inst[39216]
    Out[15]: 
    array([[     0.48 ,     -0.379,      0.792,      0.   ],
           [    -0.619,     -0.785,     -0.   ,      0.   ],
           [     0.621,     -0.49 ,     -0.611,      0.   ],
           [-12075.873,   9528.691,  11876.771,      1.   ]], dtype=float32)

    In [16]: t.sframe.m2w
    Out[16]: 
    array([[     0.48 ,     -0.379,      0.792,      0.   ],
           [    -0.619,     -0.785,     -0.   ,      0.   ],
           [     0.621,     -0.49 ,     -0.611,      0.   ],
           [-12075.873,   9528.691,  11876.771,      1.   ]], dtype=float32)

    In [17]: np.all( t.sframe.m2w  == cf.inst[39216] )
    Out[17]: False

    In [18]: np.where( t.sframe.m2w  != cf.inst[39216] )
    Out[18]: (array([0, 1, 2]), array([3, 3, 3]))

    In [19]: t.sframe.m2w - cf.inst[39216]
    Out[19]: 
    array([[ 0.,  0.,  0., -0.],
           [ 0.,  0.,  0., -0.],
           [ 0.,  0.,  0., -0.],
           [ 0.,  0.,  0.,  0.]], dtype=float32)

    In [20]: t.sframe.m2w[:,:3]
    Out[20]: 
    array([[     0.48 ,     -0.379,      0.792],
           [    -0.619,     -0.785,     -0.   ],
           [     0.621,     -0.49 ,     -0.611],
           [-12075.873,   9528.691,  11876.771]], dtype=float32)

    In [21]: np.all( t.sframe.m2w[:,:3] == cf.inst[39216,:,:3] )
    Out[21]: True




New Workflow Photon Flags : mostly handled via sphoton methods ?
---------------------------------------------------------------------------

* sensorIndex needs effort, regarding identity info collection 

::

    093     SPHOTON_METHOD unsigned idx() const {      return orient_idx & 0x7fffffffu  ;  }
     94     SPHOTON_METHOD float    orient() const {   return ( orient_idx & 0x80000000u ) ? -1.f : 1.f ; }
     95 
     96     SPHOTON_METHOD void set_orient(float orient){ orient_idx = ( orient_idx & 0x7fffffffu ) | (( orient < 0.f ? 0x1 : 0x0 ) << 31 ) ; } // clear orient     bit and then set it 
     97     SPHOTON_METHOD void set_idx( unsigned idx ){  orient_idx = ( orient_idx & 0x80000000u ) | ( 0x7fffffffu & idx ) ; }   // retain bit 31 asis 
     98 
     99     SPHOTON_METHOD unsigned flag() const {     return boundary_flag & 0xffffu ; } // flag___     = lambda p:p.view(np.uint32)[...,3,0] & 0xffff
    100     SPHOTON_METHOD unsigned boundary() const { return boundary_flag >> 16 ; }     // boundary___ = lambda p:p.view(np.uint32)[...,3,0] >> 16
    101 
        



Old Photon Flags G4Opticks::getHit : okc/OpticksPhotonFlags
-----------------------------------------------------------------

::

    1335     OpticksPhotonFlags pflag = m_hits_wrapper->getOpticksPhotonFlags(i);
    ....
    1348     hit->boundary      = pflag.boundary ;
    1349     hit->sensorIndex   = pflag.sensorIndex ;
    1350     hit->nodeIndex     = pflag.nodeIndex ;
    1351     hit->photonIndex   = pflag.photonIndex ;
    1352     hit->flag_mask     = pflag.flagMask ;
    1353 
    1354     hit->is_cerenkov       = (pflag.flagMask & CERENKOV) != 0 ;
    1355     hit->is_reemission     = (pflag.flagMask & BULK_REEMIT) != 0 ;
    1356 
    1357     // via m_sensorlib 
    1358     hit->sensor_identifier = getSensorIdentifier(pflag.sensorIndex);

::

    255 /**
    256 GPho::getOpticksPhotonFlags
    257 ---------------------------
    258 
    259 The float flags contain the bits of unsigned and signed integers with some bit packing.  
    260 These are decoded using OpticksPhotonFlags.
    261 
    262 **/
    263 
    264 OpticksPhotonFlags GPho::getOpticksPhotonFlags(unsigned i) const
    265 {
    266     glm::vec4 flgs = m_photons->getQuad_(i,3);
    267     OpticksPhotonFlags okfl(flgs);
    268     return okfl ;
    269 }






Old Integration : OPTICKS_LOG 
---------------------------------

::

    epsilon:offline blyth$ jgr OPTICKS_LOG
    ./Simulation/DetSimV2/DetSimMTUtil/src/DetFactorySvc.cc:#include "OPTICKS_LOG.hh"
    ./Simulation/DetSimV2/DetSimOptions/src/DetSim0Svc.cc:#include "OPTICKS_LOG.hh"

jcv DetSim0Svc::

    301 bool DetSim0Svc::initializeOpticks()
    302 {
    303     dumpOpticks("DetSim0Svc::initializeOpticks");
    304     assert( m_opticksMode > 0);
    305 
    306 #ifdef WITH_G4OPTICKS
    307     OPTICKS_ELOG("DetSim0Svc");
    308 #else
    309     LogError << " FATAL : non-zero opticksMode **NOT** WITH_G4OPTICKS " << std::endl ;
    310     assert(0);
    311 #endif
    312     return true ;
    313 }
    314 
    315 bool DetSim0Svc::finalizeOpticks()
    316 {
    317     dumpOpticks("DetSim0Svc::finalizeOpticks");
    318     assert( m_opticksMode > 0);
    319 
    320 #ifdef WITH_G4OPTICKS
    321     G4Opticks::Finalize();
    322 #else
    323     LogError << " FATAL : non-zero opticksMode **NOT** WITH_G4OPTICKS " << std::endl ;
    324     assert(0);
    325 #endif
    326     return true;
    327 }





Old Integration : setup : done at tail of LSExpDetectorConstruction::Construct
---------------------------------------------------------------------------------

jcv LSExpDetectorConstruction::


     199 G4VPhysicalVolume* LSExpDetectorConstruction::Construct()
     200 {
     ...
     359   m_g4opticks = LSExpDetectorConstruction_Opticks::Setup( physiWorld, m_sd, m_opticksMode );
     360 
     361   G4cout
     362       << __FILE__ << ":" << __LINE__ << " completed construction of physiWorld "
     363       << " m_opticksMode " << m_opticksMode
     364       << G4endl
     365       ;
     366 
     367   return physiWorld;
     368 }


jcv LSExpDetectorConstruction_Opticks::

    001 #pragma once
      2 
      3 class G4Opticks ;
      4 class G4VPhysicalVolume ;
      5 class G4VSensitiveDetector ;
      6 
      7 struct LSExpDetectorConstruction_Opticks
      8 {
      9     static G4Opticks* Setup(const G4VPhysicalVolume* world, const G4VSensitiveDetector* sd_, int opticksMode );
     10 };

    084 G4Opticks* LSExpDetectorConstruction_Opticks::Setup(const G4VPhysicalVolume* world, const G4VSensitiveDetector* sd_, int opticksMode )  // static
     85 {
     86     if( opticksMode == 0 ) return nullptr ;
     87     LOG(info) << "[ WITH_G4OPTICKS opticksMode " << opticksMode  ;
     88 
     89     assert(world); 
     90 
     91     // 1. pass geometry to Opticks, translate it to GPU and return sensor placements  
     92 
     93     G4Opticks* g4ok = new G4Opticks ;
     94     
     95     bool outer_volume = true ;
     96     bool profile = true ;
     97 
     98     const char* geospecific_default =   "--way --pvname pAcrylic --boundary Water///Acrylic --waymask 3 --gdmlkludge" ;  // (1): gives radius 17820
     99     const char* embedded_commandline_extra = SSys::getenvvar("LSXDC_GEOSPECIFIC", geospecific_default ) ;   
    100     LOG(info) << " embedded_commandline_extra " << embedded_commandline_extra ;
    101 
    102     g4ok->setPlacementOuterVolume(outer_volume); 
    103     g4ok->setProfile(profile); 
    104     g4ok->setEmbeddedCommandLineExtra(embedded_commandline_extra);
    105     g4ok->setGeometry(world); 
    106 
    107     const std::vector<G4PVPlacement*>& sensor_placements = g4ok->getSensorPlacements() ;       
    108     unsigned num_sensor = sensor_placements.size(); 
    109 
    110     // 2. use the placements to pass sensor data : efficiencies, categories, identifiers  
    111 
    112     const junoSD_PMT_v2* sd = dynamic_cast<const junoSD_PMT_v2*>(sd_) ;  
    113     assert(sd) ; 


    0596 void G4Opticks::setGeometry(const G4VPhysicalVolume* world)
     597 {
     598     LOG(LEVEL) << "[" ;
     599 
     600     LOG(LEVEL) << "( translateGeometry " ;
     601     GGeo* ggeo = translateGeometry( world ) ;
     602     LOG(LEVEL) << ") translateGeometry " ;
     603 
     604     if( m_standardize_geant4_materials )
     605     {
     606         standardizeGeant4MaterialProperties();
     607     }
     608 
     609     m_world = world ;
     610 
     611     setGeometry(ggeo);
     612 
     613     LOG(LEVEL) << "]" ;
     614 }

     940 GGeo* G4Opticks::translateGeometry( const G4VPhysicalVolume* top )
     941 {
     942     LOG(verbose) << "( key" ;
     943     const char* keyspec = X4PhysicalVolume::Key(top) ;
     944 
     945     bool parse_argv = false ;
     946     Opticks* ok = InitOpticks(keyspec, m_embedded_commandline_extra, parse_argv );
     947 
     948     // ok->setGPartsTransformOffset(true);  
     949     // HMM: CANNOT DO THIS PRIOR TO pre-7 
     950     // IDEA: COULD CREATE GParts TWICE WITH THE DIFFERENT SETTING AFTER pre-7 OGeo 
     951     // ACTUALLY: IT MAKES MORE SENSE TO SAVE IT ONLY IN CSG_GGeo : 
     952 
     953     const char* dbggdmlpath = ok->getDbgGDMLPath();
     954     if( dbggdmlpath != NULL )
     955     {
     956         LOG(info) << "( CGDML" ;
     957         CGDML::Export( dbggdmlpath, top );
     958         LOG(info) << ") CGDML" ;
     959     }

Old Integration : usage 
--------------------------

jcv junoSD_PMT_v2::

    1070 void junoSD_PMT_v2::EndOfEvent(G4HCofThisEvent* HCE)
    1071 {
    1072 
    1073 #ifdef WITH_G4OPTICKS
    1074     if(m_opticksMode > 0)
    1075     {
    1076         // Opticks GPU optical photon simulation and bulk hit population is done here 
    1077         m_jpmt_opticks->EndOfEvent(HCE);
    1078     }
    1079 #endif

jcv junoSD_PMT_v2_Opticks::

    118 void junoSD_PMT_v2_Opticks::EndOfEvent(G4HCofThisEvent*)
    119 {
    120     if(m_pmthitmerger_opticks == nullptr)
    121     {
    122         m_pmthitmerger_opticks = m_jpmt->getMergerOpticks();
    123     }
    124 
    125     const G4Event* event = G4RunManager::GetRunManager()->GetCurrentEvent() ;
    126     G4int eventID = event->GetEventID() ;
    127 
    128     G4Opticks* g4ok = G4Opticks::Get() ;
    129 
    130     unsigned num_gensteps = g4ok->getNumGensteps();
    131     unsigned num_photons = g4ok->getNumPhotons();
    132 
    133     LOG(info)
    134         << "["
    135         << " eventID " << eventID
    136         << " m_opticksMode " << m_opticksMode
    137         << " numGensteps " << num_gensteps
    138         << " numPhotons " << num_photons
    139         ;
    140 
    141     g4ok->propagateOpticalPhotons(eventID);




