sensor_info_into_new_workflow
===============================

* from :doc:`update_juno_opticks_integration_for_new_workflow`
* see also :doc:`instanceIdentity-into-new-workflow`


Not so keen on passing efficiencies one-by-one this way
--------------------------------------------------------

* identifiers and indices seems ok, as only one of those but 
  the other info will tend to need to be expanded

* better to establish the placement order and accept all values for
  all sensors in single API 


::

     30 struct ExampleSensor : public U4Sensor
     31 {
     32     // In reality would need ctor argument eg junoSD_PMT_v2 to lookup real values 
     33     unsigned getId(           const G4PVPlacement* pv) const { return pv->GetCopyNo() ; }
     34     float getEfficiency(      const G4PVPlacement* pv) const { return 1. ; }
     35     float getEfficiencyScale( const G4PVPlacement* pv) const { return 1. ; }
     36 }; 


Opted for::

     22 struct U4SensorIdentifier
     23 {
     24     virtual int getIdentity(const G4VPhysicalVolume* instance_outer_pv ) const = 0 ;
     25 };

     09 struct U4SensorIdentifierDefault
     10 {
     11     int getIdentity(const G4VPhysicalVolume* instance_outer_pv ) const ;
     12     static void FindSD_r( std::vector<const G4VPhysicalVolume*>& sdpv , const G4VPhysicalVolume* pv, int depth );
     13 };
     14 
     15 
     16 inline int U4SensorIdentifierDefault::getIdentity( const G4VPhysicalVolume* instance_outer_pv ) const
     17 {
     18     const G4PVPlacement* pvp = dynamic_cast<const G4PVPlacement*>(instance_outer_pv) ;
     19     int copyno = pvp ? pvp->GetCopyNo() : -1 ;
     20 
     21     std::vector<const G4VPhysicalVolume*> sdpv ;
     22     FindSD_r(sdpv, instance_outer_pv, 0 );
     23 
     24     unsigned num_sd = sdpv.size() ;
     25     int sensor_id = num_sd == 0 ? -1 : copyno ;
     26 
     27     std::cout
     28         << "U4SensorIdentifierDefault::getIdentity"
     29         << " copyno " << copyno
     30         << " num_sd " << num_sd
     31         << " sensor_id " << sensor_id
     32         ;
     33 
     34     return sensor_id ;
     35 }
     36 
     37 inline void U4SensorIdentifierDefault::FindSD_r( std::vector<const G4VPhysicalVolume*>& sdpv , const G4VPhysicalVolume* pv, int depth )
     38 {
     39     const G4LogicalVolume* lv = pv->GetLogicalVolume() ;
     40     G4VSensitiveDetector* sd = lv->GetSensitiveDetector() ;
     41     if(sd) sdpv.push_back(pv);
     42     for (size_t i=0 ; i < size_t(lv->GetNoDaughters()) ; i++ ) FindSD_r( lv->GetDaughter(i), depth+1, );
     43 }




Compare with Framework ProcessHits
-------------------------------------

::

     316 G4bool junoSD_PMT_v2::ProcessHits(G4Step * step,G4TouchableHistory*)
     317 {
     ...
     391     // == get the copy number -> pmt id
     392     int pmtID = get_pmtid(track);
     ...
     444     if (m_pmthitmerger and m_pmthitmerger->getMergeFlag()) {
     445         // == if merged, just return true. That means just update the hit
     446         // NOTE: only the time and count will be update here, the others 
     447         //       will not filled.
     448         bool ok = m_pmthitmerger->doMerge(pmtID, hittime);
     449         if (ok) {
     450             m_merge_count += 1 ;
     451             return true;
     452         }





What is the Opticks equivalent of junoSD_PMT_v2::get_pmtid ?
-------------------------------------------------------------

Opticks shifts focus to geometry preparation stage, so it doesnt have to 
be repeated for every photon.  That means:

1. duplicating sensor_id and sensor_index labels to all ~5-6 nodes of the subtree of 
   each instance within stree (formerly GGeo/GNodeLib/GNode)

2. planting sensor_id and sensor_index within the CSGFoundry inst in 
   fourth column of the transform. 

But how to get sensor_id and sensor_index in first place ?

sensor_index 
   0-based index that orders the sensors as they are 
   encountered in the standard postorder traversal of the volumes

   * this means that given a way to get sensor_id of a volume 
     can derive the sensor index within Opticks   

sensor_id
   this comes from the copyNo but that is JUNO specific so 
   cannot assume that is the 


How to label the subtrees ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

U4Tree::convertNodes_r 
     too early as the instances not yet defined 
    
stree::add_inst
     is the right place to label the tree and populate the inst 4th column, 
     but need to operate without Geant4 types within stree : so need to 
     collect sensor_id integer into the stree/snode during U4Tree::convertNodes_r 
     using the U4Sensor object passed from the framework (or copyno) 



junoSD_PMT_v2::get_pmtid
---------------------------

::

    junoSD_PMT_v2::ProcessHits dumpcount 0
    U4Touchable::Desc depth 8
     i  0 cp      0 so HamamatsuR12860_PMT_20inch_body_solid_1_4 pv                         HamamatsuR12860_PMT_20inch_body_phys
     i  1 cp      0 so HamamatsuR12860_PMT_20inch_pmt_solid_1_4 pv                          HamamatsuR12860_PMT_20inch_log_phys
     i  2 cp   9744 so             HamamatsuR12860sMask_virtual pv                                       pLPMT_Hamamatsu_R12860
     i  3 cp      0 so                              sInnerWater pv                                                  pInnerWater
     i  4 cp      0 so                           sReflectorInCD pv                                             pCentralDetector
     i  5 cp      0 so                          sOuterWaterPool pv                                              pOuterWaterPool
     i  6 cp      0 so                              sPoolLining pv                                                  pPoolLining
     i  7 cp      0 so                              sBottomRock pv                                                     pBtmRock

    junoSD_PMT_v2::ProcessHits dumpcount 1
    U4Touchable::Desc depth 8
     i  0 cp      0 so    NNVTMCPPMT_PMT_20inch_body_solid_head pv                              NNVTMCPPMT_PMT_20inch_body_phys
     i  1 cp      0 so     NNVTMCPPMT_PMT_20inch_pmt_solid_head pv                               NNVTMCPPMT_PMT_20inch_log_phys
     i  2 cp   3505 so                  NNVTMCPPMTsMask_virtual pv                                            pLPMT_NNVT_MCPPMT
     i  3 cp      0 so                              sInnerWater pv                                                  pInnerWater
     i  4 cp      0 so                           sReflectorInCD pv                                             pCentralDetector
     i  5 cp      0 so                          sOuterWaterPool pv                                              pOuterWaterPool
     i  6 cp      0 so                              sPoolLining pv                                                  pPoolLining
     i  7 cp      0 so                              sBottomRock pv                                                     pBtmRock





::

     477 int junoSD_PMT_v2::get_pmtid(G4Track* track) {
     478     int ipmt= -1;
     479     // find which pmt we are in
     480     // The following doesn't work anymore (due to new geometry optimization?)
     481     //  ipmt=fastTrack.GetEnvelopePhysicalVolume()->GetMother()->GetCopyNo();
     482     // so we do this:
     483     {
     484         const G4VTouchable* touch= track->GetTouchable();
     485         int nd= touch->GetHistoryDepth();
     486         int id=0;
     487         for (id=0; id<nd; id++) {   
     488             if (touch->GetVolume(id)==track->GetVolume()) {
     ///
     ///         iterate up stack of volumes : until find the one of this track : 
     ///         would expect that to be the first 
     ///
     489                 int idid=1;
     490                 G4VPhysicalVolume* tmp_pv=NULL;
     491                 for (idid=1; idid < (nd-id); ++idid) {
     ///
     ///            code edited to make less obtuse. 
     ///            looks like proceeds up the stack until finds a volume with siblings
     ///            in order to get the CopyNo  
     ///
     ...
     494                     G4LogicalVolume* mother_vol = touch->GetVolume(id+idid)->GetLogicalVolume();
     495                     G4LogicalVolume* daughter_vol = touch->GetVolume(id+idid-1)->GetLogicalVolume();

     497                     int no_daugh = mother_vol -> GetNoDaughters();
     498                     if (no_daugh > 1) {
     499                         int count = 0;
     500                         for (int i=0; (count<2) &&(i < no_daugh); ++i) {
     501                             if (daughter_vol->GetName()==mother_vol->GetDaughter(i)->GetLogicalVolume()->GetName()) {
     503                                 ++count;
     504                             }
     505                         }
     506                         if (count > 1) {
     507                             break;
     508                         }
     509                     }
     510                     // continue to find
     511                 }
     512                 ipmt= touch->GetReplicaNumber(id+idid-1);
     513                 break;
     514             }
     515         }
     516         if (ipmt < 0) {
     517             G4Exception("junoPMTOpticalModel: could not find envelope -- where am I !?!", // issue
     518                     "", //Error Code
     519                     FatalException, // severity
     520                     "");
     521         }
     522     }
     523 
     524     return ipmt;
     525 }


g4-cls G4VTouchable::

     34 inline
     35 G4int G4VTouchable::GetCopyNumber(G4int depth) const
     36 { 
     37   return GetReplicaNumber(depth);
     38 }


     59 inline
     60 G4VPhysicalVolume* G4TouchableHistory::GetVolume( G4int depth ) const
     61 {   
     62   return fhistory.GetVolume(CalculateHistoryIndex(depth));
     63 }
     64    
     65 inline
     66 G4VSolid* G4TouchableHistory::GetSolid( G4int depth ) const
     67 {
     68   return fhistory.GetVolume(CalculateHistoryIndex(depth))
     69                             ->GetLogicalVolume()->GetSolid();
     70 }
     71   
     72 inline
     73 G4int G4TouchableHistory::GetReplicaNumber( G4int depth ) const
     74 {
     75   return fhistory.GetReplicaNo(CalculateHistoryIndex(depth));
     76 }
     77 

     53 inline
     54 G4int G4TouchableHistory::CalculateHistoryIndex( G4int stackDepth ) const
     55 { 
     56   return (fhistory.GetDepth()-stackDepth); // was -1
     57 }

::

    098   G4ThreeVector ftlate;
     99   G4NavigationHistory fhistory;
    100 };




U4Sensor
----------

::

    epsilon:u4 blyth$ opticks-f U4Sensor
    ./u4/CMakeLists.txt:    U4Sensor.h
    ./u4/U4Sensor.h:U4Sensor.h
    ./u4/U4Sensor.h:struct U4Sensor
    ./g4cx/G4CXOpticks.hh:struct U4Sensor ; 
    ./g4cx/G4CXOpticks.hh:    const U4Sensor* sd ; 
    ./g4cx/G4CXOpticks.hh:    void setSensor(const U4Sensor* sd );
    ./g4cx/G4CXOpticks.hh:    // HMM: maybe add U4Sensor arg here, 
    ./g4cx/tests/G4CXSimulateTest.cc:#include "U4Sensor.h"
    ./g4cx/tests/G4CXSimulateTest.cc:struct ExampleSensor : public U4Sensor
    ./g4cx/G4CXOpticks.cc:void G4CXOpticks::setSensor(const U4Sensor* sd_ )
    epsilon:opticks blyth$ 
    epsilon:opticks blyth$ 

::

    120 void G4CXOpticks::setSensor(const U4Sensor* sd_ )
    121 {
    122     sd = sd_ ;
    123 }

    030 struct ExampleSensor : public U4Sensor
     31 {
     32     // In reality would need ctor argument eg junoSD_PMT_v2 to lookup real values 
     33     unsigned getId(           const G4PVPlacement* pv) const { return pv->GetCopyNo() ; }
     34     float getEfficiency(      const G4PVPlacement* pv) const { return 1. ; }
     35     float getEfficiencyScale( const G4PVPlacement* pv) const { return 1. ; }
     36 }; 




What is the effect of having non-sensitive SD volumes ?
----------------------------------------------------------

Probably no effect, as need "theStatus == Detection" anyhow
and to get "Detection" need an efficiency property with value 
greater than zero and a suitable random throw. 

BUT : it adds a complication for communicating efficiencies 

::

    411 inline
    412 void InstrumentedG4OpBoundaryProcess::DoAbsorption()
    413 {
    414               theStatus = Absorption;
    415 
    416               if ( G4BooleanRand_theEfficiency(theEfficiency) ) {
    417 
    418                  // EnergyDeposited =/= 0 means: photon has been detected
    419                  theStatus = Detection;
    420                  aParticleChange.ProposeLocalEnergyDeposit(thePhotonMomentum);
    421               }
    422               else {
    423                  aParticleChange.ProposeLocalEnergyDeposit(0.0);
    424               }
    425 
    426               NewMomentum = OldMomentum;
    427               NewPolarization = OldPolarization;
    428 
    429 //              aParticleChange.ProposeEnergy(0.0);
    430               aParticleChange.ProposeTrackStatus(fStopAndKill);
    431 }


::

    1617 G4bool InstrumentedG4OpBoundaryProcess::InvokeSD(const G4Step* pStep)
    1618 {
    1619   G4Step aStep = *pStep;
    1620 
    1621   aStep.AddTotalEnergyDeposit(thePhotonMomentum);
    1622 
    1623   G4VSensitiveDetector* sd = aStep.GetPostStepPoint()->GetSensitiveDetector();
    1624   if (sd) return sd->Hit(&aStep);
    1625   else return false;
    1626 }


    0222 G4VParticleChange*
     223 InstrumentedG4OpBoundaryProcess::PostStepDoIt(const G4Track& aTrack, const G4Step& aStep)
     224 {

     663         if ( theStatus == Detection && fInvokeSD ) InvokeSD(pStep);
     664 
     665         return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
     666 }



Check Sensors : systematically 2x the number of SD than would expect ?
------------------------------------------------------------------------

::

    epsilon:sysrap blyth$ jgr SetSensitive 
    ./Simulation/DetSimV2/PMTSim/src/Hello3inchPMTManager.cc:    body_log->SetSensitiveDetector(m_detector);
    ./Simulation/DetSimV2/PMTSim/src/Hello3inchPMTManager.cc:    inner1_log->SetSensitiveDetector(m_detector);
    ./Simulation/DetSimV2/PMTSim/src/dyw_PMT_LogicalVolume.cc:  body_log->SetSensitiveDetector(detector);
    ./Simulation/DetSimV2/PMTSim/src/dyw_PMT_LogicalVolume.cc:  inner1_log->SetSensitiveDetector(detector);
    ...


    457 void NNVTMCPPMTManager::helper_make_logical_volume()
    458 {
    459     body_log= new G4LogicalVolume
    460         ( body_solid,
    461           GlassMat,
    462           GetName()+"_body_log" );
    463 
    464     m_logical_pmt = new G4LogicalVolume
    465         ( pmt_solid,
    466           GlassMat,
    467           GetName()+"_log" );
    468 
    469     body_log->SetSensitiveDetector(m_detector);
    470 
    471     inner1_log= new G4LogicalVolume
    472         ( inner1_solid,
    473           PMT_Vacuum,
    474           GetName()+"_inner1_log" );
    475     inner1_log->SetSensitiveDetector(m_detector);
    476 

::

    desc_sensor
        nds :  lv :                                             soname : 0th 
       4997 : 106 :          HamamatsuR12860_PMT_20inch_inner1_solid_I : 70970 
       4997 : 108 :          HamamatsuR12860_PMT_20inch_body_solid_1_4 : 70969 
      12615 : 113 :            NNVTMCPPMT_PMT_20inch_inner1_solid_head : 70984 
      12615 : 115 :              NNVTMCPPMT_PMT_20inch_body_solid_head : 70983 
      25600 : 118 :                  PMT_3inch_inner1_solid_ell_helper : 194251 
      25600 : 120 :                PMT_3inch_body_solid_ell_ell_helper : 194250 
       2400 : 130 :                       PMT_20inch_veto_inner1_solid : 322257 
       2400 : 132 :                     PMT_20inch_veto_body_solid_1_2 : 322256 
      91224 :     :                                                    :  
    zth:70970
             +      snode ix:  70970 dh: 9 nc:    0 lv:106 se:      1. sf 125 :   -4997 : 8a3d4fe0109975976aef9a87c7842a63. HamamatsuR12860_PMT_20inch_inner1_solid_I
    zth:70969
            +       snode ix:  70969 dh: 8 nc:    2 lv:108 se:      0. sf 124 :   -4997 : f343253c582a107559795892ee52220f. HamamatsuR12860_PMT_20inch_body_solid_1_4
             +      snode ix:  70970 dh: 9 nc:    0 lv:106 se:      1. sf 125 :   -4997 : 8a3d4fe0109975976aef9a87c7842a63. HamamatsuR12860_PMT_20inch_inner1_solid_I
             +      snode ix:  70971 dh: 9 nc:    0 lv:107 se:     -1. sf 126 :   -4997 : fd63d016360b18a01ab74dcd01b5e32c. HamamatsuR12860_PMT_20inch_inner2_solid_1_4
    zth:70984
             +      snode ix:  70984 dh: 9 nc:    0 lv:113 se:      5. sf 131 :  -12615 : 341ae4bffe82aa82798d3886484179a6. NNVTMCPPMT_PMT_20inch_inner1_solid_head
    zth:70983
            +       snode ix:  70983 dh: 8 nc:    2 lv:115 se:      4. sf 130 :  -12615 : 067136473b80d872bffc4de42fbf2337. NNVTMCPPMT_PMT_20inch_body_solid_head
             +      snode ix:  70984 dh: 9 nc:    0 lv:113 se:      5. sf 131 :  -12615 : 341ae4bffe82aa82798d3886484179a6. NNVTMCPPMT_PMT_20inch_inner1_solid_head
             +      snode ix:  70985 dh: 9 nc:    0 lv:114 se:     -1. sf 132 :  -12615 : 946e0765de8ecaf64388ebe09c86680e. NNVTMCPPMT_PMT_20inch_inner2_solid_head
    zth:194251
            +       snode ix: 194251 dh: 8 nc:    0 lv:118 se:  35225. sf 133 :  -25600 : c301322ae66e730aac2a27836ead8b89. PMT_3inch_inner1_solid_ell_helper
    zth:194250
           +        snode ix: 194250 dh: 7 nc:    2 lv:120 se:  35224. sf 135 :  -25600 : 2485b31b2df8ec818453e3a773f02436. PMT_3inch_body_solid_ell_ell_helper
            +       snode ix: 194251 dh: 8 nc:    0 lv:118 se:  35225. sf 133 :  -25600 : c301322ae66e730aac2a27836ead8b89. PMT_3inch_inner1_solid_ell_helper
            +       snode ix: 194252 dh: 8 nc:    0 lv:119 se:     -1. sf 136 :  -25600 : 511486df0c29cd5e2e9a38b4a6d2e108. PMT_3inch_inner2_solid_ell_helper
    zth:322257
           +        snode ix: 322257 dh: 7 nc:    0 lv:130 se:  86425. sf 116 :   -2400 : 4c4aff2e5de757833006d7f55c3f2127. PMT_20inch_veto_inner1_solid
    zth:322256
          +         snode ix: 322256 dh: 6 nc:    2 lv:132 se:  86424. sf 118 :   -2400 : 38ba238fc5def688b7fe3639cc3f6c6f. PMT_20inch_veto_body_solid_1_2
           +        snode ix: 322257 dh: 7 nc:    0 lv:130 se:  86425. sf 116 :   -2400 : 4c4aff2e5de757833006d7f55c3f2127. PMT_20inch_veto_inner1_solid
           +        snode ix: 322258 dh: 7 nc:    0 lv:131 se:     -1. sf 117 :   -2400 : d2f14afe26c74ad9d618c6d18a2e25a1. PMT_20inch_veto_inner2_solid



::

     20 def desc_sensor(st):
     21     """
     22     desc_sensor
     23         nds :  lv : soname
     24        4997 : 106 : HamamatsuR12860_PMT_20inch_inner1_solid_I 
     25        4997 : 108 : HamamatsuR12860_PMT_20inch_body_solid_1_4 
     26       12615 : 113 : NNVTMCPPMT_PMT_20inch_inner1_solid_head 
     27       12615 : 115 : NNVTMCPPMT_PMT_20inch_body_solid_head 
     28       25600 : 118 : PMT_3inch_inner1_solid_ell_helper 
     29       25600 : 120 : PMT_3inch_body_solid_ell_ell_helper 
     30        2400 : 130 : PMT_20inch_veto_inner1_solid 
     31        2400 : 132 : PMT_20inch_veto_body_solid_1_2 
     32 
     33     """
     34     ws = np.where(st.nds.sensor > -1 )[0]
     35     se = st.nds.sensor[ws]
     36     xse = np.arange(len(se), dtype=np.int32)
     37     assert np.all( xse == se )  
     38     ulv, nlv = np.unique(st.nds.lvid[ws], return_counts=True)
     39     
     40     hfmt = "%7s : %3s : %s"
     41     fmt = "%7d : %3d : %s "
     42     hdr = hfmt % ("nds", "lv", "soname" )
     43     
     44     head = ["desc_sensor",hdr]
     45     body = [fmt % ( nlv[i], ulv[i], st.soname_[ulv[i]].decode() ) for i in range(len(ulv))]
     46     tail = [hfmt % ( nlv.sum(), "", "" ),]
     47     return "\n".join(head+body+tail)
     48     
     49     


::

    epsilon:offline blyth$ jgr _1_4
    ./Simulation/DetSimV2/PMTSim/src/Hamamatsu_R12860_PMTSolid.cc:				 solidname+"_1_4",
    ./Simulation/DetSimV2/PMTSim/src/Hamamatsu_R12860_PMTSolid.cc:    double neck_offset_z = -210. + m4_h/2 ;  // see _1_4 below
    ./Simulation/DetSimV2/PMTSim/src/Hamamatsu_R12860_PMTSolid.cc:    double c_cy = neck_offset_z -m4_h/2 ;    // -210. torus_z  (see _1_4 below)
    epsilon:offline blyth$ 




Should sensor_id be placed into OptixInstance .instanceId ?
------------------------------------------------------------------

::

    the returned unsigned value is used by IAS_Builder to set the OptixInstance .instanceId 
    Within CSGOptiX/CSGOptiX7.cu:: __closesthit__ch *optixGetInstanceId()* is used to 
    passes the instanceId value into "quad2* prd" (per-ray-data) which is available 
    within qudarap/qsim.h methods. 
    
    The 32 bit unsigned returned by *getInstanceIdentity* may not use the top 8 bits 
    because of an OptiX 7 limit of 24 bits, from Properties::dump::

        limitMaxInstanceId :   16777215    ffffff

    (that limit might well be raised in versions after 700)





HMM: how to split those 24 bits ? 

1. sensor id
2. sensor category (4 cat:2 bits, 8 cat: 3 bits)

::

    In [14]: for i in range(32): print(" (0x1 << %2d) - 1   %16x   %16d  %16.2f  " % (i, (0x1 << i)-1, (0x1 << i)-1, float((0x1 << i)-1)/1e6 )) 

     (0x1 <<  0) - 1                  0                  0              0.00  
     (0x1 <<  1) - 1                  1                  1              0.00  
     (0x1 <<  2) - 1                  3                  3              0.00  
     (0x1 <<  3) - 1                  7                  7              0.00  
     (0x1 <<  4) - 1                  f                 15              0.00  
     (0x1 <<  5) - 1                 1f                 31              0.00  
     (0x1 <<  6) - 1                 3f                 63              0.00  
     (0x1 <<  7) - 1                 7f                127              0.00  
     (0x1 <<  8) - 1                 ff                255              0.00  
     (0x1 <<  9) - 1                1ff                511              0.00  
     (0x1 << 10) - 1                3ff               1023              0.00  
     (0x1 << 11) - 1                7ff               2047              0.00  
     (0x1 << 12) - 1                fff               4095              0.00  
     (0x1 << 13) - 1               1fff               8191              0.01  
     (0x1 << 14) - 1               3fff              16383              0.02  
     (0x1 << 15) - 1               7fff              32767              0.03  
     (0x1 << 16) - 1               ffff              65535              0.07  
     (0x1 << 17) - 1              1ffff             131071              0.13  
     (0x1 << 18) - 1              3ffff             262143              0.26  
     (0x1 << 19) - 1              7ffff             524287              0.52  
     (0x1 << 20) - 1              fffff            1048575              1.05  
     (0x1 << 21) - 1             1fffff            2097151              2.10  
     (0x1 << 22) - 1             3fffff            4194303              4.19  
     (0x1 << 23) - 1             7fffff            8388607              8.39  
     (0x1 << 24) - 1             ffffff           16777215             16.78  
     (0x1 << 25) - 1            1ffffff           33554431             33.55  
     (0x1 << 26) - 1            3ffffff           67108863             67.11  
     (0x1 << 27) - 1            7ffffff          134217727            134.22  
     (0x1 << 28) - 1            fffffff          268435455            268.44  
     (0x1 << 29) - 1           1fffffff          536870911            536.87  
     (0x1 << 30) - 1           3fffffff         1073741823           1073.74  
     (0x1 << 31) - 1           7fffffff         2147483647           2147.48  







