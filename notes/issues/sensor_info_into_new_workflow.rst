sensor_info_into_new_workflow
===============================

* from :doc:`update_juno_opticks_integration_for_new_workflow`
* see also :doc:`instanceIdentity-into-new-workflow`


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
     ///         iterate up stack of volumes : until find the one of this track 
     ///
     489                 int idid=1;
     490                 G4VPhysicalVolume* tmp_pv=NULL;
     491                 for (idid=1; idid < (nd-id); ++idid) {
     ///
     ///            code edited to make less obtuse 
     ///            looks like proceeds to check 
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



