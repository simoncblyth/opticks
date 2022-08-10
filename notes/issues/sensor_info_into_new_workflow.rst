sensor_info_into_new_workflow
===============================

* from :doc:`update_juno_opticks_integration_for_new_workflow`


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



