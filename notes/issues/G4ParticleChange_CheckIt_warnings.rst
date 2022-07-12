G4ParticleChange_CheckIt_warnings : the Momentum Change is not unit vector  : 1e-8 
=======================================================================================


* probably need to normalize mom after transforming the input photons
* did this and see no difference : but have narrowed
* recall previously the need for double precision input photons to avoid this

  * prefer not to do that as it introduces an immediate A-B difference  



::

    BP=G4ParticleChange::CheckIt ./u4s.sh dbg


confirmed that using wide input photons avoids this
------------------------------------------------------

::

     197 void SEvt::setFrame(const sframe& fr )
     198 {
     199     frame = fr ;
     200 
     201     if(SEventConfig::IsRGModeSimtrace())
     202     {
     203         addGenstep( SFrameGenstep::MakeCenterExtentGensteps(frame) );
     204     }
     205     else if(SEventConfig::IsRGModeSimulate() && hasInputPhoton())
     206     {
     207         assert( genstep.size() == 0 ) ; // cannot mix input photon running with other genstep running  
     208 
     209         addGenstep(MakeInputPhotonGenstep(input_photon, frame));
     210 
     211         bool normalize = true ;  // normalize mom and pol after doing the transform 
     212 
     213         NP* ipt = frame.transform_photon_m2w( input_photon, normalize );
     214 
     215         //input_photon_transformed = ipt->ebyte == 8 ? NP::MakeNarrow(ipt) : ipt ;
     216         input_photon_transformed = ipt ;
     217 
     218         // narrow here to prevent Geant4 seeing double precision and Opticks float precision 
     219     }



::

    SCF::ReadNames path /home/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/41c046fe05b28cb70b1fc65d0e6b7749/1/CSG_GGeo/CSGFoundry/primname.txt names.size 0
    2022-07-13 02:32:53.974 INFO  [170511] [main@186] U4Random::desc U4Random::isReady() YES m_seqpath /home/blyth/.opticks/precooked/QSimTest/rng_sequence/rng_sequence_f_ni1000000_nj16_nk16_tranche100000/rng_sequence_f_ni100000_nj16_nk16_ioffset000000.npy m_seq_ni 100000 m_seq_nv 256
    2022-07-13 02:32:53.974 INFO  [170511] [main@189]  desc ShimG4OpAbsorption_FLOAT_ShimG4OpRayleigh_FLOAT/J000/ALL

    **************************************************************
     Geant4 version Name: geant4-10-04-patch-02 [MT]   (25-May-2018)
                           Copyright : Geant4 Collaboration
                          References : NIM A 506 (2003), 250-303
                                     : IEEE-TNS 53 (2006), 270-278
                                     : NIM A 835 (2016), 186-225
                                 WWW : http://geant4.org/
    **************************************************************

    G4GDML: Reading '/home/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/41c046fe05b28cb70b1fc65d0e6b7749/1/origin_CGDMLKludge.gdml'...
    G4GDML: Reading definitions...
    G4GDML: Reading materials...
    G4GDML: Reading solids...
    G4GDML: Reading structure...
    G4GDML: Reading setup...
    G4GDML: Reading '/home/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/41c046fe05b28cb70b1fc65d0e6b7749/1/origin_CGDMLKludge.gdml' done!
    DsG4Scintillation::DsG4Scintillation level 0 verboseLevel 0
    2022-07-13 02:34:20.946 INFO  [170511] [U4Recorder::BeginOfRunAction@80] 
    2022-07-13 02:34:20.947 INFO  [170511] [U4RecorderTest::GeneratePrimaries@130] [ fPrimaryMode I
    2022-07-13 02:34:20.950 INFO  [170511] [U4RecorderTest::GeneratePrimaries@138] ]
    2022-07-13 02:34:20.950 INFO  [170511] [U4Recorder::BeginOfEventAction@82] 
    2022-07-13 02:34:21.016 INFO  [170511] [U4Recorder::PreUserTrackingAction_Optical@153]  label.id 9000
    2022-07-13 02:34:21.070 INFO  [170511] [U4Recorder::PreUserTrackingAction_Optical@153]  label.id 8000
    2022-07-13 02:34:21.124 INFO  [170511] [U4Recorder::PreUserTrackingAction_Optical@153]  label.id 7000
    2022-07-13 02:34:21.178 INFO  [170511] [U4Recorder::PreUserTrackingAction_Optical@153]  label.id 6000
    2022-07-13 02:34:21.233 INFO  [170511] [U4Recorder::PreUserTrackingAction_Optical@153]  label.id 5000
    2022-07-13 02:34:21.287 INFO  [170511] [U4Recorder::PreUserTrackingAction_Optical@153]  label.id 4000
    2022-07-13 02:34:21.342 INFO  [170511] [U4Recorder::PreUserTrackingAction_Optical@153]  label.id 3000
    2022-07-13 02:34:21.396 INFO  [170511] [U4Recorder::PreUserTrackingAction_Optical@153]  label.id 2000
    2022-07-13 02:34:21.451 INFO  [170511] [U4Recorder::PreUserTrackingAction_Optical@153]  label.id 1000
    2022-07-13 02:34:21.504 INFO  [170511] [U4Recorder::PreUserTrackingAction_Optical@153]  label.id 0
    2022-07-13 02:34:21.504 INFO  [170511] [U4Recorder::EndOfEventAction@83] 
    2022-07-13 02:34:21.505 INFO  [170511] [U4Recorder::EndOfRunAction@81] 
    2022-07-13 02:34:21.505 INFO  [170511] [main@211] /tmp/blyth/opticks/U4RecorderTest/ShimG4OpAbsorption_FLOAT_ShimG4OpRayleigh_FLOAT/J000/ALL
    U4Random::saveProblemIdx m_problem_idx.size 0 ()
    2022-07-13 02:34:21.564 INFO  [170511] [main@216] /tmp/blyth/opticks/U4RecorderTest/ShimG4OpAbsorption_FLOAT_ShimG4OpRayleigh_FLOAT/J000/ALL
    === ./u4s.sh : logdir /tmp/blyth/opticks/U4RecorderTest
    N[blyth@localhost u4]$ 





How to switch off the warning ?
-----------------------------------


Each process has aParticleChange::

     53 G4VProcess::G4VProcess(const G4String& aName, G4ProcessType   aType )
     54                   : aProcessManager(0),
     55                 pParticleChange(0),
     56                     theNumberOfInteractionLengthLeft(-1.0),
     57                     currentInteractionLength(-1.0),
     58             theInitialNumberOfInteractionLength(-1.0),
     59                     theProcessName(aName),
     60             theProcessType(aType),
     61             theProcessSubType(-1),
     62                     thePILfactor(1.0),
     63                     enableAtRestDoIt(true),
     64                     enableAlongStepDoIt(true),
     65                     enablePostStepDoIt(true),
     66                     verboseLevel(0),
     67                     masterProcessShadow(0)
     68 
     69 {
     70   pParticleChange = &aParticleChange;
     71 }
     72 





::

    045 const G4double G4VParticleChange::accuracyForWarning = 1.0e-9;
    046 const G4double G4VParticleChange::accuracyForException = 0.001;

    047 
     48 G4VParticleChange::G4VParticleChange()
     49   :theListOfSecondaries(0),
     50    theNumberOfSecondaries(0),
     51    theSizeOftheListOfSecondaries(G4TrackFastVectorSize),
     52    theStatusChange(fAlive),
     53    theSteppingControlFlag(NormalCondition),
     54    theLocalEnergyDeposit(0.0),
     55    theNonIonizingEnergyDeposit(0.0),
     56    theTrueStepLength(0.0),
     57    theFirstStepInVolume(false),
     58    theLastStepInVolume(false),
     59    theParentWeight(1.0),
     60    isParentWeightProposed(false),
     61    fSetSecondaryWeightByProcess(false),
     62    theParentGlobalTime(0.0),
     63    verboseLevel(1), 
     64    debugFlag(false)
     65 {
     66 #ifdef G4VERBOSE
     67   // activate CHeckIt if in VERBOSE mode
     68   debugFlag = true;
     69 #endif
     70   theListOfSecondaries = new G4TrackFastVector();
     71 }




    306   protected:
    307     // CheckSecondary method is provided for debug
    308     G4bool CheckSecondary(G4Track&);
    309 
    310     G4double GetAccuracyForWarning() const;
    311     G4double GetAccuracyForException() const;
    312 
    313   protected:
    314     G4bool   debugFlag;
    315 
    316     // accuracy levels
    317     static const G4double accuracyForWarning;
    318     static const G4double accuracyForException;
    319 
    320 
    321 };


Lots of these::

    2022-07-12 21:39:55.818 INFO  [128695] [U4Recorder::BeginOfEventAction@82] 
      G4ParticleChange::CheckIt  : the Momentum Change is not unit vector !!  Difference:  5.61912e-08
    opticalphoton E=2.47473e-06 pos=-12.7039, 10.0346, 12.4523
          -----------------------------------------------
            G4ParticleChange Information  
          -----------------------------------------------
            # of 2ndaries       :                    0
          -----------------------------------------------
            Energy Deposit (MeV):                    0
            Non-ionizing Energy Deposit (MeV):                    0
            Track Status        :                Alive
            True Path Length (mm) :                  793
            Stepping Control      :                    0
            Mass (GeV)   :                    0
            Charge (eplus)   :                    0
            MagneticMoment   :                    0
                    :  =                    0*[e hbar]/[2 m]
            Position - x (mm)   :            -1.22e+04
            Position - y (mm)   :             9.65e+03
            Position - z (mm)   :              1.2e+04
            Time (ns)           :                 3.67
            Proper Time (ns)    :                    0
            Momentum Direct - x :                0.621
            Momentum Direct - y :                -0.49
            Momentum Direct - z :               -0.611
            Kinetic Energy (MeV):             2.47e-06
            Velocity  (/c):                    1
            Polarization - x    :               -0.707
            Polarization - y    :               -0.687
            Polarization - z    :               -0.168
            Touchable (pointer) :                    0
      G4ParticleChange::CheckIt  : the Momentum Change is not unit vector !!  Difference:  5.61912e-08


Huh normalizing after the transform seems to make no difference::

    2022-07-13 02:12:37.742 INFO  [157058] [U4Recorder::BeginOfEventAction@82] 
      G4ParticleChange::CheckIt  : the Momentum Change is not unit vector !!  Difference:  5.61912e-08
    opticalphoton E=2.47473e-06 pos=-11.4378, 9.03553, 11.3013
          -----------------------------------------------
            G4ParticleChange Information  
          -----------------------------------------------
            # of 2ndaries       :                    0
          -----------------------------------------------
            Energy Deposit (MeV):                    0
            Non-ionizing Energy Deposit (MeV):                    0
            Track Status        :                Alive
            True Path Length (mm) :                  790
            Stepping Control      :                    0
            Mass (GeV)   :                    0
            Charge (eplus)   :                    0
            MagneticMoment   :                    0
                    :  =                    0*[e hbar]/[2 m]
            Position - x (mm)   :            -1.19e+04
            Position - y (mm)   :             9.42e+03
            Position - z (mm)   :             1.18e+04
            Time (ns)           :                 3.65
            Proper Time (ns)    :                    0
            Momentum Direct - x :               -0.621
            Momentum Direct - y :                 0.49
            Momentum Direct - z :                0.611
            Kinetic Energy (MeV):             2.47e-06
            Velocity  (/c):                    1
            Polarization - x    :               -0.503
            Polarization - y    :               -0.848
            Polarization - z    :                0.168
            Touchable (pointer) :                    0
      G4ParticleChange::CheckIt  : the Momentum Change is not unit vector !!  Difference:  5.61912e-08



::

    506 G4bool G4ParticleChange::CheckIt(const G4Track& aTrack)
    507 {
    508   G4bool    exitWithError = false;
    509   G4double  accuracy;
    510   static G4ThreadLocal G4int nError = 0;
    511 #ifdef G4VERBOSE
    512   const  G4int maxError = 30;
    513 #endif
    514 
    515   // No check in case of "fStopAndKill" 
    516   if (GetTrackStatus() ==   fStopAndKill )  return G4VParticleChange::CheckIt(aTrack);
    517 
    518   // MomentumDirection should be unit vector
    519   G4bool itsOKforMomentum = true;
    520   if ( theEnergyChange >0.) {
    521     accuracy = std::fabs(theMomentumDirectionChange.mag2()-1.0);
    522     if (accuracy > accuracyForWarning) {
    523       itsOKforMomentum = false;
    524       nError += 1;
    525       exitWithError = exitWithError || (accuracy > accuracyForException);
    526 #ifdef G4VERBOSE
    527       if (nError < maxError) {
    528     G4cout << "  G4ParticleChange::CheckIt  : ";
    529     G4cout << "the Momentum Change is not unit vector !!"
    530            << "  Difference:  " << accuracy << G4endl;
    531     G4cout << aTrack.GetDefinition()->GetParticleName()
    532            << " E=" << aTrack.GetKineticEnergy()/MeV
    533            << " pos=" << aTrack.GetPosition().x()/m
    534            << ", " << aTrack.GetPosition().y()/m
    535            << ", " << aTrack.GetPosition().z()/m
    536            <<G4endl;
    537       }
    538 #endif
    539     }
    540   }





::

    2022-07-13 02:24:05.406 INFO  [168163] [U4Recorder::BeginOfEventAction@82] 

    Breakpoint 1, 0x00007ffff2379b10 in G4ParticleChange::CheckIt(G4Track const&) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4track.so
    Missing separate debuginfos, use: debuginfo-install bzip2-libs-1.0.6-13.el7.x86_64 cyrus-sasl-lib-2.1.26-23.el7.x86_64 expat-2.1.0-10.el7_3.x86_64 freetype-2.8-12.el7_6.1.x86_64 glibc-2.17-307.el7.1.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-37.el7_6.x86_64 libICE-1.0.9-9.el7.x86_64 libSM-1.2.2-2.el7.x86_64 libX11-1.6.7-4.el7_9.x86_64 libXau-1.0.8-2.1.el7.x86_64 libXext-1.3.3-3.el7.x86_64 libcom_err-1.42.9-13.el7.x86_64 libcurl-7.29.0-59.el7_9.1.x86_64 libicu-50.2-4.el7_7.x86_64 libidn-1.28-4.el7.x86_64 libpng-1.5.13-7.el7_2.x86_64 libselinux-2.5-14.1.el7.x86_64 libssh2-1.8.0-3.el7.x86_64 libuuid-2.23.2-59.el7_6.1.x86_64 libxcb-1.13-1.el7.x86_64 nspr-4.19.0-1.el7_5.x86_64 nss-3.36.0-7.1.el7_6.x86_64 nss-softokn-freebl-3.36.0-5.el7_5.x86_64 nss-util-3.36.0-1.1.el7_6.x86_64 openldap-2.4.44-25.el7_9.x86_64 openssl-libs-1.0.2k-24.el7_9.x86_64 pcre-8.32-17.el7.x86_64 zlib-1.2.7-18.el7.x86_64
    (gdb) bt
    #0  0x00007ffff2379b10 in G4ParticleChange::CheckIt(G4Track const&) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4track.so
    #1  0x00007ffff237fa50 in G4ParticleChangeForTransport::UpdateStepForAlongStep(G4Step*) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4track.so
    #2  0x00007ffff4492d1b in G4SteppingManager::InvokeAlongStepDoItProcs() () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4tracking.so
    #3  0x00007ffff4490b7f in G4SteppingManager::Stepping() () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4tracking.so
    #4  0x00007ffff449c472 in G4TrackingManager::ProcessOneTrack(G4Track*) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4tracking.so
    #5  0x00007ffff46d3389 in G4EventManager::DoProcessing(G4Event*) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4event.so
    #6  0x00007ffff496ea6f in G4RunManager::DoEventLoop(int, char const*, int) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4run.so
    #7  0x00007ffff496c53e in G4RunManager::BeamOn(int, char const*, int) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4run.so
    #8  0x0000000000418e6d in main (argc=1, argv=0x7fffffff5c98) at /data/blyth/junotop/opticks/u4/tests/U4RecorderTest.cc:208
    (gdb) 


g4-cls G4ParticleChangeForTransport::

    200 
    201 #ifdef G4VERBOSE
    202   if (debugFlag) CheckIt(*aTrack);
    203 #endif
    204 
    205   //  Update the G4Step specific attributes
    206   //pStep->SetStepLength( theTrueStepLength );
    207   //  pStep->AddTotalEnergyDeposit( theLocalEnergyDeposit );
    208   pStep->SetControlFlag( theSteppingControlFlag );
    209   return pStep;
    210   //  return UpdateStepInfo(pStep);
    211 }

g4-cls G4VParticleChange::

    300     // CheckIt method is activated 
    301     // if debug flag is set and 'G4VERBOSE' is defined 
    302     void   ClearDebugFlag();
    303     void   SetDebugFlag();
    304     G4bool GetDebugFlag() const;
    305 
    306   protected:
    307     // CheckSecondary method is provided for debug
    308     G4bool CheckSecondary(G4Track&);
    309 
    310     G4double GetAccuracyForWarning() const;
    311     G4double GetAccuracyForException() const;
    312 
    313   protected:
    314     G4bool   debugFlag;
    315 
    316     // accuracy levels
    317     static const G4double accuracyForWarning;
    318     static const G4double accuracyForException;
    319 
    320 
    321 };

    289 inline
    290  void G4VParticleChange::ClearDebugFlag()
    291 {
    292   debugFlag = false;
    293 }
    294 
    295 inline
    296  void G4VParticleChange::SetDebugFlag()
    297 {
    298   debugFlag = true;
    299 }
    300 
    301 inline
    302  G4bool G4VParticleChange::GetDebugFlag() const
    303 {
    304   return debugFlag;
    305 }





    (gdb) c
    Continuing.
      G4ParticleChange::CheckIt  : the Momentum Change is not unit vector !!  Difference:  5.61912e-08
    opticalphoton E=2.47473e-06 pos=-11.4378, 9.03553, 11.3013
          -----------------------------------------------
            G4ParticleChange Information  
          -----------------------------------------------
            # of 2ndaries       :                    0
          -----------------------------------------------
            Energy Deposit (MeV):                    0
            Non-ionizing Energy Deposit (MeV):                    0
            Track Status        :                Alive
            True Path Length (mm) :                  790
            Stepping Control      :                    0
            Mass (GeV)   :                    0
            Charge (eplus)   :                    0
            MagneticMoment   :                    0
                    :  =                    0*[e hbar]/[2 m]
            Position - x (mm)   :            -1.19e+04
            Position - y (mm)   :             9.42e+03
            Position - z (mm)   :             1.18e+04
            Time (ns)           :                 3.65
            Proper Time (ns)    :                    0
            Momentum Direct - x :               -0.621
            Momentum Direct - y :                 0.49
            Momentum Direct - z :                0.611
            Kinetic Energy (MeV):             2.47e-06
            Velocity  (/c):                    1
            Polarization - x    :               -0.503
            Polarization - y    :               -0.848
            Polarization - z    :                0.168
            Touchable (pointer) :                    0

    Breakpoint 1, 0x00007ffff2379b10 in G4ParticleChange::CheckIt(G4Track const&) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4track.so
    (gdb) bt
    #0  0x00007ffff2379b10 in G4ParticleChange::CheckIt(G4Track const&) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4track.so
    #1  0x00007ffff23774d6 in G4ParticleChange::UpdateStepForPostStep(G4Step*) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4track.so
    #2  0x00007ffff449310a in G4SteppingManager::InvokePSDIP(unsigned long) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4tracking.so
    #3  0x00007ffff449356b in G4SteppingManager::InvokePostStepDoItProcs() () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4tracking.so
    #4  0x00007ffff4490d3d in G4SteppingManager::Stepping() () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4tracking.so
    #5  0x00007ffff449c472 in G4TrackingManager::ProcessOneTrack(G4Track*) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4tracking.so
    #6  0x00007ffff46d3389 in G4EventManager::DoProcessing(G4Event*) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4event.so
    #7  0x00007ffff496ea6f in G4RunManager::DoEventLoop(int, char const*, int) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4run.so
    #8  0x00007ffff496c53e in G4RunManager::BeamOn(int, char const*, int) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4run.so
    #9  0x0000000000418e6d in main (argc=1, argv=0x7fffffff5c98) at /data/blyth/junotop/opticks/u4/tests/U4RecorderTest.cc:208
    (gdb) 


::

    epsilon:opticks blyth$ g4-cc ClearDebugFlag
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/processes/electromagnetic/muons/src/G4ErrorEnergyLoss.cc:  aParticleChange.ClearDebugFlag();



::

    gx
    ./ab.sh 


Sizable input photon difference when using "SEvt_setFrame_WIDE_INPUT_PHOTON"::

    cfbase:/usr/local/opticks/geocache/OKX4Test_lWorld0x5780b30_PV_g4live/g4ok_gltf/5303cd587554cb16682990189831ae83/1/CSG_GGeo 
    -------- after Fold.Load
    max_starts:11 max_slots:29
    max_starts:13 max_slots:29
    -------- after XFold
    im = np.abs(a.inphoton - b.inphoton).max()         : 0.0004882713565166341



