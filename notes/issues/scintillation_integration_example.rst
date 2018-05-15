
Scintillation Integration Example
====================================


Request
---------

::

    Hi Simon, 

    I am a post doc at McGill university. I am working on some detector optical
    simulations and would like to implement Opticks into our Geant4 simulation.
    Right now I am just wanting to get it working with the LXe example in Geant. I
    noticed that you already had a geometry file for this example. I was was
    wondering if you had examples of how you modified the Geant scintillation
    process with the CUDA code in scintillationstep.h? Any information would help. 

    Thanks,
    Thomas McElroy



::

    Hi Thomas,

    I’m happy that you are interested in getting to know Opticks.   
    What experiment are you aiming to use it with ?

    Opticks/Geant4 integrations have so far been applied to 
    detector specific (Dayabay and JUNO) customizations of 
    G4Scintillation and G4Cerenkov steered from detector specific 
    manager classes. There is no off the shelf example currently.   

    Creating an integration example is however on my short-term roadmap. 
    I will let you know when I have something for you to test.

    Process customizations are strait forward , just replacing the photon 
    generation loop with the collection of “genstep” parameters   (okop/OpMgr::addGenstep), 
    which includes  the number of photons to generate for the step, 
    and porting the generation inner loop to the GPU, as you saw in scintillationstep.h

    What is missing currently:

      - reusable steering for managing geometry, gensteps and hits  
        (all the underpinnings exist already : what is missing is the high level 
         steering code,  that needs to be extracted from simulation framework 
         specific code) 

      - CMake config setup to enable find_package(Opticks) following 
        the pattern of Geant4 examples like LXe  

    Simon



Original Approach used in Chroma days
------------------------------------------

JUNO integration done by Tao, so need to await his return to 
be certain of finding state-of-art. But it is not going to be
much different from my original for Dayabay::


    epsilon:~ blyth$ find dybgaudi -name '*.cc' | grep Scintillation 
    dybgaudi/Simulation/DetSimChroma/src/DsChromaG4Scintillation.cc
    dybgaudi/Simulation/DetSim/src/DsG4Scintillation.cc
    dybgaudi/Utilities/G4DAEChroma/src/G4DAEScintillationStepList.cc
    dybgaudi/Utilities/G4DAEChroma/src/G4DAEScintillationStep.cc
    epsilon:~ blyth$ 



Geant4 LXe example using canned scintillation ?
----------------------------------------------------

::

    epsilon:geant4_10_02_p01 blyth$ g4-cls G4Scintillation
    vi -R source/processes/electromagnetic/xrays/include/G4Scintillation.hh source/processes/electromagnetic/xrays/src/G4Scintillation.cc
    2 files to edit



::

    472 void LXeDetectorConstruction::SetMainScintYield(G4double y) {
    473   fLXe_mt->AddConstProperty("SCINTILLATIONYIELD",y/MeV);
    474 }
    475 
    476 //....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
    477 
    478 void LXeDetectorConstruction::SetWLSScintYield(G4double y) {
    479   fMPTPStyrene->AddConstProperty("SCINTILLATIONYIELD",y/MeV);
    480 }





JUNO Integration 
------------------

* http://juno.ihep.ac.cn/trac/browser/offline/trunk/Simulation/DetSimV2/G4Opticks/src/G4OpticksAnaMgr.hh



dybgaudi/Simulation/DetSimChroma/src/DsChromaG4Scintillation.cc
------------------------------------------------------------------

Hmm the below is very old code, where is the latest ?::

     579 #ifdef G4DAECHROMA_COLLECT_STEPS
     580         {
     581             //
     582             // serialize DsChromaG4Scintillation::PostStepDoIt stack, just before the photon loop
     583             // by directly G4DAEArray intems using (n,?,4) structure [float4 quads are efficient on GPU]
     584             //
     585             G4DAEChroma* chroma = G4DAEChroma::GetG4DAEChroma();
     586             G4DAEScintillationStepList* ssl = chroma->GetScintillationStepList();
     587             int* g2c = chroma->GetMaterialLookup();
     588 
     589             // this relates Geant4 materialIndex to the chroma equivalent
     590             G4int chromaMaterialIndex = g2c[materialIndex] ;
     591             G4String materialName = aMaterial->GetName();
     592 
     593             size_t ssid = 1 + ssl->GetCount() ;  // 1-based 
     594             float* ss = ssl->GetNextPointer();
     595 
     596             const G4ParticleDefinition* definition = aParticle->GetDefinition();
     597             G4ThreeVector deltaPosition = aStep.GetDeltaPosition();
     598 
     599             /*
     600             cout << "G4DAEScintillationStep " 
     601                  << " ssid " << ssid 
     602                  << " materialIndex " << materialIndex
     603                  << " chromaMaterialIndex " << chromaMaterialIndex
     604                  <<  materialName " << materialName
     605                  << " PDGEncoding " << definition->GetPDGEncoding() 
     606                  << " Num " << Num 
     607                  << endl ;
     608             */
     609 
     610             assert(chromaMaterialIndex > -1 );
     611 
     612             uif_t uifa[4] ;
     613             uifa[0].i = ssid ;  // > 0 for Scintillation
     614             uifa[1].i = aTrack.GetTrackID() ;
     615             uifa[2].i = chromaMaterialIndex ;
     616             uifa[3].i = Num ;
     617 
     618             uif_t uifb[4] ;
     619             uifb[0].i = definition->GetPDGEncoding();
     620             uifb[1].i = scnt ;   // 1:fast 2:slow
     621             uifb[2].i = 0 ;
     622             uifb[3].i = 0 ;
     623 
     624             ss[G4DAEScintillationStep::_Id]         =  uifa[0].f ;
     625             ss[G4DAEScintillationStep::_ParentID]   =  uifa[1].f ;
     626             ss[G4DAEScintillationStep::_Material]   =  uifa[2].f ;
     627             ss[G4DAEScintillationStep::_NumPhotons] =  uifa[3].f ;
     628 
     629             ss[G4DAEScintillationStep::_x0_x] = x0.x() ;
     630             ss[G4DAEScintillationStep::_x0_y] = x0.y() ;
     631             ss[G4DAEScintillationStep::_x0_z] = x0.z() ;
     632             ss[G4DAEScintillationStep::_t0] = t0 ;
     633 
     634             ss[G4DAEScintillationStep::_DeltaPosition_x] = deltaPosition.x();
     635             ss[G4DAEScintillationStep::_DeltaPosition_y] = deltaPosition.y();
     636             ss[G4DAEScintillationStep::_DeltaPosition_z] = deltaPosition.z();
     637             ss[G4DAEScintillationStep::_step_length]     = aStep.GetStepLength() ;
     638 
     639             ss[G4DAEScintillationStep::_code]      =  uifb[0].f ;
     640             ss[G4DAEScintillationStep::_charge]    =  definition->GetPDGCharge();
     641             ss[G4DAEScintillationStep::_weight]    =  weight ;
     642             ss[G4DAEScintillationStep::_MeanVelocity] = ((pPreStepPoint->GetVelocity()+ pPostStepPoint->GetVelocity())/2.);
     643 
     644             ss[G4DAEScintillationStep::_scnt]      =  uifb[1].f ;
     645             ss[G4DAEScintillationStep::_slowerRatio]  =  slowerRatio ;
     646             ss[G4DAEScintillationStep::_slowTimeConstant]  =  slowTimeConstant ;
     647             ss[G4DAEScintillationStep::_slowerTimeConstant]  =  slowerTimeConstant ;
     648 
     649             ss[G4DAEScintillationStep::_ScintillationTime]  = ScintillationTime ;
     650             ss[G4DAEScintillationStep::_ScintillationIntegralMax]  = ScintillationIntegral->GetMaxValue() ;
     651             ss[G4DAEScintillationStep::_Spare1]  = 0. ;
     652             ss[G4DAEScintillationStep::_Spare2]  = 0. ;
     653 
     654        }
     655 #endif
     656    
     657 #ifdef G4DAECHROMA_COLLECT_PHOTONS
     658         for (G4int i = 0; i < Num; i++) { //Num is # of 2ndary tracks now
     659         // Determine photon energy





