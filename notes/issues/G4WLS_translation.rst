G4WLS Translation
====================



Geant4 Sources for WLS process
--------------------------------

::

    epsilon:geant4.10.04.p02 blyth$ find source -name *WLS*.*
    source/processes/optical/include/G4VWLSTimeGeneratorProfile.hh
    source/processes/optical/include/G4WLSTimeGeneratorProfileExponential.hh
    source/processes/optical/include/G4OpWLS.hh
    source/processes/optical/include/G4WLSTimeGeneratorProfileDelta.hh
    source/processes/optical/src/G4WLSTimeGeneratorProfileExponential.cc
    source/processes/optical/src/G4VWLSTimeGeneratorProfile.cc
    source/processes/optical/src/G4WLSTimeGeneratorProfileDelta.cc
    source/processes/optical/src/G4OpWLS.cc


    epsilon:geant4.10.04.p02 blyth$ find source -type f -exec grep -H G4OpWLS {} \;
    source/physics_lists/constructors/electromagnetic/src/G4OpticalPhysics.cc:#include "G4OpWLS.hh"
    source/physics_lists/constructors/electromagnetic/src/G4OpticalPhysics.cc:    void buildCommands( G4OpWLS* op )
    source/physics_lists/constructors/electromagnetic/src/G4OpticalPhysics.cc:        G4GenericMessenger::Command& wlscmd1 = mess->DeclareMethod("setTimeProfile",&G4OpWLS::UseTimeProfile,
    source/physics_lists/constructors/electromagnetic/src/G4OpticalPhysics.cc:  G4OpWLS* OpWLSProcess = new G4OpWLS();
    source/processes/optical/include/G4OpWLS.hh:// $Id: G4OpWLS.hh 85354 2014-10-28 09:58:12Z gcosmo $
    source/processes/optical/include/G4OpWLS.hh:// File:        G4OpWLS.hh
    source/processes/optical/include/G4OpWLS.hh:#ifndef G4OpWLS_h
    ...


::

     g4-cls G4OpticalProcessIndex

     50 
     51 enum G4OpticalProcessIndex {
     52   kCerenkov,      ///< Cerenkov process index
     53   kScintillation, ///< Scintillation process index
     54   kAbsorption,    ///< Absorption process index
     55   kRayleigh,      ///< Rayleigh scattering process index
     56   kMieHG,         ///< Mie scattering process index
     57   kBoundary,      ///< Boundary process index
     58   kWLS,           ///< Wave Length Shifting process index
     59   kNoProcess      ///< Number of processes, no selected process
     60 };
     61 
     62 /// Return the name for a given optical process index
     63 G4String G4OpticalProcessName(G4int );
     64 

::

    g4-cls G4OpticalPhysics


    278   for ( G4int i=0; i<kNoProcess; i++ ) OpProcesses.push_back(NULL);
    279 
    280   // Add Optical Processes
    281 
    282   G4OpAbsorption* OpAbsorptionProcess  = new G4OpAbsorption();
    283   UIhelpers::buildCommands(OpAbsorptionProcess,DIR_CMDS"/absorption/",GUIDANCE" for absorption process");
    284   OpProcesses[kAbsorption] = OpAbsorptionProcess;
    285 
    286   G4OpRayleigh* OpRayleighScatteringProcess = new G4OpRayleigh();
    287   UIhelpers::buildCommands(OpRayleighScatteringProcess,DIR_CMDS"/rayleigh/",GUIDANCE" for Reyleigh scattering process");
    288   OpProcesses[kRayleigh] = OpRayleighScatteringProcess;
    289    
    290   G4OpMieHG* OpMieHGScatteringProcess = new G4OpMieHG();
    291   UIhelpers::buildCommands(OpMieHGScatteringProcess,DIR_CMDS"/mie/",GUIDANCE" for Mie cattering process");
    292   OpProcesses[kMieHG] = OpMieHGScatteringProcess;
    293 
    294   G4OpBoundaryProcess* OpBoundaryProcess = new G4OpBoundaryProcess();
    295   UIhelpers::buildCommands(OpBoundaryProcess,DIR_CMDS"/boundary/",GUIDANCE" for boundary process");
    296   OpBoundaryProcess->SetInvokeSD(fInvokeSD);
    297   OpProcesses[kBoundary] = OpBoundaryProcess;
    298 
    299   G4OpWLS* OpWLSProcess = new G4OpWLS();
    300   OpWLSProcess->UseTimeProfile(fProfile);
    301   UIhelpers::buildCommands(OpWLSProcess);
    302   OpProcesses[kWLS] = OpWLSProcess;
    303 
    304   G4ProcessManager * pManager = 0;
    305   pManager = G4OpticalPhoton::OpticalPhoton()->GetProcessManager();
    306 
    307   if (!pManager) {
    308      std::ostringstream o;
    309      o << "Optical Photon without a Process Manager";
    310      G4Exception("G4OpticalPhysics::ConstructProcess()","",
    311                   FatalException,o.str().c_str());
    312      return;
    313   }
    314 
    315   for ( G4int i=kAbsorption; i<=kWLS; i++ ) {
    316       if ( fProcessUse[i] ) {
    317          pManager->AddDiscreteProcess(OpProcesses[i]);
    318       }
    319   }
    320 
    321   G4Scintillation* ScintillationProcess = new G4Scintillation();
    322   ScintillationProcess->SetScintillationYieldFactor(fYieldFactor);
    323   ScintillationProcess->SetScintillationExcitationRatio(fExcitationRatio);
    324   ScintillationProcess->SetFiniteRiseTime(fFiniteRiseTime);





::

    g4-cls G4OpWLS     ## 1 opticalphoton absorbed, 0,1,... opticalphoton secondaries added

    g4-cls G4VWLSTimeGeneratorProfile              ## pure virtual base 
    g4-cls G4WLSTimeGeneratorProfileExponential    ## time = -std::log(G4UniformRand())*time_constant
    g4-cls G4WLSTimeGeneratorProfileDelta          ## time = time_constant


::


    101 G4VParticleChange*
    102 G4OpWLS::PostStepDoIt(const G4Track& aTrack, const G4Step& aStep)
    103 {
    104   aParticleChange.Initialize(aTrack);
    105  
    106   aParticleChange.ProposeTrackStatus(fStopAndKill);
    107 
    108   if (verboseLevel>0) {
    109     G4cout << "\n** Photon absorbed! **" << G4endl;
    110   }
    111  
    112   const G4Material* aMaterial = aTrack.GetMaterial();
    113 
    114   G4StepPoint* pPostStepPoint = aStep.GetPostStepPoint();
    115    
    116   G4MaterialPropertiesTable* aMaterialPropertiesTable =
    117     aMaterial->GetMaterialPropertiesTable();
    118   if (!aMaterialPropertiesTable)
    119     return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
    120 
    121   const G4MaterialPropertyVector* WLS_Intensity =
    122     aMaterialPropertiesTable->GetProperty(kWLSCOMPONENT);
    123 
    124   if (!WLS_Intensity)
    125     return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);

    ///  WLSCOMPONENT 
    ///        must be present but its value as function of wavelength is not used here, 
    ///        only used in G4OpWLS::BuildPhysicsTable

    126 
    127   G4int NumPhotons = 1;
    128 
    129   if (aMaterialPropertiesTable->ConstPropertyExists("WLSMEANNUMBERPHOTONS")) {

    ///  WLSMEANNUMBERPHOTONS
    ///        appears optional and defaulting to 1 

    130 
    131      G4double MeanNumberOfPhotons = aMaterialPropertiesTable->
    132                                     GetConstProperty(kWLSMEANNUMBERPHOTONS);
    133 
    134      NumPhotons = G4int(G4Poisson(MeanNumberOfPhotons));
    135 
    136      if (NumPhotons <= 0) {
    137         
    138         // return unchanged particle and no secondaries
    139         
    140         aParticleChange.SetNumberOfSecondaries(0);
    141         
    142         return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
    143      
    144      }
    145 
    146   }
    147 
    148   aParticleChange.SetNumberOfSecondaries(NumPhotons);




    ///
    ///  handling NumPhotons = 1 easy on GPU, as can do in same thread  
    ///   

    epsilon:geant4.10.04.p02 blyth$ find examples -type f -exec grep -H WLSMEANNUMBERPHOTONS {} \;
    epsilon:geant4.10.04.p02 blyth$ 



::

    383 G4double G4OpWLS::GetMeanFreePath(const G4Track& aTrack,
    384                          G4double ,
    385                          G4ForceCondition* )
    386 {
    387   const G4DynamicParticle* aParticle = aTrack.GetDynamicParticle();
    388   const G4Material* aMaterial = aTrack.GetMaterial();
    389 
    390   G4double thePhotonEnergy = aParticle->GetTotalEnergy();
    391 
    392   G4MaterialPropertiesTable* aMaterialPropertyTable;
    393   G4MaterialPropertyVector* AttenuationLengthVector;
    394    
    395   G4double AttenuationLength = DBL_MAX;
    396 
    397   aMaterialPropertyTable = aMaterial->GetMaterialPropertiesTable();
    398 
    399   if ( aMaterialPropertyTable ) {
    400     AttenuationLengthVector = aMaterialPropertyTable->
    401       GetProperty(kWLSABSLENGTH);
    402     if ( AttenuationLengthVector ){
    403       AttenuationLength = AttenuationLengthVector->
    404     Value(thePhotonEnergy);
    405     }
    406     else {
    407       //             G4cout << "No WLS absorption length specified" << G4endl;
    408     }
    409   }
    410   else {
    411     //           G4cout << "No WLS absortion length specified" << G4endl;
    412   }
    413  
    414   return AttenuationLength;
    415 }




::

    284 // BuildPhysicsTable for the wavelength shifting process
    285 // --------------------------------------------------
    286 
    287 void G4OpWLS::BuildPhysicsTable(const G4ParticleDefinition&)
    288 {
    289   if (theIntegralTable) {
    290      theIntegralTable->clearAndDestroy();
    291      delete theIntegralTable;
    292      theIntegralTable = NULL;
    293   }
    294 
    295   const G4MaterialTable* theMaterialTable =
    296     G4Material::GetMaterialTable();
    297   G4int numOfMaterials = G4Material::GetNumberOfMaterials();
    298  
    299   // create new physics table
    300  
    301   theIntegralTable = new G4PhysicsTable(numOfMaterials);
    302  
    303   // loop for materials
    304 
    305   for (G4int i=0 ; i < numOfMaterials; i++)
    306     {
    307       G4PhysicsOrderedFreeVector* aPhysicsOrderedFreeVector =
    308     new G4PhysicsOrderedFreeVector();
    309      
    310       // Retrieve vector of WLS wavelength intensity for
    311       // the material from the material's optical properties table.
    312 
    313       G4Material* aMaterial = (*theMaterialTable)[i];
    314 
    315       G4MaterialPropertiesTable* aMaterialPropertiesTable =
    316     aMaterial->GetMaterialPropertiesTable();
    317 
    318       if (aMaterialPropertiesTable) {
    319 
    320     G4MaterialPropertyVector* theWLSVector =
    321       aMaterialPropertiesTable->GetProperty(kWLSCOMPONENT);
    322 
    323     if (theWLSVector) {
    324       
    325       // Retrieve the first intensity point in vector
    326       // of (photon energy, intensity) pairs
    327       
    328       G4double currentIN = (*theWLSVector)[0];
    329       
    330       if (currentIN >= 0.0) {
    331         
    332         // Create first (photon energy) 



Geant4 WLS Examples
----------------------

examples/extended/optical/LXe/src/LXeDetectorConstruction.cc::

    091 void LXeDetectorConstruction::DefineMaterials(){
    ...
    131   //Fiber(PMMA)
    132   fPMMA = new G4Material("PMMA", density=1190*kg/m3,3);
    133   fPMMA->AddElement(fH,nH_PMMA);
    134   fPMMA->AddElement(fC,nC_PMMA);
    135   fPMMA->AddElement(fO,2);
    ...
    212   G4double RefractiveIndexFiber[]={ 1.60, 1.60, 1.60, 1.60};
    213   assert(sizeof(RefractiveIndexFiber) == sizeof(wls_Energy));
    214   G4double AbsFiber[]={9.00*m,9.00*m,0.1*mm,0.1*mm};
    215   assert(sizeof(AbsFiber) == sizeof(wls_Energy));
    216   G4double EmissionFib[]={1.0, 1.0, 0.0, 0.0};
    217   assert(sizeof(EmissionFib) == sizeof(wls_Energy));
    218   G4MaterialPropertiesTable* fiberProperty = new G4MaterialPropertiesTable();
    219   fiberProperty->AddProperty("RINDEX",wls_Energy,RefractiveIndexFiber,wlsnum);
    220   fiberProperty->AddProperty("WLSABSLENGTH",wls_Energy,AbsFiber,wlsnum);
    221   fiberProperty->AddProperty("WLSCOMPONENT",wls_Energy,EmissionFib,wlsnum);
    222   fiberProperty->AddConstProperty("WLSTIMECONSTANT", 0.5*ns);
    223   fPMMA->SetMaterialPropertiesTable(fiberProperty);
    224 



examples/extended/optical/LXe/src/LXeWLSFiber.cc::

     42 LXeWLSFiber::LXeWLSFiber(G4RotationMatrix *pRot,
     43                              const G4ThreeVector &tlate,
     44                              G4LogicalVolume *pMotherLogical,
     45                              G4bool pMany,
     46                              G4int pCopyNo,
     47                              LXeDetectorConstruction* c)
     48   :G4PVPlacement(pRot,tlate,
     49                  new G4LogicalVolume(new G4Box("temp",1,1,1),
     50                                      G4Material::GetMaterial("Vacuum"),
     51                                      "temp",0,0,0),
     52                  "Cladding2",pMotherLogical,pMany,pCopyNo),fConstructor(c)
     53 { 
     54   CopyValues();
     55   
     56   // The Fiber
     57   //
     58   G4Tubs* fiber_tube =
     59    new G4Tubs("Fiber",fFiber_rmin,fFiber_rmax,fFiber_z,fFiber_sphi,fFiber_ephi);
     60   
     61   G4LogicalVolume* fiber_log =
     62       new G4LogicalVolume(fiber_tube,G4Material::GetMaterial("PMMA"),
     63                           "Fiber",0,0,0);
     64   
     65   // Cladding (first layer)
     66   //
     67   G4Tubs* clad1_tube =
     68       new G4Tubs("Cladding1",fClad1_rmin,fClad1_rmax,fClad1_z,fClad1_sphi,
     69                  fClad1_ephi);
     70   
     71   G4LogicalVolume* clad1_log =
     72       new G4LogicalVolume(clad1_tube,G4Material::GetMaterial("Pethylene1"),
     73                           "Cladding1",0,0,0);
     74   







::

    epsilon:geant4.10.04.p02 blyth$ find examples -name '*.cc' -exec grep -l WLS {} \;
    examples/extended/field/field04/src/F04PhysicsList.cc
    examples/extended/optical/LXe/src/LXeWLSSlab.cc
    examples/extended/optical/LXe/src/LXeSteppingAction.cc
    examples/extended/optical/LXe/src/LXeTrajectory.cc
    examples/extended/optical/LXe/src/LXeDetectorConstruction.cc
    examples/extended/optical/LXe/src/LXePhysicsList.cc
    examples/extended/optical/LXe/src/LXeDetectorMessenger.cc
    examples/extended/optical/LXe/src/LXeWLSFiber.cc
    examples/extended/optical/LXe/src/LXeTrackingAction.cc
    examples/extended/optical/wls/wls.cc
    examples/extended/optical/wls/src/WLSStackingAction.cc
    examples/extended/optical/wls/src/WLSExtraPhysics.cc
    examples/extended/optical/wls/src/WLSSteppingActionMessenger.cc
    examples/extended/optical/wls/src/WLSPhysicsList.cc
    examples/extended/optical/wls/src/WLSRunActionMessenger.cc
    examples/extended/optical/wls/src/WLSTrajectoryPoint.cc
    examples/extended/optical/wls/src/WLSTrackingAction.cc
    examples/extended/optical/wls/src/WLSPrimaryGeneratorMessenger.cc
    examples/extended/optical/wls/src/WLSRunAction.cc
    examples/extended/optical/wls/src/WLSPhysicsListMessenger.cc
    examples/extended/optical/wls/src/WLSOpticalPhysics.cc
    examples/extended/optical/wls/src/WLSSteppingVerbose.cc
    examples/extended/optical/wls/src/WLSEventActionMessenger.cc
    examples/extended/optical/wls/src/WLSMaterials.cc
    examples/extended/optical/wls/src/WLSStepMax.cc
    examples/extended/optical/wls/src/WLSEventAction.cc
    examples/extended/optical/wls/src/WLSDetectorConstruction.cc
    examples/extended/optical/wls/src/WLSActionInitialization.cc
    examples/extended/optical/wls/src/WLSSteppingAction.cc
    examples/extended/optical/wls/src/WLSPhotonDetHit.cc
    examples/extended/optical/wls/src/WLSPrimaryGeneratorAction.cc
    examples/extended/optical/wls/src/WLSPhotonDetSD.cc
    examples/extended/optical/wls/src/WLSUserTrackInformation.cc
    examples/extended/optical/wls/src/WLSTrajectory.cc
    examples/extended/optical/wls/src/WLSDetectorMessenger.cc
    epsilon:geant4.10.04.p02 blyth$ 



How to add WLS to Opticks
--------------------------

Adding support for WLS to Opticks requires :

1. adding WLSABSLENGTH to the standard material props and getting it thru into the GPU texture 
2. using the wlsabsorption_length to give wlsabsorption_distance in propagate.h:propagate_to_boundary
3. during geometry translation assert that WLSMEANNUMBERPHOTONS is not present or has value of 1
4. ggeo/GWLSLib analogous to ggeo/GScintillatorLib that collects WLS materials and prepares the icdf buffer (equiv to BuildPhysicsTable)
5. optixrap/OWLSLib analogous to optixrap/OScintillatorLib that converts the buffer from GWLSLib into a GPU texture
6. optixrap/cu/wavelength_lookup.h  wls_lookup similar to reemission_lookup 

   * do you have several different WLS materials, or just the one ? 


WLS : does NOT need its own genstep
--------------------------------------

* scintillation and cerenkov : other particles -> opticalphotons
* WLS : opticalphotons -> opticalphotons 

* gensteps are the connection between G4Cerenkov and G4Scintillation and GPU generation loops
* WLS process needs to be entirely GPU implemented 


Re-emitting more than one photon for each absorbed photon 

* drastically more difficult as breaks the use of a single CUDA thread to handle a one photon.



Opticks propagate : needs to compare wlsabsorption_distance with absorption_distance and scattering_distance 
----------------------------------------------------------------------------------------------------------------

optixrap/cu/propagate.h::

    078 __device__ int propagate_to_boundary( Photon& p, State& s, curandState &rng)
     79 {
     80     //float speed = SPEED_OF_LIGHT/s.material1.x ;    // .x:refractive_index    (phase velocity of light in medium)
     81     float speed = s.m1group2.x ;  // .x:group_velocity  (group velocity of light in the material) see: opticks-find GROUPVEL
     82 
     83 #ifdef WITH_ALIGN_DEV
     84 #ifdef WITH_LOGDOUBLE
     85 
     86     float u_boundary_burn = curand_uniform(&rng) ;
     87     float u_scattering = curand_uniform(&rng) ;
     88     float u_absorption = curand_uniform(&rng) ;
     89 
     90     //  these two doubles brings about 100 lines of PTX with .f64
     91     //  see notes/issues/AB_SC_Position_Time_mismatch.rst      
     92     float scattering_distance = -s.material1.z*log(double(u_scattering)) ;   // .z:scattering_length
     93     float absorption_distance = -s.material1.y*log(double(u_absorption)) ;   // .y:absorption_length 
     94 
     95 #elif WITH_LOGDOUBLE_ALT
     96     float u_boundary_burn = curand_uniform(&rng) ;
     97     double u_scattering = curand_uniform_double(&rng) ;
     98     double u_absorption = curand_uniform_double(&rng) ;
     99 
    100     float scattering_distance = -s.material1.z*log(u_scattering) ;   // .z:scattering_length
    101     float absorption_distance = -s.material1.y*log(u_absorption) ;   // .y:absorption_length 
    102 
    103 #else
    104     float u_boundary_burn = curand_uniform(&rng) ;
    105     float u_scattering = curand_uniform(&rng) ;
    106     float u_absorption = curand_uniform(&rng) ;
    107     float scattering_distance = -s.material1.z*logf(u_scattering) ;   // .z:scattering_length
    108     float absorption_distance = -s.material1.y*logf(u_absorption) ;   // .y:absorption_length 
    109 #endif
    110 
    111 #else
    112     float scattering_distance = -s.material1.z*logf(curand_uniform(&rng));   // .z:scattering_length
    113     float absorption_distance = -s.material1.y*logf(curand_uniform(&rng));   // .y:absorption_length
    114 #endif
    115 
    116 #ifdef WITH_ALIGN_DEV_DEBUG
    117     rtPrintf("propagate_to_boundary  u_OpBoundary:%.9g speed:%.9g s.distance_to_boundary:%.9g \n", u_boundary_burn, speed, s.distance_to_boundary );
    118     rtPrintf("propagate_to_boundary  u_OpRayleigh:%.9g   scattering_length(s.material1.z):%.9g scattering_distance:%.9g \n", u_scattering, s.material1.z, scattering_distance );
    119     rtPrintf("propagate_to_boundary  u_OpAbsorption:%.9g   absorption_length(s.material1.y):%.9g absorption_distance:%.9g \n", u_absorption, s.material1.y, absorption_distance );
    120 #endif
    121 
    122 
    123     if (absorption_distance <= scattering_distance)
    124     {
    125         if (absorption_distance <= s.distance_to_boundary)
    126         {
    127             p.time += absorption_distance/speed ;
    128             p.position += absorption_distance*p.direction;





Need to get WLS properties into the system
---------------------------------------------

Need a GWLSLib that collects WLS materials and cooks up the icdf buffer, equivalent to the G4OpWLS::BuildPhysicsTable::


    125 NPY<float>* GScintillatorLib::createBuffer()
    126 {
    127     unsigned int ni = getNumRaw();
    128     unsigned int nj = m_icdf_length ;
    129     unsigned int nk = 1 ;
    130 
    131     LOG(LEVEL)
    132           << " ni " << ni
    133           << " nj " << nj
    134           << " nk " << nk
    135           ;
    136 
    137     NPY<float>* buf = NPY<float>::make(ni, nj, nk);
    138     buf->zero();
    139     float* data = buf->getValues();
    140 
    141     for(unsigned int i=0 ; i < ni ; i++)
    142     {
    143         GPropertyMap<float>* scint = getRaw(i) ;
    144         GProperty<float>* cdf = constructReemissionCDF(scint);
    145         assert(cdf);
    146 
    147         GProperty<float>* icdf = constructInvertedReemissionCDF(scint);
    148         assert(icdf);
    149         assert(icdf->getLength() == nj);
    150 
    151         for( unsigned int j = 0; j < nj ; ++j )
    152         {
    153             unsigned int offset = i*nj*nk + j*nk ;
    154             data[offset+0] = icdf->getValue(j);
    155         }
    156    }
    157    return buf ;
    158 }

::

    195 GProperty<float>* GScintillatorLib::constructInvertedReemissionCDF(GPropertyMap<float>* pmap)
    196 {
    197     std::string name = pmap->getShortNameString();
    198 
    199     typedef GProperty<float> P ;
    200 
    201     P* slow = getProperty(pmap, slow_component);
    202     P* fast = getProperty(pmap, fast_component);
    203     assert(slow != NULL && fast != NULL );
    204 
    205 
    206     float mxdiff = GProperty<float>::maxdiff(slow, fast);
    207     assert(mxdiff < 1e-6 );
    208 
    209     P* rrd = slow->createReversedReciprocalDomain();    // have to used reciprocal "energywise" domain for G4/NuWa agreement
    210 
    211     P* srrd = rrd->createZeroTrimmed();                 // trim extraneous zero values, leaving at most one zero at either extremity
    212 
    213     unsigned int l_srrd = srrd->getLength() ;
    214     unsigned int l_rrd = rrd->getLength()  ;
    215 
    216     if( l_srrd != l_rrd - 2)
    217     {
    218        LOG(debug)
    219            << "was expecting to trim 2 values "
    220            << " l_srrd " << l_srrd
    221            << " l_rrd " << l_rrd
    222            ;
    223     }
    224     //assert( l_srrd == l_rrd - 2); // expect to trim 2 values
    225 
    226     P* rcdf = srrd->createCDF();
    227 
    228     P* icdf = rcdf->createInverseCDF(m_icdf_length);
    229 
    230     icdf->getValues()->reciprocate();  // avoid having to reciprocate lookup results : by doing it here 
    231 
    232     return icdf ;
    233 }


To understand how inverse CDF is created play around with::

    ggeo/tests/GPropertyTest.cc
    ggeo/tests/GPropertyDebugTest.cc





TBP : tetraphenyl butadiene 
------------------------------


* https://www.science.gov/topicpages/t/tetraphenyl+butadiene+tpb

* https://arxiv.org/abs/1709.05002

  Measurements of the intrinsic quantum efficiency and absorption length of tetraphenyl butadiene thin films in the vacuum ultraviolet regime


::

    As shown in Fig. 1, the scintillation wavelengths can range from 175 nm for
    Xenon down to near 80 nm for Helium and Neon. Light of these wavelengths is
    strongly absorbed by most materials, including those commonly used for
    optical windows. Many experiments sidestep the issue of directly detecting
    VUV light though the use of wavelength shifting (WLS) films which absorb the
    VUV light and re-emit photons, typically in the visible spectrum. The visible
    photons can then easily be detected using photomultiplier tubes (PMTs).



* thin films sufficient, how thin ? 
* are multiple subsequent absorbs and re-emits within the film important ? 


photons absorbed vs photons re-emitted::

    The WLSE as defined in [16] is a “black- box” definition of the efficiency that
    includes both the intrinsic QE of the TPB as well as certain optical
    properties of the TPB film and substrate. The resulting quantity is the effi-
    ciency that a photon absorbed by the sample is reemitted from the sample. This
    measurement is thus sample dependent, including effects of scattering and
    absorption, and cannot be directly applied to other apparatus.


    Fig. 14 The measured reemission spectra of a 1.8 μm TPB film for several
    incident wavelengths. No dependence of the reemission spectrum of TPB on
    incident wavelength was observed





