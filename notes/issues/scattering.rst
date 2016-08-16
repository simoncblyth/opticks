Scattering : GdLS missing RAYLEIGH
=====================================

Argh this is flaky ... some MPT are corrupted ?

Somehow the attempt to use physics table persisting and loading for fast 
start results in a double "OpRayleigh::BuildPhysicsTable" with the 
second one with crazy properties.

Removing the lines from ggv-g4gun seem to avoid the problem with corrupt rayleigh scattering props::

    /run/particle/retrievePhysicsTable $phycache

    /run/particle/storePhysicsTable $phycache

But the backwards in time issue remains.





::

   ggv-;ggv-g4gun --dbg



::

    d #1: tid = 0x715313, 0x0000000106826f46 libG4global.dylib`G4PhysicsVector::Value(this=0x000000010d8d4ef0, theEnergy=0.0000025623521640155531, lastIdx=0x00007fff5fbfce78) const + 182 at G4PhysicsVector.cc:506, queue = 'com.apple.main-thread', stop reason = EXC_BAD_ACCESS (code=1, address=0x95376fcf8)
      * frame #0: 0x0000000106826f46 libG4global.dylib`G4PhysicsVector::Value(this=0x000000010d8d4ef0, theEnergy=0.0000025623521640155531, lastIdx=0x00007fff5fbfce78) const + 182 at G4PhysicsVector.cc:506
        frame #1: 0x0000000102fbf75b libG4processes.dylib`G4PhysicsVector::Value(this=0x000000010d8d4ef0, theEnergy=0.0000025623521640155531) const + 43 at G4PhysicsVector.icc:249
        frame #2: 0x00000001042eb2c3 libG4processes.dylib`G4OpRayleigh::GetMeanFreePath(this=0x000000010f4f1170, aTrack=0x000000010f84c8e0, (null)=0, (null)=0x000000010910e408) + 163 at G4OpRayleigh.cc:261
        frame #3: 0x00000001042d8d5c libG4processes.dylib`G4VDiscreteProcess::PostStepGetPhysicalInteractionLength(this=0x000000010f4f1170, track=0x000000010f84c8e0, previousStepSize=0, condition=0x000000010910e408) + 204 at G4VDiscreteProcess.cc:92
        frame #4: 0x0000000102ec3cd0 libG4tracking.dylib`G4VProcess::PostStepGPIL(this=0x000000010f4f1170, track=0x000000010f84c8e0, previousStepSize=0, condition=0x000000010910e408) + 80 at G4VProcess.hh:503
        frame #5: 0x0000000102ec17a0 libG4tracking.dylib`G4SteppingManager::DefinePhysicalStepLength(this=0x000000010910e280) + 304 at G4SteppingManager2.cc:172
        frame #6: 0x0000000102ebe111 libG4tracking.dylib`G4SteppingManager::Stepping(this=0x000000010910e280) + 417 at G4SteppingManager.cc:180
        frame #7: 0x0000000102ed592d libG4tracking.dylib`G4TrackingManager::ProcessOneTrack(this=0x000000010910e240, apValueG4Track=0x000000010f84c8e0) + 1357 at G4TrackingManager.cc:126
        frame #8: 0x0000000102db2e44 libG4event.dylib`G4EventManager::DoProcessing(this=0x000000010910e1b0, anEvent=0x000000010f6c9f00) + 3188 at G4EventManager.cc:185
        frame #9: 0x0000000102db3b2f libG4event.dylib`G4EventManager::ProcessOneEvent(this=0x000000010910e1b0, anEvent=0x000000010f6c9f00) + 47 at G4EventManager.cc:336
        frame #10: 0x0000000102ce0c75 libG4run.dylib`G4RunManager::ProcessOneEvent(this=0x0000000109003060, i_event=0) + 69 at G4RunManager.cc:399
        frame #11: 0x0000000102ce0ab5 libG4run.dylib`G4RunManager::DoEventLoop(this=0x0000000109003060, n_event=1, macroFile=0x0000000000000000, n_select=-1) + 101 at G4RunManager.cc:367
        frame #12: 0x0000000102cdf8e4 libG4run.dylib`G4RunManager::BeamOn(this=0x0000000109003060, n_event=1, macroFile=0x0000000000000000, n_select=-1) + 196 at G4RunManager.cc:273
        frame #13: 0x000000010155ae9d libcfg4.dylib`CG4::propagate(this=0x0000000108721880) + 605 at CG4.cc:180
        frame #14: 0x000000010000d542 CG4Test`main(argc=16, argv=0x00007fff5fbfde28) + 210 at CG4Test.cc:20
        frame #15: 0x00007fff89e755fd libdyld.dylib`start + 1
        frame #16: 0x00007fff89e755fd libdyld.dylib`start + 1
    (lldb) 


::

    (lldb) f 2
    frame #2: 0x00000001042eb2c3 libG4processes.dylib`G4OpRayleigh::GetMeanFreePath(this=0x000000010f4f1170, aTrack=0x000000010f84c8e0, (null)=0, (null)=0x000000010910e408) + 163 at G4OpRayleigh.cc:261
       258                                ((*thePhysicsTable)(material->GetIndex()));
       259    
       260    G4double rsLength = DBL_MAX;
    -> 261    if( rayleigh != NULL ) rsLength = rayleigh->Value( photonMomentum );
       262    return rsLength;
       263  }
       264  
    (lldb) p rayleigh
    (G4PhysicsOrderedFreeVector *) $0 = 0x000000010d8d4ef0
    (lldb) p *rayleigh
    (G4PhysicsOrderedFreeVector) $1 = {
      G4PhysicsVector = {
        type = 155602256
        edgeMin = 2.1988720260157543E-314
        edgeMax = 2.1988714726622309E-314
        numberOfNodes = 4450565408
        dataVector = size=0 {}
        binVector = size=0 {}
        secDerivative = size=0 {}
        useSpline = true
        dBin = 2.1988631802644311E-314
        baseBin = 2.198834967139791E-314
        verboseLevel = 155590864
      }
    }
    (lldb) p photonMomentum
    (G4double) $2 = 0.0000025623521640155531
    (lldb) p 1./photonMomentum
    (double) $3 = 390266.41772490181
    (lldb) 
    (double) $4 = 390266.41772490181
    (lldb) p 0.00123984/photonMomentum
    (double) $5 = 483.8679153520423

    (lldb) p *material
    (const G4Material) $7 = {
      fName = (std::__1::string = "/dd/Materials/GdDopedLS")
      fChemicalFormula = (std::__1::string = "")
      fDensity = 5.368943773533483E+18
      fState = kStateSolid
      fTemp = 273.14999999999998
      fPressure = 632420632.24050415
      maxNbComponents = 12
      fArrayLength = 12
      fNumberOfComponents = 12
      fNumberOfElements = 12
      theElementVector = 0x000000010924a200 size=12
      fMassFractionVector = 0x000000010924a280
      fAtomsVector = 0x000000010924a090
      fMaterialPropertiesTable = 0x000000010d8d3fb0
      fIndexInTable = 45
      VecNbOfAtomsPerVolume = 0x000000010924a330
      TotNbOfAtomsPerVolume = 9.9811164021948743E+19
      TotNbOfElectPerVolume = 2.894761854434188E+20
      fRadlen = 513.67174561270542
      fNuclInterLen = 806.93702999327354
      fIonisation = 0x000000010924a390
      fSandiaTable = 0x000000010924a4b0
      fBaseMaterial = 0x0000000000000000
      fMassOfMolecule = 0
      fMatComponents = size=0 {}



Huh where is RAYLEIGH?::

    (lldb) p *(material->fMaterialPropertiesTable)
    (G4MaterialPropertiesTable) $9 = {
      MPT = size=7 {
        [0] = {
          __cc = {
            first = (std::__1::string = "ABSLENGTH")
            second = 0x000000010d8d4870
          }
          __nc = {
            first = (std::__1::string = "ABSLENGTH")
            second = 0x000000010d8d4870
          }
        }
      }
      MPTC = size=5 {
        [0] = {
          __cc = {
            first = (std::__1::string = "FASTTIMECONSTANT")
            second = 3.6399998664855957
          }
          __nc = {
            first = (std::__1::string = "FASTTIMECONSTANT")
            second = 3.6399998664855957
          }
        }
      }
    }



::
 
    simon:ggeo blyth$ g4-cls G4OpRayleigh
    vi -R source/processes/optical/include/G4OpRayleigh.hh source/processes/optical/src/G4OpRayleigh.cc



::

    (lldb) p (*thePhysicsTable)[0]
    error: call to a function 'std::__1::vector<G4PhysicsVector*, std::__1::allocator<G4PhysicsVector*> >::operator[](unsigned long)' ('_ZNSt3__16vectorIP15G4PhysicsVectorNS_9allocatorIS2_EEEixEm') that is not present in the target
    error: 0 errors parsing expression
    error: The expression could not be prepared to run in the target
    (lldb) p (*thePhysicsTable)(0)
    (G4PhysicsVector *) $8 = 0x00000001092c5210
    (lldb) p *(*thePhysicsTable)(0)
    (G4PhysicsVector) $9 = {
      type = T_G4PhysicsLogVector
      edgeMin = 0.0001
      edgeMax = 10000000
      numberOfNodes = 78
      dataVector = size=78 {
        [0] = 0.000098746535211573539
        [1] = 0.00011643550559780508
        [2] = 0.00013726578229888829
        [3] = 0.00016178784354001203
        [4] = 0.00019073099278030389
        [5] = 0.00022463489133017729
        [6] = 0.00026595734502531338
        [7] = 0.0003026714279642655
        [8] = 0.00032894631812999808
        [9] = 0.00035223680480944263



::

    (lldb) b "G4OpRayleigh::GetMeanFreePath(G4Track const&, double, G4ForceCondition*)" 



Physics table is only created during beamOn, plus the process
has no virtuals so are forced to copy process source to debug.

::

       226  // --------------------------------------------------------
       227  void OpRayleigh::BuildPhysicsTable(const G4ParticleDefinition&)
       228  {
    -> 229    if (thePhysicsTable) {
       230       thePhysicsTable->clearAndDestroy();
       231       delete thePhysicsTable;
       232       thePhysicsTable = NULL;
    (lldb) bt
    * thread #1: tid = 0x732c57, 0x0000000101595b07 libcfg4.dylib`OpRayleigh::BuildPhysicsTable(this=0x000000010d33ee80, (null)=0x000000010904f120) + 23 at OpRayleigh.cc:229, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
      * frame #0: 0x0000000101595b07 libcfg4.dylib`OpRayleigh::BuildPhysicsTable(this=0x000000010d33ee80, (null)=0x000000010904f120) + 23 at OpRayleigh.cc:229
        frame #1: 0x0000000102d4d68f libG4run.dylib`G4VUserPhysicsList::RetrievePhysicsTable(this=0x0000000109048c10, particle=0x000000010904f120, directory=0x0000000109048c48, ascii=true) + 559 at G4VUserPhysicsList.cc:912
        frame #2: 0x0000000102d4c3d8 libG4run.dylib`G4VUserPhysicsList::BuildPhysicsTable(this=0x0000000109048c10, particle=0x000000010904f120) + 536 at G4VUserPhysicsList.cc:621
        frame #3: 0x0000000102d4beaf libG4run.dylib`G4VUserPhysicsList::BuildPhysicsTable(this=0x0000000109048c10) + 1535 at G4VUserPhysicsList.cc:583
        frame #4: 0x0000000102d2883c libG4run.dylib`G4RunManagerKernel::BuildPhysicsTables(this=0x00000001087253c0, fakeRun=false) + 76 at G4RunManagerKernel.cc:707
        frame #5: 0x0000000102d28367 libG4run.dylib`G4RunManagerKernel::RunInitialization(this=0x00000001087253c0, fakeRun=false) + 279 at G4RunManagerKernel.cc:609
        frame #6: 0x0000000102d04bb8 libG4run.dylib`G4RunManager::RunInitialization(this=0x0000000108724fb0) + 56 at G4RunManager.cc:313
        frame #7: 0x0000000102d048b2 libG4run.dylib`G4RunManager::BeamOn(this=0x0000000108724fb0, n_event=1, macroFile=0x0000000000000000, n_select=-1) + 146 at G4RunManager.cc:272
        frame #8: 0x000000010155b6ad libcfg4.dylib`CG4::propagate(this=0x0000000108721880) + 605 at CG4.cc:180
        frame #9: 0x000000010000d542 CG4Test`main(argc=16, argv=0x00007fff5fbfd870) + 210 at CG4Test.cc:20
        frame #10: 0x00007fff89e755fd libdyld.dylib`start + 1
    (lldb) 

::

      (lldb) b "OpRayleigh::BuildPhysicsTable(G4ParticleDefinition const&)" 


Huh table built twice, the 2nd time with messed up values ?::

    G4ProductionCutsTable::CheckForRetrieveCutsTable!!
    [2016-May-27 17:37:47.334780]:info: OpRayleigh::check mat /dd/Materials/PPE min 1e+06 max 1e+06 num 39
    [2016-May-27 17:37:47.334935]:info: OpRayleigh::check mat /dd/Materials/MixGas min 1e+06 max 1e+06 num 39
    [2016-May-27 17:37:47.335049]:info: OpRayleigh::check mat /dd/Materials/Air min 1e+06 max 1e+06 num 39
    [2016-May-27 17:37:47.335139]:info: OpRayleigh::check mat /dd/Materials/Bakelite min 1e+06 max 1e+06 num 39
    [2016-May-27 17:37:47.335231]:info: OpRayleigh::check mat /dd/Materials/Foam min 1e+06 max 1e+06 num 39
    [2016-May-27 17:37:47.335338]:info: OpRayleigh::check mat /dd/Materials/Aluminium min 1e+06 max 1e+06 num 39
    [2016-May-27 17:37:47.335429]:info: OpRayleigh::check mat /dd/Materials/Iron min 1e+06 max 1e+06 num 39
    [2016-May-27 17:37:47.335516]:info: OpRayleigh::check mat /dd/Materials/GdDopedLS min 500000 max 850 num 39
    ...
    [2016-May-27 17:37:47.337585]:info: OpRayleigh::check mat /dd/Materials/ADTableStainlessSteel min 1e+06 max 1e+06 num 39
    [2016-May-27 17:37:47.337683]:info: OpRayleigh::check mat /dd/Materials/Tyvek min 1e+06 max 1e+06 num 39
    [2016-May-27 17:37:47.337771]:info: OpRayleigh::check mat /dd/Materials/OwsWater min 1e+06 max 1e+06 num 39
    [2016-May-27 17:37:47.337862]:info: OpRayleigh::check mat /dd/Materials/DeadWater min 1e+06 max 1e+06 num 39
    [2016-May-27 17:37:47.337953]:info: OpRayleigh::check mat /dd/Materials/RadRock min 1e+06 max 1e+06 num 39
    [2016-May-27 17:37:47.338044]:info: OpRayleigh::check mat /dd/Materials/Rock min 1e+06 max 1e+06 num 39
    [2016-May-27 17:37:47.338134]:info: OpRayleigh::BuildPhysicsTable
    [2016-May-27 17:37:47.338232]:info: OpRayleigh::check mat /dd/Materials/PPE min 0 max 6.36599e-314 num 39
    [2016-May-27 17:37:47.338327]:info: OpRayleigh::check mat /dd/Materials/MixGas min 2.31584e+77 max 2.22383e-314 num 39
    [2016-May-27 17:37:47.338429]:info: OpRayleigh::check mat /dd/Materials/Air min -1.3899e-315 max 2.22387e-314 num 39
    [2016-May-27 17:37:47.338526]:info: OpRayleigh::check mat /dd/Materials/Bakelite min -2.68156e+154 max 2.22385e-314 num 39
    [2016-May-27 17:37:47.338627]:info: OpRayleigh::check mat /dd/Materials/Foam min 2 max 2.22384e-314 num 39
    [2016-May-27 17:37:47.338719]:info: OpRayleigh::check mat /dd/Materials/Aluminium min -1.49167e-154 max 2.22386e-314 num 39
    [2016-May-27 17:37:47.338845]:info: OpRayleigh::check mat /dd/Materials/Iron min 1.38991e-315 max 2.22384e-314 num 39
    [2016-May-27 17:37:47.338944]:info: OpRayleigh::check mat /dd/Materials/GdDopedLS min 2.68156e+154 max 2.22386e-314 num 39
    ...
    [2016-May-27 17:37:47.341240]:info: OpRayleigh::check mat /dd/Materials/ADTableStainlessSteel min -2 max 2.22394e-314 num 39
    [2016-May-27 17:37:47.341344]:info: OpRayleigh::check mat /dd/Materials/Tyvek min 1.49167e-154 max -2 num 39
    [2016-May-27 17:37:47.341436]:info: OpRayleigh::check mat /dd/Materials/OwsWater min -1.38996e-315 max 2.22395e-314 num 39
    [2016-May-27 17:37:47.341537]:info: OpRayleigh::check mat /dd/Materials/DeadWater min -2.68156e+154 max -1.38971e-315 num 39
    [2016-May-27 17:37:47.341638]:info: OpRayleigh::check mat /dd/Materials/RadRock min 2 max 2.22395e-314 num 39
    [2016-May-27 17:37:47.341730]:info: OpRayleigh::check mat /dd/Materials/Rock min -1.49167e-154 max 2.22386e-314 num 39
    [2016-May-27 17:37:47.341826]:info: OpRayleigh::BuildPhysicsTable



First::

    (lldb) bt
    * thread #1: tid = 0x733d38, 0x0000000101595797 libcfg4.dylib`OpRayleigh::BuildPhysicsTable(this=0x000000010e1f67d0, (null)=0x000000010904f120) + 23 at OpRayleigh.cc:229, queue = 'com.apple.main-thread', stop reason = breakpoint 2.1
      * frame #0: 0x0000000101595797 libcfg4.dylib`OpRayleigh::BuildPhysicsTable(this=0x000000010e1f67d0, (null)=0x000000010904f120) + 23 at OpRayleigh.cc:229
        frame #1: 0x0000000102d4d68f libG4run.dylib`G4VUserPhysicsList::RetrievePhysicsTable(this=0x0000000109048c10, particle=0x000000010904f120, directory=0x0000000109048c48, ascii=true) + 559 at G4VUserPhysicsList.cc:912
        frame #2: 0x0000000102d4c3d8 libG4run.dylib`G4VUserPhysicsList::BuildPhysicsTable(this=0x0000000109048c10, particle=0x000000010904f120) + 536 at G4VUserPhysicsList.cc:621
        frame #3: 0x0000000102d4beaf libG4run.dylib`G4VUserPhysicsList::BuildPhysicsTable(this=0x0000000109048c10) + 1535 at G4VUserPhysicsList.cc:583
        frame #4: 0x0000000102d2883c libG4run.dylib`G4RunManagerKernel::BuildPhysicsTables(this=0x00000001087253c0, fakeRun=false) + 76 at G4RunManagerKernel.cc:707
        frame #5: 0x0000000102d28367 libG4run.dylib`G4RunManagerKernel::RunInitialization(this=0x00000001087253c0, fakeRun=false) + 279 at G4RunManagerKernel.cc:609
        frame #6: 0x0000000102d04bb8 libG4run.dylib`G4RunManager::RunInitialization(this=0x0000000108724fb0) + 56 at G4RunManager.cc:313
        frame #7: 0x0000000102d048b2 libG4run.dylib`G4RunManager::BeamOn(this=0x0000000108724fb0, n_event=1, macroFile=0x0000000000000000, n_select=-1) + 146 at G4RunManager.cc:272
        frame #8: 0x000000010155b33d libcfg4.dylib`CG4::propagate(this=0x0000000108721880) + 605 at CG4.cc:180
        frame #9: 0x000000010000d542 CG4Test`main(argc=16, argv=0x00007fff5fbfd818) + 210 at CG4Test.cc:20
        frame #10: 0x00007fff89e755fd libdyld.dylib`start + 1
        frame #11: 0x00007fff89e755fd libdyld.dylib`start + 1
    (lldb) 


Second::

    (lldb) bt
    * thread #1: tid = 0x733d38, 0x0000000101595797 libcfg4.dylib`OpRayleigh::BuildPhysicsTable(this=0x000000010e1f67d0, (null)=0x000000010904f120) + 23 at OpRayleigh.cc:229, queue = 'com.apple.main-thread', stop reason = breakpoint 2.1
      * frame #0: 0x0000000101595797 libcfg4.dylib`OpRayleigh::BuildPhysicsTable(this=0x000000010e1f67d0, (null)=0x000000010904f120) + 23 at OpRayleigh.cc:229
        frame #1: 0x0000000102d4c976 libG4run.dylib`G4VUserPhysicsList::BuildPhysicsTable(this=0x0000000109048c10, particle=0x000000010904f120) + 1974 at G4VUserPhysicsList.cc:689
        frame #2: 0x0000000102d4beaf libG4run.dylib`G4VUserPhysicsList::BuildPhysicsTable(this=0x0000000109048c10) + 1535 at G4VUserPhysicsList.cc:583
        frame #3: 0x0000000102d2883c libG4run.dylib`G4RunManagerKernel::BuildPhysicsTables(this=0x00000001087253c0, fakeRun=false) + 76 at G4RunManagerKernel.cc:707
        frame #4: 0x0000000102d28367 libG4run.dylib`G4RunManagerKernel::RunInitialization(this=0x00000001087253c0, fakeRun=false) + 279 at G4RunManagerKernel.cc:609
        frame #5: 0x0000000102d04bb8 libG4run.dylib`G4RunManager::RunInitialization(this=0x0000000108724fb0) + 56 at G4RunManager.cc:313
        frame #6: 0x0000000102d048b2 libG4run.dylib`G4RunManager::BeamOn(this=0x0000000108724fb0, n_event=1, macroFile=0x0000000000000000, n_select=-1) + 146 at G4RunManager.cc:272
        frame #7: 0x000000010155b33d libcfg4.dylib`CG4::propagate(this=0x0000000108721880) + 605 at CG4.cc:180
        frame #8: 0x000000010000d542 CG4Test`main(argc=16, argv=0x00007fff5fbfd818) + 210 at CG4Test.cc:20
        frame #9: 0x00007fff89e755fd libdyld.dylib`start + 1
    (lldb) 







