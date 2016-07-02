Taming Geant4 Verbosity
=========================


FIXED Issue : g4 geometry cleanup WARNINGs
--------------------------------------------

::

    WARNING - Attempt to delete the physical volume store while geometry closed !
    WARNING - Attempt to delete the logical volume store while geometry closed !
    WARNING - Attempt to delete the solid store while geometry closed !
    WARNING - Attempt to delete the region store while geometry closed !
    Process 93045 exited with status = 0 (0x00000000) 

::

    simon:cfg4 blyth$ g4-cc "Attempt" | grep delete
    /usr/local/env/g4/geant4.10.02/source/geometry/management/src/G4PhysicalVolumeStore.cc:    G4cout << "WARNING - Attempt to delete the physical volume store"
    /usr/local/env/g4/geant4.10.02/source/geometry/management/src/G4LogicalVolumeStore.cc:    G4cout << "WARNING - Attempt to delete the logical volume store"
    /usr/local/env/g4/geant4.10.02/source/geometry/management/src/G4SolidStore.cc:    G4cout << "WARNING - Attempt to delete the solid store"
    /usr/local/env/g4/geant4.10.02/source/geometry/management/src/G4RegionStore.cc:    G4cout << "WARNING - Attempt to delete the region store"

    (lldb) b "G4PhysicalVolumeStore::Clean()"

Huh looks like system is calling dtors, i didnt think it was so polite::

    (lldb) bt
    * thread #1: tid = 0x66fc61, 0x00000001061055c4 libG4geometry.dylib`G4PhysicalVolumeStore::Clean() + 4 at G4PhysicalVolumeStore.cc:78, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
      * frame #0: 0x00000001061055c4 libG4geometry.dylib`G4PhysicalVolumeStore::Clean() + 4 at G4PhysicalVolumeStore.cc:78
        frame #1: 0x0000000106105572 libG4geometry.dylib`G4PhysicalVolumeStore::~G4PhysicalVolumeStore(this=0x00000001063e31a8) + 34 at G4PhysicalVolumeStore.cc:67
        frame #2: 0x00000001061059f5 libG4geometry.dylib`G4PhysicalVolumeStore::~G4PhysicalVolumeStore(this=0x00000001063e31a8) + 21 at G4PhysicalVolumeStore.cc:66
        frame #3: 0x00007fff8cdf07a1 libsystem_c.dylib`__cxa_finalize + 177
        frame #4: 0x00007fff8cdf0a4c libsystem_c.dylib`exit + 22
        frame #5: 0x00007fff89e75604 libdyld.dylib`start + 8

::

     74 void G4PhysicalVolumeStore::Clean()
     75 {
     76   // Do nothing if geometry is closed
     77   //
     78   if (G4GeometryManager::GetInstance()->IsGeometryClosed())
     79   {
     80     G4cout << "WARNING - Attempt to delete the physical volume store"
     81            << " while geometry closed !" << G4endl;
     82     return;
     83   }

Open geometry just before exitting::

    341 void CG4::cleanup()
    342 {
    343     LOG(info) << "CG4::cleanup opening geometry" ; 
    344     G4GeometryManager::GetInstance()->OpenGeometry();
    345 }



FIXED Issue : g4 process verbosity control
----------------------------------------------

::

   (lldb) b "G4VProcess::SetVerboseLevel(int)" 


See OpNovicePhysicsList::setProcessVerbosity called after run init.   



Issue : g4 couples table noise, comment call to DumpCutValuesTable()
------------------------------------------------------------------------

::

     371 void G4ProductionCutsTable::DumpCouples() const
     372 {
     373   G4cout << G4endl;
     374   G4cout << "========= Table of registered couples =============================="
     375          << G4endl;


     409 void OpNovicePhysicsList::SetCuts()
     410 {
     411   //  " G4VUserPhysicsList::SetCutsWithDefault" method sets
     412   //   the default cut value for all particle types
     413   //
     414   SetCutsWithDefault();
     415 
     416   //if (verboseLevel>0) DumpCutValuesTable();
     417   //   



    (lldb) bt
    * thread #1: tid = 0x643cf9, 0x0000000102f73d73 libG4processes.dylib`G4ProductionCutsTable::DumpCouples() const [inlined] std::__1::basic_ostream<char, std::__1::char_traits<char> >::operator<<(this=0x000000010681b1f8, __pf=0x0000000102f2b240)(std::__1::basic_ostream<char, std::__1::char_traits<char> >&)) at ostream:310, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
      * frame #0: 0x0000000102f73d73 libG4processes.dylib`G4ProductionCutsTable::DumpCouples() const [inlined] std::__1::basic_ostream<char, std::__1::char_traits<char> >::operator<<(this=0x000000010681b1f8, __pf=0x0000000102f2b240)(std::__1::basic_ostream<char, std::__1::char_traits<char> >&)) at ostream:310
        frame #1: 0x0000000102f73d73 libG4processes.dylib`G4ProductionCutsTable::DumpCouples(this=0x0000000105054b00) const + 67 at G4ProductionCutsTable.cc:373
        frame #2: 0x0000000102cf5d58 libG4run.dylib`G4VUserPhysicsList::DumpCutValuesTableIfRequested(this=0x00000001090dd5c0) + 72 at G4VUserPhysicsList.cc:823
        frame #3: 0x0000000102cd18b6 libG4run.dylib`G4RunManagerKernel::BuildPhysicsTables(this=0x0000000108721f60, fakeRun=false) + 198 at G4RunManagerKernel.cc:714
        frame #4: 0x0000000102cd1367 libG4run.dylib`G4RunManagerKernel::RunInitialization(this=0x0000000108721f60, fakeRun=false) + 279 at G4RunManagerKernel.cc:609
        frame #5: 0x0000000102cadbb8 libG4run.dylib`G4RunManager::RunInitialization(this=0x0000000108722e00) + 56 at G4RunManager.cc:313
        frame #6: 0x0000000102cad8b2 libG4run.dylib`G4RunManager::BeamOn(this=0x0000000108722e00, n_event=1, macroFile=0x0000000000000000, n_select=-1) + 146 at G4RunManager.cc:272
        frame #7: 0x000000010153f420 libcfg4.dylib`CG4::propagate(this=0x0000000108721520) + 752 at CG4.cc:163
        frame #8: 0x000000010000d5a2 CG4Test`main(argc=13, argv=0x00007fff5fbfdcb8) + 210 at CG4Test.cc:18
        frame #9: 0x00007fff89e755fd libdyld.dylib`start + 1
        frame #10: 0x00007fff89e755fd libdyld.dylib`start + 1
    (lldb) f 2
    frame #2: 0x0000000102cf5d58 libG4run.dylib`G4VUserPhysicsList::DumpCutValuesTableIfRequested(this=0x00000001090dd5c0) + 72 at G4VUserPhysicsList.cc:823
       820  void G4VUserPhysicsList::DumpCutValuesTableIfRequested()
       821  {
       822    if(fDisplayThreshold==0) return;
    -> 823    G4ProductionCutsTable::GetProductionCutsTable()->DumpCouples();
       824    fDisplayThreshold = 0;
       825  }
       826  
    (lldb) 




Issue : g4 init is slow, how to cache the physics tables ?
-------------------------------------------------------------

See ggv-;ggv-cache

* succeed to get faster start (about 9 seconds from beamOn to 1st step), 
  but many tables fail to be stored/retrieved

::

    # needs to be 3 to see the fails...
    /run/particle/verbose 3   

    G4VUserPhysicsList::BuildPhysicsTable   Retrieve Physics Table for e-
    G4VUserPhysicsList::RetrievePhysicsTable    Fail to retrieve Physics Table for Transportation
    Calculate Physics Table for e-
    G4VUserPhysicsList::RetrievePhysicsTable    Fail to retrieve Physics Table for eBrem
    Calculate Physics Table for e-
    G4VUserPhysicsList::RetrievePhysicsTable    Fail to retrieve Physics Table for Scintillation
    Calculate Physics Table for e-



Issue : g4 Em noise control
-----------------------------

Magic incantation, to setup construction params of Em processes::

  G4EmParameters* empar = G4EmParameters::Instance() ;
  empar->SetVerbose(0); 
  empar->SetWorkerVerbose(0); 


See my OpNovicePhysicsList/setupEmVerbosity


Issue : g4 noise control
--------------------------

::

    delta:geant4.10.02 blyth$ find . -name 'G4VEmProcess.hh'
    ./source/processes/electromagnetic/utils/include/G4VEmProcess.hh

    delta:geant4.10.02 blyth$ find . -name '*.hh' -exec grep -H public\ G4VEmProcess {} \;
    ...
    ./source/processes/electromagnetic/standard/include/G4ComptonScattering.hh:class G4ComptonScattering : public G4VEmProcess
    ./source/processes/electromagnetic/standard/include/G4CoulombScattering.hh:class G4CoulombScattering : public G4VEmProcess
    ./source/processes/electromagnetic/standard/include/G4eplusAnnihilation.hh:class G4eplusAnnihilation : public G4VEmProcess
    ./source/processes/electromagnetic/standard/include/G4GammaConversion.hh:class G4GammaConversion : public G4VEmProcess
    ./source/processes/electromagnetic/standard/include/G4NuclearStopping.hh:class G4NuclearStopping : public G4VEmProcess
    ./source/processes/electromagnetic/standard/include/G4PhotoElectricEffect.hh:class G4PhotoElectricEffect : public G4VEmProcess

    delta:geant4.10.02 blyth$ find . -name '*.cc' -exec grep -H PrintInfoProcess {} \;
    ./source/processes/electromagnetic/utils/src/G4VEmProcess.cc:      PrintInfoProcess(part); 
    ./source/processes/electromagnetic/utils/src/G4VEmProcess.cc:void G4VEmProcess::PrintInfoProcess(const G4ParticleDefinition& part)

     523 void G4VEmProcess::PrintInfoProcess(const G4ParticleDefinition& part)
     524 {
     525   if(verboseLevel > 0) {
     526     G4cout << std::setprecision(6);
     527     G4cout << G4endl << GetProcessName() << ":   for  "
     528            << part.GetParticleName();
     529     if(integral)  { G4cout << ", integral: 1 "; }
     530     if(applyCuts) { G4cout << ", applyCuts: 1 "; }
     531     G4cout << "    SubType= " << GetProcessSubType();;
     532     if(biasFactor != 1.0) { G4cout << "   BiasingFactor= " << biasFactor; }
     533     G4cout << "  BuildTable= " << buildLambdaTable;
     534     G4cout << G4endl;
     535     if(buildLambdaTable) {
     536       if(particle == &part) {

     (lldb) b "G4VEmProcess::PrintInfoProcess(G4ParticleDefinition const&)" 


Notably this only happens during first event at BeamOn during RunInitialization::

    (lldb) bt
      * thread #1: tid = 0x637b46, 0x0000000103576760 libG4processes.dylib`G4VEmProcess::PrintInfoProcess(this=0x000000010f4e55f0, part=0x000000010a2424c0) 
          + 32 at G4VEmProcess.cc:525, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
      * frame #0: 0x0000000103576760 libG4processes.dylib`G4VEmProcess::PrintInfoProcess(this=0x000000010f4e55f0, part=0x000000010a2424c0) + 32 at G4VEmProcess.cc:525
        frame #1: 0x0000000103575713 libG4processes.dylib`G4VEmProcess::BuildPhysicsTable(this=0x000000010f4e55f0, part=0x000000010a2424c0) + 1955 at G4VEmProcess.cc:415
        frame #2: 0x0000000102cf0976 libG4run.dylib`G4VUserPhysicsList::BuildPhysicsTable(this=0x000000010a23c190, particle=0x000000010a2424c0) + 1974 at G4VUserPhysicsList.cc:689
        frame #3: 0x0000000102cefb5a libG4run.dylib`G4VUserPhysicsList::BuildPhysicsTable(this=0x000000010a23c190) + 682 at G4VUserPhysicsList.cc:568
        frame #4: 0x0000000102ccc83c libG4run.dylib`G4RunManagerKernel::BuildPhysicsTables(this=0x000000010a00a690, fakeRun=false) + 76 at G4RunManagerKernel.cc:707
        frame #5: 0x0000000102ccc367 libG4run.dylib`G4RunManagerKernel::RunInitialization(this=0x000000010a00a690, fakeRun=false) + 279 at G4RunManagerKernel.cc:609
        frame #6: 0x0000000102ca8bb8 libG4run.dylib`G4RunManager::RunInitialization(this=0x000000010a009bf0) + 56 at G4RunManager.cc:313
        frame #7: 0x0000000102ca88b2 libG4run.dylib`G4RunManager::BeamOn(this=0x000000010a009bf0, n_event=1, macroFile=0x0000000000000000, n_select=-1) + 146 at G4RunManager.cc:272
        frame #8: 0x000000010153f770 libcfg4.dylib`CG4::propagate(this=0x000000010a0083c0) + 752 at CG4.cc:154
        frame #9: 0x000000010000d5a2 CG4Test`main(argc=13, argv=0x00007fff5fbfde90) + 210 at CG4Test.cc:18
        frame #10: 0x00007fff89e755fd libdyld.dylib`start + 1
    (lldb) 

At what point are the Em processes hooked up ?::

    (lldb) b "G4GammaConversion::G4GammaConversion(G4String const&, G4ProcessType)" 

    (lldb) p processName
    (const G4String) $0 = (std::__1::string = "conv")
    (lldb) bt
    * thread #1: tid = 0x63b2c2, 0x0000000103479c07 libG4processes.dylib`G4GammaConversion::G4GammaConversion(this=0x000000010da46120, processName=0x00007fff5fbfd998, type=fElectromagnetic) + 23 at G4GammaConversion.cc:90, queue = 'com.apple.main-thread', stop reason = breakpoint 1.2
      * frame #0: 0x0000000103479c07 libG4processes.dylib`G4GammaConversion::G4GammaConversion(this=0x000000010da46120, processName=0x00007fff5fbfd998, type=fElectromagnetic) + 23 at G4GammaConversion.cc:90
        frame #1: 0x00000001015689f4 libcfg4.dylib`OpNovicePhysicsList::ConstructEM(this=0x0000000109224420) + 340 at OpNovicePhysicsList.cc:201
        frame #2: 0x00000001015686d1 libcfg4.dylib`OpNovicePhysicsList::ConstructProcess(this=0x0000000109224420) + 49 at OpNovicePhysicsList.cc:110
        frame #3: 0x0000000102ccdfb2 libG4run.dylib`G4VUserPhysicsList::Construct(this=0x0000000109224420) + 162 at G4VUserPhysicsList.hh:416
        frame #4: 0x0000000102ccbb73 libG4run.dylib`G4RunManagerKernel::InitializePhysics(this=0x0000000109001200) + 291 at G4RunManagerKernel.cc:535
        frame #5: 0x0000000102cab6cf libG4run.dylib`G4RunManager::InitializePhysics(this=0x0000000109001090) + 47 at G4RunManager.cc:593
        frame #6: 0x0000000102cab57a libG4run.dylib`G4RunManager::Initialize(this=0x0000000109001090) + 186 at G4RunManager.cc:566
        frame #7: 0x000000010153ebf1 libcfg4.dylib`CG4::initialize(this=0x0000000108721520) + 545 at CG4.cc:110
        frame #8: 0x000000010000d589 CG4Test`main(argc=13, argv=0x00007fff5fbfde68) + 185 at CG4Test.cc:14
        frame #9: 0x00007fff89e755fd libdyld.dylib`start + 1
        frame #10: 0x00007fff89e755fd libdyld.dylib`start + 1
    (lldb) 

    (lldb) b "G4VProcess::SetVerboseLevel(int)" 

::

    (lldb) bt
    * thread #1: tid = 0x640010, 0x0000000103241b8f libG4processes.dylib`G4VProcess::SetVerboseLevel(this=0x000000010d8e6690, value=1) + 15 at G4VProcess.hh:439, queue = 'com.apple.main-thread', stop reason = breakpoint 1.3
      * frame #0: 0x0000000103241b8f libG4processes.dylib`G4VProcess::SetVerboseLevel(this=0x000000010d8e6690, value=1) + 15 at G4VProcess.hh:439
        frame #1: 0x000000010358fed2 libG4processes.dylib`G4VMultipleScattering::PreparePhysicsTable(this=0x000000010d8e6690, part=0x0000000109055170) + 1810 at G4VMultipleScattering.cc:252
        frame #2: 0x0000000102cf5159 libG4run.dylib`G4VUserPhysicsList::PreparePhysicsTable(this=0x0000000109047270, particle=0x0000000109055170) + 585 at G4VUserPhysicsList.cc:755
        frame #3: 0x0000000102cf4958 libG4run.dylib`G4VUserPhysicsList::BuildPhysicsTable(this=0x0000000109047270) + 168 at G4VUserPhysicsList.cc:530
        frame #4: 0x0000000102cd183c libG4run.dylib`G4RunManagerKernel::BuildPhysicsTables(this=0x0000000108721f60, fakeRun=false) + 76 at G4RunManagerKernel.cc:707
        frame #5: 0x0000000102cd1367 libG4run.dylib`G4RunManagerKernel::RunInitialization(this=0x0000000108721f60, fakeRun=false) + 279 at G4RunManagerKernel.cc:609
        frame #6: 0x0000000102cadbb8 libG4run.dylib`G4RunManager::RunInitialization(this=0x0000000108722e00) + 56 at G4RunManager.cc:313
        frame #7: 0x0000000102cad8b2 libG4run.dylib`G4RunManager::BeamOn(this=0x0000000108722e00, n_event=1, macroFile=0x0000000000000000, n_select=-1) + 146 at G4RunManager.cc:272
        frame #8: 0x000000010153f4d0 libcfg4.dylib`CG4::propagate(this=0x0000000108721520) + 752 at CG4.cc:164
        frame #9: 0x000000010000d5a2 CG4Test`main(argc=13, argv=0x00007fff5fbfdd38) + 210 at CG4Test.cc:18
        frame #10: 0x00007fff89e755fd libdyld.dylib`start + 1
    (lldb) f 1
    frame #1: 0x000000010358fed2 libG4processes.dylib`G4VMultipleScattering::PreparePhysicsTable(this=0x000000010d8e6690, part=0x0000000109055170) + 1810 at G4VMultipleScattering.cc:252
       249          fDispBeyondSafety = theParameters->LatDisplacementBeyondSafety();
       250        }
       251      }
    -> 252      if(master) { SetVerboseLevel(theParameters->Verbose()); }
       253      else {  SetVerboseLevel(theParameters->WorkerVerbose()); }
       254  
       255      // initialisation of models
    (lldb) p master
    (G4bool) $0 = true


    (lldb) p *theParameters
    (G4EmParameters) $2 = {
      ...
      verbose = 1
      workerVerbose = 0
      ...
    }






::

    conv:   for  gamma    SubType= 14  BuildTable= 1
          Lambda table from 1.022 MeV to 10 TeV, 20 bins per decade, spline: 1
          ===== EM models for the G4Region  DefaultRegionForTheWorld ======
            BetheHeitler :  Emin=        0 eV    Emax=       80 GeV
         BetheHeitlerLPM :  Emin=       80 GeV   Emax=       10 TeV



