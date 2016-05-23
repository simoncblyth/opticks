# === func-gen- : optix/cfg4/cg4 fgp optix/cfg4/cg4.bash fgn cg4 fgh optix/cfg4
cg4-src(){      echo optix/cfg4/cg4.bash ; }
cg4-source(){   echo ${BASH_SOURCE:-$(env-home)/$(cg4-src)} ; }
cg4-vi(){       vi $(cg4-source) ; }
cg4-env(){      elocal- ; }
cg4-usage(){ cat << EOU

CG4 : Full Opticks/Geant4 integration
======================================

Objective
-----------

Full geometry Geant4 and Opticks integrated, enabling:

* full geometry testing/development/comparison
* Geant4 particle gun control from within interactive Opticks (GGeoView) 

See Also
---------

* cfg4- partial integration for small test geometries only
* export- geometry exporter

Approach
---------

Geant4 and Opticks need to be using the same geometry...
 
* G4DAE for Opticks
* GDML for Geant4 

Standard export- controlled geometry exports include the .gdml
and .dae when they have a "G" and "D" in the path like the 
current standard::

  /usr/local/env/geant4/geometry/export/DayaBay_VGDX_20140414-1300/


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



Issue : with GDML geometry opticalphoton time going backwards
---------------------------------------------------------------

::
  
    ggv-;ggv-g4gun

* maybe material props (eg refractive index) are messed up 


::

    *** G4Exception : TRACK003
          issued by : G4ParticleChange::CheckIt
    momentum, energy, and/or time was illegal
    *** Event Must Be Aborted ***
    -------- EEEE -------- G4Exception-END --------- EEEE -------

      G4VParticleChange::CheckIt    : the true step length is negative  !!  Difference:  0.0577914[MeV] 
    opticalphoton E=6.60373e-06 pos=-18.0795, -799.699, -6.60502


    (lldb) b "G4ParticleChange::CheckIt(G4Track const&)" 
           b 546  # once inside   

    (lldb) bt
    * thread #1: tid = 0x61a7be, 0x0000000105e901cd libG4track.dylib`G4ParticleChange::CheckIt(this=0x00000001091e8cc0, aTrack=0x000000010ec024b0) + 909 at G4ParticleChange.cc:546, queue = 'com.apple.main-thread', stop reason = breakpoint 4.1
      * frame #0: 0x0000000105e901cd libG4track.dylib`G4ParticleChange::CheckIt(this=0x00000001091e8cc0, aTrack=0x000000010ec024b0) + 909 at G4ParticleChange.cc:546
        frame #1: 0x0000000105e9963f libG4track.dylib`G4ParticleChangeForTransport::UpdateStepForAlongStep(this=0x00000001091e8cc0, pStep=0x00000001091270d0) + 1519 at G4ParticleChangeForTransport.cc:202
        frame #2: 0x0000000102e8896e libG4tracking.dylib`G4SteppingManager::InvokeAlongStepDoItProcs(this=0x0000000109126f40) + 254 at G4SteppingManager2.cc:420
        frame #3: 0x0000000102e84168 libG4tracking.dylib`G4SteppingManager::Stepping(this=0x0000000109126f40) + 504 at G4SteppingManager.cc:191
        frame #4: 0x0000000102e9b92d libG4tracking.dylib`G4TrackingManager::ProcessOneTrack(this=0x0000000109126f00, apValueG4Track=0x000000010ec024b0) + 1357 at G4TrackingManager.cc:126
        frame #5: 0x0000000102d78e44 libG4event.dylib`G4EventManager::DoProcessing(this=0x0000000109126e70, anEvent=0x000000010eb808d0) + 3188 at G4EventManager.cc:185
        frame #6: 0x0000000102d79b2f libG4event.dylib`G4EventManager::ProcessOneEvent(this=0x0000000109126e70, anEvent=0x000000010eb808d0) + 47 at G4EventManager.cc:336
        frame #7: 0x0000000102ca6c75 libG4run.dylib`G4RunManager::ProcessOneEvent(this=0x00000001087234e0, i_event=0) + 69 at G4RunManager.cc:399
        frame #8: 0x0000000102ca6ab5 libG4run.dylib`G4RunManager::DoEventLoop(this=0x00000001087234e0, n_event=1, macroFile=0x0000000000000000, n_select=-1) + 101 at G4RunManager.cc:367
        frame #9: 0x0000000102ca58e4 libG4run.dylib`G4RunManager::BeamOn(this=0x00000001087234e0, n_event=1, macroFile=0x0000000000000000, n_select=-1) + 196 at G4RunManager.cc:273
        frame #10: 0x000000010153edb0 libcfg4.dylib`CG4::propagate(this=0x0000000108721330) + 752 at CG4.cc:137
        frame #11: 0x000000010000d5a2 CG4Test`main(argc=11, argv=0x00007fff5fbfdd50) + 210 at CG4Test.cc:18
        frame #12: 0x00007fff89e755fd libdyld.dylib`start + 1





[WORKED AROUND] Issue : old GDML export omits material properties
---------------------------------------------------------------------

Get NULL MPT in loaded model::

    147 void G4GDMLWriteMaterials::MaterialWrite(const G4Material* const materialPtr)
    148 {
    ... 
    163    if (materialPtr->GetMaterialPropertiesTable())
    164    {
    165      PropertyWrite(materialElement, materialPtr);
    166    }

    228 void G4GDMLWriteMaterials::PropertyWrite(xercesc::DOMElement* matElement,
    229                                          const G4Material* const mat)
    230 {
    ...
    241    for (mpos=pmap->begin(); mpos!=pmap->end(); mpos++)
    242    {
    243       propElement = NewElement("property");
    244       propElement->setAttributeNode(NewAttribute("name", mpos->first));
    245       propElement->setAttributeNode(NewAttribute("ref",
    246                                     GenerateName(mpos->first, mpos->second)));


No property elements in the ancient geant4 exported GDML::

    simon:cfg4 blyth$ grep property /usr/local/env/geant4/geometry/export/DayaBay_VGDX_20140414-1300/g4_00.gdml

Only that one GDML file amongst the exports, exports were copied over to D:: 

    simon:export blyth$ find . -name '*.gdml'
    ./DayaBay_VGDX_20140414-1300/g4_00.gdml
    simon:export blyth$ pwd
    /usr/local/env/geant4/geometry/export

Anyhow checking geant4.0.2p01/G4GDMLWriteMaterials::MaterialWrite does not write material properties::

    [blyth@ntugrid5 env]$ nuwa-;cd $(nuwa-g4-sdir)


* re-export DYB geometry, checking material properties, old export lacks em  

  * this not so easy, would need to backport recent GDML writer to work with nuwa 
    but the info is in the DAE, and are able to reconstruct G4 materials with 
    the properties for the geocache as done by cfg4- CPropLib, so used this 
    workaround  

  * Actually this work is closely releated to G4DAE exporter and intended 
    eventual revisit to bring up to latest G4 and maybe find way to 
    reduce pain of subsequent such syncing.
    Also note that GDML writer requires special G4 build configuration 
    so if that could be avoided in g4d- ?

  * see also export- 


DONE
-----

* OpticksResource .gdml path handling 
* Break off a CG4 singleton class from cfg4- to hold common G4 components, runmanager etc.. 
* move ggv- tests out of ggeoview- into separate .bash, check the cfg4 tests following refactor 
* add GDML loading 
* workaround lack of MPT in ancient g4 GDML export by converting from the G4DAE export  

TODO
----

* where to slot in CGDMLDetector into the machinery, cli, options, config ?

  * will need step recorder... NumpyEvt etc..
  * maybe bifurcate CCfG4 ? into TEST and FULL  CBaseDetector needed ?  

* try to get contained CG4 to generate smth 
* maybe: split CG4 into separate cg4- package rather than co-locating with cfg4-, cfg4- can depend on cg4-
* bring over, cleanup, simplify G4DAEChroma gdc- (no need for ZMQ) 
  with the customized step collecting Cerenkov and Scintillation processes
* collect other(non-photon producing processes) particle step tree somehow ? 
* step visualization 
* gun control interface, ImGui?  particle palette, shooter mode
* updated JUNO export, both DAE and GDML 




EOU
}
cg4-dir(){ echo $(env-home)/optix/cfg4 ; }
cg4-cd(){  cd $(cg4-dir); }
