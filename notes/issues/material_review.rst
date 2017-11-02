Material Review
==================




ABSLENGTH
----------

::

    simon:geant4_10_02_p01 blyth$ g4-cc ABSLENGTH
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/optical/src/G4OpAbsorption.cc:                                                GetProperty("ABSLENGTH");
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/optical/src/G4OpWLS.cc:      GetProperty("WLSABSLENGTH");

    g4-cls G4OpAbsorption


::

    119 // GetMeanFreePath
    120 // ---------------
    121 //
    122 G4double G4OpAbsorption::GetMeanFreePath(const G4Track& aTrack,
    123                          G4double ,
    124                          G4ForceCondition* )
    125 {
    126     const G4DynamicParticle* aParticle = aTrack.GetDynamicParticle();
    127         const G4Material* aMaterial = aTrack.GetMaterial();
    128 
    129     G4double thePhotonMomentum = aParticle->GetTotalMomentum();
    130 
    131     G4MaterialPropertiesTable* aMaterialPropertyTable;
    132     G4MaterialPropertyVector* AttenuationLengthVector;
    133 
    134         G4double AttenuationLength = DBL_MAX;
    135 
    136     aMaterialPropertyTable = aMaterial->GetMaterialPropertiesTable();
    137 
    138     if ( aMaterialPropertyTable ) {
    139        AttenuationLengthVector = aMaterialPropertyTable->
    140                                                 GetProperty("ABSLENGTH");
    141            if ( AttenuationLengthVector ){
    142              AttenuationLength = AttenuationLengthVector->
    143                                          Value(thePhotonMomentum);
    144            }
    145            else {
    146 //             G4cout << "No Absorption length specified" << G4endl;
    147            }
    148         }
    149         else {
    150 //           G4cout << "No Absorption length specified" << G4endl;
    151         }
    152 
    153         return AttenuationLength;
    154 }


::

    tboolean-;tboolean-sphere --okg4 -D


Use tab completion to get the signature, after get close enough that the G4 libs are loaded::

    (lldb)  b CG4::CG4
    Breakpoint 1: no locations (pending).
    WARNING:  Unable to resolve breakpoint to any actual locations.
    (lldb) r
    ...

    (lldb) b "G4OpAbsorption::GetMeanFreePath(G4Track const&, double, G4ForceCondition*)"   

    2017-11-02 17:27:35.760 INFO  [2384459] [CRunAction::BeginOfRunAction@19] CRunAction::BeginOfRunAction count 1
    Process 62357 stopped
    * thread #1: tid = 0x24624b, 0x0000000105796c34 libG4processes.dylib`G4OpAbsorption::GetMeanFreePath(G4Track const&, double, G4ForceCondition*) [inlined] G4Track::GetMaterial(this=<unavailable>) const at G4Track.icc:153, queue = 'com.apple.main-thread', stop reason = breakpoint 2.1
        frame #0: 0x0000000105796c34 libG4processes.dylib`G4OpAbsorption::GetMeanFreePath(G4Track const&, double, G4ForceCondition*) [inlined] G4Track::GetMaterial(this=<unavailable>) const at G4Track.icc:153
       150  
       151  // material
       152     inline G4Material* G4Track::GetMaterial() const
    -> 153     { return fpStep->GetPreStepPoint()->GetMaterial(); }
       154  
       155     inline G4Material* G4Track::GetNextMaterial() const
       156     { return fpStep->GetPostStepPoint()->GetMaterial(); }
    (lldb) bt
    * thread #1: tid = 0x24624b, 0x0000000105796c34 libG4processes.dylib`G4OpAbsorption::GetMeanFreePath(G4Track const&, double, G4ForceCondition*) [inlined] G4Track::GetMaterial(this=<unavailable>) const at G4Track.icc:153, queue = 'com.apple.main-thread', stop reason = breakpoint 2.1
      * frame #0: 0x0000000105796c34 libG4processes.dylib`G4OpAbsorption::GetMeanFreePath(G4Track const&, double, G4ForceCondition*) [inlined] G4Track::GetMaterial(this=<unavailable>) const at G4Track.icc:153
        frame #1: 0x0000000105796c34 libG4processes.dylib`G4OpAbsorption::GetMeanFreePath(this=0x0000000114a27e20, aTrack=0x0000000138b97770, (null)=<unavailable>, (null)=0x00000001149a8828) + 4 at G4OpAbsorption.cc:127
        frame #2: 0x0000000105793490 libG4processes.dylib`G4VDiscreteProcess::PostStepGetPhysicalInteractionLength(this=0x0000000114a27e20, track=0x0000000138b97770, previousStepSize=<unavailable>, condition=0x00000001149a8828) + 112 at G4VDiscreteProcess.cc:92
        frame #3: 0x0000000104ef2d67 libG4tracking.dylib`G4SteppingManager::DefinePhysicalStepLength() [inlined] G4VProcess::PostStepGPIL(this=0x0000000114a27e20, track=<unavailable>, previousStepSize=<unavailable>, condition=<unavailable>) + 14 at G4VProcess.hh:503
        frame #4: 0x0000000104ef2d59 libG4tracking.dylib`G4SteppingManager::DefinePhysicalStepLength(this=0x00000001149a86a0) + 249 at G4SteppingManager2.cc:172
        frame #5: 0x0000000104ef173e libG4tracking.dylib`G4SteppingManager::Stepping(this=0x00000001149a86a0) + 366 at G4SteppingManager.cc:180
        frame #6: 0x0000000104efb771 libG4tracking.dylib`G4TrackingManager::ProcessOneTrack(this=0x00000001149a8660, apValueG4Track=<unavailable>) + 913 at G4TrackingManager.cc:126
        frame #7: 0x0000000104e53727 libG4event.dylib`G4EventManager::DoProcessing(this=0x00000001149a85d0, anEvent=<unavailable>) + 1879 at G4EventManager.cc:185
        frame #8: 0x0000000104dd5611 libG4run.dylib`G4RunManager::ProcessOneEvent(this=0x000000010c59d130, i_event=0) + 49 at G4RunManager.cc:399
        frame #9: 0x0000000104dd54db libG4run.dylib`G4RunManager::DoEventLoop(this=0x000000010c59d130, n_event=60, macroFile=<unavailable>, n_select=<unavailable>) + 43 at G4RunManager.cc:367
        frame #10: 0x0000000104dd4913 libG4run.dylib`G4RunManager::BeamOn(this=0x000000010c59d130, n_event=60, macroFile=0x0000000000000000, n_select=-1) + 99 at G4RunManager.cc:273
        frame #11: 0x0000000104156363 libcfg4.dylib`CG4::propagate(this=0x000000010f100990) + 1667 at CG4.cc:336
        frame #12: 0x000000010424a28a libokg4.dylib`OKG4Mgr::propagate(this=0x00007fff5fbfe4c0) + 538 at OKG4Mgr.cc:82
        frame #13: 0x00000001000132fa OKG4Test`main(argc=27, argv=0x00007fff5fbfe5a8) + 1498 at OKG4Test.cc:57
        frame #14: 0x00007fff880d35fd libdyld.dylib`start + 1
        frame #15: 0x00007fff880d35fd libdyld.dylib`start + 1
    (lldb) 

    (lldb) p *aTrack.GetMaterial()
    (G4Material) $2 = {
      fName = (std::__1::string = "Vacuum")
      fChemicalFormula = (std::__1::string = "")
      fDensity = 0.00000062415096471204161
      fState = kStateGas
      fTemp = 293.14999999999998









interpolated domain via finebndtex is not default
--------------------------------------------------

::

    simon:issues blyth$ opticks-find makeInterpolationDomain
    ./ggeo/GBndLib.cc:                                mlib->getStandardDomain()->makeInterpolationDomain(Opticks::FINE_DOMAIN_STEP) 
    ./ggeo/GDomain.cc:GDomain<T>* GDomain<T>::makeInterpolationDomain(T step)
    ./ggeo/GPropertyMap.cc:    return new GPropertyMap<T>(this, m_standard_domain->makeInterpolationDomain(nm)); 
    ./ggeo/tests/GMaterialLibTest.cc:    GDomain<float>* idom = mlib->getStandardDomain()->makeInterpolationDomain(1.f);
    ./ggeo/GDomain.hh:     GDomain<T>* makeInterpolationDomain(T step);
    simon:opticks blyth$ 

::

     43 GBndLib* GBndLib::load(Opticks* ok, bool constituents)
     44 {
     45     GBndLib* blib = new GBndLib(ok);
     46 
     47     LOG(trace) << "GBndLib::load" ;
     48 
     49     blib->loadIndexBuffer();
     50 
     51     LOG(trace) << "GBndLib::load indexBuffer loaded" ;
     52     blib->importIndexBuffer();
     53 
     54 
     55     if(constituents)
     56     {
     57         GMaterialLib* mlib = GMaterialLib::load(ok);
     58         GSurfaceLib* slib = GSurfaceLib::load(ok);
     59         GDomain<float>* finedom = ok->hasOpt("finebndtex")
     60                             ?
     61                                 mlib->getStandardDomain()->makeInterpolationDomain(Opticks::FINE_DOMAIN_STEP)
     62                             :
     63                                 NULL
     64                             ;
     65 
     66         //assert(0); 
     67 
     68         if(finedom)
     69         {
     70             LOG(warning) << "GBndLib::load  --finebndtex option triggers interpolation of material and surface props "  ;
     71             GMaterialLib* mlib2 = new GMaterialLib(mlib, finedom );   
     72             GSurfaceLib* slib2 = new GSurfaceLib(slib, finedom );   
     73 
     74             mlib2->setBuffer(mlib2->createBuffer());
     75             slib2->setBuffer(slib2->createBuffer());
     76 
     77             blib->setStandardDomain(finedom);
     78             blib->setMaterialLib(mlib2);
     79             blib->setSurfaceLib(slib2);
     80 
     81             blib->setBuffer(blib->createBuffer());
     82         }
     83         else
     84         {
     85             blib->setMaterialLib(mlib);
     86             blib->setSurfaceLib(slib);
     87         }
     88     }
     89 
     90     LOG(trace) << "GBndLib::load DONE" ;
     91 
     92     return blib ;
     93 }




test materials
----------------

::

    746 void GMaterialLib::addTestMaterials()
    747 {
    748     typedef std::pair<std::string, std::string> SS ;
    749     typedef std::vector<SS> VSS ;
    750 
    751     VSS rix ;
    752 
    753     rix.push_back(SS("GlassSchottF2", "$OPTICKS_INSTALL_PREFIX/opticksdata/refractiveindex/tmp/glass/schott/F2.npy"));
    754     rix.push_back(SS("MainH2OHale",   "$OPTICKS_INSTALL_PREFIX/opticksdata/refractiveindex/tmp/main/H2O/Hale.npy"));
    755     
    756     // NB when adding test materials also need to set in prefs ~/.opticks/GMaterialLib
    757     //
    758     //    * priority order (for transparent materials arrange to be less than 16 for material sequence tracking)
    759     //    * color 
    760     //    * two letter abbreviation
    761     //
    762     // for these settings to be acted upon must rebuild the geocache with : "ggv -G"      
    763     //

::

    151 const G4Material* CMaterialLib::convertMaterial(const GMaterial* kmat)
    152 {
    159     const char* name = kmat->getShortName();
    160     const G4Material* prior = getG4Material(name) ;
    161     if(prior)
    162     {
    169         return prior ;
    170     }
    173     unsigned int materialIndex = m_mlib->getMaterialIndex(kmat);
    174 
    175     G4String sname = name ;
    182 
    183     G4Material* material(NULL);
    184     if(strcmp(name,"MainH2OHale")==0)
    185     {
    186         material = makeWater(name) ;
    187     }
    188     else
    189     {
    190         G4double z, a, density ;
    191         // presumably z, a and density are not relevant for optical photons 
    192         material = new G4Material(sname, z=1., a=1.01*g/mole, density=universe_mean_density );
    193     }
    198     G4MaterialPropertiesTable* mpt = makeMaterialPropertiesTable(kmat);
    199     material->SetMaterialPropertiesTable(mpt);
    200 
    201     m_ggtog4[kmat] = material ;
    202     m_g4mat[name] = material ;   // used by getG4Material(shortname) 
    203 


CMaterialLibTest : does conversions
---------------------------------------

::

    op --cmat




