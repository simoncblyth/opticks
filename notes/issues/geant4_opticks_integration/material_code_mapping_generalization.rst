Material Code Mapping Generalization
=======================================

Persisted gensteps contain material indices, in order to 
map these to actual materials it is necessary to have 
a code to material name mapping. 


Issue : Dec 2016 : Lookup fails with live g4gun
-------------------------------------------------

* HUH did not do anything substantial but it seems not to be happening anymore


::

    delta:opticksgeo blyth$ opticks-find applyLookup
    ./ok/ok.bash:G4StepNPY::applyLookup does a to b mapping between lingo which is invoked 
    ./ok/ok.bash:     553         genstep.applyLookup(0, 2);   // translate materialIndex (1st quad, 3rd number) from chroma to GGeo 

    ./optickscore/OpticksEvent.cc:    m_g4step->applyLookup(0, 2);  // jj, kk [1st quad, third value] is materialIndex
    ./optickscore/OpticksEvent.cc:    idx->applyLookup<unsigned char>(phosel_values);
    ./optickscore/tests/IndexerTest.cc:    idx->applyLookup<unsigned char>(phosel->getValues());
    ./opticksnpy/tests/NLookupTest.cc:    cs.applyLookup(0, 2); // materialIndex  (1st quad, 3rd number)

    ./opticksnpy/G4StepNPY.cpp:bool G4StepNPY::applyLookup(unsigned int index)
    ./opticksnpy/G4StepNPY.cpp:        printf(" G4StepNPY::applyLookup  %3u -> %3d  a[%s] b[%s] \n", acode, bcode, aname.c_str(), bname.c_str() );
    ./opticksnpy/G4StepNPY.cpp:        printf("G4StepNPY::applyLookup failed to translate acode %u : %s \n", acode, aname.c_str() );
    ./opticksnpy/G4StepNPY.cpp:void G4StepNPY::applyLookup(unsigned int jj, unsigned int kk)
    ./opticksnpy/G4StepNPY.cpp:            bool ok = applyLookup(index);
    ./opticksnpy/G4StepNPY.cpp:       LOG(fatal) << "G4StepNPY::applyLookup"
    ./opticksnpy/G4StepNPY.cpp:       m_npy->save("$TMP/G4StepNPY_applyLookup_FAIL.npy");
    ./opticksnpy/G4StepNPY.cpp:       dumpLookupFails("G4StepNPY::applyLookup");
    ./opticksnpy/G4StepNPY.hpp:       void applyLookup(unsigned int jj, unsigned int kk);
    ./opticksnpy/G4StepNPY.hpp:       bool applyLookup(unsigned int index);



::

    075 void OpticksRun::setGensteps(NPY<float>* gensteps)
     76 {
     77     LOG(info) << "OpticksRun::setGensteps " << gensteps->getShapeString() ;
     78 
     79     assert(m_evt && m_g4evt && "must OpticksRun::createEvent prior to OpticksRun::setGensteps");
     80 
     81     m_g4evt->setGenstepData(gensteps);
     82 
     83     passBaton(); 
     84 }
     85 
     86 void OpticksRun::passBaton()
     87 {
     88     NPY<float>* nopstep = m_g4evt->getNopstepData() ;
     89     NPY<float>* genstep = m_g4evt->getGenstepData() ;
     90 
     91     LOG(info) << "OpticksRun::passBaton"
     92               << " nopstep " << nopstep
     93               << " genstep " << genstep
     94               ;
     95 
     96 
     97    //
     98    // Not-cloning as these buffers are not actually distinct 
     99    // between G4 and OK.
    100    //
    101    // Nopstep and Genstep should be treated as owned 
    102    // by the m_g4evt not the Opticks m_evt 
    103    // where the m_evt pointers are just weak reference guests 
    104    //
    105 
    106     m_evt->setNopstepData(nopstep);
    107     m_evt->setGenstepData(genstep);
    108 }


::

    0938 void OpticksEvent::setGenstepData(NPY<float>* genstep_data, bool progenitor, const char* oac_label)
     939 {
     940     int nitems = NPYBase::checkNumItems(genstep_data);
     941     if(nitems < 1)
     942     {
     943          LOG(warning) << "OpticksEvent::setGenstepData SKIP "
     944                       << " nitems " << nitems
     945                       ;
     946          return ;
     947     }
     948 
     949     importGenstepData(genstep_data, oac_label );
     950 
     951     setBufferControl(genstep_data);
     952 
     953     m_genstep_data = genstep_data  ;
     954     m_parameters->add<std::string>("genstepDigest",   m_genstep_data->getDigestString()  );
     955 
     956     //                                                j k l sz   type        norm   iatt  item_from_dim
     957     ViewNPY* vpos = new ViewNPY("vpos",m_genstep_data,1,0,0,4,ViewNPY::FLOAT,false,false, 1);    // (x0, t0)                     2nd GenStep quad 
     958     ViewNPY* vdir = new ViewNPY("vdir",m_genstep_data,2,0,0,4,ViewNPY::FLOAT,false,false, 1);    // (DeltaPosition, step_length) 3rd GenStep quad
     959 
     960     m_genstep_vpos = vpos ;
     961 
     962     m_genstep_attr = new MultiViewNPY("genstep_attr");
     963     m_genstep_attr->add(vpos);
     964     m_genstep_attr->add(vdir);
     965 
     966     {
     967         m_num_gensteps = m_genstep_data->getShape(0) ;
     968         unsigned int num_photons = m_genstep_data->getUSum(0,3);
     969         bool resize = progenitor ;
     970         setNumPhotons(num_photons, resize); // triggers a resize   <<<<<<<<<<<<< SPECIAL HANDLING OF GENSTEP <<<<<<<<<<<<<<
     971     }
     972 }




    1046 void OpticksEvent::importGenstepData(NPY<float>* gs, const char* oac_label)
    1047 {
    1048     Parameters* gsp = gs->getParameters();
    1049     m_parameters->append(gsp);
    1050 
    1051     gs->setBufferSpec(OpticksEvent::GenstepSpec(isCompute()));
    1052 
    1053     assert(m_g4step == NULL && "OpticksEvent::importGenstepData can only do this once ");
    1054     m_g4step = new G4StepNPY(gs);
    1055 
    1056     OpticksActionControl oac(gs->getActionControlPtr());
    1057     if(oac_label)
    1058     {
    1059         LOG(debug) << "OpticksEvent::importGenstepData adding oac_label " << oac_label ;
    1060         oac.add(oac_label);
    1061     }
    1062 
    1063 
    1064     LOG(debug) << "OpticksEvent::importGenstepData"
    1065                << brief()
    1066                << " shape " << gs->getShapeString()
    1067                << " " << oac.description("oac")
    1068                ;
    1069 
    1070     if(oac("GS_LEGACY"))
    1071     {
    1072         translateLegacyGensteps(gs);
    1073     }
    1074     else if(oac("GS_TORCH"))
    1075     {
    1076         LOG(debug) << " checklabel of torch steps  " << oac.description("oac") ;
    1077         m_g4step->checklabel(TORCH);
    1078     }
    1079     else if(oac("GS_FABRICATED"))
    1080     {
    1081         m_g4step->checklabel(FABRICATED);
    1082     }
    1083     else
    1084     {
    1085         LOG(debug) << " checklabel of non-legacy (collected direct) gensteps  " << oac.description("oac") ;
    1086         m_g4step->checklabel(CERENKOV, SCINTILLATION);
    1087     }
    1088 
    1089     m_g4step->countPhotons();
    .... 
    1105 }
    1106 



    0986 void OpticksEvent::translateLegacyGensteps(NPY<float>* gs)
     987 {
     988     OpticksActionControl oac(gs->getActionControlPtr());
     989     bool gs_torch = oac.isSet("GS_TORCH") ;
     990     bool gs_legacy = oac.isSet("GS_LEGACY") ;
     991 
     992     if(!gs_legacy) return ;
     993     assert(!gs_torch); // there are no legacy torch files ?
     994 
     995     if(gs->isGenstepTranslated())
     996     {
     997         LOG(warning) << "OpticksEvent::translateLegacyGensteps already translated " ;
     998         return ;
     999     }
    1000 
    1001     gs->setGenstepTranslated();
    1002 
    1003     NLookup* lookup = gs->getLookup();
    1004     if(!lookup)
    1005             LOG(fatal) << "OpticksEvent::translateLegacyGensteps"
    1006                        << " IMPORT OF LEGACY GENSTEPS REQUIRES gs->setLookup(NLookup*) "
    1007                        << " PRIOR TO OpticksEvent::setGenstepData(gs) "
    1008                        ;
    1009 
    1010     assert(lookup);
    1011 
    1012     m_g4step->relabel(CERENKOV, SCINTILLATION);
    1013 
    1014     // CERENKOV or SCINTILLATION codes are used depending on 
    1015     // the sign of the pre-label 
    1016     // this becomes the ghead.i.x used in cu/generate.cu
    1017     // which dictates what to generate
    1018 
    1019     lookup->close("OpticksEvent::translateLegacyGensteps GS_LEGACY");
    1020 
    1021     m_g4step->setLookup(lookup);
    1022     m_g4step->applyLookup(0, 2);  // jj, kk [1st quad, third value] is materialIndex
    1023     // replaces original material indices with material lines
    1024     // for easy access to properties using boundary_lookup GPU side
    1025 
    1026 }





Legacy Approach
----------------

Translate on load
~~~~~~~~~~~~~~~~~~~

Genstep material indices are translated into GPU material lines on loading the file,
to keep things simple GPU side.

`NPY<float>* OpticksHub::loadGenstepFile()`::

    389     G4StepNPY* g4step = new G4StepNPY(gs);
    390     g4step->relabel(CERENKOV, SCINTILLATION);
    391     // which code is used depends in the sign of the pre-label 
    392     // becomes the ghead.i.x used in cu/generate.cu
    393 
    394     if(m_opticks->isDayabay())
    395     {
    396         // within GGeo this depends on GBndLib
    397         NLookup* lookup = m_ggeo ? m_ggeo->getLookup() : NULL ;
    398         if(lookup)
    399         {
    400             g4step->setLookup(lookup);
    401             g4step->applyLookup(0, 2);  // jj, kk [1st quad, third value] is materialIndex
    402             //
    403             // replaces original material indices with material lines
    404             // for easy access to properties using boundary_lookup GPU side
    405             //
    406         }
    407         else
    408         {
    409             LOG(warning) << "OpticksHub::loadGenstepFile not applying lookup" ;
    410         }
    411     }
    412     return gs ;
         

* with in memory gensteps direct from G4, need to do the 
  same thing but with the lookup will need to be different


Lookups
~~~~~~~~~

* npy-/NLookup does the mapping

::

     /// setupLookup is invoked by GGeo::loadGeometry

     620 void GGeo::setupLookup()
     621 {
     622     //  maybe this belongs in GBndLib ?
     623     //
     624     m_lookup = new NLookup() ;
     625 
     626     const char* cmmd = m_opticks->getDetectorBase() ;
     627 
     628     m_lookup->loadA( cmmd, "ChromaMaterialMap.json", "/dd/Materials/") ;
     629 
     630     std::map<std::string, unsigned int>& msu  = m_lookup->getB() ;
     631 
     632     m_bndlib->fillMaterialLineMap( msu ) ;
     633 
     634     m_lookup->crossReference();
     635 
     636     //m_lookup->dump("GGeo::setupLookup");  
     637 }



ggeo-/tests/NLookupTest.cc::

    GBndLib* blib = GBndLib::load(m_opticks, true );

    NLookup* m_lookup = new NLookup();

    const char* cmmd = m_opticks->getDetectorBase() ;

    m_lookup->loadA( cmmd , "ChromaMaterialMap.json", "/dd/Materials/") ;

    std::map<std::string, unsigned int>& msu = m_lookup->getB() ;

    blib->fillMaterialLineMap( msu ) ;     // shortname eg "GdDopedLS" to material line mapping 

    m_lookup->crossReference();

    m_lookup->dump("ggeo-/NLookupTest");



ChromaMaterialMap.json contains name to code mappings used 
for a some very old gensteps that were produced by G4DAEChroma
and which are still in use.
As the assumption of all gensteps being produced the same
way and with the same material mappings will soon become 
incorrect, need a more flexible way.

Perhaps a sidecar file to the gensteps .npy should
contain material mappings, and if it doesnt exist then 
defaults are used ?

::

    simon:DayaBay blyth$ cat ChromaMaterialMap.json | tr "," "\n"
    {"/dd/Materials/OpaqueVacuum": 18
     "/dd/Materials/Pyrex": 21
     "/dd/Materials/PVC": 20
     "/dd/Materials/NitrogenGas": 16
     "/dd/Materials/Teflon": 24
     "/dd/Materials/ESR": 9
     "/dd/Materials/MineralOil": 14


Changes
---------

* move NLookup to live up in OpticksHub in order to 
  configure it from the hub prior to geometry loading 
  when the lookup cross referencing is done
 


