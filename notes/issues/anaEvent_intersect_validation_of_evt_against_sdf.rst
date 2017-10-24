anaEvent_intersect_validation_of_evt_against_sdf
=========================================================

Approach
---------

Post propagation stage that brings together 
intersect positions recorded into Opticks events
with nnode level SDFs.  Event history selections
can yield intersect positions that should be 
very close to SDF zero.

Thoughts
-----------

I have vague recollections of implementing some of this functionality already.  
It needs to be found and elevated
to be available as a standard option.

* mentioned in :doc:`analytic_geometry_shakedown` see --dbgnode option within --gltf branch.

* BUT: --test geometry and --gltf currently not working together, so cannot use --dbgnode with test geometry

* trying to fix that runs into the GGeo monolith  :doc:`breaking_up_GGeo_monolith`


Simplify structure by 

* migrate GGeoTest from GGeo up to OpticksHub 


  
* make some inroads into the monolith by chopping off GPmtLib 


Searching commits
-----------------------

prep automated intersect debug by passing OpticksEvent down from OpticksHub into GScene::debugNodeIntersects

* https://bitbucket.org/simoncblyth/opticks/commits/e483f621013b0b474d4e6a835988cb3d67081871


anaEvent
----------

::

    simon:opticksgeo blyth$ opticks-find anaEvent
    ./ggeo/GScene.cc:    // gets invoked from OpticksHub::anaEvent 
    ./ok/OKMgr.cc:                if(!production) m_hub->anaEvent();
    ./okg4/OKG4Mgr.cc:                m_hub->anaEvent();
    ./okop/OpMgr.cc:                if(!production) m_hub->anaEvent();
    ./okop/OpMgr.cc:                if(!production) m_hub->anaEvent();
    ./optickscore/OpticksRun.cc:void OpticksRun::anaEvent()
    ./optickscore/OpticksRun.cc:    OK_PROFILE("OpticksRun::anaEvent.BEG");
    ./optickscore/OpticksRun.cc:    OK_PROFILE("OpticksRun::anaEvent.END");
    ./opticksgeo/OpticksHub.cc:void OpticksHub::anaEvent()
    ./opticksgeo/OpticksHub.cc:    LOG(info) << "OpticksHub::anaEvent" 
    ./opticksgeo/OpticksHub.cc:    m_run->anaEvent();
    ./optickscore/OpticksRun.hh:        void anaEvent(); // analysis based on saved evts 
    ./opticksgeo/OpticksHub.hh:       void anaEvent();
    simon:opticks blyth$ 
    simon:opticks blyth$ 


OpticksHub::anaEvent invoked by all top level Mgr 

::

    067 void OKMgr::propagate()
     68 {
     69     const Opticks& ok = *m_ok ;
     70 
     71     if(ok("nopropagate")) return ;
     72 
     73     bool production = m_ok->isProduction();
     74 
     75     if(ok.isLoad())
     76     {
     77          m_run->loadEvent();
     78 
     79          if(m_viz)
     80          {
     81              m_hub->target();           // if not Scene targetted, point Camera at gensteps of last created evt
     82 
     83              m_viz->uploadEvent();
     84 
     85              m_viz->indexPresentationPrep();
     86          }
     87     }
     88     else if(m_num_event > 0)
     89     {
     90         for(int i=0 ; i < m_num_event ; i++)
     91         {
     92             m_run->createEvent(i);
     93 
     94             m_run->setGensteps(m_gen->getInputGensteps());
     95 
     96             m_propagator->propagate();
     97 
     98             if(ok("save"))
     99             {
    100                 m_run->saveEvent();
    101                 if(!production) m_hub->anaEvent();
    102             }
    103 
    104             m_run->resetEvent();
    105         }
    106 
    107         m_ok->postpropagate();
    108     }
    109 }





GScene::debugNodeIntersects from OpticksHub::anaEvent when --dbgnode --gltf 
---------------------------------------------------------------------------------

For gltf branch and --dbgnode > -1

::

    371 void OpticksHub::anaEvent()
    372 {
    373     int dbgnode = m_ok->getDbgNode();
    374     LOG(info) << "OpticksHub::anaEvent"
    375               << " dbgnode " << dbgnode
    376               ;
    377 
    378     if(dbgnode > -1)
    379     {
    380         if(m_gscene)
    381         {
    382             OpticksEvent* evt = m_run->getEvent();
    383             m_gscene->debugNodeIntersects( dbgnode, evt );
    384         }
    385         else
    386         {
    387             LOG(warning) << "--dbgnode only implemented for glTF branch " ;
    388         }
    389     }
    390 
    391 
    392     m_run->anaEvent();
    393 }



How does gltf effect test geometry ?
---------------------------------------- 

* gltf is a switch that uses the GDML parsed geometry inside GScene instead of the 
  GGeo from G4DAE

* what about test geometry ? is there a GScene ? If gltf option is used there will be. 


::

     615 void GGeo::loadAnalyticFromGLTF()
     616 {
     617     LOG(info) << "GGeo::loadAnalyticFromGLTF START" ;
     618     if(!m_ok->isGLTF()) return ;
     619 #ifdef WITH_YoctoGL
     620 
     621     bool loaded = false ;
     622     m_gscene = new GScene(m_ok, this, loaded); // GGeo needed for m_bndlib 
     623 
     624 #else
     625     LOG(fatal) << "GGeo::loadAnalyticFromGLTF requires YoctoGL external " ;
     626     assert(0);
     627 #endif
     628     LOG(info) << "GGeo::loadAnalyticFromGLTF DONE" ;
     629 }


     675 void GGeo::loadAnalyticFromCache()
     676 {
     677     LOG(info) << "GGeo::loadAnalyticFromCache START" ;
     678     m_gscene = GScene::Load(m_ok, this); // GGeo needed for m_bndlib 
     679     LOG(info) << "GGeo::loadAnalyticFromCache DONE" ;
     680 }




::

     552 void GGeo::loadGeometry()
     553 {
     554     bool loaded = isLoaded() ;
     555 
     556     int gltf = m_ok->getGLTF();
     557 
     558     LOG(info) << "GGeo::loadGeometry START"
     559               << " loaded " << loaded
     560               << " gltf " << gltf
     561               ;
     562 
     563     if(!loaded)
     564     {
     565         loadFromG4DAE();
     566         save();
     567 
     568         if(gltf > 0 && gltf < 10)
     569         {
     570             loadAnalyticFromGLTF();
     571             saveAnalytic();
     572         }
     573     }
     574     else
     575     {
     576         loadFromCache();
     577         if(gltf > 0 && gltf < 10)
     578         {
     579             loadAnalyticFromCache();
     580         }
     581     }
     582 
     583     loadAnalyticPmt();
     584 
     585     if( gltf >= 10 )
     586     {
     587         LOG(info) << "GGeo::loadGeometry DEBUFFING loadAnalyticFromGLTF " ;
     588         loadAnalyticFromGLTF();
     589     }
     590 
     591     setupLookup();
     592     setupColors();
     593     setupTyp();
     594     LOG(info) << "GGeo::loadGeometry DONE" ;
     595 }


Migrate GGeoTest to OpticksHub
-----------------------------------

Move GGeoTest to like up in OpticksHub ?

* not OpticksGeometry as that is GGeo tri focussed, whereas
  OpticksHub treats ana and tri on equal footing 


* 1st : get GGeoTest to operate from GGeoBase, required GPmtLib 
* 2nd : move up to OpticksHub 




::

     810 void GGeo::modifyGeometry(const char* config)
     811 {
     812     // NB only invoked with test option : "op --test" 
     813     //   controlled from OpticksGeometry::loadGeometry 
     814 
     815     GGeoTestConfig* gtc = new GGeoTestConfig(config);
     816 
     817     assert(m_geotest == NULL);
     818     m_geotest = new GGeoTest(m_ok, gtc, this);
     819     m_geotest->modifyGeometry();
     820 }



::

    209 void OpticksGeometry::loadGeometry()
    210 {
    211     bool modify = m_ok->hasOpt("test") ;
    212 
    213     LOG(info) << "OpticksGeometry::loadGeometry START, modifyGeometry? " << modify  ;
    214 
    215     loadGeometryBase(); //  usually from cache
    216 
    217     if(!m_ggeo->isValid())
    218     {
    219         LOG(warning) << "OpticksGeometry::loadGeometry finds invalid geometry, try creating geocache with --nogeocache/-G option " ;
    220         m_ok->setExit(true);
    221         return ;
    222     }
    223 
    224     if(modify) modifyGeometry() ;
    225 
    226     // hmm is this modify approach still needed ? perhaps just loadTestGeometry ?
    227     // probably the issue is GGeo does too much ...
    228 
    229 
    230     fixGeometry();
    231 
    232     registerGeometry();
    233 
    234     if(!m_ok->isGeocache())
    235     {
    236         LOG(info) << "OpticksGeometry::loadGeometry early exit due to --nogeocache/-G option " ;
    237         m_ok->setExit(true);
    238     }
    239 
    240     // configureGeometry();  moved up to OpticksHub::init 
    241 
    242     LOG(info) << "OpticksGeometry::loadGeometry DONE " ;
    243     TIMER("loadGeometry");
    244 }



postpropagate
----------------

postpropagate currently just looking a time/memory profiles

::

    simon:env blyth$ opticks-find postprop
    ./cfg4/CG4.cc:    postpropagate();
    ./cfg4/CG4.cc:void CG4::postpropagate()
    ./cfg4/CG4.cc:    LOG(info) << "CG4::postpropagate(" << m_ok->getTagOffset() << ")"  ;
    ./cfg4/CG4.cc:    dynamic_cast<CSteppingAction*>(m_sa)->report("CG4::postpropagate");
    ./cfg4/CG4.cc:    LOG(info) << "CG4::postpropagate(" << m_ok->getTagOffset() << ") DONE"  ;
    ./cfg4/tests/CG4Test.cc:    ok.postpropagate();
    ./ok/OKMgr.cc:        m_ok->postpropagate();
    ./okg4/OKG4Mgr.cc:        m_ok->postpropagate();
    ./okop/OpMgr.cc:            m_ok->postpropagate();
    ./okop/OpMgr.cc:        m_ok->postpropagate();
    ./okop/tests/OpSeederTest.cc:    ok.postpropagate();
    ./optickscore/Opticks.cc:void Opticks::postpropagate()
    ./optickscore/Opticks.cc:   dumpProfile("Opticks::postpropagate", NULL, "OpticksRun::createEvent.BEG", 0.0001 );  // spacwith spacing at start if each evt
    ./optickscore/Opticks.cc:   dumpProfile("Opticks::postpropagate", "OPropagator::launch");  
    ./optickscore/Opticks.cc:   dumpProfile("Opticks::postpropagate", "CG4::propagate");  
    ./optickscore/Opticks.cc:   dumpParameters("Opticks::postpropagate");
    ./cfg4/CG4.hh:        void postpropagate();
    ./optickscore/Opticks.hh:       void postpropagate();
    simon:opticks blyth$ 




geometry --test with --gltf 1 asserts
--------------------------------------

Huh : which GGeoLib should --test --gltf 1 modify ?

::

    simon:opticks blyth$ tlens-concave --gltf 1 -D

    ...


    2017-10-24 11:43:32.218 INFO  [407607] [GParts::add@736]  n0   1 n1   2 num_part_add   1 num_tran_add   1 num_plan_add   0 other_part_buffer  1,4,4 other_tran_buffer  1,3,4,4 other_plan_buffer  0,4
    2017-10-24 11:43:32.219 INFO  [407607] [GMergedMesh::dumpSolids@707] GMergedMesh::combine (combined result)  ce0 gfloat4      0.000      0.000   -250.000    750.000 
        0 ce             gfloat4      0.000      0.000   -250.000    750.000  bb  mn (  -500.000   -500.000  -1000.000) mx (   500.000    500.000    500.000)
        1 ce             gfloat4      0.000      0.000      0.000    500.000  bb  mn (  -500.000   -500.000   -500.000) mx (   500.000    500.000    500.000)
        0 ni[nf/nv/nidx/pidx] ( 12, 36,  1,4294967295)  id[nidx,midx,bidx,sidx]  (  1,  1,123,  0) 
        1 ni[nf/nv/nidx/pidx] (3884,11652,  0,4294967295)  id[nidx,midx,bidx,sidx]  (  0,  0,124,  0) 
    2017-10-24 11:43:32.220 INFO  [407607] [*GGeoTest::create@152] GGeoTest::create DONE  mode PyCsgInBox
    2017-10-24 11:43:32.220 INFO  [407607] [OpticksGeometry::loadGeometry@242] OpticksGeometry::loadGeometry DONE 
    2017-10-24 11:43:32.220 INFO  [407607] [OpticksHub::loadGeometry@293] OpticksHub::loadGeometry DONE
    2017-10-24 11:43:32.220 INFO  [407607] [OpticksHub::configureGeometryTriAna@332] OpticksHub::configureGeometryTriAna restrict_mesh -1 desc OpticksHub m_ggeo 0x105e08210 m_gscene 0x1095e07b0 m_geometry 0x105e04df0 m_gen 0x0 m_gun 0x0
    2017-10-24 11:43:32.220 FATAL [407607] [OpticksHub::configureGeometryTriAna@349] OpticksHub::configureGeometryTriAna MISMATCH  nmm_a 6 nmm_t 1

    // 6 ? hmm looks like it modified the tri : should be 1:1  

    Assertion failed: (match), function configureGeometryTriAna, file /Users/blyth/opticks/opticksgeo/OpticksHub.cc, line 356.
    ...
        frame #4: 0x00000001022ae29c libOpticksGeometry.dylib`OpticksHub::configureGeometryTriAna(this=0x0000000105e00180) + 1132 at OpticksHub.cc:356
        frame #5: 0x00000001022ad138 libOpticksGeometry.dylib`OpticksHub::configureGeometry(this=0x0000000105e00180) + 56 at OpticksHub.cc:306
        frame #6: 0x00000001022ac006 libOpticksGeometry.dylib`OpticksHub::init(this=0x0000000105e00180) + 86 at OpticksHub.cc:103
        frame #7: 0x00000001022abf00 libOpticksGeometry.dylib`OpticksHub::OpticksHub(this=0x0000000105e00180, ok=0x0000000105c222b0) + 432 at OpticksHub.cc:88
        frame #8: 0x00000001022ac0ed libOpticksGeometry.dylib`OpticksHub::OpticksHub(this=0x0000000105e00180, ok=0x0000000105c222b0) + 29 at OpticksHub.cc:90
        frame #9: 0x0000000103c4d1b6 libOK.dylib`OKMgr::OKMgr(this=0x00007fff5fbfe538, argc=27, argv=0x00007fff5fbfe610, argforced=0x0000000000000000) + 262 at OKMgr.cc:46
        frame #10: 0x0000000103c4d61b libOK.dylib`OKMgr::OKMgr(this=0x00007fff5fbfe538, argc=27, argv=0x00007fff5fbfe610, argforced=0x0000000000000000) + 43 at OKMgr.cc:49
        frame #11: 0x000000010000adad OKTest`main(argc=27, argv=0x00007fff5fbfe610) + 1373 at OKTest.cc:58
        frame #12: 0x00007fff880d35fd libdyld.dylib`start + 1
    (lldb) 







for intersect checking with test geometry the GScene aint very helpful
-------------------------------------------------------------------------

* need to put fingers on the nnode SDF for the test geometry 
* split off anaEvent handling for ggeotest 

::

    tlens-;tlens-concave --gltf 1 --dbgnode 1 -D







