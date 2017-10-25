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
It needs to be found and elevated to be available as a standard option.

* mentioned in :doc:`analytic_geometry_shakedown` see --dbgnode option within --gltf branch.


Ideas
------

* recording node indices together with intersects (in some intersect debug mode)
  would allow SDF zero checks of all intersects, even in full geometries



FIXED : incompatibility of --test geometry and --gltf
---------------------------------------------------------

Fix required some refactoring, breaking somes pieces off 
the GGeo monolith  :doc:`breaking_up_GGeo_monolith`

* chop GPmtLib out of GGeo 
* repurpose OpticksGeometry to be more of an OpticksGGeo
* chop OpticksAim out of OpticksGeometry and OpticksHub 
* migrate GGeoTest from GGeo up to OpticksHub 


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

Move GGeoTest to live up in OpticksHub ?

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




FIXED : geometry --test with --gltf 1 asserts
------------------------------------------------

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



FIXED : for intersect checking with test geometry the GScene::anaEvent aint very helpful
--------------------------------------------------------------------------------------------

* need to put fingers on the nnode SDF for the test geometry
* split off anaEvent handling for ggeotest into  GGeoTest::anaEvent

::

    tlens-;tlens-concave --gltf 1 --dbgnode 1 -D   ## huh: OpenGL viz not working with gltf 1 ?
    tlens-;tlens-concave --dbgnode 1 --dbgseqhis  






::


    tlens-;tlens-concave --dbgnode 1 --dbgseqhis 0x8ccd

        ## note order reversal, node 1 is the container box
        ## every few exc : because most photons end up being SA absorbed on the container walls

    2017-10-25 13:28:13.039 INFO  [712140] [OpticksEventAna::dumpExcursions@120] OpticksEventAna::dumpExcursions seqhis ending AB or truncated seqhis : exc expected 
     seqhis               4d                 TO AB                                            tot     10 exc     10 exc/tot  1.000
     seqhis              4cd                 TO BT AB                                         tot     90 exc     90 exc/tot  1.000
     seqhis              86d                 TO SC SA                                         tot     65 exc      0 exc/tot  0.000
     seqhis              8bd                 TO BR SA                                         tot  29493 exc      0 exc/tot  0.000
     seqhis             4bcd                 TO BT BR AB                                      tot      4 exc      4 exc/tot  1.000
     seqhis             4ccd                 TO BT BT AB                                      tot     13 exc     13 exc/tot  1.000
     seqhis             86bd                 TO BR SC SA                                      tot      4 exc      0 exc/tot  0.000
     seqhis             8b6d                 TO SC BR SA                                      tot      4 exc      0 exc/tot  0.000
     seqhis             8ccd                 TO BT BT SA                                      tot 442101 exc      0 exc/tot  0.000
     seqhis            4bbcd                 TO BT BR BR AB                                   tot      1 exc      1 exc/tot  1.000
     seqhis            4cbcd                 TO BT BR BT AB                                   tot      1 exc      1 exc/tot  1.000
     seqhis            86ccd                 TO BT BT SC SA                                   tot    123 exc      0 exc/tot  0.000
     seqhis            8c6cd                 TO BT SC BT SA                                   tot     38 exc      0 exc/tot  0.000
     seqhis            8cbcd                 TO BT BR BT SA                                   tot  26267 exc      0 exc/tot  0.000
     seqhis            8cc6d                 TO SC BT BT SA                                   tot     24 exc      0 exc/tot  0.000
     seqhis           86cbcd                 TO BT BR BT SC SA                                tot     12 exc      0 exc/tot  0.000
     seqhis           8b6ccd                 TO BT BT SC BR SA                                tot      9 exc      0 exc/tot  0.000
     seqhis           8c6bcd                 TO BT BR SC BT SA                                tot      2 exc      0 exc/tot  0.000
     seqhis           8cb6cd                 TO BT SC BR BT SA                                tot     30 exc      0 exc/tot  0.000
     seqhis           8cbbcd                 TO BT BR BR BT SA                                tot   1522 exc      0 exc/tot  0.000
     seqhis           8cbc6d                 TO SC BT BR BT SA                                tot      4 exc      0 exc/tot  0.000
     seqhis           8cc6bd                 TO BR SC BT BT SA                                tot      5 exc      0 exc/tot  0.000
     seqhis          86cbbcd                 TO BT BR BR BT SC SA                             tot      1 exc      0 exc/tot  0.000
     seqhis          8cbb6cd                 TO BT SC BR BR BT SA                             tot      2 exc      0 exc/tot  0.000
     seqhis          8cbbbcd                 TO BT BR BR BR BT SA                             tot     82 exc      0 exc/tot  0.000
     seqhis          8cbbc6d                 TO SC BT BR BR BT SA                             tot      2 exc      0 exc/tot  0.000
     seqhis          8cbc6bd                 TO BR SC BT BR BT SA                             tot      1 exc      0 exc/tot  0.000
     seqhis          8cc6ccd                 TO BT BT SC BT BT SA                             tot     30 exc      0 exc/tot  0.000
     seqhis         8cbbb6cd                 TO BT SC BR BR BR BT SA                          tot      3 exc      0 exc/tot  0.000
     seqhis         8cbbbbcd                 TO BT BR BR BR BR BT SA                          tot      5 exc      0 exc/tot  0.000
     seqhis         8cbbbc6d                 TO SC BT BR BR BR BT SA                          tot      1 exc      0 exc/tot  0.000
     seqhis         8cbc6ccd                 TO BT BT SC BT BR BT SA                          tot     16 exc      0 exc/tot  0.000
     seqhis         8cc6cbcd                 TO BT BR BT SC BT BT SA                          tot      3 exc      0 exc/tot  0.000
     seqhis        8cbbb6bcd                 TO BT BR SC BR BR BR BT SA                       tot      1 exc      0 exc/tot  0.000
     seqhis        8cbbbb6cd                 TO BT SC BR BR BR BR BT SA                       tot      1 exc      0 exc/tot  0.000
     seqhis        8cbbbbbcd                 TO BT BR BR BR BR BR BT SA                       tot      1 exc      0 exc/tot  0.000
     seqhis        8cbc6cbcd                 TO BT BR BT SC BT BR BT SA                       tot      1 exc      0 exc/tot  0.000
     seqhis       8cbbbbb6cd                 TO BT SC BR BR BR BR BR BT SA                    tot      1 exc      0 exc/tot  0.000
     seqhis       8cbbc6cbcd                 TO BT BR BT SC BT BR BR BT SA                    tot      1 exc      0 exc/tot  0.000
     seqhis       bbbbbb6bcd                 TO BT BR SC BR BR BR BR BR BR                    tot      2 exc      2 exc/tot  1.000
     seqhis       bbbbbbb6cd                 TO BT SC BR BR BR BR BR BR BR                    tot     24 exc     24 exc/tot  1.000




    tlens-;tlens-concave --dbgnode 0 --dbgseqhis 0x8ccd

         ## 0 : is the lens (which whilst warming up is just a cylinder)
         ## 1 : is container box

         ## almost everything is OFF the lens, because the photons end up absorbed on container walls
         ## only a few truncated BR end with photon positions on the object 

         ## for intersect checking, either look at less precise step-by-step records or change object to have a perfectAbsorber


    2017-10-25 13:28:29.159 INFO  [712523] [OpticksEventAna::dumpExcursions@120] OpticksEventAna::dumpExcursions seqhis ending AB or truncated seqhis : exc expected 
     seqhis               4d                 TO AB                                            tot     10 exc     10 exc/tot  1.000
     seqhis              4cd                 TO BT AB                                         tot     90 exc     90 exc/tot  1.000
     seqhis              86d                 TO SC SA                                         tot     65 exc     65 exc/tot  1.000
     seqhis              8bd                 TO BR SA                                         tot  29493 exc  29493 exc/tot  1.000
     seqhis             4bcd                 TO BT BR AB                                      tot      4 exc      4 exc/tot  1.000
     seqhis             4ccd                 TO BT BT AB                                      tot     13 exc     13 exc/tot  1.000
     seqhis             86bd                 TO BR SC SA                                      tot      4 exc      4 exc/tot  1.000
     seqhis             8b6d                 TO SC BR SA                                      tot      4 exc      4 exc/tot  1.000
     seqhis             8ccd                 TO BT BT SA                                      tot 442101 exc 442101 exc/tot  1.000
     seqhis            4bbcd                 TO BT BR BR AB                                   tot      1 exc      1 exc/tot  1.000
     seqhis            4cbcd                 TO BT BR BT AB                                   tot      1 exc      1 exc/tot  1.000
     seqhis            86ccd                 TO BT BT SC SA                                   tot    123 exc    123 exc/tot  1.000
     seqhis            8c6cd                 TO BT SC BT SA                                   tot     38 exc     38 exc/tot  1.000
     seqhis            8cbcd                 TO BT BR BT SA                                   tot  26267 exc  26267 exc/tot  1.000
     seqhis            8cc6d                 TO SC BT BT SA                                   tot     24 exc     24 exc/tot  1.000
     seqhis           86cbcd                 TO BT BR BT SC SA                                tot     12 exc     12 exc/tot  1.000
     seqhis           8b6ccd                 TO BT BT SC BR SA                                tot      9 exc      9 exc/tot  1.000
     seqhis           8c6bcd                 TO BT BR SC BT SA                                tot      2 exc      2 exc/tot  1.000
     seqhis           8cb6cd                 TO BT SC BR BT SA                                tot     30 exc     30 exc/tot  1.000
     seqhis           8cbbcd                 TO BT BR BR BT SA                                tot   1522 exc   1522 exc/tot  1.000
     seqhis           8cbc6d                 TO SC BT BR BT SA                                tot      4 exc      4 exc/tot  1.000
     seqhis           8cc6bd                 TO BR SC BT BT SA                                tot      5 exc      5 exc/tot  1.000
     seqhis          86cbbcd                 TO BT BR BR BT SC SA                             tot      1 exc      1 exc/tot  1.000
     seqhis          8cbb6cd                 TO BT SC BR BR BT SA                             tot      2 exc      2 exc/tot  1.000
     seqhis          8cbbbcd                 TO BT BR BR BR BT SA                             tot     82 exc     82 exc/tot  1.000
     seqhis          8cbbc6d                 TO SC BT BR BR BT SA                             tot      2 exc      2 exc/tot  1.000
     seqhis          8cbc6bd                 TO BR SC BT BR BT SA                             tot      1 exc      1 exc/tot  1.000
     seqhis          8cc6ccd                 TO BT BT SC BT BT SA                             tot     30 exc     30 exc/tot  1.000
     seqhis         8cbbb6cd                 TO BT SC BR BR BR BT SA                          tot      3 exc      3 exc/tot  1.000
     seqhis         8cbbbbcd                 TO BT BR BR BR BR BT SA                          tot      5 exc      5 exc/tot  1.000
     seqhis         8cbbbc6d                 TO SC BT BR BR BR BT SA                          tot      1 exc      1 exc/tot  1.000
     seqhis         8cbc6ccd                 TO BT BT SC BT BR BT SA                          tot     16 exc     16 exc/tot  1.000
     seqhis         8cc6cbcd                 TO BT BR BT SC BT BT SA                          tot      3 exc      3 exc/tot  1.000
     seqhis        8cbbb6bcd                 TO BT BR SC BR BR BR BT SA                       tot      1 exc      1 exc/tot  1.000
     seqhis        8cbbbb6cd                 TO BT SC BR BR BR BR BT SA                       tot      1 exc      1 exc/tot  1.000
     seqhis        8cbbbbbcd                 TO BT BR BR BR BR BR BT SA                       tot      1 exc      1 exc/tot  1.000
     seqhis        8cbc6cbcd                 TO BT BR BT SC BT BR BT SA                       tot      1 exc      1 exc/tot  1.000
     seqhis       8cbbbbb6cd                 TO BT SC BR BR BR BR BR BT SA                    tot      1 exc      1 exc/tot  1.000
     seqhis       8cbbc6cbcd                 TO BT BR BT SC BT BR BR BT SA                    tot      1 exc      1 exc/tot  1.000
     seqhis       bbbbbb6bcd                 TO BT BR SC BR BR BR BR BR BR                    tot      2 exc      0 exc/tot  0.000
     seqhis       bbbbbbb6cd                 TO BT SC BR BR BR BR BR BR BR                    tot     24 exc      0 exc/tot  0.000





Interp Record Data
--------------------

* done at python level, but need in C++ for easy comparison against SDFs, or get SDFs into python ?


::

    simon:optickscore blyth$ opticks-find getRecordData
    ./cfg4/CRecorder.cc:    m_records = m_evt->getRecordData();
    ./ggeo/tests/RecordsNPYTest.cc:    NPY<short>* rx = evt->getRecordData();
    ./oglrap/Rdr.cc:    NPY<short>* rx = evt->getRecordData();
    ./okop/OpZeroer.cc:    NPY<short>* record = evt->getRecordData(); 
    ./optickscore/OpticksEvent.cc:NPY<short>* OpticksEvent::getRecordData()
    ./optickscore/OpticksEvent.cc:    NPY<short>* rx = getRecordData() ;
    ./optickscore/OpticksEvent.cc:    NPY<short>* rx = getRecordData();    
    ./optickscore/tests/OpticksEventTest.cc:    NPY<short>* rx = m_evt->getRecordData();
    ./opticksgeo/OpticksIdx.cc:    NPY<short>* rx = evt->getRecordData();
    ./optixrap/OEvent.cc:    NPY<short>* rx = evt->getRecordData() ;
    ./optixrap/OEvent.cc:    NPY<short>* rx = evt->getRecordData() ; 
    ./optixrap/OEvent.cc:        NPY<short>* rx = evt->getRecordData();
    ./optickscore/OpticksEvent.hh:       NPY<short>*          getRecordData();
    simon:opticks blyth$ 



tlens.py step-by-step rec positions
------------------------------------

::

    tlens-;tlens-ipy
    ...

    In [13]: evt.rpost_(slice(0,5))
    Out[13]: 
    A()sliced
    A([[[     -65.844,   -7.065, -299.997,    0.1  ],
            [ -65.844,   -7.065,  -99.994,    0.767],
            [ -65.844,   -7.065,   99.994,    1.905],
            [ -65.844,   -7.065,  500.   ,    3.239],
            [   0.   ,    0.   ,    0.   ,    0.   ]],

           [[ -33.967,   58.733, -299.997,    0.1  ],
            [ -33.967,   58.733,  -99.994,    0.767],
            [ -33.967,   58.733,   99.994,    1.889],
            [ -33.967,   58.733,  500.   ,    3.223],
            [   0.   ,    0.   ,    0.   ,    0.   ]],

           [[  20.02 ,   45.839, -299.997,    0.1  ],
            [  20.02 ,   45.839,  -99.994,    0.767],
            [  20.02 ,   45.839,   99.994,    1.91 ],
            [  20.02 ,   45.839,  500.   ,    3.244],
            [   0.   ,    0.   ,    0.   ,    0.   ]],



::

    1373     def rpost_(self, recs):
    1374         """
    1375         NB recs can be a slice, eg slice(0,5) for 1st 5 step records of each photon
    ....
    1390         """
    1391         center, extent = self.post_center_extent()
    1392         p = self.rx[:,recs,0].astype(np.float32)*extent/32767.0 + center
    1393         return p
    1394 

    1203     def rpolw_(self, recs):
    1204         """
    1205         Unlike rpol_ this works with irec slices, 
    1206         BUT note that the wavelength returned in 4th column is 
    1207         not decompressed correctly.
    1208         Due to shape shifting it is not easy to remove
    1209         """
    1210         return self.rx[:,recs,1,0:2].copy().view(np.uint8).astype(np.float32)/127.-1.



::

    In [16]: evt.rx.shape
    Out[16]: (500000, 10, 2, 4)


::

    2017-10-25 16:12:02.552 INFO  [768165] [OpticksEventDump::dumpRecords@56] OpticksEventDump::dumpRecords  shape 500000,10,2,4




tlens-ana
-------------

* dumpExcursions is for final photon position... so not so informative
* dumpStepByStepCSGExcursion shows SDF values for each tree for each rec pos : need cumulative way to represent that 


::

    simon:opticksnpy blyth$ tlens-ana
    2017-10-25 20:20:42.751 INFO  [877648] [Opticks::dumpArgs@792] Opticks::configure argc 10
      0 : OpticksEventAnaTest
      1 : --torch
      2 : --tag
      3 : 1
      4 : --cat
      5 : lens
      6 : --dbgcsgpath
      7 : /tmp/blyth/opticks/tlens-concave--
      8 : --dbgnode
      9 : 0
    ...
    2017-10-25 20:20:44.285 INFO  [877648] [OpticksEventDump::setupRecordsNPY@135] OpticksEvent::setupRecordsNPY  shape 500000,10,2,4
    2017-10-25 20:20:44.305 INFO  [877648] [OpticksEventAna::countExcursions@94] OpticksEventAna::countExcursions pho_num 500000 epsilon 0.1 dbgseqhis 0 dbgseqhis                                                 
    2017-10-25 20:20:44.489 INFO  [877648] [OpticksEventAna::countExcursions@137] OpticksEventAna::countExcursions pho_num 500000 dbgseqhis 0 dbgseqhis                                                  count 0
    2017-10-25 20:20:44.729 INFO  [877648] [OpticksEventAna::countExcursions@137] OpticksEventAna::countExcursions pho_num 500000 dbgseqhis 0 dbgseqhis                                                  count 0
    2017-10-25 20:20:44.729 INFO  [877648] [OpticksEventAna::dump@60] GGeoTest::anaEvent OpticksEventAna pho 500000,4,4 seq 500000,1,2

    2017-10-25 20:20:44.729 INFO  [877648] [OpticksEventAna::dumpExcursions@153] OpticksEventAna::dumpExcursions seqhis ending AB or truncated seqhis : exc expected 
    2017-10-25 20:20:44.729 INFO  [877648] [OpticksEventAna::dumpExcursions@163]  counts and fractions on surface of each tree (within SDF epsilon) 

     seqhis             8ccd                 TO BT BT SA                                      tot 442101 surf (       0 442101 )  frac (   0.000  1.000 ) 
     seqhis              8bd                 TO BR SA                                         tot  29493 surf (       0  29493 )  frac (   0.000  1.000 ) 
     seqhis            8cbcd                 TO BT BR BT SA                                   tot  26267 surf (       0  26267 )  frac (   0.000  1.000 ) 
     seqhis           8cbbcd                 TO BT BR BR BT SA                                tot   1522 surf (       0   1522 )  frac (   0.000  1.000 ) 
     seqhis            86ccd                 TO BT BT SC SA                                   tot    123 surf (       0    123 )  frac (   0.000  1.000 ) 
     seqhis              4cd                 TO BT AB                                         tot     90 surf (       0      0 )  frac (   0.000  0.000 ) 
     seqhis          8cbbbcd                 TO BT BR BR BR BT SA                             tot     82 surf (       0     82 )  frac (   0.000  1.000 ) 
     seqhis              86d                 TO SC SA                                         tot     65 surf (       0     65 )  frac (   0.000  1.000 ) 
     seqhis            8c6cd                 TO BT SC BT SA                                   tot     38 surf (       0     38 )  frac (   0.000  1.000 ) 
     seqhis          8cc6ccd                 TO BT BT SC BT BT SA                             tot     30 surf (       0     30 )  frac (   0.000  1.000 ) 
     seqhis           8cb6cd                 TO BT SC BR BT SA                                tot     30 surf (       0     30 )  frac (   0.000  1.000 ) 
     seqhis       bbbbbbb6cd                 TO BT SC BR BR BR BR BR BR BR                    tot     24 surf (      24      0 )  frac (   1.000  0.000 ) 
     seqhis            8cc6d                 TO SC BT BT SA                                   tot     24 surf (       0     24 )  frac (   0.000  1.000 ) 
     seqhis         8cbc6ccd                 TO BT BT SC BT BR BT SA                          tot     16 surf (       0     16 )  frac (   0.000  1.000 ) 
     seqhis             4ccd                 TO BT BT AB                                      tot     13 surf (       0      0 )  frac (   0.000  0.000 ) 
     seqhis           86cbcd                 TO BT BR BT SC SA                                tot     12 surf (       0     12 )  frac (   0.000  1.000 ) 
     seqhis               4d                 TO AB                                            tot     10 surf (       0      0 )  frac (   0.000  0.000 ) 

    2017-10-25 20:20:44.730 INFO  [877648] [OpticksEventAna::dumpStepByStepCSGExcursions@207] OpticksEventAna::dumpStepByStepCSGExcursions photon_id 0 num_pos 10 num_tree 2 seqhis             8ccd                 TO BT BT SA                                     
          post-65.844  -7.065 -299.997   0.100     199.997   -200.003
          post-65.844  -7.065 -99.994   0.767      -0.006   -400.006
          post-65.844  -7.065  99.994   1.905      -0.006   -400.006
          post-65.844  -7.065 500.000   3.239     400.000      0.000
          post  0.000   0.000   0.000   0.000 
          post  0.000   0.000   0.000   0.000 
          post  0.000   0.000   0.000   0.000 
          post  0.000   0.000   0.000   0.000 
          post  0.000   0.000   0.000   0.000 
          post  0.000   0.000   0.000   0.000 
    2017-10-25 20:20:44.730 INFO  [877648] [OpticksEventAna::dumpStepByStepCSGExcursions@207] OpticksEventAna::dumpStepByStepCSGExcursions photon_id 1 num_pos 10 num_tree 2 seqhis             8ccd                 TO BT BT SA                                     
          post-33.967  58.733 -299.997   0.100     199.997   -200.003
          post-33.967  58.733 -99.994   0.767      -0.006   -400.006
          post-33.967  58.733  99.994   1.889      -0.006   -400.006
          post-33.967  58.733 500.000   3.223     400.000      0.000
          post  0.000   0.000   0.000   0.000 
          post  0.000   0.000   0.000   0.000 
          post  0.000   0.000   0.000   0.000 
          post  0.000   0.000   0.000   0.000 
          post  0.000   0.000   0.000   0.000 
          post  0.000   0.000   0.000   0.000 
    2017-10-25 20:20:44.730 INFO  [877648] [OpticksEventAna::dumpStepByStepCSGExcursions@207] OpticksEventAna::dumpStepByStepCSGExcursions photon_id 2 num_pos 10 num_tree 2 seqhis             8ccd                 TO BT BT SA                                     
          post 20.020  45.839 -299.997   0.100     199.997   -200.003
          post 20.020  45.839 -99.994   0.767      -0.006   -400.006
          post 20.020  45.839  99.994   1.910      -0.006   -400.006
          post 20.020  45.839 500.000   3.244     400.000      0.000
          post  0.000   0.000   0.000   0.000 
          post  0.000   0.000   0.000   0.000 
          post  0.000   0.000   0.000   0.000 
          post  0.000   0.000   0.000   0.000 
          post  0.000   0.000   0.000   0.000 
          post  0.000   0.000   0.000   0.000 

