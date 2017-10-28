surface_review
================





CDetector
------------

::

    036 CDetector::CDetector(OpticksHub* hub, OpticksQuery* query)
     37   :
     38   m_hub(hub),
     39   m_ok(m_hub->getOpticks()),
     40   m_ggeo(m_hub->getGGeo()),
     41   m_blib(new CBndLib(m_hub)),
     42   m_gsurlib(m_hub->getSurLib()),   // invokes the deferred GGeo::createSurLib  
     43   m_csurlib(NULL),

    621 GSurLib* OpticksHub::getSurLib()
    622 {
    623     return m_ggeo ? m_ggeo->getSurLib() : NULL ;
    624 }


GSurLib
---------

::

     250 GSurLib* GGeo::getSurLib()
     251 {
     252     if(m_surlib == NULL) createSurLib();
     253     return m_surlib ;
     254 }

::

     701 void GGeo::createSurLib()
     702 {
     703 /*
     704     This is deferred until called upon by CG4/CGeometry so any test geometry mesh0 modifications 
     705     will have been done already when called...
     706 
     707     frame #4: 0x0000000101d1cfce libGGeo.dylib`GGeo::createSurLib(this=0x0000000109900410) + 46 at GGeo.cc:637
     708     frame #5: 0x0000000101d1cf8e libGGeo.dylib`GGeo::getSurLib(this=0x0000000109900410) + 46 at GGeo.cc:259
     709     frame #6: 0x0000000103e2eebb libcfg4.dylib`CGeometry::CGeometry(this=0x000000010e38fa60, hub=0x000000010980b170) + 91 at CGeometry.cc:33
     710     frame #7: 0x0000000103e2f56d libcfg4.dylib`CGeometry::CGeometry(this=0x000000010e38fa60, hub=0x000000010980b170) + 29 at CGeometry.cc:43
     711     frame #8: 0x0000000103ec8cc9 libcfg4.dylib`CG4::CG4(this=0x000000010e153de0, hub=0x000000010980b170) + 217 at CG4.cc:113
     712     frame #9: 0x0000000103ec919d libcfg4.dylib`CG4::CG4(this=0x000000010e153de0, hub=0x000000010980b170) + 29 at CG4.cc:134
     713     frame #10: 0x0000000103faffb3 libokg4.dylib`OKG4Mgr::OKG4Mgr(this=0x00007fff5fbfe660, argc=21, argv=0x00007fff5fbfe748) + 547 at OKG4Mgr.cc:35
     714     frame #11: 0x0000000103fb0203 libokg4.dylib`OKG4Mgr::OKG4Mgr(this=0x00007fff5fbfe660, argc=21, argv=0x00007fff5fbfe748) + 35 at OKG4Mgr.cc:41
     715     frame #12: 0x00000001000139be OKG4Test`main(argc=21, argv=0x00007fff5fbfe748) + 1486 at OKG4Test.cc:56
     716 */
     717 
     718     if(m_surlib)
     719     {
     720         LOG(warning) << "recreating GSurLib" ;
     721         delete m_surlib ;
     722     }
     723     else
     724     {
     725         LOG(info) << "deferred creation of GSurLib " ;
     726     }
     727 
     728     m_surlib = new GSurLib(this) ;
     729     //m_surlib->dump("GGeo::createSurLib");
     730 }

