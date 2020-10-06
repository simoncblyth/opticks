GGeoConvertTest_fails_with_xanalytic
======================================


FIXED
-------- 

::

     05 int main(int argc, char** argv)
      6 {
      7     OPTICKS_LOG(argc, argv);
      8 
      9     Opticks ok(argc, argv);
     10     ok.configure(); 
     11     
     12     GGeo gg(&ok);
     13     gg.loadFromCache(); 
     14     gg.dumpStats();
     15 
     16         
     17     // TODO: relocate below preps into loadFromCache ? but also needed for running  from GDML : need a higher level method than loadFromCache ?
     18     gg.close();                  // normally OpticksHub::loadGeometry
     19     gg.deferredCreateGParts();   // normally OpticksHub::init 
     20         
     21         
     22     gg.dryrun_convert();
     23             
     24     return 0 ;
     25 }       
     26     




Issue : pts assert, presumbly lack of GGeo::deferredCreateGParts ?
------------------------------------------------------------------------

::

   GGeoLib=INFO lldb_ -- GGeoConvertTest --xanalytic

   ...
     num_total_volumes 12230 num_instanced_volumes 7744 num_global_volumes 4486 num_total_faces 483996 num_total_faces_woi 2533452 (woi:without instancing) 
       0 pts Y  GPts.NumPt  4486 lvIdx ( 248 247 21 0 7 6 3 2 3 2 ... 237 238 239 240 241 242 243 244 245)
       1 pts Y  GPts.NumPt     1 lvIdx ( 1)
       2 pts Y  GPts.NumPt     1 lvIdx ( 197)
       3 pts Y  GPts.NumPt     1 lvIdx ( 198)
       4 pts Y  GPts.NumPt     1 lvIdx ( 195)
       5 pts Y  GPts.NumPt     5 lvIdx ( 47 46 43 44 45)
    2020-10-06 11:09:44.245 INFO  [2365027] [GGeoLib::dryrun_convert@459] [ nmm 6
    2020-10-06 11:09:44.245 INFO  [2365027] [GGeoLib::dryrun_convertMergedMesh@471] ( 0
    2020-10-06 11:09:44.245 INFO  [2365027] [GGeoLib::dryrun_makeOGeometry@563] ugeocode [A]
    Assertion failed: (pts && "GMergedMesh with GEOCODE_ANALYTIC must have associated GParts, see GGeo::modifyGeometry "), function dryrun_makeAnalyticGeometry, file /Users/blyth/opticks/ggeo/GGeoLib.cc, line 620.
    (lldb) bt
        frame #3: 0x00007fff589ec1ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x000000010023771b libGGeo.dylib`GGeoLib::dryrun_makeAnalyticGeometry(this=0x0000000101c0b670, mm=0x0000000101c0c090) at GGeoLib.cc:620
        frame #5: 0x0000000100236e91 libGGeo.dylib`GGeoLib::dryrun_makeOGeometry(this=0x0000000101c0b670, mm=0x0000000101c0c090) at GGeoLib.cc:571
        frame #6: 0x00000001002366ba libGGeo.dylib`GGeoLib::dryrun_makeGlobalGeometryGroup(this=0x0000000101c0b670, mm=0x0000000101c0c090) at GGeoLib.cc:503
        frame #7: 0x0000000100236641 libGGeo.dylib`GGeoLib::dryrun_convertMergedMesh(this=0x0000000101c0b670, i=0) at GGeoLib.cc:491
        frame #8: 0x0000000100236065 libGGeo.dylib`GGeoLib::dryrun_convert(this=0x0000000101c0b670) at GGeoLib.cc:463
        frame #9: 0x000000010027a85c libGGeo.dylib`GGeo::dryrun_convert(this=0x00007ffeefbfe668) at GGeo.cc:2152
        frame #10: 0x00000001000063d5 GGeoConvertTest`main(argc=2, argv=0x00007ffeefbfe9c0) at GGeoConvertTest.cc:16
        frame #11: 0x00007fff58978015 libdyld.dylib`start + 1
    (lldb) 


    (lldb) f 4
    frame #4: 0x000000010023771b libGGeo.dylib`GGeoLib::dryrun_makeAnalyticGeometry(this=0x0000000101c0b670, mm=0x0000000101c0c090) at GGeoLib.cc:620
       617 	    bool dbgmm = m_ok->getDbgMM() == int(mm->getIndex()) ;
       618 	    bool dbganalytic = m_ok->hasOpt("dbganalytic") ;
       619 	
    -> 620 	    GParts* pts = mm->getParts(); assert(pts && "GMergedMesh with GEOCODE_ANALYTIC must have associated GParts, see GGeo::modifyGeometry ");
       621 	
       622 	    if(pts->getPrimBuffer() == NULL)
       623 	    {
    (lldb) 



Need to reposition the invokation to make GGeo more self contained::

    epsilon:opticks blyth$ opticks-f deferredCreateGParts
    ./opticksgeo/OpticksHub.cc:    deferredGeometryPrep();    // GGeo::deferredCreateGParts
    ./opticksgeo/OpticksHub.cc:    m_ggeo->deferredCreateGParts() ;    
    ./ggeo/GGeo.hh:        void deferredCreateGParts(); 
    ./ggeo/GGeo.cc:GGeo::deferredCreateGParts
    ./ggeo/GGeo.cc:void GGeo::deferredCreateGParts()
    ./ggeo/GGeoTest.cc:    // below normally done in  GGeo::deferredCreateGParts  when not --test
    ./ggeo/GPts.hh:This GParts creation is done in GGeo::deferredCreateGParts
    ./notes/issues/GPts_GParts_optimization.rst:Mostly from GPts assert in GGeo::deferredCreateGParts::
    ./notes/issues/GGeo-OGeo-identity-direct-review.rst:     829     m_ggeo->deferredCreateGParts() ;
    ./notes/issues/GGeo-OGeo-identity-direct-review.rst:    1484 void GGeo::deferredCreateGParts()
    ./notes/issues/GGeoConvertTest_fails_with_xanalytic.rst:Issue : pts assert, presumbly lack of GGeo::deferredCreateGParts ?
    epsilon:opticks blyth$ 



::

     227 void OpticksHub::init()
     228 {   
     229     OK_PROFILE("_OpticksHub::init");
     230     
     231     pLOG(LEVEL,0) << "[" ;   // -1 : one notch more easily seen than LEVEL
     232     
     233     //m_composition->setCtrl(this); 
     234     
     235     add(m_fcfg);
     236     
     237     configure();
     238     // configureGeometryPrep();
     239     configureServer();
     240     configureCompositionSize();
     241 
     242     
     243     if(m_ok->isLegacy())
     244     {   
     245         LOG(fatal) << m_ok->getLegacyDesc();
     246         configureLookupA();
     247     }
     248     
     249     m_aim = new OpticksAim(this) ;
     250     
     251     if( m_ggeo == NULL )
     252     {   
     253         loadGeometry() ;
     254     }
     255     else
     256     {   
     257         adoptGeometry() ;
     258     }
     259     if(m_err) return ;
     260 
     261     
     262     // TODO:migrate these into GGeo for self-containment
     263     configureGeometry() ;      // setting mm geocode
     264     deferredGeometryPrep();    // GGeo::deferredCreateGParts
     265 
     266     
     267     m_gen = new OpticksGen(this) ;
     268     
     269     pLOG(LEVEL,0) << "]" ; 
     270     OK_PROFILE("OpticksHub::init");
     271 }
     272 



::

    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff58ac8b66 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff58c93080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff58a241ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff589ec1ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x00000001001a1b4f libGGeo.dylib`GPropertyLib::getIndex(this=0x0000000101a20080, shortname="Vacuum") at GPropertyLib.cc:399
        frame #5: 0x00000001001dfe95 libGGeo.dylib`GBnd::init(this=0x00007ffeefbfd338, flip_=false) at GBnd.cc:70
        frame #6: 0x00000001001df1b0 libGGeo.dylib`GBnd::GBnd(this=0x00007ffeefbfd338, spec_="Vacuum///Vacuum", flip_=false, mlib_=0x0000000101a20080, slib_=0x0000000101a78f10, dbgbnd_=false) at GBnd.cc:35
        frame #7: 0x00000001001e03e4 libGGeo.dylib`GBnd::GBnd(this=0x00007ffeefbfd338, spec_="Vacuum///Vacuum", flip_=false, mlib_=0x0000000101a20080, slib_=0x0000000101a78f10, dbgbnd_=false) at GBnd.cc:34
        frame #8: 0x00000001001e382a libGGeo.dylib`GBndLib::addBoundary(this=0x0000000101a1fb30, spec="Vacuum///Vacuum", flip=false) at GBndLib.cc:386
        frame #9: 0x00000001001fa2a6 libGGeo.dylib`GParts::registerBoundaries(this=0x0000000108fcc580) at GParts.cc:1257
        frame #10: 0x00000001001f8d42 libGGeo.dylib`GParts::close(this=0x0000000108fcc580) at GParts.cc:1228
        frame #11: 0x0000000100276b40 libGGeo.dylib`GGeo::deferredCreateGParts(this=0x00007ffeefbfe688) at GGeo.cc:1593
        frame #12: 0x000000010027a689 libGGeo.dylib`GGeo::dryrun_convert(this=0x00007ffeefbfe688) at GGeo.cc:2176
        frame #13: 0x00000001000063d5 GGeoConvertTest`main(argc=2, argv=0x00007ffeefbfe9e0) at GGeoConvertTest.cc:16
        frame #14: 0x00007fff58978015 libdyld.dylib`start + 1
    (lldb) f 4
    frame #4: 0x00000001001a1b4f libGGeo.dylib`GPropertyLib::getIndex(this=0x0000000101a20080, shortname="Vacuum") at GPropertyLib.cc:399
       396 	
       397 	unsigned int GPropertyLib::getIndex(const char* shortname)
       398 	{
    -> 399 	    assert( isClosed() && " must close the lib before the indices can be used, as preference sort order may be applied at the close" ); 
       400 	    
       401 	    /*
       402 	    if(!isClosed())
    (lldb) 

