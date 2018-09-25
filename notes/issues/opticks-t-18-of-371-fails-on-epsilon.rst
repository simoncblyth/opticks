opticks-t-18-of-371-fails-on-epsilon
=====================================

::

    FAILS:
      73 /120 Test #73 : NPYTest._RecordsNPYTest                       ***Exception: SegFault         0.01   
      75 /120 Test #75 : NPYTest.PhotonsNPYTest                        ***Exception: SegFault         0.01   
      2  /24  Test #2  : OpticksCoreTest.IndexerTest                   ***Exception: SegFault         0.01   
      6  /24  Test #6  : OpticksCoreTest.OpticksEventSpecTest          ***Exception: SegFault         0.01   
      25 /50  Test #25 : GGeoTest.GItemIndexTest                       ***Exception: SegFault         0.01   
      2  /15  Test #2  : ThrustRapTest.TBufTest                        ***Exception: SegFault         0.79   

      1  /33  Test #1  : CFG4Test.CMaterialLibTest                     ***Exception: Child aborted    0.29   
      2  /33  Test #2  : CFG4Test.CMaterialTest                        ***Exception: Child aborted    0.29   
      3  /33  Test #3  : CFG4Test.CTestDetectorTest                    ***Exception: Child aborted    0.97   
      5  /33  Test #5  : CFG4Test.CGDMLDetectorTest                    ***Exception: Child aborted    0.86   
      6  /33  Test #6  : CFG4Test.CGeometryTest                        ***Exception: Child aborted    0.81   
      7  /33  Test #7  : CFG4Test.CG4Test                              ***Exception: Child aborted    0.90   
      23 /33  Test #23 : CFG4Test.CInterpolationTest                   ***Exception: Child aborted    0.88   
      25 /33  Test #25 : CFG4Test.CGROUPVELTest                        ***Exception: Child aborted    0.28   
      28 /33  Test #28 : CFG4Test.CRandomEngineTest                    ***Exception: Child aborted    0.89   
      31 /33  Test #31 : CFG4Test.CCerenkovGeneratorTest               ***Exception: Child aborted    0.29   
      32 /33  Test #32 : CFG4Test.CGenstepSourceTest                   ***Exception: Child aborted    0.29   
      1  /1   Test #1  : OKG4Test.OKG4Test                             ***Exception: Child aborted    0.90   
    epsilon:build blyth$ 



After fix the BResource issue::

    totals  13  / 372 


    FAILS:
      14 /18  Test #14 : OptiXRapTest.OEventTest                       ***Exception: Child aborted    0.28   
      1  /33  Test #1  : CFG4Test.CMaterialLibTest                     ***Exception: Child aborted    0.32   
      2  /33  Test #2  : CFG4Test.CMaterialTest                        ***Exception: Child aborted    0.29   
      3  /33  Test #3  : CFG4Test.CTestDetectorTest                    ***Exception: Child aborted    0.88   
      5  /33  Test #5  : CFG4Test.CGDMLDetectorTest                    ***Exception: Child aborted    0.83   
      6  /33  Test #6  : CFG4Test.CGeometryTest                        ***Exception: Child aborted    0.81   
      7  /33  Test #7  : CFG4Test.CG4Test                              ***Exception: Child aborted    0.89   
      23 /33  Test #23 : CFG4Test.CInterpolationTest                   ***Exception: Child aborted    0.86   
      25 /33  Test #25 : CFG4Test.CGROUPVELTest                        ***Exception: Child aborted    0.28   
      28 /33  Test #28 : CFG4Test.CRandomEngineTest                    ***Exception: Child aborted    0.88   
      31 /33  Test #31 : CFG4Test.CCerenkovGeneratorTest               ***Exception: Child aborted    0.29   
      32 /33  Test #32 : CFG4Test.CGenstepSourceTest                   ***Exception: Child aborted    0.30   
      1  /1   Test #1  : OKG4Test.OKG4Test                             ***Exception: Child aborted    0.90   
    epsilon:build blyth$ 




OPTICKS_EVENT_BASE issue : added BFileTest to reproduce at lower level
----------------------------------------------------------------------------------

\_RecordsNPYTest::

    (lldb) p ubase
    (std::__1::string) $0 = "$OPTICKS_EVENT_BASE/evt/dayabay/cerenkov/1"
    (lldb) f 4
    frame #4: 0x000000010088de1f libBoostRap.dylib`BFile::FormPath(path="$OPTICKS_EVENT_BASE/evt/dayabay/cerenkov/1", sub=0x0000000000000000, name=0x0000000000000000, extra1=0x0000000000000000, extra2=0x0000000000000000) at BFile.cc:399
       396 	
       397 	   if(path[0] == '$')
       398 	   {
    -> 399 	      xpath.assign(expandvar(path));
       400 	   } 
       401 	   else if(path[0] == '~')
       402 	   {
    (lldb) 



::

    121            else if(evalue.compare("OPTICKS_EVENT_BASE")==0)
    122            {
    123                const char* evtbase = BResource::Get("evtbase") ;
    124                if( evtbase != NULL )
    125                {
    126                    evalue = evtbase ;
    127                }
    128                else
    129                {
    130                    evalue = BResource::Get("tmpuser_dir") ;
    131                    //evalue = usertmpdir("/tmp","opticks",NULL);
    132                }
    133                LOG(verbose) << "expandvar replacing OPTICKS_EVENT_BASE  with " << evalue ;
    134            }



OEventTest : resource changes tripping another assert
--------------------------------------------------------

::

    2018-09-25 10:48:04.347 INFO  [7491837] [OpticksHub::configureGeometryTri@558] OpticksHub::configureGeometryTri restrict_mesh -1 nmm 6
    2018-09-25 10:48:04.347 INFO  [7491837] [*BResource::Get@25]  label srcevtbase ret (null)
    2018-09-25 10:48:04.347 INFO  [7491837] [*BResource::Get@25]  label tmpuser_dir ret /tmp/blyth/opticks
    2018-09-25 10:48:04.347 INFO  [7491837] [*Opticks::getDirectGenstepPath@1889] Opticks::getDirectGenstepPath det dayabay typ machinery tag 1 srctagdir /tmp/blyth/opticks/evt/dayabay/machinery/1
    2018-09-25 10:48:04.347 INFO  [7491837] [*BResource::Get@25]  label srcevtbase ret (null)
    2018-09-25 10:48:04.347 INFO  [7491837] [*BResource::Get@25]  label tmpuser_dir ret /tmp/blyth/opticks
    2018-09-25 10:48:04.347 INFO  [7491837] [*Opticks::getDirectGenstepPath@1889] Opticks::getDirectGenstepPath det dayabay typ machinery tag 1 srctagdir /tmp/blyth/opticks/evt/dayabay/machinery/1
    2018-09-25 10:48:04.347 INFO  [7491837] [OpticksGen::initFromDirectGensteps@151] .
    2018-09-25 10:48:04.347 FATAL [7491837] [OpticksHub::init@189] ]
    2018-09-25 10:48:04.347 INFO  [7491837] [SLog::operator@27] OpticksHub::OpticksHub  DONE
    Assertion failed: (gs0), function main, file /Users/blyth/opticks/optixrap/tests/OEventTest.cc, line 38.
    Abort trap: 6


Why direct ? This should be using the legacy gensteps.

* OpticksGen is basing mode on existance of a direct gensteps path 
* some test is writing it ? deleting it gets OEventTest to pass

  * OEventTest itself writes this file as a result of OpticksEvent::save

* OpticksGen mode based on existance of a file, seems not a good idea

  * should only use direct gensteps in direct mode 

* direct evt paths are within the currently relevant geocache, defaulting to tmp
  for direct event paths is wrong 

* especially as legacy event paths used tmp, so this melanges the flavors 

* ... so just return NULL when OPTICKS_EVENT_BASE has not been set ? As a way to 
  handle legacy running ?

::

    epsilon:boostrap blyth$ opticks-find OPTICKS_EVENT_BASE
    ./boostrap/BFile.cc:           else if(evalue.compare("OPTICKS_EVENT_BASE")==0) 
    ./boostrap/BFile.cc:               LOG(verbose) << "expandvar replacing OPTICKS_EVENT_BASE  with " << evalue ; 
    ./boostrap/BOpticksEvent.cc:const char* BOpticksEvent::DEFAULT_DIR_TEMPLATE_NOTAG = "$OPTICKS_EVENT_BASE/evt/$1/$2" ;  // formerly "$LOCAL_BASE/env/opticks/$1/$2"
    ./boostrap/BOpticksEvent.cc:const char* BOpticksEvent::DEFAULT_DIR_TEMPLATE       = "$OPTICKS_EVENT_BASE/evt/$1/$2/$3" ;  // formerly "$LOCAL_BASE/env/opticks/$1/$2"
    ./boostrap/BOpticksEvent.cc:       LOG(debug) << "BOpticksEvent::directory_template OVERRIDE_EVENT_BASE replacing OPTICKS_EVENT_BASE with " << OVERRIDE_EVENT_BASE ; 
    ./boostrap/BOpticksEvent.cc:       boost::replace_first(deftmpl, "$OPTICKS_EVENT_BASE/evt", OVERRIDE_EVENT_BASE );
    ./boostrap/tests/BFileTest.cc:    ss.push_back("$OPTICKS_EVENT_BASE/evt/dayabay/cerenkov/1") ; 
    ./ana/ncensus.py:    c = Census("$OPTICKS_EVENT_BASE/evt")
    ./ana/nload.py:DEFAULT_BASE = "$OPTICKS_EVENT_BASE/evt"
    ./ana/base.py:        self.setdefault("OPTICKS_EVENT_BASE",      os.path.expandvars("/tmp/$USER/opticks") )
    epsilon:opticks blyth$ 

::

    epsilon:opticks blyth$ opticks-find evtbase
    ./boostrap/BFile.cc:               const char* evtbase = BResource::Get("evtbase") ; 
    ./boostrap/BFile.cc:               if( evtbase != NULL )
    ./boostrap/BFile.cc:                   evalue = evtbase ; 
    ./boostrap/BOpticksEvent.cc:srcevtbase
    ./boostrap/BOpticksEvent.cc:    const char* srcevtbase = BResource::Get("srcevtbase");   
    ./boostrap/BOpticksEvent.cc:    if( srcevtbase == NULL ) srcevtbase = BResource::Get("tmpuser_dir") ;   
    ./boostrap/BOpticksEvent.cc:    assert( srcevtbase ); 
    ./boostrap/BOpticksEvent.cc:    std::string path = BFile::FormPath(srcevtbase, "evt", det, typ, tag ); 

    ./boostrap/BOpticksResource.cc:    m_srcevtbase(NULL),
    ./boostrap/BOpticksResource.cc:    m_evtbase(NULL),
    ./boostrap/BOpticksResource.cc:    //m_srcevtbase = makeIdPathPath("evt", user, "source"); 
    ./boostrap/BOpticksResource.cc:    m_srcevtbase = makeIdPathPath("source"); 
    ./boostrap/BOpticksResource.cc:    m_res->addDir( "srcevtbase", m_srcevtbase ); 
    ./boostrap/BOpticksResource.cc:    m_evtbase = isKeySource() ? strdup(m_srcevtbase) : makeIdPathPath("tmp", user, exename ) ;  
    ./boostrap/BOpticksResource.cc:    m_res->addDir( "evtbase", m_evtbase ); 
    ./boostrap/BOpticksResource.cc://const char* BOpticksResource::getSrcEventBase() const { return m_srcevtbase ; } 
    ./boostrap/BOpticksResource.cc:const char* BOpticksResource::getEventBase() const { return m_evtbase ; } 
    ./boostrap/BOpticksResource.hh:        const char* m_srcevtbase ; 
    ./boostrap/BOpticksResource.hh:        const char* m_evtbase ; 
    epsilon:opticks blyth$ 

      

::

    142 /**
    143 
    144 BOpticksEvent::srctagdir
    145 ----------------------------
    146 
    147 srcevtbase
    148      inside the geocache keydir eg:
    149      /usr/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/792496b5e2cc08bdf5258cc12e63de9f/1/source
    150 
    151 **/
    152 
    153 const char* BOpticksEvent::srctagdir( const char* det, const char* typ, const char* tag) // static
    154 {
    155     const char* srcevtbase = BResource::Get("srcevtbase");
    156     if( srcevtbase == NULL ) srcevtbase = BResource::Get("tmpuser_dir") ;

    ///  IS THIS FALLBACK TO TMP APPROPRIATE  ?
    ///  BETTER TO EXPLICITLY SET THE srcevtbase to tmp for legacy case, analogous to DIRECT 

    157     assert( srcevtbase );
    158 
    159     std::string path = BFile::FormPath(srcevtbase, "evt", det, typ, tag );
    160     //  source/evt/g4live/natural/1/        gs.npy
    161 
    162     return strdup(path.c_str()) ;
    163 }
    164 

::

    epsilon:boostrap blyth$ opticks-find srcevtbase
    ./boostrap/BOpticksEvent.cc:srcevtbase
    ./boostrap/BOpticksEvent.cc:    const char* srcevtbase = BResource::Get("srcevtbase");   
    ./boostrap/BOpticksEvent.cc:    if( srcevtbase == NULL ) srcevtbase = BResource::Get("tmpuser_dir") ;   
    ./boostrap/BOpticksEvent.cc:    assert( srcevtbase ); 
    ./boostrap/BOpticksEvent.cc:    std::string path = BFile::FormPath(srcevtbase, "evt", det, typ, tag ); 
    ./boostrap/BOpticksResource.cc:    m_srcevtbase(NULL),
    ./boostrap/BOpticksResource.cc:    //m_srcevtbase = makeIdPathPath("evt", user, "source"); 
    ./boostrap/BOpticksResource.cc:    m_srcevtbase = makeIdPathPath("source"); 
    ./boostrap/BOpticksResource.cc:    m_res->addDir( "srcevtbase", m_srcevtbase ); 
    ./boostrap/BOpticksResource.cc:    m_evtbase = isKeySource() ? strdup(m_srcevtbase) : makeIdPathPath("tmp", user, exename ) ;  
    ./boostrap/BOpticksResource.cc://const char* BOpticksResource::getSrcEventBase() const { return m_srcevtbase ; } 
    ./boostrap/BOpticksResource.hh:        const char* m_srcevtbase ; 
    epsilon:opticks blyth$ 
    epsilon:opticks blyth$ 





::

    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff655dbb6e libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff657a6080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff655371ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff654ff1ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x000000010507ca3b libBoostRap.dylib`BFile::FormPath(path="/tmp/blyth/opticks/evt/dayabay/machinery/1", sub="gs.npy", name=0x0000000000000000, extra1=0x0000000000000000, extra2=0x0000000000000000) at BFile.cc:442
        frame #5: 0x00000001046f57c5 libOpticksCore.dylib`Opticks::getDirectGenstepPath(this=0x00007ffeefbfe758) const at Opticks.cc:1896
        frame #6: 0x00000001046f5d68 libOpticksCore.dylib`Opticks::existsDirectGenstepPath(this=0x00007ffeefbfe758) const at Opticks.cc:1936
        frame #7: 0x0000000102f40884 libOpticksGeo.dylib`OpticksGen::OpticksGen(this=0x000000010eadbfa0, hub=0x00007ffeefbfe6c0) at OpticksGen.cc:45
        frame #8: 0x0000000102f40a6d libOpticksGeo.dylib`OpticksGen::OpticksGen(this=0x000000010eadbfa0, hub=0x00007ffeefbfe6c0) at OpticksGen.cc:47
        frame #9: 0x0000000102f3a0d8 libOpticksGeo.dylib`OpticksHub::init(this=0x00007ffeefbfe6c0) at OpticksHub.cc:187
        frame #10: 0x0000000102f39e1a libOpticksGeo.dylib`OpticksHub::OpticksHub(this=0x00007ffeefbfe6c0, ok=0x00007ffeefbfe758) at OpticksHub.cc:156
        frame #11: 0x0000000102f3a22d libOpticksGeo.dylib`OpticksHub::OpticksHub(this=0x00007ffeefbfe6c0, ok=0x00007ffeefbfe758) at OpticksHub.cc:155
        frame #12: 0x0000000100009682 OEventTest`main(argc=1, argv=0x00007ffeefbfe938) at OEventTest.cc:34
        frame #13: 0x00007fff6548b015 libdyld.dylib`start + 1
        frame #14: 0x00007fff6548b015 libdyld.dylib`start + 1
    (lldb) 






::

    epsilon:optickscore blyth$ ll /tmp/blyth/opticks/evt/dayabay/machinery/1/gs.npy
    -rw-r--r--  1 blyth  wheel  1040 Sep 25 10:43 /tmp/blyth/opticks/evt/dayabay/machinery/1/gs.npy
    epsilon:optickscore blyth$ date
    Tue Sep 25 11:01:45 CST 2018

    epsilon:optickscore blyth$ xxd /tmp/blyth/opticks/evt/dayabay/machinery/1/gs.npy
    00000000: 934e 554d 5059 0100 4600 7b27 6465 7363  .NUMPY..F.{'desc
    00000010: 7227 3a20 273c 6634 272c 2027 666f 7274  r': '<f4', 'fort
    00000020: 7261 6e5f 6f72 6465 7227 3a20 4661 6c73  ran_order': Fals
    00000030: 652c 2027 7368 6170 6527 3a20 2831 302c  e, 'shape': (10,
    00000040: 2036 2c20 3429 2c20 7d20 2020 2020 200a   6, 4), }      .
    00000050: 0080 0000 0000 0000 0000 0000 0a00 0000  ................
    00000060: 0000 0000 0000 0000 0000 0000 0000 0000  ................
    00000070: 0000 0000 0000 0000 0000 0000 0000 0000  ................



::

     37     NPY<float>* gs0 = hub.getInputGensteps();
     38     assert(gs0);


::

    675 // from OpticksGen : needed by CGenerator
    676 unsigned        OpticksHub::getSourceCode() const {         return m_gen->getSourceCode() ; }
    677 
    678 NPY<float>*     OpticksHub::getInputPhotons() const    {    return m_gen->getInputPhotons() ; }
    679 NPY<float>*     OpticksHub::getInputGensteps() const {      return m_gen->getInputGensteps(); }
    680 NPY<float>*     OpticksHub::getInputPrimaries() const  {    return m_gen->getInputPrimaries() ; }

::

    045     m_direct_gensteps(m_ok->existsDirectGenstepPath() ? m_ok->loadDirectGenstep() : NULL ), 


    088 void OpticksGen::init()
     89 {
     90     if(m_direct_gensteps)
     91     {
     92         initFromDirectGensteps();
     93     }
     94     else if(m_input_primaries)
     95     {
     96         initFromPrimaries();
     97     }
     98     else if(m_emitter)
     99     {
    100         initFromEmitter();
    101     }
    102     else
    103     {
    104         initFromGensteps();
    105     }
    106 }




::

    1881 const char* Opticks::getDirectGenstepPath() const
    1882 {
    1883     const char* det = m_spec->getDet();
    1884     const char* typ = m_spec->getTyp();
    1885     const char* tag = m_spec->getTag();
    1886 
    1887     const char* srctagdir = BOpticksEvent::srctagdir(det, typ, tag );
    1888 
    1889     LOG(info) << "Opticks::getDirectGenstepPath"
    1890               << " det " << det
    1891               << " typ " << typ
    1892               << " tag " << tag
    1893               << " srctagdir " << srctagdir
    1894               ;
    1895 
    1896     std::string path = BFile::FormPath( srctagdir, "gs.npy" );
    1897     return strdup(path.c_str())  ;
    1898 }


::

    1905 std::string Opticks::getGenstepPath() const
    1906 {
    1907     const char* det = m_spec->getDet();
    1908     const char* typ = m_spec->getTyp();
    1909     const char* tag = m_spec->getTag();
    1910 
    1911     std::string path = NLoad::GenstepsPath(det, typ, tag);
    1912 
    1913     LOG(info) << "Opticks::getGenstepPath"
    1914               << " det " << det
    1915               << " typ " << typ
    1916               << " tag " << tag
    1917               << " path " << path
    1918               ;
    1919 
    1920 
    1921     return path ;
    1922 }




After fixing OpticksGen mode depending in existance of direct gensteps in non-direct mode
----------------------------------------------------------------------------------------------------

::

    totals  12  / 372 


    FAILS:
      1  /33  Test #1  : CFG4Test.CMaterialLibTest                     ***Exception: Child aborted    0.29   
      2  /33  Test #2  : CFG4Test.CMaterialTest                        ***Exception: Child aborted    0.30   
      3  /33  Test #3  : CFG4Test.CTestDetectorTest                    ***Exception: Child aborted    0.91   
      5  /33  Test #5  : CFG4Test.CGDMLDetectorTest                    ***Exception: Child aborted    0.84   
      6  /33  Test #6  : CFG4Test.CGeometryTest                        ***Exception: Child aborted    0.84   
      7  /33  Test #7  : CFG4Test.CG4Test                              ***Exception: Child aborted    0.88   
      23 /33  Test #23 : CFG4Test.CInterpolationTest                   ***Exception: Child aborted    0.88   
      25 /33  Test #25 : CFG4Test.CGROUPVELTest                        ***Exception: Child aborted    0.29   
      28 /33  Test #28 : CFG4Test.CRandomEngineTest                    ***Exception: Child aborted    0.90   
      31 /33  Test #31 : CFG4Test.CCerenkovGeneratorTest               ***Exception: Child aborted    0.29   
      32 /33  Test #32 : CFG4Test.CGenstepSourceTest                   ***Exception: Child aborted    0.29   
      1  /1   Test #1  : OKG4Test.OKG4Test                             ***Exception: Child aborted    0.94   
    epsilon:build blyth$ 


Remaing 12 fails are all the same issue, related to CPropLib sensor surface/material conversion::

    2018-09-25 16:25:46.490 INFO  [7859409] [CMaterialLib::convert@141]  g4mat 0x10dcf2a90 name Acrylic Pmin 1.512e-06 Pmax 2.0664e-05 Wmin 60 Wmax 820
    2018-09-25 16:25:46.490 INFO  [7859409] [CMaterialLib::convert@141]  g4mat 0x10dcf3850 name MineralOil Pmin 1.512e-06 Pmax 2.0664e-05 Wmin 60 Wmax 820
    2018-09-25 16:25:46.490 ERROR [7859409] [*CPropLib::makeMaterialPropertiesTable@234]  name Bialkali adding EFFICIENCY GPropertyMap  type skinsurface name /dd/Geometry/PMT/lvHeadonPmtCathodeSensorSurface
    2018-09-25 16:25:46.490 FATAL [7859409] [CPropLib::addProperties@295] missing key for prop i 1 nprop 8 matname /dd/Geometry/PMT/lvHeadonPmtCathodeSensorSurface key absorb lkey (null) ukey (null) keylocal 1
    Assertion failed: (ukey), function addProperties, file /Users/blyth/opticks/cfg4/CPropLib.cc, line 305.
    Process 77256 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff655dbb6e libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill:
    ->  0x7fff655dbb6e <+10>: jae    0x7fff655dbb78            ; <+20>
        0x7fff655dbb70 <+12>: movq   %rax, %rdi
        0x7fff655dbb73 <+15>: jmp    0x7fff655d2b00            ; cerror_nocancel
        0x7fff655dbb78 <+20>: retq   
    Target 0: (CGenstepSourceTest) stopped.
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff655dbb6e libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff657a6080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff655371ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff654ff1ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x000000010013a038 libCFG4.dylib`CPropLib::addProperties(this=0x000000010dceac40, mpt=0x000000010dcf7770, pmap=0x000000010ba166e0, _keys="EFFICIENCY", keylocal=true, constant=false) at CPropLib.cc:305
        frame #5: 0x0000000100138ee4 libCFG4.dylib`CPropLib::makeMaterialPropertiesTable(this=0x000000010dceac40, ggmat=0x000000010a73c9d0) at CPropLib.cc:238
        frame #6: 0x000000010015b7a2 libCFG4.dylib`CMaterialLib::convertMaterial(this=0x000000010dceac40, kmat=0x000000010a73c9d0) at CMaterialLib.cc:228
        frame #7: 0x000000010015a9fa libCFG4.dylib`CMaterialLib::convert(this=0x000000010dceac40) at CMaterialLib.cc:121
        frame #8: 0x000000010000eeee CGenstepSourceTest`main(argc=1, argv=0x00007ffeefbfe938) at CGenstepSourceTest.cc:39
        frame #9: 0x00007fff6548b015 libdyld.dylib`start + 1
        frame #10: 0x00007fff6548b015 libdyld.dylib`start + 1
    (lldb) 






CMaterialLibTest trying to convert all scintillator props ?::

    (lldb) f 11
    frame #11: 0x000000010013a35f libCFG4.dylib`CPropLib::makeMaterialPropertiesTable(this=0x000000010b0008c0, ggmat=0x000000010a534600) at CPropLib.cc:256
       253 	                  << " keys " << scintillator->getKeysString() 
       254 	                   ; 
       255 	        bool keylocal, constant ; 
    -> 256 	        addProperties(mpt, scintillator, "SLOWCOMPONENT,FASTCOMPONENT", keylocal=false, constant=false);
       257 	        addProperties(mpt, scintillator, "SCINTILLATIONYIELD,RESOLUTIONSCALE,YIELDRATIO,FASTTIMECONSTANT,SLOWTIMECONSTANT", keylocal=false, constant=true );
       258 	
       259 	        // NB the above skips prefixed versions of the constants: Alpha, 
    (lldb) f 10
    frame #10: 0x000000010013ae55 libCFG4.dylib`CPropLib::addProperties(this=0x000000010b0008c0, mpt=0x000000010d6cb280, pmap=0x000000010b11f170, _keys="SLOWCOMPONENT,FASTCOMPONENT", keylocal=false, constant=false) at CPropLib.cc:292
       289 	    for(unsigned int i=0 ; i<nprop ; i++)
       290 	    {
       291 	        const char* key =  pmap->getPropertyNameByIndex(i); // refractive_index absorption_length scattering_length reemission_prob
    -> 292 	        const char* lkey = m_mlib->getLocalKey(key) ;      // RINDEX ABSLENGTH RAYLEIGH REEMISSIONPROB
       293 	        const char* ukey = keylocal ? lkey : key ;
       294 	
       295 	        if(!ukey) LOG(fatal) << "CPropLib::addProperties missing key for prop " << i ; 
    (lldb) p key
    (const char *) $0 = 0x000000010b127141 "AlphaYIELDRATIO"
    (lldb) p nprop
    (unsigned int) $1 = 23
    (lldb) 


These 



