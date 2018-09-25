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



* CMaterialLib::convert issue for Bialkali  "detect" property ... this is due to the dirty way are moving 
  a surface property into a material 



Simplifying the transfer of efficieny gets down to 3 fails
---------------------------------------------------------------------

::

    ...
    CTestLog :                 g4ok :      0/     1 : 2018-09-25 18:45:00.094744 : /usr/local/opticks/build/g4ok/ctest.log 
    totals  3   / 372 


    FAILS:
      7  /33  Test #7  : CFG4Test.CG4Test                              ***Exception: Child aborted    9.70   
      32 /33  Test #32 : CFG4Test.CGenstepSourceTest                   ***Exception: Child aborted    0.30   
      1  /1   Test #1  : OKG4Test.OKG4Test                             ***Exception: Child aborted    13.10  




CG4Test + OKG4Test : shortnorm issue
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    2018-09-25 17:33:26.238 INFO  [7903324] [CRunAction::BeginOfRunAction@19] CRunAction::BeginOfRunAction count 1
    2018-09-25 17:33:26.251 INFO  [7903324] [CSensitiveDetector::Initialize@56]  HCE 0x1111bb4f0 HCE.Capacity 2 SensitiveDetectorName SD0 collectionName[0] OpHitCollectionA collectionName[1] OpHitCollectionB
    libc++abi.dylib: terminating with uncaught exception of type boost::numeric::positive_overflow: bad numeric conversion: positive overflow
    Process 83048 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff655dbb6e libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill:
    ->  0x7fff655dbb6e <+10>: jae    0x7fff655dbb78            ; <+20>
        0x7fff655dbb70 <+12>: movq   %rax, %rdi
        0x7fff655dbb73 <+15>: jmp    0x7fff655d2b00            ; cerror_nocancel
        0x7fff655dbb78 <+20>: retq   
    Target 0: (CG4Test) stopped.
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff655dbb6e libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff657a6080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff655371ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff6343bf8f libc++abi.dylib`abort_message + 245
        frame #4: 0x00007fff6343c113 libc++abi.dylib`default_terminate_handler() + 241
        frame #5: 0x00007fff64873eab libobjc.A.dylib`_objc_terminate() + 105
        frame #6: 0x00007fff634577c9 libc++abi.dylib`std::__terminate(void (*)()) + 8
        frame #7: 0x00007fff6345726f libc++abi.dylib`__cxa_throw + 121
        frame #8: 0x000000010803592f libBoostRap.dylib`boost::numeric::def_overflow_handler::operator(this=0x00007ffeefbfa7c8, r=cPosOverflow)(boost::numeric::range_check_result) at converter_policies.hpp:166
        frame #9: 0x0000000108035d32 libBoostRap.dylib`boost::numeric::convdetail::generic_range_checker<boost::numeric::conversion_traits<short, float>, boost::numeric::convdetail::LT_HalfPrevLoT<boost::numeric::conversion_traits<short, float> >, boost::numeric::convdetail::GT_HalfSuccHiT<boost::numeric::conversion_traits<short, float> >, boost::numeric::def_overflow_handler>::validate_range(s=70203.2421) at converter.hpp:294
        frame #10: 0x00000001080356a7 libBoostRap.dylib`boost::numeric::convdetail::rounding_converter<boost::numeric::conversion_traits<short, float>, boost::numeric::convdetail::generic_range_checker<boost::numeric::conversion_traits<short, float>, boost::numeric::convdetail::LT_HalfPrevLoT<boost::numeric::conversion_traits<short, float> >, boost::numeric::convdetail::GT_HalfSuccHiT<boost::numeric::conversion_traits<short, float> >, boost::numeric::def_overflow_handler>, boost::numeric::raw_converter<boost::numeric::conversion_traits<short, float> >, boost::numeric::RoundEven<float> >::convert(s=70203.2421) at converter.hpp:487
        frame #11: 0x0000000108035489 libBoostRap.dylib`short BConverter::round_to_even<short, float>(x=0x00007ffeefbfa820) at BConverter.cc:12
        frame #12: 0x0000000108035460 libBoostRap.dylib`BConverter::shortnorm(v=0, center=-16520, extent=7710.625) at BConverter.cc:18
        frame #13: 0x00000001001cd7cd libCFG4.dylib`CWriter::writeStepPoint_(this=0x00000001129ceb50, point=0x00000001118c6b50, photon=0x00000001129cf480) at CWriter.cc:197
        frame #14: 0x00000001001cd34a libCFG4.dylib`CWriter::writeStepPoint(this=0x00000001129ceb50, point=0x00000001118c6b50, flag=4096, material=13) at CWriter.cc:133
        frame #15: 0x00000001001bb7ee libCFG4.dylib`CRecorder::RecordStepPoint(this=0x00000001129cf440, point=0x00000001118c6b50, flag=4096, material=13, boundary_status=Undefined, (null)="PRE") at CRecorder.cc:468
        frame #16: 0x00000001001baae2 libCFG4.dylib`CRecorder::postTrackWriteSteps(this=0x00000001129cf440) at CRecorder.cc:398
        frame #17: 0x00000001001b9f4e libCFG4.dylib`CRecorder::postTrack(this=0x00000001129cf440) at CRecorder.cc:133
        frame #18: 0x00000001001f5611 libCFG4.dylib`CG4::postTrack(this=0x000000010a7e6060) at CG4.cc:255
        frame #19: 0x00000001001efb37 libCFG4.dylib`CTrackingAction::PostUserTrackingAction(this=0x00000001129cec80, track=0x00000001118c5560) at CTrackingAction.cc:91
        frame #20: 0x00000001020b7937 libG4tracking.dylib`G4TrackingManager::ProcessOneTrack(this=0x000000010a5dedc0, apValueG4Track=0x00000001118c5560) at G4TrackingManager.cc:140
        frame #21: 0x0000000101f7e71a libG4event.dylib`G4EventManager::DoProcessing(this=0x000000010a5ded30, anEvent=0x0000000110d4a2c0) at G4EventManager.cc:185
        frame #22: 0x0000000101f7fc2f libG4event.dylib`G4EventManager::ProcessOneEvent(this=0x000000010a5ded30, anEvent=0x0000000110d4a2c0) at G4EventManager.cc:338
        frame #23: 0x0000000101e8b9f5 libG4run.dylib`G4RunManager::ProcessOneEvent(this=0x000000010a7e6a70, i_event=0) at G4RunManager.cc:399
        frame #24: 0x0000000101e8b825 libG4run.dylib`G4RunManager::DoEventLoop(this=0x000000010a7e6a70, n_event=1, macroFile=0x0000000000000000, n_select=-1) at G4RunManager.cc:367
        frame #25: 0x0000000101e89ce1 libG4run.dylib`G4RunManager::BeamOn(this=0x000000010a7e6a70, n_event=1, macroFile=0x0000000000000000, n_select=-1) at G4RunManager.cc:273
        frame #26: 0x00000001001f6396 libCFG4.dylib`CG4::propagate(this=0x000000010a7e6060) at CG4.cc:331
        frame #27: 0x000000010000f615 CG4Test`main(argc=1, argv=0x00007ffeefbfe950) at CG4Test.cc:52
        frame #28: 0x00007fff6548b015 libdyld.dylib`start + 1
    (lldb) 


::

    (lldb) f 13
    frame #13: 0x0000000106a3e7cd libCFG4.dylib`CWriter::writeStepPoint_(this=0x000000011727af30, point=0x00000001460d0140, photon=0x000000011727ae00) at CWriter.cc:197
       194 	    const glm::vec4& td = m_evt->getTimeDomain() ; 
       195 	    const glm::vec4& wd = m_evt->getWavelengthDomain() ; 
       196 	
    -> 197 	    short posx = BConverter::shortnorm(pos.x()/mm, sd.x, sd.w ); 
       198 	    short posy = BConverter::shortnorm(pos.y()/mm, sd.y, sd.w ); 
       199 	    short posz = BConverter::shortnorm(pos.z()/mm, sd.z, sd.w ); 
       200 	    short time_ = BConverter::shortnorm(time/ns,   td.x, td.y );
    (lldb) p pos
    (G4ThreeVector) $0 = (dx = 0, dy = 0, dz = 0)
    (lldb) p sd
    (glm::vec4) $1 = {
       = (x = -16520, r = -16520, s = -16520)
       = (y = -802110, g = -802110, t = -802110)
       = (z = -7125, b = -7125, p = -7125)
       = (w = 7710.625, a = 7710.625, q = 7710.625)
    }
    (lldb) p pos.x()
    (double) $2 = 0
    (lldb) 





CGenstepSourceTest : domain mismatch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* probably are getting the wrong material

::

    CGenstepSourceTest -n
    CGenstepSourceTest -c
          # suspect plucking wrong material : causing domain mismatch ?

    CGenstepSourceTest -s  
          # expected assert as SCINTILLATION not implemented


* could be old genstep issue ? these gensteps have worked for a long time cannot ascribe to this
* but could be the old checknut of genstep lookup/translation : ie perhaps the new approach has skipped 
  some needed translation 



::

    2018-09-25 17:34:59.128 INFO  [7904431] [CMaterialLib::convert@153] CMaterialLib::convert : converted 38 ggeo materials to G4 materials 
    2018-09-25 17:34:59.128 ERROR [7904431] [GBndLib::getMaterialIndexFromLine@715]  line 12 ibnd 3 numBnd 127 imatsur 0
    2018-09-25 17:34:59.128 INFO  [7904431] [*CCerenkovGenerator::GeneratePhotonsFromGenstep@133]  genstep_idx 0 num_gs 7245 materialLine 12 materialIndex 14      post-16536.295 -802084.812 -7066.000   0.844 

    2018-09-25 17:34:59.128 INFO  [7904431] [*CCerenkovGenerator::GeneratePhotonsFromGenstep@166]  Pmin 1.55e-06 Pmax 6.2e-06 wavelength_min(nm) 199.974 wavelength_max(nm) 799.898 preVelocity 299.791 postVelocity 7.00649e-45
    2018-09-25 17:34:59.128 ERROR [7904431] [*CCerenkovGenerator::GetRINDEX@72]  aMaterial 0x10dc51340 materialIndex 14 num_material 38 Rindex 0x10dc531c0 Rindex2 0x10dc531c0
    2018-09-25 17:34:59.128 FATAL [7904431] [*CCerenkovGenerator::GeneratePhotonsFromGenstep@218]  Pmax 6.2e-06 Pmax2 2.0664e-05 dif 1.4464e-05 epsilon 1e-06 Pmax(nm) 199.974 Pmax2(nm) 60
    Assertion failed: (Pmax_match && "material mismatches genstep source material"), function GeneratePhotonsFromGenstep, file /Users/blyth/opticks/cfg4/CCerenkovGenerator.cc, line 228.
    Process 83053 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff655dbb6e libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill:
    ->  0x7fff655dbb6e <+10>: jae    0x7fff655dbb78            ; <+20>
        0x7fff655dbb70 <+12>: movq   %rax, %rdi
        0x7fff655dbb73 <+15>: jmp    0x7fff655d2b00            ; cerror_nocancel
        0x7fff655dbb78 <+20>: retq   
    Target 0: (CGenstepSourceTest) stopped.
    (lldb) 


::

    In [1]: a = np.load("/usr/local/opticks/opticksdata/gensteps/dayabay/cerenkov/./1.npy")

    In [3]: a.shape
    Out[3]: (7836, 6, 4)

    In [4]: a[:,0]
    Out[4]: 
    array([[nan,  0.,  0.,  0.],
           [nan,  0.,  0.,  0.],
           [nan,  0.,  0.,  0.],
           ...,
           [nan,  0.,  0.,  0.],
           [nan,  0.,  0.,  0.],
           [nan,  0.,  0.,  0.]], dtype=float32)

    In [5]: a[:,0].view(np.int32)
    Out[5]: 
    array([[   -1,     1,    12,    80],
           [   -2,     1,    12,   108],
           [   -3,     1,    12,    77],
           ...,
           [-7834,     1,     8,    91],
           [-7835,     1,     8,    83],
           [-7836,     1,     8,    48]], dtype=int32)

    In [8]: np.unique( a[:,0,1].view(np.int32) )
    Out[8]: 
    array([   1,    7,    8,   10,   13,   14,  127,  141,  142,  169,  177,  183,  185,  189,  194,  196,  198,  201,  204,  221,  225,  226,  229,  230,  231,  233,  234,  236,  238,  240,  242,  246,
            247,  249,  254,  260,  261,  267,  268,  272,  285,  288,  301,  302,  303,  327,  329,  331,  332,  343,  344,  386,  388,  391,  393,  394,  402,  406,  415,  417,  418,  424,  425,  427,
            431,  432,  438,  440,  441,  443,  444,  445,  446,  453,  454,  455,  456,  459,  461,  462,  465,  466,  471,  475,  479,  482,  483,  490,  493,  495,  499,  505,  507,  513,  515,  518,
            520,  522,  526,  527,  529,  530,  533,  534,  535,  539,  542,  545,  546,  547,  549,  551,  554,  561,  563,  567,  568,  572,  575,  580,  584,  586,  589,  592,  593,  595,  598,  599,
            602,  605,  610,  611,  615,  616,  619,  623,  625,  626,  632,  633,  634,  636,  638,  644,  652,  660,  665,  667,  670,  671,  674,  677,  683,  690,  691,  693,  697,  700,  702,  703,
            708,  713,  715,  716,  720,  722,  727,  730,  737,  740,  744,  748,  753,  758,  763,  765,  766,  767,  768,  771,  776,  777,  780,  781,  784,  785,  791,  793,  802,  812,  814,  817,
            819,  822,  825,  830,  834,  837,  838,  840,  842,  857,  873,  877,  888,  891,  892,  893,  895,  896,  900,  901,  903,  904,  949,  955,  960,  963,  973,  983,  987,  994,  996, 1008,
           1010, 1013, 1014, 1015, 1016, 1018, 1031, 1034, 1036, 1049, 1051, 1054, 1056, 1062, 1069, 1070, 1073, 1087, 1088, 1095, 1107, 1114, 1117, 1120, 1125, 1131, 1139, 1142, 1147, 1148, 1153, 1158,
           1160, 1206, 1210, 1213, 1224, 1243, 1244, 1245, 1247, 1259, 1261, 1262, 1263, 1264, 1272, 1279, 1287, 1292, 1293, 1296, 1298, 1300, 1303, 1304, 1307, 1309, 1314, 1332, 1336, 1339, 1344, 1348,
           1352, 1354, 1366], dtype=int32)

    In [9]: np.unique( a[:,0,2].view(np.int32) )
    Out[9]: array([ 1,  8, 10, 12, 13, 14, 19], dtype=int32)



::

     07 struct CerenkovStep
      8 {
      9     int Id    ;
     10     int ParentId ;
     11     int MaterialIndex  ;
     12     int NumPhotons ;
     13 



