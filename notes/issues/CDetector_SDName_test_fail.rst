CDetector_SDName_test_fail
============================


Axel reports::

    Hi Simon,

    I found some time now to have a closer look to the errors:

     3  /30  Test #3  : CFG4Test.CTestDetectorTest                    ***Exception: Child aborted    1.28   
      7  /30  Test #7  : CFG4Test.CG4Test                              ***Exception: Child aborted    1.22   
      21 /30  Test #21 : CFG4Test.CInterpolationTest                   ***Exception: Child aborted    1.08   
      26 /30  Test #26 : CFG4Test.CRandomEngineTest                    ***Exception: Child aborted    1.14   
      1  /1   Test #1  : OKG4Test.OKG4Test                             ***Exception: Child aborted    1.41   

    They all have the same error when running separately:

    OKG4Test: /home/gpu/opticks/cfg4/CDetector.cc:141: void CDetector::hookupSD(): Assertion `strcmp( sdname.c_str(), sdn ) == 0' failed.
    I looked at the code at /home/gpu/opticks/cfg4/CDetector.cc:141:
    assert( strcmp( sdname.c_str(), sdn ) == 0 ) ;  

    I let the two strings write to the console and got:

    2018-08-28 15:28:00.001 ERROR [6809] [CDetector::hookupSD@141]  sdn SD_AssimpGGeo sdname SD0

    At this level I got stuck as I don't really understand yet the entire code.

    Do you also get this error?

    Best regards,

    Axel


::

    123 void CDetector::hookupSD()
    124 {
    125     unsigned nlvsd = m_ggeo->getNumLVSD() ;
    126     const std::string sdname = m_sd ? m_sd->GetName() : "noSD" ;
    127     LOG(error)
    128         << " nlvsd " << nlvsd
    129         << " sd " << m_sd
    130         << " sdname " << sdname
    131         ;
    132 
    133 
    134     if(!m_sd) return ;
    135     for( unsigned i = 0 ; i < nlvsd ; i++)
    136     {
    137         std::pair<std::string,std::string> lvsd = m_ggeo->getLVSD(i) ;
    138         const char* lvn = lvsd.first.c_str();
    139         const char* sdn = lvsd.second.c_str();
    140 
    141         assert( strcmp( sdname.c_str(), sdn ) == 0 ) ;
    142 
    143         //const char* lvn = m_ggeo->getCathodeLV(i); 
    144 
    145         const G4LogicalVolume* lv = m_traverser->getLV(lvn);
    146 
    147         LOG(error)
    148              << "SetSensitiveDetector"
    149              << " lvn " << lvn
    150              << " sdn " << sdn
    151              << " lv " << lv
    152              ;
    153 
    154         if(!lv) LOG(fatal) << " no lv " << lvn ;
    155         assert(lv);
    156 
    157         const_cast<G4LogicalVolume*>(lv)->SetSensitiveDetector(m_sd) ;
    158     }
    159 }



This is curious, as I dont see the fail. However, you are doing better than me : I get nlvsd 0

    2018-08-28 22:26:08.105 ERROR [2989735] [CDetector::hookupSD@127]  nlvsd 0 sd 0x10f12c210 sdname SD0

So I tried updating my geocache `op.sh -G` to see if that gets us to the same error, and I do::

    lldb CG4Test 
    ...
    2018-08-28 22:28:46.349 INFO  [2996153] [CSurfaceLib::convert@136] CSurfaceLib  numBorderSurface 8 numSkinSurface 34
    2018-08-28 22:28:46.349 INFO  [2996153] [CDetector::attachSurfaces@339] [--dbgsurf] CDetector::attachSurfaces DONE 
    2018-08-28 22:28:46.349 ERROR [2996153] [CDetector::hookupSD@127]  nlvsd 2 sd 0x10ea45320 sdname SD0
    Assertion failed: (strcmp( sdname.c_str(), sdn ) == 0), function hookupSD, file /Users/blyth/opticks/cfg4/CDetector.cc, line 141.
    ...
        frame #3: 0x00007fff7ad201ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x00000001001ccd28 libCFG4.dylib`CDetector::hookupSD(this=0x000000010ea46160) at CDetector.cc:141
        frame #5: 0x00000001001d3815 libCFG4.dylib`CGDMLDetector::init(this=0x000000010ea46160) at CGDMLDetector.cc:78
        frame #6: 0x00000001001d34d3 libCFG4.dylib`CGDMLDetector::CGDMLDetector(this=0x000000010ea46160, hub=0x00007ffeefbfe658, query=0x000000010a725560, sd=0x000000010ea45320) at CGDMLDetector.cc:40
        frame #7: 0x00000001001d385d libCFG4.dylib`CGDMLDetector::CGDMLDetector(this=0x000000010ea46160, hub=0x00007ffeefbfe658, query=0x000000010a725560, sd=0x000000010ea45320) at CGDMLDetector.cc:38
        frame #8: 0x0000000100127b9d libCFG4.dylib`CGeometry::init(this=0x000000010ea46110) at CGeometry.cc:68
        frame #9: 0x000000010012789c libCFG4.dylib`CGeometry::CGeometry(this=0x000000010ea46110, hub=0x00007ffeefbfe658, sd=0x000000010ea45320) at CGeometry.cc:51
        frame #10: 0x0000000100127c35 libCFG4.dylib`CGeometry::CGeometry(this=0x000000010ea46110, hub=0x00007ffeefbfe658, sd=0x000000010ea45320) at CGeometry.cc:50
        frame #11: 0x00000001001f160c libCFG4.dylib`CG4::CG4(this=0x000000010e5a1160, hub=0x00007ffeefbfe658) at CG4.cc:120
        frame #12: 0x00000001001f1f8d libCFG4.dylib`CG4::CG4(this=0x000000010e5a1160, hub=0x00007ffeefbfe658) at CG4.cc:140
        frame #13: 0x000000010000f113 CG4Test`main(argc=1, argv=0x00007ffeefbfea30) at CG4Test.cc:31
        frame #14: 0x00007fff7acac015 libdyld.dylib`start + 1
    (lldb) f 4
    frame #4: 0x00000001001ccd28 libCFG4.dylib`CDetector::hookupSD(this=0x000000010ea46160) at CDetector.cc:141
       138 	        const char* lvn = lvsd.first.c_str(); 
       139 	        const char* sdn = lvsd.second.c_str(); 
       140 	
    -> 141 	        assert( strcmp( sdname.c_str(), sdn ) == 0 ) ;  
       142 	
       143 	        //const char* lvn = m_ggeo->getCathodeLV(i); 
       144 	
    (lldb) p lvsd
    (std::__1::pair<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > >) $0 = (first = "__dd__Geometry__PMT__lvHeadonPmtCathode0xc2c8d98", second = "SD_AssimpGGeo")
    (lldb) p 
         



Removing the assert moves along to another error::

    2018-08-28 22:39:00.301 INFO  [3004244] [CDetector::attachSurfaces@339] [--dbgsurf] CDetector::attachSurfaces DONE 
    2018-08-28 22:39:00.301 ERROR [3004244] [CDetector::hookupSD@127]  nlvsd 2 sd 0x7fe113e01df0 sdname SD0
    2018-08-28 22:39:00.301 ERROR [3004244] [CDetector::hookupSD@147] SetSensitiveDetector lvn __dd__Geometry__PMT__lvHeadonPmtCathode0xc2c8d98 sdn SD_AssimpGGeo lv 0x0
    2018-08-28 22:39:00.301 FATAL [3004244] [CDetector::hookupSD@154]  no lv __dd__Geometry__PMT__lvHeadonPmtCathode0xc2c8d98
    Assertion failed: (lv), function hookupSD, file /Users/blyth/opticks/cfg4/CDetector.cc, line 155.
    Abort trap: 6
    epsilon:cfg4 blyth$ 


The route cause of the problem, is that I have recently been working on a simple test geometry over in examples/Geant4/CerenkovMinimal
and it seems the changes I made to CDetector are not working with the full geometry.  


