opticks-t-fails-aug-2021-13-of-493
======================================


::

    SLOW: tests taking longer that 15 seconds
      31 /31  Test #31 : ExtG4Test.X4SurfaceTest                       Passed                         45.15        REDUCED TEST SIZE
      8  /46  Test #8  : CFG4Test.CG4Test                              Subprocess aborted***Exception:  53.21  
      1  /1   Test #1  : OKG4Test.OKG4Test                             Subprocess aborted***Exception:  67.93  


    FAILS:  13  / 493   :  Wed Aug 25 18:39:55 2021   
      25 /35  Test #25 : OptiXRapTest.interpolationTest                ***Failed                      4.97     FINDING PYTHON WITH NUMPY 

      18 /31  Test #18 : ExtG4Test.X4CSGTest                           ***Exception: SegFault         0.13     FIXED WITH local_tempStr
      20 /31  Test #20 : ExtG4Test.X4GDMLParserTest                    ***Exception: SegFault         0.14   
      21 /31  Test #21 : ExtG4Test.X4GDMLBalanceTest                   ***Exception: SegFault         0.15   


      8  /46  Test #8  : CFG4Test.CG4Test                              Subprocess aborted***Exception:  53.21       BAD FLAG 

        2021-08-25 19:40:24.060 INFO  [90759] [CTorchSource::GeneratePrimaryVertex@293]  event_gencode 6 : BAD_FLAG
        CG4Test: /home/blyth/opticks/cfg4/CCtx.cc:104: unsigned int CCtx::step_limit() const: Assertion `_ok_event_init' failed.


      32 /46  Test #32 : CFG4Test.CTreeJUNOTest                        ***Exception: SegFault         0.22   

      1  /46  Test #1  : CFG4Test.CMaterialLibTest                     Subprocess aborted***Exception:   2.40   
      2  /46  Test #2  : CFG4Test.CMaterialTest                        Subprocess aborted***Exception:   2.38   
      30 /46  Test #30 : CFG4Test.CGROUPVELTest                        Subprocess aborted***Exception:   2.44   
      38 /46  Test #38 : CFG4Test.CCerenkovGeneratorTest               Subprocess aborted***Exception:   2.38   
      39 /46  Test #39 : CFG4Test.CGenstepSourceTest                   Subprocess aborted***Exception:   2.35   



      1  /1   Test #1  : OKG4Test.OKG4Test                             Subprocess aborted***Exception:  67.93  

      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      0.87   
    O[blyth@localhost opticks]$ 




Quick Look at causes
----------------------



1. interpolationTest : python without numpy issue, SSys::RunPythonScript needs envvar to steer to correct python


CFG4 : CG4Test 
------------------


::

    39/46 Test #39: CFG4Test.CGenstepSourceTest ...............Subprocess aborted***Exception:   2.32 sec
    2021-08-25 19:40:43.807 INFO  [93237] [OpticksHub::loadGeometry@283] [ /home/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/b8bc31e2cdf88b66e3dfa9afd5ac1f2b/1
    2021-08-25 19:40:45.212 INFO  [93237] [OpticksHub::loadGeometry@315] ]
    2021-08-25 19:40:45.212 INFO  [93237] [Opticks::makeSimpleTorchStep@4218] [ts.setFrameTransform
    CGenstepSourceTest: /home/blyth/opticks/cfg4/CPropLib.cc:354: void CPropLib::addScintillatorMaterialProperties(G4MaterialPropertiesTable*, const char*): Assertion `scintillator && "non-zero reemission prob materials should has an associated raw scintillator"' failed.

    O[blyth@localhost opticks]$ gdb CMaterialTest 
    (gdb) r
    Starting program: /data/blyth/junotop/ExternalLibs/opticks/head/lib/CMaterialTest 
    [Thread debugging using libthread_db enabled]
    Using host libthread_db library "/lib64/libthread_db.so.1".
    2021-08-25 19:45:43.569 INFO  [101555] [main@74] /data/blyth/junotop/ExternalLibs/opticks/head/lib/CMaterialTest
    2021-08-25 19:45:43.579 INFO  [101555] [OpticksHub::loadGeometry@283] [ /home/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/b8bc31e2cdf88b66e3dfa9afd5ac1f2b/1
    2021-08-25 19:45:45.002 INFO  [101555] [OpticksHub::loadGeometry@315] ]
    2021-08-25 19:45:45.003 INFO  [101555] [Opticks::makeSimpleTorchStep@4218] [ts.setFrameTransform
    2021-08-25 19:45:45.003 INFO  [101555] [main@82] /data/blyth/junotop/ExternalLibs/opticks/head/lib/CMaterialTest convert 
    CMaterialTest: /home/blyth/opticks/cfg4/CPropLib.cc:354: void CPropLib::addScintillatorMaterialProperties(G4MaterialPropertiesTable*, const char*): Assertion `scintillator && "non-zero reemission prob materials should has an associated raw scintillator"' failed.

    (gdb) bt
    #3  0x00007fffe8788252 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007ffff7ad0e56 in CPropLib::addScintillatorMaterialProperties (this=0xa8facc0, mpt=0xa925420, name=0x712bd0 "LS") at /home/blyth/opticks/cfg4/CPropLib.cc:354
    #5  0x00007ffff7ad09bd in CPropLib::makeMaterialPropertiesTable (this=0xa8facc0, ggmat=0x712ad0) at /home/blyth/opticks/cfg4/CPropLib.cc:276
    #6  0x00007ffff7ae2563 in CMaterialLib::convertMaterial (this=0xa8facc0, kmat=0x712ad0) at /home/blyth/opticks/cfg4/CMaterialLib.cc:261
    #7  0x00007ffff7ae18bb in CMaterialLib::convert (this=0xa8facc0) at /home/blyth/opticks/cfg4/CMaterialLib.cc:154
    #8  0x0000000000403eaf in main (argc=1, argv=0x7fffffffa188) at /home/blyth/opticks/cfg4/tests/CMaterialTest.cc:84
    (gdb) 


::

    351 void CPropLib::addScintillatorMaterialProperties( G4MaterialPropertiesTable* mpt, const char* name )
    352 {
    353     GPropertyMap<double>* scintillator = m_sclib->getRaw(name);
    354     assert(scintillator && "non-zero reemission prob materials should has an associated raw scintillator");
    355     LOG(LEVEL)
    356         << " found corresponding scintillator from sclib "
    357         << " name " << name
    358         << " keys " << scintillator->getKeysString()
    359         ;
    360 
    361     bool keylocal = false ;
    362     bool constant = false ;
    363     addProperties(mpt, scintillator, "SLOWCOMPONENT,FASTCOMPONENT", keylocal, constant);
    364     addProperties(mpt, scintillator, "SCINTILLATIONYIELD,RESOLUTIONSCALE,YIELDRATIO,FASTTIMECONSTANT,SLOWTIMECONSTANT", keylocal, constant ); // this used constant=true formerly
    365 
    366     // NB the above skips prefixed versions of the constants: Alpha, 
    367     //addProperties(mpt, scintillator, "ALL",          keylocal=false, constant=true );
    368 }





X4 GDML tempStr fails : fixed by decoupling from Geant4 so dont have to vary by Geant4 version
-----------------------------------------------------------------------------------------------------


::

    .     Start 18: ExtG4Test.X4CSGTest
    18/31 Test #18: ExtG4Test.X4CSGTest .....................................***Exception: SegFault  0.13 sec
          Start 20: ExtG4Test.X4GDMLParserTest
    20/31 Test #20: ExtG4Test.X4GDMLParserTest ..............................***Exception: SegFault  0.14 sec
    2021-08-25 18:36:11.175 FATAL [436528] [Opticks::envkey@345]  --allownokey option prevents key checking : this is for debugging of geocache creation 
    2021-08-25 18:36:11.179 FATAL [436528] [OpticksResource::init@122]  CAUTION : are allowing no key 

          Start 21: ExtG4Test.X4GDMLBalanceTest
    21/31 Test #21: ExtG4Test.X4GDMLBalanceTest .............................***Exception: SegFault  0.15 sec



::

    (gdb) f 12
    #12 0x00000000004035cd in main (argc=1, argv=0x7fffffffa428) at /home/blyth/opticks/extg4/tests/X4CSGTest.cc:59
    59	    X4CSG::GenerateTest( solid, &ok, prefix, lvidx ) ;
    (gdb) f 11
    #11 0x00007ffff7b49d86 in X4CSG::GenerateTest (solid=0x6bc010, ok=0x7fffffffa0f0, prefix=0x40617b "$TMP/extg4/X4CSGTest", lvidx=1) at /home/blyth/opticks/extg4/X4CSG.cc:78
    78	    X4CSG xcsg(solid, ok);
    (gdb) f 10
    #10 0x00007ffff7b4a202 in X4CSG::X4CSG (this=0x7fffffff9cd0, solid_=0x6bc010, ok_=0x7fffffffa0f0) at /home/blyth/opticks/extg4/X4CSG.cc:131
    131	    index(-1)
    (gdb) f 9
    #9  0x00007ffff7b68ddb in X4GDMLParser::ToString (solid=0x6bc010, refs=false) at /home/blyth/opticks/extg4/X4GDMLParser.cc:57
    57	    X4GDMLParser parser(refs) ; 
    (gdb) f 8
    #8  0x00007ffff7b68e5c in X4GDMLParser::X4GDMLParser (this=0x7fffffff9c50, refs=false) at /home/blyth/opticks/extg4/X4GDMLParser.cc:69
    69	    writer = new X4GDMLWriteStructure(refs) ; 
    (gdb) f 7
    #7  0x00007ffff7b69942 in X4GDMLWriteStructure::X4GDMLWriteStructure (this=0x712ac0, refs=false) at /home/blyth/opticks/extg4/X4GDMLWriteStructure.cc:35
    35	    init(refs); 
    (gdb) f 6
    #6  0x00007ffff7b69a5f in X4GDMLWriteStructure::init (this=0x712ac0, refs=false) at /home/blyth/opticks/extg4/X4GDMLWriteStructure.cc:63
    63	   xercesc::XMLString::transcode("LS", tempStr, 9999);
    (gdb) p tempStr
    $1 = (XMLCh *) 0x0
    (gdb) 



1042::

    epsilon:gdml blyth$ pwd
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/persistency/gdml
    epsilon:gdml blyth$ 

    epsilon:gdml blyth$ find . -type f  -exec grep -H tempStr {} \;
    ./include/G4GDMLWrite.hh:    XMLCh tempStr[10000];
    ./src/G4GDMLWrite.cc:   xercesc::XMLString::transcode(name,tempStr,9999);
    ./src/G4GDMLWrite.cc:   xercesc::DOMAttr* att = doc->createAttribute(tempStr);
    ./src/G4GDMLWrite.cc:   xercesc::XMLString::transcode(value,tempStr,9999);
    ./src/G4GDMLWrite.cc:   att->setValue(tempStr);
    ./src/G4GDMLWrite.cc:   xercesc::XMLString::transcode(name,tempStr,9999);
    ./src/G4GDMLWrite.cc:   xercesc::DOMAttr* att = doc->createAttribute(tempStr);
    ./src/G4GDMLWrite.cc:   xercesc::XMLString::transcode(str,tempStr,9999);
    ./src/G4GDMLWrite.cc:   att->setValue(tempStr);
    ./src/G4GDMLWrite.cc:   xercesc::XMLString::transcode(name,tempStr,9999);
    ./src/G4GDMLWrite.cc:   return doc->createElement(tempStr);
    ./src/G4GDMLWrite.cc:   xercesc::XMLString::transcode("LS", tempStr, 9999);
    ./src/G4GDMLWrite.cc:     xercesc::DOMImplementationRegistry::getDOMImplementation(tempStr);
    ./src/G4GDMLWrite.cc:   xercesc::XMLString::transcode("Range", tempStr, 9999);
    ./src/G4GDMLWrite.cc:     xercesc::DOMImplementationRegistry::getDOMImplementation(tempStr);
    ./src/G4GDMLWrite.cc:   xercesc::XMLString::transcode("gdml", tempStr, 9999);
    ./src/G4GDMLWrite.cc:   doc = impl->createDocument(0,tempStr,0);
    epsilon:gdml blyth$ 




    128 
    129   protected:
    130 
    131     G4String SchemaLocation;
    132     static G4bool addPointerToName;
    133     xercesc::DOMDocument* doc;
    134     xercesc::DOMElement* extElement;
    135     xercesc::DOMElement* userinfoElement;
    136     XMLCh tempStr[10000];
    137     G4GDMLAuxListType auxList;
    138 };
    139 




1070 still the same::

    epsilon:gdml blyth$ find . -type f -exec grep -H tempStr {} \;
    ./include/G4GDMLWrite.hh:    XMLCh tempStr[10000];
    ./src/G4GDMLWrite.cc:  xercesc::XMLString::transcode(name, tempStr, 9999);
    ./src/G4GDMLWrite.cc:  xercesc::DOMAttr* att = doc->createAttribute(tempStr);
    ./src/G4GDMLWrite.cc:  xercesc::XMLString::transcode(value, tempStr, 9999);
    ./src/G4GDMLWrite.cc:  att->setValue(tempStr);
    ./src/G4GDMLWrite.cc:  xercesc::XMLString::transcode(name, tempStr, 9999);
    ./src/G4GDMLWrite.cc:  xercesc::DOMAttr* att = doc->createAttribute(tempStr);
    ./src/G4GDMLWrite.cc:  xercesc::XMLString::transcode(str, tempStr, 9999);
    ./src/G4GDMLWrite.cc:  att->setValue(tempStr);
    ./src/G4GDMLWrite.cc:  xercesc::XMLString::transcode(name, tempStr, 9999);
    ./src/G4GDMLWrite.cc:  return doc->createElement(tempStr);
    ./src/G4GDMLWrite.cc:  xercesc::XMLString::transcode("LS", tempStr, 9999);
    ./src/G4GDMLWrite.cc:  xercesc::DOMImplementationRegistry::getDOMImplementation(tempStr);
    ./src/G4GDMLWrite.cc:  xercesc::XMLString::transcode("Range", tempStr, 9999);
    ./src/G4GDMLWrite.cc:    xercesc::DOMImplementationRegistry::getDOMImplementation(tempStr);
    ./src/G4GDMLWrite.cc:  xercesc::XMLString::transcode("gdml", tempStr, 9999);
    ./src/G4GDMLWrite.cc:  doc                       = impl->createDocument(0, tempStr, 0);
    epsilon:gdml blyth$ pwd
    /usr/local/opticks_externals/g4_1070.build/geant4.10.07/source/persistency/gdml

The tempStr disappears at some point after 1070.

Old way with fixed size tempStr::

    137 xercesc::DOMAttr* G4GDMLWrite::NewAttribute(const G4String& name,
    138                                             const G4String& value)
    139 {
    140    xercesc::XMLString::transcode(name,tempStr,9999);
    141    xercesc::DOMAttr* att = doc->createAttribute(tempStr);
    142    xercesc::XMLString::transcode(value,tempStr,9999);
    143    att->setValue(tempStr);
    144    return att;
    145 }


New way::

    https://github.com/Geant4/geant4/blob/master/source/persistency/gdml/src/G4GDMLWrite.cc

    xercesc::DOMAttr* G4GDMLWrite::NewAttribute(const G4String& name,
                                                const G4String& value)
    {
      XMLCh* tempStr = NULL;
      tempStr = xercesc::XMLString::transcode(name);
      xercesc::DOMAttr* att = doc->createAttribute(tempStr);
      xercesc::XMLString::release(&tempStr);

      tempStr = xercesc::XMLString::transcode(value);
      att->setValue(tempStr);
      xercesc::XMLString::release(&tempStr);

      return att;
    }



* https://github.com/Geant4/geant4/blob/master/source/persistency/gdml/include/G4GDMLWrite.hh



::

    epsilon:opticks blyth$ git add . 
    epsilon:opticks blyth$ git commit -m "try to avoid needing to change X4GDMLWriteStructure with Geant4 version by using XMLCh local_tempStr[10000] " 
    [master 29a47cb7d] try to avoid needing to change X4GDMLWriteStructure with Geant4 version by using XMLCh local_tempStr[10000]
     3 files changed, 207 insertions(+), 7 deletions(-)
     create mode 100644 notes/issues/opticks-t-fails-aug-2021-13-of-493.rst
    epsilon:opticks blyth$ git push 
    Counting objects: 8, done.
    Delta compression using up to 8 threads.
    Compressing objects: 100% (8/8), done.
    Writing objects: 100% (8/8), 3.00 KiB | 3.00 MiB/s, done.
    Total 8 (delta 6), reused 0 (delta 0)
    To bitbucket.org:simoncblyth/opticks.git
       31a2c9e75..29a47cb7d  master -> master
    epsilon:opticks blyth$ 



