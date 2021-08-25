opticks-t-fails-aug-2021-13-of-493
======================================


::

    SLOW: tests taking longer that 15 seconds
      31 /31  Test #31 : ExtG4Test.X4SurfaceTest                       Passed                         45.15  
      8  /46  Test #8  : CFG4Test.CG4Test                              Subprocess aborted***Exception:  53.21  
      1  /1   Test #1  : OKG4Test.OKG4Test                             Subprocess aborted***Exception:  67.93  


    FAILS:  13  / 493   :  Wed Aug 25 18:39:55 2021   
      25 /35  Test #25 : OptiXRapTest.interpolationTest                ***Failed                      4.97   
      18 /31  Test #18 : ExtG4Test.X4CSGTest                           ***Exception: SegFault         0.13   
      20 /31  Test #20 : ExtG4Test.X4GDMLParserTest                    ***Exception: SegFault         0.14   
      21 /31  Test #21 : ExtG4Test.X4GDMLBalanceTest                   ***Exception: SegFault         0.15   
      1  /46  Test #1  : CFG4Test.CMaterialLibTest                     Subprocess aborted***Exception:   2.40   
      2  /46  Test #2  : CFG4Test.CMaterialTest                        Subprocess aborted***Exception:   2.38   
      8  /46  Test #8  : CFG4Test.CG4Test                              Subprocess aborted***Exception:  53.21  
      30 /46  Test #30 : CFG4Test.CGROUPVELTest                        Subprocess aborted***Exception:   2.44   
      32 /46  Test #32 : CFG4Test.CTreeJUNOTest                        ***Exception: SegFault         0.22   
      38 /46  Test #38 : CFG4Test.CCerenkovGeneratorTest               Subprocess aborted***Exception:   2.38   
      39 /46  Test #39 : CFG4Test.CGenstepSourceTest                   Subprocess aborted***Exception:   2.35   
      1  /1   Test #1  : OKG4Test.OKG4Test                             Subprocess aborted***Exception:  67.93  
      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      0.87   
    O[blyth@localhost opticks]$ 




Quick Look at causes
----------------------



1. interpolationTest : python without numpy issue, SSys::RunPythonScript needs envvar to steer to correct python



X4 GDML tempStr fails
------------------------


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


