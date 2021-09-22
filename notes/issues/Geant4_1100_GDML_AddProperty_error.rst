Geant4_1100_GDML_AddProperty_error
=====================================


* it appears that 1100 cannot read GDML with non-standard material property keys such as FASTCOMPONENT 
  because of a lack of createNewKey bool 


::

    91% tests passed, 4 tests failed out of 45

    Total Test time (real) =  37.00 sec

    The following tests FAILED:
          3 - CFG4Test.CTestDetectorTest (Child aborted)
          5 - CFG4Test.CGDMLDetectorTest (Child aborted)
          7 - CFG4Test.CGeometryTest (Child aborted)
         27 - CFG4Test.CInterpolationTest (Child aborted)
    Errors while running CTest
    Thu Sep 23 03:16:15 CST 2021


::

    (base) [simon@localhost cfg4]$ gdb CTestDetectorTest 

    (gdb) bt
    #0  0x00007fffe5c6c387 in raise () from /lib64/libc.so.6
    #1  0x00007fffe5c6da78 in abort () from /lib64/libc.so.6
    #2  0x00007fffeee92bd0 in G4Exception (originOfException=0x7fffef751900 "G4MaterialPropertiesTable::AddProperty()", exceptionCode=0x7fffef7519ae "mat221", severity=FatalException, 
        description=0x1b2c8ea8 "Attempting to create a new material property key FASTCOMPONENT without setting\ncreateNewKey parameter of AddProperty to true.")
        at /home/simon/local/opticks_externals/g4_1100.build/geant4.11.00.b01/source/global/management/src/G4Exception.cc:88
    #3  0x00007fffeee92d9d in G4Exception (originOfException=0x7fffef751900 "G4MaterialPropertiesTable::AddProperty()", exceptionCode=0x7fffef7519ae "mat221", severity=FatalException, description=...)
        at /home/simon/local/opticks_externals/g4_1100.build/geant4.11.00.b01/source/global/management/src/G4Exception.cc:104
    #4  0x00007fffef6e550c in G4MaterialPropertiesTable::AddProperty (this=0x1b2bfc30, key=..., mpv=0x1b2c46b0, createNewKey=false)
        at /home/simon/local/opticks_externals/g4_1100.build/geant4.11.00.b01/source/materials/src/G4MaterialPropertiesTable.cc:377
    #5  0x00007ffff59af432 in G4GDMLReadMaterials::PropertyRead (this=0xb5cdc10, propertyElement=0xb6c9270, material=0x1b2b7c90)
        at /home/simon/local/opticks_externals/g4_1100.build/geant4.11.00.b01/source/persistency/gdml/src/G4GDMLReadMaterials.cc:877
    #6  0x00007ffff59ae728 in G4GDMLReadMaterials::MaterialRead (this=0xb5cdc10, materialElement=0xb6c80f0)
        at /home/simon/local/opticks_externals/g4_1100.build/geant4.11.00.b01/source/persistency/gdml/src/G4GDMLReadMaterials.cc:707
    #7  0x00007ffff59af74e in G4GDMLReadMaterials::MaterialsRead (this=0xb5cdc10, materialsElement=0xb6b52a8)
        at /home/simon/local/opticks_externals/g4_1100.build/geant4.11.00.b01/source/persistency/gdml/src/G4GDMLReadMaterials.cc:920
    #8  0x00007ffff59a3a81 in G4GDMLRead::Read (this=0xb5cdc10, fileName=..., validation=false, isModule=false, strip=false)
        at /home/simon/local/opticks_externals/g4_1100.build/geant4.11.00.b01/source/persistency/gdml/src/G4GDMLRead.cc:505
    #9  0x00007ffff7b21bf2 in G4GDMLParser::Read (this=0x7fffffffb940, filename=..., validate=false) at /home/simon/local/opticks_externals/g4_1100/include/Geant4/G4GDMLParser.icc:35
    #10 0x00007ffff7b20f2b in CGDMLDetector::parseGDML (this=0xb5c51c0, 
        path=0x6d2660 "/home/simon/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/50859f8d4163ea73814016bc7008ec4d/1/origin_CGDMLKludge.gdml") at /home/simon/opticks/cfg4/CGDMLDetector.cc:121
    #11 0x00007ffff7b20d51 in CGDMLDetector::init (this=0xb5c51c0) at /home/simon/opticks/cfg4/CGDMLDetector.cc:91
    #12 0x00007ffff7b209f2 in CGDMLDetector::CGDMLDetector (this=0xb5c51c0, hub=0x7fffffffc7e0, query=0x6ccad0, sd=0xb5c2a80) at /home/simon/opticks/cfg4/CGDMLDetector.cc:63
    #13 0x00007ffff7ac6372 in CGeometry::init (this=0xb5c50e0) at /home/simon/opticks/cfg4/CGeometry.cc:99
    #14 0x00007ffff7ac6168 in CGeometry::CGeometry (this=0xb5c50e0, hub=0x7fffffffc7e0, sd=0xb5c2a80) at /home/simon/opticks/cfg4/CGeometry.cc:82
    #15 0x00007ffff7b37c07 in CG4::CG4 (this=0x7fffffffc700, hub=0x7fffffffc7e0) at /home/simon/opticks/cfg4/CG4.cc:167
    #16 0x0000000000403af9 in main (argc=1, argv=0x7fffffffd0a8) at /home/simon/opticks/cfg4/tests/CTestDetectorTest.cc:52
    (gdb) f 5
    #5  0x00007ffff59af432 in G4GDMLReadMaterials::PropertyRead (this=0xb5cdc10, propertyElement=0xb6c9270, material=0x1b2b7c90)
        at /home/simon/local/opticks_externals/g4_1100.build/geant4.11.00.b01/source/persistency/gdml/src/G4GDMLReadMaterials.cc:877
    877	    matprop->AddProperty(Strip(name), propvect);
    (gdb) f 14
    #14 0x00007ffff7ac6168 in CGeometry::CGeometry (this=0xb5c50e0, hub=0x7fffffffc7e0, sd=0xb5c2a80) at /home/simon/opticks/cfg4/CGeometry.cc:82
    82	    init();
    (gdb) f 13
    #13 0x00007ffff7ac6372 in CGeometry::init (this=0xb5c50e0) at /home/simon/opticks/cfg4/CGeometry.cc:99
    99	        detector  = static_cast<CDetector*>(new CGDMLDetector(m_hub, query, m_sd)) ; 
    (gdb) f 12
    #12 0x00007ffff7b209f2 in CGDMLDetector::CGDMLDetector (this=0xb5c51c0, hub=0x7fffffffc7e0, query=0x6ccad0, sd=0xb5c2a80) at /home/simon/opticks/cfg4/CGDMLDetector.cc:63
    63	    init();
    (gdb) f 11
    #11 0x00007ffff7b20d51 in CGDMLDetector::init (this=0xb5c51c0) at /home/simon/opticks/cfg4/CGDMLDetector.cc:91
    91	    G4VPhysicalVolume* world = parseGDML(path);
    (gdb) f 10
    #10 0x00007ffff7b20f2b in CGDMLDetector::parseGDML (this=0xb5c51c0, 
        path=0x6d2660 "/home/simon/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/50859f8d4163ea73814016bc7008ec4d/1/origin_CGDMLKludge.gdml") at /home/simon/opticks/cfg4/CGDMLDetector.cc:121
    121	    parser.Read(path, validate);
    (gdb) f 9
    #9  0x00007ffff7b21bf2 in G4GDMLParser::Read (this=0x7fffffffb940, filename=..., validate=false) at /home/simon/local/opticks_externals/g4_1100/include/Geant4/G4GDMLParser.icc:35
    35	    reader->Read(filename, validate, false, strip);
    (gdb) f 8
    #8  0x00007ffff59a3a81 in G4GDMLRead::Read (this=0xb5cdc10, fileName=..., validation=false, isModule=false, strip=false)
        at /home/simon/local/opticks_externals/g4_1100.build/geant4.11.00.b01/source/persistency/gdml/src/G4GDMLRead.cc:505
    505	      MaterialsRead(child);
    (gdb) f 7
    #7  0x00007ffff59af74e in G4GDMLReadMaterials::MaterialsRead (this=0xb5cdc10, materialsElement=0xb6b52a8)
        at /home/simon/local/opticks_externals/g4_1100.build/geant4.11.00.b01/source/persistency/gdml/src/G4GDMLReadMaterials.cc:920
    920	      MaterialRead(child);
    (gdb) f 6
    #6  0x00007ffff59ae728 in G4GDMLReadMaterials::MaterialRead (this=0xb5cdc10, materialElement=0xb6c80f0)
        at /home/simon/local/opticks_externals/g4_1100.build/geant4.11.00.b01/source/persistency/gdml/src/G4GDMLReadMaterials.cc:707
    707	      PropertyRead(child, material);
    (gdb) 

