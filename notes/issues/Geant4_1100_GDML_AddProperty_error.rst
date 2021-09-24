Geant4_1100_GDML_AddProperty_error
=====================================

1100 beta fails now stand at::

    FAILS:  5   / 492   :  Thu Sep 23 18:54:04 2021   
      30 /31  Test #30 : ExtG4Test.X4ScintillationTest                 Child aborted***Exception:     0.55   
      3  /45  Test #3  : CFG4Test.CTestDetectorTest                    Child aborted***Exception:     4.72   
      5  /45  Test #5  : CFG4Test.CGDMLDetectorTest                    Child aborted***Exception:     4.62   
      7  /45  Test #7  : CFG4Test.CGeometryTest                        Child aborted***Exception:     5.04   
      27 /45  Test #27 : CFG4Test.CInterpolationTest                   Child aborted***Exception:     4.76   



First X4ScintillationTest is from a difference in the ICDF persisted into the geocache and the one created in the test : 
suggesting an domain interpolation "::GetEnergy" difference between Geant4 1042 that created the geocache and 1100 that ran the test.

* :doc:`X4ScintillationTest_1100_icdf_all_values_slightly_off`

Found the X4ScintillationTest fail to be explained by h_Planck change between CLHEP versions::

    FAILS:  4   / 497   :  Fri Sep 24 19:43:59 2021   
      3  /45  Test #3  : CFG4Test.CTestDetectorTest                    Child aborted***Exception:     4.66   
      5  /45  Test #5  : CFG4Test.CGDMLDetectorTest                    Child aborted***Exception:     4.64   
      7  /45  Test #7  : CFG4Test.CGeometryTest                        Child aborted***Exception:     4.50   
      27 /45  Test #27 : CFG4Test.CInterpolationTest                   Child aborted***Exception:     4.63   


CFG4 fails are from a GDML issue : it appears that 1100 cannot read GDML with non-standard material property keys such as FASTCOMPONENT 
because of a lack of createNewKey bool. 


GDML Reading issue in 1100
---------------------------------


::

    (base) [simon@localhost opticks]$ gdb CGeometryTest 
    GNU gdb (GDB) Red Hat Enterprise Linux 7.6.1-114.el7
    ...
    Reading symbols from /home/simon/local/opticks/lib/CGeometryTest...done.
    (gdb) r
    Starting program: /home/simon/local/opticks/lib/CGeometryTest 
    2021-09-24 19:48:24.245 INFO  [39504] [main@61] /home/simon/local/opticks/lib/CGeometryTest
    G4GDML: Reading '/home/simon/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/50859f8d4163ea73814016bc7008ec4d/1/origin_CGDMLKludge.gdml'...
    G4GDML: Reading definitions...
    G4GDML: Reading materials...

    -------- EEEE ------- G4Exception-START -------- EEEE -------

    *** ExceptionHandler is not defined ***
    *** G4Exception : mat221
          issued by : G4MaterialPropertiesTable::AddProperty()
    Attempting to create a new material property key FASTCOMPONENT without setting
    createNewKey parameter of AddProperty to true.
    *** Fatal Exception ***
    -------- EEEE ------- G4Exception-END -------- EEEE -------


    *** G4Exception: Aborting execution ***

    Program received signal SIGABRT, Aborted.
    ...
    #2  0x00007fffeee91bd0 in G4Exception (originOfException=0x7fffef750900 "G4MaterialPropertiesTable::AddProperty()", exceptionCode=0x7fffef7509ae "mat221", severity=FatalException, 
        description=0x1b0e1768 "Attempting to create a new material property key FASTCOMPONENT without setting\ncreateNewKey parameter of AddProperty to true.")
        at /home/simon/local/opticks_externals/g4_1100.build/geant4.11.00.b01/source/global/management/src/G4Exception.cc:88
    #3  0x00007fffeee91d9d in G4Exception (originOfException=0x7fffef750900 "G4MaterialPropertiesTable::AddProperty()", exceptionCode=0x7fffef7509ae "mat221", severity=FatalException, description=...)
        at /home/simon/local/opticks_externals/g4_1100.build/geant4.11.00.b01/source/global/management/src/G4Exception.cc:104
    #4  0x00007fffef6e450c in G4MaterialPropertiesTable::AddProperty (this=0x1b0d84f0, key=..., mpv=0x1b0dcf70, createNewKey=false)
        at /home/simon/local/opticks_externals/g4_1100.build/geant4.11.00.b01/source/materials/src/G4MaterialPropertiesTable.cc:377
    #5  0x00007ffff59ae432 in G4GDMLReadMaterials::PropertyRead (this=0xb3cec40, propertyElement=0xb4e1b10, material=0x1b0d0550)
        at /home/simon/local/opticks_externals/g4_1100.build/geant4.11.00.b01/source/persistency/gdml/src/G4GDMLReadMaterials.cc:877
    #6  0x00007ffff59ad728 in G4GDMLReadMaterials::MaterialRead (this=0xb3cec40, materialElement=0xb4e0990)
        at /home/simon/local/opticks_externals/g4_1100.build/geant4.11.00.b01/source/persistency/gdml/src/G4GDMLReadMaterials.cc:707
    #7  0x00007ffff59ae74e in G4GDMLReadMaterials::MaterialsRead (this=0xb3cec40, materialsElement=0xb4cdb48)
        at /home/simon/local/opticks_externals/g4_1100.build/geant4.11.00.b01/source/persistency/gdml/src/G4GDMLReadMaterials.cc:920
    #8  0x00007ffff59a2a81 in G4GDMLRead::Read (this=0xb3cec40, fileName=..., validation=false, isModule=false, strip=false)
        at /home/simon/local/opticks_externals/g4_1100.build/geant4.11.00.b01/source/persistency/gdml/src/G4GDMLRead.cc:505
    #9  0x00007ffff7b21dcc in G4GDMLParser::Read (this=0x7fffffffb9d0, filename=..., validate=false) at /home/simon/local/opticks_externals/g4_1100/include/Geant4/G4GDMLParser.icc:35
    #10 0x00007ffff7b21105 in CGDMLDetector::parseGDML (this=0xb3c67a0, 
        path=0x6d3400 "/home/simon/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/50859f8d4163ea73814016bc7008ec4d/1/origin_CGDMLKludge.gdml") at /home/simon/opticks/cfg4/CGDMLDetector.cc:121
    #11 0x00007ffff7b20f2b in CGDMLDetector::init (this=0xb3c67a0) at /home/simon/opticks/cfg4/CGDMLDetector.cc:91
    #12 0x00007ffff7b20bcc in CGDMLDetector::CGDMLDetector (this=0xb3c67a0, hub=0x7fffffffc8e0, query=0x6cd8f0, sd=0x0) at /home/simon/opticks/cfg4/CGDMLDetector.cc:63
    #13 0x00007ffff7ac6482 in CGeometry::init (this=0x7fffffffc890) at /home/simon/opticks/cfg4/CGeometry.cc:99
    #14 0x00007ffff7ac6278 in CGeometry::CGeometry (this=0x7fffffffc890, hub=0x7fffffffc8e0, sd=0x0) at /home/simon/opticks/cfg4/CGeometry.cc:82
    #15 0x000000000040411b in main (argc=1, argv=0x7fffffffce68) at /home/simon/opticks/cfg4/tests/CGeometryTest.cc:66
    (gdb) 





