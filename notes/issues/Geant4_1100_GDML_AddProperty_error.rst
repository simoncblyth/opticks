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

Last four fails are from a GDML issue : it appears that 1100 cannot read GDML with non-standard material property keys such as FASTCOMPONENT 
because of a lack of createNewKey bool. 


X4ScintillationTest
-----------------------



No problem with 1042, O:: 

    x4 ; ipython -i tests/X4ScintillationTest.py 

    O[blyth@localhost extg4]$ ipython -i tests/X4ScintillationTest.py 
    Python 2.7.5 (default, Nov 16 2020, 22:23:17) 
    Type "copyright", "credits" or "license" for more information.

    IPython 3.2.1 -- An enhanced Interactive Python.
    ?         -> Introduction and overview of IPython's features.
    %quickref -> Quick reference.
    help      -> Python's own help system.
    object?   -> Details about 'object', use 'object??' for extra details.
    INFO:__main__:icdf_jspath:/tmp/blyth/opticks/X4ScintillationTest/g4icdf_auto.json
    INFO:__main__: num_bins : 4096 
    INFO:__main__: edge : 0.05 
    INFO:__main__: hd_factor : 20 
    INFO:__main__: name : LS 
    INFO:__main__: creator : X4Scintillation::CreateGeant4InterpolatedInverseCDF 
    INFO:__main__:icdf_compare
    a:(3, 4096) a.min    200.118 a.max    799.898
    b.(3, 4096) b.min    200.118 b.max    799.898
    ab:(3, 4096) ab.min          0 ab.max          0

    In [1]: a
    Out[1]: 
    array([[ 799.89798414,  785.89756342,  772.37880425, ...,  208.95408887,
             205.87260891,  202.8806943 ],
           [ 799.89798414,  799.18612661,  798.47553497, ...,  485.01049867,
             485.004202  ,  484.99794481],
           [ 391.46193947,  391.46039661,  391.45885375, ...,  200.40510648,
             200.26136376,  200.11782709]])

    In [2]: ab
    Out[2]: 
    array([[ 0.,  0.,  0., ...,  0.,  0.,  0.],
           [ 0.,  0.,  0., ...,  0.,  0.,  0.],
           [ 0.,  0.,  0., ...,  0.,  0.,  0.]])


With 1100 using same geoache as O, not just a few values off, all values are off at 1e-5/1e-4 level::

    x4 ; ipython -i tests/X4ScintillationTest.py 

    NFO:__main__: num_bins : 4096 
    INFO:__main__:icdf_compare
    a:(3, 4096) a.min    200.118 a.max    799.898
    b.(3, 4096) b.min    200.118 b.max    799.898
    ab:(3, 4096) ab.min 1.7579e-05 ab.max 7.02658e-05

    In [1]: a
    Out[1]: 
    array([[799.89798414, 785.89756342, 772.37880425, ..., 208.95408887,
            205.87260891, 202.8806943 ],
           [799.89798414, 799.18612661, 798.47553497, ..., 485.01049867,
            485.004202  , 484.99794481],
           [391.46193947, 391.46039661, 391.45885375, ..., 200.40510648,
            200.26136376, 200.11782709]])

    In [2]: ab
    Out[2]: 
    array([[7.02658095e-05, 6.90359639e-05, 6.78484295e-05, ...,
            1.83552509e-05, 1.80845630e-05, 1.78217429e-05],
           [7.02658095e-05, 7.02032775e-05, 7.01408566e-05, ...,
            4.26050020e-05, 4.26044490e-05, 4.26038993e-05],
           [3.43873726e-05, 3.43872371e-05, 3.43871016e-05, ...,
            1.76042787e-05, 1.75916518e-05, 1.75790431e-05]])

    In [3]: ab.shape
    Out[3]: (3, 4096)




Compare the constants::


    In [2]: import numpy as np

    In [3]: a = np.load("/tmp/simon/opticks/X4PhysicalConstantsTest/1100.npy")

    In [4]: b = np.load("/tmp/blyth/opticks/X4PhysicalConstantsTest/1042.npy")

    In [5]: a
    Out[5]: 
    array([4.13566770e-12, 2.99792458e+02, 1.23984198e-09, 1.23984198e-03,
           1.00000000e-06])

    In [6]: b
    Out[6]: 
    array([4.13566733e-12, 2.99792458e+02, 1.23984188e-09, 1.23984188e-03,
           1.00000000e-06])

    In [7]: a-b
    Out[7]: 
    array([3.63291343e-19, 0.00000000e+00, 1.08912005e-16, 1.08912005e-10,
           0.00000000e+00])



Compare the integrals, they match exactly::

    In [1]: import numpy as np

    In [2]: a = np.load("/tmp/simon/opticks/X4ScintillationTest/ScintillatorIntegral.npy")

    In [3]: b = np.load("/tmp/blyth/opticks/X4ScintillationTest/ScintillatorIntegral.npy")


    In [9]: ab = np.abs(a - b )

    In [10]: ab.min()
    Out[10]: 0.0

    In [11]: ab.max()
    Out[11]: 0.0













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

