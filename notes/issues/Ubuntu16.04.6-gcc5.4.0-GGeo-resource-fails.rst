Ubuntu16.04.6-gcc5.4.0-GGeo-resource-fails : FIXED
===================================================

Overview
---------

Fixed by doing the gdml2gltf and rebuilding geocache 
using the non-standard "OpticksGeoTest -G".
Now down to 2 expected FAILs in "om-test :opticksgeo".



Try building geocache for default DYB geometry (old route)
--------------------------------------------------------------

::

   VERBOSE=1 op.sh --gdml2gltf


Testing Partial Install
------------------------

::

    blyth@blyth-VirtualBox:~/opticks$ om-subs :opticksgeo
    okconf
    sysrap
    boostrap
    npy
    yoctoglrap
    optickscore
    ggeo
    assimprap
    openmeshrap
    opticksgeo

    blyth@blyth-VirtualBox:~/opticks$ om-test :opticksgeo
    ...
    FAILS:
      1  /1   Test #1  : OKConfTest.OKConfTest                         Child aborted***Exception:     0.15   
      1  /34  Test #1  : SysRapTest.SOKConfTest                        Child aborted***Exception:     0.18   
                                   EXPECTED : AS NO CUDA/OptiX

      10 /50  Test #10 : GGeoTest.GMaterialLibTest                     Child aborted***Exception:     0.16   
      13 /50  Test #13 : GGeoTest.GScintillatorLibTest                 ***Exception: SegFault         0.19   
      16 /50  Test #16 : GGeoTest.GBndLibTest                          Child aborted***Exception:     0.20   
      17 /50  Test #17 : GGeoTest.GBndLibInitTest                      Child aborted***Exception:     0.17   
      30 /50  Test #30 : GGeoTest.GPmtTest                             Child aborted***Exception:     0.17   
      31 /50  Test #31 : GGeoTest.BoundariesNPYTest                    Child aborted***Exception:     0.17   
      32 /50  Test #32 : GGeoTest.GAttrSeqTest                         Child aborted***Exception:     0.24   
      36 /50  Test #36 : GGeoTest.GGeoLibTest                          Child aborted***Exception:     0.16   
      37 /50  Test #37 : GGeoTest.GGeoTest                             Child aborted***Exception:     0.21   
      38 /50  Test #38 : GGeoTest.GMakerTest                           Child aborted***Exception:     0.17   
      45 /50  Test #45 : GGeoTest.GSurfaceLibTest                      Child aborted***Exception:     0.17   
      47 /50  Test #47 : GGeoTest.NLookupTest                          Child aborted***Exception:     0.19   
      48 /50  Test #48 : GGeoTest.RecordsNPYTest                       Child aborted***Exception:     0.23   
      49 /50  Test #49 : GGeoTest.GSceneTest                           Child aborted***Exception:     0.19   
      1  /3   Test #1  : OpticksGeoTest.OpticksGeoTest                 Child aborted***Exception:     0.21   
      2  /3   Test #2  : OpticksGeoTest.OpticksHubTest                 Child aborted***Exception:     0.18   
    blyth@blyth-VirtualBox:~/opticks/boostrap$ 


Ahha cannot build OKTest from ok package (as it is placed after CUDA needing subs) 
so cannot do the normal "op.sh -G" which runs OKTest to populate the geocache

* this is an example of mixed up dependency order, as should not need CUDA or OptiX to do this
  is just an accident of the om-subs ordering 

* need to find an alternate executable from opticksgeo(?) to do the honours : try "OpticksGeoTest -G"

::

    blyth@blyth-VirtualBox:~/opticks/opticksgeo$ ll /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/
    total 12
    drwxrwxr-x 3 blyth blyth 4096 Apr  6 14:13 ./
    drwxrwxr-x 4 blyth blyth 4096 Apr  6 14:13 ../
    drwxrwxr-x 2 blyth blyth 4096 Apr  6 14:13 MeshIndex/
    blyth@blyth-VirtualBox:~/opticks/opticksgeo$ 



Create geocache with "OpticksGeoTest -G" as OKTest not built without CUDA
-----------------------------------------------------------------------------

::

    blyth@blyth-VirtualBox:~/opticks/opticksgeo$ OpticksGeoTest -G
    2019-04-06 15:20:18.059 WARN  [4859] [OpticksResource::readG4Environment@508] OpticksResource::readG4Environment MISSING inipath /usr/local/opticks/externals/config/geant4.ini (create it with bash functions: g4-;g4-export-ini ) 
    2019-04-06 15:20:18.063 WARN  [4859] [OpticksHub::configure@296] OpticksHub::configure FORCED COMPUTE MODE : as remote session detected 
    2019-04-06 15:20:18.066 INFO  [4859] [OpticksHub::loadGeometry@469] OpticksHub::loadGeometry START
    2019-04-06 15:20:18.086 INFO  [4859] [OpticksGeometry::loadGeometry@87] OpticksGeometry::loadGeometry START 
    2019-04-06 15:20:18.086 ERROR [4859] [OpticksGeometry::loadGeometryBase@119] OpticksGeometry::loadGeometryBase START 
    2019-04-06 15:20:18.086 INFO  [4859] [GGeo::loadGeometry@569] GGeo::loadGeometry START loaded 0 gltf 0
    2019-04-06 15:20:18.087 ERROR [4859] [GGeo::loadFromG4DAE@617] GGeo::loadFromG4DAE START
    AssimpImporter::init verbosity 0 severity.Err Err severity.Warn Warn severity.Info no-Info severity.Debugging no-Debugging
    myStream Warn,  T0: warn
    myStream Error, T0: error
    2019-04-06 15:20:19.815 INFO  [4859] [OpticksResource::getSensorList@1140] OpticksResource::getSensorList NSensorList:  NSensor count 6888 distinct identier count 684
    2019-04-06 15:20:19.835 FATAL [4859] [GMaterialLib::setCathode@1063]  have already set that cathode GMaterial : __dd__Materials__Bialkali0xc2f2428
    2019-04-06 15:20:19.855 INFO  [4859] [GMaterialLib::addTestMaterials@1037] GMaterialLib::addTestMaterials name                  GlassSchottF2 path $OPTICKS_INSTALL_PREFIX/opticksdata/refractiveindex/tmp/glass/schott/F2.npy
    2019-04-06 15:20:19.856 INFO  [4859] [GMaterialLib::addTestMaterials@1037] GMaterialLib::addTestMaterials name                    MainH2OHale path $OPTICKS_INSTALL_PREFIX/opticksdata/refractiveindex/tmp/main/H2O/Hale.npy
    GMaterial::Summary material 48 eea9fbca69f8dc128e3c08938b81626a __dd__Materials__Bialkali0xc2f2428




After that are down to 3 fails up to opticksgeo, 2 expected
--------------------------------------------------------------

::
    blyth@blyth-VirtualBox:~/opticks$ om-test :opticksgeo
    ...
    LOGS:
    CTestLog :               okconf :      1/     1 : 2019-04-06 15:22:55.329477 : /usr/local/opticks/build/okconf/ctest.log 
    CTestLog :               sysrap :      1/    34 : 2019-04-06 15:22:55.693478 : /usr/local/opticks/build/sysrap/ctest.log 
    CTestLog :             boostrap :      0/    30 : 2019-04-06 15:22:55.877478 : /usr/local/opticks/build/boostrap/ctest.log 
    CTestLog :                  npy :      0/   118 : 2019-04-06 15:22:57.693480 : /usr/local/opticks/build/npy/ctest.log 
    CTestLog :           yoctoglrap :      0/     4 : 2019-04-06 15:22:57.769480 : /usr/local/opticks/build/yoctoglrap/ctest.log 
    CTestLog :          optickscore :      0/    26 : 2019-04-06 15:22:58.089480 : /usr/local/opticks/build/optickscore/ctest.log 
    CTestLog :                 ggeo :      1/    50 : 2019-04-06 15:23:01.505485 : /usr/local/opticks/build/ggeo/ctest.log 
    CTestLog :            assimprap :      0/     3 : 2019-04-06 15:24:59.905622 : /usr/local/opticks/build/assimprap/ctest.log 
    CTestLog :          openmeshrap :      0/     1 : 2019-04-06 15:24:59.961622 : /usr/local/opticks/build/openmeshrap/ctest.log 
    CTestLog :           opticksgeo :      0/     3 : 2019-04-06 15:25:59.241686 : /usr/local/opticks/build/opticksgeo/ctest.log 
     totals  3   / 270 


    FAILS:
      1  /1   Test #1  : OKConfTest.OKConfTest                         Child aborted***Exception:     0.16   
      1  /34  Test #1  : SysRapTest.SOKConfTest                        Child aborted***Exception:     0.17   
                           EXPECTED : FROM LACK OF OptiX/CUDA
                      
      38 /50  Test #38 : GGeoTest.GMakerTest                           Child aborted***Exception:     0.22   


GMakerTest fail from lack of ImplicitMesher support in NPY : FIXED
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* fixed by installing oimplicitmesher- and rebuilding npy- 

::

    blyth@blyth-VirtualBox:~/opticks/ggeo$ GMakerTest 
    2019-04-06 15:29:31.551 WARN  [6300] [OpticksResource::readG4Environment@508] OpticksResource::readG4Environment MISSING inipath /usr/local/opticks/externals/config/geant4.ini (create it with bash functions: g4-;g4-export-ini ) 
    ...
    GMakerTest: /home/blyth/opticks/npy/NPolygonizer.cpp:225: NTrianglesNPY* NPolygonizer::implicitMesher(): Assertion `0 && "installation does not have ImplicitMesher support"' failed.
    Aborted (core dumped)


    blyth@blyth-VirtualBox:~/opticks/ggeo$ oimplicitmesher-;oimplicitmesher--
    remote: Warning: Permanently added the RSA host key for IP address '18.205.93.2' to the list of known hosts.
    Enter passphrase for key '/home/blyth/.ssh/id_rsa':     ## password needed because for USER "blyth" the ssh bitbucket url is used
    destination directory: ImplicitMesher

::
 
    cd ~/opticks/npy
    om-conf
    om-make



Down to 2 expected fails
--------------------------

::

    blyth@blyth-VirtualBox:~/opticks$ om-test :opticksgeo
    === om-test-one : okconf          /home/blyth/opticks/okconf                                   /usr/local/opticks/build/okconf                              
    Sat Apr  6 16:55:23 CST 2019
    ...
    GS:
    CTestLog :               okconf :      1/     1 : 2019-04-06 16:55:23.682509 : /usr/local/opticks/build/okconf/ctest.log 
    CTestLog :               sysrap :      1/    34 : 2019-04-06 16:55:24.034508 : /usr/local/opticks/build/sysrap/ctest.log 
    CTestLog :             boostrap :      0/    30 : 2019-04-06 16:55:24.234507 : /usr/local/opticks/build/boostrap/ctest.log 
    CTestLog :                  npy :      0/   119 : 2019-04-06 16:55:26.554499 : /usr/local/opticks/build/npy/ctest.log 
    CTestLog :           yoctoglrap :      0/     4 : 2019-04-06 16:55:26.634498 : /usr/local/opticks/build/yoctoglrap/ctest.log 
    CTestLog :          optickscore :      0/    26 : 2019-04-06 16:55:26.994497 : /usr/local/opticks/build/optickscore/ctest.log 
    CTestLog :                 ggeo :      0/    50 : 2019-04-06 16:55:44.902432 : /usr/local/opticks/build/ggeo/ctest.log 
    CTestLog :            assimprap :      0/     3 : 2019-04-06 16:57:45.426052 : /usr/local/opticks/build/assimprap/ctest.log 
    CTestLog :          openmeshrap :      0/     1 : 2019-04-06 16:57:45.490051 : /usr/local/opticks/build/openmeshrap/ctest.log 
    CTestLog :           opticksgeo :      0/     3 : 2019-04-06 16:58:45.505897 : /usr/local/opticks/build/opticksgeo/ctest.log 
     totals  2   / 271 


    FAILS:
      1  /1   Test #1  : OKConfTest.OKConfTest                         Child aborted***Exception:     0.16   
      1  /34  Test #1  : SysRapTest.SOKConfTest                        Child aborted***Exception:     0.15   


