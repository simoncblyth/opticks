oxrap OInterpolationTest asserts at python level for lack of IDPATH GBndLib.txt
==================================================================================



FIXED : OInterpolationTest PASS depending on CInterpolationTest having been run recently (/tmp unwiped) 
----------------------------------------------------------------------------------------------------------

First run failing for lack of CInterpolationTest_interpol.npy

::

    simon:optixrap blyth$ ll /tmp/blyth/opticks/InterpolationTest/
    total 48336
    drwxr-xr-x    3 blyth  wheel       102 Oct 23 11:00 GItemList
    drwxr-xr-x    5 blyth  wheel       170 Oct 23 11:00 GBndLib
    -rw-r--r--    1 blyth  wheel  12370912 Oct 23 11:01 CInterpolationTest_interpol.npy
    drwxr-xr-x    6 blyth  wheel       204 Oct 23 11:01 .
    drwxr-xr-x  170 blyth  wheel      5780 Oct 23 11:01 ..
    -rw-r--r--    1 blyth  wheel  12370896 Oct 23 11:02 OInterpolationTest_interpol.npy
    simon:optixrap blyth$ 



Hmm SG Fail
------------

::

    2017-10-18 20:14:23.329 INFO  [30557] [OContext::close@245] OContext::close m_cfg->apply() done.
    2017-10-18 20:14:28.137 INFO  [30557] [OContext::launch@322] OContext::launch LAUNCH time: 4.80798
    No handlers could be found for logger "opticks.ana.proplib"
    Traceback (most recent call last):
      File "/home/simon/opticks/optixrap/tests/OInterpolationTest_interpol.py", line 20, in <module>
        c = np.load(os.path.expandvars(os.path.join(base,"CInterpolationTest_%s.npy" % ext))).reshape(-1,4,2,nl,4) 
      File "/usr/local/anaconda2/lib/python2.7/site-packages/numpy/lib/npyio.py", line 370, in load
        fid = open(file, "rb")
    IOError: [Errno 2] No such file or directory: '/tmp/simon/opticks/InterpolationTest/CInterpolationTest_interpol.npy'
    2017-10-18 20:14:28.272 INFO  [30557] [SSys::run@46] python /home/simon/opticks/optixrap/tests/OInterpolationTest_interpol.py rc_raw : 256 rc : 1
    2017-10-18 20:14:28.272 WARN  [30557] [SSys::run@52] SSys::run FAILED with  cmd python /home/simon/opticks/optixrap/tests/OInterpolationTest_interpol.py possibly you need to set export PATH=$OPTICKS_HOME/ana:$OPTICKS_HOME/bin:/usr/local/opticks/lib:$PATH 
    [simon@localhost opticks]$ 

    [simon@localhost opticks]$ ll /tmp/simon/opticks/InterpolationTest/CInterpolationTest_interpol.npy
    ls: cannot access /tmp/simon/opticks/InterpolationTest/CInterpolationTest_interpol.npy: No such file or directory

    [simon@localhost opticks]$ ll /tmp/simon/opticks/InterpolationTest/
    total 11712
    drwxrwxr-x. 2 simon simon     4096 Oct 18 20:10 GBndLib
    drwxrwxr-x. 2 simon simon     4096 Oct 18 20:10 GItemList
    -rw-rw-r--. 1 simon simon 11981264 Oct 18 20:14 OInterpolationTest_interpol.npy
    [simon@localhost opticks]$ 


Oct 2017 : FIXED old chestnut 
---------------------------------------

* FIXED using GBndLib::saveAllOverride and overhaul of paths in the analysis scripts


::


    2017-10-18 16:04:47.536 INFO  [151806] [OLaunchTest::init@50] OLaunchTest entry   0 width       1 height       1 ptx                          OInterpolationTest.cu.ptx prog                                 OInterpolationTest
    2017-10-18 16:04:47.536 INFO  [151806] [OLaunchTest::launch@61] OLaunchTest entry   0 width     761 height     123 ptx                          OInterpolationTest.cu.ptx prog                                 OInterpolationTest
    2017-10-18 16:04:47.536 INFO  [151806] [OContext::close@235] OContext::close numEntryPoint 1
    2017-10-18 16:04:47.536 INFO  [151806] [OContext::close@239] OContext::close setEntryPointCount done.
    2017-10-18 16:04:47.548 INFO  [151806] [OContext::close@245] OContext::close m_cfg->apply() done.
    2017-10-18 16:04:50.920 INFO  [151806] [OContext::launch@322] OContext::launch LAUNCH time: 3.37147
    Traceback (most recent call last):
      File "/Users/blyth/opticks/optixrap/tests/OInterpolationTest_interpol.py", line 13, in <module>
        blib = PropLib("GBndLib")
      File "/Users/blyth/opticks/ana/proplib.py", line 126, in __init__
        names = map(lambda _:_[:-1],file(npath).readlines())
    IOError: [Errno 2] No such file or directory: '/usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GItemList/GBndLib.txt'
    2017-10-18 16:04:51.075 INFO  [151806] [SSys::run@46] python /Users/blyth/opticks/optixrap/tests/OInterpolationTest_interpol.py rc_raw : 256 rc : 1
    2017-10-18 16:04:51.075 WARN  [151806] [SSys::run@52] SSys::run FAILED with  cmd python /Users/blyth/opticks/optixrap/tests/OInterpolationTest_interpol.py possibly you need to set export PATH=$OPTICKS_HOME/ana:$OPTICKS_HOME/bin:/usr/local/opticks/lib:$PATH 
    simon:opticks blyth$ 
    simon:opticks blyth$ 



Old geocache have the missing file::

    simon:g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae blyth$ mdfind GBndLib.txt
    /usr/local/env/geant4/geometry/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GItemList/GBndLib.txt
    /usr/local/env/geant4/geometry/export/juno/test3.fcc8b4dc9474af8826b29bf172452160.dae/GItemList/GBndLib.txt
    /Users/simon/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GItemList/GBndLib.txt

    simon:issues blyth$ cat /usr/local/env/geant4/geometry/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GItemList/GBndLib.txt
    Vacuum///Vacuum
    Vacuum///Rock
    Rock///Air
    Air/NearPoolCoverSurface//PPE
    Air///Aluminium
    Aluminium///Foam
    Foam///Bakelite
    Bakelite///Air
    Air///MixGas
    Air///Air
    Air///Iron
    Rock///Rock
    Rock///DeadWater
    DeadWater/NearDeadLinerSurface//Tyvek
    Tyvek//NearOWSLinerSurface/OwsWater
    OwsWater///Tyvek
    ...



* m_names GItemList is handled in base class GPropertyLib
* GBndLib is special (as boundaries can be dynamically added to test geometry) 
* dynamic nature means GBndLib must be closed before the names are set 
* GBndLibTest saves such a file to $TMP/GItemList/GBndLib.txt 

::

    simon:ggeo blyth$ opticks-find setNames
    ./ggeo/GPropertyLib.cc:    setNames(other->getNames());  // need setter for m_attrnames hookup
    ./ggeo/GPropertyLib.cc:    setNames(names);
    ./ggeo/GPropertyLib.cc:    setNames(names); 
    ./ggeo/GPropertyLib.cc:void GPropertyLib::setNames(GItemList* names)
    ./ggeo/tests/BoundariesNPYTest.cc:    blib->close();     //  BndLib is dynamic so requires a close before setNames is called setting the sequence for OpticksAttrSeq
    ./ggeo/GPropertyLib.hh:        void setNames(GItemList* names);
    simon:opticks blyth$ 








June 2017
------------

::

    99% tests passed, 3 tests failed out of 234

    Total Test time (real) = 118.02 sec

    The following tests FAILED:
        209 - OptiXRapTest.OInterpolationTest (Failed)     

              ## python level missing geocache file GItemList/GBndLib.txt  
              ## was GBndLib closed ?  
              ## hmm run-to-run dynamic files shouldnt be in geocache and it isnt 
              ##
              ##       ... is it persisted elsewhere now and the python was not updated ?
                                                 
        222 - cfg4Test.CMaterialLibTest (OTHER_FAULT)

              ## expecting oil
              ##  Assertion failed: (strcmp(mat.c_str(),"MineralOil")==0), function main, file /Users/blyth/opticks/cfg4/tests/CMaterialLibTest.cc, line 97.
              ##
              ## ... suspect just due to long ago change to finer wavelength sampling , disabled the test 

        223 - cfg4Test.CTestDetectorTest (OTHER_FAULT)

              ##  GGeoTest::createPmtInBox lacking m_bndlib hookup in GParts ???


* GItemList names for all GPropLib as created on closing, but seems that 
  hasnt happened for GBndLib ?


::

    simon:issues blyth$ OInterpolationTest 
    2017-06-15 12:48:30.175 INFO  [7582349] [OpticksDbg::postconfigure@49] OpticksDbg::postconfigure OpticksDbg  debug_photon  size: 0 elem: () other_photon  size: 0 elem: ()
    2017-06-15 12:48:30.350 INFO  [7582349] [*GMergedMesh::load@613] GMergedMesh::load dir /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/0 -> cachedir /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/0 index 0 version (null) existsdir 1
    2017-06-15 12:48:30.462 INFO  [7582349] [*GMergedMesh::load@613] GMergedMesh::load dir /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/1 -> cachedir /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/1 index 1 version (null) existsdir 1
    2017-06-15 12:48:30.546 INFO  [7582349] [GMaterialLib::postLoadFromCache@67] GMaterialLib::postLoadFromCache  nore 0 noab 0 nosc 0 xxre 0 xxab 0 xxsc 0 fxre 0 fxab 0 fxsc 0 groupvel 1
    2017-06-15 12:48:30.546 INFO  [7582349] [GMaterialLib::replaceGROUPVEL@552] GMaterialLib::replaceGROUPVEL  ni 38
    2017-06-15 12:48:30.546 INFO  [7582349] [GPropertyLib::getIndex@338] GPropertyLib::getIndex type GMaterialLib TRIGGERED A CLOSE  shortname [GdDopedLS]
    2017-06-15 12:48:30.546 INFO  [7582349] [GPropertyLib::close@384] GPropertyLib::close type GMaterialLib buf 38,2,39,4
    2017-06-15 12:48:30.552 INFO  [7582349] [GGeo::loadAnalyticPmt@789] GGeo::loadAnalyticPmt AnalyticPMTIndex 0 AnalyticPMTSlice ALL Path /usr/local/opticks/opticksdata/export/DayaBay/GPmt/0
    2017-06-15 12:48:30.560 INFO  [7582349] [SLog::operator@15] OpticksHub::OpticksHub DONE
     s 0 nf   0  i0 0:0  i1 434816:434816   il 0:0 
     s 1 nf   0  i0 0:0  i1 434816:434816   il 0:0 
     s 2 nf   0  i0 0:0  i1 434816:434816   il 0:0 
     ...
     s 12227 nf   0  i0 434816:434816  i1 869632:869632   il 434816:434816 
     s 12228 nf   0  i0 434816:434816  i1 869632:869632   il 434816:434816 
     s 12229 nf   0  i0 434816:434816  i1 869632:869632   il 434816:434816 
     ----- 434816 
     s 0 nf 720  i0 0:720  i1 2928:3648   il 1964688:1965408 
     s 1 nf 672  i0 720:1392  i1 3648:4320   il 1965408:1966080 
     s 2 nf 960  i0 1392:2352  i1 4320:5280   il 1966080:1967040 
     s 3 nf 480  i0 2352:2832  i1 5280:5760   il 1967040:1967520 
     s 4 nf  96  i0 2832:2928  i1 5760:5856   il 1967520:1967616 
     ----- 2928 
    2017-06-15 12:48:31.274 INFO  [7582349] [SLog::operator@15] OScene::OScene DONE
    2017-06-15 12:48:31.274 INFO  [7582349] [main@128]  ok 
    2017-06-15 12:48:31.274 INFO  [7582349] [OInterpolationTest::launch@85] OInterpolationTest::launch nb   123 nx   761 ny   984 progname             OInterpolationTest path $TMP/InterpolationTest/OInterpolationTest_interpol.npy
    2017-06-15 12:48:31.274 INFO  [7582349] [OLaunchTest::init@50] OLaunchTest entry   0 width       1 height       1 ptx                          OInterpolationTest.cu.ptx prog                                 OInterpolationTest
    2017-06-15 12:48:31.274 INFO  [7582349] [OLaunchTest::launch@61] OLaunchTest entry   0 width     761 height     123 ptx                          OInterpolationTest.cu.ptx prog                                 OInterpolationTest
    2017-06-15 12:48:31.274 INFO  [7582349] [OContext::close@219] OContext::close numEntryPoint 1
    Traceback (most recent call last):
      File "/Users/blyth/opticks/optixrap/tests/OInterpolationTest_interpol.py", line 13, in <module>
        blib = PropLib("GBndLib")
      File "/Users/blyth/opticks/ana/proplib.py", line 126, in __init__
        names = map(lambda _:_[:-1],file(npath).readlines())
    IOError: [Errno 2] No such file or directory: '/usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GItemList/GBndLib.txt'
    2017-06-15 12:48:34.919 INFO  [7582349] [SSys::run@46] python /Users/blyth/opticks/optixrap/tests/OInterpolationTest_interpol.py rc_raw : 256 rc : 1
    simon:issues blyth$ 



Pump up the verbosity to see where the huge amounts of output coming from::

    simon:issues blyth$ OInterpolationTest --OXRAP trace
    2017-06-15 12:58:43.262 INFO  [7585657] [OpticksDbg::postconfigure@49] OpticksDbg::postconfigure OpticksDbg  debug_photon  size: 0 elem: () other_photon  size: 0 elem: ()
    2017-06-15 12:58:43.434 INFO  [7585657] [*GMergedMesh::load@613] GMergedMesh::load dir /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/0 -> cachedir /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/0 index 0 version (null) existsdir 1
    2017-06-15 12:58:43.547 INFO  [7585657] [*GMergedMesh::load@613] GMergedMesh::load dir /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/1 -> cachedir /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/1 index 1 version (null) existsdir 1
    2017-06-15 12:58:43.628 INFO  [7585657] [GMaterialLib::postLoadFromCache@67] GMaterialLib::postLoadFromCache  nore 0 noab 0 nosc 0 xxre 0 xxab 0 xxsc 0 fxre 0 fxab 0 fxsc 0 groupvel 1
    2017-06-15 12:58:43.628 INFO  [7585657] [GMaterialLib::replaceGROUPVEL@552] GMaterialLib::replaceGROUPVEL  ni 38
    2017-06-15 12:58:43.628 INFO  [7585657] [GPropertyLib::getIndex@338] GPropertyLib::getIndex type GMaterialLib TRIGGERED A CLOSE  shortname [GdDopedLS]
    2017-06-15 12:58:43.629 INFO  [7585657] [GPropertyLib::close@384] GPropertyLib::close type GMaterialLib buf 38,2,39,4
    2017-06-15 12:58:43.634 INFO  [7585657] [GGeo::loadAnalyticPmt@789] GGeo::loadAnalyticPmt AnalyticPMTIndex 0 AnalyticPMTSlice ALL Path /usr/local/opticks/opticksdata/export/DayaBay/GPmt/0
    2017-06-15 12:58:43.642 INFO  [7585657] [GMergedMesh::dumpSolids@640] OpticksGeometry::loadGeometryBase mesh1 ce0 gfloat4      0.000      0.000    -18.997    149.997 
        0 ce             gfloat4      0.000      0.000    -18.997    149.997  bb bb min   -100.288   -100.288   -168.995  max    100.288    100.288    131.000 
        1 ce             gfloat4      0.005     -0.003    -18.252    146.252  bb bb min    -98.995    -99.003   -164.504  max     99.005     98.997    128.000 
        2 ce             gfloat4      0.005     -0.004     91.998     98.143  bb bb min    -98.138    -98.147     55.996  max     98.148     98.139    128.000 
        3 ce             gfloat4      0.000      0.000     13.066     98.143  bb bb min    -98.143    -98.143    -30.000  max     98.143     98.143     56.131 
        4 ce             gfloat4      0.000      0.000    -81.500     83.000  bb bb min    -27.500    -27.500   -164.500  max     27.500     27.500      1.500 
        0 ni[nf/nv/nidx/pidx] (720,362,3199,3155)  id[nidx,midx,bidx,sidx]  (3199, 47, 27,  0) 
        1 ni[nf/nv/nidx/pidx] (672,338,3200,3199)  id[nidx,midx,bidx,sidx]  (3200, 46, 28,  0) 
        2 ni[nf/nv/nidx/pidx] (960,482,3201,3200)  id[nidx,midx,bidx,sidx]  (3201, 43, 29,  3) 
        3 ni[nf/nv/nidx/pidx] (480,242,3202,3200)  id[nidx,midx,bidx,sidx]  (3202, 44, 30,  0) 
        4 ni[nf/nv/nidx/pidx] ( 96, 50,3203,3200)  id[nidx,midx,bidx,sidx]  (3203, 45, 30,  0) 
    2017-06-15 12:58:43.645 INFO  [7585657] [SLog::operator@15] OpticksHub::OpticksHub DONE
    2017-06-15 12:58:43.645 VERB  [7585657] [OScene::init@85] OScene::init START
    2017-06-15 12:58:44.215 DEBUG [7585657] [OScene::init@99] OScene::init (OContext)
    2017-06-15 12:58:44.216 DEBUG [7585657] [OContext::init@170] OContext::init  mode INTEROP num_ray_type 3
    2017-06-15 12:58:44.216 DEBUG [7585657] [OContext::setStackSize@125] OContext::setStackSize 2180
    2017-06-15 12:58:44.216 DEBUG [7585657] [OContext::setPrintIndex@131] OContext::setPrintIndex 
    2017-06-15 12:58:44.216 DEBUG [7585657] [OScene::init@114] OScene::init (OColors)
    2017-06-15 12:58:44.216 VERB  [7585657] [OConfig::configureSampler@392] OPropertyLib::configureSampler
    2017-06-15 12:58:44.216 DEBUG [7585657] [OScene::init@120] OScene::init (OSourceLib)
    2017-06-15 12:58:44.216 DEBUG [7585657] [OSourceLib::convert@17] OSourceLib::convert
    2017-06-15 12:58:44.216 DEBUG [7585657] [OSourceLib::makeSourceTexture@36] OSourceLib::makeSourceTexture  nx 1024 ny 1
    2017-06-15 12:58:44.216 VERB  [7585657] [OConfig::configureSampler@392] OPropertyLib::configureSampler
    2017-06-15 12:58:44.216 DEBUG [7585657] [OScene::init@126] OScene::init (OScintillatorLib) slice 0:1
    2017-06-15 12:58:44.216 VERB  [7585657] [OScintillatorLib::convert@21] OScintillatorLib::convert from 2,4096,1 ni 2
    2017-06-15 12:58:44.216 VERB  [7585657] [OScintillatorLib::convert@31] OScintillatorLib::convert sliced buffer with 0:1 from 2,4096,1 to 1,4096,1
    2017-06-15 12:58:44.216 VERB  [7585657] [OScintillatorLib::makeReemissionTexture@69] OScintillatorLib::makeReemissionTexture  nx 4096 ny 1 ni 1 nj 4096 nk 1 step 0.000244141 empty 0
    2017-06-15 12:58:44.216 VERB  [7585657] [OConfig::configureSampler@392] OPropertyLib::configureSampler
    2017-06-15 12:58:44.216 VERB  [7585657] [OScintillatorLib::makeReemissionTexture@95] OScintillatorLib::makeReemissionTexture DONE 
    2017-06-15 12:58:44.216 VERB  [7585657] [OScintillatorLib::convert@44] OScintillatorLib::convert DONE
    2017-06-15 12:58:44.216 DEBUG [7585657] [OScene::init@131] OScene::init (OGeo)
    2017-06-15 12:58:44.217 DEBUG [7585657] [OScene::init@133] OScene::init (OGeo) -> setTop
    2017-06-15 12:58:44.217 DEBUG [7585657] [OScene::init@135] OScene::init (OGeo) -> convert
    2017-06-15 12:58:44.217 VERB  [7585657] [OGeo::convert@168] OGeo::convert nmm 2
    2017-06-15 12:58:44.217 VERB  [7585657] [OConfig::createProgram@55] OConfig::createProgram path /usr/local/opticks/installcache/PTX/OptiXRap_generated_TriangleMesh.cu.ptx
    2017-06-15 12:58:44.217 DEBUG [7585657] [OConfig::createProgram@61] OConfig::createProgram /usr/local/opticks/installcache/PTX/OptiXRap_generated_TriangleMesh.cu.ptx:mesh_intersect
    2017-06-15 12:58:44.221 VERB  [7585657] [OConfig::createProgram@55] OConfig::createProgram path /usr/local/opticks/installcache/PTX/OptiXRap_generated_TriangleMesh.cu.ptx
    2017-06-15 12:58:44.221 DEBUG [7585657] [OConfig::createProgram@61] OConfig::createProgram /usr/local/opticks/installcache/PTX/OptiXRap_generated_TriangleMesh.cu.ptx:mesh_bounds
    2017-06-15 12:58:44.222 VERB  [7585657] [OGeo::makeTriangulatedGeometry@583] OGeo::makeTriangulatedGeometry  mmIndex 0 numFaces (PrimitiveCount) 434816 numSolids 12230 numITransforms 1
     s 0 nf   0  i0 0:0  i1 434816:434816   il 0:0 
     s 1 nf   0  i0 0:0  i1 434816:434816   il 0:0 
     s 2 nf   0  i0 0:0  i1 434816:434816   il 0:0 
     s 3 nf   0  i0 0:0  i1 434816:434816   il 0:0 
     s 4 nf   0  i0 0:0  i1 434816:434816   il 0:0 
     s 5 nf   0  i0 0:0  i1 434816:434816   il 0:0 




Another lack of GBndLib issue in CTestDetectorTest
------------------------------------------------------

* hmm probably can just move to/implement NCSG handling and drop the old commandline config based GGeoTest::createPmtInBox ?


::


    simon:cfg4 blyth$ lldb CTestDetectorTest 
    (lldb) target create "CTestDetectorTest"
    Current executable set to 'CTestDetectorTest' (x86_64).
    (lldb) r
    Process 23661 launched: '/usr/local/opticks/lib/CTestDetectorTest' (x86_64)
    2017-06-15 13:12:29.455 INFO  [7594821] [main@42] CTestDetectorTest
    2017-06-15 13:12:29.623 INFO  [7594821] [*GMergedMesh::load@632] GMergedMesh::load dir /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/0 -> cachedir /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/0 index 0 version (null) existsdir 1
    2017-06-15 13:12:29.736 INFO  [7594821] [*GMergedMesh::load@632] GMergedMesh::load dir /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/1 -> cachedir /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/1 index 1 version (null) existsdir 1
    2017-06-15 13:12:29.824 INFO  [7594821] [GMaterialLib::postLoadFromCache@67] GMaterialLib::postLoadFromCache  nore 0 noab 0 nosc 0 xxre 0 xxab 0 xxsc 0 fxre 0 fxab 0 fxsc 0 groupvel 1
    2017-06-15 13:12:29.824 INFO  [7594821] [GMaterialLib::replaceGROUPVEL@552] GMaterialLib::replaceGROUPVEL  ni 38
    2017-06-15 13:12:29.824 INFO  [7594821] [GPropertyLib::getIndex@338] GPropertyLib::getIndex type GMaterialLib TRIGGERED A CLOSE  shortname [GdDopedLS]
    2017-06-15 13:12:29.824 INFO  [7594821] [GPropertyLib::close@384] GPropertyLib::close type GMaterialLib buf 38,2,39,4
    2017-06-15 13:12:29.829 INFO  [7594821] [GGeo::loadAnalyticPmt@789] GGeo::loadAnalyticPmt AnalyticPMTIndex 0 AnalyticPMTSlice ALL Path /usr/local/opticks/opticksdata/export/DayaBay/GPmt/0
    2017-06-15 13:12:29.838 WARN  [7594821] [GGeoTest::init@54] GGeoTest::init booting from m_ggeo 
    2017-06-15 13:12:29.838 WARN  [7594821] [GMaker::init@169] GMaker::init booting from cache
    2017-06-15 13:12:29.838 INFO  [7594821] [*GMergedMesh::load@632] GMergedMesh::load dir /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/0 -> cachedir /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/0 index 0 version (null) existsdir 1
    2017-06-15 13:12:29.941 INFO  [7594821] [*GMergedMesh::load@632] GMergedMesh::load dir /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/1 -> cachedir /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/1 index 1 version (null) existsdir 1
    2017-06-15 13:12:29.945 INFO  [7594821] [GMaterialLib::postLoadFromCache@67] GMaterialLib::postLoadFromCache  nore 0 noab 0 nosc 0 xxre 0 xxab 0 xxsc 0 fxre 0 fxab 0 fxsc 0 groupvel 1
    2017-06-15 13:12:29.945 INFO  [7594821] [GMaterialLib::replaceGROUPVEL@552] GMaterialLib::replaceGROUPVEL  ni 38
    2017-06-15 13:12:29.945 INFO  [7594821] [GPropertyLib::getIndex@338] GPropertyLib::getIndex type GMaterialLib TRIGGERED A CLOSE  shortname [GdDopedLS]
    2017-06-15 13:12:29.946 INFO  [7594821] [GPropertyLib::close@384] GPropertyLib::close type GMaterialLib buf 38,2,39,4
    2017-06-15 13:12:29.949 INFO  [7594821] [GPropertyLib::getIndex@338] GPropertyLib::getIndex type GSurfaceLib TRIGGERED A CLOSE  shortname [NONE]
    2017-06-15 13:12:29.950 INFO  [7594821] [GPropertyLib::close@384] GPropertyLib::close type GSurfaceLib buf 48,2,39,4
    2017-06-15 13:12:29.950 INFO  [7594821] [*GGeoTest::createPmtInBox@152] GGeoTest::createPmtInBox  type 6 csgName box spec Rock/NONE/perfectAbsorbSurface/MineralOil container_inner_material MineralOil param 0.0000,0.0000,0.0000,300.0000
    2017-06-15 13:12:29.950 INFO  [7594821] [*GMergedMesh::load@632] GMergedMesh::load dir $OPTICKSINSTALLPREFIX/opticksdata/export/dpib/GMergedMesh/0 -> cachedir /usr/local/opticks/opticksdata/export/dpib/GMergedMesh/0 index 0 version (null) existsdir 1
    2017-06-15 13:12:29.951 INFO  [7594821] [GPropertyLib::getIndex@338] GPropertyLib::getIndex type GSurfaceLib TRIGGERED A CLOSE  shortname [NONE]
    2017-06-15 13:12:29.952 INFO  [7594821] [GPropertyLib::close@384] GPropertyLib::close type GSurfaceLib buf 48,2,39,4
    2017-06-15 13:12:29.952 INFO  [7594821] [*GMergedMesh::combine@122] GMergedMesh::combine making new mesh  index 0 solids 1 verbosity 1
    2017-06-15 13:12:29.952 INFO  [7594821] [GSolid::Dump@199] GMergedMesh::combine (source solids) numSolid 1
    2017-06-15 13:12:29.952 INFO  [7594821] [GNode::dump@196] mesh.numSolids 0 mesh.ce.0 gfloat4      0.000      0.000      0.000    300.000 
    2017-06-15 13:12:29.952 FATAL [7594821] [GMergedMesh::mergeSolidIdentity@482] GMergedMesh::mergeSolid mismatch  nodeIndex 0 m_cur_solid 6
    2017-06-15 13:12:29.952 INFO  [7594821] [GMergedMesh::dumpSolids@659] GMergedMesh::combine (combined result)  ce0 gfloat4      0.000      0.000      0.000    300.000 
        0 ce             gfloat4      0.000      0.000      0.000    300.000  bb bb min   -300.000   -300.000   -300.000  max    300.000    300.000    300.000 
        1 ce             gfloat4      0.000      0.000    -18.997    149.997  bb bb min   -100.288   -100.288   -168.995  max    100.288    100.288    131.000 
        2 ce             gfloat4      0.000      0.000    -18.247    146.247  bb bb min    -97.288    -97.288   -164.495  max     97.288     97.288    128.000 
        3 ce             gfloat4      0.005      0.004     91.998     98.143  bb bb min    -98.138    -98.139     55.996  max     98.148     98.147    128.000 
        4 ce             gfloat4      0.000      0.000     13.066     98.143  bb bb min    -98.143    -98.143    -30.000  max     98.143     98.143     56.131 
        5 ce             gfloat4      0.000      0.000    -81.500     83.000  bb bb min    -27.500    -27.500   -164.500  max     27.500     27.500      1.500 
        6 ce             gfloat4      0.000      0.000      0.000    300.000  bb bb min   -300.000   -300.000   -300.000  max    300.000    300.000    300.000 
        0 ni[nf/nv/nidx/pidx] (  0,  0,  0,4294967295)  id[nidx,midx,bidx,sidx]  (  0,  5,  0,  0) 
        1 ni[nf/nv/nidx/pidx] (720,362,  1,  0)  id[nidx,midx,bidx,sidx]  (  1,  4,  1,  0) 
        2 ni[nf/nv/nidx/pidx] (720,362,  2,  1)  id[nidx,midx,bidx,sidx]  (  2,  3,  2,  0) 
        3 ni[nf/nv/nidx/pidx] (960,482,  3,  2)  id[nidx,midx,bidx,sidx]  (  3,  0,  3,  0) 
        4 ni[nf/nv/nidx/pidx] (576,288,  4,  2)  id[nidx,midx,bidx,sidx]  (  4,  1,  4,  0) 
        5 ni[nf/nv/nidx/pidx] ( 96, 50,  5,  2)  id[nidx,midx,bidx,sidx]  (  5,  2,  4,  0) 
        6 ni[nf/nv/nidx/pidx] ( 12, 24,  0,4294967295)  id[nidx,midx,bidx,sidx]  (  0,1000,  0,  0) 
    Assertion failed: (m_bndlib), function registerBoundaries, file /Users/blyth/opticks/ggeo/GParts.cc, line 614.
    Process 23661 stopped
    * thread #1: tid = 0x73e345, 0x00007fff8f018866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff8f018866 libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill + 10:
    -> 0x7fff8f018866:  jae    0x7fff8f018870            ; __pthread_kill + 20
       0x7fff8f018868:  movq   %rax, %rdi
       0x7fff8f01886b:  jmp    0x7fff8f015175            ; cerror_nocancel
       0x7fff8f018870:  retq   
    (lldb) bt
    * thread #1: tid = 0x73e345, 0x00007fff8f018866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff8f018866 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff866b535c libsystem_pthread.dylib`pthread_kill + 92
        frame #2: 0x00007fff8d405b1a libsystem_c.dylib`abort + 125
        frame #3: 0x00007fff8d3cf9bf libsystem_c.dylib`__assert_rtn + 321
        frame #4: 0x0000000100d342a0 libGGeo.dylib`GParts::registerBoundaries(this=0x000000010b5f5d20) + 96 at GParts.cc:614
        frame #5: 0x0000000100d34219 libGGeo.dylib`GParts::close(this=0x000000010b5f5d20) + 25 at GParts.cc:607
        frame #6: 0x0000000100d5fbb8 libGGeo.dylib`GGeoTest::createPmtInBox(this=0x000000010b54e1f0) + 1368 at GGeoTest.cc:187
        frame #7: 0x0000000100d5f25e libGGeo.dylib`GGeoTest::create(this=0x000000010b54e1f0) + 126 at GGeoTest.cc:109
        frame #8: 0x0000000100d5f13d libGGeo.dylib`GGeoTest::modifyGeometry(this=0x000000010b54e1f0) + 157 at GGeoTest.cc:81
        frame #9: 0x0000000100d841fc libGGeo.dylib`GGeo::modifyGeometry(this=0x0000000107c11570, config=0x0000000000000000) + 668 at GGeo.cc:819
        frame #10: 0x00000001010f6844 libOpticksGeometry.dylib`OpticksGeometry::modifyGeometry(this=0x0000000107c12740) + 868 at OpticksGeometry.cc:263
        frame #11: 0x00000001010f5d8c libOpticksGeometry.dylib`OpticksGeometry::loadGeometry(this=0x0000000107c12740) + 572 at OpticksGeometry.cc:200
        frame #12: 0x00000001010f9e69 libOpticksGeometry.dylib`OpticksHub::loadGeometry(this=0x00007fff5fbfeae0) + 409 at OpticksHub.cc:243
        frame #13: 0x00000001010f8ffd libOpticksGeometry.dylib`OpticksHub::init(this=0x00007fff5fbfeae0) + 77 at OpticksHub.cc:94
        frame #14: 0x00000001010f8f00 libOpticksGeometry.dylib`OpticksHub::OpticksHub(this=0x00007fff5fbfeae0, ok=0x00007fff5fbfeb50) + 416 at OpticksHub.cc:81
        frame #15: 0x00000001010f90dd libOpticksGeometry.dylib`OpticksHub::OpticksHub(this=0x00007fff5fbfeae0, ok=0x00007fff5fbfeb50) + 29 at OpticksHub.cc:83
        frame #16: 0x000000010000d026 CTestDetectorTest`main(argc=1, argv=0x00007fff5fbfee58) + 950 at CTestDetectorTest.cc:48
        frame #17: 0x00007fff8a48b5fd libdyld.dylib`start + 1
        frame #18: 0x00007fff8a48b5fd libdyld.dylib`start + 1
    (lldb) f 4
    frame #4: 0x0000000100d342a0 libGGeo.dylib`GParts::registerBoundaries(this=0x000000010b5f5d20) + 96 at GParts.cc:614
       611  
       612  void GParts::registerBoundaries()
       613  {
    -> 614     assert(m_bndlib); 
       615     unsigned int nbnd = m_bndspec->getNumKeys() ; 
       616     assert( getNumParts() == nbnd );
       617     for(unsigned int i=0 ; i < nbnd ; i++)
    (lldb) 




