opticks-t-2021-aug-15-4-of-489-fails
=======================================

macOS::

    SLOW: tests taking longer that 15 seconds
      2  /6   Test #2  : OKOPTest.OpSeederTest                         Passed                         20.79  
      5  /6   Test #5  : OKOPTest.OpSnapTest                           Passed                         19.74  
      6  /6   Test #6  : OKOPTest.OpFlightPathTest                     Passed                         32.96  
      2  /5   Test #2  : OKTest.OKTest                                 Passed                         70.34  
      3  /5   Test #3  : OKTest.OTracerTest                            Passed                         16.70  
      31 /31  Test #31 : ExtG4Test.X4SurfaceTest                       Passed                         42.52  
      3  /46  Test #3  : CFG4Test.CTestDetectorTest                    Passed                         22.70  
      5  /46  Test #5  : CFG4Test.CGDMLDetectorTest                    Passed                         23.23  
      7  /46  Test #7  : CFG4Test.CGeometryTest                        Passed                         22.72  
      8  /46  Test #8  : CFG4Test.CG4Test                              Child aborted***Exception:     381.15 
      28 /46  Test #28 : CFG4Test.CInterpolationTest                   Passed                         24.49  
      1  /1   Test #1  : OKG4Test.OKG4Test                             Child aborted***Exception:     402.85 
      1  /2   Test #1  : G4OKTest.G4OKTest                             Passed                         51.27  


    FAILS:  4   / 489   :  Sun Aug 15 14:24:29 2021   
      25 /35  Test #25 : OptiXRapTest.interpolationTest                ***Failed                      11.73  
      8  /46  Test #8  : CFG4Test.CG4Test                              Child aborted***Exception:     381.15 
      1  /1   Test #1  : OKG4Test.OKG4Test                             Child aborted***Exception:     402.85 
      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      6.53   
    epsilon:opticks blyth$ 




OptiXRapTest.interpolationTest : unclear why a python load fails
--------------------------------------------------------------------


::

    97 2021-08-15 14:04:35.590 INFO  [6316496] [OLaunchTest::launch@80] OLaunchTest entry   0 width     761 height      36 ptx                               interpolationTest.cu prog                                      interpolationTest
     98 2021-08-15 14:04:39.842 INFO  [6316496] [interpolationTest::launch@158] OLaunchTest entry   0 width     761 height      36 ptx                               interpolationTest.cu prog                                      interpolationTest
     99 2021-08-15 14:04:39.853 INFO  [6316496] [interpolationTest::launch@165]  save  base $TMP/optixrap/interpolationTest name interpolationTest_interpol.npy
    100 2021-08-15 14:04:39.866 INFO  [6316496] [SSys::RunPythonScript@625]  script interpolationTest_interpol.py script_path /usr/local/opticks/bin/interpolationTest_interpol.py python_executabl    e /Users/blyth/miniconda3/bin/python
    101 [{__init__            :proplib.py:152} INFO     - ^[[32mnames : $TMP/interpolationTest/GItemList/GBndLib.txt ^[[0m
    102 [{__init__            :proplib.py:162} INFO     - ^[[32mnpath : /tmp/blyth/opticks/interpolationTest/GItemList/GBndLib.txt ^[[0m
    103 Traceback (most recent call last):
    104   File "/usr/local/opticks/bin/interpolationTest_interpol.py", line 36, in <module>
    105     blib = PropLib.load_GBndLib(base)
    106   File "/Users/blyth/opticks/ana/proplib.py", line 127, in load_GBndLib
    107     blib = cls("GBndLib", data=t, names=os.path.join(base,"GItemList/GBndLib.txt"), optical=o )
    108   File "/Users/blyth/opticks/ana/proplib.py", line 167, in __init__
    109     names = list(map(lambda _:_[:-1],open(npath,"r").readlines()))
    110 FileNotFoundError: [Errno 2] No such file or directory: '/tmp/blyth/opticks/interpolationTest/GItemList/GBndLib.txt'
    111 2021-08-15 14:04:40.034 INFO  [6316496] [SSys::run@100] /Users/blyth/miniconda3/bin/python /usr/local/opticks/bin/interpolationTest_interpol.py  rc_raw : 256 rc : 1
    112 2021-08-15 14:04:40.034 ERROR [6316496] [SSys::run@107] FAILED with  cmd /Users/blyth/miniconda3/bin/python /usr/local/opticks/bin/interpolationTest_interpol.py  RC 1
    113 2021-08-15 14:04:40.034 INFO  [6316496] [SSys::RunPythonScript@632]  RC 1
    114 


CG4Test and OKG4Test : old torch genstep running needs to follow initialization pattern of live gensteps
----------------------------------------------------------------------------------------------------------

Same fail from CG4Test and OKG4Test. The old torch gensteps used for the test 
are somehow causing the CCtx::initEvent not to get called::

    102 unsigned CCtx::step_limit() const
    103 {
    104     assert( _ok_event_init );
    105     return 1 + 2*( _steps_per_photon > _bounce_max ? _steps_per_photon : _bounce_max ) ;
    106 }
    107 


    205 /**
    206 CCtx::initEvent
    207 --------------------
    208 
    209 Collect the parameters of the OpticksEvent which 
    210 dictate what needs to be collected.
    211 
    212 **/
    213 
    214 void CCtx::initEvent(const OpticksEvent* evt)
    215 {
    216     _ok_event_init = true ;
    217     _photons_per_g4event = evt->getNumPhotonsPerG4Event() ;
    218     _steps_per_photon = evt->getMaxRec() ;   // number of points to be recorded into record buffer   
    219     _record_max = evt->getNumPhotons();      // from the genstep summation, hmm with dynamic running this will start as zero 
    220 
    221     _bounce_max = evt->getBounceMax();       // maximum bounce allowed before truncation will often be 1 less than _steps_per_photon but need not be 
    222     unsigned bounce_max_2 = evt->getMaxBounce();
    223     assert( _bounce_max == bounce_max_2 ) ; // TODO: eliminate or rename one of those
    224 
    225     const char* typ = evt->getTyp();
    226 




::

    069 2021-08-15 14:10:07.532 FATAL [6323538] [CTorchSource::configure@166] CTorchSource::configure _t 0.1 _radius 0 _pos 0.0000,0.0000,0.0000 _dir 0.0000,0.0000,1.0000 _zeaz 0.0000,1.0000,0.00    00,1.0000 _pol 0.0000,0.0000,1.0000
     70 
     71   C4FPEDetection::InvalidOperationDetection_Disable       NOT IMPLEMENTED
     72     0                      Steel_surface         ChimneySteelOpticalSurface lv lLowerChimneySteel0x4ccc9a0
     73     0           UpperChimneyTyvekSurface    UpperChimneyTyvekOpticalSurface pv1 pUpperChimneyLS0x4cca450 #0 pv2 pUpperChimneyTyvek0x4cca5f0 #0
     74     1   NNVTMCPPMT_photocathode_logsurf1     NNVTMCPPMT_Photocathode_opsurf pv1 NNVTMCPPMT_inner1_phys0x3a939d0 #0 pv2 NNVTMCPPMT_body_phys0x3a93950 #0
     75     2         NNVTMCPPMT_mirror_logsurf1           NNVTMCPPMT_Mirror_opsurf pv1 NNVTMCPPMT_inner2_phys0x3a93a80 #0 pv2 NNVTMCPPMT_body_phys0x3a93950 #0
     76     3   NNVTMCPPMT_photocathode_logsurf2     NNVTMCPPMT_Photocathode_opsurf pv1 NNVTMCPPMT_body_phys0x3a93950 #0 pv2 NNVTMCPPMT_inner1_phys0x3a939d0 #0
     77     4HamamatsuR12860_photocathode_logsurf1HamamatsuR12860_Photocathode_opsurf pv1 HamamatsuR12860_inner1_phys0x3aa1230 #0 pv2 HamamatsuR12860_body_phys0x3aa11b0 #0
     78     5    HamamatsuR12860_mirror_logsurf1      HamamatsuR12860_Mirror_opsurf pv1 HamamatsuR12860_inner2_phys0x3aa12e0 #0 pv2 HamamatsuR12860_body_phys0x3aa11b0 #0
     79     6HamamatsuR12860_photocathode_logsurf2HamamatsuR12860_Photocathode_opsurf pv1 HamamatsuR12860_body_phys0x3aa11b0 #0 pv2 HamamatsuR12860_inner1_phys0x3aa1230 #0
     80     7    PMT_3inch_photocathode_logsurf1          Photocathode_opsurf_3inch pv1 PMT_3inch_inner1_phys0x421f2d0 #0 pv2 PMT_3inch_body_phys0x421f250 #0
     81     8          PMT_3inch_absorb_logsurf1                      Absorb_opsurf pv1 PMT_3inch_inner2_phys0x421f380 #0 pv2 PMT_3inch_body_phys0x421f250 #0
     82     9    PMT_3inch_photocathode_logsurf2          Photocathode_opsurf_3inch pv1 PMT_3inch_body_phys0x421f250 #0 pv2 PMT_3inch_inner1_phys0x421f2d0 #0
     83    10          PMT_3inch_absorb_logsurf3                      Absorb_opsurf pv1 PMT_3inch_cntr_phys0x421f430 #0 pv2 PMT_3inch_body_phys0x421f250 #0
     84    11PMT_20inch_veto_photocathode_logsurf1                Photocathode_opsurf pv1 PMT_20inch_veto_inner1_phys0x3a8d550 #0 pv2 PMT_20inch_veto_body_phys0x3a8d4d0 #0
     85    12    PMT_20inch_veto_mirror_logsurf1                      Mirror_opsurf pv1 PMT_20inch_veto_inner2_phys0x3a8d600 #0 pv2 PMT_20inch_veto_body_phys0x3a8d4d0 #0
     86    13PMT_20inch_veto_photocathode_logsurf2                Photocathode_opsurf pv1 PMT_20inch_veto_body_phys0x3a8d4d0 #0 pv2 PMT_20inch_veto_inner1_phys0x3a8d550 #0
     87    14                     CDTyvekSurface              CDTyvekOpticalSurface pv1 pOuterWaterPool0x33574c0 #0 pv2 pCentralDetector0x3359290 #0
     88 2021-08-15 14:10:07.584 WARN  [6323538] [main@52]  post CG4
     89 2021-08-15 14:10:07.584 WARN  [6323538] [main@56]   post CG4::interactive
     90 2021-08-15 14:10:07.584 ERROR [6323538] [main@63]  setting gensteps 0x7fcb6d540120 numPhotons 20000
     91 2021-08-15 14:10:07.585 INFO  [6323538] [*OpticksRun::createOKEvent@158]  tagoffset 0 skipaheadstep 0 skipahead 0
     92 2021-08-15 14:10:07.585 INFO  [6323538] [main@68]  cgs T  idx   0 pho20000 off      0
     93 2021-08-15 14:10:07.607 INFO  [6323538] [*CG4::propagate@396]  calling BeamOn numG4Evt 1
     94 2021-08-15 14:16:05.502 INFO  [6323538] [CScint::Check@16]  pmanager 0x7fcb6f04e890 proc 0x0
     95 2021-08-15 14:16:05.503 INFO  [6323538] [CScint::Check@21] CProMgr n:[4] (0) name Transportation left -1 (1) name OpAbsorption left -1 (2) name OpRayleigh left -1 (3) name OpBoundary left     -1
     96 2021-08-15 14:16:05.503 INFO  [6323538] [CTorchSource::GeneratePrimaryVertex@293]  event_gencode 6 : BAD_FLAG
     97 Assertion failed: (_ok_event_init), function step_limit, file /Users/blyth/opticks/cfg4/CCtx.cc, line 104.
     98 
     99       Start  9: CFG4Test.G4MaterialTest
    100  9/46 Test  #9: CFG4Test.G4MaterialTest ...................   Passed    0.10 sec
    101       Start 10: CFG4Test.G4StringTest
    102 10/46 Test #10: CFG4Test.G4StringTest .....................   Passed    0.10 sec
    103       Start 11: CFG4Test.G4SphereTest


    126 2021-08-15 14:17:24.028 INFO  [6381838] [*CG4::propagate@396]  calling BeamOn numG4Evt 1
    127 2021-08-15 14:23:27.569 INFO  [6381838] [CScint::Check@16]  pmanager 0x7f904281f940 proc 0x0
    128 2021-08-15 14:23:27.569 INFO  [6381838] [CScint::Check@21] CProMgr n:[4] (0) name Transportation left -1 (1) name OpAbsorption left -1 (2) name OpRayleigh left -1 (3) name OpBoundary left     -1
    129 2021-08-15 14:23:27.569 INFO  [6381838] [CTorchSource::GeneratePrimaryVertex@293]  event_gencode 6 : BAD_FLAG
    130 Assertion failed: (_ok_event_init), function step_limit, file /Users/blyth/opticks/cfg4/CCtx.cc, line 104.
    131 
    132 
    133 0% tests passed, 1 tests failed out of 1
    134 








IntegrationTests.tboolean.box : failing as the code is expecting double precision property and only float in opticksaux
----------------------------------------------------------------------------------------------------------------------------------


::


    83 2021-08-15 14:24:28.363 INFO  [6399332] [OpticksHub::setupTestGeometry@358] --test modifying geometry
     84 2021-08-15 14:24:28.367 ERROR [6399332] [*NPY<double>::load@1093] NPY<T>::load failed for path [/usr/local/opticks/opticksaux/refractiveindex/tmp/glass/schott/F2.npy] use debugload with N    PYLoadTest to investigate (problems are usually from dtype mismatches)
     85 2021-08-15 14:24:28.368 ERROR [6399332] [*GProperty<double>::load@122] GProperty<T>::load FAILED for path $OPTICKS_INSTALL_PREFIX/opticksaux/refractiveindex/tmp/glass/schott/F2.npy
     86 2021-08-15 14:24:28.369 ERROR [6399332] [*NPY<double>::load@1093] NPY<T>::load failed for path [/usr/local/opticks/opticksaux/refractiveindex/tmp/main/H2O/Hale.npy] use debugload with NPY    LoadTest to investigate (problems are usually from dtype mismatches)
     87 2021-08-15 14:24:28.369 ERROR [6399332] [*GProperty<double>::load@122] GProperty<T>::load FAILED for path $OPTICKS_INSTALL_PREFIX/opticksaux/refractiveindex/tmp/main/H2O/Hale.npy
     88 2021-08-15 14:24:28.369 FATAL [6399332] [GMaterialLib::reuseBasisMaterial@1124] reuseBasisMaterial requires basis library to be present and to contain the material  GlassSchottF2
     89 Assertion failed: (mat), function reuseBasisMaterial, file /Users/blyth/opticks/ggeo/GMaterialLib.cc, line 1125.
     90 /Users/blyth/opticks/bin/o.sh: line 362: 62523 Abort trap: 6           /usr/local/opticks/lib/OKG4Test --okg4test --align --dbgskipclearzero --dbgnojumpzero --dbgkludgeflatzero --profile     --generateoverride 10000 --envkey --rendermode +global,+axis --geocenter --stack 2180 --eye 1,0,0 --up 0,0,1 --test --testconfig mode=PyCsgInBox_analytic=1_name=tboolean-box_csgpath=/tmp/    blyth/opticks/tboolean-box_outerfirst=1_autocontainer=Rock//perfectAbsorbSurface/Vacuum_autoobject=Vacuum/perfectSpecularSurface//GlassSchottF2_autoemitconfig=photons:600000,wavelength:38    0,time:0.2,posdelta:0.1,sheetmask:0x1,umin:0.45,umax:0.5


