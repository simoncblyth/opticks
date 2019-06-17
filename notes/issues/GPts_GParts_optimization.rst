GPts_GParts_optimization
============================

Context
----------

* :doc:`GPtsTest`
* :doc:`x016`


Now that have capability to `GParts::Create(GPts*)` in a deferred way.
Can that be used to reduce the memory and time consumption of 
the very heavy node tree traverse ? For example *geocache-recreate* 

* potentially postcache consumer is OGeo (which currently only has GGeoLib with its GMergedMesh), 
  but do not want to add much code there 



Issue : geocache-recreate takes lots of memory+time
--------------------------------------------------------------

* too much memory to run on lxslc

::


    geocache-recreate () 
    { 
        geocache-j1808-v4 $*
    }

    geocache-j1808-v4 () 
    { 
        geocache-j1808-v4- --csgskiplv 22 --runfolder $FUNCNAME --runcomment $(${FUNCNAME}-comment) $*
    }

    geocache-j1808-v4- () 
    { 
        opticksdata-;
        geocache-create- --gdmlpath $(opticksdata-jv4) $*
    }

    geocache-create- () 
    { 
        local iwd=$PWD;
        local tmp=$(geocache-tmp $FUNCNAME);
        mkdir -p $tmp && cd_func $tmp;
        o.sh --okx4 --g4codegen --deletegeocache $*;
        cd_func $iwd
    }





::

    2019-06-16 22:06:40.608 INFO  [37908] [X4PhysicalVolume::convertStructure@722] [ creating large tree of GVolume instances
    2019-06-16 22:07:07.457 INFO  [37908] [X4PhysicalVolume::convertStructure@742] ] tree contains GGeo::getNumVolumes() 366697


    2019-06-17 09:51:00.893 INFO  [235645] [X4PhysicalVolume::convertStructure@725] [ creating large tree of GVolume instances
    2019-06-17 09:51:21.859 INFO  [235645] [X4PhysicalVolume::convertStructure@745] ] tree contains GGeo::getNumVolumes() 366697

    ## deferring GParts only saves ~6s 




Before any change::

    2019-06-16 22:08:24.025 INFO  [37908] [BTimesTable::dump@145] Opticks::postgeocache filter: NONE
     diffListedTime           Time      DeltaTime             VM        DeltaVM
              0.000           0.000      50799.457          0.000        484.320 : OpticksRun::OpticksRun_1139392501
              0.000           0.000          0.000          0.000          0.000 : Opticks::Opticks_0
              0.043           0.043          0.043        103.880        103.880 : _OKX4Test:GGeo_0
              0.008           0.051          0.008        103.880          0.000 : OKX4Test:GGeo_0
              0.000           0.051          0.000        103.880          0.000 : _OKX4Test:X4PhysicalVolume_0
              0.000           0.051          0.000        103.880          0.000 : _X4PhysicalVolume::convertMaterials_0
              0.000           0.051          0.000        104.012          0.132 : X4PhysicalVolume::convertMaterials_0
              0.051           0.102          0.051        104.276          0.264 : _X4PhysicalVolume::convertSolids_0
              1.051           1.152          1.051        116.792         12.516 : X4PhysicalVolume::convertSolids_0
              0.000           1.152          0.000        116.792          0.000 : _X4PhysicalVolume::convertStructure_0
             26.848          28.000         26.848       3732.160       3615.368 : X4PhysicalVolume::convertStructure_0
              0.000          28.000          0.000       3732.160          0.000 : OKX4Test:X4PhysicalVolume_0
              0.000          28.000          0.000       3732.160          0.000 : GInstancer::createInstancedMergedMeshes_0
              1.762          29.762          1.762       3920.524        188.364 : GInstancer::createInstancedMergedMeshes:deltacheck_0
             13.023          42.785         13.023       4290.020        369.496 : GInstancer::createInstancedMergedMeshes:traverse_0
              0.527          43.312          0.527       4290.020          0.000 : GInstancer::createInstancedMergedMeshes:labelTree_0
              0.000          43.312          0.000       4290.020          0.000 : _GMergedMesh::Create_0
              0.117          43.430          0.117       4290.020          0.000 : GMergedMesh::Create::Count_0
              0.000          43.430          0.000       4290.020          0.000 : _GMergedMesh::Create::Allocate_0
              0.020          43.449          0.020       4339.780         49.760 : GMergedMesh::Create::Allocate_0
             13.625          57.074         13.625       5958.864       1619.084 : GMergedMesh::Create::Merge_0
              0.000          57.074          0.000       5958.996          0.132 : GMergedMesh::Create::Bounds_0

              ... elide the quick instanced meshes ...

              0.000          57.426          0.000       5968.196          0.000 : GMergedMesh::Create::Bounds_0
              0.078          58.379          0.078       5987.752          0.000 : GInstancer::createInstancedMergedMeshes:makeMergedMeshAndInstancedBuffers_0
              0.121          58.500          0.121       5987.752          0.000 : _OKX4Test:OKMgr_0
              4.352          62.852          4.352      15228.775       9241.023 : OKX4Test:OKMgr_0


Moving to deferred GParts creation, shaves ~5s and 1.3G from X4PhysicalVolume::convertStructure::

     diffListedTime           Time      DeltaTime             VM        DeltaVM
              0.000           0.000      16725.602          0.000        484.312 : OpticksRun::OpticksRun_1139392501
              0.000           0.000          0.000          0.000          0.000 : Opticks::Opticks_0
              0.012           0.012          0.012        103.880        103.880 : _OKX4Test:GGeo_0
              0.006           0.018          0.006        103.880          0.000 : OKX4Test:GGeo_0
              0.000           0.018          0.000        103.880          0.000 : _OKX4Test:X4PhysicalVolume_0
              0.000           0.018          0.000        103.880          0.000 : _X4PhysicalVolume::convertMaterials_0
              0.002           0.020          0.002        104.012          0.132 : X4PhysicalVolume::convertMaterials_0
              0.057           0.076          0.057        104.276          0.264 : _X4PhysicalVolume::convertSolids_0
              1.037           1.113          1.037        116.792         12.516 : X4PhysicalVolume::convertSolids_0
              0.000           1.113          0.000        116.792          0.000 : _X4PhysicalVolume::convertStructure_0
             21.137          22.250         21.137       2442.088       2325.296 : X4PhysicalVolume::convertStructure_0
              0.000          22.250          0.000       2442.088          0.000 : OKX4Test:X4PhysicalVolume_0
              0.002          22.252          0.002       2442.088          0.000 : GInstancer::createInstancedMergedMeshes_0
              1.678          23.930          1.678       2630.452        188.364 : GInstancer::createInstancedMergedMeshes:deltacheck_0
             12.887          36.816         12.887       2999.904        369.452 : GInstancer::createInstancedMergedMeshes:traverse_0
              0.514          37.330          0.514       2999.904          0.000 : GInstancer::createInstancedMergedMeshes:labelTree_0
              0.000          37.330          0.000       2999.904          0.000 : _GMergedMesh::Create_0
              0.109          37.439          0.109       2999.904          0.000 : GMergedMesh::Create::Count_0
              0.000          37.439          0.000       2999.904          0.000 : _GMergedMesh::Create::Allocate_0
              0.021          37.461          0.021       3049.668         49.764 : GMergedMesh::Create::Allocate_0
             13.561          51.021         13.561       4668.632       1618.964 : GMergedMesh::Create::Merge_0
              0.002          51.023          0.002       4668.796          0.164 : GMergedMesh::Create::Bounds_0
             ....
              0.000          52.254          0.000       4697.472          0.000 : GMergedMesh::Create::Bounds_0
              0.076          52.330          0.076       4697.472          0.000 : GInstancer::createInstancedMergedMeshes:makeMergedMeshAndInstancedBuffers_0
              0.119          52.449          0.119       4697.472          0.000 : _OKX4Test:OKMgr_0
              4.029          56.479          4.029      13938.203       9240.731 : OKX4Test:OKMgr_0



Where to do the deferred GParts creation ?
-----------------------------------------------

* GGeo seems natural as it involves more than one lib

1. GMeshLib to get at the NCSG solids
2. GGeoLib for the GMergedMesh 
3. GBndLib


* appropriate "could be postcache" juncture in GGeo ?

* actually better to implement in GGeo but drive it from higher level, 
  perhaps OpticksHub::init




Lack of GParts on the volume is felt first at GGeo::prepare ... GMergedMesh::mergeVolumeAnalytic
----------------------------------------------------------------------------------------------------

::

    (gdb) bt
    #0  0x00007fffe200c207 in raise () from /lib64/libc.so.6
    #1  0x00007fffe200d8f8 in abort () from /lib64/libc.so.6
    #2  0x00007fffe2005026 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007fffe20050d2 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007fffe5ce7c0b in GMergedMesh::mergeVolumeAnalytic (this=0xb0df14a0, parts=0x0, transform=0x33740a0, verbosity=0) at /home/blyth/opticks/ggeo/GMergedMesh.cc:763
    #5  0x00007fffe5ce6377 in GMergedMesh::mergeVolume (this=0xb0df14a0, volume=0x3374120, selected=true, verbosity=0) at /home/blyth/opticks/ggeo/GMergedMesh.cc:520
    #6  0x00007fffe5ce5828 in GMergedMesh::traverse_r (this=0xb0df14a0, node=0x3374120, depth=0, pass=1, verbosity=0) at /home/blyth/opticks/ggeo/GMergedMesh.cc:337
    #7  0x00007fffe5ce5101 in GMergedMesh::Create (ridx=0, base=0x0, root=0x3374120, verbosity=0) at /home/blyth/opticks/ggeo/GMergedMesh.cc:265
    #8  0x00007fffe5cc6a73 in GGeoLib::makeMergedMesh (this=0x26bf080, index=0, base=0x0, root=0x3374120, verbosity=0) at /home/blyth/opticks/ggeo/GGeoLib.cc:276
    #9  0x00007fffe5cdb174 in GInstancer::makeMergedMeshAndInstancedBuffers (this=0x26bfd70, verbosity=0) at /home/blyth/opticks/ggeo/GInstancer.cc:589
    #10 0x00007fffe5cd8eed in GInstancer::createInstancedMergedMeshes (this=0x26bfd70, delta=true, verbosity=0) at /home/blyth/opticks/ggeo/GInstancer.cc:99
    #11 0x00007fffe5cf2247 in GGeo::prepareVolumes (this=0x26b8180) at /home/blyth/opticks/ggeo/GGeo.cc:1273
    #12 0x00007fffe5cef32f in GGeo::prepare (this=0x26b8180) at /home/blyth/opticks/ggeo/GGeo.cc:683
    #13 0x00000000004052c3 in main (argc=13, argv=0x7fffffffd748) at /home/blyth/opticks/okg4/tests/OKX4Test.cc:138
    (gdb) 


Deferring that, next felt at OKMgr::OKMgr ... OGeo::makeAnalyticGeometry  which could be postcache 
-------------------------------------------------------------------------------------------------------

::

    (gdb) bt
    #0  0x00007fffe200d207 in raise () from /lib64/libc.so.6
    #1  0x00007fffe200e8f8 in abort () from /lib64/libc.so.6
    #2  0x00007fffe2006026 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007fffe20060d2 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007ffff6556a6b in OGeo::makeAnalyticGeometry (this=0x116b41370, mm=0xb0df14a0, lod=0) at /home/blyth/opticks/optixrap/OGeo.cc:615
    #5  0x00007ffff6556543 in OGeo::makeOGeometry (this=0x116b41370, mergedmesh=0xb0df14a0, lod=0) at /home/blyth/opticks/optixrap/OGeo.cc:551
    #6  0x00007ffff6554aa2 in OGeo::makeGlobalGeometryGroup (this=0x116b41370, mm=0xb0df14a0) at /home/blyth/opticks/optixrap/OGeo.cc:277
    #7  0x00007ffff65546f9 in OGeo::convertMergedMesh (this=0x116b41370, i=0) at /home/blyth/opticks/optixrap/OGeo.cc:256
    #8  0x00007ffff655421b in OGeo::convert (this=0x116b41370) at /home/blyth/opticks/optixrap/OGeo.cc:223
    #9  0x00007ffff654c6e9 in OScene::init (this=0x115e42640) at /home/blyth/opticks/optixrap/OScene.cc:144
    #10 0x00007ffff654bfc0 in OScene::OScene (this=0x115e42640, hub=0x1146bdc80, cmake_target=0x7ffff6900434 "OptiXRap", ptxrel=0x0) at /home/blyth/opticks/optixrap/OScene.cc:66
    #11 0x00007ffff68a41ba in OpEngine::OpEngine (this=0x115e429c0, hub=0x1146bdc80) at /home/blyth/opticks/okop/OpEngine.cc:48
    #12 0x00007ffff79cc758 in OKPropagator::OKPropagator (this=0x115e42d20, hub=0x1146bdc80, idx=0x1146da980, viz=0x1146da9a0) at /home/blyth/opticks/ok/OKPropagator.cc:41
    #13 0x00007ffff79cb979 in OKMgr::OKMgr (this=0x7fffffffc880, argc=13, argv=0x7fffffffd748, argforced=0x0) at /home/blyth/opticks/ok/OKMgr.cc:41
    #14 0x00000000004052f5 in main (argc=13, argv=0x7fffffffd748) at /home/blyth/opticks/okg4/tests/OKX4Test.cc:143
    (gdb) 







Seems to work, but lots of fails
------------------------------------

Mostly from GPts assert in GGeo::deferredCreateGParts::

    FAILS:
      1  /3   Test #1  : OpticksGeoTest.OpticksGeoTest                 Child aborted***Exception:     0.35   
      4  /24  Test #4  : OptiXRapTest.Roots3And4Test                   Child aborted***Exception:     1.87   
      12 /24  Test #12 : OptiXRapTest.rayleighTest                     Child aborted***Exception:     0.43   
      17 /24  Test #17 : OptiXRapTest.eventTest                        Child aborted***Exception:     0.43   
      18 /24  Test #18 : OptiXRapTest.interpolationTest                Child aborted***Exception:     0.45   
      21 /24  Test #21 : OptiXRapTest.intersectAnalyticTest.iaTorusTest Child aborted***Exception:     2.32   
      1  /5   Test #1  : OKOPTest.OpIndexerTest                        Child aborted***Exception:     0.44   
      2  /5   Test #2  : OKOPTest.OpSeederTest                         Child aborted***Exception:     0.45   
      5  /5   Test #5  : OKOPTest.OpSnapTest                           Child aborted***Exception:     0.44   
      2  /5   Test #2  : OKTest.OKTest                                 Child aborted***Exception:     0.43   
      3  /5   Test #3  : OKTest.OTracerTest                            Child aborted***Exception:     0.44   
      12 /18  Test #12 : ExtG4Test.X4PhysicalVolume2Test               ***Exception: SegFault         0.16   
      1  /34  Test #1  : CFG4Test.CMaterialLibTest                     Child aborted***Exception:     0.40   
      2  /34  Test #2  : CFG4Test.CMaterialTest                        Child aborted***Exception:     0.40   
      3  /34  Test #3  : CFG4Test.CTestDetectorTest                    Child aborted***Exception:     0.40   
      5  /34  Test #5  : CFG4Test.CGDMLDetectorTest                    Child aborted***Exception:     0.38   
      6  /34  Test #6  : CFG4Test.CGeometryTest                        Child aborted***Exception:     0.40   
      7  /34  Test #7  : CFG4Test.CG4Test                              Child aborted***Exception:     0.39   
      22 /34  Test #22 : CFG4Test.CGenstepCollectorTest                Child aborted***Exception:     0.39   
      23 /34  Test #23 : CFG4Test.CInterpolationTest                   Child aborted***Exception:     0.40   
      25 /34  Test #25 : CFG4Test.CGROUPVELTest                        Child aborted***Exception:     0.41   
      29 /34  Test #29 : CFG4Test.CRandomEngineTest                    Child aborted***Exception:     0.39   
      32 /34  Test #32 : CFG4Test.CCerenkovGeneratorTest               Child aborted***Exception:     0.40   
      33 /34  Test #33 : CFG4Test.CGenstepSourceTest                   Child aborted***Exception:     0.42   
      1  /1   Test #1  : OKG4Test.OKG4Test                             Child aborted***Exception:     0.52   



After skipping the assert::

    FAILS:  3   / 405   :  Mon Jun 17 12:58:42 2019   
      4  /24  Test #4  : OptiXRapTest.Roots3And4Test                   Child aborted***Exception:     1.81   
      21 /24  Test #21 : OptiXRapTest.intersectAnalyticTest.iaTorusTest Child aborted***Exception:     2.16   
      12 /18  Test #12 : ExtG4Test.X4PhysicalVolume2Test               ***Exception: SegFault         0.16   


X4PhysicalVolume2Test was expecting GParts on volume for access to NCSG.
After skipping that, down to normal 2.



