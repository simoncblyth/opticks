opticks-t-how-difficult-to-revive-with-new-geometry-workflow
================================================================



Issue 1 : requires OPTICKS_KEY with is an old workflow envvar
----------------------------------------------------------------

::

    epsilon:~ blyth$ echo $OPTICKS_KEY

    epsilon:~ blyth$ 
    epsilon:~ blyth$ 
    epsilon:~ blyth$ opticks-t
    === qudarap-check-installation : /Users/blyth/.opticks/rngcache/RNG rc 0
    === qudarap-check-installation : /Users/blyth/.opticks/rngcache/RNG/QCurandState_1000000_0_0.bin rc 0
    === qudarap-check-installation : /Users/blyth/.opticks/rngcache/RNG/QCurandState_3000000_0_0.bin rc 0
    === opticks-check-installation : rc 0
    === opticks-check-key : OPTICKS_KEY envvar is not defined : read the docs https://simoncblyth.bitbucket.io/opticks/docs/testing.html
    === opticks-t- : ABORT : opticks-check-key failed
    epsilon:~ blyth$ 


Issue 2 : checking $OPTICKS_PREFIX/installcache which is irrelevant in new workflow
---------------------------------------------------------------------------------------

::

    epsilon:issues blyth$ t opticks-t
    opticks-t () 
    { 
        opticks-t- $*
    }
    epsilon:issues blyth$ t opticks-t-
    opticks-t- () 
    { 
        local msg="=== $FUNCNAME : ";
        local iwd=$PWD;
        local rc=0;
        opticks-check-installation;
        rc=$?;
        [ $rc -ne 0 ] && echo $msg ABORT : missing installcache components : create with opticks-prepare-installation && return $rc;
        opticks-check-key;
        rc=$?;
        [ $rc -ne 0 ] && echo $msg ABORT : opticks-check-key failed && return $rc;
        local arg=$1;
        if [ "${arg:0:1}" == "/" -a -d "$arg" ]; then
            bdir=$arg;
            shift;
        else
            bdir=$(opticks-bdir);
        fi;
        cd_func $bdir;
        om-;
        om-test;
        cd_func $iwd
    }
    epsilon:issues blyth$ 


    epsilon:issues blyth$ t opticks-check-installation
    opticks-check-installation () 
    { 
        local msg="=== $FUNCNAME :";
        local rc=0;
        local iwd=$PWD;
        local dir=$(opticks-installcachedir);
        if [ ! -d "$dir" ]; then
            echo $msg missing dir $dir;
            rc=100;
        else
            if [ ! -d "$dir/PTX" ]; then
                echo $msg $dir/PTX : missing PTX : compiled OptiX programs created when building oxrap-;
                rc=101;
            else
                qudarap-;
                qudarap-check-installation;
                rc=$?;
            fi;
        fi;
        cd_func $iwd;
        echo $msg rc $rc;
        return $rc
    }
    epsilon:issues blyth$ 


    epsilon:issues blyth$ opticks-installcachedir
    /usr/local/opticks/installcache
    epsilon:issues blyth$ t opticks-installcachedir
    opticks-installcachedir () 
    { 
        echo $(opticks-prefix)/installcache
    }
    epsilon:issues blyth$ t opticks-prefix
    opticks-prefix () 
    { 
        echo ${OPTICKS_PREFIX:-/usr/local/opticks}
    }
    epsilon:issues blyth$ 

    epsilon:~ blyth$ l /usr/local/opticks/installcache/
    total 0
    0 drwxr-xr-x  39 blyth  staff  1248 Jun 29 19:44 ..
    0 drwxr-xr-x  81 blyth  staff  2592 Jun 27 15:02 PTX 
    0 drwxr-xr-x   7 blyth  staff   224 Jun 22  2019 OKC 
    0 drwxr-xr-x   5 blyth  staff   160 Apr  5  2018 .
    0 drwxr-xr-x   4 blyth  staff   128 Apr  5  2018 RNG 
    epsilon:~ blyth$ 


Looks like the installcache contains old workflow files only::

    epsilon:~ blyth$ find /usr/local/opticks/installcache
    /usr/local/opticks/installcache
    /usr/local/opticks/installcache/OKC
    /usr/local/opticks/installcache/OKC/GFlagIndexLocal.ini
    /usr/local/opticks/installcache/OKC/GFlagIndexSource.ini
    /usr/local/opticks/installcache/OKC/GFlagsLocal.ini
    /usr/local/opticks/installcache/OKC/OpticksFlagsAbbrevMeta.json
    /usr/local/opticks/installcache/OKC/GFlagsSource.ini

    // HMM: the OKC files might be being read somewhere (eg from python anlysis),
    //  if so will need some updates

    /usr/local/opticks/installcache/PTX
    /usr/local/opticks/installcache/PTX/UseOptiXProgramPP_generated_basicTest.cu.ptx
    /usr/local/opticks/installcache/PTX/UseOptiXGeometryInstancedOCtx_generated_box.cu.ptx
    /usr/local/opticks/installcache/PTX/OptiXRap_generated_LTminimalTest.cu.ptx
    /usr/local/opticks/installcache/PTX/OptiXRap_generated_constantbg.cu.ptx
    /usr/local/opticks/installcache/PTX/UseOptiXGeometry_generated_csg_intersect_part.cu.ptx
    /usr/local/opticks/installcache/PTX/OptiXRap_generated_cbrtTest.cu.ptx
    /usr/local/opticks/installcache/PTX/UseOptiXGeometry_generated_UseOptiXGeometry.cu.ptx
    /usr/local/opticks/installcache/PTX/OptiXRap_generated_generate_+ANGULAR_ENABLED,-WAY_ENABLED.cu.ptx
    /usr/local/opticks/installcache/PTX/OptiXRap_generated_generate_+WITH_ANGULAR,-WITH_WAY_BUFFER.cu.ptx
    /usr/local/opticks/installcache/PTX/UseOptiXGeometryInstanced_generated_sphere.cu.ptx
    /usr/local/opticks/installcache/PTX/UseOptiXTextureLayeredOKImgGeo_generated_UseOptiXTextureLayeredOKImgGeo.cu.ptx
    /usr/local/opticks/installcache/PTX/OptiXRap_generated_generate_-ANGULAR_ENABLED,-WAY_ENABLED.cu
    ...
    /usr/local/opticks/installcache/PTX/OptiXRap_generated_dirtyBufferTest.cu.ptx
    /usr/local/opticks/installcache/PTX/UseOptiXGeometry_generated_box.cu.ptx
    /usr/local/opticks/installcache/PTX/OptiXRap_generated_Roots3And4Test.cu.ptx


    /usr/local/opticks/installcache/RNG
    /usr/local/opticks/installcache/RNG/cuRANDWrapper_10240_0_0.bin
    /usr/local/opticks/installcache/RNG/cuRANDWrapper_3000000_0_0.bin

    // the ptx and these bin are not used in new workflow


The ptx from CSGOptiX now get written to $OPTICKS_PREFIX/ptx: :

    epsilon:CSGOptiX blyth$ l /usr/local/opticks/ptx/
    total 3944
       0 drwxr-xr-x   6 blyth  staff     192 Nov 12 12:15 .
     680 -rw-r--r--   1 blyth  staff  347568 Nov 12 12:15 CSGOptiX_generated_CSGOptiX6.cu.ptx
    1744 -rw-r--r--   1 blyth  staff  892637 Nov  1 15:52 CSGOptiX_generated_CSGOptiX6geo.cu.ptx
       0 drwxr-xr-x  39 blyth  staff    1248 Jun 29 19:44 ..
    1488 -rw-r--r--   1 blyth  staff  759471 Apr 30  2022 CSGOptiX_generated_geo_OptiX6Test.cu.ptx
      32 -rw-r--r--   1 blyth  staff   16226 Apr 30  2022 CSGOptiX_generated_OptiX6Test.cu.ptx
    epsilon:CSGOptiX blyth$         

    N[blyth@localhost junosw]$ echo $OPTICKS_PREFIX
    /data/blyth/junotop/ExternalLibs/opticks/head

    N[blyth@localhost junosw]$ l $OPTICKS_PREFIX/ptx/
    total 1288
       0 drwxrwxr-x.  2 blyth blyth      88 Nov 10 19:12 .
    1284 -rw-r--r--.  1 blyth blyth 1312894 Nov 10 19:11 CSGOptiX_generated_CSGOptiX7.cu.ptx
       4 -rw-r--r--.  1 blyth blyth     311 Oct 20 04:33 CSGOptiX_generated_Check.cu.ptx
       0 drwxrwxr-x. 11 blyth blyth     153 Sep 27 19:24 ..
    N[blyth@localhost junosw]$ 


The RNG now coming from .opticks::

    N[blyth@localhost junosw]$ l ~/.opticks/rngcache/RNG/
    total 14180152
          4 -rw-r--r--. 1 blyth blyth       1385 Oct  7 03:46 QCurandStateTest.log
          4 drwxrwxr-x. 2 blyth blyth       4096 Oct  7 03:46 .
     429688 -rw-rw-r--. 1 blyth blyth  440000000 Oct  7 03:38 QCurandState_10000000_0_0.bin
     128908 -rw-rw-r--. 1 blyth blyth  132000000 Oct  7 03:38 QCurandState_3000000_0_0.bin
      42972 -rw-rw-r--. 1 blyth blyth   44000000 Oct  7 03:38 QCurandState_1000000_0_0.bin
        440 -rw-rw-r--. 1 blyth blyth     450560 Sep 23  2019 cuRANDWrapper_10240_0_0.bin
    8593752 -rw-rw-r--. 1 blyth blyth 8800000000 Sep 23  2019 cuRANDWrapper_200000000_0_0.bin
          0 drwxrwxr-x. 3 blyth blyth         17 Sep 14  2019 ..
      85940 -rw-rw-r--. 1 blyth blyth   88000000 Sep 12  2019 cuRANDWrapper_2000000_0_0.bin
      42972 -rw-rw-r--. 1 blyth blyth   44000000 Jul  8  2019 cuRANDWrapper_1000000_0_0.bin
    4296876 -rw-rw-r--. 1 blyth blyth 4400000000 Jul  8  2019 cuRANDWrapper_100000000_0_0.bin
     429688 -rw-rw-r--. 1 blyth blyth  440000000 Jul  8  2019 cuRANDWrapper_10000000_0_0.bin
     128908 -rw-rw-r--. 1 blyth blyth  132000000 Jul  6  2018 cuRANDWrapper_3000000_0_0.bin
    N[blyth@localhost junosw]$ 



Issue 3 : very large number of fails, many will be from the geometry config change
------------------------------------------------------------------------------------

115/510 fails::

    SLOW: tests taking longer that 15 seconds


    FAILS:  115 / 510   :  Sat Nov 12 13:19:06 2022   
      12 /98  Test #12 : SysRapTest.SPathTest                          Child aborted***Exception:     0.03   
      25 /98  Test #25 : SysRapTest.SCFTest                            ***Failed                      0.03   
      64 /98  Test #64 : SysRapTest.SOpticksResourceTest               Child aborted***Exception:     0.03   
      78 /98  Test #78 : SysRapTest.SFrameGenstep_MakeCenterExtentGensteps_Test ***Exception: SegFault         0.03   
      87 /98  Test #87 : SysRapTest.SGeoConfigTest                     Child aborted***Exception:     0.03   
      91 /98  Test #91 : SysRapTest.SBndTest                           Child aborted***Exception:     0.03   
      92 /98  Test #92 : SysRapTest.SNameTest                          Child aborted***Exception:     0.03   
      21 /39  Test #21 : BoostRapTest.BOpticksKeyTest                  ***Failed                      0.03   
      2  /45  Test #2  : OpticksCoreTest.IndexerTest                   Child aborted***Exception:     0.03   
      8  /45  Test #8  : OpticksCoreTest.OpticksFlagsTest              Child aborted***Exception:     0.03   
      10 /45  Test #10 : OpticksCoreTest.OpticksColorsTest             Child aborted***Exception:     0.03   
      11 /45  Test #11 : OpticksCoreTest.OpticksCfgTest                Child aborted***Exception:     0.03   
      12 /45  Test #12 : OpticksCoreTest.OpticksCfg2Test               Child aborted***Exception:     0.03   
      13 /45  Test #13 : OpticksCoreTest.OpticksTest                   Child aborted***Exception:     0.03   
      14 /45  Test #14 : OpticksCoreTest.OpticksTwoTest                Child aborted***Exception:     0.03   
      15 /45  Test #15 : OpticksCoreTest.OpticksResourceTest           Child aborted***Exception:     0.03   
      20 /45  Test #20 : OpticksCoreTest.OK_PROFILE_Test               Child aborted***Exception:     0.03   
      21 /45  Test #21 : OpticksCoreTest.OpticksAnaTest                Child aborted***Exception:     0.03   
      22 /45  Test #22 : OpticksCoreTest.OpticksDbgTest                Child aborted***Exception:     0.03   
      24 /45  Test #24 : OpticksCoreTest.CompositionTest               Child aborted***Exception:     0.03   
      25 /45  Test #25 : OpticksCoreTest.Composition_vs_SGLM_Test      Child aborted***Exception:     0.04   
      28 /45  Test #28 : OpticksCoreTest.EvtLoadTest                   Child aborted***Exception:     0.03   
      29 /45  Test #29 : OpticksCoreTest.OpticksEventAnaTest           Child aborted***Exception:     0.03   
      30 /45  Test #30 : OpticksCoreTest.OpticksEventCompareTest       Child aborted***Exception:     0.03   
      31 /45  Test #31 : OpticksCoreTest.OpticksEventDumpTest          Child aborted***Exception:     0.03   
      37 /45  Test #37 : OpticksCoreTest.CfgTest                       Child aborted***Exception:     0.03   
      41 /45  Test #41 : OpticksCoreTest.OpticksEventTest              Child aborted***Exception:     0.03   
      42 /45  Test #42 : OpticksCoreTest.OpticksEventLeakTest          Child aborted***Exception:     0.03   
      43 /45  Test #43 : OpticksCoreTest.OpticksRunTest                Child aborted***Exception:     0.03   
      44 /45  Test #44 : OpticksCoreTest.FlightPathTest                Child aborted***Exception:     0.04   
      45 /45  Test #45 : OpticksCoreTest.Opticks_getOutPathTest        Child aborted***Exception:     0.04   
      4  /62  Test #4  : GGeoTest.GBufferTest                          Child aborted***Exception:     0.03   
      9  /62  Test #9  : GGeoTest.GItemListTest                        Child aborted***Exception:     0.05   
      13 /62  Test #13 : GGeoTest.GScintillatorLibTest                 Child aborted***Exception:     0.04   
      15 /62  Test #15 : GGeoTest.GSourceLibTest                       Child aborted***Exception:     0.04   
      16 /62  Test #16 : GGeoTest.GBndLibTest                          Child aborted***Exception:     0.04   
      17 /62  Test #17 : GGeoTest.GBndLibInitTest                      Child aborted***Exception:     0.04   
      27 /62  Test #27 : GGeoTest.GItemIndex2Test                      Child aborted***Exception:     0.03   
      31 /62  Test #31 : GGeoTest.GPartsCreateTest                     Child aborted***Exception:     0.03   
      32 /62  Test #32 : GGeoTest.GPtsTest                             Child aborted***Exception:     0.03   
      33 /62  Test #33 : GGeoTest.GPtTest                              Child aborted***Exception:     0.03   
      35 /62  Test #35 : GGeoTest.GGeoLoadFromDirTest                  ***Exception: SegFault         0.04   
      37 /62  Test #37 : GGeoTest.BoundariesNPYTest                    Child aborted***Exception:     0.03   
      38 /62  Test #38 : GGeoTest.GAttrSeqTest                         Child aborted***Exception:     0.03   
      39 /62  Test #39 : GGeoTest.GBBoxMeshTest                        Child aborted***Exception:     0.03   
      41 /62  Test #41 : GGeoTest.GFlagsTest                           Child aborted***Exception:     0.03   
      42 /62  Test #42 : GGeoTest.GGeoLibTest                          Child aborted***Exception:     0.03   
      43 /62  Test #43 : GGeoTest.GGeoTest                             Child aborted***Exception:     0.03   
      44 /62  Test #44 : GGeoTest.GGeoIdentityTest                     Child aborted***Exception:     0.03   
      45 /62  Test #45 : GGeoTest.GGeoConvertTest                      Child aborted***Exception:     0.03   
      46 /62  Test #46 : GGeoTest.GGeoTestTest                         Child aborted***Exception:     0.04   
      47 /62  Test #47 : GGeoTest.GMakerTest                           Child aborted***Exception:     0.04   
      48 /62  Test #48 : GGeoTest.GMergedMeshTest                      Child aborted***Exception:     0.03   
      55 /62  Test #55 : GGeoTest.GSurfaceLibTest                      Child aborted***Exception:     0.03   
      57 /62  Test #57 : GGeoTest.RecordsNPYTest                       Child aborted***Exception:     0.04   
      58 /62  Test #58 : GGeoTest.GMeshLibTest                         Child aborted***Exception:     0.03   
      59 /62  Test #59 : GGeoTest.GNodeLibTest                         Child aborted***Exception:     0.03   
      60 /62  Test #60 : GGeoTest.GPhoTest                             Child aborted***Exception:     0.03   
      61 /62  Test #61 : GGeoTest.GGeoDumpTest                         Child aborted***Exception:     0.03   
      1  /45  Test #1  : ExtG4Test.X4SolidMakerTest                    Child aborted***Exception:     0.59   
      2  /45  Test #2  : ExtG4Test.X4SolidMultiUnionTest               Child aborted***Exception:     0.10   
      16 /45  Test #16 : ExtG4Test.X4PhysicalVolumeTest                Child aborted***Exception:     0.11   
      17 /45  Test #17 : ExtG4Test.X4PhysicalVolume2Test               Child aborted***Exception:     0.11   
      21 /45  Test #21 : ExtG4Test.X4CSGTest                           Child aborted***Exception:     0.10   
      22 /45  Test #22 : ExtG4Test.X4PolyconeTest                      Child aborted***Exception:     0.10   
      24 /45  Test #24 : ExtG4Test.X4GDMLBalanceTest                   Child aborted***Exception:     0.11   
      34 /45  Test #34 : ExtG4Test.X4ScintillationTest                 Child aborted***Exception:     0.10   
      38 /45  Test #38 : ExtG4Test.X4SurfaceTest                       Child aborted***Exception:     0.16   
      40 /45  Test #40 : ExtG4Test.convertMultiUnionTest               Child aborted***Exception:     0.10   
      41 /45  Test #41 : ExtG4Test.X4IntersectSolidTest                Child aborted***Exception:     0.11   
      44 /45  Test #44 : ExtG4Test.X4MeshTest                          Child aborted***Exception:     0.16   
      1  /39  Test #1  : CSGTest.CSGNodeTest                           Child aborted***Exception:     0.04   
      3  /39  Test #3  : CSGTest.CSGIntersectSolidTest                 Child aborted***Exception:     0.03   
      5  /39  Test #5  : CSGTest.CSGPrimSpecTest                       Child aborted***Exception:     0.03   
      6  /39  Test #6  : CSGTest.CSGPrimTest                           Child aborted***Exception:     0.03   
      8  /39  Test #8  : CSGTest.CSGFoundryTest                        Child aborted***Exception:     0.04   
      9  /39  Test #9  : CSGTest.CSGFoundry_getCenterExtent_Test       Child aborted***Exception:     0.05   
      10 /39  Test #10 : CSGTest.CSGFoundry_findSolidIdx_Test          Child aborted***Exception:     0.05   
      11 /39  Test #11 : CSGTest.CSGNameTest                           Child aborted***Exception:     0.03   
      12 /39  Test #12 : CSGTest.CSGTargetTest                         Child aborted***Exception:     0.03   
      13 /39  Test #13 : CSGTest.CSGTargetGlobalTest                   Child aborted***Exception:     0.03   
      14 /39  Test #14 : CSGTest.CSGFoundry_MakeCenterExtentGensteps_Test Child aborted***Exception:     0.04   
      15 /39  Test #15 : CSGTest.CSGFoundry_getFrame_Test              Child aborted***Exception:     0.03   
      16 /39  Test #16 : CSGTest.CSGFoundry_SGeo_SEvt_Test             Child aborted***Exception:     0.04   
      17 /39  Test #17 : CSGTest.CSGFoundry_ResolveCFBase_Test         Child aborted***Exception:     0.04   
      18 /39  Test #18 : CSGTest.CSGFoundryLoadTest                    Child aborted***Exception:     0.03   
      19 /39  Test #19 : CSGTest.CSGScanTest                           Child aborted***Exception:     0.03   
      22 /39  Test #22 : CSGTest.CSGMakerTest                          Child aborted***Exception:     0.04   
      23 /39  Test #23 : CSGTest.CSGQueryTest                          Child aborted***Exception:     0.04   
      24 /39  Test #24 : CSGTest.CSGSimtraceTest                       Child aborted***Exception:     0.04   
      25 /39  Test #25 : CSGTest.CSGSimtraceRerunTest                  Child aborted***Exception:     0.05   
      26 /39  Test #26 : CSGTest.CSGSimtraceSampleTest                 Child aborted***Exception:     0.04   
      27 /39  Test #27 : CSGTest.CSGCopyTest                           Child aborted***Exception:     0.05   
      33 /39  Test #33 : CSGTest.CSGIntersectComparisonTest            Child aborted***Exception:     0.04   
      36 /39  Test #36 : CSGTest.CSGSignedDistanceFieldTest            ***Exception: Interrupt        0.03   
      37 /39  Test #37 : CSGTest.CSGGeometryTest                       ***Exception: Interrupt        0.03   
      38 /39  Test #38 : CSGTest.CSGGeometryFromGeocacheTest           Child aborted***Exception:     0.03   
      2  /3   Test #2  : GeoChainTest.GeoChainVolumeTest               ***Exception: SegFault         0.40   
      3  /20  Test #3  : QUDARapTest.QScintTest                        Child aborted***Exception:     0.03   
      4  /20  Test #4  : QUDARapTest.QCerenkovIntegralTest             ***Exception: SegFault         0.03   
      5  /20  Test #5  : QUDARapTest.QCerenkovTest                     Child aborted***Exception:     0.03   
      7  /20  Test #7  : QUDARapTest.QSimTest                          Child aborted***Exception:     0.03   
      8  /20  Test #8  : QUDARapTest.QBndTest                          Child aborted***Exception:     0.03   
      9  /20  Test #9  : QUDARapTest.QPrdTest                          Child aborted***Exception:     0.03   
      10 /20  Test #10 : QUDARapTest.QOpticalTest                      Child aborted***Exception:     0.03   
      11 /20  Test #11 : QUDARapTest.QPropTest                         ***Exception: SegFault         0.03   
      13 /20  Test #13 : QUDARapTest.QSimWithEventTest                 Child aborted***Exception:     0.03   
      18 /20  Test #18 : QUDARapTest.QMultiFilmTest                    ***Exception: SegFault         0.03   
      5  /18  Test #5  : U4Test.U4GDMLReadTest                         Child aborted***Exception:     0.10   
      7  /18  Test #7  : U4Test.U4RandomTest                           ***Exception: SegFault         0.64   
      13 /18  Test #13 : U4Test.U4TreeTest                             Child aborted***Exception:     0.15   
      1  /4   Test #1  : G4CXTest.G4CXRenderTest                       Child aborted***Exception:     1.21   
      2  /4   Test #2  : G4CXTest.G4CXSimulateTest                     Child aborted***Exception:     1.02   
      3  /4   Test #3  : G4CXTest.G4CXSimtraceTest                     Child aborted***Exception:     0.84   
      4  /4   Test #4  : G4CXTest.G4CXOpticks_setGeometry_Test         Child aborted***Exception:     0.12   
    epsilon:opticks blyth$ 



sysrap fails::

      12 /98  Test #12 : SysRapTest.SPathTest                          Child aborted***Exception:     0.03   
      25 /98  Test #25 : SysRapTest.SCFTest                            ***Failed                      0.03   
      64 /98  Test #64 : SysRapTest.SOpticksResourceTest               Child aborted***Exception:     0.03   
      78 /98  Test #78 : SysRapTest.SFrameGenstep_MakeCenterExtentGensteps_Test ***Exception: SegFault         0.03   
      87 /98  Test #87 : SysRapTest.SGeoConfigTest                     Child aborted***Exception:     0.03   
      91 /98  Test #91 : SysRapTest.SBndTest                           Child aborted***Exception:     0.03   
      92 /98  Test #92 : SysRapTest.SNameTest                          Child aborted***Exception:     0.03   
 

Rerun with om-test in sysrap::

    Total Test time (real) =   4.48 sec

    The following tests FAILED:
         12 - SysRapTest.SPathTest (Child aborted)
         25 - SysRapTest.SCFTest (Failed)
         64 - SysRapTest.SOpticksResourceTest (Child aborted)
         78 - SysRapTest.SFrameGenstep_MakeCenterExtentGensteps_Test (SEGFAULT)
         87 - SysRapTest.SGeoConfigTest (Child aborted)
         91 - SysRapTest.SBndTest (Child aborted)
         92 - SysRapTest.SNameTest (Child aborted)
    Errors while running CTest
    Sat Nov 12 13:27:07 GMT 2022
    epsilon:sysrap blyth$ 


::

    epsilon:sysrap blyth$ lldb__ SPathTest 
    HEAD
    TAIL
    /Applications/Xcode/Xcode_10_1.app/Contents/Developer/usr/bin/lldb -f SPathTest --
    (lldb) target create "/usr/local/opticks/lib/SPathTest"
    Current executable set to '/usr/local/opticks/lib/SPathTest' (x86_64).
    (lldb) r
    Process 37202 launched: '/usr/local/opticks/lib/SPathTest' (x86_64)
    2022-11-12 13:35:51.567 INFO  [53814965] [test_Resolve@204] 
                                                            $TMP :                                           /tmp/blyth/opticks
                                               $DefaultOutputDir :                            /tmp/blyth/opticks/GEOM/SPathTest
                                                    $OPTICKS_TMP :                                           /tmp/blyth/opticks
                                             $OPTICKS_EVENT_BASE :                                           /tmp/blyth/opticks
                                                    $HOME/hello  :                                          /Users/blyth/hello 
                             $TMP/somewhere/over/the/rainbow.txt :            /tmp/blyth/opticks/somewhere/over/the/rainbow.txt
                            $NON_EXISTING_EVAR/elsewhere/sub.txt :                         /tmp/blyth/opticks/elsewhere/sub.txt
    Assertion failed: (idpath), function CGDir_, file /Users/blyth/opticks/sysrap/SOpticksResource.cc, line 258.
    Process 37202 stopped
    Target 0: (SPathTest) stopped.
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff5c1fcb66 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff5c3c7080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff5c1581ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff5c1201ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x000000010022cb5d libSysRap.dylib`SOpticksResource::CGDir_(setkey=true, rel="CSG_GGeo") at SOpticksResource.cc:258
        frame #5: 0x000000010022c824 libSysRap.dylib`SOpticksResource::CGDir(setkey=true) at SOpticksResource.cc:254
        frame #6: 0x000000010022c864 libSysRap.dylib`SOpticksResource::CFBase() at SOpticksResource.cc:295
        frame #7: 0x000000010022d07b libSysRap.dylib`SOpticksResource::Get(key="CFBase") at SOpticksResource.cc:503
        frame #8: 0x00000001001574c1 libSysRap.dylib`SPath::Resolve(spec_="$CFBase/CSGFoundry/SSim", create_dirs=0) at SPath.cc:184
        frame #9: 0x000000010001e7e2 SPathTest`test_Resolve() at SPathTest.cc:227
        frame #10: 0x0000000100020a3c SPathTest`main(argc=1, argv=0x00007ffeefbfe988) at SPathTest.cc:402
        frame #11: 0x00007fff5c0ac015 libdyld.dylib`start + 1
        frame #12: 0x00007fff5c0ac015 libdyld.dylib`start + 1
    (lldb) 



After removing idpath assert down to::

    The following tests FAILED:
         25 - SysRapTest.SCFTest (Failed)
         78 - SysRapTest.SFrameGenstep_MakeCenterExtentGensteps_Test (SEGFAULT)
         91 - SysRapTest.SBndTest (SEGFAULT)
         92 - SysRapTest.SNameTest (SEGFAULT)
    Errors while running CTest
    Sat Nov 12 13:41:08 GMT 2022


Fix the above. Mostly lack of loaded resources or geometry specifics. 



okc fails
--------------

::

    okc
    om-test 


    Total Test time (real) =   6.20 sec

    The following tests FAILED:
          2 - OpticksCoreTest.IndexerTest (Child aborted)
          8 - OpticksCoreTest.OpticksFlagsTest (Child aborted)
         10 - OpticksCoreTest.OpticksColorsTest (Child aborted)
         11 - OpticksCoreTest.OpticksCfgTest (Child aborted)
         12 - OpticksCoreTest.OpticksCfg2Test (Child aborted)
         13 - OpticksCoreTest.OpticksTest (Child aborted)
         14 - OpticksCoreTest.OpticksTwoTest (Child aborted)
         15 - OpticksCoreTest.OpticksResourceTest (Child aborted)
         20 - OpticksCoreTest.OK_PROFILE_Test (Child aborted)
         21 - OpticksCoreTest.OpticksAnaTest (Child aborted)
         22 - OpticksCoreTest.OpticksDbgTest (Child aborted)
         24 - OpticksCoreTest.CompositionTest (Child aborted)
         25 - OpticksCoreTest.Composition_vs_SGLM_Test (Child aborted)
         28 - OpticksCoreTest.EvtLoadTest (Child aborted)
         29 - OpticksCoreTest.OpticksEventAnaTest (Child aborted)
         30 - OpticksCoreTest.OpticksEventCompareTest (Child aborted)
         31 - OpticksCoreTest.OpticksEventDumpTest (Child aborted)
         37 - OpticksCoreTest.CfgTest (Child aborted)
         41 - OpticksCoreTest.OpticksEventTest (Child aborted)
         42 - OpticksCoreTest.OpticksEventLeakTest (Child aborted)
         43 - OpticksCoreTest.OpticksRunTest (Child aborted)
         44 - OpticksCoreTest.FlightPathTest (Child aborted)
         45 - OpticksCoreTest.Opticks_getOutPathTest (Child aborted)
    Errors while running CTest
    Sat Nov 12 15:01:03 GMT 2022
    epsilon:optickscore blyth$ 

Removing requirement for OPTICKS_KEY gets down to::

    The following tests FAILED:
         15 - OpticksCoreTest.OpticksResourceTest (Child aborted)
         41 - OpticksCoreTest.OpticksEventTest (SEGFAULT)
         42 - OpticksCoreTest.OpticksEventLeakTest (SEGFAULT)
         43 - OpticksCoreTest.OpticksRunTest (SEGFAULT)
    Errors while running CTest
    Sat Nov 12 15:07:44 GMT 2022
    epsilon:optickscore blyth$ 

Fix the above using IDPATH_TRANSITIONAL of "/tmp" as a kludge. 





Now down to 66::

    FAILS:  66  / 510   :  Sat Nov 12 15:25:24 2022   
      13 /62  Test #13 : GGeoTest.GScintillatorLibTest                 ***Exception: SegFault         0.04   
      16 /62  Test #16 : GGeoTest.GBndLibTest                          Child aborted***Exception:     0.04   
      17 /62  Test #17 : GGeoTest.GBndLibInitTest                      Child aborted***Exception:     0.04   
      31 /62  Test #31 : GGeoTest.GPartsCreateTest                     ***Exception: SegFault         0.03   
      33 /62  Test #33 : GGeoTest.GPtTest                              ***Exception: SegFault         0.03   
      35 /62  Test #35 : GGeoTest.GGeoLoadFromDirTest                  ***Exception: SegFault         0.03   
      37 /62  Test #37 : GGeoTest.BoundariesNPYTest                    Child aborted***Exception:     0.04   
      42 /62  Test #42 : GGeoTest.GGeoLibTest                          Child aborted***Exception:     0.03   
      43 /62  Test #43 : GGeoTest.GGeoTest                             Child aborted***Exception:     0.04   
      44 /62  Test #44 : GGeoTest.GGeoIdentityTest                     Child aborted***Exception:     0.04   
      45 /62  Test #45 : GGeoTest.GGeoConvertTest                      Child aborted***Exception:     0.04   
      47 /62  Test #47 : GGeoTest.GMakerTest                           Child aborted***Exception:     0.04   
      55 /62  Test #55 : GGeoTest.GSurfaceLibTest                      Child aborted***Exception:     0.04   
      57 /62  Test #57 : GGeoTest.RecordsNPYTest                       Child aborted***Exception:     0.04   
      59 /62  Test #59 : GGeoTest.GNodeLibTest                         ***Exception: SegFault         0.04   
      60 /62  Test #60 : GGeoTest.GPhoTest                             Child aborted***Exception:     0.04   
      61 /62  Test #61 : GGeoTest.GGeoDumpTest                         Child aborted***Exception:     0.04   
      1  /45  Test #1  : ExtG4Test.X4SolidMakerTest                    Child aborted***Exception:     0.11   
      16 /45  Test #16 : ExtG4Test.X4PhysicalVolumeTest                Child aborted***Exception:     0.11   
      17 /45  Test #17 : ExtG4Test.X4PhysicalVolume2Test               Child aborted***Exception:     0.11   
      38 /45  Test #38 : ExtG4Test.X4SurfaceTest                       Child aborted***Exception:     0.16   
      41 /45  Test #41 : ExtG4Test.X4IntersectSolidTest                Child aborted***Exception:     0.10   
      44 /45  Test #44 : ExtG4Test.X4MeshTest                          Child aborted***Exception:     0.15   
      1  /39  Test #1  : CSGTest.CSGNodeTest                           Child aborted***Exception:     0.03   
      3  /39  Test #3  : CSGTest.CSGIntersectSolidTest                 ***Exception: Interrupt        0.03   
      5  /39  Test #5  : CSGTest.CSGPrimSpecTest                       Child aborted***Exception:     0.03   
      6  /39  Test #6  : CSGTest.CSGPrimTest                           Child aborted***Exception:     0.03   
      8  /39  Test #8  : CSGTest.CSGFoundryTest                        Child aborted***Exception:     0.04   
      9  /39  Test #9  : CSGTest.CSGFoundry_getCenterExtent_Test       Child aborted***Exception:     0.06   
      10 /39  Test #10 : CSGTest.CSGFoundry_findSolidIdx_Test          Child aborted***Exception:     0.05   
      11 /39  Test #11 : CSGTest.CSGNameTest                           Child aborted***Exception:     0.04   
      12 /39  Test #12 : CSGTest.CSGTargetTest                         Child aborted***Exception:     0.04   
      13 /39  Test #13 : CSGTest.CSGTargetGlobalTest                   Child aborted***Exception:     0.04   
      14 /39  Test #14 : CSGTest.CSGFoundry_MakeCenterExtentGensteps_Test Child aborted***Exception:     0.03   
      15 /39  Test #15 : CSGTest.CSGFoundry_getFrame_Test              Child aborted***Exception:     0.04   
      16 /39  Test #16 : CSGTest.CSGFoundry_SGeo_SEvt_Test             Child aborted***Exception:     0.03   
      18 /39  Test #18 : CSGTest.CSGFoundryLoadTest                    Child aborted***Exception:     0.03   
      19 /39  Test #19 : CSGTest.CSGScanTest                           Child aborted***Exception:     0.03   
      22 /39  Test #22 : CSGTest.CSGMakerTest                          Child aborted***Exception:     0.05   
      23 /39  Test #23 : CSGTest.CSGQueryTest                          Child aborted***Exception:     0.04   
      24 /39  Test #24 : CSGTest.CSGSimtraceTest                       Child aborted***Exception:     0.05   
      25 /39  Test #25 : CSGTest.CSGSimtraceRerunTest                  Child aborted***Exception:     0.04   
      26 /39  Test #26 : CSGTest.CSGSimtraceSampleTest                 Child aborted***Exception:     0.03   
      27 /39  Test #27 : CSGTest.CSGCopyTest                           Child aborted***Exception:     0.03   
      33 /39  Test #33 : CSGTest.CSGIntersectComparisonTest            Child aborted***Exception:     0.04   
      36 /39  Test #36 : CSGTest.CSGSignedDistanceFieldTest            ***Exception: Interrupt        0.03   
      37 /39  Test #37 : CSGTest.CSGGeometryTest                       ***Exception: Interrupt        0.03   
      38 /39  Test #38 : CSGTest.CSGGeometryFromGeocacheTest           ***Exception: Interrupt        0.03   
      2  /3   Test #2  : GeoChainTest.GeoChainVolumeTest               ***Exception: SegFault         0.38   
      3  /20  Test #3  : QUDARapTest.QScintTest                        ***Exception: SegFault         0.02   
      4  /20  Test #4  : QUDARapTest.QCerenkovIntegralTest             ***Exception: SegFault         0.03   
      5  /20  Test #5  : QUDARapTest.QCerenkovTest                     Child aborted***Exception:     0.03   
      7  /20  Test #7  : QUDARapTest.QSimTest                          Child aborted***Exception:     1.10   
      8  /20  Test #8  : QUDARapTest.QBndTest                          ***Exception: SegFault         0.03   
      9  /20  Test #9  : QUDARapTest.QPrdTest                          ***Exception: SegFault         0.03   
      10 /20  Test #10 : QUDARapTest.QOpticalTest                      ***Exception: SegFault         0.03   
      11 /20  Test #11 : QUDARapTest.QPropTest                         ***Exception: SegFault         0.03   
      13 /20  Test #13 : QUDARapTest.QSimWithEventTest                 Child aborted***Exception:     1.11   
      18 /20  Test #18 : QUDARapTest.QMultiFilmTest                    ***Exception: SegFault         0.02   
      5  /18  Test #5  : U4Test.U4GDMLReadTest                         Child aborted***Exception:     0.09   
      7  /18  Test #7  : U4Test.U4RandomTest                           ***Exception: SegFault         0.57   
      13 /18  Test #13 : U4Test.U4TreeTest                             Child aborted***Exception:     0.12   
      1  /4   Test #1  : G4CXTest.G4CXRenderTest                       Child aborted***Exception:     1.08   
      2  /4   Test #2  : G4CXTest.G4CXSimulateTest                     Child aborted***Exception:     1.07   
      3  /4   Test #3  : G4CXTest.G4CXSimtraceTest                     Child aborted***Exception:     0.91   
      4  /4   Test #4  : G4CXTest.G4CXOpticks_setGeometry_Test         Child aborted***Exception:     0.12   
    epsilon:optickscore blyth$ 



GGeo fails::

    The following tests FAILED:
         13 - GGeoTest.GScintillatorLibTest (SEGFAULT)
         16 - GGeoTest.GBndLibTest (Child aborted)
         17 - GGeoTest.GBndLibInitTest (Child aborted)
         31 - GGeoTest.GPartsCreateTest (SEGFAULT)
         33 - GGeoTest.GPtTest (SEGFAULT)
         35 - GGeoTest.GGeoLoadFromDirTest (SEGFAULT)
         37 - GGeoTest.BoundariesNPYTest (Child aborted)
         42 - GGeoTest.GGeoLibTest (Child aborted)
         43 - GGeoTest.GGeoTest (Child aborted)
         44 - GGeoTest.GGeoIdentityTest (Child aborted)
         45 - GGeoTest.GGeoConvertTest (Child aborted)
         47 - GGeoTest.GMakerTest (Child aborted)
         55 - GGeoTest.GSurfaceLibTest (Child aborted)
         57 - GGeoTest.RecordsNPYTest (Child aborted)
         59 - GGeoTest.GNodeLibTest (SEGFAULT)
         60 - GGeoTest.GPhoTest (Child aborted)
         61 - GGeoTest.GGeoDumpTest (Child aborted)
    Errors while running CTest
    Sat Nov 12 15:26:39 GMT 2022
    epsilon:ggeo blyth$ 


Access to geometry failures
----------------------------

::

    epsilon:ggeo blyth$ GScintillatorLibTest 
    2022-11-12 15:30:34.209 ERROR [53974744] [*NPY<double>::load@1093] NPY<T>::load failed for path [/tmp/GScintillatorLib/GScintillatorLib.npy] use debugload with NPYLoadTest to investigate (problems are usually from dtype mismatches) 
    2022-11-12 15:30:34.210 INFO  [53974744] [GPropertyLib::loadFromCache@608] Optional buffer not present  dir /tmp/GScintillatorLib name GScintillatorLib.npy
    2022-11-12 15:30:34.210 INFO  [53974744] [main@196]  GScintillatorLib.getNumRaw  0 GScintillatorLib.getNumRawOriginal  0
    [ nraw 0] nraw 0


    Segmentation fault: 11
    epsilon:ggeo blyth$ find ~/.opticks/GEOM/J004 -name GScintillatorLib.npy
    /Users/blyth/.opticks/GEOM/J004/GGeo/GScintillatorLib/GScintillatorLib.npy
    epsilon:ggeo blyth$ 


    epsilon:ggeo blyth$ GEOM=J004 source ~/opticks/bin/GEOM_.sh 
                       BASH_SOURCE : /Users/blyth/opticks/bin/GEOM_.sh 
                               gp_ : J004_GDMLPath 
                                gp :  
                               cg_ : J004_CFBaseFromGEOM 
                                cg : /Users/blyth/.opticks/GEOM/J004 
                       TMP_GEOMDIR : /tmp/blyth/opticks/GEOM/J004 
                           GEOMDIR : /Users/blyth/.opticks/GEOM/J004 
                       BASH_SOURCE : /Users/blyth/opticks/bin/GEOM_.sh 

    epsilon:ggeo blyth$ echo $CFBASE

    epsilon:ggeo blyth$ env | grep J004
    GEOMDIR=/Users/blyth/.opticks/GEOM/J004
    J004_CFBaseFromGEOM=/Users/blyth/.opticks/GEOM/J004
    epsilon:ggeo blyth$ 


How to configure GEOM for tests is unclear in new flexible geometry working environment
------------------------------------------------------------------------------------------

* GEOM jumps around between geometries and levels of geometry,
  testing cannot rely on that taking some reference value 

* GEOM does not fulfil the assumption of always pointing to a 
  fully featured geometry that the former OPTICKS_KEY did

* MAYBE : introduce a separate OPTICKS_T_GEOM
  that is expected to be less variable than GEOM 
  and which is used to control the geometry used by opticks-t 
  (actually the usage needs to be at om-test level, 
  as need to support running at single package level)

* HMM: can this be done purely at bash level ?  

* just need to get the old idpath via IDPATH_TRANSITIONAL 
  to work from transitional "$CFBaseFromGEOM/GGeo" ? 

* om-test can invoke bin/GEOM_.sh script ?

  * HMM: not scalable to have users editing something thats in repo 
    to configure where their geometry is 
  * TODO: switch to userspace for GEOM config::

       "GEOM=$OPTICKS_T_GEOM source ~/.opticks/GEOM.sh" ?


What should IDPATH_TRANSITIONAL be ? "$CFBaseFromGEOM/GGeo" ?
----------------------------------------------------------------

::

    epsilon:~ blyth$ cd ~/.opticks
    epsilon:.opticks blyth$ find . -name GScintillatorLib.npy
    ./ntds3/G4CXOpticks/GGeo/GScintillatorLib/GScintillatorLib.npy
    ./geocache/G4OKVolumeTest_World_pv_g4live/g4ok_gltf/454372a9f3c659bed5168603f4a26a22/1/GScintillatorLib/GScintillatorLib.npy
    ./geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/3dbec4dc3bdef47884fe48af781a179d/1/GScintillatorLib/GScintillatorLib.npy
    ./GEOM/J004/GGeo/GScintillatorLib/GScintillatorLib.npy
    ./GEOM/example_pet/GGeo/GScintillatorLib/GScintillatorLib.npy
    epsilon:.opticks blyth$ 

    epsilon:.opticks blyth$ cd /usr/local/opticks/geocache/
    epsilon:geocache blyth$ find . -name GScintillatorLib.npy 
    ./G4OKPMTSimTest_nnvt_body_phys_g4live/g4ok_gltf/bf7a3ecb6d69fdec7b83d46e187503f1/1/GScintillatorLib/GScintillatorLib.npy
    ./OpticksEmbedded_World_g4live/g4ok_gltf/43bc26d43bba43fc6c680afe1e9df8fa/1/GScintillatorLib/GScintillatorLib.npy
    ./OKX4Test_lWorld0x5780b30_PV_g4live/g4ok_gltf/5303cd587554cb16682990189831ae83/1/GScintillatorLib/GScintillatorLib.npy
    ./OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GScintillatorLib/GScintillatorLib.npy
    ./OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/0dce832a26eb41b58a000497a3127cb8/1/GScintillatorLib/GScintillatorLib.npy
    ./OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/742ab212f7f2da665ed627411ebdb07d/1/GScintillatorLib/GScintillatorLib.npy
    ./OKX4Test_lWorld0x68777d0_PV_g4live/g4ok_gltf/b574f652da8bb005cefa723ecf24b65b/1/GScintillatorLib/GScintillatorLib.npy
    ./G4OKVolumeTest_World_pv_g4live/g4ok_gltf/bc3ed0133d9cea75f52fa6fb60c6c988/1/GScintillatorLib/GScintillatorLib.npy
    ./G4OKVolumeTest_World_pv_g4live/g4ok_gltf/519a6b7d159ca9d99452211b9361d94e/1/GScintillatorLib/GScintillatorLib.npy
    ./G4OKVolumeTest_World_pv_g4live/g4ok_gltf/36866edf42c8c86e06bf9a520a61f11d/1/GScintillatorLib/GScintillatorLib.npy
    ./G4OKVolumeTest_World_pv_g4live/g4ok_gltf/454372a9f3c659bed5168603f4a26a22/1/GScintillatorLib/GScintillatorLib.npy
    ./G4OKVolumeTest_World_pv_g4live/g4ok_gltf/4945968a8835051c4cef2c31f2bb109a/1/GScintillatorLib/GScintillatorLib.npy
    ./G4OKVolumeTest_World_pv_g4live/g4ok_gltf/89064e4668fdbf2021363b5713f8c1ea/1/GScintillatorLib/GScintillatorLib.npy
    ./G4OKVolumeTest_World_pv_g4live/g4ok_gltf/0747692aead8f4ff52e3c3911ed6e2d3/1/GScintillatorLib/GScintillatorLib.npy
    ./CerenkovMinimal_World_g4live/g4ok_gltf/792496b5e2cc08bdf5258cc12e63de9f/1/GScintillatorLib/GScintillatorLib.npy
    ...


::

    epsilon:opticks blyth$ cd ~/.opticks/GEOM/J004
    epsilon:J004 blyth$ l
    total 83464
        0 drwxr-xr-x   4 blyth  staff       128 Nov  4 20:29 ..
        0 drwxr-xr-x   5 blyth  staff       160 Oct 11 20:25 G4CXSimtraceTest
        0 drwxr-xr-x   8 blyth  staff       256 Oct 11 16:24 .
    41472 -rw-rw-r--   1 blyth  staff  20992917 Oct 11 15:50 origin.gdml
        8 -rw-rw-r--   1 blyth  staff       190 Oct 11 15:50 origin_gdxml_report.txt
    41984 -rw-rw-r--   1 blyth  staff  20994470 Oct 11 15:50 origin_raw.gdml
        0 drwxrwxr-x  17 blyth  staff       544 Oct 11 15:50 GGeo
        0 drwxr-xr-x  13 blyth  staff       416 Oct 11 15:50 CSGFoundry

    epsilon:J004 blyth$ l GGeo/
    total 16
    0 drwxr-xr-x    8 blyth  staff   256 Oct 11 16:24 ..
    0 drwxrwxr-x   17 blyth  staff   544 Oct 11 15:50 .
    8 -rw-rw-r--    1 blyth  staff   223 Oct 11 15:50 cachemeta.json
    8 -rw-rw-r--    1 blyth  staff   160 Oct 11 15:50 runcomment.txt
    0 drwxr-xr-x    5 blyth  staff   160 Oct 11 15:50 stree
    0 drwxrwxr-x    3 blyth  staff    96 Oct 11 15:50 GBndLib
    0 drwxrwxr-x    7 blyth  staff   224 Oct 11 15:50 GItemList
    0 drwxrwxr-x    3 blyth  staff    96 Oct 11 15:50 GSourceLib
    0 drwxrwxr-x    6 blyth  staff   192 Oct 11 15:50 GScintillatorLib
    0 drwxrwxr-x    5 blyth  staff   160 Oct 11 15:50 GSurfaceLib
    0 drwxrwxr-x    4 blyth  staff   128 Oct 11 15:50 GMaterialLib
    0 drwxrwxr-x   11 blyth  staff   352 Oct 11 15:50 GNodeLib
    0 drwxrwxr-x  144 blyth  staff  4608 Oct 11 15:50 GMeshLib
    0 drwxrwxr-x  143 blyth  staff  4576 Oct 11 15:50 GMeshLibNCSG
    0 drwxrwxr-x   12 blyth  staff   384 Oct 11 15:50 GPts
    0 drwxrwxr-x   12 blyth  staff   384 Oct 11 15:50 GParts
    0 drwxrwxr-x   12 blyth  staff   384 Oct 11 15:50 GMergedMesh
    epsilon:J004 blyth$ 






::

    epsilon:tests blyth$ cat OpticksTest.sh 
    #!/bin/bash -l 

    export GEOM=J004
    source $(dirname $BASH_SOURCE)/../../bin/GEOM_.sh  
    env | grep $GEOM

    OpticksTest 



    epsilon:tests blyth$ cat GScintillatorLibTest.sh 
    #!/bin/bash -l 

    export GEOM=J004
    source $(dirname $BASH_SOURCE)/../../bin/GEOM_.sh  
    env | grep $GEOM

    GScintillatorLibTest


    
With testenv bracketing::

     566 om-testenv-dump(){
     567     local pfx="$1"
     568     if [ -z "$QUIET" ]; then 
     569         local fmt="$pfx %20s : %s \n"
     570         local vars="$(om-testenv-vars)"
     571         local var ; for var in $vars ; do printf "$fmt" "$var" "${!var}" ; done
     572     fi
     573 }
     574 
     575 om-testenv-push()
     576 {
     577     om-testenv-dump "[push"
     578 
     579     export OM_KEEP_GEOM=$GEOM
     580     export GEOM=${OPTICKS_T_GEOM:-$GEOM}
     581 
     582     source $(dirname $BASH_SOURCE)/bin/GEOM_.sh
     583 
     584     om-testenv-dump "]push"
     585 }
     586 om-testenv-pop()
     587 {
     588     om-testenv-dump "[pop "
     589 
     590     if [ -n "$OM_KEEP_GEOM" ] ; then
     591         export GEOM=$OM_KEEP_GEOM
     592         unset OM_KEEP_GEOM
     593     else
     594         unset GEOM
     595     fi
     596 
     597     om-testenv-dump "]pop "
     598 }
     599 




::

    90% tests passed, 6 tests failed out of 62

    Total Test time (real) =  14.81 sec

    The following tests FAILED:
         35 - GGeoTest.GGeoLoadFromDirTest (SEGFAULT)

         43 - GGeoTest.GGeoTest (Child aborted)
         44 - GGeoTest.GGeoIdentityTest (Child aborted)
         45 - GGeoTest.GGeoConvertTest (Child aborted)
         60 - GGeoTest.GPhoTest (Child aborted)
         61 - GGeoTest.GGeoDumpTest (Child aborted)
    Errors while running CTest
    Sat Nov 12 18:37:34 GMT 2022
    [pop           BASH_SOURCE : /Users/blyth/opticks/om.bash 
    [pop              FUNCNAME : om-testenv-dump 
    [pop          OM_KEEP_GEOM :  
    [pop                  GEOM : J004 
    [pop        OPTICKS_T_GEOM : J004 
    ]pop           BASH_SOURCE : /Users/blyth/opticks/om.bash 
    ]pop              FUNCNAME : om-testenv-dump 
    ]pop          OM_KEEP_GEOM :  
    ]pop                  GEOM :  
    ]pop        OPTICKS_T_GEOM : J004 
    epsilon:ggeo blyth$ 



GGeoTest::    

    epsilon:ggeo blyth$ om-testenv-push
    [push          BASH_SOURCE : /Users/blyth/opticks/om.bash 
    [push             FUNCNAME : om-testenv-dump 
    [push         OM_KEEP_GEOM :  
    [push                 GEOM :  
    [push       OPTICKS_T_GEOM : J004 
                       BASH_SOURCE : /Users/blyth/opticks/bin/GEOM_.sh 
                               gp_ : J004_GDMLPath 
                                gp :  
                               cg_ : J004_CFBaseFromGEOM 
                                cg : /Users/blyth/.opticks/GEOM/J004 
                       TMP_GEOMDIR : /tmp/blyth/opticks/GEOM/J004 
                           GEOMDIR : /Users/blyth/.opticks/GEOM/J004 
                       BASH_SOURCE : /Users/blyth/opticks/bin/GEOM_.sh 

    ]push          BASH_SOURCE : /Users/blyth/opticks/om.bash 
    ]push             FUNCNAME : om-testenv-dump 
    ]push         OM_KEEP_GEOM :  
    ]push                 GEOM : J004 
    ]push       OPTICKS_T_GEOM : J004 
    epsilon:ggeo blyth$ GGeoTest 
    Assertion failed: (mm->getParts() == NULL), function deferredCreateGParts, file /Users/blyth/opticks/ggeo/GGeo.cc, line 1617.
    Abort trap: 6
    epsilon:ggeo blyth$ 


    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff5c1fcb66 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff5c3c7080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff5c1581ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff5c1201ac libsystem_c.dylib`__assert_rtn + 320
      * frame #4: 0x00000001002f542a libGGeo.dylib`GGeo::deferredCreateGParts(this=0x0000000101902f00) at GGeo.cc:1617
        frame #5: 0x00000001002f35e0 libGGeo.dylib`GGeo::deferred(this=0x0000000101902f00) at GGeo.cc:655
        frame #6: 0x00000001002f2f4b libGGeo.dylib`GGeo::postLoadFromCache(this=0x0000000101902f00) at GGeo.cc:605
        frame #7: 0x00000001002eed35 libGGeo.dylib`GGeo::loadFromCache(this=0x0000000101902f00) at GGeo.cc:586
        frame #8: 0x00000001002ee8e9 libGGeo.dylib`GGeo::Load(ok=0x00007ffeefbfe620) at GGeo.cc:134
        frame #9: 0x0000000100009fd9 GGeoTest`main(argc=1, argv=0x00007ffeefbfe890) at GGeoTest.cc:426
        frame #10: 0x00007fff5c0ac015 libdyld.dylib`start + 1
    (lldb) 


    (lldb) f 4
    frame #4: 0x00000001002f542a libGGeo.dylib`GGeo::deferredCreateGParts(this=0x0000000101902f00) at GGeo.cc:1617
       1614	    for(unsigned i=0 ; i < nmm ; i++)
       1615	    {
       1616	        GMergedMesh* mm = m_geolib->getMergedMesh(i);
    -> 1617	        assert( mm->getParts() == NULL ); 
       1618	
       1619	        GPts* pts = mm->getPts(); 
       1620	        LOG_IF(fatal, pts == nullptr ) << " pts NULL, cannot create GParts for mm " << i ; 
    (lldb) p nmm
    (unsigned int) $0 = 10
    (lldb) 



Four fails for the same cause::
    
    epsilon:ggeo blyth$ GGeoIdentityTest
    Assertion failed: (mm->getParts() == NULL), function deferredCreateGParts, file /Users/blyth/opticks/ggeo/GGeo.cc, line 1617.
    Abort trap: 6
    epsilon:ggeo blyth$ GGeoTest 
    Assertion failed: (mm->getParts() == NULL), function deferredCreateGParts, file /Users/blyth/opticks/ggeo/GGeo.cc, line 1617.
    Abort trap: 6
    epsilon:ggeo blyth$ GGeoConvertTest 
    Assertion failed: (mm->getParts() == NULL), function deferredCreateGParts, file /Users/blyth/opticks/ggeo/GGeo.cc, line 1617.
    Abort trap: 6
    epsilon:ggeo blyth$ GPhoTest 
    Assertion failed: (mm->getParts() == NULL), function deferredCreateGParts, file /Users/blyth/opticks/ggeo/GGeo.cc, line 1617.
    Abort trap: 6
    epsilon:ggeo blyth$ GGeoDumpTest 
    Assertion failed: (mm->getParts() == NULL), function deferredCreateGParts, file /Users/blyth/opticks/ggeo/GGeo.cc, line 1617.
    Abort trap: 6
    epsilon:ggeo blyth$ 


    

Now down to 57::



    FAILS:  57  / 510   :  Sat Nov 12 18:53:04 2022   
      21 /129 Test #21 : NPYTest.NContourTest                          ***Exception: SegFault         0.03   FIXED

      43 /62  Test #43 : GGeoTest.GGeoTest                             Child aborted***Exception:     1.36   
      44 /62  Test #44 : GGeoTest.GGeoIdentityTest                     Child aborted***Exception:     1.36   
      45 /62  Test #45 : GGeoTest.GGeoConvertTest                      Child aborted***Exception:     1.34   
      60 /62  Test #60 : GGeoTest.GPhoTest                             Child aborted***Exception:     1.32   
      61 /62  Test #61 : GGeoTest.GGeoDumpTest                         Child aborted***Exception:     1.33   

      All above 5 are asserts from non-null GParts loaded from deferred ? 

      1  /45  Test #1  : ExtG4Test.X4SolidMakerTest                    Child aborted***Exception:     0.11    FIXED JustOrb
      2  /45  Test #2  : ExtG4Test.X4SolidMultiUnionTest               Child aborted***Exception:     0.11    FIXED GEOM clash 

      16 /45  Test #16 : ExtG4Test.X4PhysicalVolumeTest                Child aborted***Exception:     0.10    SKIPPED : API NO LONGER USED
      17 /45  Test #17 : ExtG4Test.X4PhysicalVolume2Test               Child aborted***Exception:     0.11    SKIPPED : API NO LONGER USED 
      38 /45  Test #38 : ExtG4Test.X4SurfaceTest                       Child aborted***Exception:     0.17    SKIPPED : API NO LONGER USED

    (lldb) f 4
    frame #4: 0x00000001001e8cbb libExtG4.dylib`X4PhysicalVolume::Convert(top=0x00000001081228f0, argforce="--printenabled --nogdmlpath") at X4PhysicalVolume.cc:124
       121 	
       122 	GGeo* X4PhysicalVolume::Convert(const G4VPhysicalVolume* const top, const char* argforce)
       123 	{
    -> 124 	    assert(0) ; // NOT USED IN NEW WORKFLOW : NOW USING X4Geo::Translate ? 
       125 	
       126 	    const char* key = X4PhysicalVolume::Key(top) ; 
       127 	
    (lldb) 


      40 /45  Test #40 : ExtG4Test.convertMultiUnionTest               Child aborted***Exception:     0.11     FIXED GEOM clash 
      41 /45  Test #41 : ExtG4Test.X4IntersectSolidTest                Child aborted***Exception:     0.10     FIXED GEOM clash 
      42 /45  Test #42 : ExtG4Test.X4SimtraceTest                      Child aborted***Exception:     0.10     FIXED GEOM clash  
      43 /45  Test #43 : ExtG4Test.X4IntersectVolumeTest               Child aborted***Exception:     0.21     FIXED GEOM clash  
      44 /45  Test #44 : ExtG4Test.X4MeshTest                          ***Failed                      0.10     FIXED GEOM clash  
      45 /45  Test #45 : ExtG4Test.X4VolumeMakerTest                   Child aborted***Exception:     0.11     FIXED GEOM clash

      FIXED ALL THE X4 TESTS


      1  /39  Test #1  : CSGTest.CSGNodeTest                           Child aborted***Exception:     0.03   
      3  /39  Test #3  : CSGTest.CSGIntersectSolidTest                 Child aborted***Exception:     0.03   
      5  /39  Test #5  : CSGTest.CSGPrimSpecTest                       Child aborted***Exception:     0.03   
      6  /39  Test #6  : CSGTest.CSGPrimTest                           Child aborted***Exception:     0.04   
      8  /39  Test #8  : CSGTest.CSGFoundryTest                        Child aborted***Exception:     0.04   
      9  /39  Test #9  : CSGTest.CSGFoundry_getCenterExtent_Test       Child aborted***Exception:     0.06   
      10 /39  Test #10 : CSGTest.CSGFoundry_findSolidIdx_Test          Child aborted***Exception:     0.04   
      11 /39  Test #11 : CSGTest.CSGNameTest                           Child aborted***Exception:     0.03   
      13 /39  Test #13 : CSGTest.CSGTargetGlobalTest                   Child aborted***Exception:     0.04   
      14 /39  Test #14 : CSGTest.CSGFoundry_MakeCenterExtentGensteps_Test Child aborted***Exception:     0.03   
      15 /39  Test #15 : CSGTest.CSGFoundry_getFrame_Test              Child aborted***Exception:     0.04   
      16 /39  Test #16 : CSGTest.CSGFoundry_SGeo_SEvt_Test             Child aborted***Exception:     0.03   
      19 /39  Test #19 : CSGTest.CSGScanTest                           Child aborted***Exception:     0.03   
      22 /39  Test #22 : CSGTest.CSGMakerTest                          Child aborted***Exception:     0.04   
      23 /39  Test #23 : CSGTest.CSGQueryTest                          Child aborted***Exception:     0.03   
      25 /39  Test #25 : CSGTest.CSGSimtraceRerunTest                  ***Exception: SegFault         0.11   
      26 /39  Test #26 : CSGTest.CSGSimtraceSampleTest                 ***Exception: SegFault         0.11   
      27 /39  Test #27 : CSGTest.CSGCopyTest                           Child aborted***Exception:     0.04   
      33 /39  Test #33 : CSGTest.CSGIntersectComparisonTest            Child aborted***Exception:     0.04   
      35 /39  Test #35 : CSGTest.CSGNodeScanTest                       Child aborted***Exception:     0.03   
      36 /39  Test #36 : CSGTest.CSGSignedDistanceFieldTest            Child aborted***Exception:     0.03   
      37 /39  Test #37 : CSGTest.CSGGeometryTest                       Child aborted***Exception:     0.03   
      38 /39  Test #38 : CSGTest.CSGGeometryFromGeocacheTest           Child aborted***Exception:     0.03   

      1  /3   Test #1  : GeoChainTest.GeoChainSolidTest                Child aborted***Exception:     0.12   
      2  /3   Test #2  : GeoChainTest.GeoChainVolumeTest               Child aborted***Exception:     0.16   
      3  /3   Test #3  : GeoChainTest.GeoChainNodeTest                 Child aborted***Exception:     0.10   

      3  /20  Test #3  : QUDARapTest.QScintTest                        ***Exception: SegFault         0.02   
      4  /20  Test #4  : QUDARapTest.QCerenkovIntegralTest             ***Exception: SegFault         0.03   
      5  /20  Test #5  : QUDARapTest.QCerenkovTest                     Child aborted***Exception:     0.03   
      7  /20  Test #7  : QUDARapTest.QSimTest                          Child aborted***Exception:     1.15   
      8  /20  Test #8  : QUDARapTest.QBndTest                          ***Exception: SegFault         0.03   
      9  /20  Test #9  : QUDARapTest.QPrdTest                          ***Exception: SegFault         0.03   
      10 /20  Test #10 : QUDARapTest.QOpticalTest                      ***Exception: SegFault         0.03   
      11 /20  Test #11 : QUDARapTest.QPropTest                         ***Exception: SegFault         0.03   
      13 /20  Test #13 : QUDARapTest.QSimWithEventTest                 Child aborted***Exception:     1.06   
      18 /20  Test #18 : QUDARapTest.QMultiFilmTest                    ***Exception: SegFault         0.03   

      5  /18  Test #5  : U4Test.U4GDMLReadTest                         Child aborted***Exception:     0.09   
      7  /18  Test #7  : U4Test.U4RandomTest                           ***Exception: SegFault         0.54   
      8  /18  Test #8  : U4Test.U4VolumeMakerTest                      Child aborted***Exception:     0.09   
      13 /18  Test #13 : U4Test.U4TreeTest                             Child aborted***Exception:     0.11   

    [pop           BASH_SOURCE : /Users/blyth/opticks/om.bash 
    [pop              FUNCNAME : om-testenv-dump 
    [pop          OM_KEEP_GEOM :  
    [pop                  GEOM : J004 
    [pop        OPTICKS_T_GEOM : J004 
    


CSG fails
------------

::

    epsilon:CSG blyth$ CSGNodeTest 
    2022-11-12 19:32:25.277 FATAL [54209341] [CSGFoundry::CSGFoundry@90] must SSim::Create before CSGFoundry::CSGFoundry 
    Assertion failed: (sim), function CSGFoundry, file /Users/blyth/opticks/CSG/CSGFoundry.cc, line 91.
    Abort trap: 6
    epsilon:CSG blyth$ 

    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff5c1fcb66 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff5c3c7080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff5c1581ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff5c1201ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x00000001001752e5 libCSG.dylib`CSGFoundry::CSGFoundry(this=0x0000000102002970) at CSGFoundry.cc:91
        frame #5: 0x00000001001755d5 libCSG.dylib`CSGFoundry::CSGFoundry(this=0x0000000102002970) at CSGFoundry.cc:89
        frame #6: 0x00000001001a0ed2 libCSG.dylib`CSGFoundry::Load(base="/Users/blyth/.opticks/GEOM/J004", rel="CSGFoundry") at CSGFoundry.cc:2591
        frame #7: 0x00000001001a0361 libCSG.dylib`CSGFoundry::Load_() at CSGFoundry.cc:2564
        frame #8: 0x00000001001a005d libCSG.dylib`CSGFoundry::Load() at CSGFoundry.cc:2484
        frame #9: 0x000000010002ba8d CSGNodeTest`main(argc=1, argv=0x00007ffeefbfe8e0) at CSGNodeTest.cc:8
        frame #10: 0x00007fff5c0ac015 libdyld.dylib`start + 1
    (lldb) 



CSGFoundryLoadTest.cc::

     06 int main(int argc, char** argv)
      7 {
      8     OPTICKS_LOG(argc, argv);
      9 
     10     SSim::Create() ;
     11 
     12     CSGFoundry* cf = CSGFoundry::Load() ;
     13     LOG(info) << cf->desc() ;
     14 
     15     stree* st = cf->sim->tree ;
     16     LOG(info) << st->desc() ;
     17 
     18     return 0 ;
     19 }



::        

    Target 0: (CSGFoundry_SGeo_SEvt_Test) stopped.
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = EXC_BAD_ACCESS (code=1, address=0xc)
      * frame #0: 0x00000001001657bc libCSG.dylib`qat4::getIdentity(this=0x0000000000000000, ins_idx=0x00007ffeefbfe46c, gas_idx=0x00007ffeefbfe468, sensor_identifier=0x00007ffeefbfe464, sensor_index=0x00007ffeefbfe460) const at sqat4.h:329
        frame #1: 0x000000010020515b libCSG.dylib`CSGTarget::getFrame(this=0x00000001022031f0, fr=0x00007ffeefbfe6a0, inst_idx=39216) const at CSGTarget.cc:147
        frame #2: 0x00000001001a33da libCSG.dylib`CSGFoundry::getFrame(this=0x00000001022034b0, fr=0x00007ffeefbfe6a0, inst_idx=39216) const at CSGFoundry.cc:2944
        frame #3: 0x00000001012b7e18 libSysRap.dylib`SEvt::setFrame(this=0x000000010281b200, ins_idx=39216) at SEvt.cc:342
        frame #4: 0x000000010002b255 CSGFoundry_SGeo_SEvt_Test`main(argc=1, argv=0x00007ffeefbfe8c8) at CSGFoundry_SGeo_SEvt_Test.cc:19
        frame #5: 0x00007fff5c0ac015 libdyld.dylib`start + 1
        frame #6: 0x00007fff5c0ac015 libdyld.dylib`start + 1
    (lldb) 


    (lldb) f 1
    frame #1: 0x000000010020515b libCSG.dylib`CSGTarget::getFrame(this=0x00000001022031f0, fr=0x00007ffeefbfe6a0, inst_idx=39216) const at CSGTarget.cc:147
       144 	    const qat4* _t = foundry->getInst(inst_idx); 
       145 	
       146 	    int ins_idx,  gas_idx, sensor_identifier, sensor_index ;
    -> 147 	    _t->getIdentity(ins_idx,  gas_idx, sensor_identifier, sensor_index );  
       148 	
       149 	    assert( ins_idx == inst_idx ); 
       150 	    fr.set_inst(inst_idx); 
    (lldb) p _t
    (const qat4 *) $0 = 0x0000000000000000
    (lldb) 
        

    
Check again : 9 remaining CSG fails
-----------------------------------------

::

    The following tests FAILED:
         16 - CSGTest.CSGFoundry_SGeo_SEvt_Test (SEGFAULT)
         22 - CSGTest.CSGMakerTest (Child aborted)
         33 - CSGTest.CSGIntersectComparisonTest (Child aborted)
         35 - CSGTest.CSGNodeScanTest (Child aborted)


         37 - CSGTest.CSGGeometryTest (SEGFAULT)
         38 - CSGTest.CSGGeometryFromGeocacheTest (SEGFAULT)
         25 - CSGTest.CSGSimtraceRerunTest (SEGFAULT)  
         26 - CSGTest.CSGSimtraceSampleTest (SEGFAULT)   
         36 - CSGTest.CSGSignedDistanceFieldTest (SEGFAULT)
         PREVENT FAIL WITH PROTECTIONS 


Check again : down to 4 CSG fails
--------------------------------------

::


    The following tests FAILED:
         16 - CSGTest.CSGFoundry_SGeo_SEvt_Test (SEGFAULT)
         22 - CSGTest.CSGMakerTest (Child aborted)
         33 - CSGTest.CSGIntersectComparisonTest (Child aborted)
         35 - CSGTest.CSGNodeScanTest (Child aborted)                   FIXED : GEOM clash  



    epsilon:CSG blyth$ CSGMakerTest 
    2022-11-12 20:44:57.913 INFO  [54313274] [GetNames@18]  names.size 42
    2022-11-12 20:44:57.915 INFO  [54313274] [main@38] JustOrb
    2022-11-12 20:44:57.916 INFO  [54313274] [*CSGMaker::makeSolid11@510] so.label JustOrb so.center_extent ( 0.000, 0.000, 0.000,100.000) 
    2022-11-12 20:44:57.916 INFO  [54313274] [*CSGFoundry::MakeGeom@2383]  so 0x104daf000
    2022-11-12 20:44:57.916 INFO  [54313274] [*CSGFoundry::MakeGeom@2384]  so.desc CSGSolid          JustOrb primNum/Offset     1    0 ce ( 0.000, 0.000, 0.000,100.000) 
    2022-11-12 20:44:57.916 INFO  [54313274] [*CSGFoundry::MakeGeom@2385]  fd.desc CSGFoundry  num_total 1 num_solid 1 num_prim 1 num_node 1 num_plan 0 num_tran 1 num_itra 1 num_inst 1 gas 0 meshname 1 mmlabel 1 mtime 1668285897 mtimestamp 20221112_204457 sim Y
    2022-11-12 20:44:57.916 INFO  [54313274] [main@41] CSGFoundry  num_total 1 num_solid 1 num_prim 1 num_node 1 num_plan 0 num_tran 1 num_itra 1 num_inst 1 gas 0 meshname 1 mmlabel 1 mtime 1668285897 mtimestamp 20221112_204457 sim Y
    2022-11-12 20:44:57.919 INFO  [54313274] [main@38] BoxedSphere
    2022-11-12 20:44:57.919 INFO  [54313274] [*CSGMaker::makeBoxedSphere@300] CSGMaker_makeBoxedSphere_HALFSIDE 100
    2022-11-12 20:44:57.919 INFO  [54313274] [*CSGMaker::makeBoxedSphere@301] CSGMaker_makeBoxedSphere_FACTOR   1
    2022-11-12 20:44:57.919 INFO  [54313274] [*CSGMaker::makeBoxedSphere@325]  so->center_extent ( 0.000, 0.000, 0.000,100.000) 
    2022-11-12 20:44:57.919 INFO  [54313274] [*CSGFoundry::MakeGeom@2383]  so 0x107257000
    2022-11-12 20:44:57.919 INFO  [54313274] [*CSGFoundry::MakeGeom@2384]  so.desc CSGSolid      BoxedSphere primNum/Offset     2    0 ce ( 0.000, 0.000, 0.000,100.000) 
    2022-11-12 20:44:57.919 INFO  [54313274] [*CSGFoundry::MakeGeom@2385]  fd.desc CSGFoundry  num_total 1 num_solid 1 num_prim 2 num_node 2 num_plan 0 num_tran 1 num_itra 1 num_inst 1 gas 0 meshname 1 mmlabel 1 mtime 1668285897 mtimestamp 20221112_204457 sim Y
    2022-11-12 20:44:57.919 INFO  [54313274] [main@41] CSGFoundry  num_total 1 num_solid 1 num_prim 2 num_node 2 num_plan 0 num_tran 1 num_itra 1 num_inst 1 gas 0 meshname 1 mmlabel 1 mtime 1668285897 mtimestamp 20221112_204457 sim Y
    2022-11-12 20:44:57.920 FATAL [54313274] [*CSGFoundry::getMeshName@277]  not in range midx 4294967295 meshname.size()  1
    Assertion failed: (in_range), function getMeshName, file /Users/blyth/opticks/CSG/CSGFoundry.cc, line 278.
    Abort trap: 6
    epsilon:CSG blyth$ 
    epsilon:CSG blyth$ 



       638 	
       639 	void CSGIntersectComparisonTest::loaded()
       640 	{
    -> 641 	    assert( rerun ); 
       642 	    LOG(info) << " rerun " << rerun->sstr() ; 
       643 	    num = rerun->shape[0] ; 
       644 	    for(unsigned i=0 ; i < num ; i++ )
    (lldb) 




Overall down to 25
--------------------

::


    FAILS:  25  / 507   :  Sat Nov 12 20:51:54 2022   
      43 /62  Test #43 : GGeoTest.GGeoTest                             Child aborted***Exception:     1.30   
      44 /62  Test #44 : GGeoTest.GGeoIdentityTest                     Child aborted***Exception:     1.32   
      45 /62  Test #45 : GGeoTest.GGeoConvertTest                      Child aborted***Exception:     1.31   
      60 /62  Test #60 : GGeoTest.GPhoTest                             Child aborted***Exception:     1.37   
      61 /62  Test #61 : GGeoTest.GGeoDumpTest                         Child aborted***Exception:     1.31   

      16 /39  Test #16 : CSGTest.CSGFoundry_SGeo_SEvt_Test             ***Exception: SegFault         0.03   
      22 /39  Test #22 : CSGTest.CSGMakerTest                          Child aborted***Exception:     0.03   
      33 /39  Test #33 : CSGTest.CSGIntersectComparisonTest            Child aborted***Exception:     0.03   

      1  /3   Test #1  : GeoChainTest.GeoChainSolidTest                Child aborted***Exception:     0.11   
      2  /3   Test #2  : GeoChainTest.GeoChainVolumeTest               Child aborted***Exception:     0.15   
      3  /3   Test #3  : GeoChainTest.GeoChainNodeTest                 Child aborted***Exception:     0.10   

      3  /20  Test #3  : QUDARapTest.QScintTest                        ***Exception: SegFault         0.02   
      4  /20  Test #4  : QUDARapTest.QCerenkovIntegralTest             ***Exception: SegFault         0.03   
      5  /20  Test #5  : QUDARapTest.QCerenkovTest                     Child aborted***Exception:     0.03   
      7  /20  Test #7  : QUDARapTest.QSimTest                          Child aborted***Exception:     1.26   
      8  /20  Test #8  : QUDARapTest.QBndTest                          ***Exception: SegFault         0.03   
      9  /20  Test #9  : QUDARapTest.QPrdTest                          ***Exception: SegFault         0.03   
      10 /20  Test #10 : QUDARapTest.QOpticalTest                      ***Exception: SegFault         0.03   
      11 /20  Test #11 : QUDARapTest.QPropTest                         ***Exception: SegFault         0.03   
      13 /20  Test #13 : QUDARapTest.QSimWithEventTest                 Child aborted***Exception:     1.11   
      18 /20  Test #18 : QUDARapTest.QMultiFilmTest                    ***Exception: SegFault         0.03   

      5  /18  Test #5  : U4Test.U4GDMLReadTest                         Child aborted***Exception:     0.09   
      7  /18  Test #7  : U4Test.U4RandomTest                           ***Exception: SegFault         0.54   
      8  /18  Test #8  : U4Test.U4VolumeMakerTest                      Child aborted***Exception:     0.10   
      13 /18  Test #13 : U4Test.U4TreeTest                             Child aborted***Exception:     0.11   
    [pop           BASH_SOURCE : /Users/blyth/opticks/om.bash 
    [pop              FUNCNAME : om-testenv-dump 
        



Look at GGeo fails : ALL FROM SAME ASSERT : FIXED BY SKIPPING THE ASSERT
--------------------------------------------------------------------------

::

    42/62 Test #42: GGeoTest.GGeoLibTest ....................   Passed    0.36 sec
          Start 43: GGeoTest.GGeoTest
    43/62 Test #43: GGeoTest.GGeoTest .......................Child aborted***Exception:   1.43 sec
    Assertion failed: (mm->getParts() == NULL), function deferredCreateGParts, file /Users/blyth/opticks/ggeo/GGeo.cc, line 1617.

          Start 44: GGeoTest.GGeoIdentityTest
    44/62 Test #44: GGeoTest.GGeoIdentityTest ...............Child aborted***Exception:   1.37 sec
    Assertion failed: (mm->getParts() == NULL), function deferredCreateGParts, file /Users/blyth/opticks/ggeo/GGeo.cc, line 1617.

          Start 45: GGeoTest.GGeoConvertTest
    45/62 Test #45: GGeoTest.GGeoConvertTest ................Child aborted***Exception:   1.37 sec
    Assertion failed: (mm->getParts() == NULL), function deferredCreateGParts, file /Users/blyth/opticks/ggeo/GGeo.cc, line 1617.

          Start 46: GGeoTest.GGeoTestTest
    46/62 Test #46: GGeoTest.GGeoTestTest ...................   Passed    0.03 sec





::

    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff70412b66 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff705dd080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff7036e1ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff703361ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x00000001002f618a libGGeo.dylib`GGeo::deferredCreateGParts(this=0x0000000101b00cb0) at GGeo.cc:1617
        frame #5: 0x00000001002f4340 libGGeo.dylib`GGeo::deferred(this=0x0000000101b00cb0) at GGeo.cc:655
        frame #6: 0x00000001002f3cab libGGeo.dylib`GGeo::postLoadFromCache(this=0x0000000101b00cb0) at GGeo.cc:605
        frame #7: 0x00000001002efa95 libGGeo.dylib`GGeo::loadFromCache(this=0x0000000101b00cb0) at GGeo.cc:586
        frame #8: 0x00000001002ef649 libGGeo.dylib`GGeo::Load(ok=0x00007ffeefbfe670) at GGeo.cc:134
        frame #9: 0x000000010000a099 GGeoTest`main(argc=1, argv=0x00007ffeefbfe8e8) at GGeoTest.cc:426
        frame #10: 0x00007fff702c2015 libdyld.dylib`start + 1
        frame #11: 0x00007fff702c2015 libdyld.dylib`start + 1
    (lldb) 

    (lldb) f 7
    frame #7: 0x00000001002efa95 libGGeo.dylib`GGeo::loadFromCache(this=0x0000000101b00cb0) at GGeo.cc:586
       583 	
       584 	    m_meshlib->setGGeoLib(m_geolib); 
       585 	
    -> 586 	    postLoadFromCache(); 
       587 	
       588 	    LOG(LEVEL) << "]" ; 
       589 	}
    (lldb) p m_ok->getIdPath()
    (const char *) $0 = 0x0000000101a019a0 "/Users/blyth/.opticks/GEOM/J004/GGeo"
    (lldb) 

    (lldb) f 4
    frame #4: 0x00000001002f618a libGGeo.dylib`GGeo::deferredCreateGParts(this=0x0000000101b00cb0) at GGeo.cc:1617
       1614	    for(unsigned i=0 ; i < nmm ; i++)
       1615	    {
       1616	        GMergedMesh* mm = m_geolib->getMergedMesh(i);
    -> 1617	        assert( mm->getParts() == NULL ); 
       1618	
       1619	        GPts* pts = mm->getPts(); 
       1620	        LOG_IF(fatal, pts == nullptr ) << " pts NULL, cannot create GParts for mm " << i ; 
    (lldb) 



Look at CSG Fails
--------------------


::

      16 /39  Test #16 : CSGTest.CSGFoundry_SGeo_SEvt_Test             ***Exception: SegFault         0.03   
                 AVOIDED BY BAIL OUT WHEN SEvt::Load GIVES EMPTY 

      22 /39  Test #22 : CSGTest.CSGMakerTest                          Child aborted***Exception:     0.03   

      33 /39  Test #33 : CSGTest.CSGIntersectComparisonTest            Child aborted***Exception:     0.03   
                 WAS EXPECTING A FAIL : AVOIDED ASSERT FOR THIS


CSGFoundry_SGeo_SEvt_Test : needs some protection over frame index existing ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  

::


    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = EXC_BAD_ACCESS (code=1, address=0xc)
      * frame #0: 0x00000001001693dc libCSG.dylib`qat4::getIdentity(this=0x0000000000000000, ins_idx=0x00007ffeefbfe44c, gas_idx=0x00007ffeefbfe448, sensor_identifier=0x00007ffeefbfe444, sensor_index=0x00007ffeefbfe440) const at sqat4.h:329
        frame #1: 0x000000010020a88b libCSG.dylib`CSGTarget::getFrame(this=0x00000001021030e0, fr=0x00007ffeefbfe680, inst_idx=39216) const at CSGTarget.cc:147
        frame #2: 0x00000001001a6fda libCSG.dylib`CSGFoundry::getFrame(this=0x0000000102103470, fr=0x00007ffeefbfe680, inst_idx=39216) const at CSGFoundry.cc:2944
        frame #3: 0x00000001012d08c8 libSysRap.dylib`SEvt::setFrame(this=0x000000010281b200, ins_idx=39216) at SEvt.cc:428
        frame #4: 0x000000010002c4b5 CSGFoundry_SGeo_SEvt_Test`main(argc=1, argv=0x00007ffeefbfe8a8) at CSGFoundry_SGeo_SEvt_Test.cc:19
        frame #5: 0x00007fff702c2015 libdyld.dylib`start + 1
        frame #6: 0x00007fff702c2015 libdyld.dylib`start + 1
    (lldb) f 4
    frame #4: 0x000000010002c4b5 CSGFoundry_SGeo_SEvt_Test`main(argc=1, argv=0x00007ffeefbfe8a8) at CSGFoundry_SGeo_SEvt_Test.cc:19
       16  	    sev->setGeo(fd); 
       17  	
       18  	    int ins_idx = SSys::getenvint("INS_IDX", 39216) ;
    -> 19  	    if( ins_idx >= 0 ) sev->setFrame(ins_idx); 
       20  	    std::cout << sev->descFull() ; 
       21  	  
       22  	    return 0 ; 
    (lldb) p ins_idx
    (int) $0 = 39216
    (lldb) 

    (lldb) f 3
    frame #3: 0x00000001012d08c8 libSysRap.dylib`SEvt::setFrame(this=0x000000010281b200, ins_idx=39216) at SEvt.cc:428
       425 	    LOG_IF(fatal, cf == nullptr) << "must SEvt::setGeo before being can access frames " ; 
       426 	    assert(cf); 
       427 	    sframe fr ; 
    -> 428 	    int rc = cf->getFrame(fr, ins_idx) ; 
       429 	    assert( rc == 0 );  
       430 	    fr.prepare();     
       431 	
    (lldb) f 2
    frame #2: 0x00000001001a6fda libCSG.dylib`CSGFoundry::getFrame(this=0x0000000102103470, fr=0x00007ffeefbfe680, inst_idx=39216) const at CSGFoundry.cc:2944
       2941	
       2942	int CSGFoundry::getFrame(sframe& fr, int inst_idx) const
       2943	{
    -> 2944	    return target->getFrame( fr, inst_idx ); 
       2945	}
       2946	
       2947	
    (lldb) f 1
    frame #1: 0x000000010020a88b libCSG.dylib`CSGTarget::getFrame(this=0x00000001021030e0, fr=0x00007ffeefbfe680, inst_idx=39216) const at CSGTarget.cc:147
       144 	    const qat4* _t = foundry->getInst(inst_idx); 
       145 	
       146 	    int ins_idx,  gas_idx, sensor_identifier, sensor_index ;
    -> 147 	    _t->getIdentity(ins_idx,  gas_idx, sensor_identifier, sensor_index );  
       148 	
       149 	    assert( ins_idx == inst_idx ); 
       150 	    fr.set_inst(inst_idx); 
    (lldb) p _t 
    (const qat4 *) $1 = 0x0000000000000000
    (lldb) 




CSGMakerTest failing to persist
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff70412b66 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff705dd080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff7036e1ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff703361ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x000000010017d765 libCSG.dylib`CSGFoundry::getMeshName(this=0x0000000102001870, midx=4294967295) const at CSGFoundry.cc:278
        frame #5: 0x000000010017cf03 libCSG.dylib`CSGFoundry::getPrimName(this=0x0000000102001870, pname=size=0) const at CSGFoundry.cc:264
        frame #6: 0x000000010019ac80 libCSG.dylib`CSGFoundry::save_(this=0x0000000102001870, dir_="/tmp/blyth/opticks/GEOM/CSGFoundry") const at CSGFoundry.cc:2086
        frame #7: 0x000000010019a973 libCSG.dylib`CSGFoundry::save(this=0x0000000102001870, base="/tmp/blyth/opticks/GEOM", rel="CSGFoundry") const at CSGFoundry.cc:2033
        frame #8: 0x000000010019a1fa libCSG.dylib`CSGFoundry::save(this=0x0000000102001870) const at CSGFoundry.cc:2020
        frame #9: 0x000000010002d1f5 CSGMakerTest`main(argc=1, argv=0x00007ffeefbfe8c0) at CSGMakerTest.cc:43
        frame #10: 0x00007fff702c2015 libdyld.dylib`start + 1
    (lldb) 


    (lldb) p cfbase
    (const char *) $1 = 0x00000001020016f0 "/tmp/blyth/opticks/GEOM"
    (lldb) f 7
    frame #7: 0x000000010019a973 libCSG.dylib`CSGFoundry::save(this=0x0000000102001870, base="/tmp/blyth/opticks/GEOM", rel="CSGFoundry") const at CSGFoundry.cc:2033
       2030	    std::stringstream ss ;   
       2031	    ss << base << "/" << rel ; 
       2032	    std::string dir = ss.str();   
    -> 2033	    save_(dir.c_str()); 
       2034	}
       2035	
       2036	/**
    (lldb) f 6
    frame #6: 0x000000010019ac80 libCSG.dylib`CSGFoundry::save_(this=0x0000000102001870, dir_="/tmp/blyth/opticks/GEOM/CSGFoundry") const at CSGFoundry.cc:2086
       2083	    if(meshname.size() > 0 ) NP::WriteNames( dir, "meshname.txt", meshname );
       2084	
       2085	    std::vector<std::string> primname ; 
    -> 2086	    getPrimName(primname); 
       2087	    if(primname.size() > 0 ) NP::WriteNames( dir, "primname.txt", primname );
       2088	
       2089	    if(mmlabel.size() > 0 )  NP::WriteNames( dir, "mmlabel.txt", mmlabel );
    (lldb) 

    (lldb) f 5
    frame #5: 0x000000010017cf03 libCSG.dylib`CSGFoundry::getPrimName(this=0x0000000102001870, pname=size=0) const at CSGFoundry.cc:264
       261 	    {
       262 	        const CSGPrim& pr = prim[i] ; 
       263 	        unsigned midx = num_prim == 1 ? 0 : pr.meshIdx();  // kludge avoid out-of-range for single prim CSGFoundry
    -> 264 	        const std::string& mname = getMeshName(midx); 
       265 	        LOG(debug) << " primIdx " << std::setw(4) << i << " midx " << midx << " mname " << mname  ;  
       266 	        pname.push_back(mname);  
       267 	    }
    (lldb) p num_prim
    (unsigned int) $2 = 2

    ## HUH: would have expected 1 ?

    (lldb) p midx
    (unsigned int) $3 = 4294967295
    (lldb) p ~0
    (int) $4 = -1
    (lldb) p unsigned(~0)
    (unsigned int) $5 = 4294967295
    (lldb) 


CSGMakerTest is stomping on standard geometry folder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Hmm: problem is this test assumes GEOM envvar to control the 
name and output dir but that is not practical so had changed
to CSGMakerTest_GEOM in the test but the GEOM machinery 
controlling the default directory is just ignoring that envvar. 

* HMM: the default GEOM access machinery could always check
  first for ExecutableName_GEOM before looking at GEOM : so 
  tests that need to effectively set GEOM can do so without 
  causing problems for standard GEOM access

::

    1993 const char* CSGFoundry::BASE = "$DefaultGeometryDir" ; // incorporates GEOM if defined
    1994 const char* CSGFoundry::RELDIR = "CSGFoundry"  ;
    1995 
    1996 /**
    1997 CSGFoundry::getBaseDir
    1998 -------------------------
    1999 
    2000 Returns value of CFBASE envvar if defined, otherwise resolves '$DefaultOutputDir' which 
    2001 is for example /tmp/$USER/opticks/$GEOM/SProc::ExecutableName
    2002 
    2003 **/
    2004 
    2005 const char* CSGFoundry::getBaseDir(bool create) const
    2006 {
    2007     const char* cfbase_default = SPath::Resolve(BASE, create ? DIRPATH : NOOP );  //   
    2008     const char* cfbase = SSys::getenvvar("CFBASE", cfbase_default );
    2009     return cfbase ? strdup(cfbase) : nullptr ;
    2010 }

::

    143 const char* SOpticksResource::DefaultOutputDir()
    144 {
    145     return SPath::Resolve("$TMP/GEOM", SSys::getenvvar("GEOM"), ExecutableName(), NOOP);
    146 }
    147 const char* SOpticksResource::DefaultGeometryDir()
    148 {
    149     return SPath::Resolve("$TMP/GEOM", SSys::getenvvar("GEOM"), NOOP);
    150 }
    151 const char* SOpticksResource::DefaultGeometryBase()
    152 {
    153     return SPath::Resolve("$TMP/GEOM", NOOP);
    154 }
         


    epsilon:opticks blyth$ opticks-f "SSys::getenvvar(\"GEOM"
    ./CSG/CSGSimtrace.cc:    geom(SSys::getenvvar("GEOM", "nmskSolidMaskTail")),  
    ./CSG/tests/CSGMakerTest.cc:     const char* geom = SSys::getenvvar("GEOM", nullptr ); 
    ./CSG/tests/CSGDemoTest.cc:    const char* geom = SSys::getenvvar("GEOM", "sphere" ); 
    ./CSG/CSGFoundry.cc:    if(geom == nullptr) geom = SSys::getenvvar("GEOM", "GeneralSphereDEV") ; 
    ./GeoChain/tests/GeoChainNodeTest.cc:    const char* name = SSys::getenvvar("GEOM", "sphere" ); 
    ./GeoChain/tests/GeoChainVolumeTest.cc:    const char* name = SSys::getenvvar("GEOM", name_default ); 
    ./GeoChain/tests/GeoChainSolidTest.cc:    const char* geom_ = SSys::getenvvar("GEOM", "AdditionAcrylicConstruction" ) ; 
    ./g4ok/tests/G4OKVolumeTest.cc:    const char* geom = SSys::getenvvar("GEOM", geom_default );  
    ./sysrap/SOpticksResource.cc:    return SPath::Resolve("$TMP/GEOM", SSys::getenvvar("GEOM"), ExecutableName(), NOOP); 
    ./sysrap/SOpticksResource.cc:    return SPath::Resolve("$TMP/GEOM", SSys::getenvvar("GEOM"), NOOP); 
    ./sysrap/SOpticksResource.cc:    const char* GEOM = SSys::getenvvar("GEOM") ; 
    ./sysrap/SOpticksResource.cc:    const char* geom = SSys::getenvvar("GEOM"); 
    ./sysrap/SOpticksResource.cc:    const char* geom = SSys::getenvvar("GEOM"); 
    ./sysrap/SOpticksResource.cc:    const char* geom = SSys::getenvvar("GEOM"); 
    ./u4/U4VolumeMaker.cc:const char* U4VolumeMaker::GEOM = SSys::getenvvar("GEOM", "BoxOfScintillator"); 
    epsilon:opticks blyth$ 



::

    epsilon:tests blyth$ l /tmp/blyth/opticks/GEOM/J004/CSGFoundry/
    total 80
    8 -rw-r--r--   1 blyth  wheel  192 Nov 27 16:23 inst.npy
    8 -rw-r--r--   1 blyth  wheel  192 Nov 27 16:23 itra.npy
    8 -rw-r--r--   1 blyth  wheel  192 Nov 27 16:23 tran.npy
    8 -rw-r--r--   1 blyth  wheel  192 Nov 27 16:23 node.npy
    8 -rw-r--r--   1 blyth  wheel  192 Nov 27 16:23 prim.npy
    8 -rw-r--r--   1 blyth  wheel  176 Nov 27 16:23 solid.npy
    8 -rw-r--r--   1 blyth  wheel  130 Nov 27 16:23 meta.txt
    8 -rw-r--r--   1 blyth  wheel    8 Nov 27 16:23 mmlabel.txt
    8 -rw-r--r--   1 blyth  wheel    8 Nov 27 16:23 primname.txt
    8 -rw-r--r--   1 blyth  wheel    8 Nov 27 16:23 meshname.txt
    0 drwxr-xr-x   5 blyth  wheel  160 Nov 27 14:35 ..
    0 drwxr-xr-x  12 blyth  wheel  384 Nov 27 14:35 .
    epsilon:tests blyth$ date
    Sun Nov 27 16:23:25 GMT 2022
    epsilon:tests blyth$ cat /tmp/blyth/opticks/GEOM/J004/CSGFoundry/meshname.txt
    JustOrb
    epsilon:tests blyth$ 




Down to 18/509 on laptop
---------------------------


::

    FAILS:  18  / 509   :  Sun Nov 27 17:45:03 2022   
      40 /42  Test #40 : ExtG4Test.X4IntersectVolumeTest               Child aborted***Exception:     0.16   
             LACK OF nnvtBodyPhys with PMTFastSim, fixed by switch to hamaBodyPhys

      1  /3   Test #1  : GeoChainTest.GeoChainSolidTest                Child aborted***Exception:     0.11   
      2  /3   Test #2  : GeoChainTest.GeoChainVolumeTest               Child aborted***Exception:     0.15   
      3  /3   Test #3  : GeoChainTest.GeoChainNodeTest                 Child aborted***Exception:     0.10   
            FIXED BY CHANGE TO ExecutableName_GEOM envvars  

      3  /20  Test #3  : QUDARapTest.QScintTest                        ***Exception: SegFault         0.02   
      4  /20  Test #4  : QUDARapTest.QCerenkovIntegralTest             ***Exception: SegFault         0.03   
      5  /20  Test #5  : QUDARapTest.QCerenkovTest                     Child aborted***Exception:     0.03   
      7  /20  Test #7  : QUDARapTest.QSimTest                          Child aborted***Exception:     1.47   
      8  /20  Test #8  : QUDARapTest.QBndTest                          ***Exception: SegFault         0.03   
      9  /20  Test #9  : QUDARapTest.QPrdTest                          ***Exception: SegFault         0.03   
      10 /20  Test #10 : QUDARapTest.QOpticalTest                      ***Exception: SegFault         0.03   
      11 /20  Test #11 : QUDARapTest.QPropTest                         ***Exception: SegFault         0.03   
      13 /20  Test #13 : QUDARapTest.QSimWithEventTest                 Child aborted***Exception:     1.12   
      18 /20  Test #18 : QUDARapTest.QMultiFilmTest                    ***Exception: SegFault         0.02   

      5  /19  Test #5  : U4Test.U4GDMLReadTest                         Child aborted***Exception:     0.09   
               FIXED BY ADDING J004_GDMLPathFromGEOM envvar

      7  /19  Test #7  : U4Test.U4RandomTest                           ***Exception: SegFault         0.62   
               SKIP CHECK WHEN NO SEvt

      9  /19  Test #9  : U4Test.U4VolumeMakerTest                      Child aborted***Exception:     0.09   
               

      14 /19  Test #14 : U4Test.U4TreeTest                             Child aborted***Exception:     0.11   

    [pop           BASH_SOURCE : /Users/blyth/opticks/om.bash 



u4 fails
------------


::

    test_basics@16:  rand.dump after U4Random::setSequenceIndex(0) : USING PRECOOKED RANDOMS 
    ...
     i     8 u    0.14407
     i     9 u    0.18780
    ...
       1341	int SEvt::getTagSlot() const 
       1342	{
    -> 1343	    if(evt->tag == nullptr) return -1 ; 
       1344	    const stagr& tagr = current_ctx.tagr ; 
       1345	    return tagr.slot ; 
       1346	}
    Target 0: (U4RandomTest) stopped.
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = EXC_BAD_ACCESS (code=1, address=0x20)
      * frame #0: 0x00000001062d445c libSysRap.dylib`SEvt::getTagSlot(this=0x0000000000000000) const at SEvt.cc:1343
        frame #1: 0x00000001062d4440 libSysRap.dylib`SEvt::GetTagSlot() at SEvt.cc:529
        frame #2: 0x00000001001c0f77 libU4.dylib`U4Random::check_cursor_vs_tagslot(this=0x000000010781b760) at U4Random.cc:494
        frame #3: 0x00000001001be0bc libU4.dylib`U4Random::setSequenceIndex(this=0x000000010781b760, index_=-1) at U4Random.cc:287
        frame #4: 0x000000010000bc18 U4RandomTest`test_basics(rnd=0x000000010781b760) at U4RandomTest.cc:18
        frame #5: 0x000000010000c340 U4RandomTest`main(argc=1, argv=0x00007ffeefbfe7a0) at U4RandomTest.cc:30
        frame #6: 0x00007fff702c2015 libdyld.dylib`start + 1
    (lldb) 



