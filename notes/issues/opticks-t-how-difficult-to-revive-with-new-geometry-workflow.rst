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





