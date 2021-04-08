cluster-opticks-t-shakedown
==============================


Revist : April 8th
-----------------------

::

    epsilon:~ blyth$ ssh L7    # NOPE USE THE L7 FUNCTION TO SET TERM and hence PS1

    -bash-4.2$ sj              # check the what srun will do 

    -bash-4.2$ sr              # srun 
    === gpujob-setup: blyth
    ...



    SLOW: tests taking longer that 15 seconds
      30 /56  Test #30 : GGeoTest.GPtsTest                             Passed                         36.98  


    FAILS:  9   / 453   :  Thu Apr  8 20:41:15 2021   
      22 /32  Test #22 : OptiXRapTest.interpolationTest                ***Failed                      5.38   
      3  /38  Test #3  : CFG4Test.CTestDetectorTest                    Subprocess aborted***Exception:   8.76   
      5  /38  Test #5  : CFG4Test.CGDMLDetectorTest                    Subprocess aborted***Exception:   8.94   
      7  /38  Test #7  : CFG4Test.CGeometryTest                        Subprocess aborted***Exception:   8.04   
      8  /38  Test #8  : CFG4Test.CG4Test                              Subprocess aborted***Exception:   8.45   
      26 /38  Test #26 : CFG4Test.CInterpolationTest                   Subprocess aborted***Exception:   8.05   
      32 /38  Test #32 : CFG4Test.CRandomEngineTest                    Subprocess aborted***Exception:   8.30   
      1  /1   Test #1  : OKG4Test.OKG4Test                             Subprocess aborted***Exception:   9.07   
      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      0.23   
    drwxr-xr-x  3 blyth  dyw          21 Apr  8 20:35 blyth
    gpujob-tail : rc 0
    -bash-4.2$ 


::

    L7[blyth@lxslc711 gpujob]$ t sr
    sr () 
    { 
        srun --partition=gpu --account=junogpu --gres=gpu:v100:1 $(job)
    }
    L7[blyth@lxslc711 gpujob]$ job
    /hpcfs/juno/junogpu/blyth/j/gpujob.sh
    L7[blyth@lxslc711 gpujob]$ 


    #!/bin/bash

    #SBATCH --partition=gpu
    #SBATCH --qos=debug
    #SBATCH --account=junogpu
    #SBATCH --job-name=gpujob
    #SBATCH --ntasks=1
    #SBATCH --output=/hpcfs/juno/junogpu/blyth/gpujob/%j.out
    #SBATCH --error=/hpcfs/juno/junogpu/blyth/gpujob/%j.err
    #SBATCH --mem-per-cpu=20480
    #SBATCH --gres=gpu:v100:1

    tds(){ 
        local opts="--opticks-mode 1 --no-guide_tube --pmt20inch-polycone-neck --pmt20inch-simplify-csg --evtmax 10"
        tds- $opts
    }
    tds0(){ 
        : run with opticks disabled
        local opts="--opticks-mode 0 --no-guide_tube --pmt20inch-polycone-neck --pmt20inch-simplify-csg --evtmax 10"
        tds- $opts
    }
    tds-label(){
        local label="tds";
        local arg;
        for arg in $*;
        do
            case $arg in 
                --no-guide_tube)           label="${label}_ngt"  ;;
                --pmt20inch-polycone-neck) label="${label}_pcnk" ;;
                --pmt20inch-simplify-csg)  label="${label}_sycg" ;;
            esac;
        done
        echo $label 
    }

    tds-(){ 
        local msg="=== $FUNCNAME :"
        local label=$(tds-label $*)
        local dbggdmlpath="$HOME/${label}_202103.gdml"
        echo $msg label $label dbggdmlpath $dbggdmlpath;
        export OPTICKS_EMBEDDED_COMMANDLINE_EXTRA="--dbggdmlpath $dbggdmlpath"
        local script=$JUNOTOP/offline/Examples/Tutorial/share/tut_detsim.py;
        local args="gun";
        local iwd=$PWD;
        local dir=$HOME/tds;
        mkdir -p $dir;
        cd $dir;
        local runline="python $script $* $args ";
        echo $runline;
        date;
        eval $runline;
        date;
        cd $iwd
    }

    gpujob-setup()
    {
       local msg="=== $FUNCNAME:"
       echo $msg $USER
       export JUNOTOP=/hpcfs/juno/junogpu/blyth/junotop
       export HOME=/hpcfs/juno/junogpu/blyth   # avoid /afs and control where to put .opticks/rngcache/RNG/

       source $JUNOTOP/bashrc.sh
       source $JUNOTOP/sniper/SniperRelease/cmt/setup.sh
       source $JUNOTOP/offline/JunoRelease/cmt/setup.sh
       mkdir -p /hpcfs/juno/junogpu/blyth/gpujob
       [ -z "$OPTICKS_PREFIX" ] && echo $msg MISSING OPTICKS_PREFIX && return 1
       opticks-(){ . $JUNOTOP/opticks/opticks.bash && opticks-env  ; } 
       opticks-
       env | grep OPTICKS_
       env | grep TMP
    }

    gpujob-head(){ 
       hostname 
       nvidia-smi   
       opticks-info
       opticks-paths
       #UseOptiX  TODO:use an always built executable instead of this optional one
    }
    gpujob-body()
    {
       #opticks-full-prepare  # create rngcache files
       #tds0
       #tds
       opticks-t
    }
    gpujob-tail(){
       local rc=$?    # capture the return code of prior command
       echo $FUNCNAME : rc $rc              
    }

    gpujob-setup
    gpujob-head
    gpujob-body
    gpujob-tail




::

    SLOW: tests taking longer that 15 seconds


    FAILS:  88  / 453   :  Wed Mar 24 20:01:35 2021   
      46 /55  Test #46 : SysRapTest.SPPMTest                           ***Exception: SegFault         0.38   
      15 /116 Test #15 : NPYTest.ImageNPYTest                          Subprocess aborted***Exception:   0.10   
      16 /116 Test #16 : NPYTest.ImageNPYConcatTest                    Subprocess aborted***Exception:   0.11   
               needs tmp folder


      2  /43  Test #2  : OpticksCoreTest.IndexerTest                   Subprocess aborted***Exception:   0.22   



      8  /43  Test #8  : OpticksCoreTest.OpticksFlagsTest              Subprocess aborted***Exception:   0.14   
      10 /43  Test #10 : OpticksCoreTest.OpticksColorsTest             Subprocess aborted***Exception:   0.13   
      13 /43  Test #13 : OpticksCoreTest.OpticksCfg2Test               Subprocess aborted***Exception:   0.13   
      14 /43  Test #14 : OpticksCoreTest.OpticksTest                   Subprocess aborted***Exception:   0.15   
      15 /43  Test #15 : OpticksCoreTest.OpticksTwoTest                Subprocess aborted***Exception:   0.11   
      16 /43  Test #16 : OpticksCoreTest.OpticksResourceTest           Subprocess aborted***Exception:   0.13   
      21 /43  Test #21 : OpticksCoreTest.OK_PROFILE_Test               Subprocess aborted***Exception:   0.09   
      22 /43  Test #22 : OpticksCoreTest.OpticksAnaTest                Subprocess aborted***Exception:   0.15   
      23 /43  Test #23 : OpticksCoreTest.OpticksDbgTest                Subprocess aborted***Exception:   0.11   
      25 /43  Test #25 : OpticksCoreTest.CompositionTest               Subprocess aborted***Exception:   0.12   
      28 /43  Test #28 : OpticksCoreTest.EvtLoadTest                   Subprocess aborted***Exception:   0.10   
      29 /43  Test #29 : OpticksCoreTest.OpticksEventAnaTest           Subprocess aborted***Exception:   0.15   
      30 /43  Test #30 : OpticksCoreTest.OpticksEventCompareTest       Subprocess aborted***Exception:   0.11   
      31 /43  Test #31 : OpticksCoreTest.OpticksEventDumpTest          Subprocess aborted***Exception:   0.13   
      37 /43  Test #37 : OpticksCoreTest.CfgTest                       Subprocess aborted***Exception:   0.12   
      41 /43  Test #41 : OpticksCoreTest.OpticksEventTest              Subprocess aborted***Exception:   0.14   
      42 /43  Test #42 : OpticksCoreTest.OpticksEventLeakTest          Subprocess aborted***Exception:   0.13   
      43 /43  Test #43 : OpticksCoreTest.OpticksRunTest                Subprocess aborted***Exception:   0.13   
      13 /56  Test #13 : GGeoTest.GScintillatorLibTest                 Subprocess aborted***Exception:   0.11   
      15 /56  Test #15 : GGeoTest.GSourceLibTest                       Subprocess aborted***Exception:   0.11   
      16 /56  Test #16 : GGeoTest.GBndLibTest                          Subprocess aborted***Exception:   0.10   
      17 /56  Test #17 : GGeoTest.GBndLibInitTest                      Subprocess aborted***Exception:   0.12   
      26 /56  Test #26 : GGeoTest.GItemIndex2Test                      Subprocess aborted***Exception:   0.08   
      30 /56  Test #30 : GGeoTest.GPtsTest                             Subprocess aborted***Exception:   0.15   
      34 /56  Test #34 : GGeoTest.BoundariesNPYTest                    Subprocess aborted***Exception:   0.12   
      35 /56  Test #35 : GGeoTest.GAttrSeqTest                         Subprocess aborted***Exception:   0.10   
      36 /56  Test #36 : GGeoTest.GBBoxMeshTest                        Subprocess aborted***Exception:   0.08   
      38 /56  Test #38 : GGeoTest.GFlagsTest                           Subprocess aborted***Exception:   0.13   
      39 /56  Test #39 : GGeoTest.GGeoLibTest                          Subprocess aborted***Exception:   0.16   
      40 /56  Test #40 : GGeoTest.GGeoTest                             Subprocess aborted***Exception:   0.13   
      41 /56  Test #41 : GGeoTest.GGeoIdentityTest                     Subprocess aborted***Exception:   0.12   
      42 /56  Test #42 : GGeoTest.GGeoConvertTest                      Subprocess aborted***Exception:   0.13   
      43 /56  Test #43 : GGeoTest.GGeoTestTest                         Subprocess aborted***Exception:   0.12   
      44 /56  Test #44 : GGeoTest.GMakerTest                           Subprocess aborted***Exception:   0.12   
      45 /56  Test #45 : GGeoTest.GMergedMeshTest                      Subprocess aborted***Exception:   0.14   
      51 /56  Test #51 : GGeoTest.GSurfaceLibTest                      Subprocess aborted***Exception:   0.11   
      53 /56  Test #53 : GGeoTest.RecordsNPYTest                       Subprocess aborted***Exception:   0.11   
      54 /56  Test #54 : GGeoTest.GMeshLibTest                         Subprocess aborted***Exception:   0.11   
      55 /56  Test #55 : GGeoTest.GNodeLibTest                         Subprocess aborted***Exception:   0.62   
      56 /56  Test #56 : GGeoTest.GPhoTest                             Subprocess aborted***Exception:   0.12   
      1  /3   Test #1  : OpticksGeoTest.OpticksGeoTest                 Subprocess aborted***Exception:   0.30   
      2  /3   Test #2  : OpticksGeoTest.OpticksHubTest                 Subprocess aborted***Exception:   0.09   
      3  /3   Test #3  : OpticksGeoTest.OpticksHubGGeoTest             Subprocess aborted***Exception:   0.14   
      2  /32  Test #2  : OptiXRapTest.OContextCreateTest               Subprocess aborted***Exception:   0.30   
      3  /32  Test #3  : OptiXRapTest.OScintillatorLibTest             Subprocess aborted***Exception:   0.28   
      4  /32  Test #4  : OptiXRapTest.LTOOContextUploadDownloadTest    Subprocess aborted***Exception:   0.25   
      9  /32  Test #9  : OptiXRapTest.bufferTest                       Subprocess aborted***Exception:   0.41   
      10 /32  Test #10 : OptiXRapTest.textureTest                      Subprocess aborted***Exception:   0.50   
      11 /32  Test #11 : OptiXRapTest.boundaryTest                     Subprocess aborted***Exception:   0.27   
      12 /32  Test #12 : OptiXRapTest.boundaryLookupTest               Subprocess aborted***Exception:   0.24   
      16 /32  Test #16 : OptiXRapTest.rayleighTest                     Subprocess aborted***Exception:   0.26   
      17 /32  Test #17 : OptiXRapTest.writeBufferTest                  Subprocess aborted***Exception:   0.21   
      20 /32  Test #20 : OptiXRapTest.downloadTest                     Subprocess aborted***Exception:   0.18   
      21 /32  Test #21 : OptiXRapTest.eventTest                        Subprocess aborted***Exception:   0.22   
      22 /32  Test #22 : OptiXRapTest.interpolationTest                Subprocess aborted***Exception:   0.26   
      23 /32  Test #23 : OptiXRapTest.ORngTest                         Subprocess aborted***Exception:   0.22   
      1  /5   Test #1  : OKOPTest.OpIndexerTest                        Subprocess aborted***Exception:   0.46   
      2  /5   Test #2  : OKOPTest.OpSeederTest                         Subprocess aborted***Exception:   0.23   
      3  /5   Test #3  : OKOPTest.dirtyBufferTest                      Subprocess aborted***Exception:   0.22   
      4  /5   Test #4  : OKOPTest.compactionTest                       Subprocess aborted***Exception:   0.29   
      5  /5   Test #5  : OKOPTest.OpSnapTest                           Subprocess aborted***Exception:   0.23   
      2  /5   Test #2  : OKTest.OKTest                                 Subprocess aborted***Exception:   0.22   
      3  /5   Test #3  : OKTest.OTracerTest                            Subprocess aborted***Exception:   0.22   
      5  /5   Test #5  : OKTest.TrivialTest                            Subprocess aborted***Exception:   0.21   
      3  /25  Test #3  : ExtG4Test.X4SolidTest                         Subprocess aborted***Exception:   0.21   
      10 /25  Test #10 : ExtG4Test.X4MaterialTableTest                 Subprocess aborted***Exception:   0.18   
      16 /25  Test #16 : ExtG4Test.X4CSGTest                           Subprocess aborted***Exception:   0.18   
      18 /25  Test #18 : ExtG4Test.X4GDMLParserTest                    Subprocess aborted***Exception:   0.29   
      19 /25  Test #19 : ExtG4Test.X4GDMLBalanceTest                   Subprocess aborted***Exception:   0.26   
      1  /38  Test #1  : CFG4Test.CMaterialLibTest                     Subprocess aborted***Exception:   0.71   
      2  /38  Test #2  : CFG4Test.CMaterialTest                        Subprocess aborted***Exception:   0.30   
      3  /38  Test #3  : CFG4Test.CTestDetectorTest                    Subprocess aborted***Exception:   0.28   
      5  /38  Test #5  : CFG4Test.CGDMLDetectorTest                    Subprocess aborted***Exception:   0.27   
      7  /38  Test #7  : CFG4Test.CGeometryTest                        Subprocess aborted***Exception:   0.30   
      8  /38  Test #8  : CFG4Test.CG4Test                              Subprocess aborted***Exception:   0.30   
      26 /38  Test #26 : CFG4Test.CInterpolationTest                   Subprocess aborted***Exception:   0.32   
      28 /38  Test #28 : CFG4Test.CGROUPVELTest                        Subprocess aborted***Exception:   0.36   
      31 /38  Test #31 : CFG4Test.CPhotonTest                          Subprocess aborted***Exception:   0.29   
      32 /38  Test #32 : CFG4Test.CRandomEngineTest                    Subprocess aborted***Exception:   0.31   
      35 /38  Test #35 : CFG4Test.CCerenkovGeneratorTest               Subprocess aborted***Exception:   0.34   
      36 /38  Test #36 : CFG4Test.CGenstepSourceTest                   Subprocess aborted***Exception:   0.31   
      1  /1   Test #1  : OKG4Test.OKG4Test                             Subprocess aborted***Exception:   0.75   
      1  /2   Test #1  : G4OKTest.G4OKTest                             Subprocess aborted***Exception:   0.47   
      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      0.23   
    gpujob-tail : rc 0
    L7[blyth@lxslc716 ~]$ 




Sort out TMP
----------------

* added creation of TMP OPTICKS_TMP OPTICKS_EVENT_BASE dirs to opticks-setup 
  so they get created on sourcing opticks-setup.sh 


Errors from lack of TMP dir::



    46/55 Test #46: SysRapTest.SPPMTest .......................***Exception: SegFault  0.38 sec
    2021-03-24 20:00:01.586 INFO  [253731] [test_MakeTestImage@18]  path /tmp/blyth/opticks/SPPMTest_MakeTestImage.ppm width 1024 height 512 size 1572864 yflip 1 config vertical_gradient


     14/116 Test  #14: NPYTest.NGridTest ......................   Passed    0.07 sec
            Start  15: NPYTest.ImageNPYTest
     15/116 Test  #15: NPYTest.ImageNPYTest ...................Subprocess aborted***Exception:   0.10 sec
    2021-03-24 20:00:08.987 INFO  [255504] [main@94]  load ipath /tmp/blyth/opticks/SPPMTest.ppm
    2021-03-24 20:00:08.989 INFO  [255504] [test_LoadPPM@60]  path /tmp/blyth/opticks/SPPMTest.ppm yflip 0 ncomp 3 config add_border,add_midline,add_quadline
    2021-03-24 20:00:08.989 FATAL [255504] [SPPM::readHeader@217] Could not open path: /tmp/blyth/opticks/SPPMTest.ppm
    ImageNPYTest: /hpcfs/juno/junogpu/blyth/junotop/opticks/npy/ImageNPY.cpp:100: static NPY<unsigned char>* ImageNPY::LoadPPM(const char*, bool, unsigned int, const char*, bool): Assertion `rc0 == 0 && mode == 6 && bits == 255' failed.

            Start  16: NPYTest.ImageNPYConcatTest
     16/116 Test  #16: NPYTest.ImageNPYConcatTest .............Subprocess aborted***Exception:   0.11 sec
    2021-03-24 20:00:09.100 INFO  [255506] [test_LoadPPMConcat@18] [
    2021-03-24 20:00:09.102 INFO  [255506] [test_LoadPPMConcat@29]  num_concat 3 path /tmp/blyth/opticks/SPPMTest_MakeTestImage.ppm yflip 0 ncomp 3 config0 add_border config1 add_midline
    2021-03-24 20:00:09.102 FATAL [255506] [SPPM::readHeader@217] Could not open path: /tmp/blyth/opticks/SPPMTest_MakeTestImage.ppm
    ImageNPYConcatTest: /hpcfs/juno/junogpu/blyth/junotop/opticks/npy/ImageNPY.cpp:100: static NPY<unsigned char>* ImageNPY::LoadPPM(const char*, bool, unsigned int, const char*, bool): Assertion `rc0 == 0 && mode == 6 && bits == 255' failed.

            Start  17: NPYTest.NPointTest
     17/116 Test  #17: NPYTest.NPointTest .....................   Passed    0.07 sec



Related issue note some direct /tmp writes on GPU node::

    drwxr-xr-x 3 blyth       dyw           21 Mar 24 21:50 blyth
    -rw-r--r-- 1 blyth       dyw       450560 Mar 24 20:00 cuRANDWrapper_10240_0_0.bin           FIXED
    -rw-r--r-- 1 blyth       dyw        45056 Mar 24 20:00 cuRANDWrapper_1024_0_0.bin            FIXED
    -rw-r--r-- 1 blyth       dyw         2240 Mar 24 20:01 mapOfMatPropVects_BUG.gdml            FIXED
    -rw-r--r-- 1 blyth       dyw          179 Mar 24 20:00 S_freopen_redirect_test.log           FIXED 
    -rw-r--r-- 1 blyth       dyw          570 Mar 24 20:01 simstream.txt                         FIXED
    -rw-r--r-- 1 blyth       dyw          405 Mar 24 20:00 thrust_curand_printf_redirect2.log    FIXED




Opticks::loadOriginCacheMeta_ asserts when using an OPTICKS_KEY born from live running
-----------------------------------------------------------------------------------------

* comment the assert in Opticks::loadOriginCacheMeta\_ to see what really needs the origin gdml path


::

    .     Start  2: OpticksCoreTest.IndexerTest
     2/43 Test  #2: OpticksCoreTest.IndexerTest ............................Subprocess aborted***Exception:   0.22 sec
    2021-03-24 20:00:19.628 INFO  [255811] [BOpticksKey::SetKey@90]  spec DetSim0Svc.X4PhysicalVolume.pWorld.85d8514854333c1a7c3fd50cc91507dc
    2021-03-24 20:00:19.632 INFO  [255811] [Opticks::init@438] COMPUTE_MODE forced_compute  hostname gpu016.ihep.ac.cn
    2021-03-24 20:00:19.632 INFO  [255811] [Opticks::init@447]  mandatory keyed access to geometry, opticksaux 
    2021-03-24 20:00:19.633 INFO  [255811] [Opticks::init@466] OpticksSwitches:WITH_SEED_BUFFER WITH_RECORD WITH_SOURCE WITH_ALIGN_DEV WITH_LOGDOUBLE WITH_KLUDGE_FLAT_ZERO_NOPEEK WITH_SENSORLIB 
    2021-03-24 20:00:19.633 ERROR [255811] [BOpticksKey::SetKey@78] key is already set, ignoring update with spec (null)
    2021-03-24 20:00:19.634 INFO  [255811] [BOpticksResource::initViaKey@785] 
                 BOpticksKey  :  
          spec (OPTICKS_KEY)  : DetSim0Svc.X4PhysicalVolume.pWorld.85d8514854333c1a7c3fd50cc91507dc
                     exename  : DetSim0Svc
             current_exename  : IndexerTest
                       class  : X4PhysicalVolume
                     volname  : pWorld
                      digest  : 85d8514854333c1a7c3fd50cc91507dc
                      idname  : DetSim0Svc_pWorld_g4live
                      idfile  : g4ok.gltf
                      idgdml  : g4ok.gdml
                      layout  : 1

    2021-03-24 20:00:19.659 INFO  [255811] [Opticks::loadOriginCacheMeta_@1996]  cachemetapath /hpcfs/juno/junogpu/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/85d8514854333c1a7c3fd50cc91507dc/1/cachemeta.json
    2021-03-24 20:00:19.677 INFO  [255811] [BMeta::dump@199] Opticks::loadOriginCacheMeta_
    {
        "GEOCACHE_CODE_VERSION": 9,
        "argline": "DetSim0Svc ",
        "cwd": "/hpcfs/juno/junogpu/blyth/tds",
        "location": "Opticks::updateCacheMeta",
        "rundate": "20210324_014558",
        "runfolder": "DetSim0Svc",
        "runlabel": "R0_cvd_0",
        "runstamp": 1616521558
    }
    2021-03-24 20:00:19.677 FATAL [255811] [Opticks::ExtractCacheMetaGDMLPath@2147]  FAILED TO EXTRACT ORIGIN GDMLPATH FROM METADATA argline 
     argline DetSim0Svc 
    2021-03-24 20:00:19.677 INFO  [255811] [Opticks::loadOriginCacheMeta_@2001] ExtractCacheMetaGDMLPath 
    2021-03-24 20:00:19.677 FATAL [255811] [Opticks::loadOriginCacheMeta_@2006] cachemetapath /hpcfs/juno/junogpu/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/85d8514854333c1a7c3fd50cc91507dc/1/cachemeta.json
    2021-03-24 20:00:19.677 FATAL [255811] [Opticks::loadOriginCacheMeta_@2007] argline that creates cachemetapath must include "--gdmlpath /path/to/geometry.gdml" 
    IndexerTest: /hpcfs/juno/junogpu/blyth/junotop/opticks/optickscore/Opticks.cc:2009: void Opticks::loadOriginCacheMeta_(): Assertion `m_origin_gdmlpath' failed.

          Start  3: OpticksCoreTest.CameraTest
     3/43 Test  #3: OpticksCoreTest.CameraTest .............................   Passed    0.06 sec
          Start  4: OpticksCoreTest.CameraSwiftTest





Removing the origin GDML path assert reduces fails to
-------------------------------------------------------


::

    SLOW: tests taking longer that 15 seconds
      30 /56  Test #30 : GGeoTest.GPtsTest                             Passed                         15.64  


    FAILS:  9   / 453   :  Wed Mar 24 23:45:51 2021   
      22 /32  Test #22 : OptiXRapTest.interpolationTest                ***Failed                      5.77   

      3  /38  Test #3  : CFG4Test.CTestDetectorTest                    ***Exception: SegFault         2.86   
      5  /38  Test #5  : CFG4Test.CGDMLDetectorTest                    Subprocess aborted***Exception:   2.71   
      7  /38  Test #7  : CFG4Test.CGeometryTest                        Subprocess aborted***Exception:   2.74   
      8  /38  Test #8  : CFG4Test.CG4Test                              ***Exception: SegFault         2.79   
      26 /38  Test #26 : CFG4Test.CInterpolationTest                   ***Exception: SegFault         2.79   
      32 /38  Test #32 : CFG4Test.CRandomEngineTest                    ***Exception: SegFault         2.82   
      1  /1   Test #1  : OKG4Test.OKG4Test                             ***Exception: SegFault         2.91   


      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      0.14   
    drwxr-xr-x 3 blyth       dyw           21 Mar 24 23:42 blyth
    gpujob-tail : rc 0




lack of numpy fails
---------------------

::

      22 /32  Test #22 : OptiXRapTest.interpolationTest                ***Failed                      5.77   



::

    2021-03-24 23:44:11.285 INFO  [155701] [SSys::RunPythonScript@571]  script interpolationTest_interpol.py script_path /hpcfs/juno/junogpu/blyth/junotop/ExternalLibs/opticks/head/bin/interpolationTest_interpol.py python_executable /hpcfs/juno/junogpu/blyth/junotop/ExternalLibs/Python/2.7.17/bin/python
    Traceback (most recent call last):
      File "/hpcfs/juno/junogpu/blyth/junotop/ExternalLibs/opticks/head/bin/interpolationTest_interpol.py", line 22, in <module>
        import os,sys, numpy as np, logging
    ImportError: No module named numpy
    2021-03-24 23:44:11.368 INFO  [155701] [SSys::run@100] /hpcfs/juno/junogpu/blyth/junotop/ExternalLibs/Python/2.7.17/bin/python /hpcfs/juno/junogpu/blyth/junotop/ExternalLibs/opticks/head/bin/interpolationTest_interpol.py  rc_raw : 256 rc : 1
    2021-03-24 23:44:11.368 ERROR [155701] [SSys::run@107] FAILED with  cmd /hpcfs/juno/junogpu/blyth/junotop/ExternalLibs/Python/2.7.17/bin/python /hpcfs/juno/junogpu/blyth/junotop/ExternalLibs/opticks/head/bin/interpolationTest_interpol.py  RC 1
    2021-03-24 23:44:11.368 INFO  [155701] [SSys::RunPythonScript@578]  RC 1



lack of GDML path from live OPTICKS_KEY geocache
---------------------------------------------------

::

      3  /38  Test #3  : CFG4Test.CTestDetectorTest                    ***Exception: SegFault         2.86   
      5  /38  Test #5  : CFG4Test.CGDMLDetectorTest                    Subprocess aborted***Exception:   2.71   
      7  /38  Test #7  : CFG4Test.CGeometryTest                        Subprocess aborted***Exception:   2.74   
      8  /38  Test #8  : CFG4Test.CG4Test                              ***Exception: SegFault         2.79   
      26 /38  Test #26 : CFG4Test.CInterpolationTest                   ***Exception: SegFault         2.79   
      32 /38  Test #32 : CFG4Test.CRandomEngineTest                    ***Exception: SegFault         2.82   
      1  /1   Test #1  : OKG4Test.OKG4Test                             ***Exception: SegFault         2.91   


::

    2021-03-24 23:45:10.059 ERROR [158046] [BFile::ExistsFile@515] BFile::ExistsFile BAD PATH path NULL sub NULL name NULL
    2021-03-24 23:45:10.060 ERROR [158046] [CGDMLDetector::init@79] CGDMLDetector::init PATH DOES NOT EXIST  path (null)
    2021-03-24 23:45:10.060 FATAL [158046] [Opticks::setSpaceDomain@2771]  changing w 60000 -> 0
    2021-03-24 23:45:10.060 FATAL [158046] [CTorchSource::configure@163] CTorchSource::configure _t 0.1 _radius 0 _pos 0.0000,0.0000,0.0000 _dir 0.0000,0.0000,1.0000 _zeaz 0.0000,1.0000,0.0000,1.0000 _pol 0.0000,0.0000,1.0000


Solution is to always save origin.gdml into the geocache : so will always have the GDML even from a live running geocache::

     914 GGeo* G4Opticks::translateGeometry( const G4VPhysicalVolume* top )
     915 {
     916     LOG(verbose) << "( key" ;
     917     const char* keyspec = X4PhysicalVolume::Key(top) ;
     918 
     919     bool parse_argv = false ;
     920     Opticks* ok = InitOpticks(keyspec, m_embedded_commandline_extra, parse_argv );
     921 
     922     const char* dbggdmlpath = ok->getDbgGDMLPath();
     923     if( dbggdmlpath != NULL )
     924     {
     925         LOG(info) << "( CGDML" ;
     926         CGDML::Export( dbggdmlpath, top );
     927         LOG(info) << ") CGDML" ;
     928     }

     ADDED SAVE OF origin.gdml HERE 

     929 
     930     LOG(info) << "( GGeo instanciate" ;
     931     bool live = true ;       // <--- for now this ignores preexisting cache in GGeo::init 
     932     GGeo* gg = new GGeo(ok, live) ;
     933     LOG(info) << ") GGeo instanciate " ;
     934 
     935     LOG(info) << "( GGeo populate" ;
     936     X4PhysicalVolume xtop(gg, top) ;
     937     LOG(info) << ") GGeo populate" ;
     938 
     939     LOG(info) << "( GGeo::postDirectTranslation " ;
     940     gg->postDirectTranslation();
     941     LOG(info) << ") GGeo::postDirectTranslation " ;
     942 
     943     return gg ;
     944 }
     945 


     569 void GGeo::postDirectTranslation()
     570 {
     571     LOG(LEVEL) << "[" ;
     572 
     573     prepare();     // instances are formed here     
     574 
     575     LOG(LEVEL) << "( GBndLib::fillMaterialLineMap " ;
     576     GBndLib* blib = getBndLib();
     577     blib->fillMaterialLineMap();
     578     LOG(LEVEL) << ") GBndLib::fillMaterialLineMap " ;
     579 
     580     LOG(LEVEL) << "( GGeo::save " ;
     581     save();
     582     LOG(LEVEL) << ") GGeo::save " ;
     583 
     584 
     585     deferredCreateGParts();
     586 
     587     postDirectTranslationDump();
     588 
     589     LOG(LEVEL) << "]" ;
     590 }




::

    L7[blyth@lxslc713 gpujob]$ BP=CGDMLDetector::init gdb_ CTestDetectorTest
    gdb -ex "set breakpoint pending on" -ex "break CGDMLDetector::init" -ex "info break" -ex r --args CTestDetectorTest
    Thu Mar 25 02:37:43 CST 2021



::

    L7[blyth@lxslc709 opticks]$ CTestDetectorTest
    2021-03-25 03:15:34.939 INFO  [2447] [main@44] CTestDetectorTest
    2021-03-25 03:15:34.941 INFO  [2447] [BOpticksKey::SetKey@90]  spec DetSim0Svc.X4PhysicalVolume.pWorld.85d8514854333c1a7c3fd50cc91507dc
    2021-03-25 03:15:34.943 INFO  [2447] [Opticks::init@439] COMPUTE_MODE forced_compute  hostname lxslc709.ihep.ac.cn
    ...
    2021-03-25 03:15:46.415 ERROR [2447] [OpticksGen::makeTorchstep@468]  generateoverride 0 num_photons0 10000 num_photons 10000
    2021-03-25 03:15:46.417 INFO  [2447] [BOpticksResource::IsGeant4EnvironmentDetected@296]  n 10 detect 1
    2021-03-25 03:15:46.417 ERROR [2447] [CG4::preinit@136] External Geant4 environment is detected, not changing this. 
    ...
    G4GDML: Reading '/hpcfs/juno/junogpu/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/85d8514854333c1a7c3fd50cc91507dc/1/origin.gdml'...
    G4GDML: Reading definitions...

    -------- EEEE ------- G4Exception-START -------- EEEE -------
    *** G4Exception : InvalidSize
          issued by : G4GDMLEvaluator::DefineMatrix()
    Matrix 'PPOABSLENGTH0x61a3280' is not filled correctly!
    *** Fatal Exception *** core dump ***
     **** Track information is not available at this moment
     **** Step information is not available at this moment

    -------- EEEE -------- G4Exception-END --------- EEEE -------


    *** G4Exception: Aborting execution ***
    Aborted (core dumped)
    L7[blyth@lxslc709 opticks]$ 



Manually edit origin.gdml::


    G4GDML: Reading '/hpcfs/juno/junogpu/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/85d8514854333c1a7c3fd50cc91507dc/1/origin.gdml'...
    G4GDML: Reading definitions...
    G4GDML: Reading materials...

    -------- EEEE ------- G4Exception-START -------- EEEE -------
    *** G4Exception : ReadError
          issued by : G4GDMLReadDefine::getMatrix()
    Matrix 'SCINTILLATIONYIELD' was not found!
    *** Fatal Exception *** core dump ***
     **** Track information is not available at this moment
     **** Step information is not available at this moment

    -------- EEEE -------- G4Exception-END --------- EEEE -------

::

       188       <property name="bisMSBTIMECONSTANT" ref="bisMSBTIMECONSTANT0x61aa9c0"/>
       189     <!--
       190       <property name="SCINTILLATIONYIELD" ref="SCINTILLATIONYIELD"/>
       191       <property name="RESOLUTIONSCALE" ref="RESOLUTIONSCALE"/>
       192       <property name="FASTTIMECONSTANT" ref="FASTTIMECONSTANT"/>
       193       <property name="SLOWTIMECONSTANT" ref="SLOWTIMECONSTANT"/>
       194       <property name="YIELDRATIO" ref="YIELDRATIO"/>
       195     -->
       196       <T unit="K" value="293.15"/>




::

    G4GDML: Reading '/hpcfs/juno/junogpu/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/85d8514854333c1a7c3fd50cc91507dc/1/origin.gdml'...
    G4GDML: Reading definitions...
    G4GDML: Reading materials...
    G4GDML: Reading solids...
    G4GDML: Reading structure...
    G4GDML: Reading setup...
    G4GDML: Reading '/hpcfs/juno/junogpu/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/85d8514854333c1a7c3fd50cc91507dc/1/origin.gdml' done!
    2021-03-25 03:40:51.993 FATAL [18632] [CMaterialSort::sort@83]  sorting G4MaterialTable using order kv 40
    2021-03-25 03:40:51.993 INFO  [18632] [CDetector::traverse@124] [
    2021-03-25 03:40:55.825 INFO  [18632] [CDetector::traverse@132] ]
    2021-03-25 03:40:55.825 FATAL [18632] [CGDMLDetector::addMPTLegacyGDML@192]  UNEXPECTED TO SEE ONLY SOME Geant4 MATERIALS WITHOUT MPT  nmat 17 nmat_without_mpt 5
    2021-03-25 03:40:55.826 INFO  [18632] [CGDMLDetector::addMPTLegacyGDML@223] CGDMLDetector::addMPT added MPT to 5 g4 materials 
    2021-03-25 03:40:55.826 INFO  [18632] [CGDMLDetector::standardizeGeant4MaterialProperties@239] [
    2021-03-25 03:40:55.826 FATAL [18632] [X4MaterialLib::init@106]  num_materials MISMATCH  G4Material::GetNumberOfMaterials 17 m_mlib->getNumMaterials 40
    CTestDetectorTest: /hpcfs/juno/junogpu/blyth/junotop/opticks/extg4/X4MaterialLib.cc:112: void X4MaterialLib::init(): Assertion `match' failed.

    (gdb) bt
    #3  0x00007fffe5f7c252 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007ffff77dcec8 in X4MaterialLib::init (this=0x7fffffff55c0) at /hpcfs/juno/junogpu/blyth/junotop/opticks/extg4/X4MaterialLib.cc:112
    #5  0x00007ffff77dcd69 in X4MaterialLib::X4MaterialLib (this=0x7fffffff55c0, mtab=0x7ffff209b070 <G4Material::theMaterialTable>, mlib=0x6c7cc0)
        at /hpcfs/juno/junogpu/blyth/junotop/opticks/extg4/X4MaterialLib.cc:81
    #6  0x00007ffff77dcd2f in X4MaterialLib::Standardize (mtab=0x7ffff209b070 <G4Material::theMaterialTable>, mlib=0x6c7cc0) at /hpcfs/juno/junogpu/blyth/junotop/opticks/extg4/X4MaterialLib.cc:72
    #7  0x00007ffff77dcd05 in X4MaterialLib::Standardize () at /hpcfs/juno/junogpu/blyth/junotop/opticks/extg4/X4MaterialLib.cc:67
    #8  0x00007ffff7b382ff in CGDMLDetector::standardizeGeant4MaterialProperties (this=0x8d27ad0) at /hpcfs/juno/junogpu/blyth/junotop/opticks/cfg4/CGDMLDetector.cc:240
    #9  0x00007ffff7b37895 in CGDMLDetector::init (this=0x8d27ad0) at /hpcfs/juno/junogpu/blyth/junotop/opticks/cfg4/CGDMLDetector.cc:106
    #10 0x00007ffff7b3743b in CGDMLDetector::CGDMLDetector (this=0x8d27ad0, hub=0x7fffffff66b0, query=0x6c14d0, sd=0x8d25470) at /hpcfs/juno/junogpu/blyth/junotop/opticks/cfg4/CGDMLDetector.cc:63
    #11 0x00007ffff7ae2aec in CGeometry::init (this=0x8d27a20) at /hpcfs/juno/junogpu/blyth/junotop/opticks/cfg4/CGeometry.cc:99
    #12 0x00007ffff7ae28ea in CGeometry::CGeometry (this=0x8d27a20, hub=0x7fffffff66b0, sd=0x8d25470) at /hpcfs/juno/junogpu/blyth/junotop/opticks/cfg4/CGeometry.cc:82
    #13 0x00007ffff7b4eb86 in CG4::CG4 (this=0x7fffffff68f0, hub=0x7fffffff66b0) at /hpcfs/juno/junogpu/blyth/junotop/opticks/cfg4/CG4.cc:159
    #14 0x0000000000403899 in main (argc=1, argv=0x7fffffff7108) at /hpcfs/juno/junogpu/blyth/junotop/opticks/cfg4/tests/CTestDetectorTest.cc:52
    (gdb) 


    228 /**
    229 CGDMLDetector::standardizeGeant4MaterialProperties
    230 -----------------------------------------------------
    231 
    232 Duplicates G4Opticks::standardizeGeant4MaterialProperties
    233 
    234 **/
    235 
    236 
    237 void CGDMLDetector::standardizeGeant4MaterialProperties()
    238 {
    239     LOG(info) << "[" ;
    240     X4MaterialLib::Standardize() ;
    241     LOG(info) << "]" ;
    242 }
    243 
    244 


::

    2021-03-25 03:55:35.614 FATAL [28472] [X4MaterialLib::init@107]  num_materials MISMATCH  G4Material::GetNumberOfMaterials 17 m_mlib->getNumMaterials 40
    m4   0 : Galactic
    m4   1 : LS
    m4   2 : Tyvek
    m4   3 : Acrylic
    m4   4 : Steel
    m4   5 : LatticedShellSteel
    m4   6 : PE_PA
    m4   7 : Air
    m4   8 : Vacuum
    m4   9 : Pyrex
    m4  10 : Rock
    m4  11 : vetoWater
    m4  12 : Water
    m4  13 : Scintillator
    m4  14 : Adhesive
    m4  15 : Aluminium
    m4  16 : TiO2Coating

    mt   0 : Galactic
    mt   1 : LS
                    mt   2 : LAB
                    mt   3 : ESR
    mt   4 : Tyvek
    mt   5 : Acrylic
                    mt   6 : DummyAcrylic
                    mt   7 : Teflon
    mt   8 : Steel
    mt   9 : LatticedShellSteel
                    mt  10 : StainlessSteel
                    mt  11 : Mylar
                    mt  12 : Copper
                    mt  13 : ETFE
                    mt  14 : FEP
    mt  15 : PE_PA
                    mt  16 : PA
    mt  17 : Air
    mt  18 : Vacuum
                    mt  19 : VacuumT
                    mt  20 : photocathode
                    mt  21 : photocathode_3inch
                    mt  22 : photocathode_MCP20inch
                    mt  23 : photocathode_MCP8inch
                    mt  24 : photocathode_Ham20inch
                    mt  25 : photocathode_Ham8inch
                    mt  26 : photocathode_HZC9inch
                    mt  27 : SiO2
                    mt  28 : B2O2
                    mt  29 : Na2O
    mt  30 : Pyrex
                    mt  31 : MineralOil
    mt  32 : Rock
    mt  33 : vetoWater
    mt  34 : Water
    mt  35 : Scintillator
    mt  36 : Adhesive
    mt  37 : Aluminium
                    mt  38 : TiO2
    mt  39 : TiO2Coating
    CTestDetectorTest: /hpcfs/juno/junogpu/blyth/junotop/opticks/extg4/X4MaterialLib.cc:127: void X4MaterialLib::init(): Assertion `match' failed.
    Aborted (core dumped)
    L7[blyth@lxslc709 extg4]$ 





lack of tboolean-
--------------------

::

    2/2 Test #2: IntegrationTests.tboolean.box ......***Failed    0.14 sec
    ====== /hpcfs/juno/junogpu/blyth/junotop/ExternalLibs/opticks/head/bin/tboolean.sh --generateoverride 10000 ====== PWD /hpcfs/juno/junogpu/blyth/junotop/ExternalLibs/opticks/head/build/integration/tests =================
    /hpcfs/juno/junogpu/blyth/junotop/ExternalLibs/opticks/head/bin/tboolean.sh: line 74: tboolean-: command not found
    tboolean-lv --generateoverride 10000
    /hpcfs/juno/junogpu/blyth/junotop/ExternalLibs/opticks/head/bin/tboolean.sh: line 78: tboolean-lv: command not found
    ====== /hpcfs/juno/junogpu/blyth/junotop/ExternalLibs/opticks/head/bin/tboolean.sh --generateoverride 10000 ====== PWD /hpcfs/juno/junogpu/blyth/junotop/ExternalLibs/opticks/head/build/integration/tests ============ RC 127 =======





tests that do not need GPU should be able to run on lxslc
-------------------------------------------------------------

::

    L7[blyth@lxslc713 ~]$ CTestDetectorTest 
    2021-03-25 00:59:48.073 INFO  [25341] [main@44] CTestDetectorTest
    2021-03-25 00:59:48.074 INFO  [25341] [BOpticksKey::SetKey@90]  spec DetSim0Svc.X4PhysicalVolume.pWorld.85d8514854333c1a7c3fd50cc91507dc
    2021-03-25 00:59:48.076 INFO  [25341] [Opticks::init@438] COMPUTE_MODE forced_compute  hostname lxslc713.ihep.ac.cn
    2021-03-25 00:59:48.076 INFO  [25341] [Opticks::init@447]  mandatory keyed access to geometry, opticksaux 
    2021-03-25 00:59:48.077 INFO  [25341] [Opticks::init@466] OpticksSwitches:WITH_SEED_BUFFER WITH_RECORD WITH_SOURCE WITH_ALIGN_DEV WITH_LOGDOUBLE WITH_KLUDGE_FLAT_ZERO_NOPEEK WITH_SENSORLIB 
    2021-03-25 00:59:48.077 ERROR [25341] [OpticksResource::SetupG4Environment@220] inipath /hpcfs/juno/junogpu/blyth/junotop/ExternalLibs/opticks/head/externals/config/geant4.ini
    2021-03-25 00:59:48.078 ERROR [25341] [OpticksResource::SetupG4Environment@229]  MISSING inipath /hpcfs/juno/junogpu/blyth/junotop/ExternalLibs/opticks/head/externals/config/geant4.ini (create it with bash functions: g4-;g4-export-ini ) 
    2021-03-25 00:59:48.079 ERROR [25341] [BOpticksKey::SetKey@78] key is already set, ignoring update with spec (null)
    2021-03-25 00:59:48.080 INFO  [25341] [BOpticksResource::initViaKey@785] 
                 BOpticksKey  :  
          spec (OPTICKS_KEY)  : DetSim0Svc.X4PhysicalVolume.pWorld.85d8514854333c1a7c3fd50cc91507dc
                     exename  : DetSim0Svc
             current_exename  : CTestDetectorTest
                       class  : X4PhysicalVolume
                     volname  : pWorld
                      digest  : 85d8514854333c1a7c3fd50cc91507dc
                      idname  : DetSim0Svc_pWorld_g4live
                      idfile  : g4ok.gltf
                      idgdml  : g4ok.gdml
                      layout  : 1

    2021-03-25 00:59:48.108 FATAL [25341] [Opticks::getCURANDStatePath@3656]  CURANDStatePath IS NOT READABLE  INVALID RNG config : change options --rngmax/--rngseed/--rngoffset  path /afs/ihep.ac.cn/users/b/blyth/.opticks/rngcache/RNG/cuRANDWrapper_3000000_0_0.bin rngdir /afs/ihep.ac.cn/users/b/blyth/.opticks/rngcache/RNG rngmax 3000000 rngseed 0 rngoffset 0
    CTestDetectorTest: /hpcfs/juno/junogpu/blyth/junotop/opticks/optickscore/Opticks.cc:3668: const char* Opticks::getCURANDStatePath(bool) const: Assertion `readable' failed.
    Aborted (core dumped)
    L7[blyth@lxslc713 ~]$ 


    (gdb) bt
    #0  0x00007fffe5f83387 in raise () from /lib64/libc.so.6
    #1  0x00007fffe5f84a78 in abort () from /lib64/libc.so.6
    #2  0x00007fffe5f7c1a6 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007fffe5f7c252 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007fffefc267e8 in Opticks::getCURANDStatePath (this=0x7fffffff6d10, assert_readable=true) at /hpcfs/juno/junogpu/blyth/junotop/opticks/optickscore/Opticks.cc:3668
    #5  0x00007fffefc1ba09 in Opticks::initResource (this=0x7fffffff6d10) at /hpcfs/juno/junogpu/blyth/junotop/opticks/optickscore/Opticks.cc:929
    #6  0x00007fffefc21891 in Opticks::postconfigure (this=0x7fffffff6d10) at /hpcfs/juno/junogpu/blyth/junotop/opticks/optickscore/Opticks.cc:2535
    #7  0x00007fffefc21417 in Opticks::configure (this=0x7fffffff6d10) at /hpcfs/juno/junogpu/blyth/junotop/opticks/optickscore/Opticks.cc:2500
    #8  0x00007ffff098a7a1 in OpticksHub::configure (this=0x7fffffff6c80) at /hpcfs/juno/junogpu/blyth/junotop/opticks/opticksgeo/OpticksHub.cc:412
    #9  0x00007ffff0989984 in OpticksHub::init (this=0x7fffffff6c80) at /hpcfs/juno/junogpu/blyth/junotop/opticks/opticksgeo/OpticksHub.cc:233
    #10 0x00007ffff09897d2 in OpticksHub::OpticksHub (this=0x7fffffff6c80, ok=0x7fffffff6d10) at /hpcfs/juno/junogpu/blyth/junotop/opticks/opticksgeo/OpticksHub.cc:215
    #11 0x0000000000403880 in main (argc=1, argv=0x7fffffff76d8) at /hpcfs/juno/junogpu/blyth/junotop/opticks/cfg4/tests/CTestDetectorTest.cc:50
    (gdb) 

