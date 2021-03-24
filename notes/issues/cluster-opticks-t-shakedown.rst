cluster-opticks-t-shakedown
==============================


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




