opticks-t-36-of-407-fails-on-macOS-2019-06-22
=================================================

::

    FAILS:  36  / 407   :  Sat Jun 22 12:34:44 2019   
      32 /37  Test #32 : BoostRapTest.BConverterTest                   ***Exception: Child aborted    0.01   
      AVOIDED : a difference in numerical exception percolation with clang ?    

      82 /120 Test #82 : NPYTest.NCSGLoadTest                          ***Exception: Interrupt        0.01   
      FIXED : was a badlex std::raise(SIGINT)

      89 /120 Test #89 : NPYTest.NScanTest                             ***Exception: Interrupt        0.01   
      FIXED badlex too ?

      94 /120 Test #94 : NPYTest.NTreeJUNOTest                         ***Exception: Child aborted    0.08   
      FIXED initialization ordering what was causing set_zcut(0,0) ?  Huh how did this pass on Linux ?

      113/120 Test #113: NPYTest.NSceneTest                            ***Exception: Child aborted    1.51   
      SKIPPED IT : NScene IS EOL 

      7  /29  Test #7  : OpticksCoreTest.OpticksEventSpecTest          ***Exception: SegFault         0.01   
      FIXED handle NULL m_cat

      30 /53  Test #30 : GGeoTest.GPtsTest                             ***Exception: Child aborted    0.02   
      SEE ISSUE 1 BELOW      


      40 /53  Test #40 : GGeoTest.GGeoTest                             ***Exception: Child aborted    0.20   
      41 /53  Test #41 : GGeoTest.GMakerTest                           ***Exception: Child aborted    0.03   
      52 /53  Test #52 : GGeoTest.GSceneTest                           ***Exception: Child aborted    0.19   
      1  /3   Test #1  : OpticksGeoTest.OpticksGeoTest                 ***Exception: Child aborted    0.20   
      2  /3   Test #2  : OpticksGeoTest.OpticksHubTest                 ***Exception: Child aborted    0.20   
      12 /24  Test #12 : OptiXRapTest.rayleighTest                     ***Exception: Child aborted    0.22   
      17 /24  Test #17 : OptiXRapTest.eventTest                        ***Exception: Child aborted    0.22   
      HUH : NO FAIL


      53 /53  Test #53 : GGeoTest.GMeshLibTest                         ***Exception: Child aborted    0.02   
      SEE ISSUE 1 BELOW      


      18 /24  Test #18 : OptiXRapTest.interpolationTest                ***Exception: Child aborted    0.22   
      FIXED : ana/main.py issue because of export OPTICKS_ANA_DEFAULTS="det=concentric,src=torch,tag=1,pfx=."  missed the pfx


      1  /5   Test #1  : OKOPTest.OpIndexerTest                        ***Exception: Child aborted    0.24   
      2  /5   Test #2  : OKOPTest.OpSeederTest                         ***Exception: Child aborted    0.22   
      5  /5   Test #5  : OKOPTest.OpSnapTest                           ***Exception: Child aborted    0.22   
      2  /5   Test #2  : OKTest.OKTest                                 ***Exception: Child aborted    0.22   
      3  /5   Test #3  : OKTest.OTracerTest                            ***Exception: Child aborted    0.24   
      1  /34  Test #1  : CFG4Test.CMaterialLibTest                     ***Exception: Child aborted    0.25   
      2  /34  Test #2  : CFG4Test.CMaterialTest                        ***Exception: Child aborted    0.22   
      NO FAIL


      3  /34  Test #3  : CFG4Test.CTestDetectorTest                    ***Exception: Child aborted    0.24   
      5  /34  Test #5  : CFG4Test.CGDMLDetectorTest                    ***Exception: Child aborted    0.23   
      6  /34  Test #6  : CFG4Test.CGeometryTest                        ***Exception: Child aborted    0.22   
      7  /34  Test #7  : CFG4Test.CG4Test                              ***Exception: Child aborted    0.23   
      22 /34  Test #22 : CFG4Test.CGenstepCollectorTest                ***Exception: Child aborted    0.23   
      23 /34  Test #23 : CFG4Test.CInterpolationTest                   ***Exception: Child aborted    0.23   
      25 /34  Test #25 : CFG4Test.CGROUPVELTest                        ***Exception: Child aborted    0.22   
      27 /34  Test #27 : CFG4Test.CTreeJUNOTest                        ***Exception: Child aborted    0.12   
      29 /34  Test #29 : CFG4Test.CRandomEngineTest                    ***Exception: Child aborted    0.22   
      32 /34  Test #32 : CFG4Test.CCerenkovGeneratorTest               ***Exception: Child aborted    0.22   
      33 /34  Test #33 : CFG4Test.CGenstepSourceTest                   ***Exception: Child aborted    0.22   
      1  /1   Test #1  : OKG4Test.OKG4Test                             ***Exception: Child aborted    0.28   
      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      2.90   




Now down to 4
---------------

::

    FAILS:  4   / 406   :  Sat Jun 22 14:05:12 2019   
      30 /53  Test #30 : GGeoTest.GPtsTest                             ***Exception: Child aborted    0.02   
      53 /53  Test #53 : GGeoTest.GMeshLibTest                         ***Exception: Child aborted    0.02   
      29 /34  Test #29 : CFG4Test.CRandomEngineTest                    ***Exception: Child aborted    1.39   
      FIXED : planted assert 

      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      2.72   




ISSUE 1 : GPtsTest, GMeshLibTest  : these due to the changes in GMeshLib persisting ? With individiual GMesh etc..
-----------------------------------------------------------------------------------------------------------------------


::


    epsilon:cfg4 blyth$ GPtsTest 
    2019-06-22 14:11:31.975 INFO  [17036561] [Opticks::init@313] INTEROP_MODE
    2019-06-22 14:11:31.978 INFO  [17036561] [BOpticksResource::setupViaKey@531] 
                 BOpticksKey  :  
          spec (OPTICKS_KEY)  : OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.15cf540d9c315b7f5d0adc7c3907b922
                     exename  : OKX4Test
             current_exename  : GPtsTest
                       class  : X4PhysicalVolume
                     volname  : lWorld0x4bc2710_PV
                      digest  : 15cf540d9c315b7f5d0adc7c3907b922
                      idname  : OKX4Test_lWorld0x4bc2710_PV_g4live
                      idfile  : g4ok.gltf
                      idgdml  : g4ok.gdml
                      layout  : 1

    2019-06-22 14:11:31.979 ERROR [17036561] [GItemList::load_@56]  MISSING ITEMLIST TXT  txtpath /usr/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/15cf540d9c315b7f5d0adc7c3907b922/1/GItemList/GMeshLib.txt txtname GMeshLib.txt reldir GItemList m_itemtype GMeshLib
    Assertion failed: (soliddir_exists && "GMeshLib persisted GMesh are expected to have paired GMeshLibNCSG dirs"), function loadMeshes, file /Users/blyth/opticks/ggeo/GMeshLib.cc, line 429.
    Abort trap: 6
    epsilon:cfg4 blyth$ 


    epsilon:cfg4 blyth$ GMeshLibTest 
    2019-06-22 14:12:17.538 INFO  [17037158] [Opticks::init@313] INTEROP_MODE
    2019-06-22 14:12:17.541 INFO  [17037158] [BOpticksResource::setupViaKey@531] 
                 BOpticksKey  :  
          spec (OPTICKS_KEY)  : OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.15cf540d9c315b7f5d0adc7c3907b922
                     exename  : OKX4Test
             current_exename  : GMeshLibTest
                       class  : X4PhysicalVolume
                     volname  : lWorld0x4bc2710_PV
                      digest  : 15cf540d9c315b7f5d0adc7c3907b922
                      idname  : OKX4Test_lWorld0x4bc2710_PV_g4live
                      idfile  : g4ok.gltf
                      idgdml  : g4ok.gdml
                      layout  : 1

    2019-06-22 14:12:17.542 ERROR [17037158] [GItemList::load_@56]  MISSING ITEMLIST TXT  txtpath /usr/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/15cf540d9c315b7f5d0adc7c3907b922/1/GItemList/GMeshLib.txt txtname GMeshLib.txt reldir GItemList m_itemtype GMeshLib
    Assertion failed: (soliddir_exists && "GMeshLib persisted GMesh are expected to have paired GMeshLibNCSG dirs"), function loadMeshes, file /Users/blyth/opticks/ggeo/GMeshLib.cc, line 429.
    Abort trap: 6



* following :doc:`x016` "breakage" tried to fix using tests/GItemIndex2Test.cc with non-default define WRITE_MESHNAMES_TO_GEOCACHE : but thats 
  not it, this is a direct geometry using keydir

* hence on Linux added+commit+push opticksdata-jv4 geometry to opticksdata

* pulled that down to macOS, and do : geocache-recreate 

At the tail of the log::

    2019-06-22 14:19:46.370 INFO  [17038822] [Opticks::reportGeoCacheCoordinates@727]  ok.idpath  /usr/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/a717fae122a5eda2027f6cec1d4a3f67/1
    2019-06-22 14:19:46.370 INFO  [17038822] [Opticks::reportGeoCacheCoordinates@728]  ok.keyspec OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.a717fae122a5eda2027f6cec1d4a3f67
    2019-06-22 14:19:46.370 INFO  [17038822] [Opticks::reportGeoCacheCoordinates@729]  To reuse this geometry: 
    2019-06-22 14:19:46.370 INFO  [17038822] [Opticks::reportGeoCacheCoordinates@730]    1. set envvar OPTICKS_KEY=OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.a717fae122a5eda2027f6cec1d4a3f67
    2019-06-22 14:19:46.370 INFO  [17038822] [Opticks::reportGeoCacheCoordinates@731]    2. enable envvar sensitivity with --envkey argument to Opticks executables 
    2019-06-22 14:19:46.371 FATAL [17038822] [Opticks::reportGeoCacheCoordinates@739] THE LIVE keyspec DOES NOT MATCH THAT OF THE CURRENT ENVVAR 
    2019-06-22 14:19:46.371 INFO  [17038822] [Opticks::reportGeoCacheCoordinates@740]  (envvar) OPTICKS_KEY=OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.15cf540d9c315b7f5d0adc7c3907b922
    2019-06-22 14:19:46.371 INFO  [17038822] [Opticks::reportGeoCacheCoordinates@741]  (live)   OPTICKS_KEY=OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.a717fae122a5eda2027f6cec1d4a3f67
    2019-06-22 14:19:46.371 INFO  [17038822] [Opticks::dumpRC@202]  rc 0 rcmsg : -
    2019-06-22 14:19:52.599 INFO  [17038822] [OContext::cleanUpCache@466]  RemoveDir /var/tmp/OptixCache
    BFile::RemoveDir path does not exist /var/tmp/OptixCache
    === o-main : /usr/local/opticks/lib/OKX4Test --okx4 --g4codegen --deletegeocache --gdmlpath /usr/local/opticks/opticksdata/export/juno1808/g4_00_v4.gdml --csgskiplv 22 --runfolder geocache-j1808-v4 --runcomment torus-less-skipping-just-lv-22-maskVirtual ======= PWD /tmp/blyth/opticks/geocache-create- RC 0 Sat Jun 22 14:19:53 CST 2019
    echo o-postline : dummy
    o-postline : dummy
    /Users/blyth/opticks/bin/o.sh : RC : 0


Setting that into .bash_profile to adopt the new geometry with::

    373 export OPTICKS_HOME=$HOME/opticks
    374 opticks-(){  [ -r $OPTICKS_HOME/opticks.bash ] && . $OPTICKS_HOME/opticks.bash && opticks-env $* && opticks-export ; }
    375 
    376 #export OPTICKS_KEY=OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.15cf540d9c315b7f5d0adc7c3907b922
    377 export OPTICKS_KEY=OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.a717fae122a5eda2027f6cec1d4a3f67
    378 


Now the GMeshLibTest and GPtsTest pass.




And then there was one
--------------------------

::

    FAILS:  1   / 406   :  Sat Jun 22 14:28:09 2019   
      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      2.59 



::

    [2019-06-22 14:33:05,033] p18430 {env.py    :144} WARNING  - legacy_init : OPTICKS_KEY envvar deleted for legacy running, unset IDPATH to use direct_init
    [2019-06-22 14:33:05,034] p18430 {tboolean.py:63} INFO     - pfx tboolean-box tag 1 src torch det tboolean-box c2max [1.5, 2.0, 2.5] ipython False 
    [2019-06-22 14:33:05,035] p18430 {base.py   :174} WARNING  - failed to load json from $OPTICKS_INSTALL_CACHE/OKC/OpticksFlagsAbbrevMeta.json : /usr/local/opticks/installcache/OKC/OpticksFlagsAbbrevMeta.json 
    Traceback (most recent call last):
      File "/Users/blyth/opticks/ana/tboolean.py", line 65, in <module>
        ab = AB(ok)
      File "/Users/blyth/opticks/ana/ab.py", line 192, in __init__
        self.histype = HisType()
      File "/Users/blyth/opticks/ana/histype.py", line 53, in __init__
        flags = PhotonCodeFlags() 
      File "/Users/blyth/opticks/ana/base.py", line 337, in __init__
        self.abbrev = Abbrev("$OPTICKS_INSTALL_CACHE/OKC/OpticksFlagsAbbrevMeta.json")
      File "/Users/blyth/opticks/ana/base.py", line 208, in __init__
        js = json_(path)
      File "/Users/blyth/opticks/ana/base.py", line 175, in json_
        assert 0
    AssertionError
    2019-06-22 14:33:05.094 INFO  [17072993] [SSys::run@72] tboolean.py --tagoffset 0 --tag 1 --det tboolean-box --pfx tboolean-box --src torch   rc_raw : 256 rc : 1
    2019-06-22 14:33:05.094 ERROR [17072993] [SSys::run@79] FAILED with  cmd tboolean.py --tagoffset 0 --tag 1 --det tboolean-box --pfx tboolean-box --src torch   RC 1

    2019-06-22 14:33:05.094 INFO  [17072993] [OpticksAna::run@89]  anakey tboolean cmdline tboolean.py --tagoffset 0 --tag 1 --det tboolean-box --pfx tboolean-box --src torch   interactivity 2 rc 1 rcmsg OpticksAna::run non-zero RC from ana script
    2019-06-22 14:33:05.094 FATAL [17072993] [Opticks::dumpRC@202]  rc 1 rcmsg : OpticksAna::run non-zero RC from ana script
    2019-06-22 14:33:05.094 INFO  [17072993] [SSys::WaitForInput@226] SSys::WaitForInput OpticksAna::run paused : hit RETURN to continue...



::

    epsilon:tmp blyth$ tboolean.py --tagoffset 0 --tag 1 --det tboolean-box --pfx tboolean-box --src torch
    args: /Users/blyth/opticks/ana/tboolean.py --tagoffset 0 --tag 1 --det tboolean-box --pfx tboolean-box --src torch
    [2019-06-22 14:34:54,721] p18441 {env.py    :144} WARNING  - legacy_init : OPTICKS_KEY envvar deleted for legacy running, unset IDPATH to use direct_init
    [2019-06-22 14:34:54,721] p18441 {tboolean.py:63} INFO     - pfx tboolean-box tag 1 src torch det tboolean-box c2max [1.5, 2.0, 2.5] ipython False 
    [2019-06-22 14:34:54,722] p18441 {base.py   :174} WARNING  - failed to load json from $OPTICKS_INSTALL_CACHE/OKC/OpticksFlagsAbbrevMeta.json : /usr/local/opticks/installcache/OKC/OpticksFlagsAbbrevMeta.json 
    Traceback (most recent call last):
      File "/Users/blyth/opticks/ana/tboolean.py", line 65, in <module>
        ab = AB(ok)
      File "/Users/blyth/opticks/ana/ab.py", line 192, in __init__
        self.histype = HisType()
      File "/Users/blyth/opticks/ana/histype.py", line 53, in __init__
        flags = PhotonCodeFlags() 
      File "/Users/blyth/opticks/ana/base.py", line 337, in __init__
        self.abbrev = Abbrev("$OPTICKS_INSTALL_CACHE/OKC/OpticksFlagsAbbrevMeta.json")
      File "/Users/blyth/opticks/ana/base.py", line 208, in __init__
        js = json_(path)
      File "/Users/blyth/opticks/ana/base.py", line 175, in json_
        assert 0
    AssertionError


Linux has this file::

    epsilon:integration blyth$ l /usr/local/opticks/installcache/OKC/
    total 32
    -rw-r--r--  1 blyth  staff  308 Nov 27  2018 GFlagIndexLocal.ini
    -rw-r--r--  1 blyth  staff  308 Nov 27  2018 GFlagIndexSource.ini
    -rw-r--r--  1 blyth  staff  308 Nov 27  2018 GFlagsLocal.ini
    -rw-r--r--  1 blyth  staff  308 Nov 27  2018 GFlagsSource.ini


Perhaps just rerun OpticksPrepareInstallCache_OKC

::

    epsilon:opticks blyth$ find . -name OpticksPrepareInstallCache_OKC.cc
    ./optickscore/tests/OpticksPrepareInstallCache_OKC.cc

::

    2842 /**
    2843 Opticks::prepareInstallCache
    2844 -----------------------------
    2845 
    2846 Moved save directory from IdPath to ResourceDir as
    2847 the IdPath is not really appropriate  
    2848 for things such as the flags that are a feature of an 
    2849 Opticks installation, not a feature of the geometry.
    2850 
    2851 But ResourceDir is not appropriate either as that requires 
    2852 manual management via opticksdata repo.
    2853 
    2854 
    2855 **/
    2856 
    2857 
    2858 void Opticks::prepareInstallCache(const char* dir)
    2859 {
    2860     if(dir == NULL) dir = m_resource->getOKCInstallCacheDir() ;
    2861     LOG(info) << ( dir ? dir : "NULL" )  ;
    2862     m_resource->saveFlags(dir);
    2863     m_resource->saveTypes(dir);
    2864 }


::

    epsilon:integration blyth$ ll /usr/local/opticks/installcache/OKC/
    total 40
    drwxr-xr-x  5 blyth  staff  160 Apr  5  2018 ..
    -rw-r--r--  1 blyth  staff  308 Jun 22 14:40 GFlagsSource.ini
    -rw-r--r--  1 blyth  staff  308 Jun 22 14:40 GFlagsLocal.ini
    drwxr-xr-x  7 blyth  staff  224 Jun 22 14:40 .
    -rw-r--r--  1 blyth  staff  279 Jun 22 14:40 OpticksFlagsAbbrevMeta.json
    -rw-r--r--  1 blyth  staff  308 Jun 22 14:40 GFlagIndexSource.ini
    -rw-r--r--  1 blyth  staff  308 Jun 22 14:40 GFlagIndexLocal.ini
    epsilon:integration blyth$ 



::

    2019-06-22 14:44:37.077 INFO  [17080310] [OpticksAna::run@70]  anakey tboolean enabled Y

    args: /Users/blyth/opticks/ana/tboolean.py --tagoffset 0 --tag 1 --det tboolean-box --pfx tboolean-box --src torch
    [2019-06-22 14:44:37,886] p19812 {env.py    :144} WARNING  - legacy_init : OPTICKS_KEY envvar deleted for legacy running, unset IDPATH to use direct_init
    [2019-06-22 14:44:37,887] p19812 {tboolean.py:63} INFO     - pfx tboolean-box tag 1 src torch det tboolean-box c2max [1.5, 2.0, 2.5] ipython False 
    Traceback (most recent call last):
      File "/Users/blyth/opticks/ana/tboolean.py", line 65, in <module>
        ab = AB(ok)
      File "/Users/blyth/opticks/ana/ab.py", line 195, in __init__
        self.load()
      File "/Users/blyth/opticks/ana/ab.py", line 220, in load
        a = Evt(tag=atag, src=args.src, det=args.det, pfx=args.pfx, args=args, nom="A", smry=args.smry)
      File "/Users/blyth/opticks/ana/evt.py", line 223, in __init__
        self.init_types()
      File "/Users/blyth/opticks/ana/evt.py", line 268, in init_types
        mattype = MatType(reldir=reldir)
      File "/Users/blyth/opticks/ana/mattype.py", line 109, in __init__
        material_names = ItemList("GMaterialLib", reldir=reldir)
      File "/Users/blyth/opticks/ana/base.py", line 247, in __init__
        names = map(lambda _:_[:-1],file(npath).readlines())
    IOError: [Errno 2] No such file or directory: u'/tmp/blyth/opticks/tboolean-box/GItemList/GMaterialLib.txt'
    2019-06-22 14:44:37.943 INFO  [17080310] [SSys::run@72] tboolean.py --tagoffset 0 --tag 1 --det tboolean-box --pfx tboolean-box --src torch   rc_raw : 256 rc : 1
    2019-06-22 14:44:37.943 ERROR [17080310] [SSys::run@79] FAILED with  cmd tboolean.py --tagoffset 0 --tag 1 --det tboolean-box --pfx tboolean-box --src torch   RC 1



Made the below work by defining envvar OPTICKS_EVENT_BASE as /tmp::

   tboolean-box --okg4


But still macOS bash failing to pass TESTCONFIG from tboolean-box into tboolean--
but only when run within tboolean.sh ?

Looks like a bash bug::

    epsilon:opticks blyth$ bash --version
    GNU bash, version 3.2.57(1)-release (x86_64-apple-darwin17)
    Copyright (C) 2007 Free Software Foundation, Inc.
    epsilon:opticks blyth$ 





