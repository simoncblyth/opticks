var-tmp-OptixCache-permissions-problem
===========================================

Context
----------

* :doc:`shakedown-running-from-expanded-binary-tarball`



OPTIX_CACHE_PATH Envvar Workaround
-------------------------------------

::

    [blyth@localhost ~]$ OPTIX_CACHE_PATH=/var/tmp/blyth/OptiXCache UseOptiX --ctx
    OptiX version 60000 major.minor.micro 6.0.0   Number of devices = 2 

     Device 0:                        TITAN_V    0000:A6:00.0 ordinal:0 compat[0]:1  Compute Support: 7 0  Total Memory: 12621381632 bytes 
     Device 1:                      TITAN_RTX    0000:73:00.0 ordinal:1 compat[0]:1  Compute Support: 7 5  Total Memory: 25364987904 bytes 
    all GPU names are unique, nothing to do 


    ( creating context 
    ) creating context 
     RT_FORMAT_FLOAT4 size 16
    [blyth@localhost ~]$ l /var/tmp/blyth/
    total 0
    drwxrwxr--. 2 blyth blyth 62 Jun  7 10:44 OptiXCache



Issue : /var/tmp/OptixCache doesnt have username in the path
----------------------------------------------------------------

::

    [simon@localhost opticks-dist-test]$ ./lib/UseOptiX
    OptiX 6.0.0
    Number of Devices = 2

    Device 0: TITAN V
      Compute Support: 7 0
      Total Memory: 12621381632 bytes
    Device 1: TITAN RTX
      Compute Support: 7 5
      Total Memory: 25364987904 bytes
    terminate called after throwing an instance of 'APIError'
    Aborted (core dumped)
    [simon@localhost opticks-dist-test]$ 
    [simon@localhost opticks-dist-test]$ pwd
    /tmp/blyth/opticks/opticks-dist-test
    [simon@localhost opticks-dist-test]$ 


Problem at context creation::

    [simon@localhost ~]$ which UseOptiX
    /tmp/blyth/opticks/opticks-dist-test/lib/UseOptiX
    [simon@localhost ~]$ 
    [simon@localhost ~]$ 
    [simon@localhost ~]$ UseOptiX
    OptiX 6.0.0
    Number of Devices = 2

    Device 0: TITAN V
      Compute Support: 7 0
      Total Memory: 12621381632 bytes
    Device 1: TITAN RTX
      Compute Support: 7 5
      Total Memory: 25364987904 bytes
    terminate called after throwing an instance of 'APIError'
    Aborted (core dumped)
    [simon@localhost ~]$ 


/var/tmp/OptixCache::

    [simon@localhost ~]$ strace -f -e trace=openat UseOptiX
    openat(AT_FDCWD, "/sys/devices/system/node", O_RDONLY|O_NONBLOCK|O_DIRECTORY|O_CLOEXEC) = 11
    strace: Process 329524 attached
    OptiX 6.0.0
    Number of Devices = 2

    Device 0: TITAN V
      Compute Support: 7 0
      Total Memory: 12621381632 bytes
    Device 1: TITAN RTX
      Compute Support: 7 5
      Total Memory: 25364987904 bytes
    ( creating context 
    [pid 329506] openat(AT_FDCWD, "/var/tmp/OptixCache", O_RDONLY|O_NONBLOCK|O_DIRECTORY|O_CLOEXEC) = 20
    [pid 329506] openat(AT_FDCWD, "/var/tmp", O_RDONLY|O_NONBLOCK|O_DIRECTORY|O_CLOEXEC) = 20
    [pid 329506] openat(AT_FDCWD, "/var/tmp/OptixCache", O_RDONLY|O_NONBLOCK|O_DIRECTORY|O_CLOEXEC) = 20
    terminate called after throwing an instance of 'APIError'
    [pid 329506] --- SIGABRT {si_signo=SIGABRT, si_code=SI_TKILL, si_pid=329506, si_uid=1001} ---
    [pid 329524] +++ killed by SIGABRT (core dumped) +++
    +++ killed by SIGABRT (core dumped) +++
    Aborted (core dumped)
    [simon@localhost ~]$ 


::

    blyth@localhost OptixCache]$ which sqlite3
    /usr/bin/sqlite3
    [blyth@localhost OptixCache]$ sqlite3 cache.db
    SQLite version 3.7.17 2013-05-20 00:56:22
    Enter ".help" for instructions
    Enter SQL statements terminated with a ";"
    sqlite> .tables 
    cache_data  cache_info  globals   
    sqlite> .schema
    CREATE TABLE cache_info (key VARCHAR(1024) UNIQUE ON CONFLICT REPLACE, optix_version VARCHAR(32), driver_version VARCHAR(32), size INTEGER, timestamp INTEGER);
    CREATE TABLE cache_data (key VARCHAR(1024) UNIQUE ON CONFLICT REPLACE, value BLOB);
    CREATE TABLE globals (key VARCHAR(256) UNIQUE ON CONFLICT REPLACE, value TEXT);
    CREATE INDEX cache_data_key ON cache_data(key);
    CREATE INDEX cache_info_key ON cache_info(key);
    CREATE TRIGGER cache_data_delete_info_trigger AFTER DELETE ON cache_data FOR EACH ROW BEGIN DELETE FROM cache_info WHERE key=OLD.key;END;
    CREATE TRIGGER cache_info_delete_data_trigger AFTER DELETE ON cache_info FOR EACH ROW BEGIN DELETE FROM cache_data WHERE key=OLD.key;END;
    CREATE TRIGGER total_data_size_delete_trigger AFTER DELETE ON cache_info FOR EACH ROW BEGIN UPDATE globals SET value=value - OLD.size WHERE key='total_data_size';END;
    CREATE TRIGGER total_data_size_insert_trigger AFTER INSERT ON cache_info FOR EACH ROW BEGIN UPDATE globals SET value=value + NEW.size WHERE key='total_data_size';END;
    sqlite> 


* :google:`/var/tmp/OptixCache`

* https://answers.arnoldrenderer.com/questions/16140/pre-populate-gpu-cache.html
* https://www.arnoldrenderer.com/


::

    [simon@localhost ~]$ OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig "steps=10,eyestartz=-1,eyestopz=5" --size 2560,1440,1 --embedded
    2019-04-27 22:24:50.074 INFO  [320773] [BOpticksKey::SetKey@45] from OPTICKS_KEY envvar OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.528f4cefdac670fffe846377973af10a
    2019-04-27 22:24:50.079 ERROR [320773] [OpticksResource::readG4Environment@499]  MISSING inipath /tmp/blyth/opticks/opticks-dist-test/externals/config/geant4.ini (create it with bash functions: g4-;g4-export-ini ) 
    2019-04-27 22:24:50.079 ERROR [320773] [OpticksResource::readOpticksEnvironment@523]  MISSING inipath /tmp/blyth/opticks/opticks-dist-test/opticksdata/config/opticksdata.ini (create it with bash functions: opticksdata-;opticksdata-export-ini ) 
    2019-04-27 22:24:50.079 ERROR [320773] [OpticksResource::initRunResultsDir@262] /tmp/blyth/opticks/opticks-dist-test/results/OpSnapTest/runlabel/20190427_222450
    2019-04-27 22:24:50.080 INFO  [320773] [OpticksHub::loadGeometry@480] [ /tmp/blyth/opticks/opticks-dist-test/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/528f4cefdac670fffe846377973af10a/1
    2019-04-27 22:24:50.448 WARN  [320773] [OpticksColors::load@52] OpticksColors::load FAILED no file at  dir /tmp/blyth/opticks/opticks-dist-test/opticksdata/resource/OpticksColors with name OpticksColors.json
    2019-04-27 22:24:50.453 INFO  [320773] [OpticksHub::loadGeometry@524] ]
    2019-04-27 22:24:50.453 WARN  [320773] [OpticksGen::initFromLegacyGensteps@160] OpticksGen::initFromLegacyGensteps SKIP as isNoInputGensteps OR isEmbedded  
    2019-04-27 22:24:50.454 INFO  [320773] [OScene::init@128] [
    2019-04-27 22:24:50.475 FATAL [320773] [OScene::initRTX@116]  --rtx 0 setting  OFF
    terminate called after throwing an instance of 'optix::Exception'
      what():  OptiX was unable to open the disk cache with sufficient privileges. Please make sure the database file is writeable by the current user.
    Aborted (core dumped)
    [simon@localhost ~]$ 


Remove the OptixCache from blyth account::

    [blyth@localhost UseOptiX]$ ll /var/tmp/OptixCache/
    total 57836
    drwxrwxrwt. 11 root  root      4096 Apr 26 23:45 ..
    -rw-rw-r--.  1 blyth blyth 55377920 Apr 27 11:01 cache.db
    drwxrwxr--.  2 blyth blyth       62 Apr 27 11:05 .
    -rw-rw-r--.  1 blyth blyth  3802512 Apr 27 22:20 cache.db-wal
    -rw-rw-r--.  1 blyth blyth    32768 Apr 27 22:33 cache.db-shm
    [blyth@localhost UseOptiX]$ rm -rf /var/tmp/OptixCache/


::

    [simon@localhost ~]$ ls -ld /tmp
    drwxrwxrwt. 22 root root 8192 Apr 28 09:29 /tmp
    [simon@localhost ~]$ ls -ld /var/tmp
    drwxrwxrwt. 11 root root 4096 Apr 27 22:45 /var/tmp
    [simon@localhost ~]$ 


But then it fails from blyth::

    [blyth@localhost sysrap]$ UseOptiX
    OptiX 6.0.0
    Number of Devices = 2

    Device 0: TITAN V
      Compute Support: 7 0
      Total Memory: 12621381632 bytes
    Device 1: TITAN RTX
      Compute Support: 7 5
      Total Memory: 25364987904 bytes
    ( creating context 
    terminate called after throwing an instance of 'APIError'
    Aborted (core dumped)
    [blyth@localhost sysrap]$ 



Workaround : delete the cache in OContext::cleanupCache 
-----------------------------------------------------------

Suspected side effect of slower test startup but not so::

    blyth@localhost optixrap]$ cvd 1 om-test
    === om-test-one : optixrap        /home/blyth/opticks/optixrap                                 /home/blyth/local/opticks/build/optixrap                     
    Sun Apr 28 10:52:49 CST 2019
    Test project /home/blyth/local/opticks/build/optixrap
          Start  1: OptiXRapTest.OContextCreateTest
     1/19 Test  #1: OptiXRapTest.OContextCreateTest ..............   Passed    0.28 sec
          Start  2: OptiXRapTest.OScintillatorLibTest
     2/19 Test  #2: OptiXRapTest.OScintillatorLibTest ............   Passed    0.44 sec
          Start  3: OptiXRapTest.OOTextureTest
     3/19 Test  #3: OptiXRapTest.OOTextureTest ...................   Passed    0.60 sec
          Start  4: OptiXRapTest.OOMinimalTest
     4/19 Test  #4: OptiXRapTest.OOMinimalTest ...................   Passed    0.63 sec
          Start  5: OptiXRapTest.OOMinimalRedirectTest
     5/19 Test  #5: OptiXRapTest.OOMinimalRedirectTest ...........   Passed    0.38 sec
          Start  6: OptiXRapTest.OOContextTest
     6/19 Test  #6: OptiXRapTest.OOContextTest ...................   Passed    0.52 sec
          Start  7: OptiXRapTest.OOContextLowTest
     7/19 Test  #7: OptiXRapTest.OOContextLowTest ................   Passed    0.70 sec
     ...

::

    [blyth@localhost optixrap]$ CUDA_VISIBLE_DEVICES=1 OPTICKS_KEEPCACHE=1 om-test 
    === om-test-one : optixrap        /home/blyth/opticks/optixrap                                 /home/blyth/local/opticks/build/optixrap                     
    Sun Apr 28 11:06:41 CST 2019
    Test project /home/blyth/local/opticks/build/optixrap
          Start  1: OptiXRapTest.OContextCreateTest
     1/19 Test  #1: OptiXRapTest.OContextCreateTest ..............   Passed    0.25 sec
          Start  2: OptiXRapTest.OScintillatorLibTest
     2/19 Test  #2: OptiXRapTest.OScintillatorLibTest ............   Passed    0.42 sec
          Start  3: OptiXRapTest.OOTextureTest
     3/19 Test  #3: OptiXRapTest.OOTextureTest ...................   Passed    0.55 sec
          Start  4: OptiXRapTest.OOMinimalTest
     4/19 Test  #4: OptiXRapTest.OOMinimalTest ...................   Passed    0.65 sec
          Start  5: OptiXRapTest.OOMinimalRedirectTest
     5/19 Test  #5: OptiXRapTest.OOMinimalRedirectTest ...........   Passed    0.40 sec
          Start  6: OptiXRapTest.OOContextTest
     6/19 Test  #6: OptiXRapTest.OOContextTest ...................   Passed    0.50 sec
     ...


