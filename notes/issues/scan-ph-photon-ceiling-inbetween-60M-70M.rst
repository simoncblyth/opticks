scan-ph-photon-ceiling-inbetween-60M-70M ELIMINATED
=======================================================

FIXED Issue 
------------

scan-ph scanning of "ts box" revealed a ceiling on single launch 
photon counts somewhere between 60M and 70M photons : with no error returned
just less hits obtained than expected.


suspect cycling of an unsigned int on the bytesize of the buffer : CONFIRMED
---------------------------------------------------------------------------------------

* suspect this as there is no crash and the number of hits is linearly offset from expectations

::

    In [10]: ( 1 << 32 ) - 1
    Out[10]: 4294967295             ##  4,294,967,295         ~4300 M   

* photon buffer is 4*4 floats = 16*4 = 64 bytes per photon, so probably 67,108,863

::

    In [12]: (( 1 << 32 ) - 1 ) / 64  
    Out[12]: 67108863

    In [13]: (( 1 << 32 ) - 1 ) / 64 /1e6  
    Out[13]: 67.108863



FIXED : found limit to be 67,108,863 photons arising from cycling unsigned int various buffer handling code
---------------------------------------------------------------------------------------------------------------

* fix was to move to unsigned long long for buffer byte handling in::

    npy-
    cudarap-
    thrap-
    oxrap-


TODO : profiling
-------------------

* profile times for the varions stages of big photon count handling 
* comparing times with different numbers of visible GPUs
  
* use Nsight Systems to check GPU memory utilization during big running  
* NVTX labelling


TODO : memory accounting to determine theoretical photon ceiling
---------------------------------------------------------------------

Memory::


    4*Tesla V100-SXM2-32GB              128G
    NVIDIA Quadro RTX 8000 : 48598MiB    48G
    Tesla V100-SXM2-32GB   : 32480MiB    32G
    NVIDIA TITAN RTX       : 24219MiB    24G
    NVIDIA TITAN V         : 12066MiB    12G  


* push current 100M largest attempt further : 

  * 67.1M photons corresponds to 4G of photon buffer
  * 671M will get to 40G of photon buffer : which might still work in Quadro RTX 8000 
 
* hmm what about with 4xV100, that might reach 1000M photons in one launch 



okop/compactionTest
----------------------

Pincer search to find the ceiling number of photons::

    compactionTest --generateoverride -70     ## 70M
    2019-09-21 16:25:23.324 INFO  [145715] [main@170]  num_hits 2892 x_num_hits 70000

    compactionTest --generateoverride $(( 67000000 + 150000 ))             ##  67,150,000 
    2019-09-21 16:56:29.632 INFO  [196225] [main@172]  num_hits 42 x_num_hits 67150               FAIL

    compactionTest --generateoverride $(( 67000000 + 110000 ))             ##  67,110,000          
    2019-09-21 16:59:21.337 INFO  [200568] [main@172]  num_hits 2 x_num_hits 67110                FAIL 


    compactionTest --generateoverride $(( 67000000 + 109000 ))             ##  67,109,000
    2019-09-21 17:27:29.183 INFO  [243574] [main@172]  num_hits 1 x_num_hits 67109                FAIL 



    compactionTest --generateoverride $(( 67000000 + 108000 ))             ##  67,108,000     
    2019-09-21 17:22:49.405 INFO  [236445] [main@172]  num_hits 67108 x_num_hits 67108            PASS

    compactionTest --generateoverride $(( 67000000 + 102000 ))             ##  67,102,000     
    2019-09-21 17:07:07.147 INFO  [212662] [main@172]  num_hits 67102 x_num_hits 67102            PASS

    compactionTest --generateoverride $(( 67000000 + 100000 ))             ##  67,100,000
    2019-09-21 16:51:01.424 INFO  [187542] [main@172]  num_hits 67100 x_num_hits 67100            PASS



67,108,000 PASS::

    [blyth@localhost okop]$ compactionTest --generateoverride $(( 67000000 + 108000 ))             ##  67,108,000 
    ...
    CBufSpec.Summary.cpho before TBuf : dev_ptr 0x7f9994000000 size 67108000 num_bytes 4294912000 hexdump 0 
    2019-09-21 17:22:49.365 ERROR [236445] [main@148]  created tpho  cpho.size : 67108000 num_photons : 67108000
    2019-09-21 17:22:49.365 ERROR [236445] [main@159] [ tpho.downloadSelection4x4 
    2019-09-21 17:22:49.405 ERROR [236445] [main@161] ] tpho.downloadSelection4x4 
    2019-09-21 17:22:49.405 INFO  [236445] [main@172]  num_hits 67108 x_num_hits 67108
    [blyth@localhost okop]$ 


67,109,000 FAIL::

    [blyth@localhost okop]$ compactionTest --generateoverride $(( 67000000 + 109000 ))             ##  67,109,000 
    ...
    CBufSpec.Summary.cpho before TBuf : dev_ptr 0x7f19e0000000 size 67109000 num_bytes 8704 hexdump 0      <<<<  CYCLED THE unsigned int num_butes buffer size
    2019-09-21 17:27:29.183 INFO  [243574] [main@172]  num_hits 1 x_num_hits 67109
    compactionTest: /home/blyth/opticks/okop/tests/compactionTest.cc:177: int main(int, char**): Assertion `num_hits == x_num_hits' failed.
    Aborted (core dumped)
    [blyth@localhost okop]$ 



Switched to unsigned long long in CBuf and TBuf : but still failing
----------------------------------------------------------------------

* NPYBase too ?

::

    blyth@localhost okop]$ compactionTest --generateoverride $(( 67000000 + 109000 ))
    ...
    CBufSpec.Summary.cpho before TBuf : dev_ptr 0x7fc5d0000000 size 67109000 num_bytes 4294976000 hexdump 0 
    2019-09-21 18:14:23.974 ERROR [331303] [main@148]  created tpho  cpho.size : 67109000 num_photons : 67109000
    2019-09-21 18:14:23.974 ERROR [331303] [main@159] [ tpho.downloadSelection4x4 
    2019-09-21 18:14:24.010 ERROR [331303] [main@161] ] tpho.downloadSelection4x4 
    2019-09-21 18:14:24.010 INFO  [331303] [main@172]  num_hits 1 x_num_hits 67109
    compactionTest: /home/blyth/opticks/okop/tests/compactionTest.cc:177: int main(int, char**): Assertion `num_hits == x_num_hits' failed.
    Aborted (core dumped)
    [blyth@localhost okop]$ 


At +1 from the critical number the num bytes is zero::

    [blyth@localhost okop]$ DummyPhotonsNPYTest 67108863
    2019-09-21 18:35:45.097 INFO  [367338] [main@34]  num_photons 67108863 hitmask 64
    2019-09-21 18:36:33.277 INFO  [367338] [main@41] DummyPhotonsNPY::Make (67108863,4,4)  NumBytes(0) 4294967232 NumBytes(1) 64 NumValues(0) 1073741808 NumValues(1) 16{}

    [blyth@localhost okop]$ DummyPhotonsNPYTest 67108864
    2019-09-21 18:37:58.390 INFO  [370739] [main@34]  num_photons 67108864 hitmask 64
    2019-09-21 18:38:45.284 INFO  [370739] [main@41] DummyPhotonsNPY::Make (67108864,4,4)  NumBytes(0) 0 NumBytes(1) 64 NumValues(0) 1073741824 NumValues(1) 16{}


After modifying NPYBase to handles buffer size and byte related things with ULL::

    [blyth@localhost okop]$ DummyPhotonsNPYTest 67108864
    2019-09-21 18:56:12.846 INFO  [401103] [main@34]  num_photons 67108864 hitmask 64
    2019-09-21 18:57:04.145 INFO  [401103] [main@41] DummyPhotonsNPY::Make (67108864,4,4)  NumBytes(0) 4294967296 NumBytes(1) 64 NumValues(0) 1073741824 NumValues(1) 16{}


BUT compactionTest still failing::

    [blyth@localhost opticks]$ compactionTest --generateoverride $(( 67000000 + 109000 ))
    ...
    CBufSpec.Summary.cpho before TBuf : dev_ptr 0x7f9f48000000 size 67109000 num_bytes 4294976000 hexdump 0 
    2019-09-21 19:01:11.499 ERROR [416781] [main@148]  created tpho  cpho.size : 67109000 num_photons : 67109000
    2019-09-21 19:01:11.499 ERROR [416781] [main@159] [ tpho.downloadSelection4x4 
    2019-09-21 19:01:11.536 ERROR [416781] [main@161] ] tpho.downloadSelection4x4 
    2019-09-21 19:01:11.536 INFO  [416781] [main@172]  num_hits 1 x_num_hits 67109
    compactionTest: /home/blyth/opticks/okop/tests/compactionTest.cc:177: int main(int, char**): Assertion `num_hits == x_num_hits' failed.
    Aborted (core dumped)
    [blyth@localhost opticks]$ 

Twas a truncation in OContext::upload, fixing that and it works::

    OContext=ERROR compactionTest --generateoverride $(( 67000000 + 109000 ))

    [blyth@localhost thrustrap]$ OContext=ERROR compactionTest --generateoverride $(( 67000000 + 109000 ))
    PLOG::EnvLevel adjusting loglevel by envvar   key OContext level ERROR fallback DEBUG
    2019-09-21 19:10:45.199 INFO  [433454] [Opticks::init@389] COMPUTE_MODE compute_requested 
    2019-09-21 19:10:45.199 FATAL [433454] [Opticks::init@392] OPTICKS_LEGACY_GEOMETRY_ENABLED mode is active  : ie dae src access to geometry, opticksdata  
    2019-09-21 19:10:45.205 INFO  [433454] [Opticks::initResource@773]  (legacy mode) setting IDPATH envvar for python analysis scripts [/home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae]
    2019-09-21 19:10:45.206 FATAL [433454] [Opticks::defineEventSpec@2024]  resource_pfx (null) config_pfx (null) pfx default_pfx cat (null) udet dayabay typ torch tag 1
    2019-09-21 19:10:45.206 INFO  [433454] [main@76]  generateoverride 67109000 num_photons 67109000 modulo 1000 integral_multiple 1 x_num_hits 67109 verbose 0
    2019-09-21 19:10:45.206 ERROR [433454] [main@90]  hitmask 64
    2019-09-21 19:10:45.206 ERROR [433454] [main@92] [ cpu generate 
    2019-09-21 19:11:35.030 ERROR [433454] [main@94] ] cpu generate 
    2019-09-21 19:11:35.031 ERROR [433454] [OContext::SetupOptiXCachePathEnvvar@284] envvar OPTIX_CACHE_PATH not defined setting it internally to /var/tmp/blyth/OptiXCache
    2019-09-21 19:11:35.054 INFO  [433454] [OContext::InitRTX@321]  --rtx 0 setting  OFF
    2019-09-21 19:11:35.263 INFO  [433454] [OContext::CheckDevices@207] 
    Device 0                        TITAN V ordinal 0 Compute Support: 7 0 Total Memory: 12652838912
    Device 1                      TITAN RTX ordinal 1 Compute Support: 7 5 Total Memory: 25396445184

    2019-09-21 19:11:35.263 ERROR [433454] [OContext::CheckDevices@228]  NULL frame_renderer : compute mode ? 
    2019-09-21 19:11:35.296 ERROR [433454] [OContext::init@364]  mode COMPUTE num_ray_type 3 stacksize_bytes 2180
    2019-09-21 19:11:35.305 INFO  [433454] [CDevice::Dump@222] Visible devices[0:TITAN_V 1:TITAN_RTX]
    2019-09-21 19:11:35.305 INFO  [433454] [CDevice::Dump@226] CDevice index 0 ordinal 0 name TITAN V major 7 minor 0 compute_capability 70 multiProcessorCount 80 totalGlobalMem 12652838912
    2019-09-21 19:11:35.305 INFO  [433454] [CDevice::Dump@226] CDevice index 1 ordinal 1 name TITAN RTX major 7 minor 5 compute_capability 75 multiProcessorCount 72 totalGlobalMem 25396445184
    2019-09-21 19:11:35.305 INFO  [433454] [CDevice::Dump@222] All devices[0:TITAN_V 1:TITAN_RTX]
    2019-09-21 19:11:35.305 INFO  [433454] [CDevice::Dump@226] CDevice index 0 ordinal 0 name TITAN V major 7 minor 0 compute_capability 70 multiProcessorCount 80 totalGlobalMem 12652838912
    2019-09-21 19:11:35.305 INFO  [433454] [CDevice::Dump@226] CDevice index 1 ordinal 1 name TITAN RTX major 7 minor 5 compute_capability 75 multiProcessorCount 72 totalGlobalMem 25396445184
    2019-09-21 19:11:35.747 ERROR [433454] [main@117] [ prelaunch 
    2019-09-21 19:11:35.749 ERROR [433454] [OContext::launch@677]  entry 0 width 0 height 0
    2019-09-21 19:11:35.749 ERROR [433454] [OContext::launch@687] VALIDATE time: 2e-05
    2019-09-21 19:11:35.749 ERROR [433454] [OContext::launch@694] COMPILE time: 5e-06
    2019-09-21 19:11:35.974 ERROR [433454] [OContext::launch@701] PRELAUNCH time: 0.22509
    2019-09-21 19:11:35.974 ERROR [433454] [main@119] ] prelaunch 
    2019-09-21 19:11:35.974 ERROR [433454] [main@121] [ upload 
    2019-09-21 19:11:35.974 ERROR [433454] [OContext::upload@788]  numBytes 4294976000
    2019-09-21 19:11:40.643 ERROR [433454] [main@123] ] upload 
    2019-09-21 19:11:40.643 ERROR [433454] [main@125] [ launch 
    2019-09-21 19:11:40.643 ERROR [433454] [OContext::launch@677]  entry 0 width 67109000 height 1
    2019-09-21 19:11:41.865 ERROR [433454] [OContext::launch@708] LAUNCH time: 1.22244
    2019-09-21 19:11:41.865 ERROR [433454] [main@127] ] launch 
    CBufSpec.Summary.cpho before TBuf : dev_ptr 0x7f76f8000000 size 67109000 num_bytes 4294976000 hexdump 0 
    2019-09-21 19:11:41.866 ERROR [433454] [main@148]  created tpho  cpho.size : 67109000 num_photons : 67109000
    2019-09-21 19:11:41.866 ERROR [433454] [main@159] [ tpho.downloadSelection4x4 
    2019-09-21 19:11:41.905 ERROR [433454] [main@161] ] tpho.downloadSelection4x4 
    2019-09-21 19:11:41.905 INFO  [433454] [main@172]  num_hits 67109 x_num_hits 67109


100M works too::

    [blyth@localhost thrustrap]$ OContext=ERROR compactionTest --generateoverride -100
    PLOG::EnvLevel adjusting loglevel by envvar   key OContext level ERROR fallback DEBUG
    2019-09-21 19:16:24.047 INFO  [441993] [Opticks::init@389] COMPUTE_MODE compute_requested 
    2019-09-21 19:16:24.047 FATAL [441993] [Opticks::init@392] OPTICKS_LEGACY_GEOMETRY_ENABLED mode is active  : ie dae src access to geometry, opticksdata  
    2019-09-21 19:16:24.053 INFO  [441993] [Opticks::initResource@773]  (legacy mode) setting IDPATH envvar for python analysis scripts [/home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae]
    2019-09-21 19:16:24.053 FATAL [441993] [Opticks::defineEventSpec@2024]  resource_pfx (null) config_pfx (null) pfx default_pfx cat (null) udet dayabay typ torch tag 1
    2019-09-21 19:16:24.053 INFO  [441993] [main@76]  generateoverride 100000000 num_photons 100000000 modulo 1000 integral_multiple 1 x_num_hits 100000 verbose 0
    2019-09-21 19:16:24.053 ERROR [441993] [main@90]  hitmask 64
    2019-09-21 19:16:24.053 ERROR [441993] [main@92] [ cpu generate 
    2019-09-21 19:17:37.804 ERROR [441993] [main@94] ] cpu generate 
    2019-09-21 19:17:37.804 ERROR [441993] [OContext::SetupOptiXCachePathEnvvar@284] envvar OPTIX_CACHE_PATH not defined setting it internally to /var/tmp/blyth/OptiXCache
    2019-09-21 19:17:37.824 INFO  [441993] [OContext::InitRTX@321]  --rtx 0 setting  OFF
    2019-09-21 19:17:37.965 INFO  [441993] [OContext::CheckDevices@207] 
    Device 0                        TITAN V ordinal 0 Compute Support: 7 0 Total Memory: 12652838912
    Device 1                      TITAN RTX ordinal 1 Compute Support: 7 5 Total Memory: 25396445184

    2019-09-21 19:17:37.966 ERROR [441993] [OContext::CheckDevices@228]  NULL frame_renderer : compute mode ? 
    2019-09-21 19:17:37.991 ERROR [441993] [OContext::init@364]  mode COMPUTE num_ray_type 3 stacksize_bytes 2180
    2019-09-21 19:17:37.998 INFO  [441993] [CDevice::Dump@222] Visible devices[0:TITAN_V 1:TITAN_RTX]
    2019-09-21 19:17:37.998 INFO  [441993] [CDevice::Dump@226] CDevice index 0 ordinal 0 name TITAN V major 7 minor 0 compute_capability 70 multiProcessorCount 80 totalGlobalMem 12652838912
    2019-09-21 19:17:37.998 INFO  [441993] [CDevice::Dump@226] CDevice index 1 ordinal 1 name TITAN RTX major 7 minor 5 compute_capability 75 multiProcessorCount 72 totalGlobalMem 25396445184
    2019-09-21 19:17:37.998 INFO  [441993] [CDevice::Dump@222] All devices[0:TITAN_V 1:TITAN_RTX]
    2019-09-21 19:17:37.998 INFO  [441993] [CDevice::Dump@226] CDevice index 0 ordinal 0 name TITAN V major 7 minor 0 compute_capability 70 multiProcessorCount 80 totalGlobalMem 12652838912
    2019-09-21 19:17:37.998 INFO  [441993] [CDevice::Dump@226] CDevice index 1 ordinal 1 name TITAN RTX major 7 minor 5 compute_capability 75 multiProcessorCount 72 totalGlobalMem 25396445184
    2019-09-21 19:17:38.423 ERROR [441993] [main@117] [ prelaunch 
    2019-09-21 19:17:38.425 ERROR [441993] [OContext::launch@677]  entry 0 width 0 height 0
    2019-09-21 19:17:38.425 ERROR [441993] [OContext::launch@687] VALIDATE time: 3.5e-05
    2019-09-21 19:17:38.425 ERROR [441993] [OContext::launch@694] COMPILE time: 1e-05
    2019-09-21 19:17:38.694 ERROR [441993] [OContext::launch@701] PRELAUNCH time: 0.268345
    2019-09-21 19:17:38.694 ERROR [441993] [main@119] ] prelaunch 
    2019-09-21 19:17:38.694 ERROR [441993] [main@121] [ upload 
    2019-09-21 19:17:38.694 ERROR [441993] [OContext::upload@788]  numBytes 6400000000
    2019-09-21 19:17:46.383 ERROR [441993] [main@123] ] upload 
    2019-09-21 19:17:46.383 ERROR [441993] [main@125] [ launch 
    2019-09-21 19:17:46.383 ERROR [441993] [OContext::launch@677]  entry 0 width 100000000 height 1
    2019-09-21 19:17:48.552 ERROR [441993] [OContext::launch@708] LAUNCH time: 2.1686
    2019-09-21 19:17:48.552 ERROR [441993] [main@127] ] launch 
    CBufSpec.Summary.cpho before TBuf : dev_ptr 0x7f4efa000000 size 100000000 num_bytes 6400000000 hexdump 0 
    2019-09-21 19:17:48.552 ERROR [441993] [main@148]  created tpho  cpho.size : 100000000 num_photons : 100000000
    2019-09-21 19:17:48.552 ERROR [441993] [main@159] [ tpho.downloadSelection4x4 
    2019-09-21 19:17:48.618 ERROR [441993] [main@161] ] tpho.downloadSelection4x4 
    2019-09-21 19:17:48.618 INFO  [441993] [main@172]  num_hits 100000 x_num_hits 100000
    [blyth@localhost thrustrap]$ 




