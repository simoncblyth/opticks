SG-GPU-using-tests-all-more-than-10s
=========================================

SG tests that use GPU have a big overhead.

* startup "pedestal" : YES, driver not holding onto GPU devices in headless server, unless run daemon
* try masking out GPUs : made no difference


FIXED : using daemon nvidia-persistenced 
----------------------------------------------

Total test time without G4 ~119s::


    238/241 Test #238: OKTest.OTracerTest ...........................   Passed    2.05 sec
            Start 239: OKTest.LogTest
    239/241 Test #239: OKTest.LogTest ...............................   Passed    0.02 sec
            Start 240: OKTest.VizTest
    240/241 Test #240: OKTest.VizTest ...............................   Passed    8.95 sec
            Start 241: OKTest.TrivialTest
    241/241 Test #241: OKTest.TrivialTest ...........................   Passed    0.02 sec

    100% tests passed, 0 tests failed out of 241

    Total Test time (real) = 118.74 sec
    opticks-t- : use -V to show output
    [simon@localhost opticks]$ 
    [simon@localhost opticks]$ 


Search 
-------

* :google:`cuda overhead`
* :google:`cuda startup time`
* :google:`cuda init seconds`
* :google:`nvidia persistence mode`

* https://devtalk.nvidia.com/default/topic/480579/?comment=3445429


Deprecated Persistence Mode
-------------------------------

* http://docs.nvidia.com/deploy/driver-persistence/index.html#persistence-mode


nvidia-persistenced
----------------------

::

    [root@localhost opticks]# man nvidia-persistenced
    [root@localhost opticks]# nvidia-persistenced
    [root@localhost opticks]# ps aux | grep nvidia
    root     136020 73.1  0.0   8352   728 ?        Ss   15:42   0:08 nvidia-persistenced
    root     136022  0.0  0.0 103256   852 pts/1    S+   15:42   0:00 grep nvidia
    [root@localhost opticks]# 


After starting the daemon no init delay::

    [root@localhost opticks]# nvidia-smi
    Mon Oct 23 15:43:11 2017       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 367.48                 Driver Version: 367.48                    |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  Tesla K80           On   | 0000:05:00.0     Off |                    0 |
    | N/A   33C    P8    26W / 149W |      0MiB / 11439MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
    |   1  Tesla K80           On   | 0000:06:00.0     Off |                    0 |
    | N/A   27C    P8    28W / 149W |      0MiB / 11439MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
    |   2  Tesla K80           On   | 0000:84:00.0     Off |                    0 |
    | N/A   29C    P8    25W / 149W |      0MiB / 11439MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
    |   3  Tesla K80           On   | 0000:85:00.0     Off |                    0 |
    | N/A   28C    P8    30W / 149W |      0MiB / 11439MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
                                                                                   
    +-----------------------------------------------------------------------------+
    | Processes:                                                       GPU Memory |
    |  GPU       PID  Type  Process name                               Usage      |
    |=============================================================================|
    |  No running processes found                                                 |
    +-----------------------------------------------------------------------------+
    [root@localhost opticks]# 




Also see 10s with nvidia-smi
------------------------------

::

    [simon@localhost opticks]$ date ; nvidia-smi ; date 
    Mon Oct 23 15:31:22 CST 2017
    Mon Oct 23 15:31:32 2017       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 367.48                 Driver Version: 367.48                    |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  Tesla K80           Off  | 0000:05:00.0     Off |                    0 |
    | N/A   38C    P0    57W / 149W |      0MiB / 11439MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
    |   1  Tesla K80           Off  | 0000:06:00.0     Off |                    0 |
    | N/A   31C    P0    66W / 149W |      0MiB / 11439MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
    |   2  Tesla K80           Off  | 0000:84:00.0     Off |                    0 |
    | N/A   32C    P0    57W / 149W |      0MiB / 11439MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
    |   3  Tesla K80           Off  | 0000:85:00.0     Off |                    0 |
    | N/A   30C    P0    72W / 149W |      0MiB / 11439MiB |     97%      Default |
    +-------------------------------+----------------------+----------------------+
                                                                                   
    +-----------------------------------------------------------------------------+
    | Processes:                                                       GPU Memory |
    |  GPU       PID  Type  Process name                               Usage      |
    |=============================================================================|
    |  No running processes found                                                 |
    +-----------------------------------------------------------------------------+
    Mon Oct 23 15:31:32 CST 2017
    [simon@localhost opticks]$ 




nvidia-smi
-------------

::

    [simon@localhost opticks]$ nvidia-smi -h
    NVIDIA System Management Interface -- v367.48

    NVSMI provides monitoring information for Tesla and select Quadro devices.
    The data is presented in either a plain text or an XML format, via stdout or a file.
    NVSMI also provides several management operations for changing the device state.

    Note that the functionality of NVSMI is exposed through the NVML C-based
    library. See the NVIDIA developer website for more information about NVML.
    Python wrappers to NVML are also available.  The output of NVSMI is
    not guaranteed to be backwards compatible; NVML and the bindings are backwards
    compatible.

    http://developer.nvidia.com/nvidia-management-library-nvml/
    http://pypi.python.org/pypi/nvidia-ml-py/
    Supported products:
    - Full Support
        - All Tesla products, starting with the Fermi architecture
        - All Quadro products, starting with the Fermi architecture
        - All GRID products, starting with the Kepler architecture
        - GeForce Titan products, starting with the Kepler architecture
    - Limited Support
        - All Geforce products, starting with the Fermi architecture
    nvidia-smi [OPTION1 [ARG1]] [OPTION2 [ARG2]] ...

        -h,   --help                Print usage information and exit.

    ...



10s for optix::Context::create ?
----------------------------------------------

::

    [simon@localhost optixrap]$ ORayleighTest 
    ...
    2017-10-23 15:01:18.275 INFO  [135699] [Opticks::makeSimpleTorchStep@1386] Opticks::makeSimpleTorchStep config  cfg NULL
    2017-10-23 15:01:18.275 INFO  [135699] [SLog::operator@15] OpticksHub::OpticksHub DONE
    2017-10-23 15:01:18.275 INFO  [135699] [OScene::init@92] OScene::init START
    2017-10-23 15:01:18.275 INFO  [135699] [OScene::init@105] OScene::init optix::Context::create() START 
    2017-10-23 15:01:28.225 INFO  [135699] [OScene::init@107] OScene::init optix::Context::create() DONE 
    2017-10-23 15:01:28.225 INFO  [135699] [OScene::init@111] OScene::init (OContext) stack_size_bytes: 2180
    2017-10-23 15:01:28.225 INFO  [135699] [OScene::init@129] OScene::init ggeobase identifier : GGeo


::

    089 void OScene::init()
     90 {
     91     //if(m_verbosity > 0)
     92     LOG(info) << "OScene::init START" ;
     93 
     94     m_timer->setVerbose(true);
     95     m_timer->start();
     96 
     97     std::string builder_   = m_cfg->getBuilder();
     98     std::string traverser_ = m_cfg->getTraverser();
     99     const char* builder   = builder_.empty() ? NULL : builder_.c_str() ;
    100     const char* traverser = traverser_.empty() ? NULL : traverser_.c_str() ;
    101 
    102 
    103     OContext::Mode_t mode = m_ok->isCompute() ? OContext::COMPUTE : OContext::INTEROP ;
    104 
    105     LOG(info) << "OScene::init optix::Context::create() START " ;
    106     optix::Context context = optix::Context::create();
    107     LOG(info) << "OScene::init optix::Context::create() DONE " ;
    108 



Get same ~10s with one-by-one masking:: 

     1059  CUDA_VISIBLE_DEVICES=0 ORayleighTest 
     1060  CUDA_VISIBLE_DEVICES=1 ORayleighTest 
     1061  CUDA_VISIBLE_DEVICES=2 ORayleighTest 
     1062  CUDA_VISIBLE_DEVICES=3 ORayleighTest 


Same delay seen with pure-CUDA cudaGetDevicePropertiesTest



10s pedestal for all GPU using tests ?
-------------------------------------------

::

    203/240 Test #203: CUDARapTest.cudaGetDevicePropertiesTest ......   Passed   10.70 sec
            Start 204: ThrustRapTest.CBufSpecTest
    204/240 Test #204: ThrustRapTest.CBufSpecTest ...................   Passed   10.69 sec
            Start 205: ThrustRapTest.TBufTest
    205/240 Test #205: ThrustRapTest.TBufTest .......................   Passed   10.71 sec
            Start 206: ThrustRapTest.expandTest
    206/240 Test #206: ThrustRapTest.expandTest .....................   Passed   10.70 sec
            Start 207: ThrustRapTest.iexpandTest
    207/240 Test #207: ThrustRapTest.iexpandTest ....................   Passed   10.69 sec
            Start 208: ThrustRapTest.issue628Test
    208/240 Test #208: ThrustRapTest.issue628Test ...................   Passed   10.71 sec
            Start 209: ThrustRapTest.printfTest
    209/240 Test #209: ThrustRapTest.printfTest .....................   Passed   10.70 sec
            Start 210: ThrustRapTest.repeated_rangeTest
    210/240 Test #210: ThrustRapTest.repeated_rangeTest .............   Passed   10.66 sec
            Start 211: ThrustRapTest.strided_rangeTest
    211/240 Test #211: ThrustRapTest.strided_rangeTest ..............   Passed   10.64 sec
            Start 212: ThrustRapTest.strided_repeated_rangeTest
    212/240 Test #212: ThrustRapTest.strided_repeated_rangeTest .....   Passed   10.61 sec
            Start 213: OptiXRapTest.OPropertyLibTest
    213/240 Test #213: OptiXRapTest.OPropertyLibTest ................   Passed   10.39 sec
            Start 214: OptiXRapTest.OScintillatorLibTest
    214/240 Test #214: OptiXRapTest.OScintillatorLibTest ............   Passed   11.75 sec
            Start 215: OptiXRapTest.OOTextureTest
    215/240 Test #215: OptiXRapTest.OOTextureTest ...................   Passed   12.01 sec
            Start 216: OptiXRapTest.OOMinimalTest
    216/240 Test #216: OptiXRapTest.OOMinimalTest ...................   Passed   11.97 sec
            Start 217: OptiXRapTest.OOContextTest
    217/240 Test #217: OptiXRapTest.OOContextTest ...................   Passed   11.83 sec
            Start 218: OptiXRapTest.OOContextUploadDownloadTest
    218/240 Test #218: OptiXRapTest.OOContextUploadDownloadTest .....   Passed   11.91 sec
            Start 219: OptiXRapTest.LTOOContextUploadDownloadTest
    219/240 Test #219: OptiXRapTest.LTOOContextUploadDownloadTest ...   Passed   12.01 sec
            Start 220: OptiXRapTest.OOboundaryTest
    220/240 Test #220: OptiXRapTest.OOboundaryTest ..................   Passed   11.96 sec
            Start 221: OptiXRapTest.OOboundaryLookupTest
    221/240 Test #221: OptiXRapTest.OOboundaryLookupTest ............   Passed   11.97 sec
            Start 222: OptiXRapTest.OOtex0Test
    222/240 Test #222: OptiXRapTest.OOtex0Test ......................   Passed   11.91 sec
            Start 223: OptiXRapTest.OOtexTest
    223/240 Test #223: OptiXRapTest.OOtexTest .......................   Passed   11.89 sec
            Start 224: OptiXRapTest.bufferTest
    224/240 Test #224: OptiXRapTest.bufferTest ......................   Passed   11.94 sec
            Start 225: OptiXRapTest.OEventTest
    225/240 Test #225: OptiXRapTest.OEventTest ......................   Passed   12.23 sec
            Start 226: OptiXRapTest.OInterpolationTest
    226/240 Test #226: OptiXRapTest.OInterpolationTest ..............***Failed   17.39 sec
            Start 227: OptiXRapTest.ORayleighTest
    227/240 Test #227: OptiXRapTest.ORayleighTest ...................***Exception: Other 13.49 sec
            Start 228: OptiXRapTest.intersect_analytic_test
    228/240 Test #228: OptiXRapTest.intersect_analytic_test .........   Passed   12.23 sec
            Start 229: OptiXRapTest.Roots3And4Test
    229/240 Test #229: OptiXRapTest.Roots3And4Test ..................   Passed   12.07 sec
            Start 230: OKOPTest.OpIndexerTest
    230/240 Test #230: OKOPTest.OpIndexerTest .......................   Passed   12.35 sec
            Start 231: OKOPTest.OpSeederTest
    231/240 Test #231: OKOPTest.OpSeederTest ........................   Passed   18.75 sec

    ...

    240/240 Test #240: OKTest.TrivialTest ...........................   Passed    0.02 sec

    99% tests passed, 2 tests failed out of 240

    Total Test time (real) = 482.15 sec

    The following tests FAILED:
        226 - OptiXRapTest.OInterpolationTest (Failed)         ## missing '/tmp/simon/opticks/InterpolationTest/CInterpolationTest_interpol.npy' 
        227 - OptiXRapTest.ORayleighTest (OTHER_FAULT)         ## 
    Errors while running CTest
    opticks-t- : use -V to show output
    [simon@localhost opticks]$ 



OInterpolationTest : missing file 
------------------------------------

* missing '/tmp/simon/opticks/InterpolationTest/CInterpolationTest_interpol.npy' 
* suspect the missing npy comes from cfg4 which is not yet installed on SG


ORayleighTest : rayleigh_buffer dimension ?
-------------------------------------------

* OptiX version issue presumably, SG is with 411 ? D with 380

::

    2017-10-18 20:58:48.396 INFO  [31943] [OContext::close@239] OContext::close setEntryPointCount done.
    2017-10-18 20:58:48.546 INFO  [31943] [OContext::close@245] OContext::close m_cfg->apply() done.
    terminate called after throwing an instance of 'optix::Exception'
      what():  Type mismatch (Details: Function "RTresult _rtContextLaunch2D(RTcontext, unsigned int, RTsize, RTsize)" 
      caught exception: Variable "rayleigh_buffer" assigned type Buffer(2d, 16 byte element).  Should be Buffer(1d, 16 byte element).)
    Aborted
    [simon@localhost opticks]$ 





