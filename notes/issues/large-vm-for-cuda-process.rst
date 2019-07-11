large-vm-for-cuda-process
=============================

Context
-----------

* :doc:`plugging-cfg4-leaks`


CONCLUSION : ITS A NON-ISSUE
------------------------------

* thats just how CUDA works, it does not present a performance problem


ISSUE : profiling CUDA using application on Linux shows large VM 
------------------------------------------------------------------- 

::

     OpticksProfile=ERROR ts box --generateoverride 100000

     ip tprofile.py 


CRandomEngine is pulling 5.4G
------------------------------------

::

    [blyth@localhost ana]$ ip tprofile.py

    Python 2.7.15 |Anaconda, Inc.| (default, May  1 2018, 23:32:55) 
    Type "copyright", "credits" or "license" for more information.

    IPython 5.7.0 -- An enhanced Interactive Python.
    ?         -> Introduction and overview of IPython's features.
    %quickref -> Quick reference.
    help      -> Python's own help system.
    object?   -> Details about 'object', use 'object??' for extra details.
    args: /home/blyth/opticks/ana/tprofile.py
    [2019-07-10 22:39:29,760] p284964 {<module>            :tprofile.py:83} INFO     - pfx tboolean-box tag 1 src torch det tboolean-box c2max [1.5, 2.0, 2.5] ipython True 
    path:/tmp/blyth/opticks/tboolean-box/evt/tboolean-box/torch/1/OpticksProfile.npy stamp:20190710-2239 
    lpath:/tmp/blyth/opticks/tboolean-box/evt/tboolean-box/torch/1/OpticksProfileLabels.npy stamp:20190710-2239 
     l0:_CG4::propagate l1:CG4::propagate p0:47 p1:71 v01:538.624023 
       0 :                             OpticksRun::OpticksRun :      0.000      0.000  52736.117    446.584   
       1 :                                   Opticks::Opticks :      0.000      0.000      0.000      0.000   
       2 :                                  _OpticksHub::init :      0.000      0.000      0.000      0.000   
       3 :                     _OpticksGeometry::loadGeometry :      0.012    103.748      0.012    103.748   
       4 :                      OpticksGeometry::loadGeometry :      0.441    227.472      0.430    123.724   
       5 :                               _GMergedMesh::Create :      0.477    233.216      0.035      5.744   
       6 :                         GMergedMesh::Create::Count :      0.477    233.216      0.000      0.000   
       7 :                     _GMergedMesh::Create::Allocate :      0.477    233.216      0.000      0.000   
       8 :                      GMergedMesh::Create::Allocate :      0.477    233.520      0.000      0.304   
       9 :                         GMergedMesh::Create::Merge :      0.480    234.312      0.004      0.792   
      10 :                        GMergedMesh::Create::Bounds :      0.480    234.312      0.000      0.000   
      11 :                                   OpticksHub::init :      0.609    245.596      0.129     11.284   
      12 :                                          _CG4::CG4 :      0.609    245.596      0.000      0.000   
      13 :                      _CRandomEngine::CRandomEngine :      0.609    245.596      0.000      0.000   
      14 :                       CRandomEngine::CRandomEngine :      0.984   5685.600      0.375   5440.004   
      15 :                                _CPhysics::CPhysics :      0.984   5685.600      0.000      0.000   
      16 :                                 CPhysics::CPhysics :      1.031   5687.452      0.047      1.852   
      17 :                                           CG4::CG4 :      1.039   5687.904      0.008      0.452   
      18 :                           _OpticksRun::createEvent :      2.414   9706.352      1.375   4018.448   
      19 :                            OpticksRun::createEvent :      2.414   9706.352      0.000      0.000   
      20 :                           _OKPropagator::propagate :      2.438   9706.352      0.023      0.000   
      21 :                                    _OEvent::upload :      2.469   9748.140      0.031     41.788   
      22 :                                     OEvent::upload :      2.469   9748.140      0.000      0.000   
      23 :                            _OPropagator::prelaunch :      2.480   9745.068      0.012     -3.071   
      24 :                             OPropagator::prelaunch :      3.719  10329.632      1.238    584.563   
      25 :                               _OPropagator::launch :      3.719  10329.632      0.000      0.000   
      26 :                                OPropagator::launch :      3.727  10559.008      0.008    229.376   
      27 :                          _OpIndexer::indexSequence :      3.727  10559.008      0.000      0.000   
      28 :                   _OpIndexer::indexSequenceInterop :      3.727  10559.008      0.000      0.000   
      29 :                       _OpIndexer::seqhisMakeLookup :      3.727  10559.008      0.000      0.000   
      30 :                        OpIndexer::seqhisMakeLookup :      3.738  10559.008      0.012      0.000   
      31 :                       OpIndexer::seqhisApplyLookup :      3.738  10559.008      0.000      0.000   



All from TCURAND::

      00 :                             OpticksRun::OpticksRun :      0.000      0.000  53489.781    446.584   
       1 :                                   Opticks::Opticks :      0.000      0.000      0.000      0.000   
       2 :                                  _OpticksHub::init :      0.000      0.000      0.000      0.000   
       3 :                     _OpticksGeometry::loadGeometry :      0.012    103.748      0.012    103.748   
       4 :                      OpticksGeometry::loadGeometry :      0.461    227.472      0.449    123.724   
       5 :                               _GMergedMesh::Create :      0.492    233.216      0.031      5.744   
       6 :                         GMergedMesh::Create::Count :      0.492    233.216      0.000      0.000   
       7 :                     _GMergedMesh::Create::Allocate :      0.492    233.216      0.000      0.000   
       8 :                      GMergedMesh::Create::Allocate :      0.492    233.520      0.000      0.304   
       9 :                         GMergedMesh::Create::Merge :      0.496    234.312      0.004      0.792   
      10 :                        GMergedMesh::Create::Bounds :      0.496    234.312      0.000      0.000   
      11 :                                   OpticksHub::init :      0.613    245.596      0.117     11.284   
      12 :                                          _CG4::CG4 :      0.613    245.596      0.000      0.000   
      13 :                      _CRandomEngine::CRandomEngine :      0.613    245.596      0.000      0.000   
      14 :                                  _TCURAND::TCURAND :      0.613    245.596      0.000      0.000   
      15 :                                   TCURAND::TCURAND :      0.980   5685.636      0.367   5440.040   
      16 :                       CRandomEngine::CRandomEngine :      0.980   5685.636      0.000      0.000   
      17 :                                _CPhysics::CPhysics :      0.984   5685.636      0.004      0.000   
      18 :                                 CPhysics::CPhysics :      1.023   5687.364      0.039      1.728   
      19 :                                           CG4::CG4 :      1.031   5687.904      0.008      0.540   
      20 :                           _OpticksRun::createEvent :      2.516   9706.352      1.484   4018.448   
      21 :                            OpticksRun::createEvent :      2.516   9706.352      0.000      0.000   
      22 :                           _OKPropagator::propagate :      2.547   9706.352      0.031      0.000   
      23 :                                    _OEvent::upload :      2.574   9748.140      0.027     41.788   
      24 :                                     OEvent::upload :      2.574   9748.140      0.000      0.000   
      25 :                            _OPropagator::prelaunch :      2.586   9745.068      0.012     -3.071   
      26 :                             OPropagator::prelaunch :      3.895  10329.148      1.309    584.080   
      27 :                               _OPropagator::launch :      3.895  10329.148      0.000      0.000   
      28 :                                OPropagator::launch :      3.902  10558.524      0.008    229.376   
      29 :                          _OpIndexer::indexSequence :      3.902  10558.524      0.000      0.000   
      30 :                   _OpIndexer::indexSequenceInterop :      3.902  10558.524      0.000      0.000   
      31 :                       _OpIndexer::seqhisMakeLookup :      3.902  10558.524      0.000      0.000   




Could understand 500M or so, but 10x that ?::

    In [3]: 100000*16*16*8/(1000*1000)
    Out[3]: 204



TCURANDTest also takes more than 5G  
------------------------------------------

With ni 100,000::

    cd thrustrap/tests
    OpticksProfile=ERROR TCURANDImp=ERROR TEST=TCURANDTest om-t
    ...
    2019-07-11 10:07:33.875 INFO  [430857] [Opticks::initResource@654]  (legacy mode) setting IDPATH envvar for python analysis scripts [/home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae]
    2019-07-11 10:07:33.875 ERROR [430857] [OpticksProfile::stamp@147] _TCURANDImp::TCURANDImp_0 (0.0078125,0.00683594,103.62,103.62)
    2019-07-11 10:07:34.151 ERROR [430857] [TCURANDImp<T>::init@40] TCURANDImp ox 100000,16,16
    2019-07-11 10:07:34.265 ERROR [430857] [OpticksProfile::stamp@147] TCURANDImp::TCURANDImp_0 (0.396973,0.38916,5547.4,5443.78)
    2019-07-11 10:07:34.265 ERROR [430857] [TCURANDImp<T>::setIBase@59]  ibase 0
    2019-07-11 10:07:34.441 INFO  [430857] [TCURANDTest::save@48]  save /tmp/blyth/opticks/TCURANDTest_0.npy
    (100000, 16, 16)
    ...
    2019-07-11 10:07:37.277 INFO  [430857] [OpticksProfile::dump@273] TCURANDTest dir 
    2019-07-11 10:07:37.278 INFO  [430857] [BTimesTable::dump@145] TCURANDTest filter: NONE
     diffListedTime           Time      DeltaTime             VM        DeltaVM
        0          0.000           0.000       7653.868          0.000        194.728 : OpticksRun::OpticksRun
        1          0.001           0.001          0.001          0.000          0.000 : Opticks::Opticks_0
        2          0.007           0.008          0.007        103.620        103.620 : _TCURANDImp::TCURANDImp_0
        3          0.389           0.397          0.389       5547.404       5443.784 : TCURANDImp::TCURANDImp_0
    2019-07-11 10:07:37.278 INFO  [430857] [OpticksProfile::dump@278]  npy 4,1,4 


Reduce ni to 1000, shows not much reduction::

    2019-07-11 10:15:51.649 INFO  [443658] [OpticksProfile::dump@273] TCURANDTest dir 
    2019-07-11 10:15:51.649 INFO  [443658] [BTimesTable::dump@145] TCURANDTest filter: NONE
     diffListedTime           Time      DeltaTime             VM        DeltaVM
        0          0.000           0.000       8150.994          0.000        194.728 : OpticksRun::OpticksRun_0
        1          0.000           0.000          0.000          0.000          0.000 : Opticks::Opticks_0
        2          0.007           0.008          0.007        103.620        103.620 : _TCURANDImp::TCURANDImp_0
        3          0.230           0.238          0.230       5120.124       5016.504 : TCURANDImp::TCURANDImp_0
    2019-07-11 10:15:51.649 INFO  [443658] [OpticksProfile::dump@278]  npy 4,1,4 


Pinned it down to thrust::device_vector dox taking 5G::

    2019-07-11 10:33:25.880 INFO  [12308] [main@60] ./TCURANDTest
    2019-07-11 10:33:25.881 ERROR [12308] [OpticksProfile::stamp@147] OpticksRun::OpticksRun_0 (0,9205.88,0,194.728)
    2019-07-11 10:33:25.882 ERROR [12308] [OpticksProfile::stamp@147] Opticks::Opticks_0 (0,0,0,0)
    2019-07-11 10:33:25.882 INFO  [12308] [Opticks::init@318] INTEROP_MODE
    2019-07-11 10:33:25.882 INFO  [12308] [Opticks::configure@1844]  setting CUDA_VISIBLE_DEVICES envvar internally to 1
    2019-07-11 10:33:25.888 INFO  [12308] [Opticks::initResource@654]  (legacy mode) setting IDPATH envvar for python analysis scripts [/home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae]
    2019-07-11 10:33:25.888 ERROR [12308] [OpticksProfile::stamp@147] _TCURANDImp::TCURANDImp_0 (0.00683594,0.00683594,103.62,103.62)
    2019-07-11 10:33:25.888 ERROR [12308] [OpticksProfile::stamp@147] _dvec_dox_0 (0.00683594,0,103.62,0)
    2019-07-11 10:33:26.209 ERROR [12308] [OpticksProfile::stamp@147] dvec_dox_0 (0.327148,0.320312,5118.1,5014.48)
    2019-07-11 10:33:26.209 ERROR [12308] [OpticksProfile::stamp@147] _TRngBuf::TRngBuf_0 (0.327148,0,5118.1,0)
    2019-07-11 10:33:26.209 ERROR [12308] [OpticksProfile::stamp@147] TRngBuf::TRngBuf_0 (0.327148,0,5118.1,0)
    2019-07-11 10:33:26.209 ERROR [12308] [TCURANDImp<T>::init@42] TCURANDImp ox 1000,16,16 elem 256000
    2019-07-11 10:33:26.211 ERROR [12308] [OpticksProfile::stamp@147] TCURANDImp::TCURANDImp_0 (0.329102,0.00195312,5120.12,2.02393)
    2019-07-11 10:33:26.211 ERROR [12308] [TCURANDImp<T>::setIBase@79]  ibase 0
    2019-07-11 10:33:26.214 INFO  [12308] [TCURANDTest::save@48]  save /tmp/blyth/opticks/TCURANDTest_0.npy
    2019-07-11 10:33:26.216 ERROR [12308] [TCURANDImp<T>::setIBase@79]  ibase 1000
    2019-07-11 10:33:26.217 INFO  [12308] [TCURANDTest::save@48]  save /tmp/blyth/opticks/TCURANDTest_1000.npy
    2019-07-11 10:33:26.219 ERROR [12308] [TCURANDImp<T>::setIBase@79]  ibase 2000
    2019-07-11 10:33:26.220 INFO  [12308] [TCURANDTest::save@48]  save /tmp/blyth/opticks/TCURANDTest_2000.npy
    2019-07-11 10:33:26.222 INFO  [12308] [OpticksProfile::dump@273] TCURANDTest dir 
    2019-07-11 10:33:26.223 INFO  [12308] [BTimesTable::dump@145] TCURANDTest filter: NONE
     diffListedTime           Time      DeltaTime             VM        DeltaVM
        0          0.000           0.000       9205.882          0.000        194.728 : OpticksRun::OpticksRun_0
        1          0.000           0.000          0.000          0.000          0.000 : Opticks::Opticks_0
        2          0.007           0.007          0.007        103.620        103.620 : _TCURANDImp::TCURANDImp_0
        3          0.000           0.007          0.000        103.620          0.000 : _dvec_dox_0
        4          0.320           0.327          0.320       5118.096       5014.476 : dvec_dox_0
        5          0.000           0.327          0.000       5118.096          0.000 : _TRngBuf::TRngBuf_0
        6          0.000           0.327          0.000       5118.096          0.000 : TRngBuf::TRngBuf_0
        7          0.002           0.329          0.002       5120.120          2.024 : TCURANDImp::TCURANDImp_0
    2019-07-11 10:33:26.223 INFO  [12308] [OpticksProfile::dump@278]  npy 8,1,4 


How did it manage to take 5000M when the array only 2M, probably thats CUDA context::

    In [1]: 256000*8
    Out[1]: 2048000

    In [2]: 256000*8/1e6
    Out[2]: 2.048



Search
---------

* :google:`cuda virtual memory profile`


* https://devtalk.nvidia.com/default/topic/1044446/cuda-programming-and-performance/high-virtual-memory-consumption-on-linux-for-cuda-programs-is-it-possible-to-avoid-it-/

Apparently its harmless


* https://stackoverflow.com/questions/11631191/why-does-the-cuda-runtime-reserve-80-gib-virtual-memory-upon-initialization

talonmies:

    Nothing to do with scratch space, it is the result of the addressing system
    that allows unified andressing and peer to peer access between host and
    multiple GPUs. The CUDA driver registers all the GPU(s) memory + host memory in
    a single virtual address space using the kernel's virtual memory system. It
    isn't actually memory consumption, per se, it is just a "trick" to map all the
    available address spaces into a linear virtual space for unified addressing.


Using both TITAN V and TITAN RTX pushes the VM to 9G::

    [blyth@localhost tests]$ OpticksProfile=ERROR TCURANDImp=ERROR TCURANDTest --cvd 0,1
    PLOG::EnvLevel adjusting loglevel by envvar   key OpticksProfile level ERROR fallback DEBUG
    PLOG::EnvLevel adjusting loglevel by envvar   key TCURANDImp level ERROR fallback DEBUG
    PLOG::EnvLevel adjusting loglevel by envvar   key TCURANDImp level ERROR fallback DEBUG
    2019-07-11 10:50:03.262 INFO  [39184] [main@60] TCURANDTest
    2019-07-11 10:50:03.264 ERROR [39184] [OpticksProfile::stamp@147] OpticksRun::OpticksRun_0 (0,10203.3,0,194.728)
    2019-07-11 10:50:03.264 ERROR [39184] [OpticksProfile::stamp@147] Opticks::Opticks_0 (0.000976562,0.000976562,0,0)
    2019-07-11 10:50:03.264 INFO  [39184] [Opticks::init@318] INTEROP_MODE
    2019-07-11 10:50:03.265 INFO  [39184] [Opticks::configure@1844]  setting CUDA_VISIBLE_DEVICES envvar internally to 0,1
    2019-07-11 10:50:03.271 INFO  [39184] [Opticks::initResource@654]  (legacy mode) setting IDPATH envvar for python analysis scripts [/home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae]
    2019-07-11 10:50:03.271 ERROR [39184] [OpticksProfile::stamp@147] _TCURANDImp::TCURANDImp_0 (0.0078125,0.00683594,103.62,103.62)
    2019-07-11 10:50:03.271 ERROR [39184] [OpticksProfile::stamp@147] _dvec_dox_0 (0.0078125,0,103.62,0)
    2019-07-11 10:50:03.639 ERROR [39184] [OpticksProfile::stamp@147] dvec_dox_0 (0.375977,0.368164,9442.66,9339.04)
    2019-07-11 10:50:03.639 ERROR [39184] [OpticksProfile::stamp@147] _TRngBuf::TRngBuf_0 (0.375977,0,9442.66,0)
    2019-07-11 10:50:03.639 ERROR [39184] [OpticksProfile::stamp@147] TRngBuf::TRngBuf_0 (0.375977,0,9442.66,0)
    2019-07-11 10:50:03.640 ERROR [39184] [TCURANDImp<T>::init@42] TCURANDImp ox 1000,16,16 elem 256000
    2019-07-11 10:50:03.641 ERROR [39184] [OpticksProfile::stamp@147] TCURANDImp::TCURANDImp_0 (0.37793,0.00195312,9444.7,2.04395)
    2019-07-11 10:50:03.641 ERROR [39184] [TCURANDImp<T>::setIBase@79]  ibase 0
    2019-07-11 10:50:03.645 INFO  [39184] [TCURANDTest::save@48]  save /tmp/blyth/opticks/TCURANDTest_0.npy
    2019-07-11 10:50:03.648 ERROR [39184] [TCURANDImp<T>::setIBase@79]  ibase 1000
    2019-07-11 10:50:03.649 INFO  [39184] [TCURANDTest::save@48]  save /tmp/blyth/opticks/TCURANDTest_1000.npy
    2019-07-11 10:50:03.650 ERROR [39184] [TCURANDImp<T>::setIBase@79]  ibase 2000
    2019-07-11 10:50:03.652 INFO  [39184] [TCURANDTest::save@48]  save /tmp/blyth/opticks/TCURANDTest_2000.npy
    2019-07-11 10:50:03.654 INFO  [39184] [OpticksProfile::dump@273] TCURANDTest dir 
    2019-07-11 10:50:03.654 INFO  [39184] [BTimesTable::dump@145] TCURANDTest filter: NONE
     diffListedTime           Time      DeltaTime             VM        DeltaVM
        0          0.000           0.000      10203.264          0.000        194.728 : OpticksRun::OpticksRun_0
        1          0.001           0.001          0.001          0.000          0.000 : Opticks::Opticks_0
        2          0.007           0.008          0.007        103.620        103.620 : _TCURANDImp::TCURANDImp_0
        3          0.000           0.008          0.000        103.620          0.000 : _dvec_dox_0
        4          0.368           0.376          0.368       9442.656       9339.036 : dvec_dox_0
        5          0.000           0.376          0.000       9442.656          0.000 : _TRngBuf::TRngBuf_0
        6          0.000           0.376          0.000       9442.656          0.000 : TRngBuf::TRngBuf_0
        7          0.002           0.378          0.002       9444.700          2.044 : TCURANDImp::TCURANDImp_0
    2019-07-11 10:50:03.654 INFO  [39184] [OpticksProfile::dump@278]  npy 8,1,4 
    [blyth@localhost tests]$ 


With --cvd 0 TITAN V::

     diffListedTime           Time      DeltaTime             VM        DeltaVM
        0          0.000           0.000      10292.008          0.000        194.728 : OpticksRun::OpticksRun_0
        1          0.001           0.001          0.001          0.000          0.000 : Opticks::Opticks_0
        2          0.008           0.009          0.008        103.620        103.620 : _TCURANDImp::TCURANDImp_0
        3          0.000           0.009          0.000        103.620          0.000 : _dvec_dox_0
        4          0.357           0.366          0.357       5248.212       5144.592 : dvec_dox_0
        5          0.000           0.366          0.000       5248.212          0.000 : _TRngBuf::TRngBuf_0
        6          0.000           0.366          0.000       5248.212          0.000 : TRngBuf::TRngBuf_0
        7          0.003           0.369          0.003       5250.316          2.104 : TCURANDImp::TCURANDImp_0
    2019-07-11 10:51:32.394 INFO  [41452] [OpticksProfile::dump@278]  npy 8,1,4 

With --cvd 1 TITAN RTX::

    2019-07-11 10:52:18.787 INFO  [42634] [BTimesTable::dump@145] TCURANDTest filter: NONE
     diffListedTime           Time      DeltaTime             VM        DeltaVM
        0          0.000           0.000      10338.507          0.000        194.728 : OpticksRun::OpticksRun_0
        1          0.000           0.000          0.000          0.000          0.000 : Opticks::Opticks_0
        2          0.007           0.007          0.007        103.620        103.620 : _TCURANDImp::TCURANDImp_0
        3          0.000           0.007          0.000        103.620          0.000 : _dvec_dox_0
        4          0.260           0.267          0.260       5117.992       5014.372 : dvec_dox_0
        5          0.000           0.267          0.000       5117.992          0.000 : _TRngBuf::TRngBuf_0
        6          0.000           0.267          0.000       5118.124          0.132 : TRngBuf::TRngBuf_0
        7          0.002           0.269          0.002       5120.120          1.996 : TCURANDImp::TCURANDImp_0
    2019-07-11 10:52:18.787 INFO  [42634] [OpticksProfile::dump@278]  npy 8,1,4 


