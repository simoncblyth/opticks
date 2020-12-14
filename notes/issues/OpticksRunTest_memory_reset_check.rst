OpticksRunTest_memory_reset_check
===================================

Testing how effective resetEvent is a cleaning up memory usage.

okc/OpticksRunTest::

     23 void test_OpticksRun_reset(Opticks* ok, unsigned nevt, bool cfg4evt)
     24 {
     25     unsigned num_photons = 10000 ;
     26     NPY<float>* gs0 = OpticksGenstep::MakeCandle(num_photons, 0);
     27     for(unsigned i=0 ; i < nevt ; i++)
     28     {
     29         LOG(info) << i ;
     30         gs0->setArrayContentIndex(i);
     31         ok->createEvent(gs0, cfg4evt);   // input argument gensteps are cloned by OpticksEvent 
     32         ok->resetEvent();
     33     }
     34     ok->saveProfile();
     35 }


::

    epsilon:optickscore blyth$ OpticksRunTest 10000 --profile
    epsilon:optickscore blyth$ TEST=OpticksRunTest ipython -i ~/opticks/ana/profile_.py -- --tag 0

    ##  plot of VM against time ramps up to almost 600M  : so reset is missing lots of memory 

    In [1]: pr.a                                                                                                                                                                                                                         
    Out[1]: 
    array([[    0.    , 67872.63  ,     0.    ,  4397.735 ],
           [    0.    ,     0.    ,     0.    ,     0.    ],
           [    0.0078,     0.0078,    18.875 ,    18.875 ],
           ...,
           [    8.4219,     0.    ,   573.0723,     0.    ],
           [    8.4219,     0.    ,   573.0723,     0.    ],
           [    8.4219,     0.    ,   573.0723,     0.    ]], dtype=float32)


    # take the genstep creation out of the loop 

    In [1]: print(pr.a)                                                                                                                                                                                 
    [[    0.     68256.75       0.      4389.347 ]
     [    0.         0.         0.         0.    ]
     [    0.         0.         9.437      9.437 ]
     ...
     [    7.2969     0.       553.1479     0.    ]
     [    7.2969     0.       553.1479     0.    ]
     [    7.2969     0.       553.1479     0.    ]]


    # simplifications by cloning inputs 

    In [1]: pr.a                                                                                                                                                                                   
    Out[1]: 
    array([[    0.    , 50597.656 ,     0.    ,  4397.752 ],
           [    0.    ,     0.    ,     0.    ,     0.    ],
           [    0.0078,     0.0078,     9.437 ,     9.437 ],
           ...,
           [    6.5938,     0.    ,   468.2139,     0.    ],
           [    6.5938,     0.    ,   468.2139,     0.    ],
           [    6.5938,     0.    ,   468.2139,     0.    ]], dtype=float32)


  
::

    2020-12-14 14:03:24.248 INFO  [412267] [test_OpticksRun_reset@44]  vm0 4407.19 vm1 4865.97 dvm 458.777 nevt 10000 leak_per_evt (MB) 0.0458777
 
    2020-12-14 14:10:20.138 INFO  [418961] [OpticksEvent::~OpticksEvent@218] OpticksEvent::~OpticksEvent PLACEHOLDER
    2020-12-14 14:10:20.138 INFO  [418961] [test_OpticksRun_reset@44]  vm0 4408.24 vm1 4816.68 dvm 408.445 nevt 10000 leak_per_evt (MB) 0.0408445

    ## progressively deleting more of OpticksEvent 

    2020-12-14 14:57:58.832 INFO  [449357] [test_OpticksRun_reset@44]  vm0 4407.2 vm1 4733.85 dvm 326.648 nevt 10000 leak_per_evt (MB) 0.0326648 cfg4evt 1 

    2020-12-14 15:05:58.700 INFO  [477125] [test_OpticksRun_reset@44]  vm0 4407.2 vm1 4615.36 dvm 208.155 nevt 10000 leak_per_evt (MB) 0.0208155 cfg4evt 0

 
    ## disabling profiling gives signficant reduction 

    2020-12-14 15:52:58.292 INFO  [540955] [test_OpticksRun_reset@44]  vm0 4407.2 vm1 4495.29 dvm 88.0889 nevt 10000 leak_per_evt (MB) 0.00880889 cfg4evt 0


::

    178 void OpticksRun::resetEvent()
    179 {
    180     LOG(LEVEL) << "[" ;
    181     OK_PROFILE("_OpticksRun::resetEvent");
    182     m_evt->reset();
    183     if(m_g4evt) m_g4evt->reset();
    184     OK_PROFILE("OpticksRun::resetEvent");
    185     LOG(LEVEL) << "]" ;
    186 }


Switching off the cfg4evt in OpticksRun by default almost halves the memory, 
but it is still trending up when it should not.::

    In [1]: pr.a                                                                                                                                                                                                                         
    Out[1]: 
    array([[    0.    , 74101.72  ,     0.    ,  4397.744 ],
           [    0.    ,     0.    ,     0.    ,     0.    ],
           [    0.0078,     0.0078,     9.437 ,     9.437 ],
           ...,
           [    4.9688,     0.    ,   324.5498,     0.    ],
           [    4.9688,     0.    ,   324.5498,     0.    ],
           [    4.9688,     0.    ,   324.5498,     0.    ]], dtype=float32)





