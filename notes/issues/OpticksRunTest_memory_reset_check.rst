OpticksRunTest_memory_reset_check
===================================

Testing how effective resetEvent is a cleaning up memory usage.

okc/OpticksRunTest::

     23 void test_OpticksRun_reset(Opticks* ok, unsigned nevt)
     24 {
     25     unsigned num_photons = 10000 ;
     26     NPY<float>* gs0 = OpticksGenstep::MakeCandle(num_photons, 0);
     27     for(unsigned i=0 ; i < nevt ; i++)
     28     {
     29         LOG(info) << i ;
     30         gs0->setArrayContentIndex(i);
     31         bool cfg4evt = false ;
     32         ok->createEvent(gs0, cfg4evt);
     33         ok->resetEvent();
     34     }
     35     ok->saveProfile();
     36 }



::

    epsilon:optickscore blyth$ OpticksRunTest 10000
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





