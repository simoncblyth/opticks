revive_multievent_for_profiling_overheads
============================================


Context
-----------

* :doc:`tboolean-generateoverride-photon-scanning`


Opticks photon scanning performance begs the question : what are the overheads ?






1./256. = 0.00390625
------------------------

::

    In [8]: np.arange(10, dtype=np.float64)/256.
    Out[8]: 
    array([0.        , 0.00390625, 0.0078125 , 0.01171875, 0.015625  ,
           0.01953125, 0.0234375 , 0.02734375, 0.03125   , 0.03515625])


::

     06 double BTimeStamp::RealTime()
      7 {
      8     ptime t(microsec_clock::universal_time());
      9     time_duration d = t.time_of_day();
     10     double unit = 1e9 ;
     11     return d.total_nanoseconds()/unit ;
     12 }



* https://stackoverflow.com/questions/6749621/how-to-create-a-high-resolution-timer-in-linux-to-measure-program-performance


* https://www.boost.org/doc/libs/1_43_0/doc/html/date_time/posix_time.html




Approach, "--multievent 10 --nog4propagate" runs of 1M photons each
-----------------------------------------------------------------------

::

     tmp ; rm -rf scan-ph ; OpticksProfile=ERROR ts box --pfx scan-ph --cat cvd_1_rtx_1 --generateoverride -1 --compute --production --cvd 1 --rtx 1 --multievent 10 --nog4propagate

     ip profile.py --tag 0 --cat cvd_1_rtx_1_1M

DONE
-----

* reduced time between launches from 0.25s to 0.05s by making ViewNPY lazy on bounds
* get rid of OpticksRun m_g4evt when using  "--nog4propagate" it takes 0.025s (just like m_evt)



TODO : find cause of random sprinkles of additional 0.0039
------------------------------------------------------------------

::

    In [3]: tt[:, 10:33]
    Out[3]: 
    array([[0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.0156, 0.    , 0.    , 0.    , 0.0078, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ],
           [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.0156, 0.    , 0.    , 0.    , 0.0078, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ],
           [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.0156, 0.    , 0.0039, 0.    , 0.0039, 0.    , 0.    , 0.    , 0.    , 0.0039, 0.    , 0.    , 0.    , 0.    ],
           [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.0156, 0.    , 0.    , 0.    , 0.0078, 0.    , 0.    , 0.0039, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ],
           [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.0156, 0.    , 0.    , 0.    , 0.0078, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ],
           [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.0039, 0.    , 0.0117, 0.    , 0.    , 0.    , 0.0078, 0.0039, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ],
           [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.0156, 0.    , 0.    , 0.    , 0.0078, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ],
           [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.0117, 0.    , 0.0039, 0.    , 0.0039, 0.    , 0.    , 0.0039, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ],
           [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.0156, 0.    , 0.    , 0.    , 0.0078, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ]])

    In [5]: 0.0117+0.0039
    Out[5]: 0.0156





AVOIDED ISSUE  by "--nog4propagate"
-------------------------------------------

::

    OpticksProfile=ERROR ts box --pfx scan-ph --cat cvd_1_rtx_0 --generateoverride 1 --compute --production --cvd 1 --rtx 0 --multievent 2 -D

    ...

    2019-07-16 13:39:03.901 ERROR [122188] [OpticksProfile::stamp@180] CRunAction::BeginOfRunAction_1 (7.71484,0,10672.3,0)
    2019-07-16 13:39:03.901 ERROR [122188] [OpticksProfile::stamp@180] _CInputPhotonSource::GeneratePrimaryVertex_1 (7.71484,0,10672.3,0)
    OKG4Test: /home/blyth/opticks/sysrap/STranche.cc:24: unsigned int STranche::tranche_size(unsigned int) const: Assertion `i < num_tranche && " trance indices must be from 0 to tr.num_tranche - 1 inclusive  "' failed.
    
    #4  0x00007fffe74ecd28 in STranche::tranche_size (this=0x768a670, i=1) at /home/blyth/opticks/sysrap/STranche.cc:24
    #5  0x00007ffff4c96205 in CInputPhotonSource::GeneratePrimaryVertex (this=0x768bec0, evt=0x7a6e5d0) at /home/blyth/opticks/cfg4/CInputPhotonSource.cc:174
    #6  0x00007ffff4c726de in CPrimaryGeneratorAction::GeneratePrimaries (this=0x768a560, event=0x7a6e5d0) at /home/blyth/opticks/cfg4/CPrimaryGeneratorAction.cc:15
    #7  0x00007ffff155dba7 in G4RunManager::GenerateEvent (this=0x7453020, i_event=0) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManager.cc:460
    #8  0x00007ffff155d63c in G4RunManager::ProcessOneEvent (this=0x7453020, i_event=0) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManager.cc:398
    #9  0x00007ffff155d4d7 in G4RunManager::DoEventLoop (this=0x7453020, n_event=1, macroFile=0x0, n_select=-1) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManager.cc:367
    #10 0x00007ffff155cd2d in G4RunManager::BeamOn (this=0x7453020, n_event=1, macroFile=0x0, n_select=-1) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManager.cc:273
    #11 0x00007ffff4c9d972 in CG4::propagate (this=0x61df180) at /home/blyth/opticks/cfg4/CG4.cc:345
    #12 0x00007ffff7bd49d1 in OKG4Mgr::propagate_ (this=0x7fffffffcb00) at /home/blyth/opticks/okg4/OKG4Mgr.cc:201
    #13 0x00007ffff7bd487f in OKG4Mgr::propagate (this=0x7fffffffcb00) at /home/blyth/opticks/okg4/OKG4Mgr.cc:138
    #14 0x00000000004039a9 in main (argc=42, argv=0x7fffffffce48) at /home/blyth/opticks/okg4/tests/OKG4Test.cc:9
    (gdb) 
    
    
    (gdb) f 4
    #4  0x00007fffe74ecd28 in STranche::tranche_size (this=0x768a670, i=1) at /home/blyth/opticks/sysrap/STranche.cc:24
    24      assert( i < num_tranche && " trance indices must be from 0 to tr.num_tranche - 1 inclusive  " ); 
    (gdb) p num_tranche
    $1 = 1
    (gdb) p i
    $2 = 1
    (gdb) 



* Added CInputPhotonSource::reset to zero m_gpv_count, but where to call it from ? Needs to come from OpticksRun::resetEvent

::

    112 void OKG4Mgr::propagate()
    113 {
    ...
    133     else if(m_num_event > 0)
    134     {
    135         for(int i=0 ; i < m_num_event ; i++)
    136         {
    137             m_run->createEvent(i);
    138 
    139             propagate_();
    140 
    141             if(ok("save"))
    142             {
    143                 m_run->saveEvent();
    144                 if(m_production)  m_hub->anaEvent();
    145             }
    146             m_run->resetEvent();
    147 
    148         }
    149         m_ok->postpropagate();
    150     }
    151 }   



* also am interested in Opticks multievent, not G4 so avoid with "--nog4propagate"

::

    OpticksProfile=ERROR ts box --pfx scan-ph --cat cvd_1_rtx_0 --generateoverride 1 --compute --production --cvd 1 --rtx 0 --multievent 2 -D --nog4propagate

::

     tmp
     rm -rf scan-ph
     OpticksProfile=ERROR ts box --pfx scan-ph --cat cvd_1_rtx_0 --generateoverride 1 --compute --production --cvd 1 --rtx 0 --multievent 10 --nog4propagate
     ## single photon for machinery check 

     tmp ; rm -rf scan-ph ; OpticksProfile=ERROR ts box --pfx scan-ph --cat cvd_1_rtx_1 --generateoverride -1 --compute --production --cvd 1 --rtx 1 --multievent 10 --nog4propagate

     ip profile.py --tag 0 --cat cvd_1_rtx_1

     ## 1M photons 

     ip profile.py --tag 0



Want to see multievent profile plot, 


Time Between 1M launches
--------------------------------

Time between launches around 0.25s,  FIXED the largest contributor, now down to 0.05

::

    In [8]: tt = pr.t[np.where(pr.l == "_OPropagator::launch" )]

    In [9]: tt
    Out[9]: array([5.1367, 5.3652, 5.6133, 5.8438, 6.0898, 6.3281, 6.5762, 6.8203, 7.0664, 7.3105], dtype=float32)

    In [11]: np.diff(tt)
    Out[11]: array([0.2285, 0.248 , 0.2305, 0.2461, 0.2383, 0.248 , 0.2441, 0.2461, 0.2441], dtype=float32)

    In [12]: np.diff(tt).shape
    Out[12]: (9,)


::

    # /home/blyth/local/opticks/tmp/scan-ph/evt/cvd_1_rtx_0/torch/OpticksProfile.npy              20190716-1507 

    In [1]: pr.times()
    Out[1]: array([5.0332, 5.3809, 5.6523, 5.9238, 6.2109, 6.4941, 6.7715, 7.0586, 7.3496, 7.6348], dtype=float32)

    In [2]: np.diff(pr.times())
    Out[2]: array([0.3477, 0.2715, 0.2715, 0.2871, 0.2832, 0.2773, 0.2871, 0.291 , 0.2852], dtype=float32)




::

    ip profile.py --tag 0 --cat cvd_1_rtx_1

    [2019-07-16 15:25:55,873] p285696 {<module>            :profile.py:307} INFO     - tagdir: /home/blyth/local/opticks/tmp/scan-ph/evt/cvd_1_rtx_1/torch 
    [2019-07-16 15:25:55,874] p285696 {__init__            :profile.py:24} INFO     -  tagdir:/home/blyth/local/opticks/tmp/scan-ph/evt/cvd_1_rtx_1/torch name:pro tag:torch g4:False 
    pro
      /home/blyth/local/opticks/tmp/scan-ph/evt/cvd_1_rtx_1/torch/OpticksProfile.npy              20190716-1509 
      /home/blyth/local/opticks/tmp/scan-ph/evt/cvd_1_rtx_1/torch/OpticksProfileAcc.npy           20190716-1509 
    slice(0, 1, None)
        idx :                                              label :          t          v         dt         dv   
          0 :                             OpticksRun::OpticksRun :     0.0000     0.0000 25787.1367   446.6280   
        idx :                                              label :          t          v         dt         dv   

    In [1]: pr.times()
    Out[1]: array([5.1367, 5.3652, 5.6133, 5.8438, 6.0898, 6.3281, 6.5762, 6.8203, 7.0664, 7.3105], dtype=float32)

    In [2]: np.diff(pr.times())
    Out[2]: array([0.2285, 0.248 , 0.2305, 0.2461, 0.2383, 0.248 , 0.2441, 0.2461, 0.2441], dtype=float32)

        



::

    194 :                               _OPropagator::launch :     6.8203 10284.9277     0.0000     0.0000   
    195 :                                OPropagator::launch :     6.8301 10284.9277     0.0098     0.0000   
    196 :                            OKPropagator::propagate :     6.8301 10284.9277     0.0000     0.0000   
    197 :                       _OEvent::downloadHitsCompute :     6.8301 10284.9277     0.0000     0.0000   
    198 :                        OEvent::downloadHitsCompute :     6.8320 10284.9277     0.0020     0.0000   
    199 :                   OKPropagator::propagate-download :     6.8320 10284.9277     0.0000     0.0000   
    200 :                             _OpticksRun::saveEvent :     6.8320 10284.9277     0.0000     0.0000   
    201 :                                _OpticksEvent::save :     6.8320 10284.9277     0.0000     0.0000   
    202 :                                 OpticksEvent::save :     6.8320 10284.9277     0.0000     0.0000   
    203 :                                _OpticksEvent::save :     6.8477 10284.9277     0.0156     0.0000   *** 
    204 :                                 OpticksEvent::save :     6.8477 10284.9277     0.0000     0.0000   
    205 :                              OpticksRun::saveEvent :     6.8613 10284.9277     0.0137     0.0000   ***
    206 :                            _OpticksRun::resetEvent :     6.8613 10284.9277     0.0000     0.0000   
    207 :                             OpticksRun::resetEvent :     6.8613 10284.9277     0.0000     0.0000   
    208 :                           _OpticksRun::createEvent :     6.8613 10284.9277     0.0000     0.0000   
    209 :                            OpticksRun::createEvent :     6.8613 10284.9277     0.0000     0.0000   
                  /// whats happening in here
    210 :                           _OKPropagator::propagate :     7.0508 10284.9277     0.1895     0.0000   ***
    211 :                                    _OEvent::upload :     7.0508 10284.9277     0.0000     0.0000   
    212 :                                     OEvent::upload :     7.0645 10284.9277     0.0137     0.0000   
    213 :         _OpSeeder::seedPhotonsFromGenstepsViaOptiX :     7.0645 10284.9277     0.0000     0.0000   
    214 :          OpSeeder::seedPhotonsFromGenstepsViaOptiX :     7.0664 10284.9277     0.0020     0.0000   
    215 :                               _OPropagator::launch :     7.0664 10284.9277     0.0000     0.0000   
    216 :                                OPropagator::launch :     7.0742 10284.9277     0.0078     0.0000   





::

    133     else if(m_num_event > 0)
    134     {
    135         for(int i=0 ; i < m_num_event ; i++)
    136         {
    137             m_run->createEvent(i);
    138 
    139             propagate_();
    140 
    141             if(ok("save"))
    142             {
    143                 m_run->saveEvent();
    144                 if(!m_production)  m_hub->anaEvent();
    145             }
    146             m_run->resetEvent();
    147 
    148         }
    149         m_ok->postpropagate();
    150     }


::

    188 void OKG4Mgr::propagate_()
    189 {
    190     bool align = m_ok->isAlign();
    191 
    192     if(m_generator->hasGensteps())   // TORCH
    193     {
    194          NPY<float>* gs = m_generator->getGensteps() ;
    195          m_run->setGensteps(gs);
    196 
    197          if(align)
    198              m_propagator->propagate();
    199 
    200          if(!m_nog4propagate)
    201              m_g4->propagate();
    202     }
    203     else   // no-gensteps : G4GUN or PRIMARYSOURCE
    204     {
    205          NPY<float>* gs = m_g4->propagate() ;
    206 
    207          if(!gs) LOG(fatal) << "CG4::propagate failed to return gensteps" ;
    208          assert(gs);
    209 
    210          m_run->setGensteps(gs);
    211     }
    212 
    213     if(!align)
    214         m_propagator->propagate();
    215 }




Mostly from OpticksEvent::setSourceData

* doing it twice for G4 and OK


::

      368          0.000           6.941          0.000      10284.952          0.000 : _OpticksRun::createEvent_8
      369          0.000           6.941          0.000      10284.952          0.000 : _OpticksEvent::setNopstepData_8
      370          0.000           6.941          0.000      10284.952          0.000 : OpticksEvent::setNopstepData_8
      371          0.000           6.941          0.000      10284.952          0.000 : _OpticksEvent::setNopstepData_8
      372          0.000           6.941          0.000      10284.952          0.000 : OpticksEvent::setNopstepData_8
      373          0.000           6.941          0.000      10284.952          0.000 : OpticksRun::createEvent_8
      374          0.000           6.941          0.000      10284.952          0.000 : _OpticksRun::setGensteps_8
      375          0.000           6.941          0.000      10284.952          0.000 : _OpticksRun::importGensteps_8
      376          0.000           6.941          0.000      10284.952          0.000 : _OpticksRun::importGenstepData_8
      377          0.000           6.941          0.000      10284.952          0.000 : OpticksRun::importGenstepData_8
      378          0.002           6.943          0.002      10284.952          0.000 : _OpticksEvent::setGenstepData_8
      379          0.000           6.943          0.000      10284.952          0.000 : OpticksEvent::setGenstepData_8
      380          0.000           6.943          0.000      10284.952          0.000 : _OpticksEvent::setGenstepData_8
      381          0.000           6.943          0.000      10284.952          0.000 : OpticksEvent::setGenstepData_8
      382          0.000           6.943          0.000      10284.952          0.000 : _OpticksEvent::setSourceData_8
      383          0.094           7.037          0.094      10284.952          0.000 : OpticksEvent::setSourceData_8
      384          0.000           7.037          0.000      10284.952          0.000 : _OpticksEvent::setSourceData_8
      385          0.098           7.135          0.098      10284.952          0.000 : OpticksEvent::setSourceData_8
      386          0.000           7.135          0.000      10284.952          0.000 : _OpticksEvent::setNopstepData_8
      387          0.000           7.135          0.000      10284.952          0.000 : OpticksEvent::setNopstepData_8
      388          0.000           7.135          0.000      10284.952          0.000 : OpticksRun::importGensteps_8
      389          0.000           7.135          0.000      10284.952          0.000 : OpticksRun::setGensteps_8
      390          0.000           7.135          0.000      10284.952          0.000 : _OKPropagator::propagate_8
      391          0.000           7.135          0.000      10284.952          0.000 : _OEvent::upload_8
      392          0.014           7.148          0.014      10284.952          0.000 : OEvent::upload_8
      393          0.000           7.148          0.000      10284.952          0.000 : _OpSeeder::seedPhotonsFromGenstepsViaOptiX_8
      394          0.000           7.148          0.000      10284.952          0.000 : OpSeeder::seedPhotonsFromGenstepsViaOptiX_8
      395          0.000           7.148          0.000      10284.952          0.000 : _OPropagator::launch_8
      396          0.010           7.158          0.010      10284.952          0.000 : OPropagator::launch_8
      397          0.000           7.158          0.000      10284.952          0.000 : OKPropagator::propagate_8
      398          0.000           7.158          0.000      10284.952          0.000 : _OEvent::downloadHitsCompute_8
      399          0.002           7.160          0.002      10284.952          0.000 : OEvent::downloadHitsCompute_8
      400          0.000           7.160          0.000      10284.952          0.000 : OKPropagator::propagate-download_8
      401          0.000           7.160          0.000      10284.952          0.000 : _OpticksRun::saveEvent_8
      402          0.000           7.160          0.000      10284.952          0.000 : _OpticksEvent::save_8
      403          0.004           7.164          0.004      10284.952          0.000 : OpticksEvent::save_8
      404          0.035           7.199          0.035      10284.952          0.000 : _OpticksEvent::save_8
      405          0.002           7.201          0.002      10284.952          0.000 : OpticksEvent::save_8
      406          0.023           7.225          0.023      10284.952          0.000 : OpticksRun::saveEvent_8
      407          0.000           7.225          0.000      10284.952          0.000 : _OpticksRun::resetEvent_8
      408          0.000           7.225          0.000      10284.952          0.000 : OpticksRun::resetEvent_8




After making ViewNPY lazy about evaluating bounds, reduce time between launches from about 0.25s to 0.05s::


    [blyth@localhost opticks]$ ip profile.py --tag 0 --cat cvd_1_rtx_1
    Python 2.7.15 |Anaconda, Inc.| (default, May  1 2018, 23:32:55) 
    Type "copyright", "credits" or "license" for more information.

    IPython 5.7.0 -- An enhanced Interactive Python.
    ?         -> Introduction and overview of IPython's features.
    %quickref -> Quick reference.
    help      -> Python's own help system.
    object?   -> Details about 'object', use 'object??' for extra details.
    defaults det g4live cat cvd_1_rtx_0 src torch tag 1 pfx scan-ph 
    [2019-07-16 16:47:17,526] p439937 {<module>            :profile.py:307} INFO     - tagdir: /home/blyth/local/opticks/tmp/scan-ph/evt/cvd_1_rtx_1/torch 
    [2019-07-16 16:47:17,526] p439937 {__init__            :profile.py:24} INFO     -  tagdir:/home/blyth/local/opticks/tmp/scan-ph/evt/cvd_1_rtx_1/torch name:pro tag:torch g4:False 
    pro
      /home/blyth/local/opticks/tmp/scan-ph/evt/cvd_1_rtx_1/torch/OpticksProfile.npy              20190716-1626 
      /home/blyth/local/opticks/tmp/scan-ph/evt/cvd_1_rtx_1/torch/OpticksProfileAcc.npy           20190716-1626 
    slice(0, 1, None)
        idx :                                              label :          t          v         dt         dv   
          0 :                             OpticksRun::OpticksRun :     0.0000     0.0000 30372.6953   446.6280   
        idx :                                              label :          t          v         dt         dv   
    launch t0 %r  [5.0078 5.0723 5.1152 5.1621 5.2129 5.2695 5.3262 5.3887 5.457  5.5273]
    launch t1 %r  [5.0156 5.0801 5.123  5.1699 5.2207 5.2754 5.332  5.3945 5.4629 5.5352]
    launch                avg     0.0070   t1-t0 array([0.0078, 0.0078, 0.0078, 0.0078, 0.0078, 0.0059, 0.0059, 0.0059, 0.0059, 0.0078], dtype=float32)   
    times between starts  avg     0.0577   np.diff(t0) array([0.0645, 0.043 , 0.0469, 0.0508, 0.0566, 0.0566, 0.0625, 0.0684, 0.0703], dtype=float32) 
    times between stops   avg     0.0577   np.diff(t1) array([0.0645, 0.043 , 0.0469, 0.0508, 0.0547, 0.0566, 0.0625, 0.0684, 0.0723], dtype=float32) 
     between-launch     0.0577  launch-time     0.0070   overhead ratio     8.2099 





OpticksEvent::save is next in line, can be halved by avoiding m_g4evt with --nog4propagate
---------------------------------------------------------------------------------------------

::

      400          0.000           5.377          0.000      10284.960          0.000 : _OpticksRun::createEvent_8
      401          0.000           5.377          0.000      10284.960          0.000 : _OpticksEvent::setNopstepData_8
      402          0.000           5.377          0.000      10284.960          0.000 : OpticksEvent::setNopstepData_8
      403          0.000           5.377          0.000      10284.960          0.000 : _OpticksEvent::setNopstepData_8
      404          0.000           5.377          0.000      10284.960          0.000 : OpticksEvent::setNopstepData_8
      405          0.000           5.377          0.000      10284.960          0.000 : OpticksRun::createEvent_8
      406          0.000           5.377          0.000      10284.960          0.000 : _OpticksRun::setGensteps_8
      407          0.000           5.377          0.000      10284.960          0.000 : _OpticksRun::importGensteps_8
      408          0.000           5.377          0.000      10284.960          0.000 : _OpticksRun::importGenstepData_8
      409          0.000           5.377          0.000      10284.960          0.000 : OpticksRun::importGenstepData_8
      410          0.000           5.377          0.000      10284.960          0.000 : _OpticksEvent::setGenstepData_8
      411          0.000           5.377          0.000      10284.960          0.000 : OpticksEvent::setGenstepData_8
      412          0.000           5.377          0.000      10284.960          0.000 : _OpticksEvent::setGenstepData_8
      413          0.000           5.377          0.000      10284.960          0.000 : OpticksEvent::setGenstepData_8
      414          0.000           5.377          0.000      10284.960          0.000 : _OpticksEvent::setSourceData_8
      415          0.002           5.379          0.002      10284.960          0.000 : _OpticksEvent::setSourceData.MultiViewNPY_8
      416          0.000           5.379          0.000      10284.960          0.000 : OpticksEvent::setSourceData.MultiViewNPY_8
      417          0.000           5.379          0.000      10284.960          0.000 : OpticksEvent::setSourceData_8
      418          0.000           5.379          0.000      10284.960          0.000 : _OpticksEvent::setSourceData_8
      419          0.000           5.379          0.000      10284.960          0.000 : _OpticksEvent::setSourceData.MultiViewNPY_8
      420          0.000           5.379          0.000      10284.960          0.000 : OpticksEvent::setSourceData.MultiViewNPY_8
      421          0.000           5.379          0.000      10284.960          0.000 : OpticksEvent::setSourceData_8
      422          0.000           5.379          0.000      10284.960          0.000 : _OpticksEvent::setNopstepData_8
      423          0.000           5.379          0.000      10284.960          0.000 : OpticksEvent::setNopstepData_8
      424          0.000           5.379          0.000      10284.960          0.000 : OpticksRun::importGensteps_8
      425          0.000           5.379          0.000      10284.960          0.000 : OpticksRun::setGensteps_8
      426          0.000           5.379          0.000      10284.960          0.000 : _OKPropagator::propagate_8
      427          0.000           5.379          0.000      10284.960          0.000 : _OEvent::upload_8
      428          0.008           5.387          0.008      10284.960          0.000 : OEvent::upload_8
      429          0.000           5.387          0.000      10284.960          0.000 : _OpSeeder::seedPhotonsFromGenstepsViaOptiX_8
      430          0.000           5.387          0.000      10284.960          0.000 : OpSeeder::seedPhotonsFromGenstepsViaOptiX_8
      431          0.000           5.387          0.000      10284.960          0.000 : _OPropagator::launch_8
      432          0.006           5.393          0.006      10284.960          0.000 : OPropagator::launch_8
      433          0.002           5.395          0.002      10284.960          0.000 : OKPropagator::propagate_8
      434          0.000           5.395          0.000      10284.960          0.000 : _OEvent::downloadHitsCompute_8
      435          0.002           5.396          0.002      10284.960          0.000 : OEvent::downloadHitsCompute_8
      436          0.000           5.396          0.000      10284.960          0.000 : OKPropagator::propagate-download_8
      437          0.000           5.396          0.000      10284.960          0.000 : _OpticksRun::saveEvent_8
      438          0.000           5.396          0.000      10284.960          0.000 : _OpticksEvent::save_8
      439          0.002           5.398          0.002      10284.960          0.000 : OpticksEvent::save_8
      440          0.023           5.422          0.023      10284.960          0.000 : _OpticksEvent::save_8
      441          0.002           5.424          0.002      10284.960          0.000 : OpticksEvent::save_8
      442          0.025           5.449          0.025      10284.960          0.000 : OpticksRun::saveEvent_8
      443          0.000           5.449          0.000      10284.960          0.000 : _OpticksRun::resetEvent_8
      444          0.000           5.449          0.000      10284.960          0.000 : OpticksRun::resetEvent_8



Skipping creation and saving of the report in production greatly reduces OpticksEvent::save leaving next up OEvent::upload 
-------------------------------------------------------------------------------------------------------------------------------

* taking twice the time of the launch 


::

    In [6]: pr[w0[1]:w1[1]]
    Out[6]: 
    pro
      /home/blyth/local/opticks/tmp/scan-ph/evt/cvd_1_rtx_1/torch/OpticksProfile.npy              20190716-1737 
      /home/blyth/local/opticks/tmp/scan-ph/evt/cvd_1_rtx_1/torch/OpticksProfileAcc.npy           20190716-1737 
    slice(74, 107, None)
        idx :                                              label :          t          v         dt         dv   
         74 :                           _OpticksRun::createEvent :     4.9102 10284.9238     0.0000     0.0000   
         75 :                      _OpticksEvent::setNopstepData :     4.9102 10284.9238     0.0000     0.0000   
         76 :                       OpticksEvent::setNopstepData :     4.9102 10284.9238     0.0000     0.0000   
         77 :                            OpticksRun::createEvent :     4.9102 10284.9238     0.0000     0.0000   
         78 :                           _OpticksRun::setGensteps :     4.9102 10284.9238     0.0000     0.0000   
         79 :                        _OpticksRun::importGensteps :     4.9102 10284.9238     0.0000     0.0000   
         80 :                     _OpticksRun::importGenstepData :     4.9102 10284.9238     0.0000     0.0000   
         81 :                      OpticksRun::importGenstepData :     4.9102 10284.9238     0.0000     0.0000   
         82 :                      _OpticksEvent::setGenstepData :     4.9102 10284.9238     0.0000     0.0000   
         83 :                       OpticksEvent::setGenstepData :     4.9102 10284.9238     0.0000     0.0000   
         84 :                       _OpticksEvent::setSourceData :     4.9102 10284.9238     0.0000     0.0000   
         85 :          _OpticksEvent::setSourceData.MultiViewNPY :     4.9102 10284.9238     0.0000     0.0000   
         86 :           OpticksEvent::setSourceData.MultiViewNPY :     4.9102 10284.9238     0.0000     0.0000   
         87 :                        OpticksEvent::setSourceData :     4.9102 10284.9238     0.0000     0.0000   
         88 :                      _OpticksEvent::setNopstepData :     4.9102 10284.9238     0.0000     0.0000   
         89 :                         OpticksRun::importGensteps :     4.9102 10284.9238     0.0000     0.0000   
         90 :                            OpticksRun::setGensteps :     4.9102 10284.9238     0.0000     0.0000   
         91 :                           _OKPropagator::propagate :     4.9102 10284.9238     0.0000     0.0000   
         92 :                                    _OEvent::upload :     4.9102 10284.9238     0.0000     0.0000   
         93 :                                     OEvent::upload :     4.9258 10284.9238     0.0156     0.0000   
         94 :         _OpSeeder::seedPhotonsFromGenstepsViaOptiX :     4.9258 10284.9238     0.0000     0.0000   
         95 :          OpSeeder::seedPhotonsFromGenstepsViaOptiX :     4.9258 10284.9238     0.0000     0.0000   
         96 :                               _OPropagator::launch :     4.9258 10284.9238     0.0000     0.0000   
         97 :                                OPropagator::launch :     4.9336 10284.9238     0.0078     0.0000   
         98 :                            OKPropagator::propagate :     4.9336 10284.9238     0.0000     0.0000   
         99 :                       _OEvent::downloadHitsCompute :     4.9336 10284.9238     0.0000     0.0000   
        100 :                        OEvent::downloadHitsCompute :     4.9336 10284.9238     0.0000     0.0000   
        101 :                   OKPropagator::propagate-download :     4.9336 10284.9238     0.0000     0.0000   
        102 :                             _OpticksRun::saveEvent :     4.9336 10284.9238     0.0000     0.0000   
        103 :                                _OpticksEvent::save :     4.9336 10284.9238     0.0000     0.0000   
        104 :                                 OpticksEvent::save :     4.9336 10284.9238     0.0000     0.0000   
        105 :                              OpticksRun::saveEvent :     4.9336 10284.9238     0.0000     0.0000   
        106 :                            _OpticksRun::resetEvent :     4.9336 10284.9238     0.0000     0.0000   
        idx :                                              label :          t          v         dt         dv   

    In [7]: 0.0156 + 0.0078
    Out[7]: 0.023399999999999997

    In [8]: (0.0156 + 0.0078)/0.0078
    Out[8]: 2.9999999999999996

    In [9]: (0.0156)/0.0078
    Out[9]: 2.0


quantization problem with the timing, need longer times 
---------------------------------------------------------

* ignoring this quantization, see that upload for 1M photons takes twice the time as the launch 
* this is the end of the road for input photons, need proper on GPU generation to take overhead checking further

::

    ip profile.py --tag 0 --cat cvd_1_rtx_1
    .. 
      /home/blyth/local/opticks/tmp/scan-ph/evt/cvd_1_rtx_1/torch/OpticksProfile.npy              20190716-2127 
      /home/blyth/local/opticks/tmp/scan-ph/evt/cvd_1_rtx_1/torch/OpticksProfileAcc.npy           20190716-2127 
    slice(0, 1, None)
        idx :                                              label :          t          v         dt         dv   
          0 :                             OpticksRun::OpticksRun :     0.0000     0.0000 48431.7070   446.6280   
        idx :                                              label :          t          v         dt         dv   
    launch t0 %r  [4.8789 4.9062 4.9336 4.957  4.9883 5.0117 5.043  5.0664 5.0938 5.1211]
    launch t1 %r  [4.8867 4.9141 4.9414 4.9648 4.9922 5.0195 5.0469 5.0742 5.1016 5.1289]
    launch                avg     0.0070   t1-t0 array([0.0078, 0.0078, 0.0078, 0.0078, 0.0039, 0.0078, 0.0039, 0.0078, 0.0078, 0.0078], dtype=float32)   
    times between starts  avg     0.0269   np.diff(t0) array([0.0273, 0.0273, 0.0234, 0.0312, 0.0234, 0.0312, 0.0234, 0.0273, 0.0273], dtype=float32) 
    times between stops   avg     0.0269   np.diff(t1) array([0.0273, 0.0273, 0.0234, 0.0273, 0.0273, 0.0273, 0.0273, 0.0273, 0.0273], dtype=float32) 
     between-launch     0.0269  launch-time     0.0070   betweenLaunch/launch      3.8272 (perfect=1) 
    pr[w0[1]:w1[1]]
    pro
      /home/blyth/local/opticks/tmp/scan-ph/evt/cvd_1_rtx_1/torch/OpticksProfile.npy              20190716-2127 
      /home/blyth/local/opticks/tmp/scan-ph/evt/cvd_1_rtx_1/torch/OpticksProfileAcc.npy           20190716-2127 
    slice(78, 115, None)
        idx :                                              label :          t          v         dt         dv   
         78 :                           _OpticksRun::createEvent :     4.8867 10284.9316     0.0000     0.0000   
         79 :                      _OpticksEvent::setNopstepData :     4.8867 10284.9316     0.0000     0.0000   
         80 :                       OpticksEvent::setNopstepData :     4.8867 10284.9316     0.0000     0.0000   
         81 :                            OpticksRun::createEvent :     4.8867 10284.9316     0.0000     0.0000   
         82 :                           _OpticksRun::setGensteps :     4.8867 10284.9316     0.0000     0.0000   
         83 :                        _OpticksRun::importGensteps :     4.8906 10284.9316     0.0039     0.0000   
         84 :                     _OpticksRun::importGenstepData :     4.8906 10284.9316     0.0000     0.0000   
         85 :                      OpticksRun::importGenstepData :     4.8906 10284.9316     0.0000     0.0000   
         86 :                      _OpticksEvent::setGenstepData :     4.8906 10284.9316     0.0000     0.0000   
         87 :                       OpticksEvent::setGenstepData :     4.8906 10284.9316     0.0000     0.0000   
         88 :                       _OpticksEvent::setSourceData :     4.8906 10284.9316     0.0000     0.0000   
         89 :          _OpticksEvent::setSourceData.MultiViewNPY :     4.8906 10284.9316     0.0000     0.0000   
         90 :           OpticksEvent::setSourceData.MultiViewNPY :     4.8906 10284.9316     0.0000     0.0000   
         91 :                        OpticksEvent::setSourceData :     4.8906 10284.9316     0.0000     0.0000   
         92 :                      _OpticksEvent::setNopstepData :     4.8906 10284.9316     0.0000     0.0000   
         93 :                         OpticksRun::importGensteps :     4.8906 10284.9316     0.0000     0.0000   
         94 :                            OpticksRun::setGensteps :     4.8906 10284.9316     0.0000     0.0000   
         95 :                           _OKPropagator::propagate :     4.8906 10284.9316     0.0000     0.0000   
         96 :                                    _OEvent::upload :     4.8906 10284.9316     0.0000     0.0000   
         97 :                            _OEvent::uploadGensteps :     4.8906 10284.9316     0.0000     0.0000   
         98 :                             OEvent::uploadGensteps :     4.8906 10284.9316     0.0000     0.0000   
         99 :                              _OEvent::uploadSource :     4.8906 10284.9316     0.0000     0.0000   
        100 :                               OEvent::uploadSource :     4.9062 10284.9316     0.0156     0.0000   ####
        101 :                                     OEvent::upload :     4.9062 10284.9316     0.0000     0.0000   
        102 :         _OpSeeder::seedPhotonsFromGenstepsViaOptiX :     4.9062 10284.9316     0.0000     0.0000   
        103 :          OpSeeder::seedPhotonsFromGenstepsViaOptiX :     4.9062 10284.9316     0.0000     0.0000   
        104 :                               _OPropagator::launch :     4.9062 10284.9316     0.0000     0.0000   
        105 :                                OPropagator::launch :     4.9141 10284.9316     0.0078     0.0000    ####  
        106 :                            OKPropagator::propagate :     4.9141 10284.9316     0.0000     0.0000   
        107 :                       _OEvent::downloadHitsCompute :     4.9141 10284.9316     0.0000     0.0000   
        108 :                        OEvent::downloadHitsCompute :     4.9141 10284.9316     0.0000     0.0000   
        109 :                   OKPropagator::propagate-download :     4.9141 10284.9316     0.0000     0.0000   
        110 :                             _OpticksRun::saveEvent :     4.9141 10284.9316     0.0000     0.0000   
        111 :                                _OpticksEvent::save :     4.9141 10284.9316     0.0000     0.0000   
        112 :                                 OpticksEvent::save :     4.9141 10284.9316     0.0000     0.0000   
        113 :                              OpticksRun::saveEvent :     4.9141 10284.9316     0.0000     0.0000   
        114 :                            _OpticksRun::resetEvent :     4.9141 10284.9316     0.0000     0.0000   
        idx :                                              label :          t          v         dt         dv   








::

    In [1]: tt
    Out[1]: 
    array([[0.    , 0.    , 0.    , 0.    , 0.    , 0.0039, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.0156,
            0.    , 0.    , 0.    , 0.    , 0.0078, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ],
           [0.    , 0.    , 0.    , 0.    , 0.    , 0.0039, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.0156,
            0.    , 0.    , 0.    , 0.    , 0.0078, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ],
           [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.0156,
            0.    , 0.    , 0.    , 0.    , 0.0078, 0.    , 0.    , 0.0039, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ],
           [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.0156,
            0.    , 0.    , 0.0039, 0.    , 0.0039, 0.    , 0.0039, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ],
           [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.0039, 0.    , 0.    , 0.0117,
            0.    , 0.    , 0.    , 0.    , 0.0078, 0.    , 0.    , 0.0039, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ],
           [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.0156,
            0.    , 0.    , 0.    , 0.0039, 0.0039, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.0039, 0.    , 0.    ],
           [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.0156,
            0.    , 0.    , 0.    , 0.    , 0.0078, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ],
           [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.0039, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.0156,
            0.    , 0.    , 0.    , 0.    , 0.0078, 0.    , 0.    , 0.0039, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ],
           [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.0156,
            0.    , 0.    , 0.    , 0.    , 0.0078, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ]])

    In [2]: 0.0039*np.arange(5)
    Out[2]: array([0.    , 0.0039, 0.0078, 0.0117, 0.0156])





Try 10M : time of upload and launch about the same
------------------------------------------------------




::

     tmp ; rm -rf scan-ph ; OpticksProfile=ERROR ts box --pfx scan-ph --cat cvd_1_rtx_1 --generateoverride -10 --rngmax 10 --compute --production --cvd 1 --rtx 1 --multievent 10 --nog4propagate

     tmp ; rm -rf scan-ph ; ts box --pfx scan-ph --cat cvd_1_rtx_1 --generateoverride 10000000 --compute --production --nog4propagate --rngmax 10 --cvd 1 --rtx 1 --multievent 10 

     ip profile.py --tag 0 --cat cvd_1_rtx_1


::

    [2019-07-16 22:06:47,584] p34569 {<module>            :profile.py:314} INFO     - tagdir: /home/blyth/local/opticks/tmp/scan-ph/evt/cvd_1_rtx_1/torch 
    [2019-07-16 22:06:47,584] p34569 {__init__            :profile.py:24} INFO     -  tagdir:/home/blyth/local/opticks/tmp/scan-ph/evt/cvd_1_rtx_1/torch name:pro tag:torch g4:False 
    pro
      /home/blyth/local/opticks/tmp/scan-ph/evt/cvd_1_rtx_1/torch/OpticksProfile.npy              20190716-2152 
      /home/blyth/local/opticks/tmp/scan-ph/evt/cvd_1_rtx_1/torch/OpticksProfileAcc.npy           20190716-2152 
    slice(0, 1, None)
        idx :                                              label :          t          v         dt         dv   
          0 :                             OpticksRun::OpticksRun :     0.0000     0.0000 49946.0977   446.6280   
        idx :                                              label :          t          v         dt         dv   
    launch t0 %r  [18.1367 18.2969 18.4531 18.6016 18.75   18.957  19.1094 19.2617 19.4102 19.5625]
    launch t1 %r  [18.2148 18.3711 18.5234 18.6719 18.8203 19.0312 19.1836 19.332  19.4844 19.6328]
    launch                avg     0.0727   t1-t0 array([0.0781, 0.0742, 0.0703, 0.0703, 0.0703, 0.0742, 0.0742, 0.0703, 0.0742, 0.0703], dtype=float32)   
    times between starts  avg     0.1584   np.diff(t0) array([0.1602, 0.1562, 0.1484, 0.1484, 0.207 , 0.1523, 0.1523, 0.1484, 0.1523], dtype=float32) 
    times between stops   avg     0.1576   np.diff(t1) array([0.1562, 0.1523, 0.1484, 0.1484, 0.2109, 0.1523, 0.1484, 0.1523, 0.1484], dtype=float32) 
     between-launch     0.1584  launch-time     0.0727   betweenLaunch/launch      2.1804 (perfect=1) 











