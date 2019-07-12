geant4-beamOn-profiling
========================

Context
----------

* :doc:`plugging-cfg4-leaks`


ISSUE : old estimate broken by more detailed profiling, need a better Geant4 "launch" time estimate : DONE
------------------------------------------------------------------------------------------------------------

* plumped for : CRunAction::BeginOfRunAction -> CRunAction::EndOfRunAction 
  with CRandomEngine::setupTranche accumulated time subtracted 
* see opticks.ana.profile  ana/profile.py 


Rationale
------------

Excluding CRandomEngine::setupTranche makes every G4 event take the same time(a good sign) 
a bit over 1 second (1.023/1.029/...) for 10,000 photons 

Measure:

* 10.6s for 100k photons, 0.1M 

Extrapolates:

* 1s for 0.010M
* 100s for 1M       1.67 min
* 1,000s for 10M      16.66 min        (measured 1060.47s)
* 10,000s for 100M    166.66 min   2.77 hrs
* 100,000s for 1000M  1666.66 min  27.77 hrs

* CRandomEngine::setupTranche is inside the brackets CRunAction::BeginOfRunAction CRunAction::EndOfRunAction
  so added persisting of accumulators and used it subtract off accumulated setupTranche time 

  * its quite a small effect though in low stats testing anyhow

Around 1% more for GeneratePrimaryVertex, which is translating input photons::

    In [6]: 0.012/1.031
    Out[6]: 0.011639185257032008


Hmm : tranche size is 100,000 photons each G4 event covers 10,000 photons : so one tranche setup per 10 G4 events
--------------------------------------------------------------------------------------------------------------------

::

     53 CRandomEngine::CRandomEngine(CG4* g4)
     54     :
     55     m_g4(g4),
     ... 
     71     m_locseq(m_alignlevel > 1 ? new BLocSeq<unsigned long long>(m_skipdupe) : NULL ),
     72     m_tranche_size(100000),
     73     m_tranche_id(-1),
     74     m_tranche_ibase(-1),


ip profile.py from 100k testing split into 10 G4Event of 10k photons each 
-------------------------------------------------------------------------------------


100k testing::

      OpticksProfile=ERROR ts box --generateoverride 100000   

::

    [blyth@localhost ana]$ ip profile.py
    Python 2.7.15 |Anaconda, Inc.| (default, May  1 2018, 23:32:55) 
    Type "copyright", "credits" or "license" for more information.

    IPython 5.7.0 -- An enhanced Interactive Python.
    ?         -> Introduction and overview of IPython's features.
    %quickref -> Quick reference.
    help      -> Python's own help system.
    object?   -> Details about 'object', use 'object??' for extra details.
    args: /home/blyth/opticks/ana/profile.py
    [2019-07-11 19:09:43,410] p422440 {<module>            :profile.py:159} INFO     - pfx tboolean-box tag 1 src torch det tboolean-box c2max [1.5, 2.0, 2.5] ipython True 
    path:/tmp/blyth/opticks/tboolean-box/evt/tboolean-box/torch/1/OpticksProfile.npy stamp:20190711-1732 
    lpath:/tmp/blyth/opticks/tboolean-box/evt/tboolean-box/torch/1/OpticksProfileLabels.npy stamp:20190711-1732 
    acpath:/tmp/blyth/opticks/tboolean-box/evt/tboolean-box/torch/1/OpticksProfileAcc.npy stamp:20190711-1732 
    lacpath:/tmp/blyth/opticks/tboolean-box/evt/tboolean-box/torch/1/OpticksProfileAccLabels.npy stamp:20190711-1732 
    [2019-07-11 19:09:43,416] p422440 {delta               :profile.py:128} INFO     -  l0:          _OPropagator::launch l1:           OPropagator::launch p0: 47 p1: 48  (v0:   10321.5 v1:   10550.8 dv:     229.4 )  ( t0:    3.1836 t1:    3.1914 dt:    0.0078 )  
    [2019-07-11 19:09:43,416] p422440 {delta               :profile.py:128} INFO     -  l0:  CRunAction::BeginOfRunAction l1:    CRunAction::EndOfRunAction p0: 70 p1:113  (v0:   10599.8 v1:   11138.4 dv:     538.6 )  ( t0:    4.6250 t1:   15.4805 dt:   10.8555 )  
    ab.pro
          okp     0.0078     g4r 10.8555    stt 0.1816     g4p 10.6739           g4p/okp 1366.2570     
    slice(0, 10, None)
     idx :                                              label :          t          v         dt         dv   
       0 :                             OpticksRun::OpticksRun :     0.0000     0.0000 34307.3359   446.6040   
       1 :                                   Opticks::Opticks :     0.0000     0.0000     0.0000     0.0000   
       2 :                                  _OKG4Mgr::OKG4Mgr :     0.0000     0.0000     0.0000     0.0000   
       3 :                                  _OpticksHub::init :     0.0000     0.0000     0.0000     0.0000   
       4 :                     _OpticksGeometry::loadGeometry :     0.0117   103.7480     0.0117   103.7480   
       5 :                      OpticksGeometry::loadGeometry :     0.4570   227.4720     0.4453   123.7240   
       6 :                               _GMergedMesh::Create :     0.4883   233.2200     0.0312     5.7480   
       7 :                         GMergedMesh::Create::Count :     0.4922   233.2200     0.0039     0.0000   
       8 :                     _GMergedMesh::Create::Allocate :     0.4922   233.2200     0.0000     0.0000   
       9 :                      GMergedMesh::Create::Allocate :     0.4922   233.5200     0.0000     0.3000   
     idx :                                              label :          t          v         dt         dv   

    In [1]: 


Adjusting slices::

    In [2]: op[47:48+1]
    Out[2]: 
    ab.pro
          okp     0.0078     g4r 10.8555    stt 0.1816     g4p 10.6739           g4p/okp 1366.2570     
    slice(47, 49, None)
     idx :                                              label :          t          v         dt         dv   
      47 :                               _OPropagator::launch :     3.1836 10321.4551     0.0000     0.0000   
      48 :                                OPropagator::launch :     3.1914 10550.8320     0.0078   229.3760   
     idx :                                              label :          t          v         dt         dv   


    In [3]: op[70:113+1]
    Out[3]: 
    ab.pro
          okp     0.0078     g4r 10.8555    stt 0.1816     g4p 10.6739           g4p/okp 1366.2570     
    slice(70, 114, None)
     idx :                                              label :          t          v         dt         dv   
      70 :                       CRunAction::BeginOfRunAction :     4.6250 10599.8242     1.3359     0.0000   
      71 :         _CInputPhotonSource::GeneratePrimaryVertex :     4.6250 10599.8242     0.0000     0.0000   
      72 :          CInputPhotonSource::GeneratePrimaryVertex :     4.6367 10599.8242     0.0117     0.0000   
      73 :                   CEventAction::BeginOfEventAction :     4.6367 10599.8242     0.0000     0.0000   
      74 :                       _CRandomEngine::setupTranche :     4.6445 10599.8242     0.0078     0.0000   
      75 :                        CRandomEngine::setupTranche :     4.8242 11124.1113     0.1797   524.2881   
      76 :                     CEventAction::EndOfEventAction :     5.8828 11124.1113     1.0586     0.0000   
      77 :         _CInputPhotonSource::GeneratePrimaryVertex :     5.8867 11124.1113     0.0039     0.0000   
      78 :          CInputPhotonSource::GeneratePrimaryVertex :     5.8984 11126.1602     0.0117     2.0479   
      79 :                   CEventAction::BeginOfEventAction :     5.8984 11126.1602     0.0000     0.0000   
      80 :                     CEventAction::EndOfEventAction :     6.9453 11126.1602     1.0469     0.0000   
      81 :         _CInputPhotonSource::GeneratePrimaryVertex :     6.9453 11126.1602     0.0000     0.0000   
      82 :          CInputPhotonSource::GeneratePrimaryVertex :     6.9570 11126.1602     0.0117     0.0000   
      83 :                   CEventAction::BeginOfEventAction :     6.9570 11126.1602     0.0000     0.0000   
      84 :                     CEventAction::EndOfEventAction :     8.0039 11126.1602     1.0469     0.0000   
      85 :         _CInputPhotonSource::GeneratePrimaryVertex :     8.0078 11126.1602     0.0039     0.0000   
      86 :          CInputPhotonSource::GeneratePrimaryVertex :     8.0195 11130.2559     0.0117     4.0967   
      87 :                   CEventAction::BeginOfEventAction :     8.0195 11130.2559     0.0000     0.0000   
      88 :                     CEventAction::EndOfEventAction :     9.0703 11130.2559     1.0508     0.0000   
      89 :         _CInputPhotonSource::GeneratePrimaryVertex :     9.0703 11130.2559     0.0000     0.0000   
      90 :          CInputPhotonSource::GeneratePrimaryVertex :     9.0820 11130.2559     0.0117     0.0000   
      91 :                   CEventAction::BeginOfEventAction :     9.0820 11130.2559     0.0000     0.0000   
      92 :                     CEventAction::EndOfEventAction :    10.1289 11130.2559     1.0469     0.0000   
      93 :         _CInputPhotonSource::GeneratePrimaryVertex :    10.1289 11130.2559     0.0000     0.0000   
      94 :          CInputPhotonSource::GeneratePrimaryVertex :    10.1406 11130.2559     0.0117     0.0000   
      95 :                   CEventAction::BeginOfEventAction :    10.1406 11130.2559     0.0000     0.0000   
      96 :                     CEventAction::EndOfEventAction :    11.2109 11130.2559     1.0703     0.0000   
      97 :         _CInputPhotonSource::GeneratePrimaryVertex :    11.2109 11130.2559     0.0000     0.0000   
      98 :          CInputPhotonSource::GeneratePrimaryVertex :    11.2227 11138.4473     0.0117     8.1914   
      99 :                   CEventAction::BeginOfEventAction :    11.2227 11138.4473     0.0000     0.0000   
     100 :                     CEventAction::EndOfEventAction :    12.2852 11138.4473     1.0625     0.0000   
     101 :         _CInputPhotonSource::GeneratePrimaryVertex :    12.2852 11138.4473     0.0000     0.0000   
     102 :          CInputPhotonSource::GeneratePrimaryVertex :    12.2969 11138.4473     0.0117     0.0000   
     103 :                   CEventAction::BeginOfEventAction :    12.2969 11138.4473     0.0000     0.0000   
     104 :                     CEventAction::EndOfEventAction :    13.3633 11138.4473     1.0664     0.0000   
     105 :         _CInputPhotonSource::GeneratePrimaryVertex :    13.3633 11138.4473     0.0000     0.0000   
     106 :          CInputPhotonSource::GeneratePrimaryVertex :    13.3750 11138.4473     0.0117     0.0000   
     107 :                   CEventAction::BeginOfEventAction :    13.3750 11138.4473     0.0000     0.0000   
     108 :                     CEventAction::EndOfEventAction :    14.4141 11138.4473     1.0391     0.0000   
     109 :         _CInputPhotonSource::GeneratePrimaryVertex :    14.4180 11138.4473     0.0039     0.0000   
     110 :          CInputPhotonSource::GeneratePrimaryVertex :    14.4258 11138.4473     0.0078     0.0000   
     111 :                   CEventAction::BeginOfEventAction :    14.4258 11138.4473     0.0000     0.0000   
     112 :                     CEventAction::EndOfEventAction :    15.4766 11138.4473     1.0508     0.0000   
     113 :                         CRunAction::EndOfRunAction :    15.4805 11138.4473     0.0039     0.0000   
     idx :                                              label :          t          v         dt         dv   



Old simple way of "launch" timing, includes some initialization
-----------------------------------------------------------------------

::

    348     LOG(info) << " calling BeamOn numG4Evt " << numG4Evt ;
    349     OK_PROFILE("_CG4::propagate");
    350 
    351     m_runManager->BeamOn(numG4Evt);
    352 
    353     OK_PROFILE("CG4::propagate");
    354     LOG(info) << " calling BeamOn numG4Evt " << numG4Evt << " DONE " ;


100k testing::

      OpticksProfile=ERROR ts box --generateoverride 100000   


ip profile.py::


      66 :                       _OEvent::downloadHitsInterop :      3.920  10580.956      0.000      0.000   
      67 :                        OEvent::downloadHitsInterop :      3.924  10580.956      0.004      0.000   
      68 :                   OKPropagator::propagate-download :      3.924  10580.956      0.000      0.000   
      69 :                                    _CG4::propagate :      3.953  10602.832      0.029     21.876   
      /////////
      ///////// whats G4 doing in here for 1.3 s ???? before starting the run ?  
      /////////
      70 :                       CRunAction::BeginOfRunAction :      5.293  10602.832      1.340      0.000   
      71 :                   CEventAction::BeginOfEventAction :      5.311  10603.856      0.018      1.024   
      72 :                        CRandomEngine::setupTranche :      5.318  10605.509      0.008      1.652   
      73 :                     CEventAction::EndOfEventAction :      6.553  11129.797      1.234    524.288   
      74 :                   CEventAction::BeginOfEventAction :      6.566  11131.845      0.014      2.048   
      75 :                     CEventAction::EndOfEventAction :      7.594  11131.845      1.027      0.000   
      76 :                   CEventAction::BeginOfEventAction :      7.607  11131.845      0.014      0.000   
      77 :                     CEventAction::EndOfEventAction :      8.645  11131.845      1.037      0.000   
      78 :                   CEventAction::BeginOfEventAction :      8.660  11135.940      0.016      4.096   
      79 :                     CEventAction::EndOfEventAction :      9.715  11135.940      1.055      0.000   
      80 :                   CEventAction::BeginOfEventAction :      9.727  11135.940      0.012      0.000   
      81 :                     CEventAction::EndOfEventAction :     10.762  11135.940      1.035      0.000   
      82 :                   CEventAction::BeginOfEventAction :     10.775  11135.940      0.014      0.000   
      83 :                     CEventAction::EndOfEventAction :     11.818  11135.940      1.043      0.000   
      84 :                   CEventAction::BeginOfEventAction :     11.836  11144.133      0.018      8.192   
      85 :                     CEventAction::EndOfEventAction :     12.863  11144.133      1.027      0.000   
      86 :                   CEventAction::BeginOfEventAction :     12.875  11144.133      0.012      0.000   
      87 :                     CEventAction::EndOfEventAction :     13.932  11144.133      1.057      0.000   
      88 :                   CEventAction::BeginOfEventAction :     13.943  11144.133      0.012      0.000   
      89 :                     CEventAction::EndOfEventAction :     15.002  11144.133      1.059      0.000   
      90 :                   CEventAction::BeginOfEventAction :     15.018  11144.133      0.016      0.000   
      91 :                     CEventAction::EndOfEventAction :     16.049  11144.133      1.031      0.000   
      92 :                         CRunAction::EndOfRunAction :     16.051  11144.133      0.002      0.000   
      93 :                                     CG4::propagate :     16.051  11144.133      0.000      0.000   



This does GPU launches to generate randoms for aligned running

::

    205 void CRandomEngine::setupTranche(int tranche_id)
    206 {
    207     m_ok->accumulateStart(m_setupTranche_acc) ;
    208     OK_PROFILE("_CRandomEngine::setupTranche");
    209 
    210     m_tranche_id = tranche_id ;
    211     m_tranche_ibase = m_tranche_id*m_tranche_size ;
    212 
    213     LOG(LEVEL)
    214         << " DYNAMIC_CURAND "
    215         << " m_tranche_id " << m_tranche_id
    216         << " m_tranche_size " << m_tranche_size
    217         << " m_tranche_ibase " << m_tranche_ibase
    218         ;
    219 
    220     m_tcurand->setIBase(m_tranche_ibase);   // <-- does GPU launch to init curand and generate the randoms
    221     checkTranche();
    222 
    223     OK_PROFILE("CRandomEngine::setupTranche");
    224     m_ok->accumulateStop(m_setupTranche_acc) ;
    225 }


::

      OpticksProfile=ERROR ts box --generateoverride 100000   


      069          0.043           4.008          0.043      10605.960         23.439 : _CG4::propagate_0
       70          1.342           5.350          1.342      10605.960          0.000 : CRunAction::BeginOfRunAction_0
       71          0.002           5.352          0.002      10605.960          0.000 : _CInputPhotonSource::GeneratePrimaryVertex_0
       72          0.014           5.365          0.014      10605.960          0.000 : CInputPhotonSource::GeneratePrimaryVertex_0
       73          0.000           5.365          0.000      10605.960          0.000 : CEventAction::BeginOfEventAction_0

       74          0.012           5.377          0.012      10605.960          0.000 : _CRandomEngine::setupTranche_0
       75          0.211           5.588          0.211      11130.248        524.288 : CRandomEngine::setupTranche_0

       76          1.023           6.611          1.023      11130.248          0.000 : CEventAction::EndOfEventAction_0
       77          0.002           6.613          0.002      11130.248          0.000 : _CInputPhotonSource::GeneratePrimaryVertex_0
       78          0.012           6.625          0.012      11132.297          2.049 : CInputPhotonSource::GeneratePrimaryVertex_0
       79          0.000           6.625          0.000      11132.297          0.000 : CEventAction::BeginOfEventAction_0
       80          1.023           7.648          1.023      11132.297          0.000 : CEventAction::EndOfEventAction_0
       81          0.002           7.650          0.002      11132.297          0.000 : _CInputPhotonSource::GeneratePrimaryVertex_0
       82          0.012           7.662          0.012      11132.297          0.000 : CInputPhotonSource::GeneratePrimaryVertex_0
       83          0.000           7.662          0.000      11132.297          0.000 : CEventAction::BeginOfEventAction_0
       84          1.023           8.686          1.023      11132.297          0.000 : CEventAction::EndOfEventAction_0
       85          0.002           8.688          0.002      11132.297          0.000 : _CInputPhotonSource::GeneratePrimaryVertex_0
       86          0.012           8.699          0.012      11136.393          4.096 : CInputPhotonSource::GeneratePrimaryVertex_0
       87          0.000           8.699          0.000      11136.393          0.000 : CEventAction::BeginOfEventAction_0
       88          1.029           9.729          1.029      11136.393          0.000 : CEventAction::EndOfEventAction_0
       89          0.002           9.730          0.002      11136.393          0.000 : _CInputPhotonSource::GeneratePrimaryVertex_0
       90          0.012           9.742          0.012      11136.393          0.000 : CInputPhotonSource::GeneratePrimaryVertex_0
       91          0.000           9.742          0.000      11136.393          0.000 : CEventAction::BeginOfEventAction_0
       92          1.021          10.764          1.021      11136.393          0.000 : CEventAction::EndOfEventAction_0
       93          0.000          10.764          0.000      11136.393          0.000 : _CInputPhotonSource::GeneratePrimaryVertex_0
       94          0.012          10.775          0.012      11136.393          0.000 : CInputPhotonSource::GeneratePrimaryVertex_0
       95          0.000          10.775          0.000      11136.393          0.000 : CEventAction::BeginOfEventAction_0
       96          1.031          11.807          1.031      11136.393          0.000 : CEventAction::EndOfEventAction_0
       97          0.000          11.807          0.000      11136.393          0.000 : _CInputPhotonSource::GeneratePrimaryVertex_0
       98          0.016          11.822          0.016      11144.584          8.191 : CInputPhotonSource::GeneratePrimaryVertex_0
       99          0.000          11.822          0.000      11144.584          0.000 : CEventAction::BeginOfEventAction_0
      100          1.035          12.857          1.035      11144.584          0.000 : CEventAction::EndOfEventAction_0
      101          0.002          12.859          0.002      11144.584          0.000 : _CInputPhotonSource::GeneratePrimaryVertex_0
      102          0.010          12.869          0.010      11144.584          0.000 : CInputPhotonSource::GeneratePrimaryVertex_0
      103          0.000          12.869          0.000      11144.584          0.000 : CEventAction::BeginOfEventAction_0
      104          1.027          13.896          1.027      11144.584          0.000 : CEventAction::EndOfEventAction_0
      105          0.002          13.898          0.002      11144.584          0.000 : _CInputPhotonSource::GeneratePrimaryVertex_0
      106          0.012          13.910          0.012      11144.584          0.000 : CInputPhotonSource::GeneratePrimaryVertex_0
      107          0.000          13.910          0.000      11144.584          0.000 : CEventAction::BeginOfEventAction_0
      108          1.023          14.934          1.023      11144.584          0.000 : CEventAction::EndOfEventAction_0
      109          0.002          14.936          0.002      11144.584          0.000 : _CInputPhotonSource::GeneratePrimaryVertex_0
      110          0.012          14.947          0.012      11144.584          0.000 : CInputPhotonSource::GeneratePrimaryVertex_0
      111          0.000          14.947          0.000      11144.584          0.000 : CEventAction::BeginOfEventAction_0
      112          1.027          15.975          1.027      11144.584          0.000 : CEventAction::EndOfEventAction_0
      113          0.000          15.975          0.000      11144.584          0.000 : CRunAction::EndOfRunAction_0
      114          0.000          15.975          0.000      11144.584          0.000 : CG4::propagate_0




::

      066          0.002           3.807          0.002      10580.956          0.000 : _OEvent::downloadHitsInterop_0
       67          0.000           3.807          0.000      10580.956          0.000 : OEvent::downloadHitsInterop_0
       68          0.000           3.807          0.000      10580.956          0.000 : OKPropagator::propagate-download_0
       69          0.027           3.834          0.027      10604.393         23.437 : _CG4::propagate_0
       70          1.344           5.178          1.344      10604.393          0.000 : CRunAction::BeginOfRunAction_0
       71          0.000           5.178          0.000      10604.393          0.000 : _CInputPhotonSource::GeneratePrimaryVertex_0         ## invoked by G4RunManager::GenerateEvent
       72          0.012           5.189          0.012      10605.416          1.023 : CInputPhotonSource::GeneratePrimaryVertex_0
       73          0.002           5.191          0.002      10605.416          0.000 : CEventAction::BeginOfEventAction_0
       74          0.008           5.199          0.008      10605.572          0.156 : CRandomEngine::setupTranche_0
       75          1.193           6.393          1.193      11129.860        524.288 : CEventAction::EndOfEventAction_0
       ////////// smth different about the 1st event ?    
       76          0.000           6.393          0.000      11129.860          0.000 : _CInputPhotonSource::GeneratePrimaryVertex_0
       77          0.012           6.404          0.012      11131.908          2.048 : CInputPhotonSource::GeneratePrimaryVertex_0
       78          0.002           6.406          0.002      11131.908          0.000 : CEventAction::BeginOfEventAction_0
       79          1.014           7.420          1.014      11131.908          0.000 : CEventAction::EndOfEventAction_0
       80          0.000           7.420          0.000      11131.908          0.000 : _CInputPhotonSource::GeneratePrimaryVertex_0
       81          0.012           7.432          0.012      11131.908          0.000 : CInputPhotonSource::GeneratePrimaryVertex_0
       82          0.000           7.432          0.000      11131.908          0.000 : CEventAction::BeginOfEventAction_0
       83          1.014           8.445          1.014      11131.908          0.000 : CEventAction::EndOfEventAction_0
       84          0.002           8.447          0.002      11131.908          0.000 : _CInputPhotonSource::GeneratePrimaryVertex_0
       85          0.012           8.459          0.012      11136.004          4.096 : CInputPhotonSource::GeneratePrimaryVertex_0
       86          0.000           8.459          0.000      11136.004          0.000 : CEventAction::BeginOfEventAction_0
       87          1.027           9.486          1.027      11136.004          0.000 : CEventAction::EndOfEventAction_0
       88          0.002           9.488          0.002      11136.004          0.000 : _CInputPhotonSource::GeneratePrimaryVertex_0
       89          0.010           9.498          0.010      11136.004          0.000 : CInputPhotonSource::GeneratePrimaryVertex_0
       90          0.000           9.498          0.000      11136.004          0.000 : CEventAction::BeginOfEventAction_0
       91          1.041          10.539          1.041      11136.004          0.000 : CEventAction::EndOfEventAction_0
       92          0.000          10.539          0.000      11136.004          0.000 : _CInputPhotonSource::GeneratePrimaryVertex_0
       93          0.012          10.551          0.012      11136.004          0.000 : CInputPhotonSource::GeneratePrimaryVertex_0
       94          0.000          10.551          0.000      11136.004          0.000 : CEventAction::BeginOfEventAction_0
       95          1.018          11.568          1.018      11136.004          0.000 : CEventAction::EndOfEventAction_0
       96          0.002          11.570          0.002      11136.004          0.000 : _CInputPhotonSource::GeneratePrimaryVertex_0
       97          0.012          11.582          0.012      11144.196          8.192 : CInputPhotonSource::GeneratePrimaryVertex_0
       98          0.000          11.582          0.000      11144.196          0.000 : CEventAction::BeginOfEventAction_0
       99          1.025          12.607          1.025      11144.196          0.000 : CEventAction::EndOfEventAction_0
      100          0.000          12.607          0.000      11144.196          0.000 : _CInputPhotonSource::GeneratePrimaryVertex_0
      101          0.012          12.619          0.012      11144.196          0.000 : CInputPhotonSource::GeneratePrimaryVertex_0
      102          0.000          12.619          0.000      11144.196          0.000 : CEventAction::BeginOfEventAction_0
      103          1.016          13.635          1.016      11144.196          0.000 : CEventAction::EndOfEventAction_0
      104          0.000          13.635          0.000      11144.196          0.000 : _CInputPhotonSource::GeneratePrimaryVertex_0
      105          0.010          13.645          0.010      11144.196          0.000 : CInputPhotonSource::GeneratePrimaryVertex_0
      106          0.000          13.645          0.000      11144.196          0.000 : CEventAction::BeginOfEventAction_0
      107          1.018          14.662          1.018      11144.196          0.000 : CEventAction::EndOfEventAction_0
      108          0.002          14.664          0.002      11144.196          0.000 : _CInputPhotonSource::GeneratePrimaryVertex_0
      109          0.010          14.674          0.010      11144.196          0.000 : CInputPhotonSource::GeneratePrimaryVertex_0
      110          0.000          14.674          0.000      11144.196          0.000 : CEventAction::BeginOfEventAction_0
      111          1.018          15.691          1.018      11144.196          0.000 : CEventAction::EndOfEventAction_0
      112          0.000          15.691          0.000      11144.196          0.000 : CRunAction::EndOfRunAction_0
      113          0.000          15.691          0.000      11144.196          0.000 : CG4::propagate_0
      114          0.002          15.693          0.002      11144.196          0.000 : _OpticksEvent::indexPhotonsCPU_0
      115          0.084          15.777          0.084      11144.196          0.000 : OpticksEvent::indexPhotonsCPU_0
      116          0.000          15.777          0.000      11144.196          0.000 : _OpticksEvent::collectPhotonHitsCPU_0
      117          0.008          15.785          0.008      11144.196          0.000 : OpticksEvent::collectPhotonHitsCPU_0
      118          0.006          15.791          0.006      11144.196          0.000 : _OpticksRun::saveEvent_0
      119          0.000          15.791          0.000      11144.196          0.000 : _OpticksEvent::save_0





g4-cls G4RunManager::


    262 void G4RunManager::BeamOn(G4int n_event,const char* macroFile,G4int n_select)
    263 {
    264   if(n_event<=0) { fakeRun = true; }
    265   else { fakeRun = false; }
    266   G4bool cond = ConfirmBeamOnCondition();
    267   if(cond)
    268   {
    269     numberOfEventToBeProcessed = n_event;
    270     numberOfEventProcessed = 0;
    271     ConstructScoringWorlds();
    272     RunInitialization();
    273     DoEventLoop(n_event,macroFile,n_select);
    274     RunTermination();
    275   }
    276   fakeRun = false;
    277 }


    360 void G4RunManager::DoEventLoop(G4int n_event,const char* macroFile,G4int n_select)
    361 {
    362   InitializeEventLoop(n_event,macroFile,n_select);
    363 
    364 // Event loop
    365   for(G4int i_event=0; i_event<n_event; i_event++ )
    366   {
    367     ProcessOneEvent(i_event);
    368     TerminateOneEvent();
    369     if(runAborted) break;
    370   }
    371 
    372   // For G4MTRunManager, TerminateEventLoop() is invoked after all threads are finished.
    373   if(runManagerType==sequentialRM) TerminateEventLoop();
    374 }

    396 void G4RunManager::ProcessOneEvent(G4int i_event)
    397 {
    398   currentEvent = GenerateEvent(i_event);
    399   eventManager->ProcessOneEvent(currentEvent);
    400   AnalyzeEvent(currentEvent);
    401   UpdateScoring();
    402   if(i_event<n_select_msg) G4UImanager::GetUIpointer()->ApplyCommand(msgText);
    403 }
    404 
    405 void G4RunManager::TerminateOneEvent()
    406 {
    407   StackPreviousEvent(currentEvent);
    408   currentEvent = 0;
    409   numberOfEventProcessed++;
    410 }


H



