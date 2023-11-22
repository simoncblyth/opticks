sevt_SAB_slow_with_1M_and_order_2000_histories
================================================


Overview
----------

Obvious speedup in would be to preselect removing 
low frequency histories. But that dont look
possible in the python/numpy way.

Maybe need to do in C++ ?   

Issue sevt.py:SAB taking minutes for only 1M photons ? Due to many distinct histories
---------------------------------------------------------------------------------------


::

    INFO:opticks.ana.pvplt:SEvt.__init__  symbol b pid -1 opt  off [0. 0. 0.] 
    SEvt symbol b pid -1 opt  off [0. 0. 0.] b.f.base /data/blyth/opticks/GEOM/J23_1_0_rc3_ok0/G4CXTest/ALL0/n001 
    [--- ab = SAB(a,b) ----
    ^C---------------------------------------------------------------------------
    KeyboardInterrupt                         Traceback (most recent call last)
    ~/opticks/g4cx/tests/G4CXTest_GEOM.py in <module>
         46 
         47     print("[--- ab = SAB(a,b) ----")
    ---> 48     ab = None if a is None or b is None else SAB(a,b)
         49     print("]--- ab = SAB(a,b) ----")
         50 

    ~/opticks/sysrap/sevt.py in __init__(self, a, b)
        730             qcf0 = None
        731         else:
    --> 732             qcf = QCF( a.q, b.q, symbol="qcf")
        733             qcf0 = QCFZero(qcf) if "ZERO" in os.environ else None
        734         pass

    ~/opticks/ana/qcf.py in __init__(self, _aq, _bq, symbol)
         78 
         79         for i, q in enumerate(qu):
    ---> 80             ai_ = np.where(aqu.u == q )[0]           # find indices in the a and b unique lists
         81             bi_ = np.where(bqu.u == q )[0]
         82             ai = ai_[0] if len(ai_) == 1 else -1

    KeyboardInterrupt: 
    > /Users/blyth/opticks/ana/qcf.py(80)__init__()
         78 
         79         for i, q in enumerate(qu):
    ---> 80             ai_ = np.where(aqu.u == q )[0]           # find indices in the a and b unique lists
         81             bi_ = np.where(bqu.u == q )[0]
         82             ai = ai_[0] if len(ai_) == 1 else -1

    ipdb>               

    ipdb> p len(a.q)
    1000000
    ipdb> p len(b.q)
    1000000
    ipdb>

    ipdb> p len(aqu.u)
    79908
    ipdb> p len(bqu.u)
    79726



Forming this array is the thing that take the time::

    In [5]: ab.qcf.ab.shape
    Out[5]: (137467, 3, 2)

    In [7]: ab.qcf.ab.reshape(-1,6)[:100]
    Out[7]:
    array([[     0,      0,      2,      5, 127200, 127884],
           [    -1,      1,     -1,  44497,      0,      1],
           [     1,     -1, 532164,     -1,      1,      0],
           [    -1,      2,     -1,  58429,      0,      1],
           [    -1,      3,     -1, 199377,      0,      1],
           [     2,      4,     36,     80,   9410,   9351],
           [     3,      5,  25404,  21834,     27,     25],
           [     4,      6,   2809,  39996,    143,    108],
           [     5,      7,  95685,  91478,      2,      2],
           [     6,      8, 201662, 123891,      3,      7],
           [    -1,      9,     -1, 114142,      0,      1],
           [    -1,     10,     -1, 769322,      0,      1],
           [     7,     -1, 492829,     -1,      1,      0],
           [     8,     -1, 337860,     -1,      1,      0],
              ^^^^^^^^^^^^  ^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^
              internal        external first      counts  
              A,B             index
              indices




DONE : Forming sequence histories in C++
-----------------------------------------

Straightforward::

    sysrap/tests/OpticksPhotonTest.sh
    sysrap/tests/sseq_test.sh

::

    In [8]: ab.qcf.aq
    Out[8]:
    array([[b'TO BT BT DR BT SA                                                                               '],
           [b'TO SC SC SC BT BT BT BT BT BT SA                                                                '],
           [b'TO AB                                                                                           '],
           [b'TO RE RE BT BT BT BT BT BT SA                                                                   '],
           [b'TO SC SC SC BT BT BT BT DR BT BT BT BT BT BT SA                                                 '],
           ...,
           [b'TO AB                                                                                           '],
           [b'TO AB                                                                                           '],
           [b'TO SC BT BT BT BT BT BT SA                                                                      '],
           [b'TO BT BT BT BT BT BT BR BT BT BT BT BT BT BT BT SA                                              '],
           [b'TO BT BT BT BT BT BT SD                                                                         ']], dtype='|S96')

::

    epsilon:~ blyth$ ~/opticks/sysrap/tests/OpticksPhotonTest.sh
    2023-11-22 20:37:38.691 INFO  [10083845] [main@258]  sysrap.OpticksPhotonTest 
    OpticksPhotonTest:test_load_seq
     _path $TMP/GEOM/$GEOM/G4CXTest/ALL0/p001/seq.npy
     path  /data/blyth/opticks/GEOM/J23_1_0_rc3_ok0/G4CXTest/ALL0/p001/seq.npy
     a (1000000, 2, 2, )
    TO BT BT DR BT SA                                                                               
    TO SC SC SC BT BT BT BT BT BT SA                                                                
    TO AB                                                                                           
    TO RE RE BT BT BT BT BT BT SA                                                                   
    TO SC SC SC BT BT BT BT DR BT BT BT BT BT BT SA                                                 
    ...
    TO AB                                                                                           
    TO AB                                                                                           
    TO SC BT BT BT BT BT BT SA                                                                      
    TO BT BT BT BT BT BT BR BT BT BT BT BT BT BT BT SA                                              
    TO BT BT BT BT BT BT SD                                                                         
    epsilon:~ blyth$ 




Left field : use 128 bit big int
------------------------------------

::

    sysrap/tests/sbigint_test.cc
    sysrap/tests/sbigint_test.sh




