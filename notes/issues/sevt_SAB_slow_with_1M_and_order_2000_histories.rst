sevt_SAB_slow_with_1M_and_order_2000_histories
================================================


Overview
----------

Obvious speedup in would be to preselect removing 
low frequency histories. But that dont look
possible in the python/numpy way.

Maybe need to do in C++ ?   



HMM recall doing something similar before with thrust : for the OpenGL/imgui category interface
--------------------------------------------------------------------------------------------------


TODO: need to bring the below into new workflow, it does most 
of np.unique on device already (no first indices though) 

::

    epsilon:cudarap blyth$ opticks-f thrust::sort 
    ./thrustrap/TSparse_.cu:   thrust::sort_by_key 
    ./thrustrap/TSparse_.cu:    thrust::sort(data.begin(), data.end());
    ./thrustrap/TSparse_.cu:    thrust::sort_by_key( 
    epsilon:opticks blyth$ 

::

    115 template <typename T>
    116 void TSparse<T>::count_unique()
    117 {
    118     typedef typename thrust::device_vector<T>::iterator Iterator;
    119 
    120     thrust::device_ptr<T> psrc = thrust::device_pointer_cast((T*)m_source.dev_ptr) ;
    121 
    122     strided_range<Iterator> src( psrc + m_source.begin, psrc + m_source.end, m_source.stride );
    123 
    124     thrust::device_vector<T> data(src.begin(), src.end());  // copy to avoid sorting original
    125 
    126     thrust::sort(data.begin(), data.end());
    127 
    128     // inner_product of sorted data with shifted by one self finds "edges" between values 
    129     m_num_unique = thrust::inner_product(
    130                                   data.begin(),data.end() - 1,   // first1, last1  
    131                                              data.begin() + 1,   // first2
    132                                                        int(1),   // output type init 
    133                                           thrust::plus<int>(),   // reduction operator
    134                                     thrust::not_equal_to<T>()    // pair-by-pair operator, returning 1 at edges 
    135                                       );
    136 
    137 
    138 #ifdef DEBUG
    139     printf("TSparse<T>::count_unique m_num_unique %d \n", m_num_unique) ;
    140 #endif
    141 
    142     m_values.resize(m_num_unique);
    143     m_counts.resize(m_num_unique);
    144 
    145     // find all unique key values with their counts
    146     thrust::reduce_by_key(
    147                                 data.begin(),    // keys_first
    148                                   data.end(),    // keys_last 
    149            thrust::constant_iterator<int>(1),    // values_first 
    150                             m_values.begin(),    // keys_output 
    151                             m_counts.begin()     // values_output
    152                          );
    153 
    154    // *reduce_by_key* is a generalization of reduce to key-value pairs. For each group
    155    // of consecutive keys in the range [keys_first, keys_last) that are equal,
    156    // reduce_by_key copies the first element of the group to the keys_output. 
    157    // The corresponding values in the range are reduced using the plus and the result
    158    // copied to values_output.
    159    //
    160    // As *data* is sorted this means get each unique key once in m_values together
    161    // the occurrent count in m_counts    
    162 
    163     thrust::sort_by_key(
    164                           m_counts.begin(),     // keys_first
    165                             m_counts.end(),     // keys_last 
    166                           m_values.begin(),     // values_first
    167                      thrust::greater<int>()
    168                        );
    169 
    170     // sorts keys and values into descending key order, as the counts are in 
    171     // the key slots this sorts in descending count order
    172 
    173 }




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



