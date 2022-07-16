RaindropRockAirWaterSmall_make_plot
======================================



For illustration purposes it helps to use input photons 
within the simtrace plane so add input_photons name "UpXZ1000_f8.npy"

::

    In [2]: cuss(a.seq[:,0])
    Out[2]: 
    CUSS([['w0', '                   TO BT BT SA', '           36045', '            8805'],
          ['w1', '                      TO BR SA', '            2237', '             599'],
          ['w2', '                TO BT BR BT SA', '          576461', '             503'],
          ['w3', '             TO BT BR BR BT SA', '         9223117', '              43'],
          ['w4', '                      TO BT AB', '            1229', '              28'],
          ['w5', '          TO BT BR BR BR BT SA', '       147569613', '              13'],
          ['w6', '       TO BT BR BR BR BR BT SA', '      2361113549', '               2'],
          ['w7', '                TO BT SC BT SA', '          575181', '               2'],
          ['w8', '                   TO BT BR AB', '           19405', '               2'],
          ['w9', '          TO BT BR BR BR BR AB', '        79412173', '               1'],
          ['w10', '                TO BT BT SC SA', '          552141', '               1'],
          ['w11', '                      TO SC SA', '            2157', '               1']], dtype=object)

::

    In [3]: w5
    Out[3]: array([ 142, 3305, 3339, 3451, 3474, 3922, 6281, 6795, 7237, 7516, 8521, 9358, 9719])

::

    epsilon:issues blyth$ gx
    /Users/blyth/opticks/g4cx
    epsilon:g4cx blyth$ gx ; PIDX=142 ./gxt.sh 



    In [4]: a.record[w5,0,0]
    Out[4]: 
    array([[-38.662,  28.287, -99.   ,   0.   ],
           [ 34.02 , -33.343, -99.   ,   0.   ],
           [ 18.038,  45.336, -99.   ,   0.   ],
           [-41.82 ,  22.947, -99.   ,   0.   ],
           [-26.552, -23.051, -99.   ,   0.   ],
           [ -8.587, -38.941, -99.   ,   0.   ],
           [-17.089, -45.711, -99.   ,   0.   ],
           [ 41.805, -15.498, -99.   ,   0.   ],
           [ 16.744,  45.667, -99.   ,   0.   ],
           [-38.983, -29.055, -99.   ,   0.   ],
           [-27.737, -38.445, -99.   ,   0.   ],
           [ 45.595, -13.386, -99.   ,   0.   ],
           [-40.295,  18.016, -99.   ,   0.   ]], dtype=float32)

    In [5]: a.record[w6,0,0]
    Out[5]: 
    array([[-43.501,  19.577, -99.   ,   0.   ],
           [ 46.132,  15.809, -99.   ,   0.   ]], dtype=float32)




High stats 1M : A-B
----------------------


::

    In [4]: wm = np.where( A.t.view('|S64') == B.t2.view('|S64') )[0] 
    In [5]: wm                                                                                                                                                  
    Out[5]: 
    array([     0,      1,      2,      3,      4,      5,      6,      7,      8,      9,     10,     11,     12,     13,     14,     15, ..., 999984, 999985, 999986, 999987, 999988, 999989, 999990,
           999991, 999992, 999993, 999994, 999995, 999996, 999997, 999998, 999999])

    In [6]: len(wm)                                                                                                                                             
    Out[6]: 999978

    In [7]: len(we)                                                                                                                                             
    Out[7]: 22

    In [10]: len(wm)                                                                                                                                            
    Out[10]: 999978

    In [11]: s_wm = a.seq[wm,0]                                                                                                                                 

    In [12]: o_wm = cuss(s_wm, wm)                                                                                                                              

    In [13]: print(o_wm)                                                                                                                                        
    [['w0' '                   TO BT BT SA' '           36045' '          885127']
     ['w1' '                      TO BR SA' '            2237' '           59974']
     ['w2' '                TO BT BR BT SA' '          576461' '           46257']
     ['w3' '             TO BT BR BR BT SA' '         9223117' '            4725']
     ['w4' '                      TO BT AB' '            1229' '            2180']
     ['w5' '          TO BT BR BR BR BT SA' '       147569613' '             947']
     ['w6' '       TO BT BR BR BR BR BT SA' '      2361113549' '             218']
     ['w7' '                TO BT SC BT SA' '          575181' '             188']
     ['w8' '                   TO BT BR AB' '           19405' '             107']
     ['w9' '    TO BT BR BR BR BR BR BT SA' '     37777816525' '              71']
     ['w10' '                      TO SC SA' '            2157' '              45']
     ['w11' '                TO BT BT SC SA' '          552141' '              28']
     ['w12' ' TO BT BR BR BR BR BR BR BT SA' '    604445064141' '              24']
     ['w13' '             TO BT BR SC BT SA' '         9202637' '              12']
     ['w14' '                TO BT BR BR AB' '          310221' '              11']
     ['w15' '                TO SC BT BT SA' '          576621' '               9']
     ['w16' '          TO BT BT SC BT BT SA' '       147614925' '               9']
     ['w17' '             TO BT SC BR BT SA' '         9221837' '               8']
     ['w18' '                         TO AB' '              77' '               7']
     ['w19' ' TO BT BR BR BR BR BR BR BR BT' '    875028003789' '               6']
     ['w20' '             TO BT BR BT SC SA' '         8833997' '               4']
     ['w21' '                   TO BR SC SA' '           34493' '               4']
     ['w22' '             TO BT BR BR BR AB' '         4963277' '               3']
     ['w23' ' TO BT BR BR BR BR BR BR BR BR' '    806308527053' '               2']
     ['w24' ' TO BT BR SC BR BR BR BR BR BT' '    875027983309' '               1']
     ['w25' '             TO SC BT BR BT SA' '         9223277' '               1']
     ['w26' '          TO BT BR BR BR BR AB' '        79412173' '               1']
     ['w27' '          TO BT SC BR BR BT SA' '       147568333' '               1']
     ['w28' '       TO BT BR SC BR BR BT SA' '      2361093069' '               1']
     ['w29' '                   TO SC BR SA' '           35693' '               1']
     ['w30' '                   TO BT BT AB' '           19661' '               1']
     ['w31' '    TO BT BR BR BR BR BR BR AB' '     20329511885' '               1']
     ['w32' '    TO BT SC BR BR BR BR BT SA' '     37777815245' '               1']
     ['w33' '    TO BT BT SC BT BR BR BT SA' '     37777861837' '               1']
     ['w34' '                      TO BR AB' '            1213' '               1']
     ['w35' '             TO BT BT SC BR SA' '         9137357' '               1']]

    In [14]:                        

::

    In [20]: a.record[:,0,1,3] = 1.    


    In [29]: np.abs(a.record[wm] - b.record[wm]).max()                                                                                                          
    Out[29]: 0.018722534

    In [29]: np.abs(a.record[wm] - b.record[wm]).max()                                                                                                          
    Out[29]: 0.018722534

    In [30]: np.where( np.abs(a.record[wm] - b.record[wm]) > 0.01 )                                                                                             
    Out[30]: 
    (array([ 18157, 125121, 467717, 499537, 624529, 695184, 759091, 779861, 938053]),
     array([1, 1, 3, 1, 1, 2, 4, 1, 1]),
     array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
     array([2, 2, 2, 2, 2, 0, 2, 2, 2]))

    In [31]: dv_wm = np.unique(np.where( np.abs(a.record[wm] - b.record[wm]) > 0.01 )[0])                                                                       

    In [32]: dv_wm                                                                                                                                              
    Out[32]: array([ 18157, 125121, 467717, 499537, 624529, 695184, 759091, 779861, 938053])

    In [33]: seqhis_(a.seq[dv_wm,0] )                                                                                                                           
    Out[33]: 
    ['TO AB',
     'TO BT BT SA',
     'TO BT BT SA',
     'TO BT BT SA',
     'TO BT BT SA',
     'TO BT BT SA',
     'TO BT BT SA',
     'TO BT BT SA',
     'TO BT BT SA']

    In [34]: wm[dv_wm]                                                                                                                                          
    Out[34]: array([ 18157, 125124, 467729, 499549, 624543, 695198, 759108, 779878, 938075])

    In [35]: dv_wm = wm[dv_wm]                                                                                                                                  

    In [36]: seqhis_(a.seq[dv_wm,0])                                                                                                                            
    Out[36]: 
    ['TO AB',
     'TO AB',
     'TO BT BT AB',
     'TO AB',
     'TO AB',
     'TO BR AB',
     'TO SC BT BT SA',
     'TO AB',
     'TO AB']

    In [37]:                                  



    In [46]: a.record[wm][dv_wm_]                                                                                                                               
    Out[46]: array([-82.325, -38.217,  64.72 , -72.788, -82.925, -37.115,   8.126, -75.773, -82.925], dtype=float32)

    In [47]: b.record[wm][dv_wm_]                                                                                                                               
    Out[47]: array([-82.311, -38.203,  64.733, -72.774, -82.907, -37.13 ,   8.115, -75.754, -82.907], dtype=float32)




