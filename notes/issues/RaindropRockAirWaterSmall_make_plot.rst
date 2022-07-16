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



