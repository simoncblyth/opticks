OKX4Test_planBuffer_mm0
==========================

Nothing smoking with the planes

::

    args: /opt/local/bin/ipython -i /tmp/blyth/opticks/bin/ab/ab-i.py
    [2018-07-04 21:00:37,274] p35833 {/Users/blyth/opticks/ana/mesh.py:37} INFO - Mesh for idpath : /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1 
    [2018-07-04 21:00:38,298] p35833 {/Users/blyth/opticks/ana/mesh.py:37} INFO - Mesh for idpath : /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1 
    [2018-07-04 21:00:39,278] p35833 {/Users/blyth/opticks/ana/mesh.py:37} INFO - Mesh for idpath : /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103 
    [2018-07-04 21:00:39,279] p35833 {/Users/blyth/opticks/ana/mesh.py:37} INFO - Mesh for idpath : /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1 
     partIdx:     4 count:     1 tc: 12 tcn:            cylinder gta: 1 gtb: 1 mx:       1.0   
     partIdx:    18 count:     2 tc: 12 tcn:            cylinder gta: 1 gtb: 1 mx:       1.0   
     partIdx:  3496 count:     3 tc: 12 tcn:            cylinder gta: 1 gtb: 1 mx:       1.0   
     partIdx:  3510 count:     4 tc: 12 tcn:            cylinder gta: 1 gtb: 1 mx:       1.0   
     num_parts 11984  num_discrepant :     4   cut:0.0005  

    In [1]: la.shape
    Out[1]: (672, 4)

    In [2]: lb.shape
    Out[2]: (672, 4)

    In [3]: la
    Out[3]: 
    array([[     -0.    ,       0.    ,      -1.    ,    9095.985 ],
           [      0.    ,       0.    ,       1.    ,   -8089.015 ],
           [      0.8379,       0.5458,       0.    , -451659.72  ],
           ...,
           [      0.9778,      -0.2096,       0.    ,  154027.8   ],
           [      0.    ,       0.    ,       1.    ,   -7107.49  ],
           [      0.    ,       0.    ,      -1.    ,    7157.51  ]], dtype=float32)

    In [4]: lb
    Out[4]: 
    array([[     -0.    ,       0.    ,      -1.    ,    9095.985 ],
           [      0.    ,       0.    ,       1.    ,   -8089.015 ],
           [      0.8379,       0.5458,       0.    , -451659.72  ],
           ...,
           [      0.9778,      -0.2096,       0.    ,  154027.8   ],
           [      0.    ,       0.    ,       1.    ,   -7107.49  ],
           [      0.    ,       0.    ,      -1.    ,    7157.51  ]], dtype=float32)

    In [5]: la - lb 
    Out[5]: 
    array([[0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.],
           ...,
           [0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.]], dtype=float32)

    In [6]: lab = la - lb 

    In [7]: lab.max()
    Out[7]: 0.15625

    In [8]: lab.min()
    Out[8]: -0.125

    In [10]: np.max(lab, axis=1).shape
    Out[10]: (672,)

    In [11]: xlab = np.max(lab, axis=1)

    In [12]: np.where( xlab > 0.1)
    Out[12]: (array([ 24,  34,  64,  74, 104, 114, 144, 154, 360, 365, 400, 405, 440, 445, 480, 485]),)

    In [13]: wlab = np.where( xlab > 0.1)

    In [14]: la[wlab]
    Out[14]: 
    array([[      0.8231,      -0.5678,       0.    ,  442361.62  ],
           [      0.5678,       0.8231,       0.    , -665388.9   ],
           [      0.8231,      -0.5678,       0.    ,  442361.62  ],
           [      0.5678,       0.8231,       0.    , -665388.9   ],
           [      0.8231,      -0.5678,       0.    ,  442361.62  ],
           [      0.5678,       0.8231,       0.    , -665388.9   ],
           [      0.8231,      -0.5678,       0.    ,  442361.62  ],
           [      0.5678,       0.8231,       0.    , -665388.9   ],
           [      0.8231,      -0.5678,       0.    ,  447666.56  ],
           [      0.9836,       0.1805,       0.    , -156807.05  ],
           [      0.8231,      -0.5678,       0.    ,  447666.56  ],
           [      0.9836,       0.1805,       0.    , -156807.05  ],
           [      0.8231,      -0.5678,       0.    ,  447666.56  ],
           [      0.9836,       0.1805,       0.    , -156807.05  ],
           [      0.8231,      -0.5678,       0.    ,  447666.56  ],
           [      0.9836,       0.1805,       0.    , -156807.05  ]], dtype=float32)

    In [15]: lb[wlab]
    Out[15]: 
    array([[      0.8231,      -0.5678,       0.    ,  442361.47  ],
           [      0.5678,       0.8231,       0.    , -665389.    ],
           [      0.8231,      -0.5678,       0.    ,  442361.47  ],
           [      0.5678,       0.8231,       0.    , -665389.    ],
           [      0.8231,      -0.5678,       0.    ,  442361.47  ],
           [      0.5678,       0.8231,       0.    , -665389.    ],
           [      0.8231,      -0.5678,       0.    ,  442361.47  ],
           [      0.5678,       0.8231,       0.    , -665389.    ],
           [      0.8231,      -0.5678,       0.    ,  447666.4   ],
           [      0.9836,       0.1805,       0.    , -156807.16  ],
           [      0.8231,      -0.5678,       0.    ,  447666.4   ],
           [      0.9836,       0.1805,       0.    , -156807.16  ],
           [      0.8231,      -0.5678,       0.    ,  447666.4   ],
           [      0.9836,       0.1805,       0.    , -156807.16  ],
           [      0.8231,      -0.5678,       0.    ,  447666.4   ],
           [      0.9836,       0.1805,       0.    , -156807.16  ]], dtype=float32)

    In [16]: 


