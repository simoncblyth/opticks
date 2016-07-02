Comparing Opticks with G4 for test geometry and light sources
=================================================================

summary
--------

* P polarized matches well (with polvec in initial direction of the photon) 

* S polarized are way off, looks to be due to Op playing Russian Roulette with S/P Polarization choice

  * TODO: implement the G4 way of doing things in Op 


TODO
-----

* check with simple box-in-box geometry with planar reflection


::

    ---- P ----- 
    Evt(6,"torch","rainbow","P Op")
                    8ccd     943087 :                           TORCH BT BT SA 
                     8bd      30954 :                              TORCH BR SA 
                   8cbcd      19554 :                        TORCH BT BR BT SA 
                  8cbbcd       2540 :                     TORCH BT BR BR BT SA 
                   86ccd       1239 :                        TORCH BT BT BS SA 
                 8cbbbcd        756 :                  TORCH BT BR BR BR BT SA 
                     86d        530 :                              TORCH BS SA 
                8cbbbbcd        310 :               TORCH BT BR BR BR BR BT SA 
                     4cd        172 :                              TORCH BT AB 
               8cbbbbbcd        146 :            TORCH BT BR BR BR BR BR BT SA 
                    4ccd        135 :                           TORCH BT BT AB 
                   8c6cd        132 :                        TORCH BT BS BT SA 
              bbbbbbbbcd         96 :         TORCH BT BR BR BR BR BR BR BR BR 
              8cbbbbbbcd         76 :         TORCH BT BR BR BR BR BR BR BT SA 
                      4d         51 :                                 TORCH AB 
    tot: 1000000
    Evt(-6,"torch","rainbow","P G4")
                    8ccd     943093 :                           TORCH BT BT SA 
                     8bd      31029 :                              TORCH BR SA 
                   8cbcd      19431 :                        TORCH BT BR BT SA 
                  8cbbcd       2601 :                     TORCH BT BR BR BT SA 
                   86ccd       1212 :                        TORCH BT BT BS SA 
                 8cbbbcd        757 :                  TORCH BT BR BR BR BT SA 
                     86d        517 :                              TORCH BS SA 
                8cbbbbcd        316 :               TORCH BT BR BR BR BR BT SA 
                     4cd        202 :                              TORCH BT AB 
               8cbbbbbcd        144 :            TORCH BT BR BR BR BR BR BT SA 
                   8c6cd        144 :                        TORCH BT BS BT SA 
              8bbbbbbbcd        127 :         TORCH BT BR BR BR BR BR BR BR SA 
                    4ccd        105 :                           TORCH BT BT AB 
              8cbbbbbbcd         77 :         TORCH BT BR BR BR BR BR BR BT SA 
                      4d         54 :                                 TORCH AB 
    tot: 1000000
    ---- S ----- 
    Evt(5,"torch","rainbow","S Op")
                    8ccd     819264 :                           TORCH BT BT SA 
                     8bd     101941 :                              TORCH BR SA 
                   8cbcd      61864 :                        TORCH BT BR BT SA 
                  8cbbcd       9691 :                     TORCH BT BR BR BT SA 
                 8cbbbcd       2557 :                  TORCH BT BR BR BR BT SA 
                8cbbbbcd       1049 :               TORCH BT BR BR BR BR BT SA 
                   86ccd       1047 :                        TORCH BT BT BS SA 
                     86d        507 :                              TORCH BS SA 
               8cbbbbbcd        485 :            TORCH BT BR BR BR BR BR BT SA 
              bbbbbbbbcd        311 :         TORCH BT BR BR BR BR BR BR BR BR 
              8cbbbbbbcd        262 :         TORCH BT BR BR BR BR BR BR BT SA 
                     4cd        162 :                              TORCH BT AB 
                    86bd        140 :                           TORCH BR BS SA 
              cbbbbbbbcd        136 :         TORCH BT BR BR BR BR BR BR BR BT 
                   8c6cd        128 :                        TORCH BT BS BT SA 
    tot: 1000000
    Evt(-5,"torch","rainbow","S G4")
                    8ccd     881262 :                           TORCH BT BT SA 
                     8bd      66471 :                              TORCH BR SA 
                   8cbcd      40644 :                        TORCH BT BR BT SA 
                  8cbbcd       6131 :                     TORCH BT BR BR BT SA 
                 8cbbbcd       1706 :                  TORCH BT BR BR BR BT SA 
                   86ccd       1065 :                        TORCH BT BT BS SA 
                8cbbbbcd        708 :               TORCH BT BR BR BR BR BT SA 
                     86d        484 :                              TORCH BS SA 
               8cbbbbbcd        319 :            TORCH BT BR BR BR BR BR BT SA 
              8bbbbbbbcd        297 :         TORCH BT BR BR BR BR BR BR BR SA 
                     4cd        170 :                              TORCH BT AB 
              8cbbbbbbcd        160 :         TORCH BT BR BR BR BR BR BR BT SA 
                   8c6cd        115 :                        TORCH BT BS BT SA 
                    4ccd         99 :                           TORCH BT BT AB 
                    86bd         69 :                           TORCH BR BS SA 
    tot: 1000000



ggv-rainbow P pol
-------------------

::

    In [8]: evt_op.p.rpol_(0)
    Out[8]: 
    array([[ 1.,  0.,  0.],
           [ 1.,  0.,  0.],
           [ 1.,  0.,  0.],
           ..., 
           [ 1.,  0.,  0.],
           [ 1.,  0.,  0.],
           [ 1.,  0.,  0.]])

    In [9]: evt_g4.p.rpol_(0)
    Out[9]: 
    array([[ 1.,  0.,  0.],
           [ 1.,  0.,  0.],
           [ 1.,  0.,  0.],
           ..., 
           [ 1.,  0.,  0.],
           [ 1.,  0.,  0.],
           [ 1.,  0.,  0.]])



ggv-rainbow S pol
-------------------


::

    In [4]: evt_g4.s.rpol_(0)
    Out[4]: 
    array([[ 0.    , -0.0709,  1.    ],
           [ 0.    ,  0.1496,  0.9921],
           [ 0.    , -0.1024,  0.9921],
           ..., 
           [ 0.    ,  0.0866,  1.    ],
           [ 0.    , -0.0551,  1.    ],
           [ 0.    ,  0.0236,  1.    ]])

    In [5]: evt_op.s.rpol_(0)
    Out[5]: 
    array([[ 0.    , -0.9291,  0.378 ],
           [ 0.    , -0.9685,  0.2441],
           [ 0.    ,  0.    ,  1.    ],
           ..., 
           [ 0.    , -0.937 , -0.3465],
           [ 0.    ,  0.4016,  0.9134],
           [ 0.    ,  1.    , -0.0551]])


::

    In [32]: evt_g4.s.rpost_(0)
    Out[32]: 
    array([[-600.0183,  -42.7747,  -26.7342,    0.    ],
           [-600.0183,   88.5891,  -44.203 ,    0.    ],
           [-600.0183,  -63.3198,  -62.6606,    0.    ],
           ..., 
           [-600.0183,   52.4064,  -78.1518,    0.    ],
           [-600.0183,  -32.5938,  -64.8213,    0.    ],
           [-600.0183,   15.9673,   20.9845,    0.    ]])

    In [34]: pos_g4 = evt_g4.s.rpost_(0)[:,:3]

    In [35]: phi_g4 = np.arctan2(pos_g4[:,2], pos_g4[:,1])    # flat -pi:pi 



    In [38]: pos_op = evt_op.s.rpost_(0)[:,:3]

    In [39]: pos_op
    Out[39]: 
    array([[-599.9817,   32.4473,   79.6899],
           [-599.9817,   23.6579,   93.0204],
           [-599.9817,   19.7394,    0.0366],
           ..., 
           [-599.9817,  -33.9488,   91.4823],
           [-599.9817,   20.5817,   -9.0091],
           [-599.9817,   -0.9888,  -18.9703]])

    In [40]: phi_op = np.arctan2(pos_op[:,1], pos_op[:,2])     # flat -pi:pi



::

    In [44]: opx = np.zeros((len(phi_op),3), dtype=np.float32)

    In [49]: opx[:,1] = -np.sin(phi_op)

    In [50]: opx[:,2] = np.cos(phi_op)

    In [51]: opx
    Out[51]: 
    array([[ 0.    , -0.9262,  0.3771],
           [ 0.    , -0.9691,  0.2465],
           [ 0.    , -0.0019,  1.    ],
           ..., 
           [ 0.    , -0.9375, -0.3479],
           [ 0.    ,  0.401 ,  0.9161],
           [ 0.    ,  0.9986, -0.0521]], dtype=float32)


    In [52]: evt_op.s.rpol_(0)
    Out[52]: 
    array([[ 0.    , -0.9291,  0.378 ],
           [ 0.    , -0.9685,  0.2441],
           [ 0.    ,  0.    ,  1.    ],
           ..., 
           [ 0.    , -0.937 , -0.3465],
           [ 0.    ,  0.4016,  0.9134],
           [ 0.    ,  1.    , -0.0551]])





Polarization Progression
--------------------------

S Op pol vector stays alive modulo sign flips, never leaking into X::


    In [6]: evt_op.s.rpol_(0)
    Out[6]: 
    array([[ 0.    , -0.9291,  0.378 ],
           [ 0.    , -0.9685,  0.2441],
           [ 0.    ,  0.    ,  1.    ],
           ..., 
           [ 0.    , -0.937 , -0.3465],
           [ 0.    ,  0.4016,  0.9134],
           [ 0.    ,  1.    , -0.0551]])

    In [7]: evt_op.s.rpol_(1)
    Out[7]: 
    array([[ 0.    , -0.9291,  0.378 ],
           [ 0.    , -0.9685,  0.2441],
           [ 0.    ,  0.    ,  1.    ],
           ..., 
           [ 0.    , -0.937 , -0.3465],
           [ 0.    ,  0.4016,  0.9134],
           [ 0.    ,  1.    , -0.0551]])

    In [8]: evt_op.s.rpol_(2)
    Out[8]: 
    array([[ 0.    ,  0.9291, -0.378 ],
           [ 0.    ,  0.9685, -0.2441],
           [ 0.    ,  0.    , -1.    ],
           ..., 
           [ 0.    ,  0.937 ,  0.3465],
           [ 0.    , -0.4016, -0.9134],
           [ 0.    , -1.    ,  0.0551]])

    In [9]: evt_op.s.rpol_(3)
    Out[9]: 
    array([[ 0.    ,  0.9291, -0.378 ],
           [ 0.    ,  0.9685, -0.2441],
           [ 0.    ,  0.    , -1.    ],
           ..., 
           [ 0.    ,  0.937 ,  0.3465],
           [ 0.    , -0.4016, -0.9134],
           [ 0.    , -1.    ,  0.0551]])

    In [10]: evt_op.s.rpol_(4)
    Out[10]: 
    array([[-1.    , -1.    , -1.    ],
           [ 0.    ,  0.9685, -0.2441],
           [-1.    , -1.    , -1.    ],
           ..., 
           [-1.    , -1.    , -1.    ],
           [-1.    , -1.    , -1.    ],
           [-1.    , -1.    , -1.    ]])

    In [11]: 




G4 pol different approach::

    In [11]: evt_g4.s.rpol_(0)
    Out[11]: 
    array([[ 0.    ,  0.8504, -0.5276],
           [ 0.    , -0.8976, -0.4488],
           [ 0.    ,  0.7087, -0.7008],
           ..., 
           [ 0.    ,  0.0787,  1.    ],
           [ 0.    ,  0.2283, -0.9764],
           [ 0.    ,  0.2835,  0.9606]])

    In [12]: evt_g4.s.rpol_(1)
    Out[12]: 
    array([[-0.063 ,  0.8425, -0.5276],
           [-0.1417,  0.8189,  0.5591],
           [-0.0079,  0.7087, -0.7008],
           ..., 
           [-0.3307,  0.2283, -0.9134],
           [-0.1811,  0.6299,  0.7559],
           [-0.1102,  0.7638, -0.6378]])

    In [13]: evt_g4.s.rpol_(2)
    Out[13]: 
    array([[ 0.126 ,  0.1102, -0.9843],
           [-0.1417,  0.8189,  0.5591],
           [ 0.0079,  0.7008, -0.7165],
           ..., 
           [ 0.622 ,  0.0787,  0.7717],
           [ 0.3622,  0.2283, -0.9055],
           [ 0.2126,  0.2835,  0.937 ]])

    In [14]: evt_g4.s.rpol_(3)
    Out[14]: 
    array([[ 0.126 ,  0.1102, -0.9843],
           [-1.    , -1.    , -1.    ],
           [ 0.0079,  0.7008, -0.7165],
           ..., 
           [ 0.622 ,  0.0787,  0.7717],
           [ 0.3622,  0.2283, -0.9055],
           [ 0.2126,  0.2835,  0.937 ]])

    In [15]: 



