Comparing Opticks with G4 for test geometry and light sources
=================================================================

summary
--------

* ggv-rainbow splits match well for S-pol but not for P-pol

  * TODO: debug screwy polarizations


TODO
-----

* check with simple box-in-box geometry with planar reflection


ggv-rainbow
-------------

S splits are close::

    In [2]: cu = evt_g4.s.history_table(slice(0,15))
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

    In [3]: cu = evt_op.s.history_table(slice(0,15))
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



Huh, S is actually P polarization::

    In [14]: evt_op.s.rpol_(0)
    Out[14]: 
    array([[ 1.,  0.,  0.],
           [ 1.,  0.,  0.],
           [ 1.,  0.,  0.],
           ..., 
           [ 1.,  0.,  0.],
           [ 1.,  0.,  0.],
           [ 1.,  0.,  0.]])

    In [15]: evt_g4.s.rpol_(0)
    Out[15]: 
    array([[ 1.,  0.,  0.],
           [ 1.,  0.,  0.],
           [ 1.,  0.,  0.],
           ..., 
           [ 1.,  0.,  0.],
           [ 1.,  0.,  0.],
           [ 1.,  0.,  0.]])



P polarized splits are way off, much more reflection with Opticks::

    In [4]: cu = evt_op.p.history_table(slice(0,15))
                    8ccd     819264 :                           TORCH BT BT SA 
                     8bd     101941 :                              TORCH BR SA     <<<< lots more P-pol reflection from Opticks
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

    In [5]: cu = evt_g4.p.history_table(slice(0,15))
                    8ccd     881251 :                           TORCH BT BT SA 
                     8bd      66403 :                              TORCH BR SA    <<< lot less P-pol reflection from Geant4
                   8cbcd      40672 :                        TORCH BT BR BT SA 
                  8cbbcd       6011 :                     TORCH BT BR BR BT SA 
                 8cbbbcd       1792 :                  TORCH BT BR BR BR BT SA 
                   86ccd       1110 :                        TORCH BT BT BS SA 
                8cbbbbcd        682 :               TORCH BT BR BR BR BR BT SA 
                     86d        511 :                              TORCH BS SA 
              8bbbbbbbcd        322 :         TORCH BT BR BR BR BR BR BR BR SA 
               8cbbbbbcd        291 :            TORCH BT BR BR BR BR BR BT SA 
              8cbbbbbbcd        171 :         TORCH BT BR BR BR BR BR BR BT SA 
                     4cd        165 :                              TORCH BT AB 
                   8c6cd        135 :                        TORCH BT BS BT SA 
                    4ccd        119 :                           TORCH BT BT AB 
                    86bd         86 :                           TORCH BR BS SA 
    tot: 1000000




So called P is screwed up, with inconsistent axis::


    In [16]: evt_g4.p.rpol_(0)
    Out[16]: 
    array([[-0.0709,  1.    ,  0.    ],
           [ 0.1496,  0.9921,  0.    ],
           [-0.1024,  0.9921,  0.    ],
           ..., 
           [ 0.    ,  1.    ,  0.    ],
           [ 0.0787,  1.    ,  0.    ],
           [ 0.    ,  1.    ,  0.    ]])

    In [17]: evt_op.p.rpol_(0)
    Out[17]: 
    array([[ 0.    , -0.9291,  0.378 ],
           [ 0.    , -0.9685,  0.2441],
           [ 0.    ,  0.    ,  1.    ],
           ..., 
           [ 0.    , -0.937 , -0.3465],
           [ 0.    ,  0.4016,  0.9134],
           [ 0.    ,  1.    , -0.0551]])





