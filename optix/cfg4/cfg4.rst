Comparing Opticks with G4 for test geometry and light sources
=================================================================

Sources are not yet wavelength aligned (but have a quick look at the splits).

S polarized splits look close::

    In [1]: run rainbow_scatter.py    

    In [4]: cu = evt_g4.s.history_table(slice(0,15))
                    8ccd     943506 :                           TORCH BT BT SA 
                     8bd      30896 :                              TORCH BR SA 
                   8cbcd      19251 :                        TORCH BT BR BT SA 
                  8cbbcd       2614 :                     TORCH BT BR BR BT SA 
                   86ccd       1153 :                        TORCH BT BT BS SA 
                 8cbbbcd        774 :                  TORCH BT BR BR BR BT SA 
                     86d        523 :                              TORCH BS SA 
                8cbbbbcd        289 :               TORCH BT BR BR BR BR BT SA 
                     4cd        165 :                              TORCH BT AB 
              8bbbbbbbcd        143 :         TORCH BT BR BR BR BR BR BR BR SA 
               8cbbbbbcd        134 :            TORCH BT BR BR BR BR BR BT SA 
                   8c6cd        124 :                        TORCH BT BS BT SA 
                    4ccd        121 :                           TORCH BT BT AB 
              8cbbbbbbcd         76 :         TORCH BT BR BR BR BR BR BR BT SA 
                      4d         64 :                                 TORCH AB 
    tot: 1000000

    In [5]: cu = evt_op.s.history_table(slice(0,15))
                    8ccd     943254 :                           TORCH BT BT SA 
                     8bd      30797 :                              TORCH BR SA 
                   8cbcd      19689 :                        TORCH BT BR BT SA 
                  8cbbcd       2455 :                     TORCH BT BR BR BT SA 
                   86ccd       1184 :                        TORCH BT BT BS SA 
                 8cbbbcd        740 :                  TORCH BT BR BR BR BT SA 
                     86d        526 :                              TORCH BS SA 
                8cbbbbcd        295 :               TORCH BT BR BR BR BR BT SA 
                     4cd        164 :                              TORCH BT AB 
               8cbbbbbcd        157 :            TORCH BT BR BR BR BR BR BT SA 
                    4ccd        126 :                           TORCH BT BT AB 
                   8c6cd        122 :                        TORCH BT BS BT SA 
              bbbbbbbbcd        111 :         TORCH BT BR BR BR BR BR BR BR BR 
              8cbbbbbbcd         86 :         TORCH BT BR BR BR BR BR BR BT SA 
                      4d         53 :                                 TORCH AB 
    tot: 1000000



P polarized splits are way off, much less reflection (but wavelengths not yet aligned)::

    In [6]: cu = evt_op.p.history_table(slice(0,15))
                    8ccd     818192 :                           TORCH BT BT SA 
                     8bd     102792 :                              TORCH BR SA 
                   8cbcd      62017 :                        TORCH BT BR BT SA 
                  8cbbcd       9689 :                     TORCH BT BR BR BT SA 
                 8cbbbcd       2703 :                  TORCH BT BR BR BR BT SA 
                8cbbbbcd       1038 :               TORCH BT BR BR BR BR BT SA 
                   86ccd        983 :                        TORCH BT BT BS SA 
                     86d        503 :                              TORCH BS SA 
               8cbbbbbcd        466 :            TORCH BT BR BR BR BR BR BT SA 
              bbbbbbbbcd        339 :         TORCH BT BR BR BR BR BR BR BR BR 
              8cbbbbbbcd        237 :         TORCH BT BR BR BR BR BR BR BT SA 
              cbbbbbbbcd        158 :         TORCH BT BR BR BR BR BR BR BR BT 
                     4cd        151 :                              TORCH BT AB 
                    86bd        145 :                           TORCH BR BS SA 
                   8c6cd        119 :                        TORCH BT BS BT SA 
    tot: 1000000

    In [7]: cu = evt_g4.p.history_table(slice(0,15))
                    8ccd     882102 :                           TORCH BT BT SA 
                     8bd      66277 :                              TORCH BR SA 
                   8cbcd      40151 :                        TORCH BT BR BT SA 
                  8cbbcd       5956 :                     TORCH BT BR BR BT SA 
                 8cbbbcd       1685 :                  TORCH BT BR BR BR BT SA 
                   86ccd       1076 :                        TORCH BT BT BS SA 
                8cbbbbcd        647 :               TORCH BT BR BR BR BR BT SA 
                     86d        530 :                              TORCH BS SA 
               8cbbbbbcd        311 :            TORCH BT BR BR BR BR BR BT SA 
              8bbbbbbbcd        301 :         TORCH BT BR BR BR BR BR BR BR SA 
              8cbbbbbbcd        177 :         TORCH BT BR BR BR BR BR BR BT SA 
                     4cd        165 :                              TORCH BT AB 
                   8c6cd        116 :                        TORCH BT BS BT SA 
                    4ccd        105 :                           TORCH BT BT AB 
                    86bd         99 :                           TORCH BR BS SA 
    tot: 1000000


