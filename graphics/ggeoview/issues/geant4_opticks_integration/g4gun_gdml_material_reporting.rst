G4GUN Material Reporting
==========================

Approach
----------

* Material/lookup tables OR smth needs adapting to GDML/G4 running ?


Fishy flags in history
------------------------

* unexpected zero flags in history
* all those BT are fishy, or smth wrong with material recording ...

::
    In [1]: run g4gun.py
    Evt(-1,"G4Gun","G4Gun","G4Gun/G4Gun/-1 : ", seqs="[]")


    In [2]: print evt.history.table
                            -1:G4Gun 
                      4f        0.365             35       [2 ] G4GUN AB
               4cccccccf        0.333             32       [9 ] G4GUN BT BT BT BT BT BT BT AB
                    4ccf        0.052              5       [4 ] G4GUN BT BT AB
                4cc0cccf        0.052              5       [8 ] G4GUN BT BT BT ?0? BT BT AB
              cccbcccccf        0.052              5       [10] G4GUN BT BT BT BT BT BR BT BT BT
                  4ccccf        0.031              3       [6 ] G4GUN BT BT BT BT AB
                     4cf        0.010              1       [3 ] G4GUN BT AB
                4ccccccf        0.010              1       [8 ] G4GUN BT BT BT BT BT BT AB
               4cc0ccc6f        0.010              1       [9 ] G4GUN SC BT BT BT ?0? BT BT AB
              cccccccccf        0.010              1       [10] G4GUN BT BT BT BT BT BT BT BT BT
              4cc00cc0cf        0.010              1       [10] G4GUN BT ?0? BT BT ?0? ?0? BT BT AB
              4cc0cbb0cf        0.010              1       [10] G4GUN BT ?0? BR BR BT ?0? BT BT AB
              4ccccccc6f        0.010              1       [10] G4GUN SC BT BT BT BT BT BT BT AB
              cbcccccccf        0.010              1       [10] G4GUN BT BT BT BT BT BT BT BR BT
              ccbccccc6f        0.010              1       [10] G4GUN SC BT BT BT BT BT BR BT BT
              ccbccccccf        0.010              1       [10] G4GUN BT BT BT BT BT BT BR BT BT
              4cbccccccf        0.010              1       [10] G4GUN BT BT BT BT BT BT BR BT AB
                                  96         1.00 

    In [3]: print evt.material.table
                            -1:G4Gun 
                      11        0.365             35       [2 ] Gd Gd
               111111111        0.344             33       [9 ] Gd Gd Gd Gd Gd Gd Gd Gd Gd
              1111111111        0.135             13       [10] Gd Gd Gd Gd Gd Gd Gd Gd Gd Gd
                11111111        0.062              6       [8 ] Gd Gd Gd Gd Gd Gd Gd Gd
                    1111        0.052              5       [4 ] Gd Gd Gd Gd
                  111111        0.031              3       [6 ] Gd Gd Gd Gd Gd Gd
                     111        0.010              1       [3 ] Gd Gd Gd
                                  96         1.00 


    In [10]: evt.material.table
    Out[10]: 
                            -1:G4Gun 
                      11        0.928          22630       [2 ] Gd Gd
               111111111        0.041           1003       [9 ] Gd Gd Gd Gd Gd Gd Gd Gd Gd
              1111111111        0.023            550       [10] Gd Gd Gd Gd Gd Gd Gd Gd Gd Gd
                    1111        0.003             73       [4 ] Gd Gd Gd Gd
                 1111111        0.002             41       [7 ] Gd Gd Gd Gd Gd Gd Gd
                  111111        0.002             39       [6 ] Gd Gd Gd Gd Gd Gd
                11111111        0.001             26       [8 ] Gd Gd Gd Gd Gd Gd Gd Gd
                     111        0.000             12       [3 ] Gd Gd Gd
                   11111        0.000              8       [5 ] Gd Gd Gd Gd Gd
                               24382         1.00 



