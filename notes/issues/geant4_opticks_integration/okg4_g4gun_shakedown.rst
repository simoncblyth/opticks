OKG4 G4GUN Shakedown
======================

::

    OKG4Test --g4gun --compute --save


OKG4 running does both:

* G4 simulation/propagation with Opticks style recording 
* OK propagation using gensteps collected from G4

The gensteps specify position and number of photons, 
so same photon count between them is a given.  

The history and material differences are all bugs to fix. 


Peculiarities
---------------

* scintillation dialed down by material override in cfg4
  but this will have same effect on G4 and OK 


Known sources of difference
----------------------------

* G4 is using stock (not DYB) scintillation but Opticks scintillation 
  not updated to handle stock gensteps   
  (this result is the MI)


::


    tokg4.py --src g4gun 

      A:seqhis_ana    1:dayabay 
                  41        0.354         492589       [2 ] CK AB
                   3        0.352         489714       [1 ] MI
             8cccc51        0.026          36768       [7 ] CK RE BT BT BT BT SA
                 451        0.025          34271       [3 ] CK RE AB
          cccbccccc1        0.019          26612       [10] CK BT BT BT BT BT BR BT BT BT
          cccccccc51        0.015          20339       [10] CK RE BT BT BT BT BT BT BT BT
            8cccc551        0.012          16259       [8 ] CK RE RE BT BT BT BT SA
                4551        0.010          14281       [4 ] CK RE RE AB
          ccbccccc51        0.008          11194       [10] CK RE BT BT BT BT BT BR BT BT
          ccccccc551        0.006           8303       [10] CK RE RE BT BT BT BT BT BT BT
           8cccc5551        0.005           7498       [9 ] CK RE RE RE BT BT BT BT SA
                 4c1        0.005           6533       [3 ] CK BT AB
               45551        0.004           6196       [5 ] CK RE RE RE AB
            4ccccc51        0.004           6007       [8 ] CK RE BT BT BT BT BT AB
          cbccccc551        0.004           5890       [10] CK RE RE BT BT BT BT BT BR BT
               4cc51        0.004           5550       [5 ] CK RE BT BT AB
          cccccc5551        0.004           4915       [10] CK RE RE RE BT BT BT BT BT BT
          cacccccc51        0.003           4779       [10] CK RE BT BT BT BT BT BT SR BT
           8cccccc51        0.003           4191       [9 ] CK RE BT BT BT BT BT BT SA
              4ccc51        0.003           4137       [6 ] CK RE BT BT BT AB
                         1392904         1.00 
       B:seqhis_ana   -1:dayabay 
                  4f        0.837        1166339       [2 ] GN AB
                 4cf        0.094         130309       [3 ] GN BT AB
          cccbcccccf        0.021          28980       [10] GN BT BT BT BT BT BR BT BT BT
                 4bf        0.007           9402       [3 ] GN BR AB
          bbbbbbbbbf        0.004           5184       [10] GN BR BR BR BR BR BR BR BR BR
                4ccf        0.003           4226       [4 ] GN BT BT AB
                 40f        0.002           3381       [3 ] GN ?0? AB
          ccbccccccf        0.002           2936       [10] GN BT BT BT BT BT BT BR BT BT
          ccbcccc0cf        0.002           2288       [10] GN BT ?0? BT BT BT BT BR BT BT
               4cccf        0.001           1879       [5 ] GN BT BT BT AB
          c00b00cccf        0.001           1669       [10] GN BT BT BT ?0? ?0? BR ?0? ?0? BT
             4cccccf        0.001           1585       [7 ] GN BT BT BT BT BT AB
            b00cc0cf        0.001           1498       [8 ] GN BT ?0? BT BT ?0? ?0? BR
          bcccbcccbf        0.001           1335       [10] GN BR BT BT BT BR BT BT BT BR
            8ccccccf        0.001           1260       [8 ] GN BT BT BT BT BT BT SA
          ccccbccccf        0.001           1116       [10] GN BT BT BT BT BR BT BT BT BT
          cbcccc0ccf        0.001            986       [10] GN BT BT ?0? BT BT BT BT BR BT
          bccccccccf        0.001            952       [10] GN BT BT BT BT BT BT BT BT BR
          cccbccbccf        0.001            914       [10] GN BT BT BR BT BT BR BT BT BT
              4ccccf        0.001            767       [6 ] GN BT BT BT BT AB
                         1392904         1.00 
       A:seqmat_ana    1:dayabay 
                   0        0.352         489714       [1 ] ?0?
                  11        0.233         323915       [2 ] Gd Gd
                  22        0.063          88210       [2 ] LS LS
             4432311        0.024          33745       [7 ] Gd Gd Ac LS Ac MO MO
                 111        0.021          29143       [3 ] Gd Gd Gd
                  44        0.020          28252       [2 ] MO MO
                  33        0.020          28178       [2 ] Ac Ac
                  ff        0.016          22849       [2 ] Ai Ai
          3343343231        0.015          21303       [10] Gd Ac LS Ac MO Ac Ac MO Ac Ac
            44323111        0.012          16966       [8 ] Gd Gd Gd Ac LS Ac MO MO
                1111        0.009          13196       [4 ] Gd Gd Gd Gd
          4433432311        0.006           8987       [10] Gd Gd Ac LS Ac MO Ac Ac MO MO
           443231111        0.006           8181       [9 ] Gd Gd Gd Gd Ac LS Ac MO MO
          4432311111        0.005           6275       [10] Gd Gd Gd Gd Gd Ac LS Ac MO MO
               11111        0.004           6007       [5 ] Gd Gd Gd Gd Gd
          fff3432311        0.003           4573       [10] Gd Gd Ac LS Ac MO Ac Ai Ai Ai
          3334323111        0.003           4443       [10] Gd Gd Gd Ac LS Ac MO Ac Ac Ac
          3343231111        0.003           3595       [10] Gd Gd Gd Gd Ac LS Ac MO Ac Ac
                   6        0.003           3563       [1 ] Iw
            aa332311        0.002           3450       [8 ] Gd Gd Ac LS Ac Ac ES ES
                         1392904         1.00 
       B:seqmat_ana   -1:dayabay 
                  11        0.837        1166374       [2 ] Gd Gd
                 111        0.103         143334       [3 ] Gd Gd Gd
          1111111111        0.046          63409       [10] Gd Gd Gd Gd Gd Gd Gd Gd Gd Gd
                1111        0.004           6236       [4 ] Gd Gd Gd Gd
               11111        0.003           3749       [5 ] Gd Gd Gd Gd Gd
            11111111        0.002           3447       [8 ] Gd Gd Gd Gd Gd Gd Gd Gd
             1111111        0.002           2367       [7 ] Gd Gd Gd Gd Gd Gd Gd
              111111        0.002           2107       [6 ] Gd Gd Gd Gd Gd Gd
           111111111        0.001           1881       [9 ] Gd Gd Gd Gd Gd Gd Gd Gd Gd
                         1392904         1.00 



