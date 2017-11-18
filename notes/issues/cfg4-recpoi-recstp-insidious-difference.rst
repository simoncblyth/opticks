cfg4-recpoi-recstp-insidious-difference
=============================================


Separate run recpoi/recstp difference that doesnt appear with simultaneous recpoi+recstp ?
-----------------------------------------------------------------------------------------------



::


    [2017-11-18 17:14:14,101] p1353 {/Users/blyth/opticks/ana/ab.py:146} INFO - AB.init_point DONE
    AB(1,torch,tboolean-truncate)  None 0 
    A tboolean-truncate/torch/  1 :  20171118-1642 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-truncate/torch/1/fdom.npy () 
    B tboolean-truncate/torch/ -1 :  20171118-1642 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-truncate/torch/-1/fdom.npy (recstp) 
    Rock//perfectSpecularSurface/Vacuum
    /tmp/blyth/opticks/tboolean-truncate--
    .                seqhis_ana  1:tboolean-truncate   -1:tboolean-truncate        c2        ab        ba 
    .                             100000    100000         4.82/9 =  0.54  (pval:0.850 prob:0.150)  
    0000       aaaaaaaaad     99603     99633             0.00        1.000 +- 0.003        1.000 +- 0.003  [10] TO SR SR SR SR SR SR SR SR SR
    0001       aaa6aaaaad        49        42             0.54        1.167 +- 0.167        0.857 +- 0.132  [10] TO SR SR SR SR SR SC SR SR SR
    0002       aaaaa6aaad        45        42             0.10        1.071 +- 0.160        0.933 +- 0.144  [10] TO SR SR SR SC SR SR SR SR SR
    0003       6aaaaaaaad        41        49             0.71        0.837 +- 0.131        1.195 +- 0.171  [10] TO SR SR SR SR SR SR SR SR SC
    0004       aaaaaa6aad        40        30             1.43        1.333 +- 0.211        0.750 +- 0.137  [10] TO SR SR SC SR SR SR SR SR SR
    0005       a6aaaaaaad        39        31             0.91        1.258 +- 0.201        0.795 +- 0.143  [10] TO SR SR SR SR SR SR SR SC SR
    0006       aaaa6aaaad        38        36             0.05        1.056 +- 0.171        0.947 +- 0.158  [10] TO SR SR SR SR SC SR SR SR SR
    0007       aaaaaaaa6d        38        36             0.05        1.056 +- 0.171        0.947 +- 0.158  [10] TO SC SR SR SR SR SR SR SR SR
    0008       aa6aaaaaad        36        31             0.37        1.161 +- 0.194        0.861 +- 0.155  [10] TO SR SR SR SR SR SR SC SR SR
    0009       aaaaaaa6ad        35        42             0.64        0.833 +- 0.141        1.200 +- 0.185  [10] TO SR SC SR SR SR SR SR SR SR
    0010         4aaaaaad         9         4             0.00        2.250 +- 0.750        0.444 +- 0.222  [8 ] TO SR SR SR SR SR SR AB
    0011            4aaad         5         2             0.00        2.500 +- 1.118        0.400 +- 0.283  [5 ] TO SR SR SR AB
    0012          4aaaaad         5         4             0.00        1.250 +- 0.559        0.800 +- 0.400  [7 ] TO SR SR SR SR SR AB
    0013       4aaaaaaaad         4         4             0.00        1.000 +- 0.500        1.000 +- 0.500  [10] TO SR SR SR SR SR SR SR SR AB
    0014              4ad         4         6             0.00        0.667 +- 0.333        1.500 +- 0.612  [3 ] TO SR AB
    0015               4d         4         3             0.00        1.333 +- 0.667        0.750 +- 0.433  [2 ] TO AB
    0016        4aaaaaaad         2         2             0.00        1.000 +- 0.707        1.000 +- 0.707  [9 ] TO SR SR SR SR SR SR SR AB
    0017             4aad         2         2             0.00        1.000 +- 0.707        1.000 +- 0.707  [4 ] TO SR SR AB
    0018           4aaaad         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [6 ] TO SR SR SR SR AB
    .                             100000    100000         4.82/9 =  0.54  (pval:0.850 prob:0.150)  

    [2017-11-18 17:18:01,852] p1613 {/Users/blyth/opticks/ana/ab.py:141} INFO - AB.init_point DONE
    AB(1,torch,tboolean-truncate)  None 0 
    A tboolean-truncate/torch/  1 :  20171118-1717 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-truncate/torch/1/fdom.npy () 
    B tboolean-truncate/torch/ -1 :  20171118-1717 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-truncate/torch/-1/fdom.npy (recpoi) 
    Rock//perfectSpecularSurface/Vacuum
    /tmp/blyth/opticks/tboolean-truncate--
    .                seqhis_ana  1:tboolean-truncate   -1:tboolean-truncate        c2        ab        ba 
    .                             100000    100000         5.76/9 =  0.64  (pval:0.764 prob:0.236)  
    0000       aaaaaaaaad     99603     99629             0.00        1.000 +- 0.003        1.000 +- 0.003  [10] TO SR SR SR SR SR SR SR SR SR
    0001       aaa6aaaaad        49        32             3.57        1.531 +- 0.219        0.653 +- 0.115  [10] TO SR SR SR SR SR SC SR SR SR
    0002       aaaaa6aaad        45        45             0.00        1.000 +- 0.149        1.000 +- 0.149  [10] TO SR SR SR SC SR SR SR SR SR
    0003       6aaaaaaaad        41        35             0.47        1.171 +- 0.183        0.854 +- 0.144  [10] TO SR SR SR SR SR SR SR SR SC
    0004       aaaaaa6aad        40        31             1.14        1.290 +- 0.204        0.775 +- 0.139  [10] TO SR SR SC SR SR SR SR SR SR
    0005       a6aaaaaaad        39        37             0.05        1.054 +- 0.169        0.949 +- 0.156  [10] TO SR SR SR SR SR SR SR SC SR
    0006       aaaa6aaaad        38        37             0.01        1.027 +- 0.167        0.974 +- 0.160  [10] TO SR SR SR SR SC SR SR SR SR
    0007       aaaaaaaa6d        38        35             0.12        1.086 +- 0.176        0.921 +- 0.156  [10] TO SC SR SR SR SR SR SR SR SR
    0008       aa6aaaaaad        36        41             0.32        0.878 +- 0.146        1.139 +- 0.178  [10] TO SR SR SR SR SR SR SC SR SR
    0009       aaaaaaa6ad        35        37             0.06        0.946 +- 0.160        1.057 +- 0.174  [10] TO SR SC SR SR SR SR SR SR SR
    0010         4aaaaaad         9         4             0.00        2.250 +- 0.750        0.444 +- 0.222  [8 ] TO SR SR SR SR SR SR AB
    0011            4aaad         5         1             0.00        5.000 +- 2.236        0.200 +- 0.200  [5 ] TO SR SR SR AB
    0012          4aaaaad         5         5             0.00        1.000 +- 0.447        1.000 +- 0.447  [7 ] TO SR SR SR SR SR AB
    0013       4aaaaaaaad         4         4             0.00        1.000 +- 0.500        1.000 +- 0.500  [10] TO SR SR SR SR SR SR SR SR AB
    0014              4ad         4         3             0.00        1.333 +- 0.667        0.750 +- 0.433  [3 ] TO SR AB
    0015               4d         4         8             0.00        0.500 +- 0.250        2.000 +- 0.707  [2 ] TO AB
    0016        4aaaaaaad         2         4             0.00        0.500 +- 0.354        2.000 +- 1.000  [9 ] TO SR SR SR SR SR SR SR AB
    0017             4aad         2         6             0.00        0.333 +- 0.236        3.000 +- 1.225  [4 ] TO SR SR AB
    0018           4aaaad         1         5             0.00        0.200 +- 0.200        5.000 +- 2.236  [6 ] TO SR SR SR SR AB
    0019       aa6aaa6aad         0         1             0.00        0.000 +- 0.000        0.000 +- 0.000  [10] TO SR SR SC SR SR SR SC SR SR
    .                             100000    100000         5.76/9 =  0.64  (pval:0.764 prob:0.236)  




SeqTable
-----------

* add ordering "self/other/max" argument to seq.py:SeqTable.compare allowing to fix the seqhis ordering 
  for easier comparison of separate runs

::

    tboolean-truncate-ip

    In [8]: ab.seqhis_tab.cu
    Out[8]: 
    array([[733007751853,        99603,        99633],
           [732940642989,           49,           42],
           [458129844909,           41,           49],




