cfg4-recpoi-recstp-insidious-difference
=============================================


recpoi/recstp truncating track at a different juncture will thow off the random sequence
------------------------------------------------------------------------------------------

* so the problem doesnt really matter, nevertheless it would be good to 
  arrange some way of comparing algos in separate runs but with aligned random sequences

  * suppose need then to use recstp step_limit even in recpoi, 
    so are spinning G4 wheels not collecting points (beyond point limit and before step limit) 
    so as to get the random sequences of the separate algo aligned  


Revealed by single discrepant dumping::

    delta:cfg4 blyth$ tboolean-;tboolean-truncate --okg4 --dindex 161 --dbgzero



Separate run recpoi/recstp difference that doesnt appear with simultaneous recpoi+recstp ?
-----------------------------------------------------------------------------------------------

::


    tboolean-;TBOOLEAN_TAG="1,2" tboolean-truncate-ip


    In [2]: a = ab.a.seqhis
    A([733007751853, 733007751853, 733007751853, ..., 733007751853, 733007751853, 733007751853], dtype=uint64)

    In [3]: b = ab.b.seqhis
    A([733007751853, 733007751853, 733007751853, ..., 733007751853, 733007751853, 731934010029], dtype=uint64)

    In [4]: b.shape
    Out[4]: (100000,)

    In [14]: c = np.where( a != b )[0]

    In [15]: c.shape
    Out[15]: (735,)

    In [19]: c = np.array(c, dtype=np.uint32 )

    In [22]: np.save("/tmp/c.npy", c )



    delta:opticksnpy blyth$ NPY2Test
    2017-11-18 19:34:14.626 VERB  [5688735] [BFile::FormPath@291] BFile::FormPath path /tmp/c.npy sub (null) name (null) prepare 0
    2017-11-18 19:34:14.627 INFO  [5688735] [int>::load@514] NPY<T>::load /tmp/c.npy
    2017-11-18 19:34:14.627 INFO  [5688735] [int>::dump@1350] NPY::dump (735) 

    (  0)         161 
    (  1)         268 
    (  2)         312 
    (  3)         491 
    (  4)         673 
    (  5)         766 
    (  6)         767 
    (  7)         963 
    (  8)        1155 
    (  9)        1159 
    ( 10)        1284 
    ( 11)        1298 
    ( 12)        1316 
    ( 13)        1402 
    ( 14)        1755 

    In [23]: c[:10]
    Out[23]: array([ 161,  268,  312,  491,  673,  766,  767,  963, 1155, 1159], dtype=uint32)



    np.set_printoptions(formatter={'int':hex})

    In [47]: cc = np.zeros( (len(c), 2), dtype=np.uint64 )

    In [48]: cc[:,0] = a[c]

    In [49]: cc[:,1] = b[c]

    In [50]: cc[:100]
    Out[50]: 
    array([[0xaaaaa6aaadL, 0xaaaaaaaaadL],
           [0xaaaaaaaaadL, 0xaaaaaaaa6dL],
           [0x4adL, 0xaaaaaaaaadL],
           [0xaaaaa6aaadL, 0xaaaaaaaaadL],
           [0xaaaaaaaaadL, 0xaaaaaaa6adL],
           [0xaaaaaaaa6dL, 0xaaaaaaaaadL],
           [0xaaaaaaa6adL, 0xaaaaaaaaadL],
           [0xaaaaaaaaadL, 0xaaaaaa6aadL],
           [0xaaaa6aaaadL, 0xaaaaaaaaadL],
           [0xaaaaaa6aadL, 0xaaaaaaaaadL],
           [0xaaaa6aaaadL, 0xaaaaaaaaadL],
           [0xaaaaaaaaadL, 0xaaaaaa6aadL],
           [0xaaaaaaaaadL, 0x6aaaaaaaadL],
           [0x4dL, 0xaaaaaaaaadL],
           [0xaaa6aaaaadL, 0xaaaaaaaaadL],
           [0xaa6aaaaaadL, 0xaaaaaaaaadL],
           [0xaaaaa6aaadL, 0xaaaaaaaaadL],


    tboolean-;tboolean-truncate --okg4 --dindex $TMP/c.npy 

    tboolean-;tboolean-truncate --okg4 --dindex 161 --dbgzero





    delta:opticks blyth$ TBOOLEAN_TAG="1,2" tboolean-truncate-p
    args: /Users/blyth/opticks/ana/tboolean.py --det tboolean-truncate --tag 1,2 --nosmry
    ok.smry 0 
    [2017-11-18 19:18:28,257] p4552 {/Users/blyth/opticks/ana/tboolean.py:17} INFO - tag None src torch det tboolean-truncate c2max 2.0 ipython False 
    [2017-11-18 19:18:28,257] p4552 {/Users/blyth/opticks/ana/ab.py:81} INFO - AB.load START smry 0 
    [2017-11-18 19:18:28,412] p4552 {/Users/blyth/opticks/ana/ab.py:108} INFO - AB.load DONE 
    [2017-11-18 19:18:28,416] p4552 {/Users/blyth/opticks/ana/ab.py:150} INFO - AB.init_point START
    [2017-11-18 19:18:28,418] p4552 {/Users/blyth/opticks/ana/ab.py:152} INFO - AB.init_point DONE
    AB(1,2,torch,tboolean-truncate)  None 0 
    A tboolean-truncate/torch/ -1 :  20171118-1905 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-truncate/torch/-1/fdom.npy (recstp) 
    B tboolean-truncate/torch/ -2 :  20171118-1902 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-truncate/torch/-2/fdom.npy (recpoi) 
    Rock//perfectSpecularSurface/Vacuum
    /tmp/blyth/opticks/tboolean-truncate--
    .                seqhis_ana  -1:tboolean-truncate   -2:tboolean-truncate        c2        ab        ba 
    .                             100000    100000         6.07/9 =  0.67  (pval:0.733 prob:0.267)  
    0000       aaaaaaaaad     99633     99629             0.00        1.000 +- 0.003        1.000 +- 0.003  [10] TO SR SR SR SR SR SR SR SR SR
    0001       6aaaaaaaad        49        35             2.33        1.400 +- 0.200        0.714 +- 0.121  [10] TO SR SR SR SR SR SR SR SR SC
    0002       aaa6aaaaad        42        32             1.35        1.312 +- 0.203        0.762 +- 0.135  [10] TO SR SR SR SR SR SC SR SR SR
    0003       aaaaaaa6ad        42        37             0.32        1.135 +- 0.175        0.881 +- 0.145  [10] TO SR SC SR SR SR SR SR SR SR
    0004       aaaaa6aaad        42        45             0.10        0.933 +- 0.144        1.071 +- 0.160  [10] TO SR SR SR SC SR SR SR SR SR
    0005       aaaa6aaaad        36        37             0.01        0.973 +- 0.162        1.028 +- 0.169  [10] TO SR SR SR SR SC SR SR SR SR
    0006       aaaaaaaa6d        36        35             0.01        1.029 +- 0.171        0.972 +- 0.164  [10] TO SC SR SR SR SR SR SR SR SR
    0007       aa6aaaaaad        31        41             1.39        0.756 +- 0.136        1.323 +- 0.207  [10] TO SR SR SR SR SR SR SC SR SR
    0008       a6aaaaaaad        31        37             0.53        0.838 +- 0.150        1.194 +- 0.196  [10] TO SR SR SR SR SR SR SR SC SR
    0009       aaaaaa6aad        30        31             0.02        0.968 +- 0.177        1.033 +- 0.186  [10] TO SR SR SC SR SR SR SR SR SR
    0010              4ad         6         3             0.00        2.000 +- 0.816        0.500 +- 0.289  [3 ] TO SR AB
    0011       4aaaaaaaad         4         4             0.00        1.000 +- 0.500        1.000 +- 0.500  [10] TO SR SR SR SR SR SR SR SR AB
    0012         4aaaaaad         4         4             0.00        1.000 +- 0.500        1.000 +- 0.500  [8 ] TO SR SR SR SR SR SR AB
    0013          4aaaaad         4         5             0.00        0.800 +- 0.400        1.250 +- 0.559  [7 ] TO SR SR SR SR SR AB
    0014               4d         3         8             0.00        0.375 +- 0.217        2.667 +- 0.943  [2 ] TO AB
    0015            4aaad         2         1             0.00        2.000 +- 1.414        0.500 +- 0.500  [5 ] TO SR SR SR AB
    0016        4aaaaaaad         2         4             0.00        0.500 +- 0.354        2.000 +- 1.000  [9 ] TO SR SR SR SR SR SR SR AB
    0017             4aad         2         6             0.00        0.333 +- 0.236        3.000 +- 1.225  [4 ] TO SR SR AB
    0018           4aaaad         1         5             0.00        0.200 +- 0.200        5.000 +- 2.236  [6 ] TO SR SR SR SR AB
    0019       aa6aaa6aad         0         1             0.00        0.000 +- 0.000        0.000 +- 0.000  [10] TO SR SR SC SR SR SR SC SR SR
    .                             100000    100000         6.07/9 =  0.67  (pval:0.733 prob:0.267)  




::


    tboolean-;TBOOLEAN_TAG=1 tboolean-truncate --okg4     # recstp
    tboolean-;TBOOLEAN_TAG=1 tboolean-truncate-p


    [2017-11-18 19:05:37,173] p4232 {/Users/blyth/opticks/ana/ab.py:141} INFO - AB.init_point DONE
    AB(1,torch,tboolean-truncate)  None 0 
    A tboolean-truncate/torch/  1 :  20171118-1905 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-truncate/torch/1/fdom.npy () 
    B tboolean-truncate/torch/ -1 :  20171118-1905 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-truncate/torch/-1/fdom.npy (recstp) 
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


    tboolean-;TBOOLEAN_TAG=2 tboolean-truncate --okg4 --recpoi
    tboolean-;TBOOLEAN_TAG=2 tboolean-truncate-p

    args: /Users/blyth/opticks/ana/tboolean.py --det tboolean-truncate --tag 2 --nosmry
    ok.smry 0 
    [2017-11-18 19:02:24,149] p3975 {/Users/blyth/opticks/ana/tboolean.py:17} INFO - tag 2 src torch det tboolean-truncate c2max 2.0 ipython False 
    [2017-11-18 19:02:24,150] p3975 {/Users/blyth/opticks/ana/ab.py:81} INFO - AB.load START smry 0 
    [2017-11-18 19:02:24,296] p3975 {/Users/blyth/opticks/ana/ab.py:97} INFO - AB.load DONE 
    [2017-11-18 19:02:24,299] p3975 {/Users/blyth/opticks/ana/ab.py:139} INFO - AB.init_point START
    [2017-11-18 19:02:24,302] p3975 {/Users/blyth/opticks/ana/ab.py:141} INFO - AB.init_point DONE
    AB(2,torch,tboolean-truncate)  None 0 
    A tboolean-truncate/torch/  2 :  20171118-1902 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-truncate/torch/2/fdom.npy () 
    B tboolean-truncate/torch/ -2 :  20171118-1902 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-truncate/torch/-2/fdom.npy (recpoi) 
    Rock//perfectSpecularSurface/Vacuum
    /tmp/blyth/opticks/tboolean-truncate--
    .                seqhis_ana  2:tboolean-truncate   -2:tboolean-truncate        c2        ab        ba 
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




