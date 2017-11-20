tboolean-trapezoid-no-stuck-tracks-but-still-drastic-difference
===================================================================


::

    tboolean-;tboolean-trapezoid --okg4
    tboolean-;tboolean-trapezoid-p


::

    simon:opticks blyth$ tboolean-;tboolean-trapezoid-p
    args: /Users/blyth/opticks/ana/tboolean.py --det tboolean-trapezoid --tag 1 --nosmry
    ok.smry 0 
    [2017-11-20 19:19:08,265] p81694 {/Users/blyth/opticks/ana/tboolean.py:17} INFO - tag 1 src torch det tboolean-trapezoid c2max 2.0 ipython False 
    [2017-11-20 19:19:08,265] p81694 {/Users/blyth/opticks/ana/ab.py:81} INFO - AB.load START smry 0 
    [2017-11-20 19:19:08,278] p81694 {/Users/blyth/opticks/ana/evt.py:405} WARNING -  z : -501.000 501.000 : tot 100000 over 15 0.000  under 0 0.000 : mi   -500.000 mx    597.379  
    [2017-11-20 19:19:08,342] p81694 {/Users/blyth/opticks/ana/evt.py:405} WARNING -  z : -501.000 501.000 : tot 100000 over 20 0.000  under 0 0.000 : mi   -500.000 mx    595.662  
    [2017-11-20 19:19:08,413] p81694 {/Users/blyth/opticks/ana/ab.py:108} INFO - AB.load DONE 
    [2017-11-20 19:19:08,417] p81694 {/Users/blyth/opticks/ana/ab.py:150} INFO - AB.init_point START
    [2017-11-20 19:19:08,419] p81694 {/Users/blyth/opticks/ana/ab.py:152} INFO - AB.init_point DONE
    AB(1,torch,tboolean-trapezoid)  None 0 
    A tboolean-trapezoid/torch/  1 :  20171120-1917 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-trapezoid/torch/1/fdom.npy () 
    B tboolean-trapezoid/torch/ -1 :  20171120-1917 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-trapezoid/torch/-1/fdom.npy (recstp) 
    Rock//perfectAbsorbSurface/Vacuum,Vacuum///GlassSchottF2
    /tmp/blyth/opticks/tboolean-trapezoid--
    .                seqhis_ana  1:tboolean-trapezoid   -1:tboolean-trapezoid        c2        ab        ba 
    .                             100000    100000      4589.45/8 = 573.68  (pval:0.000 prob:1.000)  
    0000              8cd     97643     90544           267.80        1.078 +- 0.003        0.927 +- 0.003  [3 ] TO BT SA
    0001            8cccd      1753      7329          3423.45        0.239 +- 0.006        4.181 +- 0.049  [5 ] TO BT BT BT SA
    0002          8cbbccd       171       717           335.72        0.238 +- 0.018        4.193 +- 0.157  [7 ] TO BT BT BR BR BT SA
    0003             8bcd       137       544           243.24        0.252 +- 0.022        3.971 +- 0.170  [4 ] TO BT BR SA
    0004         8cbbbccd        93       414           203.24        0.225 +- 0.023        4.452 +- 0.219  [8 ] TO BT BT BR BR BR BT SA
    0005             86cd        91       105             1.00        0.867 +- 0.091        1.154 +- 0.113  [4 ] TO BT SC SA
    0006           8cbccd        58       193            72.61        0.301 +- 0.039        3.328 +- 0.240  [6 ] TO BT BT BR BT SA
    0007        8cbbbbccd        12        64            35.58        0.188 +- 0.054        5.333 +- 0.667  [9 ] TO BT BT BR BR BR BR BT SA
    0008               4d        10        14             0.00        0.714 +- 0.226        1.400 +- 0.374  [2 ] TO AB
    0009       8cbbbbbccd         9        24             6.82        0.375 +- 0.125        2.667 +- 0.544  [10] TO BT BT BR BR BR BR BR BT SA
    0010              4cd         7        14             0.00        0.500 +- 0.189        2.000 +- 0.535  [3 ] TO BT AB
    0011               3d         6         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [2 ] TO MI
    0012             8c6d         4         8             0.00        0.500 +- 0.250        2.000 +- 0.707  [4 ] TO SC BT SA
    0013             8ccd         3         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [4 ] TO BT BT SA
    0014           86cccd         3         2             0.00        1.500 +- 0.866        0.667 +- 0.471  [6 ] TO BT BT BT SC SA
    0015           8cc6cd         0         1             0.00        0.000 +- 0.000        0.000 +- 0.000  [6 ] TO BT SC BT BT SA
    0016        86cbbbccd         0         1             0.00        0.000 +- 0.000        0.000 +- 0.000  [9 ] TO BT BT BR BR BR BT SC SA
    0017       bbbbbb6ccd         0         1             0.00        0.000 +- 0.000        0.000 +- 0.000  [10] TO BT BT SC BR BR BR BR BR BR
    0018           8ccc6d         0         1             0.00        0.000 +- 0.000        0.000 +- 0.000  [6 ] TO SC BT BT BT SA
    0019            4cccd         0         2             0.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] TO BT BT BT AB
    .                             100000    100000      4589.45/8 = 573.68  (pval:0.000 prob:1.000)  
    .                pflags_ana  1:tboolean-trapezoid   -1:tboolean-trapezoid        c2        ab        ba 
    .                             100000    100000       907.98/2 = 453.99  (pval:0.000 prob:1.000)  
    0000             1880     99399     97873            11.80        1.016 +- 0.003        0.985 +- 0.003  [3 ] TO|BT|SA
    0001             1c80       480      1956           894.33        0.245 +- 0.011        4.075 +- 0.092  [4 ] TO|BT|BR|SA
    0002             18a0        98       118             1.85        0.831 +- 0.084        1.204 +- 0.111  [4 ] TO|BT|SA|SC
    0003             1008        10        14             0.00        0.714 +- 0.226        1.400 +- 0.374  [2 ] TO|AB
    0004             1808         7        16             0.00        0.438 +- 0.165        2.286 +- 0.571  [3 ] TO|BT|AB
    0005             1004         6         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [2 ] TO|MI
    0006             1c20         0         1             0.00        0.000 +- 0.000        0.000 +- 0.000  [4 ] TO|BT|BR|SC
    0007             1c00         0        15             0.00        0.000 +- 0.000        0.000 +- 0.000  [3 ] TO|BT|BR
    0008             1024         0         6             0.00        0.000 +- 0.000        0.000 +- 0.000  [3 ] TO|SC|MI
    0009             1ca0         0         1             0.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] TO|BT|BR|SA|SC
    .                             100000    100000       907.98/2 = 453.99  (pval:0.000 prob:1.000)  
    .                seqmat_ana  1:tboolean-trapezoid   -1:tboolean-trapezoid        c2        ab        ba 
    .                             100000    100000      4544.54/7 = 649.22  (pval:0.000 prob:1.000)  
    0000              121     97643     90544           267.80        1.078 +- 0.003        0.927 +- 0.003  [3 ] Rk Vm Rk
    0001            12321      1753      7329          3423.45        0.239 +- 0.006        4.181 +- 0.049  [5 ] Rk Vm F2 Vm Rk
    0002             1221       231       649           198.55        0.356 +- 0.023        2.810 +- 0.110  [4 ] Rk Vm Vm Rk
    0003          1233321       171       717           335.72        0.238 +- 0.018        4.193 +- 0.157  [7 ] Rk Vm F2 F2 F2 Vm Rk
    0004         12333321        93       414           203.24        0.225 +- 0.023        4.452 +- 0.219  [8 ] Rk Vm F2 F2 F2 F2 Vm Rk
    0005           123321        58       194            73.40        0.299 +- 0.039        3.345 +- 0.240  [6 ] Rk Vm F2 F2 Vm Rk
    0006        123333321        12        64            35.58        0.188 +- 0.054        5.333 +- 0.667  [9 ] Rk Vm F2 F2 F2 F2 F2 Vm Rk
    0007               11        10        14             0.00        0.714 +- 0.226        1.400 +- 0.374  [2 ] Rk Rk
    0008       1233333321         9        24             6.82        0.375 +- 0.125        2.667 +- 0.544  [10] Rk Vm F2 F2 F2 F2 F2 F2 Vm Rk
    0009              221         7        14             0.00        0.500 +- 0.189        2.000 +- 0.535  [3 ] Rk Vm Vm
    0010                1         6         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [1 ] Rk
    0011             1211         4         8             0.00        0.500 +- 0.250        2.000 +- 0.707  [4 ] Rk Rk Vm Rk
    0012           122321         3         2             0.00        1.500 +- 0.866        0.667 +- 0.471  [6 ] Rk Vm F2 Vm Vm Rk
    0013        122333321         0         1             0.00        0.000 +- 0.000        0.000 +- 0.000  [9 ] Rk Vm F2 F2 F2 F2 Vm Vm Rk
    0014              111         0         6             0.00        0.000 +- 0.000        0.000 +- 0.000  [3 ] Rk Rk Rk
    0015            22321         0         2             0.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] Rk Vm F2 Vm Vm
    0016           123211         0         1             0.00        0.000 +- 0.000        0.000 +- 0.000  [6 ] Rk Rk Vm F2 Vm Rk
    0017           123221         0         1             0.00        0.000 +- 0.000        0.000 +- 0.000  [6 ] Rk Vm Vm F2 Vm Rk
    0018       3333333321         0         8             0.00        0.000 +- 0.000        0.000 +- 0.000  [10] Rk Vm F2 F2 F2 F2 F2 F2 F2 F2
    0019       2333333321         0         8             0.00        0.000 +- 0.000        0.000 +- 0.000  [10] Rk Vm F2 F2 F2 F2 F2 F2 F2 Vm
    .                             100000    100000      4544.54/7 = 649.22  (pval:0.000 prob:1.000)  
               /tmp/blyth/opticks/evt/tboolean-trapezoid/torch/1 d2c4fcd0904d65ff43c7213c27711113 0b3b71d4268a97985d35d2b796eedbdb  100000    -1.0000 INTEROP_MODE 
    {u'verbosity': u'0', u'resolution': u'40', u'poly': u'IM', u'ctrl': u'0'}
    [2017-11-20 19:19:08,425] p81694 {/Users/blyth/opticks/ana/tboolean.py:25} INFO - early exit as non-interactive
    simon:opticks blyth$ 
