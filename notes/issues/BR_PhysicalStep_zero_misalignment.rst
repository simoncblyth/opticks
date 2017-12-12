BR_PhysicalStep_zero_misalignment
==================================


Strategy
----------

Understand what G4 condition yields the zero-step, detect it 
on Opticks side and burn the requisite number(4) of RNG 
to stay in alignment. 

This strategy turned out to be impossible, instead did an 
rng rewind on G4 side for the zero steps.


AB/SC Position/Time Differs
------------------------------

* :doc:`AB_SC_Position_Time_mismatch`


With backtracking of rng sequence by CRandomEngine to keep zero-steps aligned
--------------------------------------------------------------------------------

* down to 33/100000 misaligned, all going awry after a scatter 


::

    tboolean-;tboolean-box-ip

    In [2]: len(ab.maligned)
    Out[2]: 33

    In [3]: ab.dumpline(ab.maligned)
          0    595 :                                           TO SC SA                                  TO SC BT BR BT SA 
          1   1230 :                                        TO BR SC SA                            TO BR SC BT BR BR BT SA 
          2   2413 :                                     TO BT BT SC SA                         TO BT BT SC BT BR BR BT SA 
          3   4608 :                      TO BT SC BR BR BR BR BR BT SA                                     TO BT SC BT SA 
          4   5729 :                               TO BT BT SC BT BT SA                                     TO BT BT SC SA 
          5   9041 :                                     TO BT SC BT SA                               TO BT SC BR BR BT SA 
          6  13997 :                                     TO BT BT SC SA                               TO BT BT SC BT BT SA 
          7  14510 :                                           TO SC SA                                  TO SC BT BR BT SA 
          8  14747 :                                     TO BT SC BT SA                      TO BT SC BR BR BR BR BR BR BR 
          9  17968 :                                  TO BT SC BR BT SA                                     TO BT SC BT SA 
         10  26635 :                                     TO BT SC BT SA                      TO BT SC BR BR BR BR BR BR BR 
         11  30272 :                                  TO BT BR SC BT SA                      TO BT BR SC BR BR BR BR BR BR 
         12  36621 :                                     TO BT SC BT SA                      TO BT SC BR BR BR BR BR BR BR 
         13  38253 :                                  TO SC BT BR BT SA                                           TO SC SA 
         14  44026 :                                     TO BT BT SC SA                               TO BT BT SC BT BT SA 
         15  49786 :                                     TO BT BT SC SA                            TO BT BT SC BT BR BT SA 
         16  53964 :                                     TO BT SC BT SA                      TO BT SC BR BR BR BR BR BR BR 
         17  58189 :                                           TO SC SA                                  TO SC BT BR BT SA 
         18  61921 :                      TO BT SC BR BR BR BR BR BR BR                                     TO BT SC BT SA 
         19  64663 :                               TO SC BT BR BR BT SA                                     TO SC BT BT SA 
         20  65850 :                                     TO BT BT SC SA                            TO BT BT SC BT BR BT SA 
         21  69653 :                      TO BT SC BR BR BR BR BR BR BR                                  TO BT SC BR BT SA 
         22  76467 :                      TO BT BR SC BR BR BR BR BT SA                                  TO BT BR SC BT SA 
         23  77962 :                               TO BT BR SC BR BT SA                            TO BT BR SC BR BR BT SA 
         24  84260 :                               TO BT BT SC BT BT SA                                     TO BT BT SC SA 
         25  86722 :                      TO BT SC BR BR BR BR BR BR BR                                     TO BT SC BT SA 
         26  90322 :                                     TO BT BT SC SA                                  TO BT BT SC BR SA 
         27  91760 :                      TO BT SC BR BR BR BR BR BR BR                                     TO BT SC BT SA 
         28  93259 :                                  TO BT SC BR BT SA                                     TO BT SC BT SA 
         29  94773 :                      TO BT SC BR BR BR BR BR BR BR                                     TO BT SC BT SA 
         30  94891 :                                     TO BT SC BT SA                      TO BT SC BR BR BR BR BR BR BR 
         31  95967 :                            TO BT BT SC BT BR BT SA                                     TO BT BT SC SA 
         32  97887 :                                     TO SC BT BT SA                                  TO SC BT BR BT SA 

    In [4]: 33./100000
    Out[4]: 0.00033




::

    tboolean-;tboolean-box --okg4 --align -D

    2017-12-11 20:57:07.073 INFO  [1770376] [OpticksAna::run@66] OpticksAna::run anakey tboolean enabled Y
    args: /Users/blyth/opticks/ana/tboolean.py --tag 1 --tagoffset 0 --det tboolean-box --src torch
    [2017-12-11 20:57:07,411] p84253 {/Users/blyth/opticks/ana/tboolean.py:58} INFO - tag 1 src torch det tboolean-box c2max 2.0 ipython False 
    [2017-12-11 20:57:07,411] p84253 {/Users/blyth/opticks/ana/ab.py:92} INFO - ab START
    AB(1,torch,tboolean-box)  None 0 
    A tboolean-box/torch/  1 :  20171211-2057 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-box/torch/1/fdom.npy () 
    B tboolean-box/torch/ -1 :  20171211-2057 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-box/torch/-1/fdom.npy (recstp) 
    Rock//perfectAbsorbSurface/Vacuum,Vacuum///GlassSchottF2
    /tmp/blyth/opticks/tboolean-box--
    .                seqhis_ana  1:tboolean-box   -1:tboolean-box        c2        ab        ba 
    .                             100000    100000         0.24/7 =  0.03  (pval:1.000 prob:0.000)  
    0000             8ccd     87777     87777             0.00        1.000 +- 0.003        1.000 +- 0.003  [4 ] TO BT BT SA
    0001              8bd      6312      6312             0.00        1.000 +- 0.013        1.000 +- 0.013  [3 ] TO BR SA
    0002            8cbcd      5420      5420             0.00        1.000 +- 0.014        1.000 +- 0.014  [5 ] TO BT BR BT SA
    0003           8cbbcd       349       349             0.00        1.000 +- 0.054        1.000 +- 0.054  [6 ] TO BT BR BR BT SA
    0004              86d        31        29             0.07        1.069 +- 0.192        0.935 +- 0.174  [3 ] TO SC SA
    0005            86ccd        27        24             0.18        1.125 +- 0.217        0.889 +- 0.181  [5 ] TO BT BT SC SA
    0006          8cbbbcd        26        26             0.00        1.000 +- 0.196        1.000 +- 0.196  [7 ] TO BT BR BR BR BT SA
    0007              4cd        16        16             0.00        1.000 +- 0.250        1.000 +- 0.250  [3 ] TO BT AB
    0008       bbbbbbb6cd         9         9             0.00        1.000 +- 0.333        1.000 +- 0.333  [10] TO BT SC BR BR BR BR BR BR BR
    0009            8c6cd         6         7             0.00        0.857 +- 0.350        1.167 +- 0.441  [5 ] TO BT SC BT SA
    0010         8cbc6ccd         3         4             0.00        0.750 +- 0.433        1.333 +- 0.667  [8 ] TO BT BT SC BT BR BT SA
    0011             4ccd         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [4 ] TO BT BT AB
    0012          8cc6ccd         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [7 ] TO BT BT SC BT BT SA
    0013               4d         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [2 ] TO AB
    0014           86cbcd         2         2             0.00        1.000 +- 0.707        1.000 +- 0.707  [6 ] TO BT BR BT SC SA
    0015           8cb6cd         2         1             0.00        2.000 +- 1.414        0.500 +- 0.500  [6 ] TO BT SC BR BT SA
    0016       8cbbbbb6cd         1         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [10] TO BT SC BR BR BR BR BR BT SA
    0017           8c6bcd         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [6 ] TO BT BR SC BT SA
    0018            8cc6d         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [5 ] TO SC BT BT SA
    0019          8cb6bcd         1         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [7 ] TO BT BR SC BR BT SA
    .                             100000    100000         0.24/7 =  0.03  (pval:1.000 prob:0.000)  
    .                pflags_ana  1:tboolean-box   -1:tboolean-box        c2        ab        ba 
    .                             100000    100000         0.62/6 =  0.10  (pval:0.996 prob:0.004)  
    0000             1880     87777     87777             0.00        1.000 +- 0.003        1.000 +- 0.003  [3 ] TO|BT|SA
    0001             1480      6312      6312             0.00        1.000 +- 0.013        1.000 +- 0.013  [3 ] TO|BR|SA
    0002             1c80      5795      5795             0.00        1.000 +- 0.013        1.000 +- 0.013  [4 ] TO|BT|BR|SA
    0003             18a0        37        35             0.06        1.057 +- 0.174        0.946 +- 0.160  [4 ] TO|BT|SA|SC
    0004             10a0        31        29             0.07        1.069 +- 0.192        0.935 +- 0.174  [3 ] TO|SA|SC
    0005             1808        19        19             0.00        1.000 +- 0.229        1.000 +- 0.229  [3 ] TO|BT|AB
    0006             1ca0        14        18             0.50        0.778 +- 0.208        1.286 +- 0.303  [5 ] TO|BT|BR|SA|SC
    0007             1c20         9        10             0.00        0.900 +- 0.300        1.111 +- 0.351  [4 ] TO|BT|BR|SC
    0008             1008         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [2 ] TO|AB
    0009             1c08         2         2             0.00        1.000 +- 0.707        1.000 +- 0.707  [4 ] TO|BT|BR|AB
    0010             14a0         1         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [4 ] TO|BR|SA|SC
    .                             100000    100000         0.62/6 =  0.10  (pval:0.996 prob:0.004)  
    .                seqmat_ana  1:tboolean-box   -1:tboolean-box        c2        ab        ba 
    .                             100000    100000         0.18/6 =  0.03  (pval:1.000 prob:0.000)  
    0000             1232     87777     87777             0.00        1.000 +- 0.003        1.000 +- 0.003  [4 ] Vm F2 Vm Rk
    0001              122      6343      6341             0.00        1.000 +- 0.013        1.000 +- 0.013  [3 ] Vm Vm Rk
    0002            12332      5426      5427             0.00        1.000 +- 0.014        1.000 +- 0.014  [5 ] Vm F2 F2 Vm Rk
    0003           123332       352       351             0.00        1.003 +- 0.053        0.997 +- 0.053  [6 ] Vm F2 F2 F2 Vm Rk
    0004          1233332        27        27             0.00        1.000 +- 0.192        1.000 +- 0.192  [7 ] Vm F2 F2 F2 F2 Vm Rk
    0005            12232        27        24             0.18        1.125 +- 0.217        0.889 +- 0.181  [5 ] Vm F2 Vm Vm Rk
    0006              332        16        16             0.00        1.000 +- 0.250        1.000 +- 0.250  [3 ] Vm F2 F2
    0007       3333333332         9        10             0.00        0.900 +- 0.300        1.111 +- 0.351  [10] Vm F2 F2 F2 F2 F2 F2 F2 F2 F2
    0008          1232232         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [7 ] Vm F2 Vm Vm F2 Vm Rk
    0009               22         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [2 ] Vm Vm
    0010             2232         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [4 ] Vm F2 Vm Vm
    0011         12332232         3         4             0.00        0.750 +- 0.433        1.333 +- 0.667  [8 ] Vm F2 Vm Vm F2 F2 Vm Rk
    0012       1233333332         2         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [10] Vm F2 F2 F2 F2 F2 F2 F2 Vm Rk
    0013           122332         2         2             0.00        1.000 +- 0.707        1.000 +- 0.707  [6 ] Vm F2 F2 Vm Vm Rk
    0014           123222         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [6 ] Vm Vm Vm F2 Vm Rk
    0015            12322         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [5 ] Vm Vm F2 Vm Rk
    0016             3332         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [4 ] Vm F2 F2 F2
    0017          1233322         1         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [7 ] Vm Vm F2 F2 F2 Vm Rk
    0018           123322         1         4             0.00        0.250 +- 0.250        4.000 +- 2.000  [6 ] Vm Vm F2 F2 Vm Rk
    0019            33332         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [5 ] Vm F2 F2 F2 F2
    .                             100000    100000         0.18/6 =  0.03  (pval:1.000 prob:0.000)  
    ab.a.metadata                  /tmp/blyth/opticks/evt/tboolean-box/torch/1 e3b4ee8211178b213c6da01bfd4f9be2 3a624e7d0fc57237b2ecd23c0c9cdd25  100000    -1.0000 INTEROP_MODE 
    ab.a.metadata.csgmeta0 {u'containerscale': u'3', u'container': u'1', u'ctrl': u'0', u'verbosity': u'0', u'poly': u'IM', u'emitconfig': u'photons:100000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x1,umin:0.45,umax:0.55,vmin:0.45,vmax:0.55', u'resolution': u'20', u'emit': -1}
    rpost_dv maxdvmax:899.990478225 maxdv:[0.013763847773677895, 0.0, 0.0, 0.0, 881.2716452528459, 899.9904782250435, 0.0, 0.055055391094704476, 299.9968260750145, 420.14145329142127, 0.49549851985227633, 331.39216284676655, 0.49549851985227633, 553.6370128482924, 781.9517197180089, 0.04129154332102303, 0.04129154332102303] 
     0000            :                    TO BT BT SA :   87777    87777  :     87777 1404432/     12: 0.000  mx/mn/av 0.01376/     0/1.176e-07  eps:0.0002    
     0001            :                       TO BR SA :    6312     6312  :      6312   75744/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0002            :                 TO BT BR BT SA :    5420     5420  :      5420  108400/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0003            :              TO BT BR BR BT SA :     349      349  :       349    8376/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0004            :                       TO SC SA :      31       29  :        28     336/    133: 0.396  mx/mn/av  881.3/     0/ 64.55  eps:0.0002    
     0005            :                 TO BT BT SC SA :      27       24  :        21     420/     98: 0.233  mx/mn/av    900/     0/ 28.19  eps:0.0002    
     0006            :           TO BT BR BR BR BT SA :      26       26  :        26     728/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0007            :                       TO BT AB :      16       16  :        16     192/     21: 0.109  mx/mn/av 0.05506/     0/0.003815  eps:0.0002    
     0008            :  TO BT SC BR BR BR BR BR BR BR :       9        9  :         4     160/    115: 0.719  mx/mn/av    300/     0/ 61.75  eps:0.0002    
     0010            :        TO BT BT SC BT BR BT SA :       3        4  :         2      64/     27: 0.422  mx/mn/av  420.1/     0/ 28.15  eps:0.0002    
     0011            :                    TO BT BT AB :       3        3  :         3      48/      6: 0.125  mx/mn/av 0.4955/     0/0.02962  eps:0.0002    
     0012            :           TO BT BT SC BT BT SA :       3        3  :         1      28/     10: 0.357  mx/mn/av  331.4/     0/ 29.67  eps:0.0002    
     0013            :                          TO AB :       3        3  :         3      24/      6: 0.250  mx/mn/av 0.4955/     0/0.05985  eps:0.0002    
     0014            :              TO BT BR BT SC SA :       2        2  :         2      48/     10: 0.208  mx/mn/av  553.6/     0/ 38.49  eps:0.0002    
     0022            :              TO BR SC BT BT SA :       1        1  :         1      24/     11: 0.458  mx/mn/av    782/     0/ 52.77  eps:0.0002    
     0025            :                    TO BT BR AB :       1        1  :         1      16/      1: 0.062  mx/mn/av 0.04129/     0/0.002581  eps:0.0002    
     0026            :                 TO BT BR BR AB :       1        1  :         1      20/      2: 0.100  mx/mn/av 0.04129/     0/0.002095  eps:0.0002    
    rpol_dv maxdvmax:1.98425197601 maxdv:[0.0, 0.0, 0.0, 0.0, 1.9842519760131836, 1.9685039520263672, 0.0, 0.0, 1.8346457481384277, 1.9133858680725098, 0.0, 0.20472443103790283, 0.0, 1.9527559280395508, 1.1338582038879395, 0.0, 0.0] 
     0000            :                    TO BT BT SA :   87777    87777  :     87777 1053324/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0001            :                       TO BR SA :    6312     6312  :      6312   56808/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0002            :                 TO BT BR BT SA :    5420     5420  :      5420   81300/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0003            :              TO BT BR BR BT SA :     349      349  :       349    6282/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0004            :                       TO SC SA :      31       29  :        28     252/    168: 0.667  mx/mn/av  1.984/     0/ 0.375  eps:0.0002    
     0005            :                 TO BT BT SC SA :      27       24  :        21     315/    124: 0.394  mx/mn/av  1.969/     0/0.2309  eps:0.0002    
     0006            :           TO BT BR BR BR BT SA :      26       26  :        26     546/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0007            :                       TO BT AB :      16       16  :        16     144/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0008            :  TO BT SC BR BR BR BR BR BR BR :       9        9  :         4     120/     96: 0.800  mx/mn/av  1.835/     0/0.4668  eps:0.0002    
     0010            :        TO BT BT SC BT BR BT SA :       3        4  :         2      48/     30: 0.625  mx/mn/av  1.913/     0/0.2126  eps:0.0002    
     0011            :                    TO BT BT AB :       3        3  :         3      36/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0012            :           TO BT BT SC BT BT SA :       3        3  :         1      21/     12: 0.571  mx/mn/av 0.2047/     0/0.05024  eps:0.0002    
     0013            :                          TO AB :       3        3  :         3      18/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0014            :              TO BT BR BT SC SA :       2        2  :         2      36/     12: 0.333  mx/mn/av  1.953/     0/0.2454  eps:0.0002    
     0022            :              TO BR SC BT BT SA :       1        1  :         1      18/     12: 0.667  mx/mn/av  1.134/     0/0.3911  eps:0.0002    
     0025            :                    TO BT BR AB :       1        1  :         1      12/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0026            :                 TO BT BR BR AB :       1        1  :         1      15/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
    /Users/blyth/opticks/ana/dv.py:58: RuntimeWarning: invalid value encountered in greater
      discrep = dv[dv>eps]
    ox_dv maxdvmax:900.0 maxdv:[5.960464477539063e-08, 1.401298464324817e-45, 5.960464477539063e-08, 5.960464477539063e-08, 881.2715454101562, 900.0, 5.960464477539063e-08, 0.050258636474609375, 200.0, 420.14764404296875, 0.49346923828125, 331.3966979980469, nan, 553.6422119140625, 781.9554443359375, 0.0469207763671875, 0.04579925537109375] 
     0000            :                    TO BT BT SA :   87777    87777  :     87777 1404432/      0: 0.000  mx/mn/av 5.96e-08/     0/3.725e-09  eps:0.0002    
     0001            :                       TO BR SA :    6312     6312  :      6312  100992/      0: 0.000  mx/mn/av 1.401e-45/     0/8.758e-47  eps:0.0002    
     0002            :                 TO BT BR BT SA :    5420     5420  :      5420   86720/      0: 0.000  mx/mn/av 5.96e-08/     0/3.725e-09  eps:0.0002    
     0003            :              TO BT BR BR BT SA :     349      349  :       349    5584/      0: 0.000  mx/mn/av 5.96e-08/     0/3.725e-09  eps:0.0002    
     0004            :                       TO SC SA :      31       29  :        28     448/    266: 0.594  mx/mn/av  881.3/     0/ 48.62  eps:0.0002    
     0005            :                 TO BT BT SC SA :      27       24  :        21     336/    197: 0.586  mx/mn/av    900/     0/ 35.45  eps:0.0002    
     0006            :           TO BT BR BR BR BT SA :      26       26  :        26     416/      0: 0.000  mx/mn/av 5.96e-08/     0/3.725e-09  eps:0.0002    
     0007            :                       TO BT AB :      16       16  :        16     256/     32: 0.125  mx/mn/av 0.05026/     0/0.003003  eps:0.0002    
     0008            :  TO BT SC BR BR BR BR BR BR BR :       9        9  :         4      64/     40: 0.625  mx/mn/av    200/     0/ 16.18  eps:0.0002    
     0010            :        TO BT BT SC BT BR BT SA :       3        4  :         2      32/     18: 0.562  mx/mn/av  420.1/     0/    31  eps:0.0002    
     0011            :                    TO BT BT AB :       3        3  :         3      48/      6: 0.125  mx/mn/av 0.4935/     0/0.02979  eps:0.0002    
     0012            :           TO BT BT SC BT BT SA :       3        3  :         1      16/     10: 0.625  mx/mn/av  331.4/     0/ 43.43  eps:0.0002    
     0013            :                          TO AB :       3        3  :         3      48/      6: 0.125  mx/mn/av    nan/   nan/   nan  eps:0.0002    
     0014            :              TO BT BR BT SC SA :       2        2  :         2      32/     20: 0.625  mx/mn/av  553.6/     0/ 58.01  eps:0.0002    
     0022            :              TO BR SC BT BT SA :       1        1  :         1      16/     10: 0.625  mx/mn/av    782/     0/ 56.74  eps:0.0002    
     0025            :                    TO BT BR AB :       1        1  :         1      16/      2: 0.125  mx/mn/av 0.04692/     0/0.00295  eps:0.0002    
     0026            :                 TO BT BR BR AB :       1        1  :         1      16/      2: 0.125  mx/mn/av 0.0458/     0/0.00288  eps:0.0002    
    c2p : {'seqmat_ana': 0.029716760282539143, 'pflags_ana': 0.1037037037037037, 'seqhis_ana': 0.034733893557422971} c2pmax: 0.103703703704  CUT ok.c2max 2.0  RC:0 
    rmxs_ : {'rpol_dv': 1.9842519760131836, 'rpost_dv': 899.9904782250435} rmxs_max_: 899.990478225  CUT ok.rdvmax 0.1  RC:88 
    pmxs_ : {'ox_dv': 900.0} pmxs_max_: 900.0  CUT ok.pdvmax 0.001  RC:99 
    [2017-12-11 20:57:08,234] p84253 {/Users/blyth/opticks/ana/tboolean.py:73} INFO - early exit as non-interactive




Perhaps backtrack the sequence on G4 side ?
-----------------------------------------------

Hmm detecting on Opticks side seems impossible, requires seeing into future ?

Conversely its straightforward to detect on G4 side, and then 
backtrack the sequence. Actually this kinda makes sense the problem 
of burning steps is on G4 side, so its a lot easier to deal with there.

* implementing CRandomEngine::jump to backtrack on the sequence
* hmm could keep track of rng consumption per step, so know what to jump in
  order to backtrack if the step turns out to be zero-step  


::

    simon:cfg4 blyth$ thrust_curand_printf 1230
    thrust_curand_printf
     i0 1230 i1 1231 q0 0 q1 16
     id:1230 thread_offset:0 seq0:0 seq1:16 
     0.001117  0.502647  0.601504  0.938713 
     0.753801  0.999847  0.438020  0.714032 
     0.330404  0.570742  0.375909  0.784978 
     0.892654  0.441063  0.773742  0.556839 
    simon:cfg4 blyth$ 


Turnaround dump
----------------

::

    2017-12-11 16:05:40.024 ERROR [1689439] [CRandomEngine::pretrack@256] CRandomEngine::pretrack record_id:  ctx.record_id 0 index 1230 mask.size 1
    //                                    opticks.ana.cfg4lldb.CRandomEngine_cc_210.[0] : 0.001117024919949472 1  
    //                                    opticks.ana.cfg4lldb.CRandomEngine_cc_210.[1] : 0.50264734029769897 2  
    //                                    opticks.ana.cfg4lldb.CRandomEngine_cc_210.[2] : 0.60150414705276489 3  
    G4SteppingManager2_cc_181_ : Dumping lengths collected by _181 after PostStep process loop  
    //                                                  .fCurrentProcess.theProcessName :  OpBoundary  
    //                                                                   .physIntLength :  1.79769e+308  
    //                                                  .fCurrentProcess.theProcessName :  OpRayleigh  
    //                                                                   .physIntLength :  687866  
    //                                                  .fCurrentProcess.theProcessName :  OpAbsorption  
    //                                                                   .physIntLength :  5.08322e+06  
    //                                                  .fCurrentProcess.theProcessName :  Transportation  
    //                                                                   .physIntLength :  1.79769e+308  

    //                                opticks.ana.cfg4lldb.G4Transportation_cc_517_.[0] : AlongStepGetPhysicalInteractionLength Exit  
    //                                                                             this : G4Transportation_cc_517 
    //                                                                   /startPosition :  ( -37.879   11.823 -449.900)  
    //                                                                /startMomentumDir :  (  -0.000   -0.000    1.000)  
    //                                                                       /newSafety :  0.100006  
    //                                                            .fGeometryLimitedStep : True 
    //                                                              .fFirstStepInVolume : True 
    //                                                               .fLastStepInVolume : False 
    //                                                                .fMomentumChanged : False 
    //                                                           .fTransportEndPosition :  ( -37.879   11.823 -100.000)  
    //                                                        .fTransportEndMomentumDir :  (  -0.000   -0.000    1.000)  
    //                                                               .fEndPointDistance :  349.9  
    //                                               .fParticleChange.thePositionChange :  (   0.000    0.000    0.000)  
    //                                      .fParticleChange.theMomentumDirectionChange :  (   0.000    0.000    0.000)  
    Process 75886 stopped
    * thread #1: tid = 0x19c75f, 0x0000000105b5a3ce libG4processes.dylib`G4Transportation::AlongStepGetPhysicalInteractionLength(this=0x0000000110964190, track=<unavailable>, (null)=<unavailable>, currentMinimumStep=<unavailable>, currentSafety=<unavailable>, selection=<unavailable>) + 3486 at G4Transportation.cc:518, queue = 'com.apple.main-thread', stop reason = breakpoint 6.1
        frame #0: 0x0000000105b5a3ce libG4processes.dylib`G4Transportation::AlongStepGetPhysicalInteractionLength(this=0x0000000110964190, track=<unavailable>, (null)=<unavailable>, currentMinimumStep=<unavailable>, currentSafety=<unavailable>, selection=<unavailable>) + 3486 at G4Transportation.cc:518
       515    fParticleChange.ProposeTrueStepLength(geometryStepLength) ;
       516  
       517    return geometryStepLength ;
    -> 518  }
       519  
       520  //////////////////////////////////////////////////////////////////////////
       521  //
    (lldb) c
    Process 75886 resuming

    //                              opticks.ana.cfg4lldb.G4SteppingManager2_cc_270_.[0] : Near end of DefinePhysicalStepLength : Inside MAXofAlongStepLoops after AlongStepGPIL 
    //                                                                             this : G4SteppingManager2_cc_270 
    //                                                  .fCurrentProcess.theProcessName :  Transportation  
    //                                                                   .physIntLength :  349.9  
    //                                                                    .PhysicalStep :  349.9  
    //                                                                     .fStepStatus :  fGeomBoundary  
    //                                    opticks.ana.cfg4lldb.CRandomEngine_cc_210.[3] : 0.93871349096298218 4  

    //                               opticks.ana.cfg4lldb.G4TrackingManager_cc_131_.[0] : Step Conclusion : TrackingManager step loop just after Stepping()  
    //                                                                             this : G4TrackingManager 
    //                                                   .fpSteppingManager.fStepStatus :  fGeomBoundary  
    //                                                  .fpSteppingManager.PhysicalStep :  349.9  
    //                                .fpSteppingManager.fCurrentProcess.theProcessName :  OpBoundary  
    //                                .fpSteppingManager.fStep.fpPreStepPoint.fPosition :  ( -37.879   11.823 -449.900)  
    //                              .fpSteppingManager.fStep.fpPreStepPoint.fGlobalTime :  0.2  
    //                       .fpSteppingManager.fStep.fpPreStepPoint.fMomentumDirection :  (  -0.000   -0.000    1.000)  
    //                               .fpSteppingManager.fStep.fpPostStepPoint.fPosition :  ( -37.879   11.823 -100.000)  
    //                             .fpSteppingManager.fStep.fpPostStepPoint.fGlobalTime :  1.36714  
    //                      .fpSteppingManager.fStep.fpPostStepPoint.fMomentumDirection :  (   0.000    0.000   -1.000)  
    //                                                             CRandomEngine_cc_210 : 4 
    //                                                         G4Transportation_cc_517_ : 1 
    //                                                        G4TrackingManager_cc_131_ : 1 
    //                                                       G4SteppingManager2_cc_270_ : 1 
    //                                                       G4SteppingManager2_cc_181_ : 0 
    //                                    opticks.ana.cfg4lldb.CRandomEngine_cc_210.[4] : 0.75380146503448486 5  
    //                                    opticks.ana.cfg4lldb.CRandomEngine_cc_210.[5] : 0.99984675645828247 6  
    //                                    opticks.ana.cfg4lldb.CRandomEngine_cc_210.[6] : 0.43801957368850708 7  
    G4SteppingManager2_cc_181_ : Dumping lengths collected by _181 after PostStep process loop  
    //                                                  .fCurrentProcess.theProcessName :  OpBoundary  
    //                                                                   .physIntLength :  1.79769e+308  
    //                                                  .fCurrentProcess.theProcessName :  OpRayleigh  
    //                                                                   .physIntLength :  153.255  
    //                                                  .fCurrentProcess.theProcessName :  OpAbsorption  
    //                                                                   .physIntLength :  825492  
    //                                                  .fCurrentProcess.theProcessName :  Transportation  
    //                                                                   .physIntLength :  1.79769e+308  

    //                                opticks.ana.cfg4lldb.G4Transportation_cc_517_.[1] : AlongStepGetPhysicalInteractionLength Exit  
    //                                                                             this : G4Transportation_cc_517 
    //                                                                   /startPosition :  ( -37.879   11.823 -100.000)  
    //                                                                /startMomentumDir :  (   0.000    0.000   -1.000)  
    //                                                                       /newSafety :  0  
    //                                                            .fGeometryLimitedStep : True 
    //                                                              .fFirstStepInVolume : True 
    //                                                               .fLastStepInVolume : False 
    //                                                                .fMomentumChanged : False 
    //                                                           .fTransportEndPosition :  ( -37.879   11.823 -100.000)  
    //                                                        .fTransportEndMomentumDir :  (   0.000    0.000   -1.000)  
    //                                                               .fEndPointDistance :  0  
    //                                               .fParticleChange.thePositionChange :  ( -37.879   11.823 -100.000)  
    //                                      .fParticleChange.theMomentumDirectionChange :  (  -0.000   -0.000    1.000)  
    Process 75886 stopped
    * thread #1: tid = 0x19c75f, 0x0000000105b5a3ce libG4processes.dylib`G4Transportation::AlongStepGetPhysicalInteractionLength(this=0x0000000110964190, track=<unavailable>, (null)=<unavailable>, currentMinimumStep=<unavailable>, currentSafety=<unavailable>, selection=<unavailable>) + 3486 at G4Transportation.cc:518, queue = 'com.apple.main-thread', stop reason = breakpoint 6.1
        frame #0: 0x0000000105b5a3ce libG4processes.dylib`G4Transportation::AlongStepGetPhysicalInteractionLength(this=0x0000000110964190, track=<unavailable>, (null)=<unavailable>, currentMinimumStep=<unavailable>, currentSafety=<unavailable>, selection=<unavailable>) + 3486 at G4Transportation.cc:518
       515    fParticleChange.ProposeTrueStepLength(geometryStepLength) ;
       516  
       517    return geometryStepLength ;
    -> 518  }
       519  
       520  //////////////////////////////////////////////////////////////////////////
       521  //
    (lldb) 




Smouldering evidence : PhysicalStep-zero/StepTooSmall results in RNG mis-alignment 
------------------------------------------------------------------------------------

Some G4 technicality yields zero step at BR, that means the lucky scatter 
throw that Opticks saw was not seen by G4 : as the sequence gets out of alignment.

::

    WITH_ALIGN_DEV_DEBUG photon_id:0 bounce:0 
    propagate_to_boundary  u_boundary_burn:    0.0011 
    propagate_to_boundary  u_scattering:    0.5026   scattering_distance:687866.4375 
    propagate_to_boundary  u_absorption:    0.6015   absorption_distance:5083218.0000 
    propagate_at_boundary  u_reflect:       0.93871  reflect:1   TransCoeff:   0.93847 
    WITH_ALIGN_DEV_DEBUG photon_id:0 bounce:1 
    propagate_to_boundary  u_boundary_burn:    0.7538 
    propagate_to_boundary  u_scattering:    0.9998   scattering_distance:  153.2073 
    propagate_to_boundary  u_absorption:    0.4380   absorption_distance:8254916.0000 
    rayleigh_scatter
    WITH_ALIGN_DEV_DEBUG photon_id:0 bounce:2 
    propagate_to_boundary  u_boundary_burn:    0.2825 
    propagate_to_boundary  u_scattering:    0.4325   scattering_distance:838178.1875 
    propagate_to_boundary  u_absorption:    0.9078   absorption_distance:966772.9375 
    propagate_at_surface   u_surface:       0.9121 
    propagate_at_surface   u_surface_burn:       0.2018 





::

    tboolean-;tboolean-box --okg4 --align --mask 1230 --pindex 0 -D

    2017-12-11 14:57:46.221 ERROR [1667660] [CRandomEngine::pretrack@256] CRandomEngine::pretrack record_id:  ctx.record_id 0 index 1230 mask.size 1
    //                                    opticks.ana.cfg4lldb.CRandomEngine_cc_210.[0] : 0.001117024919949472 1  
    //                                    opticks.ana.cfg4lldb.CRandomEngine_cc_210.[1] : 0.50264734029769897 2  
    //                                    opticks.ana.cfg4lldb.CRandomEngine_cc_210.[2] : 0.60150414705276489 3  
    G4SteppingManager2_cc_181_ : Dumping lengths collected by _181 ffter PostStep process loop  
    //                                                  .fCurrentProcess.theProcessName :  OpBoundary  
    //                                                                   .physIntLength :  1.79769e+308  
    //                                                  .fCurrentProcess.theProcessName :  OpRayleigh  
    //                                                                   .physIntLength :  687866  
    //                                                  .fCurrentProcess.theProcessName :  OpAbsorption  
    //                                                                   .physIntLength :  5.08322e+06  
    //                                                  .fCurrentProcess.theProcessName :  Transportation  
    //                                                                   .physIntLength :  1.79769e+308  

    //                              opticks.ana.cfg4lldb.G4SteppingManager2_cc_270_.[0] : Near end of DefinePhysicalStepLength : Inside MAXofAlongStepLoops after AlongStepGPIL 
    //                                                                             this : G4SteppingManager2_cc_270 
    //                                                  .fCurrentProcess.theProcessName :  Transportation  
    //                                                                   .physIntLength :  349.9  
    //                                                                    .PhysicalStep :  349.9  
    //                                                                     .fStepStatus :  fGeomBoundary  
    //                                    opticks.ana.cfg4lldb.CRandomEngine_cc_210.[3] : 0.93871349096298218 4  

    //                               opticks.ana.cfg4lldb.G4TrackingManager_cc_131_.[0] : Step Conclusion : TrackingManager step loop just after Stepping()  
    //                                                                             this : G4TrackingManager 
    //                                                   .fpSteppingManager.fStepStatus :  fGeomBoundary  
    //                                                  .fpSteppingManager.PhysicalStep :  349.9  
    //                                .fpSteppingManager.fCurrentProcess.theProcessName :  OpBoundary  
    //                                .fpSteppingManager.fStep.fpPreStepPoint.fPosition :  ( -37.879   11.823 -449.900)  
    //                              .fpSteppingManager.fStep.fpPreStepPoint.fGlobalTime :  0.2  
    //                       .fpSteppingManager.fStep.fpPreStepPoint.fMomentumDirection :  (  -0.000   -0.000    1.000)  
    //                               .fpSteppingManager.fStep.fpPostStepPoint.fPosition :  ( -37.879   11.823 -100.000)  
    //                             .fpSteppingManager.fStep.fpPostStepPoint.fGlobalTime :  1.36714  
    //                      .fpSteppingManager.fStep.fpPostStepPoint.fMomentumDirection :  (   0.000    0.000   -1.000)  
    //                                                             CRandomEngine_cc_210 : 4 
    //                                                        G4TrackingManager_cc_131_ : 1 
    //                                                       G4SteppingManager2_cc_270_ : 1 
    //                                                       G4SteppingManager2_cc_181_ : 0 
    //                                    opticks.ana.cfg4lldb.CRandomEngine_cc_210.[4] : 0.75380146503448486 5  
    //                                    opticks.ana.cfg4lldb.CRandomEngine_cc_210.[5] : 0.99984675645828247 6  
    //                                    opticks.ana.cfg4lldb.CRandomEngine_cc_210.[6] : 0.43801957368850708 7  
    G4SteppingManager2_cc_181_ : Dumping lengths collected by _181 ffter PostStep process loop  
    //                                                  .fCurrentProcess.theProcessName :  OpBoundary  
    //                                                                   .physIntLength :  1.79769e+308  
    //                                                  .fCurrentProcess.theProcessName :  OpRayleigh  
    //                                                                   .physIntLength :  153.255  
    //                                                  .fCurrentProcess.theProcessName :  OpAbsorption  
    //                                                                   .physIntLength :  825492  
    //                                                  .fCurrentProcess.theProcessName :  Transportation  
    //                                                                   .physIntLength :  1.79769e+308  

    //                              opticks.ana.cfg4lldb.G4SteppingManager2_cc_270_.[1] : Near end of DefinePhysicalStepLength : Inside MAXofAlongStepLoops after AlongStepGPIL 
    //                                                                             this : G4SteppingManager2_cc_270 
    //                                                  .fCurrentProcess.theProcessName :  Transportation  
    //                                                                   .physIntLength :  0  
    //                                                                    .PhysicalStep :  0  
    //                                                                     .fStepStatus :  fGeomBoundary  

    //                               opticks.ana.cfg4lldb.G4TrackingManager_cc_131_.[1] : Step Conclusion : TrackingManager step loop just after Stepping()  
    //                                                                             this : G4TrackingManager 
    //                                                   .fpSteppingManager.fStepStatus :  fGeomBoundary  
    //                                                  .fpSteppingManager.PhysicalStep :  0  
    //                                .fpSteppingManager.fCurrentProcess.theProcessName :  OpBoundary  
    //                                .fpSteppingManager.fStep.fpPreStepPoint.fPosition :  ( -37.879   11.823 -100.000)  
    //                              .fpSteppingManager.fStep.fpPreStepPoint.fGlobalTime :  1.36714  
    //                       .fpSteppingManager.fStep.fpPreStepPoint.fMomentumDirection :  (   0.000    0.000   -1.000)  
    //                               .fpSteppingManager.fStep.fpPostStepPoint.fPosition :  ( -37.879   11.823 -100.000)  
    //                             .fpSteppingManager.fStep.fpPostStepPoint.fGlobalTime :  1.36714  
    //                      .fpSteppingManager.fStep.fpPostStepPoint.fMomentumDirection :  (   0.000    0.000   -1.000)  
    //                                                             CRandomEngine_cc_210 : 7 
    //                                                        G4TrackingManager_cc_131_ : 2 
    //                                                       G4SteppingManager2_cc_270_ : 2 
    //                                                       G4SteppingManager2_cc_181_ : 0 
    //                                    opticks.ana.cfg4lldb.CRandomEngine_cc_210.[7] : 0.71403157711029053 8  
    //                                    opticks.ana.cfg4lldb.CRandomEngine_cc_210.[8] : 0.33040395379066467 9  
    //                                    opticks.ana.cfg4lldb.CRandomEngine_cc_210.[9] : 0.57074165344238281 10  
    G4SteppingManager2_cc_181_ : Dumping lengths collected by _181 ffter PostStep process loop  
    //                                                  .fCurrentProcess.theProcessName :  OpBoundary  
    //                                                                   .physIntLength :  1.79769e+308  
    //                                                  .fCurrentProcess.theProcessName :  OpRayleigh  
    //                                                                   .physIntLength :  1.10744e+06  
    //                                                  .fCurrentProcess.theProcessName :  OpAbsorption  
    //                                                                   .physIntLength :  5.60819e+06  
    //                                                  .fCurrentProcess.theProcessName :  Transportation  
    //                                                                   .physIntLength :  1.79769e+308  

    //                              opticks.ana.cfg4lldb.G4SteppingManager2_cc_270_.[2] : Near end of DefinePhysicalStepLength : Inside MAXofAlongStepLoops after AlongStepGPIL 
    //                                                                             this : G4SteppingManager2_cc_270 
    //                                                  .fCurrentProcess.theProcessName :  Transportation  
    //                                                                   .physIntLength :  350  
    //                                                                    .PhysicalStep :  350  
    //                                                                     .fStepStatus :  fGeomBoundary  
    //                                   opticks.ana.cfg4lldb.CRandomEngine_cc_210.[10] : 0.37590867280960083 11  
    //                                   opticks.ana.cfg4lldb.CRandomEngine_cc_210.[11] : 0.78497833013534546 12  

    //                               opticks.ana.cfg4lldb.G4TrackingManager_cc_131_.[2] : Step Conclusion : TrackingManager step loop just after Stepping()  
    //                                                                             this : G4TrackingManager 
    //                                                   .fpSteppingManager.fStepStatus :  fGeomBoundary  
    //                                                  .fpSteppingManager.PhysicalStep :  350  
    //                                .fpSteppingManager.fCurrentProcess.theProcessName :  OpBoundary  
    //                                .fpSteppingManager.fStep.fpPreStepPoint.fPosition :  ( -37.879   11.823 -100.000)  
    //                              .fpSteppingManager.fStep.fpPreStepPoint.fGlobalTime :  1.36714  
    //                       .fpSteppingManager.fStep.fpPreStepPoint.fMomentumDirection :  (   0.000    0.000   -1.000)  
    //                               .fpSteppingManager.fStep.fpPostStepPoint.fPosition :  ( -37.879   11.823 -450.000)  
    //                             .fpSteppingManager.fStep.fpPostStepPoint.fGlobalTime :  2.53462  
    //                      .fpSteppingManager.fStep.fpPostStepPoint.fMomentumDirection :  (   0.000    0.000   -1.000)  
    //                                                             CRandomEngine_cc_210 : 12 
    //                                                        G4TrackingManager_cc_131_ : 3 
    //                                                       G4SteppingManager2_cc_270_ : 3 
    //                                                       G4SteppingManager2_cc_181_ : 0 
    2017-12-11 14:57:46.775 INFO  [1667660] [CRunAction::EndOfRunAction@23] CRunAction::EndOfRunAction count 1

