random_alignment_iterating
============================

RNG aligned comparisons
-------------------------

::

    tboolean-;tboolean-box --okg4 --align --alignlevel 0 -D


Initial deviation Observations
-----------------------------------

* overall good history matching (including histories with SC|AB) except for 1 "BR BR" surprise

  * surprisingly bad "TO BT BR BR BT SA :     349      346  :        28 "
   
* value matching good for BT|BR|SA 

  * poor "TO BT BR BR BT SA" from accidental history alignments : need to history align this first 

* value matching totally off for "SC"

* "TO AB" "TO BT AB" value matching looks to be trying 



Initial deviation comparison with rng aligned simulations 
---------------------------------------------------------------

::


    simon:optixrap blyth$ tboolean-;tboolean-box-ip
    ...
    rpost_dv maxdvmax:899.990478225 maxdv:[0.013763847773677895, 0.0, 0.0, 0.0, 881.2716452528459, 899.9904782250435, 0.055055391094704476, 299.9968260750145, 420.14145329142127, 0.49549851985227633, 331.39216284676655, 0.49549851985227633] 
     0000            :                    TO BT BT SA :   87777    87777  :     87777 1404432/     12: 0.000  mx/mn/av 0.01376/     0/1.176e-07  eps:0.0002    
     0001            :                       TO BR SA :    6312     6312  :      6310   75720/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0002            :                 TO BT BR BT SA :    5420     5438  :      5090  101800/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0003            :              TO BT BR BR BT SA :     349      346  :        28     672/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0004            :                       TO SC SA :      31       29  :        28     336/    133: 0.396  mx/mn/av  881.3/     0/ 64.55  eps:0.0002    
     0005            :                 TO BT BT SC SA :      27       24  :        21     420/     98: 0.233  mx/mn/av    900/     0/ 28.19  eps:0.0002    
     0007            :                       TO BT AB :      16       16  :        16     192/     21: 0.109  mx/mn/av 0.05506/     0/0.003815  eps:0.0002    
     0008            :  TO BT SC BR BR BR BR BR BR BR :       9        9  :         4     160/    115: 0.719  mx/mn/av    300/     0/ 61.75  eps:0.0002    
     0010            :        TO BT BT SC BT BR BT SA :       3        4  :         2      64/     27: 0.422  mx/mn/av  420.1/     0/ 28.15  eps:0.0002    
     0011            :                    TO BT BT AB :       3        3  :         3      48/      6: 0.125  mx/mn/av 0.4955/     0/0.02962  eps:0.0002    
     0012            :           TO BT BT SC BT BT SA :       3        3  :         1      28/     10: 0.357  mx/mn/av  331.4/     0/ 29.67  eps:0.0002    
     0013            :                          TO AB :       3        3  :         3      24/      6: 0.250  mx/mn/av 0.4955/     0/0.05985  eps:0.0002    
    rpol_dv maxdvmax:1.98425197601 maxdv:[0.0, 0.0, 0.0, 0.0, 1.9842519760131836, 1.9685039520263672, 0.0, 1.8346457481384277, 1.9133858680725098, 0.0, 0.20472443103790283, 0.0] 
     0000            :                    TO BT BT SA :   87777    87777  :     87777 1053324/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0001            :                       TO BR SA :    6312     6312  :      6310   56790/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0002            :                 TO BT BR BT SA :    5420     5438  :      5090   76350/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0003            :              TO BT BR BR BT SA :     349      346  :        28     504/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0004            :                       TO SC SA :      31       29  :        28     252/    168: 0.667  mx/mn/av  1.984/     0/ 0.375  eps:0.0002    
     0005            :                 TO BT BT SC SA :      27       24  :        21     315/    124: 0.394  mx/mn/av  1.969/     0/0.2309  eps:0.0002    
     0007            :                       TO BT AB :      16       16  :        16     144/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0008            :  TO BT SC BR BR BR BR BR BR BR :       9        9  :         4     120/     96: 0.800  mx/mn/av  1.835/     0/0.4668  eps:0.0002    
     0010            :        TO BT BT SC BT BR BT SA :       3        4  :         2      48/     30: 0.625  mx/mn/av  1.913/     0/0.2126  eps:0.0002    
     0011            :                    TO BT BT AB :       3        3  :         3      36/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0012            :           TO BT BT SC BT BT SA :       3        3  :         1      21/     12: 0.571  mx/mn/av 0.2047/     0/0.05024  eps:0.0002    
     0013            :                          TO AB :       3        3  :         3      18/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
    /Users/blyth/opticks/ana/dv.py:58: RuntimeWarning: invalid value encountered in greater
      discrep = dv[dv>eps]
    ox_dv maxdvmax:900.0 maxdv:[5.960464477539063e-08, 1.401298464324817e-45, 5.960464477539063e-08, 5.960464477539063e-08, 881.2715454101562, 900.0, 0.050258636474609375, 200.0, 420.14764404296875, 0.49346923828125, 331.3966979980469, nan] 
     0000            :                    TO BT BT SA :   87777    87777  :     87777 1404432/      0: 0.000  mx/mn/av 5.96e-08/     0/3.725e-09  eps:0.0002    
     0001            :                       TO BR SA :    6312     6312  :      6310  100960/      0: 0.000  mx/mn/av 1.401e-45/     0/8.758e-47  eps:0.0002    
     0002            :                 TO BT BR BT SA :    5420     5438  :      5090   81440/      0: 0.000  mx/mn/av 5.96e-08/     0/3.725e-09  eps:0.0002    
     0003            :              TO BT BR BR BT SA :     349      346  :        28     448/      0: 0.000  mx/mn/av 5.96e-08/     0/3.725e-09  eps:0.0002    
     0004            :                       TO SC SA :      31       29  :        28     448/    266: 0.594  mx/mn/av  881.3/     0/ 48.62  eps:0.0002    
     0005            :                 TO BT BT SC SA :      27       24  :        21     336/    197: 0.586  mx/mn/av    900/     0/ 35.45  eps:0.0002    
     0007            :                       TO BT AB :      16       16  :        16     256/     32: 0.125  mx/mn/av 0.05026/     0/0.003003  eps:0.0002    
     0008            :  TO BT SC BR BR BR BR BR BR BR :       9        9  :         4      64/     40: 0.625  mx/mn/av    200/     0/ 16.18  eps:0.0002    
     0010            :        TO BT BT SC BT BR BT SA :       3        4  :         2      32/     18: 0.562  mx/mn/av  420.1/     0/    31  eps:0.0002    
     0011            :                    TO BT BT AB :       3        3  :         3      48/      6: 0.125  mx/mn/av 0.4935/     0/0.02979  eps:0.0002    
     0012            :           TO BT BT SC BT BT SA :       3        3  :         1      16/     10: 0.625  mx/mn/av  331.4/     0/ 43.43  eps:0.0002    
     0013            :                          TO AB :       3        3  :         3      48/      6: 0.125  mx/mn/av    nan/   nan/   nan  eps:0.0002    
    c2p : {'seqmat_ana': 0.61238839507426712, 'pflags_ana': 0.024720449274528971, 'seqhis_ana': 0.55513237781188451} c2pmax: 0.612388395074  CUT ok.c2max 2.0  RC:0 
    rmxs_ : {'rpol_dv': 1.9842519760131836, 'rpost_dv': 899.9904782250435} rmxs_max_: 899.990478225  CUT ok.rdvmax 0.1  RC:88 
    pmxs_ : {'ox_dv': 900.0} pmxs_max_: 900.0  CUT ok.pdvmax 0.001  RC:99 

    In [1]: 



Initial chisq comp : too good as not-indep samples
-----------------------------------------------------

::

    simon:optixrap blyth$ tboolean-;tboolean-box-ip
    args: /opt/local/bin/ipython -i -- /Users/blyth/opticks/ana/tboolean.py --det tboolean-box --tag 1
    [2017-12-08 14:22:26,171] p11292 {/Users/blyth/opticks/ana/base.py:335} INFO - envvar OPTICKS_ANA_DEFAULTS -> defaults {'src': 'torch', 'tag': '1', 'det': 'concentric'} 
    args: /Users/blyth/opticks/ana/tboolean.py --det tboolean-box --tag 1
    [2017-12-08 14:22:26,173] p11292 {/Users/blyth/opticks/ana/tboolean.py:27} INFO - tag 1 src torch det tboolean-box c2max 2.0 ipython True 
    AB(1,torch,tboolean-box)  None 0 
    A tboolean-box/torch/  1 :  20171208-1407 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-box/torch/1/fdom.npy () 
    B tboolean-box/torch/ -1 :  20171208-1407 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-box/torch/-1/fdom.npy (recstp) 
    Rock//perfectAbsorbSurface/Vacuum,Vacuum///GlassSchottF2
    /tmp/blyth/opticks/tboolean-box--
    .                seqhis_ana  1:tboolean-box   -1:tboolean-box        c2        ab        ba 
    .                             100000    100000         3.89/7 =  0.56  (pval:0.793 prob:0.207)  
    0000             8ccd     87777     87777             0.00        1.000 +- 0.003        1.000 +- 0.003  [4 ] TO BT BT SA
    0001              8bd      6312      6312             0.00        1.000 +- 0.013        1.000 +- 0.013  [3 ] TO BR SA
    0002            8cbcd      5420      5438             0.03        0.997 +- 0.014        1.003 +- 0.014  [5 ] TO BT BR BT SA
    0003           8cbbcd       349       346             0.01        1.009 +- 0.054        0.991 +- 0.053  [6 ] TO BT BR BR BT SA
    0004              86d        31        29             0.07        1.069 +- 0.192        0.935 +- 0.174  [3 ] TO SC SA
    0005            86ccd        27        24             0.18        1.125 +- 0.217        0.889 +- 0.181  [5 ] TO BT BT SC SA
    0006          8cbbbcd        26        14             3.60        1.857 +- 0.364        0.538 +- 0.144  [7 ] TO BT BR BR BR BT SA
    0007              4cd        16        16             0.00        1.000 +- 0.250        1.000 +- 0.250  [3 ] TO BT AB
    0008       bbbbbbb6cd         9         9             0.00        1.000 +- 0.333        1.000 +- 0.333  [10] TO BT SC BR BR BR BR BR BR BR
    0009            8c6cd         6         7             0.00        0.857 +- 0.350        1.167 +- 0.441  [5 ] TO BT SC BT SA
    0010         8cbc6ccd         3         4             0.00        0.750 +- 0.433        1.333 +- 0.667  [8 ] TO BT BT SC BT BR BT SA
    0011             4ccd         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [4 ] TO BT BT AB
    0012          8cc6ccd         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [7 ] TO BT BT SC BT BT SA
    0013               4d         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [2 ] TO AB
    0014           86cbcd         2         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [6 ] TO BT BR BT SC SA
    0015           8cb6cd         2         1             0.00        2.000 +- 1.414        0.500 +- 0.500  [6 ] TO BT SC BR BT SA
    0016       8cbbbbb6cd         1         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [10] TO BT SC BR BR BR BR BR BT SA
    0017           8c6bcd         1         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [6 ] TO BT BR SC BT SA
    0018            8cc6d         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [5 ] TO SC BT BT SA
    0019          8cb6bcd         1         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [7 ] TO BT BR SC BR BT SA
    .                             100000    100000         3.89/7 =  0.56  (pval:0.793 prob:0.207)  
    .                pflags_ana  1:tboolean-box   -1:tboolean-box        c2        ab        ba 
    .                             100000    100000         0.12/5 =  0.02  (pval:1.000 prob:0.000)  
    0000             1880     87777     87777             0.00        1.000 +- 0.003        1.000 +- 0.003  [3 ] TO|BT|SA
    0001             1480      6312      6312             0.00        1.000 +- 0.013        1.000 +- 0.013  [3 ] TO|BR|SA
    0002             1c80      5795      5799             0.00        0.999 +- 0.013        1.001 +- 0.013  [4 ] TO|BT|BR|SA
    0003             18a0        37        35             0.06        1.057 +- 0.174        0.946 +- 0.160  [4 ] TO|BT|SA|SC
    0004             10a0        31        29             0.07        1.069 +- 0.192        0.935 +- 0.174  [3 ] TO|SA|SC
    0005             1808        19        19             0.00        1.000 +- 0.229        1.000 +- 0.229  [3 ] TO|BT|AB
    0006             1ca0        14        13             0.00        1.077 +- 0.288        0.929 +- 0.258  [5 ] TO|BT|BR|SA|SC
    0007             1c20         9        10             0.00        0.900 +- 0.300        1.111 +- 0.351  [4 ] TO|BT|BR|SC
    0008             1008         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [2 ] TO|AB
    0009             1c08         2         1             0.00        2.000 +- 1.414        0.500 +- 0.500  [4 ] TO|BT|BR|AB
    0010             14a0         1         2             0.00        0.500 +- 0.500        2.000 +- 1.414  [4 ] TO|BR|SA|SC
    .                             100000    100000         0.12/5 =  0.02  (pval:1.000 prob:0.000)  
    .                seqmat_ana  1:tboolean-box   -1:tboolean-box        c2        ab        ba 
    .                             100000    100000         3.67/6 =  0.61  (pval:0.721 prob:0.279)  
    0000             1232     87777     87777             0.00        1.000 +- 0.003        1.000 +- 0.003  [4 ] Vm F2 Vm Rk
    0001              122      6343      6341             0.00        1.000 +- 0.013        1.000 +- 0.013  [3 ] Vm Vm Rk
    0002            12332      5426      5445             0.03        0.997 +- 0.014        1.004 +- 0.014  [5 ] Vm F2 F2 Vm Rk
    0003           123332       352       347             0.04        1.014 +- 0.054        0.986 +- 0.053  [6 ] Vm F2 F2 F2 Vm Rk
    0004          1233332        27        15             3.43        1.800 +- 0.346        0.556 +- 0.143  [7 ] Vm F2 F2 F2 F2 Vm Rk
    0005            12232        27        24             0.18        1.125 +- 0.217        0.889 +- 0.181  [5 ] Vm F2 Vm Vm Rk
    0006              332        16        16             0.00        1.000 +- 0.250        1.000 +- 0.250  [3 ] Vm F2 F2
    0007       3333333332         9        10             0.00        0.900 +- 0.300        1.111 +- 0.351  [10] Vm F2 F2 F2 F2 F2 F2 F2 F2 F2
    0008             2232         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [4 ] Vm F2 Vm Vm
    0009          1232232         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [7 ] Vm F2 Vm Vm F2 Vm Rk
    0010               22         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [2 ] Vm Vm
    0011         12332232         3         4             0.00        0.750 +- 0.433        1.333 +- 0.667  [8 ] Vm F2 Vm Vm F2 F2 Vm Rk
    0012       1233333332         2         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [10] Vm F2 F2 F2 F2 F2 F2 F2 Vm Rk
    0013           122332         2         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [6 ] Vm F2 F2 Vm Vm Rk
    0014            12322         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [5 ] Vm Vm F2 Vm Rk
    0015          1233322         1         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [7 ] Vm Vm F2 F2 F2 Vm Rk
    0016           123322         1         4             0.00        0.250 +- 0.250        4.000 +- 2.000  [6 ] Vm Vm F2 F2 Vm Rk
    0017           123222         1         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [6 ] Vm Vm Vm F2 Vm Rk
    0018             3332         1         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [4 ] Vm F2 F2 F2
    0019            33332         1         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] Vm F2 F2 F2 F2
    .                             100000    100000         3.67/6 =  0.61  (pval:0.721 prob:0.279)  
    ab.a.metadata                  /tmp/blyth/opticks/evt/tboolean-box/torch/1 e3b4ee8211178b213c6da01bfd4f9be2 3a624e7d0fc57237b2ecd23c0c9cdd25  100000    -1.0000 INTEROP_MODE 
    ab.a.metadata.csgmeta0 {u'containerscale': u'3', u'container': u'1', u'ctrl': u'0', u'verbosity': u'0', u'poly': u'IM', u'emitconfig': u'photons:100000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x1,umin:0.45,umax:0.55,vmin:0.45,vmax:0.55', u'resolution': u'20', u'emit': -1}
 



Iteration Approach 1 : Directly select/dump non-history aligned records
---------------------------------------------------------------------------------

* 0.7% history mismatch 

Of the 717/100000, many with different BR counts between the simulations.

::

    In [47]: np.where( ab.a.seqhis == ab.b.seqhis )[0].shape
    Out[47]: (99283,)

    In [48]: np.where( ab.a.seqhis != ab.b.seqhis )[0].shape
    Out[48]: (717,)

    In [50]: maligned = np.where( ab.a.seqhis != ab.b.seqhis )[0]

    In [4]: ab.dumpline(slice(0,1000,50))
          0      0 :                                        TO BT BT SA                                        TO BT BT SA 
          1     50 :                                        TO BT BT SA                                        TO BT BT SA 
          2    100 :                                     TO BT BR BT SA                                     TO BT BR BT SA 
          3    150 :                                        TO BT BT SA                                        TO BT BT SA 
          4    200 :                                           TO BR SA                                           TO BR SA 
          5    250 :                                        TO BT BT SA                                        TO BT BT SA 
          6    300 :                                        TO BT BT SA                                        TO BT BT SA 
          7    350 :                                        TO BT BT SA                                        TO BT BT SA 
          8    400 :                                        TO BT BT SA                                        TO BT BT SA 
          9    450 :                                        TO BT BT SA                                        TO BT BT SA 


    In [2]: ab.dumpline(ab.maligned)
          0    107 :                               TO BT BR BR BR BT SA                                     TO BT BR BT SA 
          1    130 :                                     TO BT BR BT SA                                  TO BT BR BR BT SA 
          2    355 :                                     TO BT BR BT SA                                  TO BT BR BR BT SA 
          3    370 :                                  TO BT BR BR BT SA                                     TO BT BR BT SA 
          4    595 :                                           TO SC SA                                  TO SC BT BR BT SA 
          5    858 :                                  TO BT BR BR BT SA                                     TO BT BR BT SA 
          6    906 :                                     TO BT BR BT SA                                  TO BT BR BR BT SA 
          7    942 :                                  TO BT BR BR BT SA                                     TO BT BR BT SA 
          8    996 :                                  TO BT BR BR BT SA                                     TO BT BR BT SA 
          9   1043 :                                     TO BT BR BT SA                                  TO BT BR BR BT SA 
         10   1230 :                                        TO BR SC SA                                           TO BR SA 
         11   1302 :                                     TO BT BR BT SA                                  TO BT BR BR BT SA 
         12   1363 :                                     TO BT BR BT SA                                  TO BT BR BR BT SA 
         13   1696 :                                  TO BT BR BR BT SA                                     TO BT BR BT SA 
         14   1717 :                                  TO BT BR BR BT SA                                     TO BT BR BT SA 
         15   1822 :                                     TO BT BR BT SA                                  TO BT BR BR BT SA 
         16   1907 :                                     TO BT BR BT SA                      TO BT BR SC BR BR BR BR BR BR 
         17   2094 :                                     TO BT BR BT SA                                  TO BT BR BR BT SA 
         18   2111 :                                  TO BT BR BR BT SA                                     TO BT BR BT SA 
         19   2180 :                               TO BT BR BR BR BT SA                                     TO BT BR BT SA 
         20   2333 :                                     TO BT BR BT SA                                  TO BT BR BR BT SA 
        ...
        676  94587 :                                     TO BT BR BT SA                                  TO BT BR BR BT SA 
        677  94773 :                      TO BT SC BR BR BR BR BR BR BR                                     TO BT SC BT SA 
        678  94891 :                                     TO BT SC BT SA                      TO BT SC BR BR BR BR BR BR BR 
        679  94934 :                                  TO BT BR BR BT SA                                     TO BT BR BT SA 
        680  95204 :                                  TO BT BR BR BT SA                                     TO BT BR BT SA 
        681  95266 :                                     TO BT BR BT SA                                  TO BT BR BR BT SA 
        682  95287 :                                  TO BT BR BR BT SA                                     TO BT BR BT SA 
        683  95614 :                               TO BT BR BR BR BT SA                                  TO BT BR BR BT SA 
        684  95722 :                                  TO BT BR BT SC SA                                     TO BT BR BT SA 
        685  95967 :                            TO BT BT SC BT BR BT SA                                     TO BT BT SC SA 
        686  96040 :                                     TO BT BR BT SA                                  TO BT BR BR BT SA 
        687  96258 :                                  TO BT BR BR BT SA                                     TO BT BR BT SA 
        688  96292 :                                     TO BT BR BT SA                                  TO BT BR BR BT SA 
        689  96365 :                                     TO BT BR BT SA                                  TO BT BR BR BT SA 
        690  96480 :                                     TO BT BR BT SA                                  TO BT BR BR BT SA 
        691  96698 :                                  TO BT BR BR BT SA                                     TO BT BR BT SA 
        692  96764 :                                     TO BT BR BT SA                                  TO BT BR BR BT SA 
        693  96942 :                                     TO BT BR BT SA                               TO BT BR BR BR BT SA 
        694  96952 :                                  TO BT BR BR BT SA                                     TO BT BR BT SA 
        695  97230 :                                     TO BT BR BT SA                                  TO BT BR BR BT SA 
        696  97378 :                                     TO BT BR BT SA                                  TO BT BR BR BT SA 
        697  97449 :                                     TO BT BR BT SA                                  TO BT BR BR BT SA 
        698  97607 :                               TO BT BR BR BR BT SA                                     TO BT BR BT SA 
        699  97649 :                                     TO BT BR BT SA                                  TO BT BR BR BT SA 
        700  97697 :                                     TO BT BR BT SA                                  TO BT BR BR BT SA 
        701  97887 :                                     TO SC BT BT SA                                  TO SC BT BR BT SA 
        702  97981 :                                     TO BT BR BT SA                                  TO BT BR BR BT SA 
        703  98012 :                                  TO BT BR BR BT SA                                     TO BT BR BT SA 
        704  98146 :                                     TO BT BR BT SA                                  TO BT BR BR BT SA 
        705  98235 :                                  TO BT BR BR BT SA                                     TO BT BR BT SA 
        706  98514 :                                     TO BT BR BT SA                                  TO BT BR BR BT SA 
        707  98577 :                                  TO BT BR BR BT SA                                     TO BT BR BT SA 
        708  98680 :                                     TO BT BR BT SA                                  TO BT BR BR BT SA 
        709  98756 :                                     TO BT BR BT SA                                  TO BT BR BR BT SA 
        710  99009 :                                     TO BT BR BT SA                                  TO BT BR BR BT SA 
        711  99250 :                                  TO BT BR BR BT SA                                     TO BT BR BT SA 
        712  99293 :                                  TO BT BR BR BT SA                                     TO BT BR BT SA 
        713  99331 :                                  TO BT BR BR BT SA                                     TO BT BR BT SA 
        714  99413 :                                  TO BT BR BR BT SA                                     TO BT BR BT SA 
        715  99702 :                                     TO BT BR BT SA                                  TO BT BR BR BT SA 
        716  99895 :                                     TO BT BR BT SA                                  TO BT BR BR BT SA 




Approach 2 : dindex dumping
-------------------------------

Dump

::

    In [38]: ab.a.dindex("TO BT BR BR BT SA")
    Out[38]: '--dindex=360,370,858,942,996,1696,1717,2111,2340,3040'

    In [39]: ab.b.dindex("TO BT BR BR BT SA")
    Out[39]: '--dindex=130,355,360,906,1043,1302,1363,1822,2094,2333'


"TO BT BR BR BT SA" all accidentals
-------------------------------------

::

    In [38]: ab.a.dindex("TO BT BR BR BT SA")
    Out[38]: '--dindex=360,370,858,942,996,1696,1717,2111,2340,3040'

    In [39]: ab.b.dindex("TO BT BR BR BT SA")
    Out[39]: '--dindex=130,355,360,906,1043,1302,1363,1822,2094,2333'


"TO SC SA" looks totally off
-------------------------------------

::

    In [13]: ab.aselhis = "TO SC SA"

    In [14]: ab.a.rpost()[:5]
    Out[14]: 
    A()sliced
    A([[[  -4.3907,   17.3287, -449.8989,    0.2002],
            [  -4.3907,   17.3287, -273.3225,    0.7892],
            [ -56.9548,   26.1788, -449.9952,    1.4045]],

           [[  41.3191,   32.5377, -449.8989,    0.2002],
            [  41.3191,   32.5377, -122.8423,    1.2909],
            [ 114.006 , -197.6626, -449.9952,    2.6472]],

           [[   0.1652,  -17.3287, -449.8989,    0.2002],
            [   0.1652,  -17.3287, -385.5667,    0.4144],
            [-422.1647, -449.9952,  -61.2629,    2.7033]],

           [[ -33.1984,  -38.7177, -449.8989,    0.2002],
            [ -33.1984,  -38.7177, -313.0312,    0.6568],
            [ 320.0232,  231.5492, -449.9952,    2.2089]],

           [[ -11.9057,  -18.6775, -449.8989,    0.2002],
            [ -11.9057,  -18.6775, -376.0971,    0.4462],
            [ 218.9553,  449.9952, -297.7946,    2.2083]]])

    In [15]: ab.b.rpost()[:5]
    Out[15]: 
    A()sliced
    A([[[  -4.3907,   17.3287, -449.8989,    0.2002],
            [  -4.3907,   17.3287, -273.2812,    0.7892],
            [ 283.3839, -141.685 , -449.9952,    2.0344]],

           [[  41.3191,   32.5377, -449.8989,    0.2002],
            [  41.3191,   32.5377, -122.801 ,    1.2909],
            [-121.4935,  217.6477, -449.9952,    2.6576]],

           [[   0.1652,  -17.3287, -449.8989,    0.2002],
            [   0.1652,  -17.3287, -385.5254,    0.4144],
            [-449.9952,  284.5538, -393.3432,    2.223 ]],

           [[ -33.1984,  -38.7177, -449.8989,    0.2002],
            [ -33.1984,  -38.7177, -312.9761,    0.6568],
            [-449.9952,  227.5577, -202.1083,    2.3475]],

           [[ -11.9057,  -18.6775, -449.8989,    0.2002],
            [ -11.9057,  -18.6775, -376.0421,    0.4462],
            [-449.9952,  -75.8113, -296.5146,    1.944 ]]])



"TO AB" "TO BT AB" looks to be trying to do the same thing : velocity bug again perhaps ?
--------------------------------------------------------------------------------------------


::

    In [10]: ab.aselhis = "TO AB"

    In [11]: ab.a.rpost()
    Out[11]: 
    A()sliced
    A([[[  32.3038,  -30.831 , -449.8989,    0.2002],
            [  32.3038,  -30.831 , -381.2311,    0.4291]],

           [[ -14.9751,   25.2704, -449.8989,    0.2002],
            [ -14.9751,   25.2704, -282.9021,    0.7569]],

           [[ -32.0422,    6.9507, -449.8989,    0.2002],
            [ -32.0422,    6.9507, -224.4608,    0.9522]]])

    In [12]: ab.b.rpost()
    Out[12]: 
    A()sliced
    A([[[  32.3038,  -30.831 , -449.8989,    0.2002],
            [  32.3038,  -30.831 , -380.7631,    0.4309]],

           [[ -14.9751,   25.2704, -449.8989,    0.2002],
            [ -14.9751,   25.2704, -282.4066,    0.7587]],

           [[ -32.0422,    6.9507, -449.8989,    0.2002],
            [ -32.0422,    6.9507, -223.9929,    0.9534]]])


    In [16]: ab.aselhis = "TO BT AB"

    In [17]: ab.a.rpost()[:5]
    Out[17]: 
    A()sliced
    A([[[  16.3102,   14.3006, -449.8989,    0.2002],
            [  16.3102,   14.3006,  -99.9944,    1.3672],
            [  16.3102,   14.3006,  -39.4197,    1.7341]],

           [[  31.3816,   15.6633, -449.8989,    0.2002],
            [  31.3816,   15.6633,  -99.9944,    1.3672],
            [  31.3816,   15.6633,   57.7393,    2.3231]],

           [[ -25.1053,  -17.6315, -449.8989,    0.2002],
            [ -25.1053,  -17.6315,  -99.9944,    1.3672],
            [ -25.1053,  -17.6315,   11.0661,    2.0399]],

           [[  12.3186,   34.038 , -449.8989,    0.2002],
            [  12.3186,   34.038 ,  -99.9944,    1.3672],
            [  12.3186,   34.038 ,   38.6076,    2.2071]],

           [[ -41.2503,   29.1518, -449.8989,    0.2002],
            [ -41.2503,   29.1518,  -99.9944,    1.3672],
            [ -41.2503,   29.1518,   38.1259,    2.204 ]]])

    In [18]: ab.b.rpost()[:5]
    Out[18]: 
    A()sliced
    A([[[  16.3102,   14.3006, -449.8989,    0.2002],
            [  16.3102,   14.3006,  -99.9944,    1.3672],
            [  16.3102,   14.3006,  -39.3784,    1.7347]],

           [[  31.3816,   15.6633, -449.8989,    0.2002],
            [  31.3816,   15.6633,  -99.9944,    1.3672],
            [  31.3816,   15.6633,   57.7806,    2.3231]],

           [[ -25.1053,  -17.6315, -449.8989,    0.2002],
            [ -25.1053,  -17.6315,  -99.9944,    1.3672],
            [ -25.1053,  -17.6315,   11.1074,    2.0405]],

           [[  12.3186,   34.038 , -449.8989,    0.2002],
            [  12.3186,   34.038 ,  -99.9944,    1.3672],
            [  12.3186,   34.038 ,   38.6489,    2.2071]],

           [[ -41.2503,   29.1518, -449.8989,    0.2002],
            [ -41.2503,   29.1518,  -99.9944,    1.3672],
            [ -41.2503,   29.1518,   38.1671,    2.2047]]])



