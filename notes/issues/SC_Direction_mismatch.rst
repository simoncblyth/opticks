SC_Direction_mismatch
=======================

Supect RNG getting off-by-one for the 6 
---------------------------------------------

* :doc:`RNG_seq_off_by_one`





After move to cu/rayleigh.h:rayleigh_scatter_align and fixing polz bug
------------------------------------------------------------------------

* Up to 99994/100000 history aligned, all 6 maligned have different number of subsequent BR after an SC ? 
* chisq hits zero in seqhis/seqmsk 

::
    tboolean-;tboolean-box-ip

    In [1]: ab.maligned
    Out[1]: array([ 1230,  9041, 14510, 49786, 69653, 77962])

    In [2]: ab.dumpline(ab.maligned)
          0   1230 :                               TO BR SC BT BR BT SA                            TO BR SC BT BR BR BT SA 
          1   9041 :                         TO BT SC BR BR BR BR BT SA                               TO BT SC BR BR BT SA 
          2  14510 :                               TO SC BT BR BR BT SA                                  TO SC BT BR BT SA 
          3  49786 :                         TO BT BT SC BT BR BR BT SA                            TO BT BT SC BT BR BT SA 
          4  69653 :                               TO BT SC BR BR BT SA                                  TO BT SC BR BT SA 
          5  77962 :                               TO BT BR SC BR BT SA                            TO BT BR SC BR BR BT SA 


    tboolean-;tboolean-box-ip

    simon:optixrap blyth$ tboolean-;tboolean-box-ip
    args: /opt/local/bin/ipython -i -- /Users/blyth/opticks/ana/tboolean.py --det tboolean-box --tag 1
    [2017-12-12 16:29:21,953] p26699 {/Users/blyth/opticks/ana/base.py:335} INFO - envvar OPTICKS_ANA_DEFAULTS -> defaults {'src': 'torch', 'tag': '1', 'det': 'concentric'} 
    args: /Users/blyth/opticks/ana/tboolean.py --det tboolean-box --tag 1
    [2017-12-12 16:29:21,955] p26699 {/Users/blyth/opticks/ana/tboolean.py:58} INFO - tag 1 src torch det tboolean-box c2max 2.0 ipython True 
    [2017-12-12 16:29:21,955] p26699 {/Users/blyth/opticks/ana/ab.py:92} INFO - ab START
    AB(1,torch,tboolean-box)  None 0 
    A tboolean-box/torch/  1 :  20171212-1626 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-box/torch/1/fdom.npy () 
    B tboolean-box/torch/ -1 :  20171212-1626 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-box/torch/-1/fdom.npy (recstp) 
    Rock//perfectAbsorbSurface/Vacuum,Vacuum///GlassSchottF2
    /tmp/blyth/opticks/tboolean-box--
    .                seqhis_ana  1:tboolean-box   -1:tboolean-box        c2        ab        ba 
    .                             100000    100000         0.00/7 =  0.00  (pval:1.000 prob:0.000)  
    0000             8ccd     87777     87777             0.00        1.000 +- 0.003        1.000 +- 0.003  [4 ] TO BT BT SA
    0001              8bd      6312      6312             0.00        1.000 +- 0.013        1.000 +- 0.013  [3 ] TO BR SA
    0002            8cbcd      5420      5420             0.00        1.000 +- 0.014        1.000 +- 0.014  [5 ] TO BT BR BT SA
    0003           8cbbcd       349       349             0.00        1.000 +- 0.054        1.000 +- 0.054  [6 ] TO BT BR BR BT SA
    0004              86d        29        29             0.00        1.000 +- 0.186        1.000 +- 0.186  [3 ] TO SC SA
    0005          8cbbbcd        26        26             0.00        1.000 +- 0.196        1.000 +- 0.196  [7 ] TO BT BR BR BR BT SA
    0006            86ccd        24        24             0.00        1.000 +- 0.204        1.000 +- 0.204  [5 ] TO BT BT SC SA
    0007              4cd        16        16             0.00        1.000 +- 0.250        1.000 +- 0.250  [3 ] TO BT AB
    0008       bbbbbbb6cd         9         9             0.00        1.000 +- 0.333        1.000 +- 0.333  [10] TO BT SC BR BR BR BR BR BR BR
    0009            8c6cd         7         7             0.00        1.000 +- 0.378        1.000 +- 0.378  [5 ] TO BT SC BT SA
    0010         8cbc6ccd         3         4             0.00        0.750 +- 0.433        1.333 +- 0.667  [8 ] TO BT BT SC BT BR BT SA
    0011             4ccd         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [4 ] TO BT BT AB
    0012          8cc6ccd         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [7 ] TO BT BT SC BT BT SA
    0013           8cbc6d         3         4             0.00        0.750 +- 0.433        1.333 +- 0.667  [6 ] TO SC BT BR BT SA
    0014               4d         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [2 ] TO AB
    0015           86cbcd         2         2             0.00        1.000 +- 0.707        1.000 +- 0.707  [6 ] TO BT BR BT SC SA
    0016        8cbbc6ccd         2         1             0.00        2.000 +- 1.414        0.500 +- 0.500  [9 ] TO BT BT SC BT BR BR BT SA
    0017           8b6ccd         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [6 ] TO BT BT SC BR SA
    0018           8c6bcd         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [6 ] TO BT BR SC BT SA
    0019        8cbbbb6cd         1         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [9 ] TO BT SC BR BR BR BR BT SA
    .                             100000    100000         0.00/7 =  0.00  (pval:1.000 prob:0.000)  
    .                pflags_ana  1:tboolean-box   -1:tboolean-box        c2        ab        ba 
    .                             100000    100000         0.00/6 =  0.00  (pval:1.000 prob:0.000)  
    0000             1880     87777     87777             0.00        1.000 +- 0.003        1.000 +- 0.003  [3 ] TO|BT|SA
    0001             1480      6312      6312             0.00        1.000 +- 0.013        1.000 +- 0.013  [3 ] TO|BR|SA
    0002             1c80      5795      5795             0.00        1.000 +- 0.013        1.000 +- 0.013  [4 ] TO|BT|BR|SA
    0003             18a0        35        35             0.00        1.000 +- 0.169        1.000 +- 0.169  [4 ] TO|BT|SA|SC
    0004             10a0        29        29             0.00        1.000 +- 0.186        1.000 +- 0.186  [3 ] TO|SA|SC
    0005             1808        19        19             0.00        1.000 +- 0.229        1.000 +- 0.229  [3 ] TO|BT|AB
    0006             1ca0        18        18             0.00        1.000 +- 0.236        1.000 +- 0.236  [5 ] TO|BT|BR|SA|SC
    0007             1c20        10        10             0.00        1.000 +- 0.316        1.000 +- 0.316  [4 ] TO|BT|BR|SC
    0008             1008         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [2 ] TO|AB
    0009             1c08         2         2             0.00        1.000 +- 0.707        1.000 +- 0.707  [4 ] TO|BT|BR|AB
    .                             100000    100000         0.00/6 =  0.00  (pval:1.000 prob:0.000)  
    .                seqmat_ana  1:tboolean-box   -1:tboolean-box        c2        ab        ba 
    .                             100000    100000         0.02/6 =  0.00  (pval:1.000 prob:0.000)  
    0000             1232     87777     87777             0.00        1.000 +- 0.003        1.000 +- 0.003  [4 ] Vm F2 Vm Rk
    0001              122      6341      6341             0.00        1.000 +- 0.013        1.000 +- 0.013  [3 ] Vm Vm Rk
    0002            12332      5427      5427             0.00        1.000 +- 0.014        1.000 +- 0.014  [5 ] Vm F2 F2 Vm Rk
    0003           123332       350       351             0.00        0.997 +- 0.053        1.003 +- 0.054  [6 ] Vm F2 F2 F2 Vm Rk
    0004          1233332        28        27             0.02        1.037 +- 0.196        0.964 +- 0.186  [7 ] Vm F2 F2 F2 F2 Vm Rk
    0005            12232        24        24             0.00        1.000 +- 0.204        1.000 +- 0.204  [5 ] Vm F2 Vm Vm Rk
    0006              332        16        16             0.00        1.000 +- 0.250        1.000 +- 0.250  [3 ] Vm F2 F2
    0007       3333333332        10        10             0.00        1.000 +- 0.316        1.000 +- 0.316  [10] Vm F2 F2 F2 F2 F2 F2 F2 F2 F2
    0008          1232232         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [7 ] Vm F2 Vm Vm F2 Vm Rk
    0009               22         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [2 ] Vm Vm
    0010             2232         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [4 ] Vm F2 Vm Vm
    0011           123322         3         4             0.00        0.750 +- 0.433        1.333 +- 0.667  [6 ] Vm Vm F2 F2 Vm Rk
    0012         12332232         3         4             0.00        0.750 +- 0.433        1.333 +- 0.667  [8 ] Vm F2 Vm Vm F2 F2 Vm Rk
    0013           122332         2         2             0.00        1.000 +- 0.707        1.000 +- 0.707  [6 ] Vm F2 F2 Vm Vm Rk
    0014        123332232         2         1             0.00        2.000 +- 1.414        0.500 +- 0.500  [9 ] Vm F2 Vm Vm F2 F2 F2 Vm Rk
    0015        123333332         1         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [9 ] Vm F2 F2 F2 F2 F2 F2 Vm Rk
    0016           123222         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [6 ] Vm Vm Vm F2 Vm Rk
    0017            12322         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [5 ] Vm Vm F2 Vm Rk
    0018           122232         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [6 ] Vm F2 Vm Vm Vm Rk
    0019             3332         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [4 ] Vm F2 F2 F2
    .                             100000    100000         0.02/6 =  0.00  (pval:1.000 prob:0.000)  
    ab.a.metadata                  /tmp/blyth/opticks/evt/tboolean-box/torch/1 d22bf96e76fc2a4d120fc679faa1f607 c73dd7e7dad8c7e239794d2f2eda381c  100000    -1.0000 INTEROP_MODE 
    ab.a.metadata.csgmeta0 {u'containerscale': u'3', u'container': u'1', u'ctrl': u'0', u'verbosity': u'0', u'poly': u'IM', u'emitconfig': u'photons:100000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x1,umin:0.45,umax:0.55,vmin:0.45,vmax:0.55', u'resolution': u'20', u'emit': -1}
    rpost_dv maxdvmax:0.0137638477737 maxdv:[0.013763847773677895, 0.0, 0.0, 0.0, 0.013763847773674343, 0.0, 0.0, 0.0, 0.013763847773674343, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
     0000            :                    TO BT BT SA :   87777    87777  :     87777 1404432/     12: 0.000  mx/mn/av 0.01376/     0/1.176e-07  eps:0.0002    
     0001            :                       TO BR SA :    6312     6312  :      6312   75744/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0002            :                 TO BT BR BT SA :    5420     5420  :      5420  108400/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0003            :              TO BT BR BR BT SA :     349      349  :       349    8376/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0004            :                       TO SC SA :      29       29  :        29     348/      1: 0.003  mx/mn/av 0.01376/     0/3.955e-05  eps:0.0002    
     0005            :           TO BT BR BR BR BT SA :      26       26  :        26     728/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0006            :                 TO BT BT SC SA :      24       24  :        24     480/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0007            :                       TO BT AB :      16       16  :        16     192/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0008            :  TO BT SC BR BR BR BR BR BR BR :       9        9  :         9     360/      1: 0.003  mx/mn/av 0.01376/     0/3.823e-05  eps:0.0002    
     0009            :                 TO BT SC BT SA :       7        7  :         7     140/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0010            :        TO BT BT SC BT BR BT SA :       3        4  :         3      96/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0011            :                    TO BT BT AB :       3        3  :         3      48/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0012            :           TO BT BT SC BT BT SA :       3        3  :         3      84/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0013            :              TO SC BT BR BT SA :       3        4  :         3      72/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0014            :                          TO AB :       3        3  :         3      24/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0015            :              TO BT BR BT SC SA :       2        2  :         2      48/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0016            :     TO BT BT SC BT BR BR BT SA :       2        1  :         1      36/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0017            :              TO BT BT SC BR SA :       1        1  :         1      24/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0018            :              TO BT BR SC BT SA :       1        1  :         1      24/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0020            :  TO BT BR SC BR BR BR BR BR BR :       1        1  :         1      40/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0021            :                 TO SC BT BT SA :       1        1  :         1      20/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0024            :              TO BR SC BT BT SA :       1        1  :         1      24/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0026            :                    TO BT BR AB :       1        1  :         1      16/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0027            :                 TO BT BR BR AB :       1        1  :         1      20/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
    rpol_dv maxdvmax:0.0 maxdv:[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
     0000            :                    TO BT BT SA :   87777    87777  :     87777 1053324/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0001            :                       TO BR SA :    6312     6312  :      6312   56808/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0002            :                 TO BT BR BT SA :    5420     5420  :      5420   81300/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0003            :              TO BT BR BR BT SA :     349      349  :       349    6282/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0004            :                       TO SC SA :      29       29  :        29     261/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0005            :           TO BT BR BR BR BT SA :      26       26  :        26     546/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0006            :                 TO BT BT SC SA :      24       24  :        24     360/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0007            :                       TO BT AB :      16       16  :        16     144/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0008            :  TO BT SC BR BR BR BR BR BR BR :       9        9  :         9     270/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0009            :                 TO BT SC BT SA :       7        7  :         7     105/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0010            :        TO BT BT SC BT BR BT SA :       3        4  :         3      72/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0011            :                    TO BT BT AB :       3        3  :         3      36/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0012            :           TO BT BT SC BT BT SA :       3        3  :         3      63/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0013            :              TO SC BT BR BT SA :       3        4  :         3      54/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0014            :                          TO AB :       3        3  :         3      18/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0015            :              TO BT BR BT SC SA :       2        2  :         2      36/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0016            :     TO BT BT SC BT BR BR BT SA :       2        1  :         1      27/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0017            :              TO BT BT SC BR SA :       1        1  :         1      18/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0018            :              TO BT BR SC BT SA :       1        1  :         1      18/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0020            :  TO BT BR SC BR BR BR BR BR BR :       1        1  :         1      30/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0021            :                 TO SC BT BT SA :       1        1  :         1      15/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0024            :              TO BR SC BT BT SA :       1        1  :         1      18/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0026            :                    TO BT BR AB :       1        1  :         1      12/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0027            :                 TO BT BR BR AB :       1        1  :         1      15/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
    /Users/blyth/opticks/ana/dv.py:58: RuntimeWarning: invalid value encountered in greater
      discrep = dv[dv>eps]
    ox_dv maxdvmax:0.000457763671875 maxdv:[5.960464477539063e-08, 1.401298464324817e-45, 5.960464477539063e-08, 5.960464477539063e-08, 0.0002593994140625, 5.960464477539063e-08, 0.000156402587890625, 7.62939453125e-06, 0.00020599365234375, 0.0003662109375, 0.000457763671875, 2.384185791015625e-07, 3.0517578125e-05, 6.103515625e-05, nan, 9.918212890625e-05, 0.0001373291015625, 4.57763671875e-05, 3.0517578125e-05, 6.103515625e-05, 4.76837158203125e-05, 6.103515625e-05, 2.384185791015625e-07, 7.62939453125e-06] 
     0000            :                    TO BT BT SA :   87777    87777  :     87777 1404432/      0: 0.000  mx/mn/av 5.96e-08/     0/3.725e-09  eps:0.0002    
     0001            :                       TO BR SA :    6312     6312  :      6312  100992/      0: 0.000  mx/mn/av 1.401e-45/     0/8.758e-47  eps:0.0002    
     0002            :                 TO BT BR BT SA :    5420     5420  :      5420   86720/      0: 0.000  mx/mn/av 5.96e-08/     0/3.725e-09  eps:0.0002    
     0003            :              TO BT BR BR BT SA :     349      349  :       349    5584/      0: 0.000  mx/mn/av 5.96e-08/     0/3.725e-09  eps:0.0002    
     0004            :                       TO SC SA :      29       29  :        29     464/      1: 0.002  mx/mn/av 0.0002594/     0/4.576e-06  eps:0.0002    
     0005            :           TO BT BR BR BR BT SA :      26       26  :        26     416/      0: 0.000  mx/mn/av 5.96e-08/     0/3.725e-09  eps:0.0002    
     0006            :                 TO BT BT SC SA :      24       24  :        24     384/      0: 0.000  mx/mn/av 0.0001564/     0/3.292e-06  eps:0.0002    
     0007            :                       TO BT AB :      16       16  :        16     256/      0: 0.000  mx/mn/av 7.629e-06/     0/1.793e-07  eps:0.0002    
     0008            :  TO BT SC BR BR BR BR BR BR BR :       9        9  :         9     144/      1: 0.007  mx/mn/av 0.000206/     0/1.301e-05  eps:0.0002    
     0009            :                 TO BT SC BT SA :       7        7  :         7     112/      3: 0.027  mx/mn/av 0.0003662/     0/1.423e-05  eps:0.0002    
     0010            :        TO BT BT SC BT BR BT SA :       3        4  :         3      48/      3: 0.062  mx/mn/av 0.0004578/     0/2.935e-05  eps:0.0002    
     0011            :                    TO BT BT AB :       3        3  :         3      48/      0: 0.000  mx/mn/av 2.384e-07/     0/1.366e-08  eps:0.0002    
     0012            :           TO BT BT SC BT BT SA :       3        3  :         3      48/      0: 0.000  mx/mn/av 3.052e-05/     0/2.627e-06  eps:0.0002    
     0013            :              TO SC BT BR BT SA :       3        4  :         3      48/      0: 0.000  mx/mn/av 6.104e-05/     0/4.239e-06  eps:0.0002    
     0014            :                          TO AB :       3        3  :         3      48/      0: 0.000  mx/mn/av    nan/   nan/   nan  eps:0.0002    
     0015            :              TO BT BR BT SC SA :       2        2  :         2      32/      0: 0.000  mx/mn/av 9.918e-05/     0/5.985e-06  eps:0.0002    
     0016            :     TO BT BT SC BT BR BR BT SA :       2        1  :         1      16/      0: 0.000  mx/mn/av 0.0001373/     0/1.151e-05  eps:0.0002    
     0017            :              TO BT BT SC BR SA :       1        1  :         1      16/      0: 0.000  mx/mn/av 4.578e-05/     0/4.216e-06  eps:0.0002    
     0018            :              TO BT BR SC BT SA :       1        1  :         1      16/      0: 0.000  mx/mn/av 3.052e-05/     0/2.253e-06  eps:0.0002    
     0020            :  TO BT BR SC BR BR BR BR BR BR :       1        1  :         1      16/      0: 0.000  mx/mn/av 6.104e-05/     0/3.823e-06  eps:0.0002    
     0021            :                 TO SC BT BT SA :       1        1  :         1      16/      0: 0.000  mx/mn/av 4.768e-05/     0/6.839e-06  eps:0.0002    
     0024            :              TO BR SC BT BT SA :       1        1  :         1      16/      0: 0.000  mx/mn/av 6.104e-05/     0/9.572e-06  eps:0.0002    
     0026            :                    TO BT BR AB :       1        1  :         1      16/      0: 0.000  mx/mn/av 2.384e-07/     0/1.49e-08  eps:0.0002    
     0027            :                 TO BT BR BR AB :       1        1  :         1      16/      0: 0.000  mx/mn/av 7.629e-06/     0/5.066e-07  eps:0.0002    
    c2p : {'seqmat_ana': 0.0032680586175593308, 'pflags_ana': 0.0, 'seqhis_ana': 0.0} c2pmax: 0.00326805861756  CUT ok.c2max 2.0  RC:0 
    rmxs_ : {'rpol_dv': 0.0, 'rpost_dv': 0.013763847773677895} rmxs_max_: 0.0137638477737  CUT ok.rdvmax 0.1  RC:0 
    pmxs_ : {'ox_dv': 0.000457763671875} pmxs_max_: 0.000457763671875  CUT ok.pdvmax 0.001  RC:0 









Finding reciprocal constant bug
---------------------------------


::

    //                              opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[3] :    4   4  : 0.47344869375228882  
    //                              opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[4] :    5   5  : 0.021312465891242027  
    //                              opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[5] :    6   6  : 0.91965359449386597  
    //                              opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[6] :    7   7  : 0.078489311039447784  
    2017-12-12 16:13:38.055 INFO  [98676] [*OpRayleigh::PostStepDoIt@195]  cosTheta : -0.904715


    //                                 opticks.ana.cfg4lldb.OpRayleigh_cc_EndWhile_.[0] : EndWhile 
    //                                                                             this : OpRayleigh_cc_EndWhile 
    //                                                                  .theProcessName :  OpRayleigh  
    //                                                                    .thePILfactor :  1  
    //                                               .aParticleChange.thePositionChange :  (  -4.386   17.332 -273.276)  
    //                                           .aParticleChange.thePolarizationChange :  (   0.000   -1.000    0.000)  
    //                                      .aParticleChange.theMomentumDirectionChange :  (  -0.000   -0.000    1.000)  
    //                                                                            /rand :  5.77835  
    //                                                                        /constant :  -0.426018  
    //                                                                        /cosTheta :  -0.904715  
    //                                                                        /CosTheta :  -0.473449  
    //                                                                        /SinTheta :  0.880821  
    //                                                                          /CosPhi :  0.875256  
    //                                                                          /SinPhi :  -0.48366  
    //                                                            /OldMomentumDirection :  (  -0.000   -0.000    1.000)  
    //                                                            /NewMomentumDirection :  (   0.771   -0.426   -0.473)  
    //                                                                 /OldPolarization :  (   0.000   -1.000    0.000)  
    //                                                                 /NewPolarization :  (   0.363    0.905   -0.223)  
    //                              opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[7] :    8   8  : 0.74589663743972778  


    2017-12-12 16:13:39.013 ERROR [98676] [OPropagator::launch@183] LAUNCH NOW
    generate photon_id 0 
    WITH_ALIGN_DEV_DEBUG photon_id:0 bounce:0 
    propagate_to_boundary  u_boundary_burn:   0.6131115556 speed:      299.79245 
    propagate_to_boundary  u_scattering:   0.9998233914   scattering_length(s.material1.z):        1000000 scattering_distance:    176.6241608 
    propagate_to_boundary  u_absorption:   0.4490413368   absorption_length(s.material1.y):       10000000 absorption_distance:      8006403.5 
    rayleigh_scatter_align p.direction (-0 -0 1) 
    rayleigh_scatter_align p.polarization (0 -1 0) 
    rayleigh_scatter_align.do u0:0.473449 u1:0.0213125 u2:0.919654 u3:0.0784893 u4:0.745897 
    rayleigh_scatter_align.do constant        (-2.34732) 

    rayleigh_scatter_align.do newDirection    (0.770944 -0.426018 -0.473449)       <<<< matched

    rayleigh_scatter_align.do newPolarization (0.852141 -9.606e-09 -0.523313)       <<<< nope
 
    rayleigh_scatter_align.do doCosTheta 9.606e-09 doCosTheta2 9.22752e-17   looping 1   
    rayleigh_scatter_align.do u0:0.365573 u1:0.341214 u2:0.151641 u3:0.370584 u4:0.0321803 
    rayleigh_scatter_align.do constant        (1.31818) 
    rayleigh_scatter_align.do newDirection    (0.539306 0.75862 -0.365573) 
    rayleigh_scatter_align.do newPolarization (-0.82775 -2.3496e-08 0.561097) 
    rayleigh_scatter_align.do doCosTheta 2.3496e-08 doCosTheta2 5.52063e-16   looping 1   
    rayleigh_scatter_align.do u0:0.467722 u1:0.0983188 u2:0.420935 u3:0.211523 u4:0.689299 
    rayleigh_scatter_align.do constant        (2.37387) 
    rayleigh_scatter_align.do newDirection    (-0.777034 0.421253 -0.467722) 
    rayleigh_scatter_align.do newPolarization (0.856762 3.66324e-09 0.515712) 
    rayleigh_scatter_align.do doCosTheta -3.66324e-09 doCosTheta2 1.34193e-17   looping 1   
    rayleigh_scatter_align.do u0:0.358324 u1:0.447504 u2:0.921221 u3:0.984192 u4:0.385099 



Masked debug run on 1st aligned "TO SC SA"
--------------------------------------------

::

    tboolean-;tboolean-box-ip


    In [10]: ab.aselhis = "TO SC SA"

    In [17]: ab.a.dindex("TO SC SA")
    Out[17]: '--dindex=420,595,1198,2658,5113,6058,10409,13143,13162,14510'

    In [18]: ab.b.dindex("TO SC SA")
    Out[18]: '--dindex=420,1198,2658,5113,6058,10409,13143,13162,17035,26237'


::

   tboolean-;tboolean-box --okg4 --align --mask 420 -DD --pindex 0 


::

    2017-12-12 14:15:42.449 ERROR [63610] [CRandomEngine::pretrack@258] CRandomEngine::pretrack record_id:  ctx.record_id 0 index 420 mask.size 1
    //                              opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[0] :    1   1  : 0.61311155557632446  
    //                              opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[1] :    2   2  : 0.99982339143753052  
    //                              opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[2] :    3   3  : 0.44904133677482605  
    G4SteppingManager2_cc_181_ : Dumping lengths collected by _181 after PostStep process loop  
    //                                                  .fCurrentProcess.theProcessName :  OpBoundary  
    //                                                                   .physIntLength :  1.79769e+308  
    //                                                  .fCurrentProcess.theProcessName :  OpRayleigh  
    //                                                                   .physIntLength :  176.624  
    //                                                  .fCurrentProcess.theProcessName :  OpAbsorption  
    //                                                                   .physIntLength :  8.0064e+06  
    //                                                  .fCurrentProcess.theProcessName :  Transportation  
    //                                                                   .physIntLength :  1.79769e+308  

    //                                opticks.ana.cfg4lldb.G4Transportation_cc_517_.[0] : AlongStepGetPhysicalInteractionLength Exit  
    //                                                                             this : G4Transportation_cc_517 
    //                                                                   /startPosition :  (  -4.386   17.332 -449.900)  
    //                                                                /startMomentumDir :  (  -0.000   -0.000    1.000)  
    //                                                                       /newSafety :  0.100006  
    //                                                            .fGeometryLimitedStep : False 
    //                                                              .fFirstStepInVolume : True 
    //                                                               .fLastStepInVolume : False 
    //                                                                .fMomentumChanged : False 
    //                                                          .fShortStepOptimisation : False 
    //                                                           .fTransportEndPosition :  (  -4.386   17.332 -273.276)  
    //                                                        .fTransportEndMomentumDir :  (  -0.000   -0.000    1.000)  
    //                                                               .fEndPointDistance :  176.624  
    //                                               .fParticleChange.thePositionChange :  (   0.000    0.000    0.000)  
    //                                      .fParticleChange.theMomentumDirectionChange :  (   0.000    0.000    0.000)  
    //                                               .fLinearNavigator.fNumberZeroSteps :  0  
    //                                               .fLinearNavigator.fLastStepWasZero : False 

    //                              opticks.ana.cfg4lldb.G4SteppingManager2_cc_270_.[0] : Near end of DefinePhysicalStepLength : Inside MAXofAlongStepLoops after AlongStepGPIL 
    //                                                                             this : G4SteppingManager2_cc_270 
    //                                                  .fCurrentProcess.theProcessName :  Transportation  
    //                                                                   .physIntLength :  176.624  
    //                                                                    .PhysicalStep :  176.624  
    //                                                                     .fStepStatus :  fPostStepDoItProc  
    //                              opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[3] :    4   4  : 0.47344869375228882  
    //                              opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[4] :    5   5  : 0.021312465891242027  
    //                              opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[5] :    6   6  : 0.91965359449386597  
    //                              opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[6] :    7   7  : 0.078489311039447784  
    //                              opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[7] :    8   8  : 0.74589663743972778  


::

    2017-12-12 15:02:02.745 ERROR [76405] [OPropagator::launch@183] LAUNCH NOW
    generate photon_id 0 
    WITH_ALIGN_DEV_DEBUG photon_id:0 bounce:0 
    propagate_to_boundary  u_boundary_burn:   0.6131115556 speed:      299.79245 
    propagate_to_boundary  u_scattering:   0.9998233914   scattering_length(s.material1.z):        1000000 scattering_distance:    176.6241608 
    propagate_to_boundary  u_absorption:   0.4490413368   absorption_length(s.material1.y):       10000000 absorption_distance:      8006403.5 
    rayleigh_scatter_align
    rayleigh_scatter_align.do u0:0.473449 u1:0.0213125 u2:0.919654 u3:0.0784893 u4:0.745897 
    WITH_ALIGN_DEV_DEBUG photon_id:0 bounce:1 
    propagate_to_boundary  u_boundary_burn:   0.3655731678 speed:      299.79245 
    propagate_to_boundary  u_scattering:    0.341214478   scattering_length(s.material1.z):        1000000 scattering_distance:        1075244 
    propagate_to_boundary  u_absorption:   0.1516411602   absorption_length(s.material1.y):       10000000 absorption_distance:       18862384 
    propagate_at_surface   u_surface:       0.3706 
    propagate_at_surface   u_surface_burn:       0.0322 
    2017-12-12 15:02:02.760 ERROR [76405] [OPropagator::launch@185] LAUNCH DONE





    simon:optixrap blyth$ thrust_curand_printf 420
    thrust_curand_printf
     i0 420 i1 421 q0 0 q1 16 logf N
     id: 420 thread_offset:0 seq0:0 seq1:16 
     0.613112  0.999823  0.449041  0.473449 
     0.021312  0.919654  0.078489  0.745897 
     0.365573  0.341214  0.151641  0.370584 
     0.032180  0.467722  0.098319  0.420935 
    simon:optixrap blyth$ 


    //                         opticks.ana.cfg4lldb.OpRayleigh_cc_ExitPostStepDoIt_.[0] : ExitPostStepDoIt 
    //                                                                             this : OpRayleigh_cc_ExitPostStepDoIt 
    //                                                                  .theProcessName :  OpRayleigh  
    //                                                                    .thePILfactor :  1  
    //                                               .aParticleChange.thePositionChange :  (  -4.386   17.332 -273.276)  
    //                                           .aParticleChange.thePolarizationChange :  (   0.363    0.905   -0.223)  
    //                                      .aParticleChange.theMomentumDirectionChange :  (   0.771   -0.426   -0.473)  


    In [3]: brp = ab.b.rpost()[0]

    In [5]: brp
    Out[5]: 
    A()sliced
    A([[  -4.3907,   17.3287, -449.8989,    0.2002],
           [  -4.3907,   17.3287, -273.2812,    0.7892],
           [ 283.3839, -141.685 , -449.9952,    2.0344]])

    In [6]: d = brp[2] - brp[1] ; d
    Out[6]: 
    A([ 287.7745, -159.0137, -176.714 ,    1.2452])


    In [10]: d/np.sqrt(np.dot(d,d))
    Out[10]: 
    A()sliced
    A([ 0.771 , -0.426 , -0.4734])





    In [1]: ab.a.rpost()
    Out[1]: 
    A()sliced
    A([[[  -4.3907,   17.3287, -449.8989,    0.2002],
            [  -4.3907,   17.3287, -273.2812,    0.7892],
            [ -56.9686,   26.1926, -449.9952,    1.4051]]])

    In [2]: ab.b.rpost()
    Out[2]: 
    A()sliced
    A([[[  -4.3907,   17.3287, -449.8989,    0.2002],
            [  -4.3907,   17.3287, -273.2812,    0.7892],
            [ 283.3839, -141.685 , -449.9952,    2.0344]]])



With cu/rayleigh.h:rayleigh_scatter_align matched rpost (for 1-do only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


::

    In [1]: ab.a.rpost()
    Out[1]: 
    A()sliced
    A([[    [  -4.3907,   17.3287, -449.8989,    0.2002],
            [  -4.3907,   17.3287, -273.2812,    0.7892],
            [ 283.3839, -141.685 , -449.9952,    2.0344]]])

    In [2]: ab.b.rpost()
    Out[2]: 
    A()sliced
    A([[    [  -4.3907,   17.3287, -449.8989,    0.2002],
            [  -4.3907,   17.3287, -273.2812,    0.7892],
            [ 283.3839, -141.685 , -449.9952,    2.0344]]])

    In [3]: ab.a.rpost() - ab.b.rpost()
    Out[3]: 
    A()sliced
    A([[[ 0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.]]])



rpol not matched::

    In [4]: ab.a.rpol()
    Out[4]: 
    A()sliced
    A([[[ 0.    , -1.    ,  0.    ],
            [-0.7717, -1.    ,  0.4724],
            [-0.7717, -1.    ,  0.4724]]], dtype=float32)

    In [5]: ab.b.rpol()
    Out[5]: 
    A()sliced
    A([[[ 0.    , -1.    ,  0.    ],
            [ 0.3622,  0.9055, -0.2205],
            [ 0.3622,  0.9055, -0.2205]]], dtype=float32)




AFTER LOG DOUBLE FIX SC POSITIONS MATCHING, BUT NOT THE SCATTER DIRECTION
---------------------------------------------------------------------------

See :doc:`AB_SC_Position_Time_mismatch`


::

    tboolean-;tboolean-box-ip


    In [10]: ab.aselhis = "TO SC SA"

    In [11]: ab.a.rpost()

    In [15]: ab.a.rpost() - ab.b.rpost()
    Out[15]: 
    A()sliced
    A([[[   0.    ,    0.    ,    0.    ,    0.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ],
            [-340.3524,  167.8777,    0.    ,   -0.6293]],

           [[   0.    ,    0.    ,    0.    ,    0.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ],
            [ 235.5132, -415.3379,    0.    ,   -0.0098]],

           [[   0.    ,    0.    ,    0.    ,    0.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ],
            [  27.8305, -734.549 ,  332.1354,    0.4804]],

           [[   0.    ,    0.    ,    0.    ,    0.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ],
            [ 770.1423,    4.0879, -247.8869,   -0.1379]],

           [[   0.    ,    0.    ,    0.    ,    0.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ],
            [ 668.9505,  525.8065,   -1.2387,    0.2643]],


    In [17]: ab.a.dindex("TO SC SA")
    Out[17]: '--dindex=420,595,1198,2658,5113,6058,10409,13143,13162,14510'

    In [18]: ab.b.dindex("TO SC SA")
    Out[18]: '--dindex=420,1198,2658,5113,6058,10409,13143,13162,17035,26237'





Following AB decision there is a reemission throw for which there is no G4 equivalent.
But its the end of the line for that RNG sub-seq so this will have no effect 
so long as not in scintillator.

::

    085     if (absorption_distance <= scattering_distance)
     86     {
     87         if (absorption_distance <= s.distance_to_boundary)
     88         {
     89             p.time += absorption_distance/speed ;
     90             p.position += absorption_distance*p.direction;
     91 
     92             float uniform_sample_reemit = curand_uniform(&rng);
     93             if (uniform_sample_reemit < s.material1.w)                       // .w:reemission_prob
     94             {
     95                 // no materialIndex input to reemission_lookup as both scintillators share same CDF 
     96                 // non-scintillators have zero reemission_prob
     97                 p.wavelength = reemission_lookup(curand_uniform(&rng));
     98                 p.direction = uniform_sphere(&rng);
     99                 p.polarization = normalize(cross(uniform_sphere(&rng), p.direction));
    100                 p.flags.i.x = 0 ;   // no-boundary-yet for new direction
    101 
    102                 s.flag = BULK_REEMIT ;
    103                 return CONTINUE;
    104             }
    105             else
    106             {
    107                 s.flag = BULK_ABSORB ;
    108                 return BREAK;
    109             }
    110         }
    111         //  otherwise sail to boundary  
    112     }
    113     else
    114     {
    115         if (scattering_distance <= s.distance_to_boundary)
    116         {
    117             p.time += scattering_distance/speed ;
    118             p.position += scattering_distance*p.direction;
    119 
    120             rayleigh_scatter(p, rng);
    121 



