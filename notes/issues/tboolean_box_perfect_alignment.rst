tboolean_box_perfect_alignment
==================================

observations
--------------

* all 3 chisq are zero 
* 14 rpost values out of 1.5 million deviate > eps 0.0002, all deviations are 0.01376 (looks like domain compression bin-edge)
* rpol perfect match (rpol are extremely domain compressed so larger bins)
* 8 ox values out of 1.5 million deviate > eps 0.0002, all 8 from photons that scatter
* ox values have many float precision level discrep
* 3 "TO AB" photons have NaN, fixed by skipping flags in ox_dv comparison see  :doc:`prosecute_some_TO_AB_NaN`

small deviations : no concerns
--------------------------------

Looked into in :doc:`tboolean_box_perfect_alignment_small_deviations` 


malign check
--------------

::

    tboolean-;TBOOLEAN_TAG=3 tboolean-box-ip

    In [1]: ab.maligned
    Out[1]: array([], dtype=int64)


extracts
----------

::

    tboolean-;TBOOLEAN_TAG=3 tboolean-box --okg4 --align --dbgskipclearzero --dbgnojumpzero --dbgkludgeflatzero



* 14 rpost values out of 1.5 million deviate > eps 0.0002 : all deviations are 0.01376 (looks like domain compression bin-edge)

::

    rpost_dv maxdvmax:0.0137638477737 maxdv:[0.013763847773677895, 0.0, 0.0, 0.0, 0.013763847773674343, 0.0, 0.0, 0.0, 0.013763847773674343, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
     0000            :                    TO BT BT SA :   87777    87777  :     87777 1404432/     12: 0.000  mx/mn/av 0.01376/     0/1.176e-07  eps:0.0002    
     0004            :                       TO SC SA :      29       29  :        29     348/      1: 0.003  mx/mn/av 0.01376/     0/3.955e-05  eps:0.0002    
     0008            :  TO BT SC BR BR BR BR BR BR BR :       9        9  :         9     360/      1: 0.003  mx/mn/av 0.01376/     0/3.823e-05  eps:0.0002    
 

* 8 ox values out of 1.5 million deviate > eps 0.0002, all 8 from photons that scatter
* many float precision level discrep
* some nans in "TO AB" to chase

:: 

    /Users/blyth/opticks/ana/dv.py:58: RuntimeWarning: invalid value encountered in greater
      discrep = dv[dv>eps]
    ox_dv maxdvmax:0.000457763671875 maxdv:[5.960464477539063e-08, 1.401298464324817e-45, 5.960464477539063e-08, 5.960464477539063e-08, 0.0002593994140625, 5.960464477539063e-08, 0.000156402587890625, 7.62939453125e-06, 0.00020599365234375, 0.0003662109375, 0.000457763671875, 2.384185791015625e-07, 3.0517578125e-05, 6.103515625e-05, nan, 9.918212890625e-05, 0.0001373291015625, 4.57763671875e-05, 3.0517578125e-05, 0.0001220703125, 6.103515625e-05, 4.76837158203125e-05, 0.00016832351684570312, 0.0001373291015625, 6.103515625e-05, 0.00019073486328125, 2.384185791015625e-07, 7.62939453125e-06, 0.00018310546875] 
     0000            :                    TO BT BT SA :   87777    87777  :     87777 1404432/      0: 0.000  mx/mn/av 5.96e-08/     0/3.725e-09  eps:0.0002    

     0004            :                       TO SC SA :      29       29  :        29     464/      1: 0.002  mx/mn/av 0.0002594/     0/4.576e-06  eps:0.0002    

     0006            :                 TO BT BT SC SA :      24       24  :        24     384/      0: 0.000  mx/mn/av 0.0001564/     0/3.292e-06  eps:0.0002    

     0008            :  TO BT SC BR BR BR BR BR BR BR :       9        9  :         9     144/      1: 0.007  mx/mn/av 0.000206/     0/1.301e-05  eps:0.0002    
     0009            :                 TO BT SC BT SA :       7        7  :         7     112/      3: 0.027  mx/mn/av 0.0003662/     0/1.423e-05  eps:0.0002    
     0010            :        TO BT BT SC BT BR BT SA :       3        3  :         3      48/      3: 0.062  mx/mn/av 0.0004578/     0/2.935e-05  eps:0.0002    

     0014            :                          TO AB :       3        3  :         3      48/      0: 0.000  mx/mn/av    nan/   nan/   nan  eps:0.0002    

     0016            :     TO BT BT SC BT BR BR BT SA :       2        2  :         2      32/      0: 0.000  mx/mn/av 0.0001373/     0/9.584e-06  eps:0.0002    
     0019            :     TO BT SC BR BR BR BR BT SA :       1        1  :         1      16/      0: 0.000  mx/mn/av 0.0001221/     0/1.114e-05  eps:0.0002    
     0022            :           TO BT BR SC BR BT SA :       1        1  :         1      16/      0: 0.000  mx/mn/av 0.0001683/     0/1.737e-05  eps:0.0002    
     0023            :           TO BR SC BT BR BT SA :       1        1  :         1      16/      0: 0.000  mx/mn/av 0.0001373/     0/8.614e-06  eps:0.0002    
     0025            :           TO SC BT BR BR BT SA :       1        1  :         1      16/      0: 0.000  mx/mn/av 0.0001907/     0/1.969e-05  eps:0.0002    
     0028            :           TO BT SC BR BR BT SA :       1        1  :         1      16/      0: 0.000  mx/mn/av 0.0001831/     0/1.821e-05  eps:0.0002    
    c2p : {'seqmat_ana': 0.0, 'pflags_ana': 0.0, 'seqhis_ana': 0.0} c2pmax: 0.0  CUT ok.c2max 2.0  RC:0 
 



full report
--------------

::

    tboolean-;TBOOLEAN_TAG=3 tboolean-box --okg4 --align --dbgskipclearzero --dbgnojumpzero --dbgkludgeflatzero

    2017-12-17 14:54:56.822 INFO  [1207955] [OpticksAna::run@66] OpticksAna::run anakey tboolean enabled Y
    args: /Users/blyth/opticks/ana/tboolean.py --tag 3 --tagoffset 0 --det tboolean-box --src torch
    [2017-12-17 14:54:57,158] p32971 {/Users/blyth/opticks/ana/tboolean.py:62} INFO - tag 3 src torch det tboolean-box c2max 2.0 ipython False 
    [2017-12-17 14:54:57,158] p32971 {/Users/blyth/opticks/ana/ab.py:110} INFO - ab START
    ab.a.metadata:                 /tmp/blyth/opticks/evt/tboolean-box/torch/3 d22bf96e76fc2a4d120fc679faa1f607 c73dd7e7dad8c7e239794d2f2eda381c  100000    -1.0000 INTEROP_MODE 
    AB(3,torch,tboolean-box)  None 0 
    A tboolean-box/torch/  3 :  20171217-1454 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-box/torch/3/fdom.npy () 
    B tboolean-box/torch/ -3 :  20171217-1454 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-box/torch/-3/fdom.npy (recstp) 
    Rock//perfectAbsorbSurface/Vacuum,Vacuum///GlassSchottF2
    /tmp/blyth/opticks/tboolean-box--
    .                seqhis_ana  3:tboolean-box   -3:tboolean-box        c2        ab        ba 
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
    0010         8cbc6ccd         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [8 ] TO BT BT SC BT BR BT SA
    0011             4ccd         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [4 ] TO BT BT AB
    0012          8cc6ccd         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [7 ] TO BT BT SC BT BT SA
    0013           8cbc6d         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [6 ] TO SC BT BR BT SA
    0014               4d         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [2 ] TO AB
    0015           86cbcd         2         2             0.00        1.000 +- 0.707        1.000 +- 0.707  [6 ] TO BT BR BT SC SA
    0016        8cbbc6ccd         2         2             0.00        1.000 +- 0.707        1.000 +- 0.707  [9 ] TO BT BT SC BT BR BR BT SA
    0017           8b6ccd         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [6 ] TO BT BT SC BR SA
    0018           8c6bcd         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [6 ] TO BT BR SC BT SA
    0019        8cbbbb6cd         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [9 ] TO BT SC BR BR BR BR BT SA
    .                             100000    100000         0.00/7 =  0.00  (pval:1.000 prob:0.000)  
    .                pflags_ana  3:tboolean-box   -3:tboolean-box        c2        ab        ba 
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
    .                seqmat_ana  3:tboolean-box   -3:tboolean-box        c2        ab        ba 
    .                             100000    100000         0.00/6 =  0.00  (pval:1.000 prob:0.000)  
    0000             1232     87777     87777             0.00        1.000 +- 0.003        1.000 +- 0.003  [4 ] Vm F2 Vm Rk
    0001              122      6341      6341             0.00        1.000 +- 0.013        1.000 +- 0.013  [3 ] Vm Vm Rk
    0002            12332      5427      5427             0.00        1.000 +- 0.014        1.000 +- 0.014  [5 ] Vm F2 F2 Vm Rk
    0003           123332       350       350             0.00        1.000 +- 0.053        1.000 +- 0.053  [6 ] Vm F2 F2 F2 Vm Rk
    0004          1233332        28        28             0.00        1.000 +- 0.189        1.000 +- 0.189  [7 ] Vm F2 F2 F2 F2 Vm Rk
    0005            12232        24        24             0.00        1.000 +- 0.204        1.000 +- 0.204  [5 ] Vm F2 Vm Vm Rk
    0006              332        16        16             0.00        1.000 +- 0.250        1.000 +- 0.250  [3 ] Vm F2 F2
    0007       3333333332        10        10             0.00        1.000 +- 0.316        1.000 +- 0.316  [10] Vm F2 F2 F2 F2 F2 F2 F2 F2 F2
    0008          1232232         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [7 ] Vm F2 Vm Vm F2 Vm Rk
    0009               22         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [2 ] Vm Vm
    0010             2232         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [4 ] Vm F2 Vm Vm
    0011           123322         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [6 ] Vm Vm F2 F2 Vm Rk
    0012         12332232         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [8 ] Vm F2 Vm Vm F2 F2 Vm Rk
    0013           122332         2         2             0.00        1.000 +- 0.707        1.000 +- 0.707  [6 ] Vm F2 F2 Vm Vm Rk
    0014        123332232         2         2             0.00        1.000 +- 0.707        1.000 +- 0.707  [9 ] Vm F2 Vm Vm F2 F2 F2 Vm Rk
    0015        123333332         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [9 ] Vm F2 F2 F2 F2 F2 F2 Vm Rk
    0016           123222         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [6 ] Vm Vm Vm F2 Vm Rk
    0017            12322         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [5 ] Vm Vm F2 Vm Rk
    0018           122232         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [6 ] Vm F2 Vm Vm Vm Rk
    0019             3332         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [4 ] Vm F2 F2 F2
    .                             100000    100000         0.00/6 =  0.00  (pval:1.000 prob:0.000)  
    ab.a.metadata:                 /tmp/blyth/opticks/evt/tboolean-box/torch/3 d22bf96e76fc2a4d120fc679faa1f607 c73dd7e7dad8c7e239794d2f2eda381c  100000    -1.0000 INTEROP_MODE 
    ab.a.metadata.csgmeta0:{u'containerscale': u'3', u'container': u'1', u'ctrl': u'0', u'verbosity': u'0', u'poly': u'IM', u'emitconfig': u'photons:100000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x1,umin:0.45,umax:0.55,vmin:0.45,vmax:0.55', u'resolution': u'20', u'emit': -1}
    rpost_dv maxdvmax:0.0137638477737 maxdv:[0.013763847773677895, 0.0, 0.0, 0.0, 0.013763847773674343, 0.0, 0.0, 0.0, 0.013763847773674343, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
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
     0010            :        TO BT BT SC BT BR BT SA :       3        3  :         3      96/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0011            :                    TO BT BT AB :       3        3  :         3      48/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0012            :           TO BT BT SC BT BT SA :       3        3  :         3      84/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0013            :              TO SC BT BR BT SA :       3        3  :         3      72/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0014            :                          TO AB :       3        3  :         3      24/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0015            :              TO BT BR BT SC SA :       2        2  :         2      48/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0016            :     TO BT BT SC BT BR BR BT SA :       2        2  :         2      72/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0017            :              TO BT BT SC BR SA :       1        1  :         1      24/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0018            :              TO BT BR SC BT SA :       1        1  :         1      24/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0019            :     TO BT SC BR BR BR BR BT SA :       1        1  :         1      36/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0020            :  TO BT BR SC BR BR BR BR BR BR :       1        1  :         1      40/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0021            :                 TO SC BT BT SA :       1        1  :         1      20/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0022            :           TO BT BR SC BR BT SA :       1        1  :         1      28/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0023            :           TO BR SC BT BR BT SA :       1        1  :         1      28/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0024            :              TO BR SC BT BT SA :       1        1  :         1      24/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0025            :           TO SC BT BR BR BT SA :       1        1  :         1      28/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0026            :                    TO BT BR AB :       1        1  :         1      16/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0027            :                 TO BT BR BR AB :       1        1  :         1      20/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0028            :           TO BT SC BR BR BT SA :       1        1  :         1      28/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
    rpol_dv maxdvmax:0.0 maxdv:[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
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
     0010            :        TO BT BT SC BT BR BT SA :       3        3  :         3      72/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0011            :                    TO BT BT AB :       3        3  :         3      36/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0012            :           TO BT BT SC BT BT SA :       3        3  :         3      63/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0013            :              TO SC BT BR BT SA :       3        3  :         3      54/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0014            :                          TO AB :       3        3  :         3      18/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0015            :              TO BT BR BT SC SA :       2        2  :         2      36/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0016            :     TO BT BT SC BT BR BR BT SA :       2        2  :         2      54/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0017            :              TO BT BT SC BR SA :       1        1  :         1      18/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0018            :              TO BT BR SC BT SA :       1        1  :         1      18/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0019            :     TO BT SC BR BR BR BR BT SA :       1        1  :         1      27/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0020            :  TO BT BR SC BR BR BR BR BR BR :       1        1  :         1      30/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0021            :                 TO SC BT BT SA :       1        1  :         1      15/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0022            :           TO BT BR SC BR BT SA :       1        1  :         1      21/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0023            :           TO BR SC BT BR BT SA :       1        1  :         1      21/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0024            :              TO BR SC BT BT SA :       1        1  :         1      18/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0025            :           TO SC BT BR BR BT SA :       1        1  :         1      21/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0026            :                    TO BT BR AB :       1        1  :         1      12/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0027            :                 TO BT BR BR AB :       1        1  :         1      15/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0028            :           TO BT SC BR BR BT SA :       1        1  :         1      21/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
    /Users/blyth/opticks/ana/dv.py:58: RuntimeWarning: invalid value encountered in greater
      discrep = dv[dv>eps]
    ox_dv maxdvmax:0.000457763671875 maxdv:[5.960464477539063e-08, 1.401298464324817e-45, 5.960464477539063e-08, 5.960464477539063e-08, 0.0002593994140625, 5.960464477539063e-08, 0.000156402587890625, 7.62939453125e-06, 0.00020599365234375, 0.0003662109375, 0.000457763671875, 2.384185791015625e-07, 3.0517578125e-05, 6.103515625e-05, nan, 9.918212890625e-05, 0.0001373291015625, 4.57763671875e-05, 3.0517578125e-05, 0.0001220703125, 6.103515625e-05, 4.76837158203125e-05, 0.00016832351684570312, 0.0001373291015625, 6.103515625e-05, 0.00019073486328125, 2.384185791015625e-07, 7.62939453125e-06, 0.00018310546875] 
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
     0010            :        TO BT BT SC BT BR BT SA :       3        3  :         3      48/      3: 0.062  mx/mn/av 0.0004578/     0/2.935e-05  eps:0.0002    
     0011            :                    TO BT BT AB :       3        3  :         3      48/      0: 0.000  mx/mn/av 2.384e-07/     0/1.366e-08  eps:0.0002    
     0012            :           TO BT BT SC BT BT SA :       3        3  :         3      48/      0: 0.000  mx/mn/av 3.052e-05/     0/2.627e-06  eps:0.0002    
     0013            :              TO SC BT BR BT SA :       3        3  :         3      48/      0: 0.000  mx/mn/av 6.104e-05/     0/4.239e-06  eps:0.0002    
     0014            :                          TO AB :       3        3  :         3      48/      0: 0.000  mx/mn/av    nan/   nan/   nan  eps:0.0002    
     0015            :              TO BT BR BT SC SA :       2        2  :         2      32/      0: 0.000  mx/mn/av 9.918e-05/     0/5.985e-06  eps:0.0002    
     0016            :     TO BT BT SC BT BR BR BT SA :       2        2  :         2      32/      0: 0.000  mx/mn/av 0.0001373/     0/9.584e-06  eps:0.0002    
     0017            :              TO BT BT SC BR SA :       1        1  :         1      16/      0: 0.000  mx/mn/av 4.578e-05/     0/4.216e-06  eps:0.0002    
     0018            :              TO BT BR SC BT SA :       1        1  :         1      16/      0: 0.000  mx/mn/av 3.052e-05/     0/2.253e-06  eps:0.0002    
     0019            :     TO BT SC BR BR BR BR BT SA :       1        1  :         1      16/      0: 0.000  mx/mn/av 0.0001221/     0/1.114e-05  eps:0.0002    
     0020            :  TO BT BR SC BR BR BR BR BR BR :       1        1  :         1      16/      0: 0.000  mx/mn/av 6.104e-05/     0/3.823e-06  eps:0.0002    
     0021            :                 TO SC BT BT SA :       1        1  :         1      16/      0: 0.000  mx/mn/av 4.768e-05/     0/6.839e-06  eps:0.0002    
     0022            :           TO BT BR SC BR BT SA :       1        1  :         1      16/      0: 0.000  mx/mn/av 0.0001683/     0/1.737e-05  eps:0.0002    
     0023            :           TO BR SC BT BR BT SA :       1        1  :         1      16/      0: 0.000  mx/mn/av 0.0001373/     0/8.614e-06  eps:0.0002    
     0024            :              TO BR SC BT BT SA :       1        1  :         1      16/      0: 0.000  mx/mn/av 6.104e-05/     0/9.572e-06  eps:0.0002    
     0025            :           TO SC BT BR BR BT SA :       1        1  :         1      16/      0: 0.000  mx/mn/av 0.0001907/     0/1.969e-05  eps:0.0002    
     0026            :                    TO BT BR AB :       1        1  :         1      16/      0: 0.000  mx/mn/av 2.384e-07/     0/1.49e-08  eps:0.0002    
     0027            :                 TO BT BR BR AB :       1        1  :         1      16/      0: 0.000  mx/mn/av 7.629e-06/     0/5.066e-07  eps:0.0002    
     0028            :           TO BT SC BR BR BT SA :       1        1  :         1      16/      0: 0.000  mx/mn/av 0.0001831/     0/1.821e-05  eps:0.0002    
    c2p : {'seqmat_ana': 0.0, 'pflags_ana': 0.0, 'seqhis_ana': 0.0} c2pmax: 0.0  CUT ok.c2max 2.0  RC:0 
    rmxs_ : {'rpol_dv': 0.0, 'rpost_dv': 0.013763847773677895} rmxs_max_: 0.0137638477737  CUT ok.rdvmax 0.1  RC:0 
    pmxs_ : {'ox_dv': 0.000457763671875} pmxs_max_: 0.000457763671875  CUT ok.pdvmax 0.001  RC:0 
    [2017-12-17 14:54:58,055] p32971 {/Users/blyth/opticks/ana/tboolean.py:70} INFO - early exit as non-interactive
    2017-12-17 14:54:58.092 INFO  [1207955] [SSys::run@50] tboolean.py --tag 3 --tagoffset 0 --det tboolean-box --src torch   rc_raw : 0 rc : 0
    2017-12-17 14:54:58.093 INFO  [1207955] [OpticksAna::run@81] OpticksAna::run anakey tboolean cmdline tboolean.py --tag 3 --tagoffset 0 --det tboolean-box --src torch   interactivity 2 rc 0 rcmsg -
    2017-12-17 14:54:58.093 INFO  [1207955] [SSys::WaitForInput@151] SSys::WaitForInput OpticksAna::run paused : hit RETURN to continue...


