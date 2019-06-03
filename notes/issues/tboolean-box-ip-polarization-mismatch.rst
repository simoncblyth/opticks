tboolean-box-ip-polarization-mismatch FIXED bug in CInputPhotonSource
========================================================================

* context :doc:`tboolean-resurrection` 


issue : Non-aligned history matching looks ok, but polarizations totally off.
---------------------------------------------------------------------------------

::

    [blyth@localhost tmp]$ tboolean-box-ip
    Python 2.7.15 |Anaconda, Inc.| (default, May  1 2018, 23:32:55) 
    Type "copyright", "credits" or "license" for more information.

    IPython 5.7.0 -- An enhanced Interactive Python.
    ?         -> Introduction and overview of IPython's features.
    %quickref -> Quick reference.
    help      -> Python's own help system.
    object?   -> Details about 'object', use 'object??' for extra details.
    args: /home/blyth/opticks/ana/tboolean.py --det tboolean-box --tag 1
    [2019-06-02 22:52:28,482] p83507 {/home/blyth/opticks/ana/base.py:259} WARNING - legacy_init : OPTICKS_KEY envvar deleted for legacy running, unset IDPATH to use direct_init
    [2019-06-02 22:52:28,483] p83507 {/home/blyth/opticks/ana/tboolean.py:63} INFO - pfx . tag 1 src torch det tboolean-box c2max 2.0 ipython True 
    [2019-06-02 22:52:28,483] p83507 {/home/blyth/opticks/ana/ab.py:110} INFO - ab START
    [2019-06-02 22:52:28,483] p83507 {/home/blyth/opticks/ana/evt.py:303} INFO - loaded metadata from tboolean-box/./evt/tboolean-box/torch/1 
    [2019-06-02 22:52:28,484] p83507 {/home/blyth/opticks/ana/evt.py:304} INFO - metadata                      tboolean-box/./evt/tboolean-box/torch/1 1362486dd26c8761b615260e66bbcdb5 481c2dd37d4d0c5641ef2411a6cdac12  100000    -1.0000 COMPUTE_MODE  
    [2019-06-02 22:52:28,485] p83507 {/home/blyth/opticks/ana/base.py:709} INFO - txt GMaterialLib reldir  /tmp/tboolean-box/GItemList 
    [2019-06-02 22:52:28,585] p83507 {/home/blyth/opticks/ana/evt.py:303} INFO - loaded metadata from tboolean-box/./evt/tboolean-box/torch/-1 
    [2019-06-02 22:52:28,585] p83507 {/home/blyth/opticks/ana/evt.py:304} INFO - metadata                     tboolean-box/./evt/tboolean-box/torch/-1 8d873be21dd0936ff3aba7604cbcedd5 c77ce477d608f6186283c16a2939190d  100000    -1.0000 COMPUTE_MODE  
    [2019-06-02 22:52:28,586] p83507 {/home/blyth/opticks/ana/base.py:709} INFO - txt GMaterialLib reldir  /tmp/tboolean-box/GItemList 
    [2019-06-02 22:52:28,694] p83507 {/home/blyth/opticks/ana/seq.py:284} INFO -  c2sum 6.021615496489548 ndf 7 c2p 0.8602307852127925 c2_pval 0.5372276976947495 
    [2019-06-02 22:52:28,695] p83507 {/home/blyth/opticks/ana/seq.py:284} INFO -  c2sum 5.021453892508648 ndf 6 c2p 0.8369089820847746 c2_pval 0.5410644529781723 
    ab.a.metadata:                     tboolean-box/./evt/tboolean-box/torch/1 1362486dd26c8761b615260e66bbcdb5 481c2dd37d4d0c5641ef2411a6cdac12  100000    -1.0000 COMPUTE_MODE 
    [2019-06-02 22:52:28,700] p83507 {/home/blyth/opticks/ana/seq.py:284} INFO -  c2sum 6.021615496489548 ndf 7 c2p 0.8602307852127925 c2_pval 0.5372276976947495 
    [2019-06-02 22:52:28,702] p83507 {/home/blyth/opticks/ana/seq.py:284} INFO -  c2sum 5.021453892508648 ndf 6 c2p 0.8369089820847746 c2_pval 0.5410644529781723 
    [2019-06-02 22:52:28,703] p83507 {/home/blyth/opticks/ana/seq.py:284} INFO -  c2sum 7.17732761606031 ndf 5 c2p 1.4354655232120621 c2_pval 0.20778275552986425 
    AB(1,torch,tboolean-box)  None 0 
    A ./tboolean-box/torch/  1 :  20190602-2208 maxbounce:9 maxrec:10 maxrng:3000000 tboolean-box/./evt/tboolean-box/torch/1/fdom.npy () 
    B ./tboolean-box/torch/ -1 :  20190602-2208 maxbounce:9 maxrec:10 maxrng:3000000 tboolean-box/./evt/tboolean-box/torch/-1/fdom.npy (recstp) 
    Rock//perfectAbsorbSurface/Vacuum,Vacuum///GlassSchottF2
    tboolean-box
    .                seqhis_ana  1:tboolean-box:.   -1:tboolean-box:.        c2        ab        ba 
    .                             100000    100000         6.02/7 =  0.86  (pval:0.537 prob:0.463)  
    0000             8ccd     87777     87920             0.12        0.998 +- 0.003        1.002 +- 0.003  [4 ] TO BT BT SA
    0001              8bd      6312      6107             3.38        1.034 +- 0.013        0.968 +- 0.012  [3 ] TO BR SA
    0002            8cbcd      5420      5463             0.17        0.992 +- 0.013        1.008 +- 0.014  [5 ] TO BT BR BT SA
    0003           8cbbcd       349       360             0.17        0.969 +- 0.052        1.032 +- 0.054  [6 ] TO BT BR BR BT SA
    0004              86d        29        37             0.97        0.784 +- 0.146        1.276 +- 0.210  [3 ] TO SC SA
    0005          8cbbbcd        26        23             0.18        1.130 +- 0.222        0.885 +- 0.184  [7 ] TO BT BR BR BR BT SA
    0006            86ccd        24        26             0.08        0.923 +- 0.188        1.083 +- 0.212  [5 ] TO BT BT SC SA
    0007              4cd        16        22             0.95        0.727 +- 0.182        1.375 +- 0.293  [3 ] TO BT AB
    0008       bbbbbbb6cd         9         6             0.00        1.500 +- 0.500        0.667 +- 0.272  [10] TO BT SC BR BR BR BR BR BR BR
    0009            8c6cd         7         4             0.00        1.750 +- 0.661        0.571 +- 0.286  [5 ] TO BT SC BT SA
    0010         8cbc6ccd         3         4             0.00        0.750 +- 0.433        1.333 +- 0.667  [8 ] TO BT BT SC BT BR BT SA
    0011             4ccd         3         8             0.00        0.375 +- 0.217        2.667 +- 0.943  [4 ] TO BT BT AB
    0012           8cbc6d         3         2             0.00        1.500 +- 0.866        0.667 +- 0.471  [6 ] TO SC BT BR BT SA
    0013               4d         3         4             0.00        0.750 +- 0.433        1.333 +- 0.667  [2 ] TO AB
    0014           86cbcd         2         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [6 ] TO BT BR BT SC SA
    0015          8cc6ccd         2         2             0.00        1.000 +- 0.707        1.000 +- 0.707  [7 ] TO BT BT SC BT BT SA
    0016        8cbbc6ccd         2         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [9 ] TO BT BT SC BT BR BR BT SA
    0017           8b6ccd         1         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [6 ] TO BT BT SC BR SA
    0018           8c6bcd         1         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [6 ] TO BT BR SC BT SA
    0019        8cbbbb6cd         1         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [9 ] TO BT SC BR BR BR BR BT SA
    .                             100000    100000         6.02/7 =  0.86  (pval:0.537 prob:0.463)  
    .                pflags_ana  1:tboolean-box:.   -1:tboolean-box:.        c2        ab        ba 
    .                             100000    100000         7.18/5 =  1.44  (pval:0.208 prob:0.792)  
    0000             1880     87777     87920             0.12        0.998 +- 0.003        1.002 +- 0.003  [3 ] TO|BT|SA
    0001             1480      6312      6107             3.38        1.034 +- 0.013        0.968 +- 0.012  [3 ] TO|BR|SA
    0002             1c80      5795      5846             0.22        0.991 +- 0.013        1.009 +- 0.013  [4 ] TO|BT|BR|SA
    0003             18a0        35        34             0.01        1.029 +- 0.174        0.971 +- 0.167  [4 ] TO|BT|SA|SC
    0004             10a0        29        37             0.97        0.784 +- 0.146        1.276 +- 0.210  [3 ] TO|SA|SC
    0005             1808        19        30             2.47        0.633 +- 0.145        1.579 +- 0.288  [3 ] TO|BT|AB
    0006             1ca0        18        12             0.00        1.500 +- 0.354        0.667 +- 0.192  [5 ] TO|BT|BR|SA|SC
    0007             1c20        10         7             0.00        1.429 +- 0.452        0.700 +- 0.265  [4 ] TO|BT|BR|SC
    0008             1008         3         4             0.00        0.750 +- 0.433        1.333 +- 0.667  [2 ] TO|AB
    0009             1c08         2         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [4 ] TO|BT|BR|AB
    0010             14a0         0         2             0.00        0.000 +- 0.000        0.000 +- 0.000  [4 ] TO|BR|SA|SC
    0011             1408         0         1             0.00        0.000 +- 0.000        0.000 +- 0.000  [3 ] TO|BR|AB
    .                             100000    100000         7.18/5 =  1.44  (pval:0.208 prob:0.792)  
    .                seqmat_ana  1:tboolean-box:.   -1:tboolean-box:.        c2        ab        ba 
    .                             100000    100000         5.02/6 =  0.84  (pval:0.541 prob:0.459)  
    0000             1232     87777     87920             0.12        0.998 +- 0.003        1.002 +- 0.003  [4 ] Vm G2 Vm Rk
    0001              122      6341      6144             3.11        1.032 +- 0.013        0.969 +- 0.012  [3 ] Vm Vm Rk
    0002            12332      5427      5467             0.15        0.993 +- 0.013        1.007 +- 0.014  [5 ] Vm G2 G2 Vm Rk
    0003           123332       350       365             0.31        0.959 +- 0.051        1.043 +- 0.055  [6 ] Vm G2 G2 G2 Vm Rk
    0004          1233332        28        24             0.31        1.167 +- 0.220        0.857 +- 0.175  [7 ] Vm G2 G2 G2 G2 Vm Rk
    0005            12232        24        26             0.08        0.923 +- 0.188        1.083 +- 0.212  [5 ] Vm G2 Vm Vm Rk
    0006              332        16        22             0.95        0.727 +- 0.182        1.375 +- 0.293  [3 ] Vm G2 G2
    0007       3333333332        10         7             0.00        1.429 +- 0.452        0.700 +- 0.265  [10] Vm G2 G2 G2 G2 G2 G2 G2 G2 G2
    0008             2232         3         8             0.00        0.375 +- 0.217        2.667 +- 0.943  [4 ] Vm G2 Vm Vm
    0009               22         3         4             0.00        0.750 +- 0.433        1.333 +- 0.667  [2 ] Vm Vm
    0010         12332232         3         4             0.00        0.750 +- 0.433        1.333 +- 0.667  [8 ] Vm G2 Vm Vm G2 G2 Vm Rk
    0011           123322         3         2             0.00        1.500 +- 0.866        0.667 +- 0.471  [6 ] Vm Vm G2 G2 Vm Rk
    0012          1232232         2         2             0.00        1.000 +- 0.707        1.000 +- 0.707  [7 ] Vm G2 Vm Vm G2 Vm Rk
    0013           122332         2         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [6 ] Vm G2 G2 Vm Vm Rk
    0014        123332232         2         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [9 ] Vm G2 Vm Vm G2 G2 G2 Vm Rk
    0015           123222         1         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [6 ] Vm Vm Vm G2 Vm Rk
    0016            33332         1         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] Vm G2 G2 G2 G2
    0017             3332         1         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [4 ] Vm G2 G2 G2
    0018            12322         1         2             0.00        0.500 +- 0.500        2.000 +- 1.414  [5 ] Vm Vm G2 Vm Rk
    0019           122232         1         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [6 ] Vm G2 Vm Vm Vm Rk
    .                             100000    100000         5.02/6 =  0.84  (pval:0.541 prob:0.459)  
    ab.a.metadata:                     tboolean-box/./evt/tboolean-box/torch/1 1362486dd26c8761b615260e66bbcdb5 481c2dd37d4d0c5641ef2411a6cdac12  100000    -1.0000 COMPUTE_MODE 
    ab.a.metadata.csgmeta0:{u'containerscale': 3.0, u'container': 1, u'ctrl': 0, u'verbosity': u'0', u'poly': u'IM', u'emitconfig': u'photons:100000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x1,umin:0.45,umax:0.55,vmin:0.45,vmax:0.55', u'resolution': u'20', u'emit': -1}
    rpost_dv maxdvmax:0.0137638477737 maxdv:[0.013763847773677895, 0.0, 0.0, 0.0] 
      idx        msg :                            sel :    lcu1     lcu2  :     nitem   nelem/  ndisc: fdisc  mx/mn/av     mx/    mn/   avg  eps:eps    
     0000            :                    TO BT BT SA :   87777    87920  :     77162 1234592/     44: 0.000  mx/mn/av 0.01376/     0/4.905e-07  eps:0.0002    
     0001            :                       TO BR SA :    6312     6107  :       374    4488/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0002            :                 TO BT BR BT SA :    5420     5463  :       305    6100/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0003            :              TO BT BR BR BT SA :     349      360  :         1      24/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    


    rpol_dv maxdvmax:2.00787401199 maxdv:[2.007874011993408, 1.5511810779571533, 1.629921317100525, 1.0] 
      idx        msg :                            sel :    lcu1     lcu2  :     nitem   nelem/  ndisc: fdisc  mx/mn/av     mx/    mn/   avg  eps:eps    
     0000            :                    TO BT BT SA :   87777    87920  :     77162  925944/ 841073: 0.908  mx/mn/av  2.008/     0/0.6792  eps:0.0002    
     0001            :                       TO BR SA :    6312     6107  :       374    3366/   2978: 0.885  mx/mn/av  1.551/     0/0.6794  eps:0.0002    
     0002            :                 TO BT BR BT SA :    5420     5463  :       305    4575/   4225: 0.923  mx/mn/av   1.63/     0/0.6784  eps:0.0002    
     0003            :              TO BT BR BR BT SA :     349      360  :         1      18/     17: 0.944  mx/mn/av      1/     0/0.6601  eps:0.0002    
    ox_dv maxdvmax:1.09949862957 maxdv:[1.0994986295700073, 1.0990633964538574, 1.0989869832992554, 0.9956643581390381] 
      idx        msg :                            sel :    lcu1     lcu2  :     nitem   nelem/  ndisc: fdisc  mx/mn/av     mx/    mn/   avg  eps:eps    
     0000            :                    TO BT BT SA :   87777    87920  :     77162  925944/ 231336: 0.250  mx/mn/av  1.099/     0/0.1705  eps:0.0002    
     0001            :                       TO BR SA :    6312     6107  :       374    4488/   1122: 0.250  mx/mn/av  1.099/     0/0.1707  eps:0.0002    
     0002            :                 TO BT BR BT SA :    5420     5463  :       305    3660/    914: 0.250  mx/mn/av  1.099/     0/0.1703  eps:0.0002    
     0003            :              TO BT BR BR BT SA :     349      360  :         1      12/      3: 0.250  mx/mn/av 0.9957/     0/0.1649  eps:0.0002    
    c2p : {'seqmat_ana': 0.8369089820847746, 'pflags_ana': 1.4354655232120621, 'seqhis_ana': 0.8602307852127925} c2pmax: 1.4354655232120621  CUT ok.c2max 2.0  RC:0 
    rmxs_ : {'rpol_dv': 2.007874011993408, 'rpost_dv': 0.013763847773677895} rmxs_max_: 2.00787401199  CUT ok.rdvmax 0.1  RC:88 
    pmxs_ : {'ox_dv': 1.0994986295700073} pmxs_max_: 1.09949862957  CUT ok.pdvmax 0.001  RC:99 


    In [1]: a.polw[:10]
    Out[1]: 
    A()sliced
    A([[  0.,  -1.,   0., 380.],
       [  0.,  -1.,   0., 380.],
       [ -0.,   1.,  -0., 380.],
       [  0.,  -1.,   0., 380.],
       [  0.,  -1.,   0., 380.],
       [ -0.,   1.,  -0., 380.],
       [  0.,  -1.,   0., 380.],
       [ -0.,   1.,  -0., 380.],
       [  0.,  -1.,   0., 380.],
       [  0.,  -1.,   0., 380.]], dtype=float32)

    In [2]: b.polw[:10]
    Out[2]: 
    A()sliced
    A([[  0.025 ,  -0.0768,  -0.9967, 380.    ],
       [ -0.0592,  -0.0006,  -0.9982, 380.    ],
       [ -0.0898,   0.0217,  -0.9957, 380.    ],
       [  0.063 ,  -0.079 ,  -0.9949, 380.    ],
       [ -0.0462,  -0.0731,  -0.9963, 380.    ],
       [ -0.0557,  -0.07  ,  -0.996 , 380.    ],
       [ -0.0197,  -0.0518,  -0.9985, 380.    ],
       [ -0.0193,  -0.0654,  -0.9977, 380.    ],
       [  0.0686,  -0.0345,  -0.997 , 380.    ],
       [  0.0108,   0.0621,  -0.998 , 380.    ]], dtype=float32)



After fixed bug in CInputPhotonSource, the deviations are back where they should be
---------------------------------------------------------------------------------------

::

    ab.a.metadata:                     tboolean-box/./evt/tboolean-box/torch/1 1362486dd26c8761b615260e66bbcdb5 481c2dd37d4d0c5641ef2411a6cdac12  100000    -1.0000 COMPUTE_MODE 
    ab.a.metadata.csgmeta0:{u'containerscale': 3.0, u'container': 1, u'ctrl': 0, u'verbosity': u'0', u'poly': u'IM', u'emitconfig': u'photons:100000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x1,umin:0.45,umax:0.55,vmin:0.45,vmax:0.55', u'resolution': u'20', u'emit': -1}
    rpost_dv maxdvmax:0.0137638477737 maxdv:[0.013763847773677895, 0.0, 0.0] 
      idx        msg :                            sel :    lcu1     lcu2  :     nitem   nelem/  ndisc: fdisc  mx/mn/av     mx/    mn/   avg  eps:eps    
     0000            :                    TO BT BT SA :   87777    87940  :     77193 1235088/     44: 0.000  mx/mn/av 0.01376/     0/4.903e-07  eps:0.0002    
     0001            :                       TO BR SA :    6312     6069  :       399    4788/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0002            :                 TO BT BR BT SA :    5420     5478  :       295    5900/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
    rpol_dv maxdvmax:0.0 maxdv:[0.0, 0.0, 0.0] 
      idx        msg :                            sel :    lcu1     lcu2  :     nitem   nelem/  ndisc: fdisc  mx/mn/av     mx/    mn/   avg  eps:eps    
     0000            :                    TO BT BT SA :   87777    87940  :     77193  926316/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0001            :                       TO BR SA :    6312     6069  :       399    3591/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0002            :                 TO BT BR BT SA :    5420     5478  :       295    4425/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
    ox_dv maxdvmax:4.76837158203e-07 maxdv:[2.384185791015625e-07, 0.0, 4.76837158203125e-07] 
      idx        msg :                            sel :    lcu1     lcu2  :     nitem   nelem/  ndisc: fdisc  mx/mn/av     mx/    mn/   avg  eps:eps    
     0000            :                    TO BT BT SA :   87777    87940  :     77193  926316/      0: 0.000  mx/mn/av 2.384e-07/     0/2.484e-08  eps:0.0002    
     0001            :                       TO BR SA :    6312     6069  :       399    4788/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0002            :                 TO BT BR BT SA :    5420     5478  :       295    3540/      0: 0.000  mx/mn/av 4.768e-07/     0/4.47e-08  eps:0.0002    
    c2p : {'seqmat_ana': 1.1208101425498735, 'pflags_ana': 1.6975355284382387, 'seqhis_ana': 1.0496205532998364} c2pmax: 1.6975355284382387  CUT ok.c2max 2.0  RC:0 
    rmxs_ : {'rpol_dv': 0.0, 'rpost_dv': 0.013763847773677895} rmxs_max_: 0.0137638477737  CUT ok.rdvmax 0.1  RC:0 
    pmxs_ : {'ox_dv': 4.76837158203125e-07} pmxs_max_: 4.76837158203e-07  CUT ok.pdvmax 0.001  RC:0 







trace polarization for emitconfig input photons 
-------------------------------------------------

::

    [blyth@localhost opticks]$ tboolean-box--
    import logging
    log = logging.getLogger(__name__)
    from opticks.ana.base import opticks_main
    from opticks.analytic.polyconfig import PolyConfig
    from opticks.analytic.csg import CSG  

    # 0x3f is all 6 
    autoemitconfig="photons:600000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x1,umin:0.45,umax:0.55,vmin:0.45,vmax:0.55,diffuse:1,ctmindiffuse:0.5,ctmaxdiffuse:1.0"
    args = opticks_main(csgpath="tboolean-box", autoemitconfig=autoemitconfig)

    #emitconfig = "photons:100000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x1,umin:0.25,umax:0.75,vmin:0.25,vmax:0.75" 
    #emitconfig = "photons:1,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x1,umin:0.25,umax:0.75,vmin:0.25,vmax:0.75" 
    emitconfig = "photons:100000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x1,umin:0.45,umax:0.55,vmin:0.45,vmax:0.55" 

    CSG.kwa = dict(poly="IM",resolution="20", verbosity="0", ctrl=0, containerscale=3.0, emitconfig=emitconfig  )

    container = CSG("box", emit=-1, boundary='Rock//perfectAbsorbSurface/Vacuum', container=1 )  # no param, container="1" switches on auto-sizing

    box = CSG("box3", param=[300,300,200,0], emit=0,  boundary="Vacuum///GlassSchottF2" )

    CSG.Serialize([container, box], args )
    [blyth@localhost opticks]$ 



* emitconfig sheetmask:0x1 picks the sheets (ie sides) of the box to be emissive and the (umin umax vmin vmax) 
  is the (u,v) size of the emissive patch in range (0:1,0:1) 
* emit=-1 on the container means directed opposite to the outwards normal, ie inwards
* *NEmitConfig* parses the emitconfig string 
* *NEmitPhotonsNPY::init* 

::

     20 glm::vec3 nglmext::least_parallel_axis( const glm::vec3& dir )
     21 {
     22     glm::vec3 adir(glm::abs(dir));
     23     glm::vec3 lpa(0) ;
     24 
     25     if( adir.x <= adir.y && adir.x <= adir.z )
     26     {
     27         lpa.x = 1.f ;
     28     }
     29     else if( adir.y <= adir.x && adir.y <= adir.z )
     30     {
     31         lpa.y = 1.f ;
     32     }
     33     else
     34     {
     35         lpa.z = 1.f ;
     36     }
     37     return lpa ;
     38 }
     39 
     40 glm::vec3 nglmext::pick_transverse_direction( const glm::vec3& dir, bool dump)
     41 {
     42     glm::vec3 lpa = least_parallel_axis(dir) ;
     43     glm::vec3 trd = glm::normalize( glm::cross( lpa, dir )) ;
     44 
     45     if(dump)
     46     {
     47         std::cout
     48                   << "nglext::pick_transverse_direction"
     49                   << " dir " << gpresent(dir)
     50                   << " lpa " << gpresent(lpa)
     51                   << " trd " << gpresent(trd)
     52                   << std::endl
     53                   ;
     54     }
     55     return trd ;
     56 }



::

    nglext::pick_transverse_direction dir (     -0.000    -0.000     1.000) lpa (      1.000     0.000     0.000) trd (      0.000    -1.000     0.000)
    nglext::pick_transverse_direction dir (     -0.000    -0.000     1.000) lpa (      1.000     0.000     0.000) trd (      0.000    -1.000     0.000)
    nglext::pick_transverse_direction dir (     -0.000    -0.000     1.000) lpa (      1.000     0.000     0.000) trd (      0.000    -1.000     0.000)
    nglext::pick_transverse_direction dir (     -0.000    -0.000     1.000) lpa (      1.000     0.000     0.000) trd (      0.000    -1.000     0.000)
    nglext::pick_transverse_direction dir (     -0.000    -0.000     1.000) lpa (      1.000     0.000     0.000) trd (      0.000    -1.000     0.000)
     i      0 pos (     11.291   -34.645  -449.900) nrm (      0.000     0.000    -1.000) dir (     -0.000    -0.000     1.000) pol (      0.000    -1.000     0.000)
     i      1 pos (    -26.689    -0.283  -449.900) nrm (      0.000     0.000    -1.000) dir (     -0.000    -0.000     1.000) pol (      0.000    -1.000     0.000)
     i      2 pos (    -40.564     9.816  -449.900) nrm (      0.000     0.000    -1.000) dir (     -0.000    -0.000     1.000) pol (      0.000    -1.000     0.000)
     i      3 pos (     28.491   -35.738  -449.900) nrm (      0.000     0.000    -1.000) dir (     -0.000    -0.000     1.000) pol (      0.000    -1.000     0.000)
     i      4 pos (    -20.879   -32.995  -449.900) nrm (      0.000     0.000    -1.000) dir (     -0.000    -0.000     1.000) pol (      0.000    -1.000     0.000)

    // bottom face of box, normal -Z downwards, pol in -Y : as reported in OpticksEvent

::

    2019-06-03 10:31:06.140 INFO  [287782] [NEmitPhotonsNPY::init@165]  i      0 pos (     11.291   -34.645  -449.900) nrm (      0.000     0.000    -1.000) dir (     -0.000    -0.000     1.000) pol (      0.000    -1.000     0.000) posnrm (      0.025    -0.077    -0.997)
    2019-06-03 10:31:06.140 INFO  [287782] [NEmitPhotonsNPY::init@165]  i      1 pos (    -26.689    -0.283  -449.900) nrm (      0.000     0.000    -1.000) dir (     -0.000    -0.000     1.000) pol (      0.000    -1.000     0.000) posnrm (     -0.059    -0.001    -0.998)
    2019-06-03 10:31:06.141 INFO  [287782] [NEmitPhotonsNPY::init@165]  i      2 pos (    -40.564     9.816  -449.900) nrm (      0.000     0.000    -1.000) dir (     -0.000    -0.000     1.000) pol (      0.000    -1.000     0.000) posnrm (     -0.090     0.022    -0.996)
    2019-06-03 10:31:06.141 INFO  [287782] [NEmitPhotonsNPY::init@165]  i      3 pos (     28.491   -35.738  -449.900) nrm (      0.000     0.000    -1.000) dir (     -0.000    -0.000     1.000) pol (      0.000    -1.000     0.000) posnrm (      0.063    -0.079    -0.995)
    2019-06-03 10:31:06.141 INFO  [287782] [NEmitPhotonsNPY::init@165]  i      4 pos (    -20.879   -32.995  -449.900) nrm (      0.000     0.000    -1.000) dir (     -0.000    -0.000     1.000) pol (      0.000    -1.000     0.000) posnrm (     -0.046    -0.073    -0.996)
    2019-06-03 10:31:06.141 INFO  [287782] [NEmitPhotonsNPY::init@165]  i      5 pos (    -25.172   -31.610  -449.900) nrm (      0.000     0.000    -1.000) dir (     -0.000    -0.000     1.000) pol (      0.000    -1.000     0.000) posnrm (     -0.056    -0.070    -0.996)
    2019-06-03 10:31:06.141 INFO  [287782] [NEmitPhotonsNPY::init@165]  i      6 pos (     -8.879   -23.347  -449.900) nrm (      0.000     0.000    -1.000) dir (     -0.000    -0.000     1.000) pol (      0.000    -1.000     0.000) posnrm (     -0.020    -0.052    -0.998)
    2019-06-03 10:31:06.141 INFO  [287782] [NEmitPhotonsNPY::init@165]  i      7 pos (     -8.717   -29.508  -449.900) nrm (      0.000     0.000    -1.000) dir (     -0.000    -0.000     1.000) pol (      0.000    -1.000     0.000) posnrm (     -0.019    -0.065    -0.998)
    2019-06-03 10:31:06.141 INFO  [287782] [NEmitPhotonsNPY::init@165]  i      8 pos (     30.958   -15.557  -449.900) nrm (      0.000     0.000    -1.000) dir (     -0.000    -0.000     1.000) pol (      0.000    -1.000     0.000) posnrm (      0.069    -0.034    -0.997)
    2019-06-03 10:31:06.142 INFO  [287782] [NEmitPhotonsNPY::init@165]  i      9 pos (      4.875    27.993  -449.900) nrm (      0.000     0.000    -1.000) dir (     -0.000    -0.000     1.000) pol (      0.000    -1.000     0.000) posnrm (      0.011     0.062    -0.998)
    2019-06-03 10:31:06.236 ERROR [287782] [OpticksGen::setInputPhotons@277] OpticksGen::setInputPhotons ox 100000,4,4 ox.hasMsk N



Hmm bizarre, G4 reported pol are posnrm (normalized position) : that looks like a to be implemented placeholder::

    In [2]: b.polw[:10]
    Out[2]: 
    A()sliced
    A([[  0.025 ,  -0.0768,  -0.9967, 380.    ],
       [ -0.0592,  -0.0006,  -0.9982, 380.    ],
       [ -0.0898,   0.0217,  -0.9957, 380.    ],
       [  0.063 ,  -0.079 ,  -0.9949, 380.    ],
       [ -0.0462,  -0.0731,  -0.9963, 380.    ],
       [ -0.0557,  -0.07  ,  -0.996 , 380.    ],
       [ -0.0197,  -0.0518,  -0.9985, 380.    ],
       [ -0.0193,  -0.0654,  -0.9977, 380.    ],
       [  0.0686,  -0.0345,  -0.997 , 380.    ],
       [  0.0108,   0.0621,  -0.998 , 380.    ]], dtype=float32)



FIXED polarization bug in cfg4/CInputPhotonSource.cc


