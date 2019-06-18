tboolean_box_perfect_alignment_revisted
============================================

Context
----------

* :doc:`tboolean_box_perfect_alignment`

Try to get this going again



analysis commandline
-----------------------

::

    tboolean-;tboolean-box-ip --tag 3

    tboolean-box-ip () {  OPTICKS_EVENT_BASE=/tmp TESTNAME=${FUNCNAME/-ip} tboolean-ipy- $* ; }
    tboolean-ipy- () {    ipython --pdb $(which tboolean.py) -i -- --det $TESTNAME --pfx $TESTNAME --tag $(tboolean-tag) $* ; } 


grab an old commandline
----------------------------

::

    tboolean-;TBOOLEAN_TAG=3 tboolean-box --okg4 --align --dbgskipclearzero --dbgnojumpzero --dbgkludgeflatzero

Works first time, thats unusual

* comparing with the old one can see that need to switch off the skips for aligned running. 
* how to auto determine aligned running from metadata ?


::

    2019-06-18 14:31:59.894 INFO  [447362] [OpticksEventAna::dump@232] GGeoTest::anaEvent OpticksEventAna pho 100000,4,4 seq 100000,1,2
    2019-06-18 14:31:59.894 INFO  [447362] [OpticksAna::run@70]  anakey tboolean enabled Y

    args: /home/blyth/opticks/ana/tboolean.py --tagoffset 0 --tag 3 --det tboolean-box --pfx tboolean-box --src torch
    [2019-06-18 14:32:00,729] p448493 {tboolean.py:63} INFO     - pfx tboolean-box tag 3 src torch det tboolean-box c2max 2.0 ipython False 
    ab.a.metadata:[/tmp/tboolean-box/evt/tboolean-box/torch/3                   ox:963f84ba3a1a6c894054ebbcf0fb1ea9 rx:3db691ffd21dfa48c062c0f19bb0fdb0 np: 100000 pr:    0.0078 INTEROP_MODE]
    ab.b.metadata:[/tmp/tboolean-box/evt/tboolean-box/torch/-3                  ox:dfa123a7382e32bfebb7eb741ccaa749 rx:e1c46ce4b32c1c7e00f1378e807aa972 np: 100000 pr:   26.8359 INTEROP_MODE]
    AB(3,torch,tboolean-box)  None 0 
    A tboolean-box/tboolean-box/torch/  3 :  20190618-1431 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/tboolean-box/evt/tboolean-box/torch/3/fdom.npy () 
    B tboolean-box/tboolean-box/torch/ -3 :  20190618-1431 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/tboolean-box/evt/tboolean-box/torch/-3/fdom.npy (recstp) 
    Rock//perfectAbsorbSurface/Vacuum,Vacuum///GlassSchottF2
    tboolean-box
    .                seqhis_ana  3:tboolean-box:tboolean-box   -3:tboolean-box:tboolean-box        c2        ab        ba 
    .                             100000    100000         0.00/7 =  0.00  (pval:1.000 prob:0.000)  
    0000             8ccd     87782     87782             0.00        1.000 +- 0.003        1.000 +- 0.003  [4 ] TO BT BT SA
    0001              8bd      6313      6313             0.00        1.000 +- 0.013        1.000 +- 0.013  [3 ] TO BR SA
    0002            8cbcd      5420      5420             0.00        1.000 +- 0.014        1.000 +- 0.014  [5 ] TO BT BR BT SA
    0003           8cbbcd       349       349             0.00        1.000 +- 0.054        1.000 +- 0.054  [6 ] TO BT BR BR BT SA
    0004              86d        29        29             0.00        1.000 +- 0.186        1.000 +- 0.186  [3 ] TO SC SA
    0005          8cbbbcd        26        26             0.00        1.000 +- 0.196        1.000 +- 0.196  [7 ] TO BT BR BR BR BT SA
    0006            86ccd        24        24             0.00        1.000 +- 0.204        1.000 +- 0.204  [5 ] TO BT BT SC SA
    0007              4cd        16        16             0.00        1.000 +- 0.250        1.000 +- 0.250  [3 ] TO BT AB
    0008       bbbbbbb6cd         9         9             0.00        1.000 +- 0.333        1.000 +- 0.333  [10] TO BT SC BR BR BR BR BR BR BR
    0009            8c6cd         7         7             0.00        1.000 +- 0.378        1.000 +- 0.378  [5 ] TO BT SC BT SA
    0010         8cbc6ccd         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [8 ] TO BT BT SC BT BR BT SA
    0011           8cbc6d         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [6 ] TO SC BT BR BT SA
    0012           86cbcd         2         2             0.00        1.000 +- 0.707        1.000 +- 0.707  [6 ] TO BT BR BT SC SA
    0013          8cc6ccd         2         3             0.00        0.667 +- 0.471        1.500 +- 0.866  [7 ] TO BT BT SC BT BT SA
    0014        8cbbc6ccd         2         2             0.00        1.000 +- 0.707        1.000 +- 0.707  [9 ] TO BT BT SC BT BR BR BT SA
    0015           8b6ccd         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [6 ] TO BT BT SC BR SA
    0016           8c6bcd         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [6 ] TO BT BR SC BT SA
    0017        8cbbbb6cd         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [9 ] TO BT SC BR BR BR BR BT SA
    0018       bbbbbb6bcd         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [10] TO BT BR SC BR BR BR BR BR BR
    0019            8cc6d         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [5 ] TO SC BT BT SA
    .                             100000    100000         0.00/7 =  0.00  (pval:1.000 prob:0.000)  
    .                pflags_ana  3:tboolean-box:tboolean-box   -3:tboolean-box:tboolean-box        c2        ab        ba 
    .                             100000    100000         0.00/6 =  0.00  (pval:1.000 prob:0.000)  
    0000             1880     87782     87782             0.00        1.000 +- 0.003        1.000 +- 0.003  [3 ] TO|BT|SA
    0001             1480      6313      6313             0.00        1.000 +- 0.013        1.000 +- 0.013  [3 ] TO|BR|SA
    0002             1c80      5795      5795             0.00        1.000 +- 0.013        1.000 +- 0.013  [4 ] TO|BT|BR|SA
    0003             18a0        35        35             0.00        1.000 +- 0.169        1.000 +- 0.169  [4 ] TO|BT|SA|SC
    0004             10a0        29        29             0.00        1.000 +- 0.186        1.000 +- 0.186  [3 ] TO|SA|SC
    0005             1ca0        18        18             0.00        1.000 +- 0.236        1.000 +- 0.236  [5 ] TO|BT|BR|SA|SC
    0006             1808        16        16             0.00        1.000 +- 0.250        1.000 +- 0.250  [3 ] TO|BT|AB
    0007             1c20        10        10             0.00        1.000 +- 0.316        1.000 +- 0.316  [4 ] TO|BT|BR|SC
    0008             1c08         2         2             0.00        1.000 +- 0.707        1.000 +- 0.707  [4 ] TO|BT|BR|AB
    .                             100000    100000         0.00/6 =  0.00  (pval:1.000 prob:0.000)  
    .                seqmat_ana  3:tboolean-box:tboolean-box   -3:tboolean-box:tboolean-box        c2        ab        ba 
    .                             100000    100000         0.00/6 =  0.00  (pval:1.000 prob:0.000)  
    0000             3414     87782     87782             0.00        1.000 +- 0.003        1.000 +- 0.003  [4 ] Va G2 Va Ro
    0001              344      6342      6342             0.00        1.000 +- 0.013        1.000 +- 0.013  [3 ] Va Va Ro
    0002            34114      5427      5427             0.00        1.000 +- 0.014        1.000 +- 0.014  [5 ] Va G2 G2 Va Ro
    0003           341114       350       350             0.00        1.000 +- 0.053        1.000 +- 0.053  [6 ] Va G2 G2 G2 Va Ro
    0004          3411114        28        28             0.00        1.000 +- 0.189        1.000 +- 0.189  [7 ] Va G2 G2 G2 G2 Va Ro
    0005            34414        24        24             0.00        1.000 +- 0.204        1.000 +- 0.204  [5 ] Va G2 Va Va Ro
    0006              114        16        16             0.00        1.000 +- 0.250        1.000 +- 0.250  [3 ] Va G2 G2
    0007       1111111114        10        10             0.00        1.000 +- 0.316        1.000 +- 0.316  [10] Va G2 G2 G2 G2 G2 G2 G2 G2 G2
    0008           341144         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [6 ] Va Va G2 G2 Va Ro
    0009         34114414         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [8 ] Va G2 Va Va G2 G2 Va Ro
    0010           344114         2         2             0.00        1.000 +- 0.707        1.000 +- 0.707  [6 ] Va G2 G2 Va Va Ro
    0011          3414414         2         3             0.00        0.667 +- 0.471        1.500 +- 0.866  [7 ] Va G2 Va Va G2 Va Ro
    0012        341114414         2         2             0.00        1.000 +- 0.707        1.000 +- 0.707  [9 ] Va G2 Va Va G2 G2 G2 Va Ro
    0013          3411144         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [7 ] Va Va G2 G2 G2 Va Ro
    0014            34144         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [5 ] Va Va G2 Va Ro
    0015          3411444         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [7 ] Va Va Va G2 G2 Va Ro
    0016            11114         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [5 ] Va G2 G2 G2 G2
    0017        341111114         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [9 ] Va G2 G2 G2 G2 G2 G2 Va Ro
    0018           341414         1         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [6 ] Va G2 Va G2 Va Ro
    0019             1114         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [4 ] Va G2 G2 G2
    .                             100000    100000         0.00/6 =  0.00  (pval:1.000 prob:0.000)  
    ab.a.metadata:/tmp/tboolean-box/evt/tboolean-box/torch/3                   ox:963f84ba3a1a6c894054ebbcf0fb1ea9 rx:3db691ffd21dfa48c062c0f19bb0fdb0 np: 100000 pr:    0.0078 INTEROP_MODE
    ab.a.metadata.csgmeta0:{u'containerscale': 3.0, u'container': 1, u'ctrl': 0, u'verbosity': u'0', u'poly': u'IM', u'emitconfig': u'photons:100000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x1,umin:0.45,umax:0.55,vmin:0.45,vmax:0.55', u'resolution': u'20', u'emit': -1}
    ab.rpost_dv maxdvmax: 0.01376 maxdv: 0.01376        0  0.01376        0        0  skip:SC AB RE
      idx        msg :                            sel :    lcu1     lcu2  :       nitem     nelem/    ndisc: fdisc  mx/mn/av        mx/       mn/      avg  eps:eps    
     0000            :                    TO BT BT SA :   87782    87782  :       87782   1404512/       52: 0.000  mx/mn/av   0.01376/        0/5.096e-07  eps:0.0002    
     0001            :                       TO BR SA :    6313     6313  :        6313     75756/        0: 0.000  mx/mn/av         0/        0/        0  eps:0.0002    
     0002            :                 TO BT BR BT SA :    5420     5420  :        5420    108400/        5: 0.000  mx/mn/av   0.01376/        0/6.349e-07  eps:0.0002    
     0003            :              TO BT BR BR BT SA :     349      349  :         349      8376/        0: 0.000  mx/mn/av         0/        0/        0  eps:0.0002    
     0005            :           TO BT BR BR BR BT SA :      26       26  :          26       728/        0: 0.000  mx/mn/av         0/        0/        0  eps:0.0002    
    ab.rpol_dv maxdvmax:       0 maxdv:       0        0        0        0        0  skip:SC AB RE
      idx        msg :                            sel :    lcu1     lcu2  :       nitem     nelem/    ndisc: fdisc  mx/mn/av        mx/       mn/      avg  eps:eps    
     0000            :                    TO BT BT SA :   87782    87782  :       87782   1053384/        0: 0.000  mx/mn/av         0/        0/        0  eps:0.0002    
     0001            :                       TO BR SA :    6313     6313  :        6313     56817/        0: 0.000  mx/mn/av         0/        0/        0  eps:0.0002    
     0002            :                 TO BT BR BT SA :    5420     5420  :        5420     81300/        0: 0.000  mx/mn/av         0/        0/        0  eps:0.0002    
     0003            :              TO BT BR BR BT SA :     349      349  :         349      6282/        0: 0.000  mx/mn/av         0/        0/        0  eps:0.0002    
     0005            :           TO BT BR BR BR BT SA :      26       26  :          26       546/        0: 0.000  mx/mn/av         0/        0/        0  eps:0.0002    
    ab.ox_dv maxdvmax:4.768e-07 maxdv:2.384e-07        0 4.768e-07 4.768e-07 4.768e-07  skip:SC AB RE
      idx        msg :                            sel :    lcu1     lcu2  :       nitem     nelem/    ndisc: fdisc  mx/mn/av        mx/       mn/      avg  eps:eps    
     0000            :                    TO BT BT SA :   87782    87782  :       87782   1053384/        0: 0.000  mx/mn/av 2.384e-07/        0/2.484e-08  eps:0.0002    
     0001            :                       TO BR SA :    6313     6313  :        6313     75756/        0: 0.000  mx/mn/av         0/        0/        0  eps:0.0002    
     0002            :                 TO BT BR BT SA :    5420     5420  :        5420     65040/        0: 0.000  mx/mn/av 4.768e-07/        0/ 4.47e-08  eps:0.0002    
     0003            :              TO BT BR BR BT SA :     349      349  :         349      4188/        0: 0.000  mx/mn/av 4.768e-07/        0/ 4.47e-08  eps:0.0002    
     0005            :           TO BT BR BR BR BT SA :      26       26  :          26       312/        0: 0.000  mx/mn/av 4.768e-07/        0/ 4.47e-08  eps:0.0002    
    c2p : {'seqmat_ana': 0.0, 'pflags_ana': 0.0, 'seqhis_ana': 0.0} c2pmax: 0.0  CUT ok.c2max 2.0  RC:0 
    rmxs_ : {'rpol_dv': 0.0, 'rpost_dv': 0.013763847773677895} rmxs_max_: 0.0137638477737  CUT ok.rdvmax 0.1  RC:0 
    pmxs_ : {'ox_dv': 4.76837158203125e-07} pmxs_max_: 4.76837158203e-07  CUT ok.pdvmax 0.001  RC:0 
    [2019-06-18 14:32:01,391] p448493 {tboolean.py:71} INFO     - early exit as non-interactive
    2019-06-18 14:32:01.524 INFO  [447362] [SSys::run@71] tboolean.py --tagoffset 0 --tag 3 --det tboolean-box --pfx tboolean-box --src torch   rc_raw : 0 rc : 0




Without the skips things aint so rosy, possibly the log(double) removal

* add OpticksSwitches to event metadata, prior to flipping any 


::

    [blyth@localhost issues]$ tboolean-box-ip --tag 3
    Python 2.7.15 |Anaconda, Inc.| (default, May  1 2018, 23:32:55) 
    Type "copyright", "credits" or "license" for more information.

    IPython 5.7.0 -- An enhanced Interactive Python.
    ?         -> Introduction and overview of IPython's features.
    %quickref -> Quick reference.
    help      -> Python's own help system.
    object?   -> Details about 'object', use 'object??' for extra details.
    args: /home/blyth/opticks/ana/tboolean.py --det tboolean-box --pfx tboolean-box --tag 1 --tag 3
    [2019-06-18 15:35:53,705] p91741 {tboolean.py:63} INFO     - pfx tboolean-box tag 3 src torch det tboolean-box c2max 2.0 ipython True 
    ab.cfm nph:  100000 A:    0.0078 B:   26.8359 B/A: 3435.0000 INTEROP_MODE ALIGN 
    ab.a.metadata:/tmp/tboolean-box/evt/tboolean-box/torch/3                   ox:963f84ba3a1a6c894054ebbcf0fb1ea9 rx:3db691ffd21dfa48c062c0f19bb0fdb0 np: 100000 pr:    0.0078 INTEROP_MODE
    ab.b.metadata:/tmp/tboolean-box/evt/tboolean-box/torch/-3                  ox:dfa123a7382e32bfebb7eb741ccaa749 rx:e1c46ce4b32c1c7e00f1378e807aa972 np: 100000 pr:   26.8359 INTEROP_MODE
    AB(3,torch,tboolean-box)  None 0 
    A tboolean-box/tboolean-box/torch/  3 :  20190618-1431 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/tboolean-box/evt/tboolean-box/torch/3/fdom.npy () 
    B tboolean-box/tboolean-box/torch/ -3 :  20190618-1431 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/tboolean-box/evt/tboolean-box/torch/-3/fdom.npy (recstp) 
    Rock//perfectAbsorbSurface/Vacuum,Vacuum///GlassSchottF2
    tboolean-box
    .                seqhis_ana  3:tboolean-box:tboolean-box   -3:tboolean-box:tboolean-box        c2        ab        ba 
    .                             100000    100000         0.00/7 =  0.00  (pval:1.000 prob:0.000)  
    0000             8ccd     87782     87782             0.00        1.000 +- 0.003        1.000 +- 0.003  [4 ] TO BT BT SA
    0001              8bd      6313      6313             0.00        1.000 +- 0.013        1.000 +- 0.013  [3 ] TO BR SA
    0002            8cbcd      5420      5420             0.00        1.000 +- 0.014        1.000 +- 0.014  [5 ] TO BT BR BT SA
    0003           8cbbcd       349       349             0.00        1.000 +- 0.054        1.000 +- 0.054  [6 ] TO BT BR BR BT SA
    0004              86d        29        29             0.00        1.000 +- 0.186        1.000 +- 0.186  [3 ] TO SC SA
    0005          8cbbbcd        26        26             0.00        1.000 +- 0.196        1.000 +- 0.196  [7 ] TO BT BR BR BR BT SA
    0006            86ccd        24        24             0.00        1.000 +- 0.204        1.000 +- 0.204  [5 ] TO BT BT SC SA
    0007              4cd        16        16             0.00        1.000 +- 0.250        1.000 +- 0.250  [3 ] TO BT AB
    0008       bbbbbbb6cd         9         9             0.00        1.000 +- 0.333        1.000 +- 0.333  [10] TO BT SC BR BR BR BR BR BR BR
    0009            8c6cd         7         7             0.00        1.000 +- 0.378        1.000 +- 0.378  [5 ] TO BT SC BT SA
    0010         8cbc6ccd         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [8 ] TO BT BT SC BT BR BT SA
    0011           8cbc6d         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [6 ] TO SC BT BR BT SA
    0012           86cbcd         2         2             0.00        1.000 +- 0.707        1.000 +- 0.707  [6 ] TO BT BR BT SC SA
    0013          8cc6ccd         2         3             0.00        0.667 +- 0.471        1.500 +- 0.866  [7 ] TO BT BT SC BT BT SA
    0014        8cbbc6ccd         2         2             0.00        1.000 +- 0.707        1.000 +- 0.707  [9 ] TO BT BT SC BT BR BR BT SA
    0015           8b6ccd         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [6 ] TO BT BT SC BR SA
    0016           8c6bcd         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [6 ] TO BT BR SC BT SA
    0017        8cbbbb6cd         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [9 ] TO BT SC BR BR BR BR BT SA
    0018       bbbbbb6bcd         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [10] TO BT BR SC BR BR BR BR BR BR
    0019            8cc6d         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [5 ] TO SC BT BT SA
    .                             100000    100000         0.00/7 =  0.00  (pval:1.000 prob:0.000)  
    .                pflags_ana  3:tboolean-box:tboolean-box   -3:tboolean-box:tboolean-box        c2        ab        ba 
    .                             100000    100000         0.00/6 =  0.00  (pval:1.000 prob:0.000)  
    0000             1880     87782     87782             0.00        1.000 +- 0.003        1.000 +- 0.003  [3 ] TO|BT|SA
    0001             1480      6313      6313             0.00        1.000 +- 0.013        1.000 +- 0.013  [3 ] TO|BR|SA
    0002             1c80      5795      5795             0.00        1.000 +- 0.013        1.000 +- 0.013  [4 ] TO|BT|BR|SA
    0003             18a0        35        35             0.00        1.000 +- 0.169        1.000 +- 0.169  [4 ] TO|BT|SA|SC
    0004             10a0        29        29             0.00        1.000 +- 0.186        1.000 +- 0.186  [3 ] TO|SA|SC
    0005             1ca0        18        18             0.00        1.000 +- 0.236        1.000 +- 0.236  [5 ] TO|BT|BR|SA|SC
    0006             1808        16        16             0.00        1.000 +- 0.250        1.000 +- 0.250  [3 ] TO|BT|AB
    0007             1c20        10        10             0.00        1.000 +- 0.316        1.000 +- 0.316  [4 ] TO|BT|BR|SC
    0008             1c08         2         2             0.00        1.000 +- 0.707        1.000 +- 0.707  [4 ] TO|BT|BR|AB
    .                             100000    100000         0.00/6 =  0.00  (pval:1.000 prob:0.000)  
    .                seqmat_ana  3:tboolean-box:tboolean-box   -3:tboolean-box:tboolean-box        c2        ab        ba 
    .                             100000    100000         0.00/6 =  0.00  (pval:1.000 prob:0.000)  
    0000             3414     87782     87782             0.00        1.000 +- 0.003        1.000 +- 0.003  [4 ] Va G2 Va Ro
    0001              344      6342      6342             0.00        1.000 +- 0.013        1.000 +- 0.013  [3 ] Va Va Ro
    0002            34114      5427      5427             0.00        1.000 +- 0.014        1.000 +- 0.014  [5 ] Va G2 G2 Va Ro
    0003           341114       350       350             0.00        1.000 +- 0.053        1.000 +- 0.053  [6 ] Va G2 G2 G2 Va Ro
    0004          3411114        28        28             0.00        1.000 +- 0.189        1.000 +- 0.189  [7 ] Va G2 G2 G2 G2 Va Ro
    0005            34414        24        24             0.00        1.000 +- 0.204        1.000 +- 0.204  [5 ] Va G2 Va Va Ro
    0006              114        16        16             0.00        1.000 +- 0.250        1.000 +- 0.250  [3 ] Va G2 G2
    0007       1111111114        10        10             0.00        1.000 +- 0.316        1.000 +- 0.316  [10] Va G2 G2 G2 G2 G2 G2 G2 G2 G2
    0008           341144         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [6 ] Va Va G2 G2 Va Ro
    0009         34114414         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [8 ] Va G2 Va Va G2 G2 Va Ro
    0010           344114         2         2             0.00        1.000 +- 0.707        1.000 +- 0.707  [6 ] Va G2 G2 Va Va Ro
    0011          3414414         2         3             0.00        0.667 +- 0.471        1.500 +- 0.866  [7 ] Va G2 Va Va G2 Va Ro
    0012        341114414         2         2             0.00        1.000 +- 0.707        1.000 +- 0.707  [9 ] Va G2 Va Va G2 G2 G2 Va Ro
    0013          3411144         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [7 ] Va Va G2 G2 G2 Va Ro
    0014            34144         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [5 ] Va Va G2 Va Ro
    0015          3411444         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [7 ] Va Va Va G2 G2 Va Ro
    0016            11114         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [5 ] Va G2 G2 G2 G2
    0017        341111114         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [9 ] Va G2 G2 G2 G2 G2 G2 Va Ro
    0018           341414         1         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [6 ] Va G2 Va G2 Va Ro
    0019             1114         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [4 ] Va G2 G2 G2
    .                             100000    100000         0.00/6 =  0.00  (pval:1.000 prob:0.000)  
    ab.cfm nph:  100000 A:    0.0078 B:   26.8359 B/A: 3435.0000 INTEROP_MODE ALIGN 
    ab.a.metadata:/tmp/tboolean-box/evt/tboolean-box/torch/3                   ox:963f84ba3a1a6c894054ebbcf0fb1ea9 rx:3db691ffd21dfa48c062c0f19bb0fdb0 np: 100000 pr:    0.0078 INTEROP_MODE
    ab.b.metadata:/tmp/tboolean-box/evt/tboolean-box/torch/-3                  ox:dfa123a7382e32bfebb7eb741ccaa749 rx:e1c46ce4b32c1c7e00f1378e807aa972 np: 100000 pr:   26.8359 INTEROP_MODE
    ab.a.metadata.csgmeta0:{u'containerscale': 3.0, u'container': 1, u'ctrl': 0, u'verbosity': u'0', u'poly': u'IM', u'emitconfig': u'photons:100000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x1,umin:0.45,umax:0.55,vmin:0.45,vmax:0.55', u'resolution': u'20', u'emit': -1}
    ab.rpost_dv maxdvmax:  0.4129 maxdv: 0.01376        0  0.01376        0   0.4129        0   0.1652  0.05506   0.2753  0.05506   0.1652  0.09635  0.05506  0.04129   0.1376  0.05506  0.04129   0.1239  0.05506  0.05506  0.09635  0.09635  0.05506  0.08258  0.04129  0.04129  0.09635  skip:
      idx        msg :                            sel :    lcu1     lcu2  :       nitem     nelem/    ndisc: fdisc  mx/mn/av        mx/       mn/      avg  eps:eps    
     0000            :                    TO BT BT SA :   87782    87782  :       87782   1404512/       52: 0.000  mx/mn/av   0.01376/        0/5.096e-07  eps:0.0002    
     0001            :                       TO BR SA :    6313     6313  :        6313     75756/        0: 0.000  mx/mn/av         0/        0/        0  eps:0.0002    
     0002            :                 TO BT BR BT SA :    5420     5420  :        5420    108400/        5: 0.000  mx/mn/av   0.01376/        0/6.349e-07  eps:0.0002    
     0003            :              TO BT BR BR BT SA :     349      349  :         349      8376/        0: 0.000  mx/mn/av         0/        0/        0  eps:0.0002    
     0004            :                       TO SC SA :      29       29  :          29       348/       97: 0.279  mx/mn/av    0.4129/        0/  0.01187  eps:0.0002    
     0005            :           TO BT BR BR BR BT SA :      26       26  :          26       728/        0: 0.000  mx/mn/av         0/        0/        0  eps:0.0002    
     0006            :                 TO BT BT SC SA :      24       24  :          24       480/       74: 0.154  mx/mn/av    0.1652/        0/ 0.005867  eps:0.0002    
     0007            :                       TO BT AB :      16       16  :          16       192/       21: 0.109  mx/mn/av   0.05506/        0/ 0.003815  eps:0.0002    
     0008            :  TO BT SC BR BR BR BR BR BR BR :       9        9  :           9       360/      138: 0.383  mx/mn/av    0.2753/        0/  0.01186  eps:0.0002    
     0009            :                 TO BT SC BT SA :       7        7  :           7       140/       31: 0.221  mx/mn/av   0.05506/        0/ 0.006135  eps:0.0002    
     0010            :        TO BT BT SC BT BR BT SA :       3        3  :           3        96/       29: 0.302  mx/mn/av    0.1652/        0/  0.01381  eps:0.0002    
     0011            :              TO SC BT BR BT SA :       3        3  :           3        72/       29: 0.403  mx/mn/av   0.09635/        0/  0.01114  eps:0.0002    
     0012            :              TO BT BR BT SC SA :       2        2  :           2        48/        4: 0.083  mx/mn/av   0.05506/        0/ 0.004014  eps:0.0002    
     0013            :           TO BT BT SC BT BT SA :       2        3  :           2        56/       12: 0.214  mx/mn/av   0.04129/        0/ 0.003474  eps:0.0002    
     0014            :     TO BT BT SC BT BR BR BT SA :       2        2  :           2        72/       25: 0.347  mx/mn/av    0.1376/        0/  0.01249  eps:0.0002    
     0015            :              TO BT BT SC BR SA :       1        1  :           1        24/        6: 0.250  mx/mn/av   0.05506/        0/ 0.005212  eps:0.0002    
     0016            :              TO BT BR SC BT SA :       1        1  :           1        24/        2: 0.083  mx/mn/av   0.04129/        0/ 0.001746  eps:0.0002    
     0017            :     TO BT SC BR BR BR BR BT SA :       1        1  :           1        36/       10: 0.278  mx/mn/av    0.1239/        0/  0.01229  eps:0.0002    
     0018            :  TO BT BR SC BR BR BR BR BR BR :       1        1  :           1        40/       15: 0.375  mx/mn/av   0.05506/        0/ 0.008694  eps:0.0002    
     0019            :                 TO SC BT BT SA :       1        1  :           1        20/        5: 0.250  mx/mn/av   0.05506/        0/ 0.006912  eps:0.0002    
     0020            :           TO BT BR SC BR BT SA :       1        1  :           1        28/        7: 0.250  mx/mn/av   0.09635/        0/  0.00793  eps:0.0002    
     0021            :           TO BR SC BT BR BT SA :       1        1  :           1        28/       10: 0.357  mx/mn/av   0.09635/        0/  0.01086  eps:0.0002    
     0022            :              TO BR SC BT BT SA :       1        1  :           1        24/        8: 0.333  mx/mn/av   0.05506/        0/  0.01037  eps:0.0002    
     0023            :           TO SC BT BR BR BT SA :       1        1  :           1        28/        9: 0.321  mx/mn/av   0.08258/        0/  0.01084  eps:0.0002    
     0025            :                    TO BT BR AB :       1        1  :           1        16/        1: 0.062  mx/mn/av   0.04129/        0/ 0.002581  eps:0.0002    
     0026            :                 TO BT BR BR AB :       1        1  :           1        20/        2: 0.100  mx/mn/av   0.04129/        0/ 0.002095  eps:0.0002    
     0027            :           TO BT SC BR BR BT SA :       1        1  :           1        28/        8: 0.286  mx/mn/av   0.09635/        0/ 0.009427  eps:0.0002    
    ab.rpol_dv maxdvmax:       0 maxdv:       0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0  skip:
      idx        msg :                            sel :    lcu1     lcu2  :       nitem     nelem/    ndisc: fdisc  mx/mn/av        mx/       mn/      avg  eps:eps    
     0000            :                    TO BT BT SA :   87782    87782  :       87782   1053384/        0: 0.000  mx/mn/av         0/        0/        0  eps:0.0002    
     0001            :                       TO BR SA :    6313     6313  :        6313     56817/        0: 0.000  mx/mn/av         0/        0/        0  eps:0.0002    
     0002            :                 TO BT BR BT SA :    5420     5420  :        5420     81300/        0: 0.000  mx/mn/av         0/        0/        0  eps:0.0002    
     0003            :              TO BT BR BR BT SA :     349      349  :         349      6282/        0: 0.000  mx/mn/av         0/        0/        0  eps:0.0002    
     0004            :                       TO SC SA :      29       29  :          29       261/        0: 0.000  mx/mn/av         0/        0/        0  eps:0.0002    
     0005            :           TO BT BR BR BR BT SA :      26       26  :          26       546/        0: 0.000  mx/mn/av         0/        0/        0  eps:0.0002    
     0006            :                 TO BT BT SC SA :      24       24  :          24       360/        0: 0.000  mx/mn/av         0/        0/        0  eps:0.0002    
     0007            :                       TO BT AB :      16       16  :          16       144/        0: 0.000  mx/mn/av         0/        0/        0  eps:0.0002    
     0008            :  TO BT SC BR BR BR BR BR BR BR :       9        9  :           9       270/        0: 0.000  mx/mn/av         0/        0/        0  eps:0.0002    
     0009            :                 TO BT SC BT SA :       7        7  :           7       105/        0: 0.000  mx/mn/av         0/        0/        0  eps:0.0002    
     0010            :        TO BT BT SC BT BR BT SA :       3        3  :           3        72/        0: 0.000  mx/mn/av         0/        0/        0  eps:0.0002    
     0011            :              TO SC BT BR BT SA :       3        3  :           3        54/        0: 0.000  mx/mn/av         0/        0/        0  eps:0.0002    
     0012            :              TO BT BR BT SC SA :       2        2  :           2        36/        0: 0.000  mx/mn/av         0/        0/        0  eps:0.0002    
     0013            :           TO BT BT SC BT BT SA :       2        3  :           2        42/        0: 0.000  mx/mn/av         0/        0/        0  eps:0.0002    
     0014            :     TO BT BT SC BT BR BR BT SA :       2        2  :           2        54/        0: 0.000  mx/mn/av         0/        0/        0  eps:0.0002    
     0015            :              TO BT BT SC BR SA :       1        1  :           1        18/        0: 0.000  mx/mn/av         0/        0/        0  eps:0.0002    
     0016            :              TO BT BR SC BT SA :       1        1  :           1        18/        0: 0.000  mx/mn/av         0/        0/        0  eps:0.0002    
     0017            :     TO BT SC BR BR BR BR BT SA :       1        1  :           1        27/        0: 0.000  mx/mn/av         0/        0/        0  eps:0.0002    
     0018            :  TO BT BR SC BR BR BR BR BR BR :       1        1  :           1        30/        0: 0.000  mx/mn/av         0/        0/        0  eps:0.0002    
     0019            :                 TO SC BT BT SA :       1        1  :           1        15/        0: 0.000  mx/mn/av         0/        0/        0  eps:0.0002    
     0020            :           TO BT BR SC BR BT SA :       1        1  :           1        21/        0: 0.000  mx/mn/av         0/        0/        0  eps:0.0002    
     0021            :           TO BR SC BT BR BT SA :       1        1  :           1        21/        0: 0.000  mx/mn/av         0/        0/        0  eps:0.0002    
     0022            :              TO BR SC BT BT SA :       1        1  :           1        18/        0: 0.000  mx/mn/av         0/        0/        0  eps:0.0002    
     0023            :           TO SC BT BR BR BT SA :       1        1  :           1        21/        0: 0.000  mx/mn/av         0/        0/        0  eps:0.0002    
     0025            :                    TO BT BR AB :       1        1  :           1        12/        0: 0.000  mx/mn/av         0/        0/        0  eps:0.0002    
     0026            :                 TO BT BR BR AB :       1        1  :           1        15/        0: 0.000  mx/mn/av         0/        0/        0  eps:0.0002    
     0027            :           TO BT SC BR BR BT SA :       1        1  :           1        21/        0: 0.000  mx/mn/av         0/        0/        0  eps:0.0002    
    ab.ox_dv maxdvmax:  0.4052 maxdv:2.384e-07        0 4.768e-07 4.768e-07   0.4052 4.768e-07   0.1665  0.05026   0.0637  0.04944  0.04845   0.0361  0.04932   0.0188  0.04688  0.02348 0.0005379  0.04691  0.03917  0.02551  0.04883  0.04253  0.04694  0.03683  0.04692   0.0458  0.04547  skip:
      idx        msg :                            sel :    lcu1     lcu2  :       nitem     nelem/    ndisc: fdisc  mx/mn/av        mx/       mn/      avg  eps:eps    
     0000            :                    TO BT BT SA :   87782    87782  :       87782   1053384/        0: 0.000  mx/mn/av 2.384e-07/        0/2.484e-08  eps:0.0002    
     0001            :                       TO BR SA :    6313     6313  :        6313     75756/        0: 0.000  mx/mn/av         0/        0/        0  eps:0.0002    
     0002            :                 TO BT BR BT SA :    5420     5420  :        5420     65040/        0: 0.000  mx/mn/av 4.768e-07/        0/ 4.47e-08  eps:0.0002    
     0003            :              TO BT BR BR BT SA :     349      349  :         349      4188/        0: 0.000  mx/mn/av 4.768e-07/        0/ 4.47e-08  eps:0.0002    
     0004            :                       TO SC SA :      29       29  :          29       348/       63: 0.181  mx/mn/av    0.4052/        0/ 0.008075  eps:0.0002    
     0005            :           TO BT BR BR BR BT SA :      26       26  :          26       312/        0: 0.000  mx/mn/av 4.768e-07/        0/ 4.47e-08  eps:0.0002    
     0006            :                 TO BT BT SC SA :      24       24  :          24       288/       42: 0.146  mx/mn/av    0.1665/        0/ 0.005635  eps:0.0002    
     0007            :                       TO BT AB :      16       16  :          16       192/       32: 0.167  mx/mn/av   0.05026/        0/ 0.004004  eps:0.0002    
     0008            :  TO BT SC BR BR BR BR BR BR BR :       9        9  :           9       108/       22: 0.204  mx/mn/av    0.0637/        0/ 0.004439  eps:0.0002    
     0009            :                 TO BT SC BT SA :       7        7  :           7        84/       16: 0.190  mx/mn/av   0.04944/        0/  0.00295  eps:0.0002    
     0010            :        TO BT BT SC BT BR BT SA :       3        3  :           3        36/        7: 0.194  mx/mn/av   0.04845/        0/ 0.004128  eps:0.0002    
     0011            :              TO SC BT BR BT SA :       3        3  :           3        36/        6: 0.167  mx/mn/av    0.0361/        0/ 0.003522  eps:0.0002    
     0012            :              TO BT BR BT SC SA :       2        2  :           2        24/        2: 0.083  mx/mn/av   0.04932/        0/ 0.004077  eps:0.0002    
     0013            :           TO BT BT SC BT BT SA :       2        3  :           2        24/        6: 0.250  mx/mn/av    0.0188/        0/ 0.001654  eps:0.0002    
     0014            :     TO BT BT SC BT BR BR BT SA :       2        2  :           2        24/        4: 0.167  mx/mn/av   0.04688/        0/ 0.004159  eps:0.0002    
     0015            :              TO BT BT SC BR SA :       1        1  :           1        12/        3: 0.250  mx/mn/av   0.02348/        0/ 0.002064  eps:0.0002    
     0016            :              TO BT BR SC BT SA :       1        1  :           1        12/        1: 0.083  mx/mn/av 0.0005379/        0/4.969e-05  eps:0.0002    
     0017            :     TO BT SC BR BR BR BR BT SA :       1        1  :           1        12/        2: 0.167  mx/mn/av   0.04691/        0/ 0.003942  eps:0.0002    
     0018            :  TO BT BR SC BR BR BR BR BR BR :       1        1  :           1        12/        3: 0.250  mx/mn/av   0.03917/        0/ 0.004313  eps:0.0002    
     0019            :                 TO SC BT BT SA :       1        1  :           1        12/        2: 0.167  mx/mn/av   0.02551/        0/ 0.002358  eps:0.0002    
     0020            :           TO BT BR SC BR BT SA :       1        1  :           1        12/        2: 0.167  mx/mn/av   0.04883/        0/ 0.004109  eps:0.0002    
     0021            :           TO BR SC BT BR BT SA :       1        1  :           1        12/        3: 0.250  mx/mn/av   0.04253/        0/ 0.005257  eps:0.0002    
     0022            :              TO BR SC BT BT SA :       1        1  :           1        12/        1: 0.083  mx/mn/av   0.04694/        0/ 0.003932  eps:0.0002    
     0023            :           TO SC BT BR BR BT SA :       1        1  :           1        12/        2: 0.167  mx/mn/av   0.03683/        0/ 0.003549  eps:0.0002    
     0025            :                    TO BT BR AB :       1        1  :           1        12/        2: 0.167  mx/mn/av   0.04692/        0/ 0.003934  eps:0.0002    
     0026            :                 TO BT BR BR AB :       1        1  :           1        12/        2: 0.167  mx/mn/av    0.0458/        0/  0.00384  eps:0.0002    
     0027            :           TO BT SC BR BR BT SA :       1        1  :           1        12/        2: 0.167  mx/mn/av   0.04547/        0/ 0.003826  eps:0.0002    
    c2p : {'seqmat_ana': 0.0, 'pflags_ana': 0.0, 'seqhis_ana': 0.0} c2pmax: 0.0  CUT ok.c2max 2.0  RC:0 
    rmxs_ : {'rpol_dv': 0.0, 'rpost_dv': 0.4129154332102303} rmxs_max_: 0.41291543321  CUT ok.rdvmax 0.1  RC:88 
    pmxs_ : {'ox_dv': 0.4051971435546875} pmxs_max_: 0.405197143555  CUT ok.pdvmax 0.001  RC:99 


