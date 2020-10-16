tboolean-box-analysis-fail
=============================

Prior:

* :doc:`G4Opticks_GGeo_rejig_shakedown`


::

    LV=box GGeoTest=INFO tboolean.sh --generateoverride 10000 -D

::

    ...
    [2020-10-16 16:38:21,666] p25995 {<module>            :tboolean.py:48} INFO     - ]AB
    ab.pro
          ap.tim 0.0312         bp.tim 0.0312          bp.tim/ap.tim 1.0000        
    ab.pro.ap
      /tmp/blyth/opticks/tboolean-box/evt/tboolean-box/torch/1/OpticksProfile.npy                 20201016-1638 
      /tmp/blyth/opticks/tboolean-box/evt/tboolean-box/torch/1/OpticksProfileAcc.npy              20201016-1638 
    slice(73, 75, None)
        idx :                                              label :          t          v         dt         dv   
         73 :                       _OPropagator::launch :     8.9648 31124.8535     0.0000     0.0000   
         74 :                        OPropagator::launch :     8.9961 31124.8535     0.0312     0.0000   
        idx :                                              label :          t          v         dt         dv   
    ab.pro.bp
      /tmp/blyth/opticks/tboolean-box/evt/tboolean-box/torch/-1/OpticksProfile.npy                20201016-1638 
      /tmp/blyth/opticks/tboolean-box/evt/tboolean-box/torch/-1/OpticksProfileAcc.npy             20201016-1638 
    slice(73, 75, None)
        idx :                                              label :          t          v         dt         dv   
         73 :                       _OPropagator::launch :     8.9648 31124.8535     0.0000     0.0000   
         74 :                        OPropagator::launch :     8.9961 31124.8535     0.0312     0.0000   
        idx :                                              label :          t          v         dt         dv   
    ab.cfm
    nph:   10000 A:    0.0312 B:    0.0000 B/A:       0.0 INTEROP_MODE ALIGN non-reflectcheat non-utaildebug 
    ab.a.metadata:/tmp/blyth/opticks/tboolean-box/evt/tboolean-box/torch/1     ox:1883f724a87d4505dd6fb5d8abbb0e48 rx:54b2d50f25bc564a8515cd83220e62d1 np:  10000 pr:    0.0312 INTEROP_MODE
    ab.b.metadata:/tmp/blyth/opticks/tboolean-box/evt/tboolean-box/torch/-1    ox:a3fed5a8352efa831f19c35dc31919b2 rx:01932254e8b99fd5c8affdffdab943ab np:  10000 pr:    0.0000 INTEROP_MODE
    WITH_SEED_BUFFER WITH_RECORD WITH_SOURCE WITH_ALIGN_DEV WITH_LOGDOUBLE WITH_KLUDGE_FLAT_ZERO_NOPEEK 
    {u'containerscale': 3.0, u'container': 1, u'ctrl': 0, u'verbosity': 0, u'containerautosize': 1, u'poly': u'IM', u'emitconfig': u'photons:100000,wavelength:380,time:0.0,posdelta:0.1,sheetmask:0x1,umin:0.45,umax:0.55,vmin:0.45,vmax:0.55', u'resolution': 20, u'emit': -1}
    .
    ab.mal
    aligned     8837/  10000 : 0.8837 : 0,1,3,4,6,8,9,10,12,13,14,15,16,17,18,20,21,22,23,24,25,26,27,29,30 
    maligned    1163/  10000 : 0.1163 : 2,5,7,11,19,28,32,38,39,47,53,64,92,100,107,120,130,137,165,168,176,185,193,200,209 
    slice(0, 25, None)
          0      2 : * :                                           TO BR MI                                           TO BR SA 
          1      5 : * :                                           TO BR MI                                           TO BR SA 
          2      7 : * :                                           TO BR MI                                           TO BR SA 
          3     11 : * :                                           TO BR MI                                           TO BR SA 
          4     19 : * :                                           TO BR MI                                           TO BR SA 
          5     28 : * :                                           TO BR MI                                           TO BR SA 
          6     32 : * :                                     TO BT BR BT MI                                     TO BT BR BT SA 
          7     38 : * :                                           TO BR MI                                           TO BR SA 
          8     39 : * :                                           TO BR MI                                           TO BR SA 
          9     47 : * :                                     TO BT BR BT MI                                     TO BT BR BT SA 
         10     53 : * :                                     TO BT BR BT MI                                     TO BT BR BT SA 
         11     64 : * :                                           TO BR MI                                           TO BR SA 
         12     92 : * :                                           TO BR MI                                           TO BR SA 
         13    100 : * :                                     TO BT BR BT MI                                     TO BT BR BT SA 
         14    107 : * :                               TO BT BR BR BR BT MI                               TO BT BR BR BR BT SA 
         15    120 : * :                                           TO BR MI                                           TO BR SA 
         16    130 : * :                                     TO BT BR BT MI                                     TO BT BR BT SA 
         17    137 : * :                                           TO BR MI                                           TO BR SA 
         18    165 : * :                                           TO BR MI                                           TO BR SA 
         19    168 : * :                                     TO BT BR BT MI                                     TO BT BR BT SA 
         20    176 : * :                                           TO BR MI                                           TO BR SA 
         21    185 : * :                                           TO BR MI                                           TO BR SA 
         22    193 : * :                                     TO BT BR BT MI                                     TO BT BR BT SA 
         23    200 : * :                                           TO BR MI                                           TO BR SA 
         24    209 : * :                                           TO BR MI                                           TO BR SA 
    ab.mal.migtab
      580                3bd                       TO BR MI                8bd                       TO BR SA  
      562              3cbcd                 TO BT BR BT MI              8cbcd                 TO BT BR BT SA  
        6            3cbbbcd           TO BT BR BR BR BT MI            8cbbbcd           TO BT BR BR BR BT SA  
        5                36d                       TO SC MI                86d                       TO SC SA  
        3                 4d                          TO AB               8ccd                    TO BT BT SA  
        1              3cc6d                 TO SC BT BT MI             8cbc6d              TO SC BT BR BT SA  
        1              3b6bd                 TO BR SC BR MI            8cbc6bd           TO BR SC BT BR BT SA  
        1               8ccd                    TO BT BT SA                4cd                       TO BT AB  
        1             8b6ccd              TO BT BT SC BR SA          8cbbc6ccd     TO BT BT SC BT BR BR BT SA  
        1              3cbcd                 TO BT BR BT MI                4cd                       TO BT AB  
        1            3cc6ccd           TO BT BT SC BT BT MI            8cc6ccd           TO BT BT SC BT BT SA  
        1              3c6cd                 TO BT SC BT MI          8cbbbb6cd     TO BT SC BR BR BR BR BT SA  
    .
    ab
    AB(1,torch,tboolean-box)  None 0     file_photons 10k   load_slice 0:100k:   loaded_photons 10k  
    A tboolean-box/tboolean-box/torch/  1 :  20201016-1638 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/tboolean-box/evt/tboolean-box/torch/1/fdom.npy () 
    B tboolean-box/tboolean-box/torch/ -1 :  20201016-1638 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/tboolean-box/evt/tboolean-box/torch/-1/fdom.npy (recstp) 
    Rock//perfectAbsorbSurface/Vacuum,Vacuum///GlassSchottF2
    /tmp/blyth/opticks/tboolean-box
    .
    ab.ahis
    .            all_seqhis_ana  1:tboolean-box:tboolean-box   -1:tboolean-box:tboolean-box        c2        ab        ba 
    .                              10000     10000      2285.00/5 = 457.00  (pval:1.000 prob:0.000)  
    0000             8ccd      8805      8807     -2             0.00        1.000 +- 0.011        1.000 +- 0.011  [4 ] TO BT BT SA
    0001              3bd       580         0    580           580.00        0.000 +- 0.000        0.000 +- 0.000  [3 ] TO BR MI
    0002            3cbcd       563         0    563           563.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] TO BT BR BT MI
    0003           8cbbcd        29        29      0             0.00        1.000 +- 0.186        1.000 +- 0.186  [6 ] TO BT BR BR BT SA
    0004          3cbbbcd         6         0      6             0.00        0.000 +- 0.000        0.000 +- 0.000  [7 ] TO BT BR BR BR BT MI
    0005              36d         5         0      5             0.00        0.000 +- 0.000        0.000 +- 0.000  [3 ] TO SC MI
    0006               4d         3         0      3             0.00        0.000 +- 0.000        0.000 +- 0.000  [2 ] TO AB
    0007            86ccd         2         2      0             0.00        1.000 +- 0.707        1.000 +- 0.707  [5 ] TO BT BT SC SA
    0008           8b6ccd         1         0      1             0.00        0.000 +- 0.000        0.000 +- 0.000  [6 ] TO BT BT SC BR SA
    0009            3b6bd         1         0      1             0.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] TO BR SC BR MI
    0010            3c6cd         1         0      1             0.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] TO BT SC BT MI
    0011          3cc6ccd         1         0      1             0.00        0.000 +- 0.000        0.000 +- 0.000  [7 ] TO BT BT SC BT BT MI
    0012            8c6cd         1         1      0             0.00        1.000 +- 1.000        1.000 +- 1.000  [5 ] TO BT SC BT SA
    0013            3cc6d         1         0      1             0.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] TO SC BT BT MI
    0014             4ccd         1         1      0             0.00        1.000 +- 1.000        1.000 +- 1.000  [4 ] TO BT BT AB
    0015            8cbcd         0       562   -562           562.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] TO BT BR BT SA
    0016          8cc6ccd         0         1     -1             0.00        0.000 +- 0.000        0.000 +- 0.000  [7 ] TO BT BT SC BT BT SA
    0017              8bd         0       580   -580           580.00        0.000 +- 0.000        0.000 +- 0.000  [3 ] TO BR SA
    0018           8cbc6d         0         1     -1             0.00        0.000 +- 0.000        0.000 +- 0.000  [6 ] TO SC BT BR BT SA
    0019              4cd         0         2     -2             0.00        0.000 +- 0.000        0.000 +- 0.000  [3 ] TO BT AB
    .                              10000     10000      2285.00/5 = 457.00  (pval:1.000 prob:0.000)  
    ab.flg
    .                pflags_ana  1:tboolean-box:tboolean-box   -1:tboolean-box:tboolean-box        c2        ab        ba 
    .                              10000     10000      2244.37/4 = 561.09  (pval:1.000 prob:0.000)  
    0000             1880      8805      8807     -2             0.00        1.000 +- 0.011        1.000 +- 0.011  [3 ] TO|BT|SA
    0001             1404       580         0    580           580.00        0.000 +- 0.000        0.000 +- 0.000  [3 ] TO|BR|MI
    0002             1c04       569         0    569           569.00        0.000 +- 0.000        0.000 +- 0.000  [4 ] TO|BT|BR|MI
    0003             1c80        29       597   -568           515.37        0.049 +- 0.009       20.586 +- 0.843  [4 ] TO|BT|BR|SA
    0004             1024         5         0      5             0.00        0.000 +- 0.000        0.000 +- 0.000  [3 ] TO|SC|MI
    0005             18a0         3         4     -1             0.00        0.750 +- 0.433        1.333 +- 0.667  [4 ] TO|BT|SA|SC
    0006             1008         3         0      3             0.00        0.000 +- 0.000        0.000 +- 0.000  [2 ] TO|AB
    0007             1824         3         0      3             0.00        0.000 +- 0.000        0.000 +- 0.000  [4 ] TO|BT|SC|MI
    0008             1808         1         3     -2             0.00        0.333 +- 0.333        3.000 +- 1.732  [3 ] TO|BT|AB
    0009             1ca0         1         4     -3             0.00        0.250 +- 0.250        4.000 +- 2.000  [5 ] TO|BT|BR|SA|SC
    0010             1424         1         0      1             0.00        0.000 +- 0.000        0.000 +- 0.000  [4 ] TO|BR|SC|MI
    0011             10a0         0         5     -5             0.00        0.000 +- 0.000        0.000 +- 0.000  [3 ] TO|SA|SC
    0012             1480         0       580   -580           580.00        0.000 +- 0.000        0.000 +- 0.000  [3 ] TO|BR|SA
    .                              10000     10000      2244.37/4 = 561.09  (pval:1.000 prob:0.000)  
    ab.mat
    .                seqmat_ana  1:tboolean-box:tboolean-box   -1:tboolean-box:tboolean-box        c2        ab        ba 
    .                              10000     10000     19944.00/5 = 3988.80  (pval:1.000 prob:0.000)  
    0000             3441      9369         0   9369          9369.00        0.000 +- 0.000        0.000 +- 0.000  [4 ] G2 Va Va Ro
    0001               31       585         0    585           585.00        0.000 +- 0.000        0.000 +- 0.000  [2 ] G2 Ro
    0002           344441        35         0     35            35.00        0.000 +- 0.000        0.000 +- 0.000  [6 ] G2 Va Va Va Va Ro
    0003            34441         3         0      3             0.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] G2 Va Va Va Ro
    0004               11         3         0      3             0.00        0.000 +- 0.000        0.000 +- 0.000  [2 ] G2 G2
    0005           341441         2         0      2             0.00        0.000 +- 0.000        0.000 +- 0.000  [6 ] G2 Va Va G2 Va Ro
    0006             3131         1         0      1             0.00        0.000 +- 0.000        0.000 +- 0.000  [4 ] G2 Ro G2 Ro
    0007             3411         1         0      1             0.00        0.000 +- 0.000        0.000 +- 0.000  [4 ] G2 G2 Va Ro
    0008             4441         1         0      1             0.00        0.000 +- 0.000        0.000 +- 0.000  [4 ] G2 Va Va Va
    0009              114         0         2     -2             0.00        0.000 +- 0.000        0.000 +- 0.000  [3 ] Va G2 G2
    0010          3414414         0         1     -1             0.00        0.000 +- 0.000        0.000 +- 0.000  [7 ] Va G2 Va Va G2 Va Ro
    0011            34414         0         2     -2             0.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] Va G2 Va Va Ro
    0012           341144         0         1     -1             0.00        0.000 +- 0.000        0.000 +- 0.000  [6 ] Va Va G2 G2 Va Ro
    0013        341111114         0         1     -1             0.00        0.000 +- 0.000        0.000 +- 0.000  [9 ] Va G2 G2 G2 G2 G2 G2 Va Ro
    0014           341114         0        29    -29             0.00        0.000 +- 0.000        0.000 +- 0.000  [6 ] Va G2 G2 G2 Va Ro
    0015             3414         0      8807   -8807          8807.00        0.000 +- 0.000        0.000 +- 0.000  [4 ] Va G2 Va Ro
    0016          3411444         0         1     -1             0.00        0.000 +- 0.000        0.000 +- 0.000  [7 ] Va Va Va G2 G2 Va Ro
    0017            34114         0       563   -563           563.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] Va G2 G2 Va Ro
    0018              344         0       585   -585           585.00        0.000 +- 0.000        0.000 +- 0.000  [3 ] Va Va Ro
    0019             4414         0         1     -1             0.00        0.000 +- 0.000        0.000 +- 0.000  [4 ] Va G2 Va Va
    .                              10000     10000     19944.00/5 = 3988.80  (pval:1.000 prob:0.000)  
    ab.cfm
    nph:   10000 A:    0.0312 B:    0.0000 B/A:       0.0 INTEROP_MODE ALIGN non-reflectcheat non-utaildebug 
    ab.a.metadata:/tmp/blyth/opticks/tboolean-box/evt/tboolean-box/torch/1     ox:1883f724a87d4505dd6fb5d8abbb0e48 rx:54b2d50f25bc564a8515cd83220e62d1 np:  10000 pr:    0.0312 INTEROP_MODE
    ab.b.metadata:/tmp/blyth/opticks/tboolean-box/evt/tboolean-box/torch/-1    ox:a3fed5a8352efa831f19c35dc31919b2 rx:01932254e8b99fd5c8affdffdab943ab np:  10000 pr:    0.0000 INTEROP_MODE
    WITH_SEED_BUFFER WITH_RECORD WITH_SOURCE WITH_ALIGN_DEV WITH_LOGDOUBLE WITH_KLUDGE_FLAT_ZERO_NOPEEK 
    {u'containerscale': 3.0, u'container': 1, u'ctrl': 0, u'verbosity': 0, u'containerautosize': 1, u'poly': u'IM', u'emitconfig': u'photons:100000,wavelength:380,time:0.0,posdelta:0.1,sheetmask:0x1,umin:0.45,umax:0.55,vmin:0.45,vmax:0.55', u'resolution': 20, u'emit': -1}
    .
    AB(1,torch,tboolean-box)  None 0     file_photons 10k   load_slice 0:100k:   loaded_photons 10k  
    ab.rpost_dv
    maxdvmax:0.9530  ndvp:8835  level:FATAL  RC:1       skip:
                     :                                :                   :                       :  8835  8835  8835 : 0.0151 0.0220 0.0289 :                                    
      idx        msg :                            sel :    lcu1     lcu2  :       nitem     nelem :  nwar  nerr  nfat :   fwar   ferr   ffat :        mx        mn       avg      
     0000            :                    TO BT BT SA :    8805     8807  :        8804    140864 :  8804  8804  8804 : 0.0625 0.0625 0.0625 :    0.9530    0.9530    0.0596   :                FATAL :   > dvmax[2] 0.0289  
     0003            :              TO BT BR BR BT SA :      29       29  :          29       696 :    29    29    29 : 0.0417 0.0417 0.0417 :    0.9530    0.9530    0.0397   :                FATAL :   > dvmax[2] 0.0289  
     0007            :                 TO BT BT SC SA :       2        2  :           2        40 :     2     2     2 : 0.0500 0.0500 0.0500 :    0.9530    0.9530    0.0477   :                FATAL :   > dvmax[2] 0.0289  
    .
    ab.rpol_dv
    maxdvmax:2.0000  ndvp:  29  level:FATAL  RC:1       skip:
                     :                                :                   :                       :    29    29    29 : 0.0078 0.0118 0.0157 :                                    
      idx        msg :                            sel :    lcu1     lcu2  :       nitem     nelem :  nwar  nerr  nfat :   fwar   ferr   ffat :        mx        mn       avg      
     0000            :                    TO BT BT SA :    8805     8807  :        8804    105648 :     0     0     0 : 0.0000 0.0000 0.0000 :    0.0000    0.0000    0.0000   :                 INFO :  
     0003            :              TO BT BR BR BT SA :      29       29  :          29       522 :    29    29    29 : 0.0556 0.0556 0.0556 :    2.0000    2.0000    0.1111   :                FATAL :   > dvmax[2] 0.0157  
     0007            :                 TO BT BT SC SA :       2        2  :           2        30 :     0     0     0 : 0.0000 0.0000 0.0000 :    0.0000    0.0000    0.0000   :                 INFO :  
    .
    ab.ox_dv
    maxdvmax:0.6813  ndvp:8835  level:FATAL  RC:1       skip:
                     :                                :                   :                       :  8835  8835    29 : 0.1000 0.2500 0.5000 :                                    
      idx        msg :                            sel :    lcu1     lcu2  :       nitem     nelem :  nwar  nerr  nfat :   fwar   ferr   ffat :        mx        mn       avg      
     0000            :                    TO BT BT SA :    8805     8807  :        8804    422592 :  8804  8804     0 : 0.0208 0.0208 0.0000 :    0.4083    0.4083    0.0085   :                ERROR :   > dvmax[1] 0.2500  
     0003            :              TO BT BR BR BT SA :      29       29  :          29      2088 :    29    29    29 : 0.0139 0.0139 0.0139 :    0.6813    0.6813    0.0095   :                FATAL :   > dvmax[2] 0.5000  
     0007            :                 TO BT BT SC SA :       2        2  :           2       120 :     2     2     0 : 0.0167 0.0167 0.0000 :    0.4083    0.4083    0.0068   :                ERROR :   > dvmax[1] 0.2500  
    .
    AB(1,torch,tboolean-box)  None 0     file_photons 10k   load_slice 0:100k:   loaded_photons 10k  
    RC 0x07
    ab.cfm
    nph:   10000 A:    0.0312 B:    0.0000 B/A:       0.0 INTEROP_MODE ALIGN non-reflectcheat non-utaildebug 
    ab.a.metadata:/tmp/blyth/opticks/tboolean-box/evt/tboolean-box/torch/1     ox:1883f724a87d4505dd6fb5d8abbb0e48 rx:54b2d50f25bc564a8515cd83220e62d1 np:  10000 pr:    0.0312 INTEROP_MODE
    ab.b.metadata:/tmp/blyth/opticks/tboolean-box/evt/tboolean-box/torch/-1    ox:a3fed5a8352efa831f19c35dc31919b2 rx:01932254e8b99fd5c8affdffdab943ab np:  10000 pr:    0.0000 INTEROP_MODE
    WITH_SEED_BUFFER WITH_RECORD WITH_SOURCE WITH_ALIGN_DEV WITH_LOGDOUBLE WITH_KLUDGE_FLAT_ZERO_NOPEEK 
    {u'containerscale': 3.0, u'container': 1, u'ctrl': 0, u'verbosity': 0, u'containerautosize': 1, u'poly': u'IM', u'emitconfig': u'photons:100000,wavelength:380,time:0.0,posdelta:0.1,sheetmask:0x1,umin:0.45,umax:0.55,vmin:0.45,vmax:0.55', u'resolution': 20, u'emit': -1}
    .
    [2020-10-16 16:38:21,671] p25995 {<module>            :tboolean.py:54} CRITICAL -  RC 0x07 0b111 
    [2020-10-16 16:38:21,671] p25995 {<module>            :tboolean.py:57} INFO     - early exit as non-interactive
    2020-10-16 16:38:21.685 INFO  [10546477] [SSys::run@100] /opt/local/bin/python /Users/blyth/opticks/ana/tboolean.py --tagoffset 0 --tag 1 --cat tboolean-box --pfx tboolean-box --src torch --show   rc_raw : 1792 rc : 7
    2020-10-16 16:38:21.685 ERROR [10546477] [SSys::run@107] FAILED with  cmd /opt/local/bin/python /Users/blyth/opticks/ana/tboolean.py --tagoffset 0 --tag 1 --cat tboolean-box --pfx tboolean-box --src torch --show   RC 7

    2020-10-16 16:38:21.685 INFO  [10546477] [OpticksAna::run@129]  anakey tboolean cmdline /opt/local/bin/python /Users/blyth/opticks/ana/tboolean.py --tagoffset 0 --tag 1 --cat tboolean-box --pfx tboolean-box --src torch --show   interactivity 2 rc 7 rcmsg OpticksAna::run non-zero RC from ana script
    2020-10-16 16:38:21.685 FATAL [10546477] [Opticks::dumpRC@239]  rc 7 rcmsg : OpticksAna::run non-zero RC from ana script
    2020-10-16 16:38:21.685 INFO  [10546477] [SSys::WaitForInput@341] SSys::WaitForInput OpticksAna::run paused : hit RETURN to continue...



Rerun just the analysis::

 
    /Users/blyth/opticks/ana/tboolean.py --tagoffset 0 --tag 1 --cat tboolean-box --pfx tboolean-box --src torch --show

    /opt/local/bin/python /Users/blyth/opticks/ana/tboolean.py --tagoffset 0 --tag 1 --cat tboolean-box --pfx tboolean-box --src torch --show 


Notice py3/py2 difference with the py3 photon histories being mangled::

    epsilon:issues blyth$ /opt/local/bin/python /Users/blyth/opticks/ana/tboolean.py --tagoffset 0 --tag 1 --cat tboolean-box --pfx tboolean-box --src torch --show  2> /tmp/ana_py2.log
    epsilon:issues blyth$ /Users/blyth/opticks/ana/tboolean.py --tagoffset 0 --tag 1 --cat tboolean-box --pfx tboolean-box --src torch --show  2> /tmp/ana_py3.log

    vimdiff /tmp/ana_py2.log /tmp/ana_py3.log 


Lots of question marks from py3::

    <   580                3bd                       TO BR MI                8bd                       TO BR SA  
    <   562              3cbcd                 TO BT BR BT MI              8cbcd                 TO BT BR BT SA  
    <     6            3cbbbcd           TO BT BR BR BR BT MI            8cbbbcd           TO BT BR BR BR BT SA  
    <     5                36d                       TO SC MI                86d                       TO SC SA  
    <     3                 4d                          TO AB               8ccd                    TO BT BT SA  
    <     1              3cc6d                 TO SC BT BT MI             8cbc6d              TO SC BT BR BT SA  
    <     1              3b6bd                 TO BR SC BR MI            8cbc6bd           TO BR SC BT BR BT SA  
    <     1               8ccd                    TO BT BT SA                4cd                       TO BT AB  
    <     1             8b6ccd              TO BT BT SC BR SA          8cbbc6ccd     TO BT BT SC BT BR BR BT SA  
    <     1              3cbcd                 TO BT BR BT MI                4cd                       TO BT AB  
    <     1            3cc6ccd           TO BT BT SC BT BT MI            8cc6ccd           TO BT BT SC BT BT SA  
    <     1              3c6cd                 TO BT SC BT MI          8cbbbb6cd     TO BT SC BR BR BR BR BT SA  
    ---
    >   580                3bd                  ?13? ?11? ?3?                8bd                  ?13? ?11? ?8?  
    >   562              3cbcd        ?13? ?12? ?11? ?12? ?3?              8cbcd        ?13? ?12? ?11? ?12? ?8?  
    >     6            3cbbbcd ?13? ?12? ?11? ?11? ?11? ?12? ?3?            8cbbbcd ?13? ?12? ?11? ?11? ?11? ?12? ?8?  
    >     5                36d                   ?13? ?6? ?3?                86d                   ?13? ?6? ?8?  
    >     3                 4d                       ?13? ?4?               8ccd             ?13? ?12? ?12? ?8?  
    >     1              3cc6d         ?13? ?6? ?12? ?12? ?3?             8cbc6d    ?13? ?6? ?12? ?11? ?12? ?8?  
    >     1              3b6bd         ?13? ?11? ?6? ?11? ?3?            8cbc6bd ?13? ?11? ?6? ?12? ?11? ?12? ?8?  
    >     1               8ccd             ?13? ?12? ?12? ?8?                4cd                  ?13? ?12? ?4?  
    >     1             8b6ccd    ?13? ?12? ?12? ?6? ?11? ?8?          8cbbc6ccd ?13? ?12? ?12? ?6? ?12? ?11? ?11? ?12? ?8?  
    >     1              3cbcd        ?13? ?12? ?11? ?12? ?3?                4cd                  ?13? ?12? ?4?  
    >     1            3cc6ccd ?13? ?12? ?12? ?6? ?12? ?12? ?3?            8cc6ccd ?13? ?12? ?12? ?6? ?12? ?12? ?8?  
    >     1              3c6cd         ?13? ?12? ?6? ?12? ?3?          8cbbbb6cd ?13? ?12? ?6? ?11? ?11? ?11? ?11? ?12? ?8?  









