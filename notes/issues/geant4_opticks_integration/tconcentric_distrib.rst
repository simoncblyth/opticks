tconcentric_distrib
=======================


Look for presentational hists
-------------------------------

::

    In [8]: st[np.logical_and(st.st.na > 1000, st.st.distc2 > 1.1)]
    Out[8]: 
    ABStat 41 iv,is,na,nb,reclab,X,Y,Z,T,A,B,C,W,seqc2,distc2 
    === == ===== ===== ============================================== ===== ===== ===== ===== ===== ===== ===== ===== ===== ====== 
    iv  is na    nb    reclab                                         X     Y     Z     T     A     B     C     W     seqc2 distc2 
    === == ===== ===== ============================================== ===== ===== ===== ===== ===== ===== ===== ===== ===== ====== 
    27  5  20238 20140 TO RE [BT] BT BT BT SA                          1.28  1.60  0.91  1.03  1.15  0.86  1.08  1.34  0.24  1.11  
    28  5  20238 20140 TO RE BT [BT] BT BT SA                          1.31  1.65  0.80  0.96  1.18  0.87  1.15  1.34  0.24  1.13  
    29  5  20238 20140 TO RE BT BT [BT] BT SA                          1.58  1.97  1.26  0.89  1.14  0.87  1.13  1.34  0.24  1.29  
    30  5  20238 20140 TO RE BT BT BT [BT] SA                          1.46  2.11  1.21  1.07  1.17  0.82  1.17  1.34  0.24  1.30  
    31  5  20238 20140 TO RE BT BT BT BT [SA]                          1.72  2.19  1.03  1.01  1.17  0.82  1.17  1.34  0.24  1.35  
    35  6  10214 10357 TO BT BT [SC] BT BT SA                          1.62  0.00  0.00  1.24  0.93  1.32  1.10  0.00  0.99  1.11  
    51  8  7540  7710  TO BT BT BT BT [DR] SA                          0.00  0.00  0.00  0.00  1.18  1.13  1.06  0.00  1.90  1.12  
    52  8  7540  7710  TO BT BT BT BT DR [SA]                          1.12  1.17  1.16  1.22  1.18  1.13  1.06  0.00  1.90  1.14  
    69  11 5339  5269  TO BT BT BT BT [DR] BT BT BT BT BT BT BT BT SA  0.00  0.00  0.00  0.00  1.32  1.45  1.02  0.00  0.46  1.26  
    70  11 5339  5269  TO BT BT BT BT DR [BT] BT BT BT BT BT BT BT SA  0.48  1.56  0.80  1.33  1.57  1.14  0.97  0.00  0.46  1.19  
    71  11 5339  5269  TO BT BT BT BT DR BT [BT] BT BT BT BT BT BT SA  2.20  1.20  0.71  0.91  1.54  1.20  0.92  0.00  0.46  1.13  
    73  11 5339  5269  TO BT BT BT BT DR BT BT BT [BT] BT BT BT BT SA  1.13  1.21  0.82  0.85  1.50  1.23  0.96  0.00  0.46  1.10  
    75  11 5339  5269  TO BT BT BT BT DR BT BT BT BT BT [BT] BT BT SA  1.04  1.34  1.05  0.91  1.37  1.52  1.06  0.00  0.46  1.17  
    76  11 5339  5269  TO BT BT BT BT DR BT BT BT BT BT BT [BT] BT SA  1.04  1.14  1.15  0.66  1.41  1.47  1.02  0.00  0.46  1.15  
    77  11 5339  5269  TO BT BT BT BT DR BT BT BT BT BT BT BT [BT] SA  1.19  1.20  1.08  0.62  1.42  1.22  1.25  0.00  0.46  1.17  
    78  11 5339  5269  TO BT BT BT BT DR BT BT BT BT BT BT BT BT [SA]  1.23  1.34  1.08  1.93  1.42  1.22  1.25  0.00  0.46  1.23  
    85  12 5111  4940  TO BT BT RE BT BT [SA]                          1.47  1.41  1.07  1.61  1.14  1.12  0.92  0.70  2.91  1.13  
    94  14 4494  4469  TO BT BT BT BT [DR] BT BT BT BT SA              0.00  0.00  0.00  0.00  1.34  2.07  1.35  0.00  0.07  1.44  
    95  14 4494  4469  TO BT BT BT BT DR [BT] BT BT BT SA              1.91  1.37  0.87  1.59  1.43  1.58  1.27  0.00  0.07  1.29  
    96  14 4494  4469  TO BT BT BT BT DR BT [BT] BT BT SA              1.52  1.37  0.86  1.67  1.19  1.46  1.20  0.00  0.07  1.20  
    97  14 4494  4469  TO BT BT BT BT DR BT BT [BT] BT SA              1.15  1.13  1.14  2.06  1.17  1.30  1.74  0.00  0.07  1.23  
    98  14 4494  4469  TO BT BT BT BT DR BT BT BT [BT] SA              1.13  1.17  1.14  1.52  1.12  1.40  1.38  0.00  0.07  1.18  
    99  14 4494  4469  TO BT BT BT BT DR BT BT BT BT [SA]              1.32  1.06  1.16  2.55  1.12  1.40  1.38  0.00  0.07  1.20  
    118 16 2670  2675  TO SC SC BT BT BT BT [SA]                       1.05  1.31  0.65  1.12  1.08  1.30  1.01  0.00  0.00  1.10  
    145 20 1815  1805  TO RE RE RE BT BT BT [BT] SA                    1.57  2.66  0.78  1.27  1.39  0.00  0.00  0.77  0.03  1.18  
    148 21 1691  1643  TO [RE] RE AB                                   1.33  0.00  0.00  1.30  0.00  0.00  0.00  1.12  0.69  1.23  
    173 23 1544  1521  TO BT BT BT BT SC BT [BT] BT BT SA              0.96  0.83  1.28  0.98  1.39  1.26  0.25  0.00  0.17  1.11  
    176 23 1544  1521  TO BT BT BT BT SC BT BT BT BT [SA]              1.16  0.00  1.52  0.72  1.19  1.34  2.02  0.00  0.17  1.14  
    221 29 1105  1168  TO BT BT RE BT [BT] BT BT BT BT SA              3.64  0.62  1.60  0.91  0.62  0.00  0.00  1.24  1.75  1.16  
    222 29 1105  1168  TO BT BT RE BT BT [BT] BT BT BT SA              0.02  0.00  5.38  1.61  2.43  0.00  0.00  1.24  1.75  1.43  
    223 29 1105  1168  TO BT BT RE BT BT BT [BT] BT BT SA              0.00 16.13  5.83  1.58  0.39  0.00  0.00  1.24  1.75  1.67  
    224 29 1105  1168  TO BT BT RE BT BT BT BT [BT] BT SA              0.00  0.78  0.00  1.48  1.69  0.00  0.00  1.24  1.75  1.29  
    225 29 1105  1168  TO BT BT RE BT BT BT BT BT [BT] SA             11.64  0.00  0.00  1.45  1.05  0.00  0.00  1.24  1.75  1.47  
    226 29 1105  1168  TO BT BT RE BT BT BT BT BT BT [SA]              0.00  0.00  0.00  1.90  1.05  0.00  0.00  1.24  1.75  1.45  
    230 30 1132  1108  TO RE SC [BT] BT BT BT SA                       2.39  0.00  0.00  1.05  0.00  0.00  0.00  1.49  0.26  1.35  
    231 30 1132  1108  TO RE SC BT [BT] BT BT SA                       1.91  0.00  0.00  0.96  0.00  0.00  0.00  1.49  0.26  1.23  
    232 30 1132  1108  TO RE SC BT BT [BT] BT SA                       0.01  0.00  0.00  1.28  0.00  0.00  0.00  1.49  0.26  1.28  
    233 30 1132  1108  TO RE SC BT BT BT [BT] SA                       1.25  0.00  0.00  1.12  0.00  0.00  0.00  1.49  0.26  1.22  
    240 31 1067  1013  TO BT BT BT BT [DR] BT BT AB                    0.00  0.00  0.00  0.00  0.00  0.91  1.48  0.00  1.40  1.10  
    242 31 1067  1013  TO BT BT BT BT DR BT [BT] AB                    0.70  1.22  1.08  0.66  1.34  1.81  1.46  0.00  1.40  1.14  
    247 32 1036  988   TO RE BT [BT] AB                                0.98  3.04  0.38  1.11  0.00  0.00  0.00  1.14  1.14  1.11  
    === == ===== ===== ============================================== ===== ===== ===== ===== ===== ===== ===== ===== ===== ====== 





Comparing interpolation blib 
--------------------------------

::

    run bnd.py

    In [26]: i1
    Out[26]: 
    GBndLib
    /tmp/blyth/opticks/InterpolationTest/CInterpolationTest_interpol.npy
    /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GItemList/GBndLib.txt
    /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GBndLib/GBndLibOptical.npy
    20161117-1600
    20160709-1558
    20160709-1558

    In [27]: i2
    Out[27]: 
    GBndLib
    /tmp/blyth/opticks/InterpolationTest/OInterpolationTest_interpol.npy
    /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GItemList/GBndLib.txt
    /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GBndLib/GBndLibOptical.npy
    20161117-1845
    20160709-1558
    20160709-1558


    i2.dat.ijk = 20,0,1
    i1.dat.ijk = 20,0,1


    In [23]: (i1.dat.d[:,0] - i2.dat.d[:,0]).max()
    Out[23]: 0.99961853

    In [24]: (i1.dat.d[:,0] - i2.dat.d[:,0]).min()
    Out[24]: -1.8506927

MineralOil seems not especially bad GROUPVEL discrep::


    Acrylic 
    [[  -0.0077    0.        0.        0.       -1.8155    0.        0.        0.    ]
     [   0.0083   39.3328  350.2969    0.        0.9839    0.        0.        0.    ]]
    Bialkali 
    [[ 0.     -2.5545  0.      0.      0.      0.      0.      0.    ]
     [ 0.      1.9261  0.      0.      0.      0.      0.      0.    ]]
    DeadWater 
    [[ -0.0001 -91.4863   0.       0.      -0.1297   0.       0.       0.    ]
     [  0.      64.2637   0.       0.       0.0995   0.       0.       0.    ]]
    GdDopedLS 
    [[  -0.008   -88.0518    0.       -0.004    -1.8517    0.        0.        0.    ]
     [   0.0083  193.9531  350.2969    0.0014    1.0106    0.        0.        0.    ]]
    IwsWater 
    [[ -0.0001 -91.4863   0.       0.      -0.1297   0.       0.       0.    ]
     [  0.      64.2637   0.       0.       0.0995   0.       0.       0.    ]]
    LiquidScintillator 
    [[  -0.008   -94.8359    0.       -0.004    -1.8517    0.        0.        0.    ]
     [   0.0083  187.0273  350.2969    0.0014    1.0106    0.        0.        0.    ]]
    MineralOil 
    [[  -0.0076  -74.7598    0.        0.       -1.8507    0.        0.        0.    ]
     [   0.0081  173.9316  350.2969    0.        0.9996    0.        0.        0.    ]]
    OwsWater 
    [[ -0.0001 -91.4863   0.       0.      -0.1297   0.       0.       0.    ]
     [  0.      64.2637   0.       0.       0.0995   0.       0.       0.    ]]
    Pyrex 
    [[ 0.     -2.5545  0.      0.      0.      0.      0.      0.    ]
     [ 0.      1.9261  0.      0.      0.      0.      0.      0.    ]]
    Teflon 
    [[  -0.0077    0.        0.        0.       -1.8155    0.        0.        0.    ]
     [   0.0083   39.3328  350.2969    0.        0.9839    0.        0.        0.    ]]
    Water 
    [[ -0.0001 -91.4863   0.       0.      -0.1297   0.       0.       0.    ]
     [  0.      64.2637   0.       0.       0.0995   0.       0.       0.    ]]




After move to c2shape comparisons : listing seq points with sum of distrib chi2 > 20 
---------------------------------------------------------------------------------------

* T : well known groupvel interpolation issue dominates

* Notable that worst distrib chi2 offenders almost all starting "TO BT BT BT BT DR .."
  but possible that this is just because these are long lived photons
  so groupvel interpolation differences are mounting up causing the times to diverge

  the chi2 aint increaing with steps though, but adaptive binning makes
  this hard to interpret 

* BUT still machinery issues with binning...

* DONE: check chi2 with absolute bins rather then the current somewhat dodgy adaptive binning 
* DONE: combine chi2 into distc2 (with c2shape)
* TODO: revisit OpInterpolationTest OInterpolationTest and compare wavelength "scans" of GROUPVEL 
* TODO: do np.diff time/position groupvel calcs for the bad chi2 seqs 
* TODO: add W wavelength to qwns, replacing the derivative and duplicitous R (which is only useful for specific geometry origins anyhow) 





Switching R to W as R duplicates XY::

    In [2]: st[st.st.distc2 > 1.5]
    Out[2]: 
    ABStat 32 iv,is,na,nb,reclab,X,Y,Z,T,A,B,C,W,seqc2,distc2 
    === == ==== ==== ================================================= ===== ===== ===== ===== ===== ===== ===== ===== ===== ====== 
    iv  is na   nb   reclab                                            X     Y     Z     T     A     B     C     W     seqc2 distc2 
    === == ==== ==== ================================================= ===== ===== ===== ===== ===== ===== ===== ===== ===== ====== 
    223 29 1105 1168 TO BT BT RE BT BT BT [BT] BT BT SA                 0.00 16.13  5.83  1.58  0.39  0.00  0.00  1.24  1.75  1.67  
    332 43 545  539  TO BT BT BT BT DR BT BT BT BT SC BT BT BT BT [SA]  0.00  0.00  0.00  2.00  0.00  0.00  0.00  0.00  0.03  1.78  
    360 47 414  451  TO RE [BT] BT RE BT BT SA                          0.32  0.00  0.00  3.62  0.00  0.00  0.00  1.06  1.58  1.63  
    361 47 414  451  TO RE BT [BT] RE BT BT SA                          0.40  0.00  0.00  3.81  0.00  0.00  0.00  1.06  1.58  1.70  
    380 49 385  333  TO RE BT [BT] SC BT BT SA                          0.31  0.00  0.00  2.50  0.00  0.00  0.00  1.68  3.77  1.61  
    381 49 385  333  TO RE BT BT [SC] BT BT SA                          0.00  0.00  0.00  1.64  0.00  0.00  0.00  1.68  3.77  1.51  
    387 50 381  372  TO SC [BT] BT RE BT BT SA                          1.06  0.00  0.00  2.19  5.46  0.95  0.00  0.00  0.11  1.54  
    388 50 381  372  TO SC BT [BT] RE BT BT SA                          1.32  0.00  0.00  2.39  5.46  0.95  0.00  0.00  0.11  1.64  
    412 53 348  367  TO [SC] BT BT BT BT DR BT BT BT BT BT BT BT BT SA  0.00  0.00  0.00  2.16  0.00  0.94  0.00  0.00  0.50  1.56  
    441 55 339  320  TO BT BT SC BT BT [SC] BT BT BT BT SA              0.00  0.00  0.00  2.41  0.00  0.00  0.00  0.00  0.55  1.88  
    506 63 277  302  TO SC SC [AB]                                      0.00  0.00  0.00  7.18  0.00  0.00  0.00  0.00  1.08  2.39  
    513 64 295  291  TO BT BT SC SC BT [BT] SA                          2.24  0.00  0.00  2.85  0.00  0.00  0.00  0.00  0.03  1.99  
    514 64 295  291  TO BT BT SC SC BT BT [SA]                          2.24  0.00  0.00  4.43  0.00  0.00  0.00  0.00  0.03  2.22  
    570 69 255  243  TO SC BT BT SC BT BT BT BT [BT] BT SA              0.00  0.00  0.00  4.18  0.00  0.00  0.00  0.00  0.29  2.09  
    572 69 255  243  TO SC BT BT SC BT BT BT BT BT BT [SA]              0.00  0.00  0.00  6.09  0.00  0.00  0.00  0.00  0.29  2.03  
    589 71 212  239  TO BT BT BT BT DR BT BT [RE] BT BT SA              0.00  0.00  0.00  1.26  0.00  0.00  0.00  4.37  1.62  1.88  
    590 71 212  239  TO BT BT BT BT DR BT BT RE [BT] BT SA              0.00  0.00  0.00  0.00  0.00  0.00  0.00  4.37  1.62  4.37  
    591 71 212  239  TO BT BT BT BT DR BT BT RE BT [BT] SA              0.00  0.00  0.00  0.00  0.00  0.00  0.00  4.37  1.62  4.37  
    592 71 212  239  TO BT BT BT BT DR BT BT RE BT BT [SA]              0.00  0.00  0.00  0.00  0.00  0.00  0.00  4.37  1.62  4.37  
    613 73 201  231  TO BT BT SC [BT] BT SC SA                          4.35  0.00  0.00  0.46  3.03  0.00  0.00  0.00  2.08  1.58  
    636 75 166  181  TO BT BT RE BT BT RE BT [BT] BT BT SA              1.51  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.65  1.51  
    650 77 179  180  TO BT BT BT BT SC [BT] BT SC BT BT SA              0.54  5.52  0.00  2.82  0.00  0.00  0.00  0.00  0.00  1.67  
    660 78 171  154  TO RE RE RE [RE] AB                                0.00  4.45  6.69  0.00  0.00  0.00  0.00  0.00  0.89  2.79  
    667 79 147  170  TO BT BT RE RE [BT] BT AB                          5.09  0.00  0.00  1.17  0.00  0.00  0.00  0.00  1.67  1.56  
    699 82 162  158  TO SC BT BT [BT] BT DR AB                          0.00  0.00  0.00  4.64  0.00  0.00  0.00  0.00  0.05  1.55  
    701 82 162  158  TO SC BT BT BT BT [DR] AB                          0.00  0.00  0.00  5.72  0.00  0.00  0.00  0.00  0.05  1.91  
    729 85 132  150  TO RE BT BT BT BT [DR] BT BT BT BT BT BT BT BT SA  0.00  0.00  0.00  3.15  0.00  0.00  0.00  0.00  1.15  1.57  
    730 85 132  150  TO RE BT BT BT BT DR [BT] BT BT BT BT BT BT BT SA  0.00  0.00  0.00  4.77  0.00  0.00  0.00  0.00  1.15  2.38  
    731 85 132  150  TO RE BT BT BT BT DR BT [BT] BT BT BT BT BT BT SA  0.00  0.00  0.00  2.69  0.00  0.00  0.00  0.00  1.15  2.69  
    744 86 135  145  TO RE RE BT BT [RE] BT BT SA                       0.00  0.00  0.00  9.35  0.00  0.00  0.00  0.00  0.36  4.67  
    789 91 130  119  TO BT BT SC [RE] BT BT SA                          0.00  0.00  0.00  1.65  0.00  0.00  0.00  0.00  0.49  1.65  
    825 94 127  106  TO BT BT SC [BT] BT RE BT BT BT BT SA              4.61  0.00  0.00  0.00  0.00  0.00  0.00  0.00  1.89  1.54  
    === == ==== ==== ================================================= ===== ===== ===== ===== ===== ===== ===== ===== ===== ====== 




After the GROUPVEL material fix::

    In [4]: st[st.st.distc2 > 2]
    Out[4]: 
    ABStat 19 iv,is,na,nb,reclab,X,Y,Z,T,A,B,C,R,seqc2,distc2 
    === == === === ================================================= ===== ===== ===== ===== ===== ===== ===== ===== ===== ====== 
    iv  is na  nb  reclab                                            X     Y     Z     T     A     B     C     R     seqc2 distc2 
    === == === === ================================================= ===== ===== ===== ===== ===== ===== ===== ===== ===== ====== 
    360 47 414 451 TO RE [BT] BT RE BT BT SA                          0.32  0.00  0.00  3.62  0.00  0.00  0.00  2.75  1.58  2.19  
    361 47 414 451 TO RE BT [BT] RE BT BT SA                          0.40  0.00  0.00  3.81  0.00  0.00  0.00  2.02  1.58  2.12  
    441 55 339 320 TO BT BT SC BT BT [SC] BT BT BT BT SA              0.00  0.00  0.00  2.41  0.00  0.00  0.00  0.00  0.55  2.11  
    506 63 277 302 TO SC SC [AB]                                      0.00  0.00  0.00  7.18  0.00  0.00  0.00  0.00  1.08  3.59  
    570 69 255 243 TO SC BT BT SC BT BT BT BT [BT] BT SA              0.00  0.00  0.00  4.18  0.00  0.00  0.00  0.59  0.29  2.39  
    572 69 255 243 TO SC BT BT SC BT BT BT BT BT BT [SA]              0.00  0.00  0.00  6.09  0.00  0.00  0.00  0.00  0.29  3.04  
    592 71 212 239 TO BT BT BT BT DR BT BT RE BT BT [SA]              0.00  0.00  0.00  0.00  0.00  0.00  0.00  3.36  1.62  3.36  
    624 74 164 202 TO BT BT SC BT BT BT [BT] BT BT AB                 0.00  0.00  0.00  0.00  0.00  0.00  0.00  4.43  3.95  4.43  
    635 75 166 181 TO BT BT RE BT BT RE [BT] BT BT BT SA              1.08  0.00  0.00  0.00  0.00  0.00  0.00  4.13  0.65  2.33  
    638 75 166 181 TO BT BT RE BT BT RE BT BT BT [BT] SA              0.00  0.00  0.00  0.00  0.00  0.00  0.00  4.85  0.65  4.85  
    653 77 179 180 TO BT BT BT BT SC BT BT SC [BT] BT SA              0.23  0.00  0.00  0.00  0.00  0.00  0.00  4.19  0.00  2.15  
    660 78 171 154 TO RE RE RE [RE] AB                                0.00  4.45  6.69  0.00  0.00  0.00  0.00  0.00  0.89  2.79  
    699 82 162 158 TO SC BT BT [BT] BT DR AB                          0.00  0.00  0.00  4.64  0.00  0.00  0.00  0.00  0.05  2.32  
    701 82 162 158 TO SC BT BT BT BT [DR] AB                          0.00  0.00  0.00  5.72  0.00  0.00  0.00  0.00  0.05  2.86  
    732 85 132 150 TO RE BT BT BT BT DR BT BT [BT] BT BT BT BT BT SA  0.00  0.00  0.00  0.00  0.00  0.00  0.00  2.92  1.15  2.92  
    734 85 132 150 TO RE BT BT BT BT DR BT BT BT BT [BT] BT BT BT SA  0.00  0.00  0.00  0.00  0.00  0.00  0.00  3.89  1.15  3.89  
    735 85 132 150 TO RE BT BT BT BT DR BT BT BT BT BT [BT] BT BT SA  0.00  0.00  0.00  0.00  0.00  0.00  0.00  2.64  1.15  2.64  
    744 86 135 145 TO RE RE BT BT [RE] BT BT SA                       0.00  0.00  0.00  9.35  0.00  0.00  0.00  0.00  0.36  4.67  
    745 86 135 145 TO RE RE BT BT RE [BT] BT SA                       0.00  0.00  0.00  0.00  0.00  0.00  0.00  4.68  0.36  4.68  
    === == === === ================================================= ===== ===== ===== ===== ===== ===== ===== ===== ===== ======  


::

    tconcentric-;tconcentric-d --noplot --rehist --sel 0:100    # recreate histograms for first 100 seq lines 




    ip>  run abstat.py   # load and examine the stats


    In [12]: st[st.st.distc2 > 10]
    Out[12]: 
    ABStat 17 iv,is,na,nb,reclab,X,Y,Z,T,A,B,C,R,seqc2,distc2 
    === == ====== ====== ================================================= ===== ===== ===== ========== ===== ===== ===== ===== ===== ========= 
    iv  is na     nb     reclab                                            X     Y     Z     T          A     B     C     R     seqc2 distc2    
    === == ====== ====== ================================================= ===== ===== ===== ========== ===== ===== ===== ===== ===== ========= 
    5   0  669843 671267 TO BT BT BT BT [SA]                                0.00  0.00  0.00 1341110.00  0.00  0.00  0.00  0.00  1.51 191587.14 
    51  8  7540   7694   TO BT BT BT BT [DR] SA                             0.00  0.00  0.00 15234.00    1.06  1.11  1.07  0.00  1.56 33.61     
    69  11 5339   5367   TO BT BT BT BT [DR] BT BT BT BT BT BT BT BT SA     0.00  0.00  0.00 10706.00    1.07  1.10  0.89  0.00  0.07 54.54     
    94  14 4494   4420   TO BT BT BT BT [DR] BT BT BT BT SA                 0.00  0.00  0.00 8914.00     1.33  2.02  1.60  0.00  0.61 34.49     
    124 17 2432   2472   TO BT BT BT BT [DR] AB                             0.00  0.00  0.00 4904.00     1.61  0.92  1.27  0.00  0.33 35.86     
    190 25 1260   1263   TO BT BT BT BT [DR] BT BT BT BT AB                 0.00  0.00  0.00 2523.00     0.30  1.09  0.62  0.00  0.00 55.57     
    240 31 1067   1019   TO BT BT BT BT [DR] BT BT AB                       0.00  0.00  0.00 2086.00     0.00  1.23  1.16  0.00  1.10 30.47     
    269 36 933    958    TO BT BT BT BT [DR] SC SA                          0.00  0.00  0.00 1891.00     0.00  1.58  0.86  0.00  0.33 58.27     
    312 42 545    566    TO BT BT BT BT [DR] BT BT BT BT SC BT BT BT BT SA  0.00  0.00  0.00 1111.00     0.00  0.81  0.76  0.00  0.40 40.29     
    346 45 507    517    TO BT BT BT BT [DR] BT BT SC BT BT SA              0.00  0.00  0.00 1024.00     0.00  0.40  0.71  0.00  0.10 49.16     
    532 66 285    239    TO BT BT BT BT [DR] BT BT BT BT BT BT AB           0.00  0.00  0.00 524.00      0.00  0.73  1.66  0.00  4.04 44.26     
    545 67 266    270    TO BT BT BT BT [DR] BT BT BT BT BT BT BT BT AB     0.00  0.00  0.00 536.00      0.00  2.08  1.44  0.00  0.03 42.20     
    578 70 212    242    TO BT BT BT BT [DR] BT BT RE BT BT SA              0.00  0.00  0.00 454.00      0.00  0.48  0.00  0.00  1.98 64.99     
    590 71 237    222    TO BT BT BT BT [DR] BT BT BT BT RE BT BT BT BT SA  0.00  0.00  0.00 459.00      0.00  0.55  0.28  0.00  0.49 35.54     
    794 91 126    131    TO BT BT BT BT [DR] BT BT BT BT BT BT BT BT DR BT  0.00  0.00  0.00 257.00      0.00  0.29  2.55  0.00  0.10 43.31     
    826 94 129    117    TO BT BT BT BT [DR] SC BT BT BT BT SA              0.00  0.00  0.00 246.00      0.00  0.00  0.00  0.00  0.59 61.50     
    886 99 126    123    TO BT BT BT BT [DR] BT BT SC BT BT BT BT BT BT SA  0.00  0.00  0.00 249.00      0.00  2.66  0.00  0.00  0.04 41.94     
    === == ====== ====== ================================================= ===== ===== ===== ========== ===== ===== ===== ===== ===== ========= 

    ## looks like reclab irec stuck ???

    In [17]: st[st.st.distc2 > 5]
    Out[17]: 
    ABStat 21 iv,is,na,nb,reclab,X,Y,Z,T,A,B,C,R,seqc2,distc2 
    === == ====== ====== ================================================= ===== ===== ===== ========== ===== ===== ===== ===== ===== ========= 
    iv  is na     nb     reclab                                            X     Y     Z     T          A     B     C     R     seqc2 distc2    
    === == ====== ====== ================================================= ===== ===== ===== ========== ===== ===== ===== ===== ===== ========= 
    5   0  669843 671267 TO BT BT BT BT [SA]                                0.00  0.00  0.00 1341110.00  0.00  0.00  0.00  0.00  1.51 191587.14 
    20  3  28955  28649  TO BT BT BT BT [AB]                                1.23  0.00  0.00 105.35      0.00  0.00  0.00  1.23  1.63  9.60     
    51  8  7540   7694   TO BT BT BT BT [DR] SA                             0.00  0.00  0.00 15234.00    1.06  1.11  1.07  0.00  1.56 33.61     
    69  11 5339   5367   TO BT BT BT BT [DR] BT BT BT BT BT BT BT BT SA     0.00  0.00  0.00 10706.00    1.07  1.10  0.89  0.00  0.07 54.54     
    94  14 4494   4420   TO BT BT BT BT [DR] BT BT BT BT SA                 0.00  0.00  0.00 8914.00     1.33  2.02  1.60  0.00  0.61 34.49     
    124 17 2432   2472   TO BT BT BT BT [DR] AB                             0.00  0.00  0.00 4904.00     1.61  0.92  1.27  0.00  0.33 35.86     
    190 25 1260   1263   TO BT BT BT BT [DR] BT BT BT BT AB                 0.00  0.00  0.00 2523.00     0.30  1.09  0.62  0.00  0.00 55.57     
    240 31 1067   1019   TO BT BT BT BT [DR] BT BT AB                       0.00  0.00  0.00 2086.00     0.00  1.23  1.16  0.00  1.10 30.47     
    269 36 933    958    TO BT BT BT BT [DR] SC SA                          0.00  0.00  0.00 1891.00     0.00  1.58  0.86  0.00  0.33 58.27     
    312 42 545    566    TO BT BT BT BT [DR] BT BT BT BT SC BT BT BT BT SA  0.00  0.00  0.00 1111.00     0.00  0.81  0.76  0.00  0.40 40.29     
    346 45 507    517    TO BT BT BT BT [DR] BT BT SC BT BT SA              0.00  0.00  0.00 1024.00     0.00  0.40  0.71  0.00  0.10 49.16     
    532 66 285    239    TO BT BT BT BT [DR] BT BT BT BT BT BT AB           0.00  0.00  0.00 524.00      0.00  0.73  1.66  0.00  4.04 44.26     
    545 67 266    270    TO BT BT BT BT [DR] BT BT BT BT BT BT BT BT AB     0.00  0.00  0.00 536.00      0.00  2.08  1.44  0.00  0.03 42.20     
    578 70 212    242    TO BT BT BT BT [DR] BT BT RE BT BT SA              0.00  0.00  0.00 454.00      0.00  0.48  0.00  0.00  1.98 64.99     
    590 71 237    222    TO BT BT BT BT [DR] BT BT BT BT RE BT BT BT BT SA  0.00  0.00  0.00 459.00      0.00  0.55  0.28  0.00  0.49 35.54     
    591 71 237    222    TO BT BT BT BT DR [BT] BT BT BT RE BT BT BT BT SA  0.82  0.00  0.00 40.35       0.00  1.16  0.74  0.06  0.49  7.66     
    592 71 237    222    TO BT BT BT BT DR BT [BT] BT BT RE BT BT BT BT SA  0.33  0.00  0.00 30.58       0.00  1.16  0.74  0.00  0.49  5.79     
    660 78 167    168    TO BT BT RE BT BT RE BT BT BT [BT] SA              0.00  0.00  0.00  0.00       0.00  0.00  0.00  5.48  0.00  5.48     
    794 91 126    131    TO BT BT BT BT [DR] BT BT BT BT BT BT BT BT DR BT  0.00  0.00  0.00 257.00      0.00  0.29  2.55  0.00  0.10 43.31     
    826 94 129    117    TO BT BT BT BT [DR] SC BT BT BT BT SA              0.00  0.00  0.00 246.00      0.00  0.00  0.00  0.00  0.59 61.50     
    886 99 126    123    TO BT BT BT BT [DR] BT BT SC BT BT BT BT BT BT SA  0.00  0.00  0.00 249.00      0.00  2.66  0.00  0.00  0.04 41.94     
    === == ====== ====== ================================================= ===== ===== ===== ========== ===== ===== ===== ===== ===== ========= 


Maybe not stuck, perhaps a problem with GROUPVEL in MineralOil ?::

    In [14]: st[312:322]
    Out[14]: 
    ABStat 10 iv,is,na,nb,reclab,X,Y,Z,T,A,B,C,R,seqc2,distc2 
    === == === === ================================================= ===== ===== ===== ======= ===== ===== ===== ===== ===== ====== 
    iv  is na  nb  reclab                                            X     Y     Z     T       A     B     C     R     seqc2 distc2 
    === == === === ================================================= ===== ===== ===== ======= ===== ===== ===== ===== ===== ====== 
    312 42 545 566 TO BT BT BT BT [DR] BT BT BT BT SC BT BT BT BT SA  0.00  0.00  0.00 1111.00  0.00  0.81  0.76  0.00  0.40 40.29  
    313 42 545 566 TO BT BT BT BT DR [BT] BT BT BT SC BT BT BT BT SA  0.04  0.87  1.35 66.68    0.01  1.07  0.81  1.02  0.40  3.15  
    314 42 545 566 TO BT BT BT BT DR BT [BT] BT BT SC BT BT BT BT SA  0.73  0.91  1.13 47.93    0.41  1.04  0.71  1.24  0.40  2.45  
    315 42 545 566 TO BT BT BT BT DR BT BT [BT] BT SC BT BT BT BT SA  0.66  0.00  1.43  8.92    0.00  0.96  0.96  0.12  0.40  2.01  
    316 42 545 566 TO BT BT BT BT DR BT BT BT [BT] SC BT BT BT BT SA  0.43  1.85  0.00  7.79    0.41  0.85  0.76  0.73  0.40  1.75  
    317 42 545 566 TO BT BT BT BT DR BT BT BT BT [SC] BT BT BT BT SA  0.00  0.00  0.00  0.52    0.00  0.00  0.00  2.35  0.40  0.56  
    318 42 545 566 TO BT BT BT BT DR BT BT BT BT SC [BT] BT BT BT SA  0.00  0.00  0.00  0.71    0.00  0.00  0.00  0.27  0.40  0.55  
    319 42 545 566 TO BT BT BT BT DR BT BT BT BT SC BT [BT] BT BT SA  0.00  0.00  0.00  0.79    0.00  0.00  0.00  0.66  0.40  0.70  
    320 42 545 566 TO BT BT BT BT DR BT BT BT BT SC BT BT [BT] BT SA  0.00  0.00  0.00  1.03    0.00  0.00  0.00  2.73  0.40  1.39  
    321 42 545 566 TO BT BT BT BT DR BT BT BT BT SC BT BT BT [BT] SA  0.00  0.00  0.00  1.22    0.00  0.00  0.00  0.24  0.40  0.87  
    === == === === ================================================= ===== ===== ===== ======= ===== ===== ===== ===== ===== ====== 



Load the 8 qwn point histos::

    cfh-;cfh "TO BT BT BT BT [AB]"

* note that auto-binning is coming up with too few time bins here


DONE machinery shakedown
-----------------------------

* adopt less expensive approach

  * eg do not need to spawn CF for all seqhis lines, now that can easily switch psel 
  * decouple distrib chi2 from plotting 
  * develop summary info on the distrib chi2, available without plotting 

* fix chi2 handling for trivial same distrib


multiplot slice(0,10) quick look
----------------------------------

* t discrep, known GROUPVEL problem still there : now that have G4 and OP live both
  at once can fix this 

* RESOLVED : yz polarization distribs followin DR SURFACE_DREFLECT are discrepant, see  :doc:`SURFACE_DREFLECT_diffuse_reflection` 


tconcentric agreement sufficient to move on to distribs 
----------------------------------------------------------

::

    imon:geant4_opticks_integration blyth$ tconcentric.py 
    /Users/blyth/opticks/ana/tconcentric.py
    [2016-11-07 21:02:25,728] p57180 {/Users/blyth/opticks/ana/tconcentric.py:208} INFO - tag 1 src torch det concentric c2max 2.0 ipython False 
    [2016-11-07 21:02:26,521] p57180 {/Users/blyth/opticks/ana/evt.py:400} INFO - pflags2(=seq2msk(seqhis)) and pflags  match
    [2016-11-07 21:02:26,823] p57180 {/Users/blyth/opticks/ana/evt.py:474} WARNING - _init_selection with psel None : resetting selection to original 
    [2016-11-07 21:02:29,802] p57180 {/Users/blyth/opticks/ana/evt.py:400} INFO - pflags2(=seq2msk(seqhis)) and pflags  match
    [2016-11-07 21:02:30,100] p57180 {/Users/blyth/opticks/ana/evt.py:474} WARNING - _init_selection with psel None : resetting selection to original 
    CF a concentric/torch/  1 :  20161107-1741 maxbounce:15 maxrec:16 maxrng:3000000 /tmp/blyth/opticks/evt/concentric/torch/1/fdom.npy 
    CF b concentric/torch/ -1 :  20161107-1741 maxbounce:15 maxrec:16 maxrng:3000000 /tmp/blyth/opticks/evt/concentric/torch/-1/fdom.npy 
    [2016-11-07 21:02:32,288] p57180 {/Users/blyth/opticks/ana/seq.py:410} INFO - compare dbgseq 0 dbgmsk 0 
    .                seqhis_ana      noname       noname           c2           ab           ba 
    .                               1000000      1000000       329.90/352 =  0.94  (pval:0.796 prob:0.204)  
       0               8ccccd        669843       671267             1.51        0.998 +- 0.001        1.002 +- 0.001  [6 ] TO BT BT BT BT SA
       1                   4d         83950        83637             0.58        1.004 +- 0.003        0.996 +- 0.003  [2 ] TO AB
       2              8cccc6d         45490        45054             2.10        1.010 +- 0.005        0.990 +- 0.005  [7 ] TO SC BT BT BT BT SA
       3               4ccccd         28955        28649             1.63        1.011 +- 0.006        0.989 +- 0.006  [6 ] TO BT BT BT BT AB
       4                 4ccd         23187        23254             0.10        0.997 +- 0.007        1.003 +- 0.007  [4 ] TO BT BT AB
       5              8cccc5d         20239        19946             2.14        1.015 +- 0.007        0.986 +- 0.007  [7 ] TO RE BT BT BT BT SA
       6              86ccccd         10176        10396             2.35        0.979 +- 0.010        1.022 +- 0.010  [7 ] TO BT BT BT BT SC SA
       7              8cc6ccd         10214        10304             0.39        0.991 +- 0.010        1.009 +- 0.010  [7 ] TO BT BT SC BT BT SA
       8              89ccccd          7605         7694             0.52        0.988 +- 0.011        1.012 +- 0.012  [7 ] TO BT BT BT BT DR SA
       9             8cccc55d          5970         5814             2.07        1.027 +- 0.013        0.974 +- 0.013  [8 ] TO RE RE BT BT BT BT SA
      10                  45d          5780         5658             1.30        1.022 +- 0.013        0.979 +- 0.013  [3 ] TO RE AB
      11      8cccccccc9ccccd          5348         5367             0.03        0.996 +- 0.014        1.004 +- 0.014  [15] TO BT BT BT BT DR BT BT BT BT BT BT BT BT SA
      12              8cc5ccd          5113         4868             6.01        1.050 +- 0.015        0.952 +- 0.014  [7 ] TO BT BT RE BT BT SA
      13                  46d          4797         4815             0.03        0.996 +- 0.014        1.004 +- 0.014  [3 ] TO SC AB
      14          8cccc9ccccd          4525         4420             1.23        1.024 +- 0.015        0.977 +- 0.015  [11] TO BT BT BT BT DR BT BT BT BT SA
      15          8cccccc6ccd          3317         3333             0.04        0.995 +- 0.017        1.005 +- 0.017  [11] TO BT BT SC BT BT BT BT BT BT SA
      16             8cccc66d          2670         2734             0.76        0.977 +- 0.019        1.024 +- 0.020  [8 ] TO SC SC BT BT BT BT SA
      17              49ccccd          2312         2472             5.35        0.935 +- 0.019        1.069 +- 0.022  [7 ] TO BT BT BT BT DR AB
      18              4cccc6d          2043         2042             0.00        1.000 +- 0.022        1.000 +- 0.022  [7 ] TO SC BT BT BT BT AB
      19            8cccc555d          1819         1762             0.91        1.032 +- 0.024        0.969 +- 0.023  [9 ] TO RE RE RE BT BT BT BT SA
    .                               1000000      1000000       329.90/352 =  0.94  (pval:0.796 prob:0.204)  
    [2016-11-07 21:02:32,429] p57180 {/Users/blyth/opticks/ana/seq.py:410} INFO - compare dbgseq 0 dbgmsk 0 
    .                pflags_ana  1:concentric   -1:concentric           c2           ab           ba 
    .                               1000000      1000000        50.71/42 =  1.21  (pval:0.168 prob:0.832)  
       0                 1880        669843       671267             1.51        0.998 +- 0.001        1.002 +- 0.001  [3 ] TO|BT|SA
       1                 1008         83950        83637             0.58        1.004 +- 0.003        0.996 +- 0.003  [2 ] TO|AB
       2                 18a0         79906        79772             0.11        1.002 +- 0.004        0.998 +- 0.004  [4 ] TO|BT|SA|SC
       3                 1808         54172        53852             0.95        1.006 +- 0.004        0.994 +- 0.004  [3 ] TO|BT|AB
       4                 1890         38518        37832             6.16        1.018 +- 0.005        0.982 +- 0.005  [4 ] TO|BT|SA|RE
       5                 1980         17803        17843             0.04        0.998 +- 0.007        1.002 +- 0.008  [4 ] TO|BT|DR|SA
       6                 1828          8788         9013             2.84        0.975 +- 0.010        1.026 +- 0.011  [4 ] TO|BT|SC|AB
       7                 1018          8204         8002             2.52        1.025 +- 0.011        0.975 +- 0.011  [3 ] TO|RE|AB
       8                 18b0          7901         7879             0.03        1.003 +- 0.011        0.997 +- 0.011  [5 ] TO|BT|SA|SC|RE
       9                 1818          6024         5941             0.58        1.014 +- 0.013        0.986 +- 0.013  [4 ] TO|BT|RE|AB
      10                 1908          5425         5463             0.13        0.993 +- 0.013        1.007 +- 0.014  [4 ] TO|BT|DR|AB
      11                 1028          5089         5153             0.40        0.988 +- 0.014        1.013 +- 0.014  [3 ] TO|SC|AB
      12                 19a0          4963         4928             0.12        1.007 +- 0.014        0.993 +- 0.014  [5 ] TO|BT|DR|SA|SC
      13                 1990          1506         1541             0.40        0.977 +- 0.025        1.023 +- 0.026  [5 ] TO|BT|DR|SA|RE
      14                 1838          1540         1535             0.01        1.003 +- 0.026        0.997 +- 0.025  [5 ] TO|BT|SC|RE|AB
      15                 1928          1048         1085             0.64        0.966 +- 0.030        1.035 +- 0.031  [5 ] TO|BT|DR|SC|AB
      16                 1038           770          776             0.02        0.992 +- 0.036        1.008 +- 0.036  [4 ] TO|SC|RE|AB
      17                 1920           775          759             0.17        1.021 +- 0.037        0.979 +- 0.036  [4 ] TO|BT|DR|SC
      18                 1918           619          609             0.08        1.016 +- 0.041        0.984 +- 0.040  [5 ] TO|BT|DR|RE|AB
      19                 1910           482          410             5.81        1.176 +- 0.054        0.851 +- 0.042  [4 ] TO|BT|DR|RE
    .                               1000000      1000000        50.71/42 =  1.21  (pval:0.168 prob:0.832)  
    [2016-11-07 21:02:32,459] p57180 {/Users/blyth/opticks/ana/seq.py:410} INFO - compare dbgseq 0 dbgmsk 0 
    .                seqmat_ana      noname       noname           c2           ab           ba 
    .                               1000000      1000000       206.81/228 =  0.91  (pval:0.840 prob:0.160)  
       0               343231        669845       671267             1.51        0.998 +- 0.001        1.002 +- 0.001  [6 ] Gd Ac LS Ac MO Ac
       1                   11         83950        83637             0.58        1.004 +- 0.003        0.996 +- 0.003  [2 ] Gd Gd
       2              3432311         65732        65001             4.09        1.011 +- 0.004        0.989 +- 0.004  [7 ] Gd Gd Ac LS Ac MO Ac
       3               443231         28955        28649             1.63        1.011 +- 0.006        0.989 +- 0.006  [6 ] Gd Ac LS Ac MO MO
       4                 2231         23188        23254             0.09        0.997 +- 0.007        1.003 +- 0.007  [4 ] Gd Ac LS LS
       5              3443231         17781        18090             2.66        0.983 +- 0.007        1.017 +- 0.008  [7 ] Gd Ac LS Ac MO MO Ac
       6              3432231         15327        15172             0.79        1.010 +- 0.008        0.990 +- 0.008  [7 ] Gd Ac LS LS Ac MO Ac
       7             34323111         10934        10826             0.54        1.010 +- 0.010        0.990 +- 0.010  [8 ] Gd Gd Gd Ac LS Ac MO Ac
       8                  111         10577        10474             0.50        1.010 +- 0.010        0.990 +- 0.010  [3 ] Gd Gd Gd
       9      343231323443231          6964         7001             0.10        0.995 +- 0.012        1.005 +- 0.012  [15] Gd Ac LS Ac MO MO Ac LS Ac Gd Ac LS Ac MO Ac
      10          34323443231          6069         5954             1.10        1.019 +- 0.013        0.981 +- 0.013  [11] Gd Ac LS Ac MO MO Ac LS Ac MO Ac
      11          34323132231          4422         4532             1.35        0.976 +- 0.015        1.025 +- 0.015  [11] Gd Ac LS LS Ac Gd Ac LS Ac MO Ac
      12              4443231          3040         3272             8.53        0.929 +- 0.017        1.076 +- 0.019  [7 ] Gd Ac LS Ac MO MO MO
      13              4432311          3008         3002             0.01        1.002 +- 0.018        0.998 +- 0.018  [7 ] Gd Gd Ac LS Ac MO MO
      14            343231111          2859         2860             0.00        1.000 +- 0.019        1.000 +- 0.019  [9 ] Gd Gd Gd Gd Ac LS Ac MO Ac
      15                22311          2791         2754             0.25        1.013 +- 0.019        0.987 +- 0.019  [5 ] Gd Gd Ac LS LS
      16                 1111          2446         2437             0.02        1.004 +- 0.020        0.996 +- 0.020  [4 ] Gd Gd Gd Gd
      17             34322311          1999         1869             4.37        1.070 +- 0.024        0.935 +- 0.022  [8 ] Gd Gd Ac LS LS Ac MO Ac
      18             34322231          1844         1872             0.21        0.985 +- 0.023        1.015 +- 0.023  [8 ] Gd Ac LS LS LS Ac MO Ac
      19                22231          1790         1825             0.34        0.981 +- 0.023        1.020 +- 0.024  [5 ] Gd Ac LS LS LS
    .                               1000000      1000000       206.81/228 =  0.91  (pval:0.840 prob:0.160)  
    [2016-11-07 21:02:32,513] p57180 {/Users/blyth/opticks/ana/evt.py:750} WARNING - missing a_ana hflags_ana 
    [2016-11-07 21:02:32,513] p57180 {/Users/blyth/opticks/ana/tconcentric.py:213} INFO - early exit as non-interactive


