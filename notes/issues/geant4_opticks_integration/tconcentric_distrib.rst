tconcentric_distrib
=======================



After move to c2shape comparisons : listing seq points with sum of distrib chi2 > 20 
---------------------------------------------------------------------------------------

* T : well known groupvel interpolation issue dominates

* Notable that worst distrib chi2 offenders almost all starting "TO BT BT BT BT DR .."
  but possible that this is just because these are long lived photons
  so groupvel interpolation differences are mounting up causing the times to diverge

  the chi2 aint increaing with steps though, but adaptive binning makes
  this hard to interpret 


* BUT still machinery issues with binning...


* TODO: check chi2 with absolute bins rather then the current somewhat dodgy adaptive binning 
* TODO: work out how to combine chi2 into distc2 (with c2shape)
* TODO: revisit OpInterpolationTest OInterpolationTest and compare wavelength "scans" of GROUPVEL 
* TODO: do np.diff time/position groupvel calcs for the bad chi2 seqs 
* TODO: add W wavelength to qwns, replacing the derivative and duplicitous R (which is only useful for specific geometry origins anyhow) 



::

    tconcentric-d --noplot --rehist --sel 0:100    # recreate histograms for first 100 seq lines 


    ip>  run abstat.py   # load and examine the stats


    In [8]: st[np.where( np.sum(ar, axis=1) > 20 )]
    Out[8]: 
    ABStat 22 iv,is,na,nb,reclab,X,Y,Z,T,A,B,C,R,seqc2,distc2 
    === == ===== ===== ================================================= ===== ===== ===== ====== ===== ===== ===== ===== ===== ====== 
    iv  is na    nb    reclab                                            X     Y     Z     T      A     B     C     R     seqc2 distc2 
    === == ===== ===== ================================================= ===== ===== ===== ====== ===== ===== ===== ===== ===== ====== 
    20  3  28955 28649 TO BT BT BT BT [AB]                                1.54  0.00  0.00 21.68   0.00  0.00  0.00  1.54  1.63  0.00  

    70  11 5339  5367  TO BT BT BT BT DR [BT] BT BT BT BT BT BT BT SA     0.20  0.89  0.36 279.71  1.27  1.21  0.95  0.08  0.07  0.00  
    71  11 5339  5367  TO BT BT BT BT DR BT [BT] BT BT BT BT BT BT SA     0.19  1.03  0.63 265.43  1.19  1.11  1.07  0.00  0.07  0.00  
    72  11 5339  5367  TO BT BT BT BT DR BT BT [BT] BT BT BT BT BT SA     1.86  0.99  0.45 106.13  1.08  1.17  0.99  0.66  0.07  0.00  
    73  11 5339  5367  TO BT BT BT BT DR BT BT BT [BT] BT BT BT BT SA     1.50  1.23  0.28 44.62   1.16  1.13  1.09  0.54  0.07  0.00  
    76  11 5339  5367  TO BT BT BT BT DR BT BT BT BT BT BT [BT] BT SA     1.46  1.37  1.03 23.99   1.11  1.39  1.03  1.39  0.07  0.00  
    77  11 5339  5367  TO BT BT BT BT DR BT BT BT BT BT BT BT [BT] SA     1.57  1.37  0.94 24.41   0.96  1.25  1.19  1.26  0.07  0.00  
    78  11 5339  5367  TO BT BT BT BT DR BT BT BT BT BT BT BT BT [SA]     1.05  1.24  0.99 79.07   0.96  1.25  1.19  1.01  0.07  0.00  
    ##            straight thru, diffuse reflect then thru all 8 layers to
 
    95  14 4494  4420  TO BT BT BT BT DR [BT] BT BT BT SA                 3.26  1.15  1.22 20.52   1.34  1.55  1.22  0.47  0.61  0.00  
    96  14 4494  4420  TO BT BT BT BT DR BT [BT] BT BT SA                 3.80  1.41  1.34 18.53   1.27  1.48  1.30  0.70  0.61  0.00  
    97  14 4494  4420  TO BT BT BT BT DR BT BT [BT] BT SA                 1.94  1.04  1.23 10.81   1.35  1.11  1.63  1.91  0.61  0.00  
    98  14 4494  4420  TO BT BT BT BT DR BT BT BT [BT] SA                 1.86  1.30  1.52 13.31   1.11  1.29  1.55  1.78  0.61  0.00  
    99  14 4494  4420  TO BT BT BT BT DR BT BT BT BT [SA]                 1.82  1.09  1.33 44.96   1.11  1.29  1.55  1.21  0.61  0.00  
    ##                         4 layers after DR             


    191 25 1260  1263  TO BT BT BT BT DR [BT] BT BT BT AB                 4.67  1.38  1.29 74.20   0.74  1.13  0.52  2.45  0.00  0.00  
    192 25 1260  1263  TO BT BT BT BT DR BT [BT] BT BT AB                 5.07  1.24  0.67 11.46   0.51  1.60  0.57  2.37  0.00  0.00  

    241 31 1067  1019  TO BT BT BT BT DR [BT] BT AB                       1.58  1.48  0.98 20.81   1.32  1.83  0.92  1.04  1.10  0.00  
    242 31 1067  1019  TO BT BT BT BT DR BT [BT] AB                       0.82  1.40  0.96 20.32   1.29  2.07  1.01  0.53  1.10  0.00  

    313 42 545   566   TO BT BT BT BT DR [BT] BT BT BT SC BT BT BT BT SA  0.04  1.18  1.35 78.15   0.01  1.07  0.81  0.01  0.40  0.00  
    314 42 545   566   TO BT BT BT BT DR BT [BT] BT BT SC BT BT BT BT SA  0.01  1.03  1.32 77.45   0.41  1.04  0.71  0.01  0.40  0.00  

    546 67 266   270   TO BT BT BT BT DR [BT] BT BT BT BT BT BT BT AB     0.80  2.05  0.71 15.21   0.00  0.61  1.60  0.15  0.03  0.00  

    591 71 237   222   TO BT BT BT BT DR [BT] BT BT BT RE BT BT BT BT SA  0.89  1.25  0.57 17.58   0.00  1.16  0.74  0.01  0.49  0.00  
    592 71 237   222   TO BT BT BT BT DR BT [BT] BT BT RE BT BT BT BT SA  2.18  1.36  0.66 17.57   0.00  1.16  0.74  0.01  0.49  0.00  
    === == ===== ===== ================================================= ===== ===== ===== ====== ===== ===== ===== ===== ===== ====== 

    ABStat 22 iv,is,na,nb,reclab,X,Y,Z,T,A,B,C,R,seqc2,distc2 


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


