ncsg_okg4_shakedown
======================



tboolean-torus
-----------------

::

    tboolean-;tboolean-torus --okg4 
    tboolean-;tboolean-torus --okg4 --load --vizg4



NEXT
------

* currently using test material GlassShottF2, move to material care about 
* do some purely positional checks : profiting from the identical input photons 
  maybe add MaxVacuum with FLT_MAX extreme absorption_length   scattering_length



SC/AB in Vacuum
------------------

::

    simon:opticks blyth$ op --mat OpaqueVacuum
    === op-cmdline-binary-match : finds 1st argument with associated binary : --mat
    ubin /usr/local/opticks/lib/GMaterialLibTest cfm --mat cmdline --mat OpaqueVacuum
    === op-export : OPTICKS_BINARY /usr/local/opticks/lib/GMaterialLibTest
    256 -rwxr-xr-x  1 blyth  staff  129400 Nov  1 14:45 /usr/local/opticks/lib/GMaterialLibTest
    proceeding.. : /usr/local/opticks/lib/GMaterialLibTest --mat OpaqueVacuum
      0 : /usr/local/opticks/lib/GMaterialLibTest
      1 : --mat
      2 : OpaqueVacuum
    option '--mat' is ambiguous and matches '--materialdbg', and '--materialprefix'
    2017-11-01 20:58:00.795 INFO  [2230869] [main@109]  ok 
    2017-11-01 20:58:00.799 INFO  [2230869] [GMaterialLib::postLoadFromCache@67] GMaterialLib::postLoadFromCache  nore 0 noab 0 nosc 0 xxre 0 xxab 0 xxsc 0 fxre 0 fxab 0 fxsc 0 groupvel 1
    2017-11-01 20:58:00.799 INFO  [2230869] [GMaterialLib::replaceGROUPVEL@552] GMaterialLib::replaceGROUPVEL  ni 38
    2017-11-01 20:58:00.799 INFO  [2230869] [GPropertyLib::getIndex@338] GPropertyLib::getIndex type GMaterialLib TRIGGERED A CLOSE  shortname [GdDopedLS]
    2017-11-01 20:58:00.800 INFO  [2230869] [GPropertyLib::close@384] GPropertyLib::close type GMaterialLib buf 38,2,39,4
    2017-11-01 20:58:00.801 INFO  [2230869] [GPropertyLib::dumpDomain@161] GPropertyLib::dumpDomain
    2017-11-01 20:58:00.801 INFO  [2230869] [GPropertyLib::dumpDomain@163]  low/high/step  low  60 high 820 step 20 dscale 0.00123984 dscale/low 2.0664e-05 dscale/high 1.512e-06
    2017-11-01 20:58:00.801 INFO  [2230869] [GPropertyLib::dumpDomain@172] GPropertyLib::dumpDomain GC::nanometer 1e-06 GC::h_Planck 4.13567e-12 GC::c_light (mm/ns ~299.792) 299.792 dscale 0.00123984
    2017-11-01 20:58:00.801 INFO  [2230869] [main@115]  after load 
    F2 ri : b0ad5d685c9b6bfb9cbcb3d68e3a3024 : 101 
    d     320.000   2500.000
    v       1.696      1.582
    2017-11-01 20:58:00.801 INFO  [2230869] [GMaterialLib::Summary@220] dump NumMaterials 39 NumFloat4 2
    2017-11-01 20:58:00.801 INFO  [2230869] [GPropertyLib::getIndex@338] GPropertyLib::getIndex type GMaterialLib TRIGGERED A CLOSE  shortname [OpaqueVacuum]
    2017-11-01 20:58:00.802 INFO  [2230869] [GPropertyLib::close@384] GPropertyLib::close type GMaterialLib buf 39,2,39,4
                  domain    refractive_index   absorption_length   scattering_length     reemission_prob      group_velocity
                      60                   1               1e+06               1e+06                   0             299.792
                      80                   1               1e+06               1e+06                   0             299.792
                     100                   1               1e+06               1e+06                   0             299.792
                     120                   1               1e+06               1e+06                   0             299.792
                     140                   1               1e+06               1e+06                   0             299.792
                     160                   1               1e+06               1e+06                   0             299.792
                     180                   1               1e+06               1e+06                   0             299.792
                     200                   1               1e+06               1e+06                   0             299.792




tboolean-box also shows BR discrep
-------------------------------------------

* hmm are the material props being translated correctly ?


::

    tboolean-box --okg4

    simon:opticksgeo blyth$ tboolean-;tboolean-box-p
    args: /Users/blyth/opticks/ana/tboolean.py --det tboolean-box --tag 1
    ok.smry 1 
    [2017-11-01 20:50:38,288] p20501 {/Users/blyth/opticks/ana/tboolean.py:17} INFO - tag 1 src torch det tboolean-box c2max 2.0 ipython False 
    [2017-11-01 20:50:38,288] p20501 {/Users/blyth/opticks/ana/ab.py:80} INFO - AB.load START smry 1 
    [2017-11-01 20:50:38,331] p20501 {/Users/blyth/opticks/ana/evt.py:392} WARNING -  x : -600.000 600.000 : tot 600000 over 13 0.000  under 22 0.000 : mi   -600.000 mx    600.000  
    [2017-11-01 20:50:38,339] p20501 {/Users/blyth/opticks/ana/evt.py:392} WARNING -  y : -600.000 600.000 : tot 600000 over 6 0.000  under 8 0.000 : mi   -600.000 mx    600.000  
    [2017-11-01 20:50:38,349] p20501 {/Users/blyth/opticks/ana/evt.py:392} WARNING -  z : -600.000 600.000 : tot 600000 over 8 0.000  under 5 0.000 : mi   -600.000 mx    600.000  
    [2017-11-01 20:50:39,004] p20501 {/Users/blyth/opticks/ana/ab.py:96} INFO - AB.load DONE 
    [2017-11-01 20:50:39,008] p20501 {/Users/blyth/opticks/ana/ab.py:125} INFO - AB.init_point START
    [2017-11-01 20:50:39,010] p20501 {/Users/blyth/opticks/ana/ab.py:127} INFO - AB.init_point DONE
    AB(1,torch,tboolean-box)  None 0 
    A tboolean-box/torch/  1 :  20171101-2049 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-box/torch/1/fdom.npy 
    B tboolean-box/torch/ -1 :  20171101-2049 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-box/torch/-1/fdom.npy 
    .                seqhis_ana  1:tboolean-box   -1:tboolean-box        c2        ab        ba 
    .                             600000    600000        16.79/6 =  2.80  (pval:0.010 prob:0.990)  
    0000     570058    570041             0.00  TO SA
    0001      25702     25962             1.31  TO BT BT SA
    0002       1799      1594            12.39  TO BR SA
    0003       1536      1498             0.48  TO BT BR BT SA
    0004        694       698             0.01  TO SC SA
    0005         97        82             1.26  TO BT BR BR BT SA
    0006         56        69             1.35  TO AB
    0007         15         8             0.00  TO BT BT SC SA
    0008         11        11             0.00  TO SC BT BT SA
    0009         10         3             0.00  TO BT BR BR BR BT SA
    0010          6         7             0.00  TO BT AB
    0011          6         5             0.00  TO SC BT BR BT SA
    0012          2         5             0.00  TO BT SC BR BR BR BR BR BR BR
    0013          1         4             0.00  TO SC BR SA
    0014          3         3             0.00  TO BT SC BR BT SA
    0015          1         3             0.00  TO SC BT BR BR BT SA
    0016          0         3             0.00  TO BT SC BT SA
    0017          0         1             0.00  TO BT BR BT SC SA
    0018          0         1             0.00  TO SC BT BR BR BR BR BT SA
    0019          1         0             0.00  TO BT BR SC BR BR BR BT SA
    .                             600000    600000        16.79/6 =  2.80  (pval:0.010 prob:0.990)  
    .                pflags_ana  1:tboolean-box   -1:tboolean-box        c2        ab        ba 



Avoid the touching container : see BR discrep
------------------------------------------------

::

    simon:opticksgeo blyth$ tboolean-;tboolean-torus-p
    args: /Users/blyth/opticks/ana/tboolean.py --det tboolean-torus --tag 1
    ok.smry 1 
    [2017-11-01 20:40:38,373] p20189 {/Users/blyth/opticks/ana/tboolean.py:17} INFO - tag 1 src torch det tboolean-torus c2max 2.0 ipython False 
    [2017-11-01 20:40:38,373] p20189 {/Users/blyth/opticks/ana/ab.py:80} INFO - AB.load START smry 1 
    [2017-11-01 20:40:38,441] p20189 {/Users/blyth/opticks/ana/evt.py:392} WARNING -  x : -150.500 150.500 : tot 600000 over 105 0.000  under 86 0.000 : mi   -150.500 mx    150.500  
    [2017-11-01 20:40:38,449] p20189 {/Users/blyth/opticks/ana/evt.py:392} WARNING -  y : -150.500 150.500 : tot 600000 over 77 0.000  under 93 0.000 : mi   -150.500 mx    150.500  
    [2017-11-01 20:40:39,460] p20189 {/Users/blyth/opticks/ana/ab.py:96} INFO - AB.load DONE 
    [2017-11-01 20:40:39,482] p20189 {/Users/blyth/opticks/ana/ab.py:125} INFO - AB.init_point START
    [2017-11-01 20:40:39,498] p20189 {/Users/blyth/opticks/ana/ab.py:127} INFO - AB.init_point DONE
    AB(1,torch,tboolean-torus)  None 0 
    A tboolean-torus/torch/  1 :  20171101-2039 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-torus/torch/1/fdom.npy 
    B tboolean-torus/torch/ -1 :  20171101-2039 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-torus/torch/-1/fdom.npy 
    .                seqhis_ana  1:tboolean-torus   -1:tboolean-torus        c2        ab        ba 
    .                             600000    600000      1052.10/42 = 25.05  (pval:0.000 prob:1.000)  
    0000     196365    205447           205.28  TO BT BT SA
    0001     100590     96737            75.23  TO BT BR BT SA
    0002      94658     94651             0.00  TO SA
    0003      54961     52006            81.63  TO BR SA
    0004      42289     45580           123.26  TO BT BT BT BT SA
    0005      33255     29115           274.81  TO BT BR BR BR BR BR BR BR BR
    0006      16959     18197            43.60  TO BT BR BR BR BT SA
    0007      15456     14218            51.65  TO BT BR BR BR BR BT SA
    0008      10597     11409            29.96  TO BT BR BR BT SA
    0009      11331     10678            19.37  TO BT BR BR BR BR BR BT SA
    0010       6901      5817            92.39  TO BT BR BR BR BR BR BR BR BT
    0011       6804      6464             8.71  TO BT BR BR BR BR BR BR BT SA
    0012       3139      3022             2.22  TO BT BT BR SA
    0013       1852      1917             1.12  TO BT BT BT BR BT SA
    0014       1402      1516             4.45  TO BT BT BR BT BT SA
    0015        711       652             2.55  TO BT BT BT BR BT BT BT SA
    0016        470       454             0.28  TO BR BT BT SA
    0017        408       361             2.87  TO BT BR BR BT BT BT SA
    0018        292       260             1.86  TO BT BT BT BR BR BT SA
    0019        196       187             0.21  TO BT BT BR BR SA
    .                             600000    600000      1052.10/42 = 25.05  (pval:0.000 prob:1.000)  
    .                pflags_ana  1:tboolean-torus   -1:tboolean-torus        c2        ab        ba 



with overtight (touching container) : crazy MI
------------------------------------------------

::

    simon:opticksgeo blyth$ tboolean-torus-p
    args: /Users/blyth/opticks/ana/tboolean.py --det tboolean-torus --tag 1
    ok.smry 1 
    [2017-11-01 20:30:41,828] p19231 {/Users/blyth/opticks/ana/tboolean.py:17} INFO - tag 1 src torch det tboolean-torus c2max 2.0 ipython False 
    [2017-11-01 20:30:41,828] p19231 {/Users/blyth/opticks/ana/ab.py:80} INFO - AB.load START smry 1 
    [2017-11-01 20:30:41,900] p19231 {/Users/blyth/opticks/ana/evt.py:392} WARNING -  x : -150.000 150.000 : tot 600000 over 80 0.000  under 83 0.000 : mi   -150.000 mx    150.000  
    [2017-11-01 20:30:41,907] p19231 {/Users/blyth/opticks/ana/evt.py:392} WARNING -  y : -150.000 150.000 : tot 600000 over 88 0.000  under 76 0.000 : mi   -150.000 mx    150.000  
    [2017-11-01 20:30:43,012] p19231 {/Users/blyth/opticks/ana/ab.py:96} INFO - AB.load DONE 
    [2017-11-01 20:30:43,104] p19231 {/Users/blyth/opticks/ana/ab.py:125} INFO - AB.init_point START
    [2017-11-01 20:30:43,125] p19231 {/Users/blyth/opticks/ana/ab.py:127} INFO - AB.init_point DONE
    AB(1,torch,tboolean-torus)  None 0 
    A tboolean-torus/torch/  1 :  20171101-2028 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-torus/torch/1/fdom.npy 
    B tboolean-torus/torch/ -1 :  20171101-2028 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-torus/torch/-1/fdom.npy 
    .                seqhis_ana  1:tboolean-torus   -1:tboolean-torus        c2        ab        ba 
    .                             600000    600000     58933.95/53 = 1111.96  (pval:0.000 prob:1.000)  
    0000     151079    207121          8768.02  TO BT BT SA
    0001     101285     98084            51.39  TO BT BR BT SA
    0002      88847     88850             0.00  TO SA
    0003      54915     52564            51.43  TO BR SA
    0004      42258     46593           211.50  TO BT BT BT BT SA
    0005      39350         0         39350.00  TO BT MI
    0006      33754     29379           303.18  TO BT BR BR BR BR BR BR BR BR
    0007      17192     18450            44.40  TO BT BR BR BR BT SA
    0008      15683     14282            65.50  TO BT BR BR BR BR BT SA
    0009      10562     11662            54.45  TO BT BR BR BT SA
    0010      11270     10721            13.71  TO BT BR BR BR BR BR BT SA
    0011       8175         0          8175.00  TO MI
    0012       7183      5915           122.75  TO BT BR BR BR BR BR BR BR BT
    0013       6754      6707             0.16  TO BT BR BR BR BR BR BR BT SA
    0014       3201      3075             2.53  TO BT BT BR SA
    0015       1871      2019             5.63  TO BT BT BT BR BT SA
    0016       1378      1422             0.69  TO BT BT BR BT BT SA
    0017        683       633             1.90  TO BT BT BT BR BT BT BT SA
    0018        486       457             0.89  TO BR BT BT SA
    0019        462         0           462.00  TO BT BT BT SA
    .                             600000    600000     58933.95/53 = 1111.96  (pval:0.000 prob:1.000)  



poor chi2 : but wasting most of the stats
-------------------------------------------

::

    simon:opticksgeo blyth$ tboolean-;tboolean-torus-p
    args: /Users/blyth/opticks/ana/tboolean.py --det tboolean-torus --tag 1
    ok.smry 1 
    [2017-11-01 20:21:41,719] p18277 {/Users/blyth/opticks/ana/tboolean.py:17} INFO - tag 1 src torch det tboolean-torus c2max 2.0 ipython False 
    [2017-11-01 20:21:41,719] p18277 {/Users/blyth/opticks/ana/ab.py:80} INFO - AB.load START smry 1 
    [2017-11-01 20:21:41,758] p18277 {/Users/blyth/opticks/ana/evt.py:392} WARNING -  x : -400.000 400.000 : tot 600000 over 868 0.001  under 785 0.001 : mi   -400.000 mx    400.000  
    [2017-11-01 20:21:41,766] p18277 {/Users/blyth/opticks/ana/evt.py:392} WARNING -  y : -400.000 400.000 : tot 600000 over 802 0.001  under 813 0.001 : mi   -400.000 mx    400.000  
    [2017-11-01 20:21:41,773] p18277 {/Users/blyth/opticks/ana/evt.py:392} WARNING -  z : -400.000 400.000 : tot 600000 over 1998 0.003  under 1944 0.003 : mi   -400.000 mx    400.000  
    [2017-11-01 20:21:42,467] p18277 {/Users/blyth/opticks/ana/ab.py:96} INFO - AB.load DONE 
    [2017-11-01 20:21:42,477] p18277 {/Users/blyth/opticks/ana/ab.py:125} INFO - AB.init_point START
    [2017-11-01 20:21:42,485] p18277 {/Users/blyth/opticks/ana/ab.py:127} INFO - AB.init_point DONE
    AB(1,torch,tboolean-torus)  None 0 
    A tboolean-torus/torch/  1 :  20171101-2000 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-torus/torch/1/fdom.npy 
    B tboolean-torus/torch/ -1 :  20171101-2000 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-torus/torch/-1/fdom.npy 
    .                seqhis_ana  1:tboolean-torus   -1:tboolean-torus        c2        ab        ba 
    .                             600000    600000        65.09/19 =  3.43  (pval:0.000 prob:1.000)  
    0000     562547    562537             0.00  TO SA
    0001      20117     20771            10.46  TO BT BT SA
    0002       5625      5365             6.15  TO BT BR BT SA
    0003       3780      3428            17.19  TO BR SA
    0004       2050      2168             3.30  TO BT BT BT BT SA
    0005       1577      1402            10.28  TO BT BR BR BR BR BR BR BR BR
    0006        768       858             4.98  TO BT BR BR BR BT SA
    0007        748       688             2.51  TO BT BR BR BR BR BT SA
    0008        593       601             0.05  TO BT BR BR BT SA
    0009        516       510             0.04  TO BT BR BR BR BR BR BT SA
    0010        458       472             0.21  TO SC SA
    0011        327       278             3.97  TO BT BR BR BR BR BR BR BR BT
    0012        289       311             0.81  TO BT BR BR BR BR BR BR BT SA
    0013        156       156             0.00  TO BT BT BR SA
    0014         88        87             0.01  TO BT BT BT BR BT SA
    0015         54        73             2.84  TO BT BT BR BT BT SA
    0016         62        58             0.13  TO BR BT BT SA
    0017         41        41             0.00  TO AB
    0018         26        35             1.33  TO BT BT BT BR BT BT BT SA
    0019         26        33             0.83  TO BT BR BR BT BT BT SA
    .                             600000    600000        65.09/19 =  3.43  (pval:0.000 prob:1.000)  



tboolean_torus with CPU side photons
---------------------------------------

Emitted input photons are exactly the same in both simulations, 
so should be able to get very close matching. After turn off things
scattering/absorption ? Perhaps use different flavors of vacuum to do this ? 



Difference in ox flags causes different np dumping::

    simon:ana blyth$ ox.py --det tboolean-torus  --tag 1 
    args: /Users/blyth/opticks/ana/ox.py --det tboolean-torus --tag 1
    [2017-11-01 18:21:31,501] p15395 {/Users/blyth/opticks/ana/ox.py:32} INFO - loaded ox /tmp/blyth/opticks/evt/tboolean-torus/torch/1/ox.npy 20171101-1515 shape (600000, 4, 4) 
    [[[-386.263  -310.873   400.        2.8685]
      [  -0.       -0.        1.        1.    ]
      [   0.       -1.        0.      380.    ]
      [   0.        0.        0.        0.    ]]

     [[ -14.892  -262.1473  400.        2.8685]
      [  -0.       -0.        1.        1.    ]
      [   0.       -1.        0.      380.    ]
      [   0.        0.        0.        0.    ]]

     [[ 333.2202 -201.3483  400.        2.8685]
      [  -0.       -0.        1.        1.    ]
      [   0.       -1.        0.      380.    ]
      [   0.        0.        0.        0.    ]]

     ..., 
     [[-174.9729 -400.      253.6111    2.8685]
      [  -0.       -1.       -0.        1.    ]
      [   0.        0.       -1.      380.    ]
      [   0.        0.        0.        0.    ]]

     [[ 259.2407 -400.     -149.578     2.8685]
      [  -0.       -1.       -0.        1.    ]
      [   0.        0.       -1.      380.    ]
      [   0.        0.        0.        0.    ]]

     [[ -64.378  -400.     -129.1872    2.8685]
      [  -0.       -1.       -0.        1.    ]
      [   0.        0.       -1.      380.    ]
      [   0.        0.        0.        0.    ]]]


::

    simon:ana blyth$ ox.py --det tboolean-torus  --tag -1 
    args: /Users/blyth/opticks/ana/ox.py --det tboolean-torus --tag -1
    [2017-11-01 18:21:48,799] p15402 {/Users/blyth/opticks/ana/ox.py:32} INFO - loaded ox /tmp/blyth/opticks/evt/tboolean-torus/torch/-1/ox.npy 20171101-1515 shape (600000, 4, 4) 
    [[[ -3.8626e+02  -3.1087e+02   4.0000e+02   2.8685e+00]
      [ -0.0000e+00  -0.0000e+00   1.0000e+00   1.0000e+00]
      [  0.0000e+00  -1.0000e+00   0.0000e+00   3.8000e+08]
      [  2.8026e-45   0.0000e+00   1.5400e-36   5.9191e-42]]

     [[ -1.4892e+01  -2.6215e+02   4.0000e+02   2.8685e+00]
      [ -0.0000e+00  -0.0000e+00   1.0000e+00   1.0000e+00]
      [  0.0000e+00  -1.0000e+00   0.0000e+00   3.8000e+08]
      [  2.8026e-45   0.0000e+00   1.5400e-36   5.9191e-42]]

     [[  3.3322e+02  -2.0135e+02   4.0000e+02   2.8685e+00]
      [ -0.0000e+00  -0.0000e+00   1.0000e+00   1.0000e+00]
      [  0.0000e+00  -1.0000e+00   0.0000e+00   3.8000e+08]
      [  2.8026e-45   0.0000e+00   1.5400e-36   5.9191e-42]]

     ..., 
     [[ -1.7497e+02  -4.0000e+02   2.5361e+02   2.8685e+00]
      [ -0.0000e+00  -1.0000e+00  -0.0000e+00   1.0000e+00]
      [  0.0000e+00   0.0000e+00  -1.0000e+00   3.8000e+08]
      [  2.8026e-45   0.0000e+00   1.5400e-36   5.9191e-42]]

     [[  2.5924e+02  -4.0000e+02  -1.4958e+02   2.8685e+00]
      [ -0.0000e+00  -1.0000e+00  -0.0000e+00   1.0000e+00]
      [  0.0000e+00   0.0000e+00  -1.0000e+00   3.8000e+08]
      [  2.8026e-45   0.0000e+00   1.5400e-36   5.9191e-42]]

     [[ -6.4378e+01  -4.0000e+02  -1.2919e+02   2.8685e+00]
      [ -0.0000e+00  -1.0000e+00  -0.0000e+00   1.0000e+00]
      [  0.0000e+00   0.0000e+00  -1.0000e+00   3.8000e+08]
      [  2.8026e-45   0.0000e+00   1.5400e-36   5.9191e-42]]]
    simon:ana blyth$ 


