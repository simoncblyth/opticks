WIP tds3gun_nonaligned_comparison
================================================


Possible Fix : add Implicit BorderSurface for RINDEX_NoRINDEX borders
--------------------------------------------------------------------------

This is adding explicit Opticks/GGeo border surfaces to emulate implicit 
Geant4 SURFACE_ABSORB behaviour for photons going from material with RINDEX to material without.

* :doc:`GSurfaceLib__addImplicitBorderSurface_RINDEX_NoRINDEX`


Issue : G4/OK divergence for photons hitting the Tyvek, lack of "SI BT BT SA" with OK
----------------------------------------------------------------------------------------

After suppressing G4 microStep in CRecorder can compare histories.

* first thing with history comparison is to look for zeros 
* also use various ordering criteria
* looks to be a lack of "BT BT SA" with OK 

::

    In [5]: ab.a.seqhis_ana.table.compare(ab.b.seqhis_ana.table)                                                                                                                                                                            
    [{compare             :seq.py    :631} INFO     - cfordering_key for noshortname?
    Out[5]: 
    noshortname?
    .                  cfo:self  1:g4live:tds3gun   -1:g4live:tds3gun        c2        ab        ba 
    .                              11278     11278      1766.28/69 = 25.60  (pval:0.000 prob:1.000)  
    0000               42      1653      1665    -12             0.04        0.993 +- 0.024        1.007 +- 0.025  [2 ] SI AB
    0001            7ccc2      1292      1230     62             1.52        1.050 +- 0.029        0.952 +- 0.027  [5 ] SI BT BT BT SD
    0002            8ccc2       590       674    -84             5.58        0.875 +- 0.036        1.142 +- 0.044  [5 ] SI BT BT BT SA
    ^^^^^^^^^^^^^^^
    0003           7ccc62       581       552     29             0.74        1.053 +- 0.044        0.950 +- 0.040  [6 ] SI SC BT BT BT SD
    0004              452       422       534   -112            13.12        0.790 +- 0.038        1.265 +- 0.055  [3 ] SI RE AB
    0005              462       392       367     25             0.82        1.068 +- 0.054        0.936 +- 0.049  [3 ] SI SC AB
    0006           7ccc52       380       397    -17             0.37        0.957 +- 0.049        1.045 +- 0.052  [6 ] SI RE BT BT BT SD
    0007             4cc2       278       127    151            56.30        2.189 +- 0.131        0.457 +- 0.041  [4 ] SI BT BT AB
    0008           8ccc62       251       267    -16             0.49        0.940 +- 0.059        1.064 +- 0.065  [6 ] SI SC BT BT BT SA
    0009          7ccc662       219       213      6             0.08        1.028 +- 0.069        0.973 +- 0.067  [7 ] SI SC SC BT BT BT SD
    0010            4cc62       197        71    126            59.24        2.775 +- 0.198        0.360 +- 0.043  [5 ] SI SC BT BT AB
    0011          7ccc652       157       159     -2             0.01        0.987 +- 0.079        1.013 +- 0.080  [7 ] SI RE SC BT BT BT SD
    0012           8ccc52       154       188    -34             3.38        0.819 +- 0.066        1.221 +- 0.089  [6 ] SI RE BT BT BT SA
    0013               41       142       144     -2             0.01        0.986 +- 0.083        1.014 +- 0.085  [2 ] CK AB
    0014             4662       137       121     16             0.99        1.132 +- 0.097        0.883 +- 0.080  [4 ] SI SC SC AB
    0015             4552       124       142    -18             1.22        0.873 +- 0.078        1.145 +- 0.096  [4 ] SI RE RE AB
    0016             4652       121       112      9             0.35        1.080 +- 0.098        0.926 +- 0.087  [4 ] SI RE SC AB
    0017            4cc52       117        32     85            48.49        3.656 +- 0.338        0.274 +- 0.048  [5 ] SI RE BT BT AB
    0018          7ccc552       102       124    -22             2.14        0.823 +- 0.081        1.216 +- 0.109  [7 ] SI RE RE BT BT BT SD
    0019           4cc662        82        23     59            33.15        3.565 +- 0.394        0.280 +- 0.058  [6 ] SI SC SC BT BT AB
    0020           4cccc2        77         1     76            74.05       77.000 +- 8.775        0.013 +- 0.013  [6 ] SI BT BT BT BT AB
    0021          4ccccc2        75         0     75            75.00        0.000 +- 0.000        0.000 +- 0.000  [7 ] SI BT BT BT BT BT AB
    0022         7ccc6662        75        72      3             0.06        1.042 +- 0.120        0.960 +- 0.113  [8 ] SI SC SC SC BT BT BT SD
    0023          49cccc2        70         0     70            70.00        0.000 +- 0.000        0.000 +- 0.000  [7 ] SI BT BT BT BT DR AB
    ^^^^^^^^^^^^^^^^^   DR      
    0024          8ccc662        69       107    -38             8.20        0.645 +- 0.078        1.551 +- 0.150  [7 ] SI SC SC BT BT BT SA
    0025           8cccc2        64        81    -17             1.99        0.790 +- 0.099        1.266 +- 0.141  [6 ] SI BT BT BT BT SA
    0026          7ccc562        59        31     28             8.71        1.903 +- 0.248        0.525 +- 0.094  [7 ] SI SC RE BT BT BT SD
    0027          8ccc652        59        95    -36             8.42        0.621 +- 0.081        1.610 +- 0.165  [7 ] SI RE SC BT BT BT SA
    0028          8ccc552        58        62     -4             0.13        0.935 +- 0.123        1.069 +- 0.136  [7 ] SI RE RE BT BT BT SA
    0029         7ccc6652        56        54      2             0.04        1.037 +- 0.139        0.964 +- 0.131  [8 ] SI RE SC SC BT BT BT SD
    0030              4c2        56        49      7             0.47        1.143 +- 0.153        0.875 +- 0.125  [3 ] SI BT AB
    0031           7cccc2        53       154   -101            49.28        0.344 +- 0.047        2.906 +- 0.234  [6 ] SI BT BT BT BT SD
    0032             4562        52        40     12             1.57        1.300 +- 0.180        0.769 +- 0.122  [4 ] SI SC RE AB



    In [7]: ab.a.seqhis_ana.table.compare(ab.b.seqhis_ana.table, ordering="sum")                                                                                                                                                            
    [{compare             :seq.py    :631} INFO     - cfordering_key for noshortname?
    Out[7]: 
    noshortname?
    .                   cfo:sum  1:g4live:tds3gun   -1:g4live:tds3gun        c2        ab        ba 
    .                              11278     11278      1766.28/69 = 25.60  (pval:0.000 prob:1.000)  
    0000               42      1653      1665    -12             0.04        0.993 +- 0.024        1.007 +- 0.025  [2 ] SI AB
    0001            7ccc2      1292      1230     62             1.52        1.050 +- 0.029        0.952 +- 0.027  [5 ] SI BT BT BT SD
    0002            8ccc2       590       674    -84             5.58        0.875 +- 0.036        1.142 +- 0.044  [5 ] SI BT BT BT SA
    0003           7ccc62       581       552     29             0.74        1.053 +- 0.044        0.950 +- 0.040  [6 ] SI SC BT BT BT SD
    0004              452       422       534   -112            13.12        0.790 +- 0.038        1.265 +- 0.055  [3 ] SI RE AB
    0005           7ccc52       380       397    -17             0.37        0.957 +- 0.049        1.045 +- 0.052  [6 ] SI RE BT BT BT SD
    0006              462       392       367     25             0.82        1.068 +- 0.054        0.936 +- 0.049  [3 ] SI SC AB
    0007           8ccc62       251       267    -16             0.49        0.940 +- 0.059        1.064 +- 0.065  [6 ] SI SC BT BT BT SA
    0008             8cc2         0       464   -464           464.00        0.000 +- 0.000        0.000 +- 0.000  [4 ] SI BT BT SA
    ^^^^^^^^
    0009          7ccc662       219       213      6             0.08        1.028 +- 0.069        0.973 +- 0.067  [7 ] SI SC SC BT BT BT SD
    0010             4cc2       278       127    151            56.30        2.189 +- 0.131        0.457 +- 0.041  [4 ] SI BT BT AB
    0011           8ccc52       154       188    -34             3.38        0.819 +- 0.066        1.221 +- 0.089  [6 ] SI RE BT BT BT SA
    0012          7ccc652       157       159     -2             0.01        0.987 +- 0.079        1.013 +- 0.080  [7 ] SI RE SC BT BT BT SD
    0013               41       142       144     -2             0.01        0.986 +- 0.083        1.014 +- 0.085  [2 ] CK AB
    0014            4cc62       197        71    126            59.24        2.775 +- 0.198        0.360 +- 0.043  [5 ] SI SC BT BT AB
    0015             4552       124       142    -18             1.22        0.873 +- 0.078        1.145 +- 0.096  [4 ] SI RE RE AB
    0016             4662       137       121     16             0.99        1.132 +- 0.097        0.883 +- 0.080  [4 ] SI SC SC AB
    0017             4652       121       112      9             0.35        1.080 +- 0.098        0.926 +- 0.087  [4 ] SI RE SC AB
    0018          7ccc552       102       124    -22             2.14        0.823 +- 0.081        1.216 +- 0.109  [7 ] SI RE RE BT BT BT SD
    0019           7cccc2        53       154   -101            49.28        0.344 +- 0.047        2.906 +- 0.234  [6 ] SI BT BT BT BT SD
    0020            8cc62         0       186   -186           186.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] SI SC BT BT SA
    ^^^^^^^^^^^  
    0021          8ccc662        69       107    -38             8.20        0.645 +- 0.078        1.551 +- 0.150  [7 ] SI SC SC BT BT BT SA


    In [10]: ab.a.seqhis_ana.table.compare(ab.b.seqhis_ana.table, ordering="other")[:20]                                                                                                                                                    
    [{compare             :seq.py    :631} INFO     - cfordering_key for noshortname?
    Out[10]: 
    noshortname?
    .                 cfo:other  1:g4live:tds3gun   -1:g4live:tds3gun        c2        ab        ba 
    .                              11278     11278      1766.28/69 = 25.60  (pval:0.000 prob:1.000)  
    0000               42      1653      1665    -12             0.04        0.993 +- 0.024        1.007 +- 0.025  [2 ] SI AB
    0001            7ccc2      1292      1230     62             1.52        1.050 +- 0.029        0.952 +- 0.027  [5 ] SI BT BT BT SD
    0002            8ccc2       590       674    -84             5.58        0.875 +- 0.036        1.142 +- 0.044  [5 ] SI BT BT BT SA
    0003           7ccc62       581       552     29             0.74        1.053 +- 0.044        0.950 +- 0.040  [6 ] SI SC BT BT BT SD
    0004              452       422       534   -112            13.12        0.790 +- 0.038        1.265 +- 0.055  [3 ] SI RE AB
    0005             8cc2         0       464   -464           464.00        0.000 +- 0.000        0.000 +- 0.000  [4 ] SI BT BT SA
    ^^^^^^^^^^^
    0006           7ccc52       380       397    -17             0.37        0.957 +- 0.049        1.045 +- 0.052  [6 ] SI RE BT BT BT SD
    0007              462       392       367     25             0.82        1.068 +- 0.054        0.936 +- 0.049  [3 ] SI SC AB
    0008           8ccc62       251       267    -16             0.49        0.940 +- 0.059        1.064 +- 0.065  [6 ] SI SC BT BT BT SA
    0009          7ccc662       219       213      6             0.08        1.028 +- 0.069        0.973 +- 0.067  [7 ] SI SC SC BT BT BT SD
    0010           8ccc52       154       188    -34             3.38        0.819 +- 0.066        1.221 +- 0.089  [6 ] SI RE BT BT BT SA
    0011            8cc62         0       186   -186           186.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] SI SC BT BT SA
    ^^^^^^^^^^^
    0012          7ccc652       157       159     -2             0.01        0.987 +- 0.079        1.013 +- 0.080  [7 ] SI RE SC BT BT BT SD
    0013           7cccc2        53       154   -101            49.28        0.344 +- 0.047        2.906 +- 0.234  [6 ] SI BT BT BT BT SD
    0014               41       142       144     -2             0.01        0.986 +- 0.083        1.014 +- 0.085  [2 ] CK AB
    0015             4552       124       142    -18             1.22        0.873 +- 0.078        1.145 +- 0.096  [4 ] SI RE RE AB
    0016            8cc52         0       138   -138           138.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] SI RE BT BT SA
    ^^^^^^^^^^^
    0017             4cc2       278       127    151            56.30        2.189 +- 0.131        0.457 +- 0.041  [4 ] SI BT BT AB
    0018          7ccc552       102       124    -22             2.14        0.823 +- 0.081        1.216 +- 0.109  [7 ] SI RE RE BT BT BT SD
    0019             4662       137       121     16             0.99        1.132 +- 0.097        0.883 +- 0.080  [4 ] SI SC SC AB
    .                              11278     11278      1766.28/69 = 25.60  (pval:0.000 prob:1.000)  







change cfordering to sum as it gives a better overview
---------------------------------------------------------

::

    epsilon:ana blyth$ tds3gun.sh 1


    ab
    AB(1,natural,g4live)  None 0     file_photons 11278   load_slice 0:100k:   loaded_photons 11278  
    A tds3gun/g4live/natural/  1 :  20210613-1141 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/tds3gun/evt/g4live/natural/1/fdom.npy () 
    B tds3gun/g4live/natural/ -1 :  20210613-1141 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/tds3gun/evt/g4live/natural/-1/fdom.npy (recstp) 
    .
    '#ab.__str__.ahis'
    ab.ahis
    .    all_seqhis_ana  cfo:sum  1:g4live:tds3gun   -1:g4live:tds3gun        c2        ab        ba 
    .                              11278     11278      1766.28/69 = 25.60  (pval:0.000 prob:1.000)  
    0000               42      1653      1665    -12             0.04        0.993 +- 0.024        1.007 +- 0.025  [2 ] SI AB
    0001            7ccc2      1292      1230     62             1.52        1.050 +- 0.029        0.952 +- 0.027  [5 ] SI BT BT BT SD
    0002            8ccc2       590       674    -84             5.58        0.875 +- 0.036        1.142 +- 0.044  [5 ] SI BT BT BT SA
    0003           7ccc62       581       552     29             0.74        1.053 +- 0.044        0.950 +- 0.040  [6 ] SI SC BT BT BT SD
    0004              452       422       534   -112            13.12        0.790 +- 0.038        1.265 +- 0.055  [3 ] SI RE AB
    0005           7ccc52       380       397    -17             0.37        0.957 +- 0.049        1.045 +- 0.052  [6 ] SI RE BT BT BT SD
    0006              462       392       367     25             0.82        1.068 +- 0.054        0.936 +- 0.049  [3 ] SI SC AB
    0007           8ccc62       251       267    -16             0.49        0.940 +- 0.059        1.064 +- 0.065  [6 ] SI SC BT BT BT SA
    0008             8cc2         0       464   -464           464.00        0.000 +- 0.000        0.000 +- 0.000  [4 ] SI BT BT SA
    ^^^^^^^^^^^^^^^^^^^

    0009          7ccc662       219       213      6             0.08        1.028 +- 0.069        0.973 +- 0.067  [7 ] SI SC SC BT BT BT SD
    0010             4cc2       278       127    151            56.30        2.189 +- 0.131        0.457 +- 0.041  [4 ] SI BT BT AB
    0011           8ccc52       154       188    -34             3.38        0.819 +- 0.066        1.221 +- 0.089  [6 ] SI RE BT BT BT SA
    0012          7ccc652       157       159     -2             0.01        0.987 +- 0.079        1.013 +- 0.080  [7 ] SI RE SC BT BT BT SD
    0013               41       142       144     -2             0.01        0.986 +- 0.083        1.014 +- 0.085  [2 ] CK AB
    0014            4cc62       197        71    126            59.24        2.775 +- 0.198        0.360 +- 0.043  [5 ] SI SC BT BT AB
    0015             4552       124       142    -18             1.22        0.873 +- 0.078        1.145 +- 0.096  [4 ] SI RE RE AB
    0016             4662       137       121     16             0.99        1.132 +- 0.097        0.883 +- 0.080  [4 ] SI SC SC AB
    0017             4652       121       112      9             0.35        1.080 +- 0.098        0.926 +- 0.087  [4 ] SI RE SC AB
    0018          7ccc552       102       124    -22             2.14        0.823 +- 0.081        1.216 +- 0.109  [7 ] SI RE RE BT BT BT SD
    0019           7cccc2        53       154   -101            49.28        0.344 +- 0.047        2.906 +- 0.234  [6 ] SI BT BT BT BT SD
    .                              11278     11278      1766.28/69 = 25.60  (pval:0.000 prob:1.000)  
    '#ab.__str__.flg'
    ab.flg
    .       pflags_ana  cfo:sum  1:g4live:tds3gun   -1:g4live:tds3gun        c2        ab        ba 
    .                              11278     11278      8579.65/43 = 199.53  (pval:0.000 prob:1.000)  
    0000                a      1653      1665    -12             0.04        0.993 +- 0.024        1.007 +- 0.025  [2 ] AB|SI
    0001              882       654      1224   -570           173.00        0.534 +- 0.021        1.872 +- 0.053  [3 ] BT|SA|SI
    0002              842         0      1389   -1389          1389.00        0.000 +- 0.000        0.000 +- 0.000  [3 ] BT|SD|SI
    ^^^^^^^^^^^^^^^^ pflags looking  inconsistent with seqhis ?

    0003               1a       586       742   -156            18.33        0.790 +- 0.033        1.266 +- 0.046  [3 ] RE|AB|SI
    0004              8a2       387       766   -379           124.58        0.505 +- 0.026        1.979 +- 0.072  [4 ] BT|SA|SC|SI
    0005               2a       601       540     61             3.26        1.113 +- 0.045        0.899 +- 0.039  [3 ] SC|AB|SI
    0006              862         0      1020   -1020          1020.00        0.000 +- 0.000        0.000 +- 0.000  [4 ] BT|SD|SC|SI
    0007               3a       431       421     10             0.12        1.024 +- 0.049        0.977 +- 0.048  [4 ] SC|RE|AB|SI
    0008             4842       797         0    797           797.00        0.000 +- 0.000        0.000 +- 0.000  [4 ] EX|BT|SD|SI
    0009              892       244       522   -278           100.89        0.467 +- 0.030        2.139 +- 0.094  [4 ] BT|SA|RE|SI
    0010              80a       542       196    346           162.22        2.765 +- 0.119        0.362 +- 0.026  [3 ] BT|AB|SI
    0011              82a       516       170    346           174.51        3.035 +- 0.134        0.329 +- 0.025  [4 ] BT|SC|AB|SI
    0012              8b2       198       478   -280           115.98        0.414 +- 0.029        2.414 +- 0.110  [5 ] BT|SA|SC|RE|SI
    0013              852         0       662   -662           662.00        0.000 +- 0.000        0.000 +- 0.000  [4 ] BT|SD|RE|SI
    0014             4862       591         0    591           591.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] EX|BT|SD|SC|SI
    0015              872         0       569   -569           569.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] BT|SD|SC|RE|SI
    0016             8842       548         0    548           548.00        0.000 +- 0.000        0.000 +- 0.000  [4 ] EC|BT|SD|SI
    0017              83a       325       102    223           116.46        3.186 +- 0.177        0.314 +- 0.031  [5 ] BT|SC|RE|AB|SI
    0018             8862       352         0    352           352.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] EC|BT|SD|SC|SI
    0019              81a       258        93    165            77.56        2.774 +- 0.173        0.360 +- 0.037  [4 ] BT|RE|AB|SI
    .                              11278     11278      8579.65/43 = 199.53  (pval:0.000 prob:1.000)  
    ab.mat
    .       seqmat_ana  cfo:sum  1:g4live:tds3gun   -1:g4live:tds3gun        c2        ab        ba 
    .                              11278     11278      3177.83/62 = 51.26  (pval:0.000 prob:1.000)  
    0000               11      1795      1809    -14             0.05        0.992 +- 0.023        1.008 +- 0.024  [2 ] LS LS
    0001            defb1      1770      1765      5             0.01        1.003 +- 0.024        0.997 +- 0.024  [5 ] LS Ac Wa Py Va
    0002           defb11      1266      1340    -74             2.10        0.945 +- 0.027        1.058 +- 0.029  [6 ] LS LS Ac Wa Py Va
    0003              111       831       914    -83             3.95        0.909 +- 0.032        1.100 +- 0.036  [3 ] LS LS LS
    0004          defb111       682       762    -80             4.43        0.895 +- 0.034        1.117 +- 0.040  [7 ] LS LS LS Ac Wa Py Va
    0005             1111       442       422     20             0.46        1.047 +- 0.050        0.955 +- 0.046  [4 ] LS LS LS LS
    0006         defb1111       327       374    -47             3.15        0.874 +- 0.048        1.144 +- 0.059  [8 ] LS LS LS LS Ac Wa Py Va
    0007             3fb1         0       451   -451           451.00        0.000 +- 0.000        0.000 +- 0.000  [4 ] LS Ac Wa Ty

    0008            11111       206       214     -8             0.15        0.963 +- 0.067        1.039 +- 0.071  [5 ] LS LS LS LS LS
    0009        defb11111       158       189    -31             2.77        0.836 +- 0.067        1.196 +- 0.087  [9 ] LS LS LS LS LS Ac Wa Py Va
    0010            3fb11         0       313   -313           313.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] LS LS Ac Wa Ty

    0011             ffb1       118       128    -10             0.41        0.922 +- 0.085        1.085 +- 0.096  [4 ] LS Ac Wa Wa
    0012            eeb11       238         0    238           238.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] LS LS Ac Py Py
    0013           111111       106        92     14             0.99        1.152 +- 0.112        0.868 +- 0.090  [6 ] LS LS LS LS LS LS
    0014            ffb11        79       102    -23             2.92        0.775 +- 0.087        1.291 +- 0.128  [5 ] LS LS Ac Wa Wa
    0015           3fb111         0       179   -179           179.00        0.000 +- 0.000        0.000 +- 0.000  [6 ] LS LS LS Ac Wa Ty

    0016           deffb1        91        87      4             0.09        1.046 +- 0.110        0.956 +- 0.102  [6 ] LS Ac Wa Wa Py Va
    0017       defb111111        75        89    -14             1.20        0.843 +- 0.097        1.187 +- 0.126  [10] LS LS LS LS LS LS Ac Wa Py Va
    0018             eeb1       162         0    162           162.00        0.000 +- 0.000        0.000 +- 0.000  [4 ] LS Ac Py Py
    0019           eeb111       137         0    137           137.00        0.000 +- 0.000        0.000 +- 0.000  [6 ] LS LS LS Ac Py Py
    .                              11278     11278      3177.83/62 = 51.26  (pval:0.000 prob:1.000)  
    #ab.cfm



Lack of OK "BT BT SA" issue looks to be of the same size as photons reaching the Tyvek.

::

    In [5]: b.selmat = "*Ty*"                                                                                                                                                                                                               
    In [6]: b.mat[:30]                                                                                                                                                                                                                      
    Out[6]: 
    seqmat_ana
    .                     cfo:-  -1:g4live:tds3gun 
    .                               1363         1.00 
    0000             3fb1        0.331         451        [4 ] LS Ac Wa Ty
    0001            3fb11        0.230         313        [5 ] LS LS Ac Wa Ty
    0002           3fb111        0.131         179        [6 ] LS LS LS Ac Wa Ty
    0003            3fbb1        0.081         110        [5 ] LS Ac Ac Wa Ty
    0004          3fb1111        0.062          85        [7 ] LS LS LS LS Ac Wa Ty
    0005           3fbb11        0.033          45        [6 ] LS LS Ac Ac Wa Ty
    0006         3fb11111        0.033          45        [8 ] LS LS LS LS LS Ac Wa Ty
    0007          3fbb111        0.015          20        [7 ] LS LS LS Ac Ac Wa Ty
    0008           3ffb11        0.011          15        [6 ] LS LS Ac Wa Wa Ty
    0009            3ffb1        0.008          11        [5 ] LS Ac Wa Wa Ty
    0010        3fb111111        0.007          10        [9 ] LS LS LS LS LS LS Ac Wa Ty
    0011         3fbb1111        0.007          10        [8 ] LS LS LS LS Ac Ac Wa Ty
    0012       3feeeefb11        0.007           9        [10] LS LS Ac Wa Py Py Py Py Wa Ty
    0013        3fbb11111        0.006           8        [9 ] LS LS LS LS LS Ac Ac Wa Ty
    0014       3feeefb111        0.006           8        [10] LS LS LS Ac Wa Py Py Py Wa Ty
    0015          3ffb111        0.006           8        [7 ] LS LS LS Ac Wa Wa Ty
    0016        3feeefb11        0.006           8        [9 ] LS LS Ac Wa Py Py Py Wa Ty
    0017       3fb1111111        0.004           5        [10] LS LS LS LS LS LS LS Ac Wa Ty
    0018       3feeeffb11        0.003           4        [10] LS LS Ac Wa Wa Py Py Py Wa Ty
    0019        3feeeefb1        0.003           4        [9 ] LS Ac Wa Py Py Py Py Wa Ty
    0020         3ffb1111        0.002           3        [8 ] LS LS LS LS Ac Wa Wa Ty
    0021          3ffbb11        0.001           2        [7 ] LS LS Ac Ac Wa Wa Ty
    0022         3ffbb111        0.001           2        [8 ] LS LS LS Ac Ac Wa Wa Ty
    0023       3fbb11bb11        0.001           1        [10] LS LS Ac Ac LS LS Ac Ac Wa Ty
    0024       3fbb111111        0.001           1        [10] LS LS LS LS LS LS Ac Ac Wa Ty
    0025        3ffb11111        0.001           1        [9 ] LS LS LS LS LS Ac Wa Wa Ty
    0026        3ffffb111        0.001           1        [9 ] LS LS LS Ac Wa Wa Wa Wa Ty
    0027           3ffbb1        0.001           1        [6 ] LS Ac Ac Wa Wa Ty
    0028           3fffb1        0.001           1        [6 ] LS Ac Wa Wa Wa Ty
    0029          3fffb11        0.001           1        [7 ] LS LS Ac Wa Wa Wa Ty
    .                               1363         1.00 



With G4 all the Tyvek reachers get SA, that was an artifical kludge from NoRINDEX yielding NAN_ABORT::
       
    In [6]: b.mat[:30]                                                                                                                                                                                                                      
    Out[6]: 
    seqmat_ana
    .                     cfo:-  -1:g4live:tds3gun 
    .                               1363         1.00 
    0000             3fb1        0.331         451        [4 ] LS Ac Wa Ty
    0001            3fb11        0.230         313        [5 ] LS LS Ac Wa Ty
    0002           3fb111        0.131         179        [6 ] LS LS LS Ac Wa Ty
    0003            3fbb1        0.081         110        [5 ] LS Ac Ac Wa Ty
    0004          3fb1111        0.062          85        [7 ] LS LS LS LS Ac Wa Ty
    0005           3fbb11        0.033          45        [6 ] LS LS Ac Ac Wa Ty
    0006         3fb11111        0.033          45        [8 ] LS LS LS LS LS Ac Wa Ty
    0007          3fbb111        0.015          20        [7 ] LS LS LS Ac Ac Wa Ty
    0008           3ffb11        0.011          15        [6 ] LS LS Ac Wa Wa Ty
    0009            3ffb1        0.008          11        [5 ] LS Ac Wa Wa Ty
    0010        3fb111111        0.007          10        [9 ] LS LS LS LS LS LS Ac Wa Ty
    0011         3fbb1111        0.007          10        [8 ] LS LS LS LS Ac Ac Wa Ty
    0012       3feeeefb11        0.007           9        [10] LS LS Ac Wa Py Py Py Py Wa Ty
    0013        3fbb11111        0.006           8        [9 ] LS LS LS LS LS Ac Ac Wa Ty
    0014       3feeefb111        0.006           8        [10] LS LS LS Ac Wa Py Py Py Wa Ty
    0015          3ffb111        0.006           8        [7 ] LS LS LS Ac Wa Wa Ty
    0016        3feeefb11        0.006           8        [9 ] LS LS Ac Wa Py Py Py Wa Ty
    0017       3fb1111111        0.004           5        [10] LS LS LS LS LS LS LS Ac Wa Ty
    0018       3feeeffb11        0.003           4        [10] LS LS Ac Wa Wa Py Py Py Wa Ty
    0019        3feeeefb1        0.003           4        [9 ] LS Ac Wa Py Py Py Py Wa Ty
    0020         3ffb1111        0.002           3        [8 ] LS LS LS LS Ac Wa Wa Ty
    0021          3ffbb11        0.001           2        [7 ] LS LS Ac Ac Wa Wa Ty
    0022         3ffbb111        0.001           2        [8 ] LS LS LS Ac Ac Wa Wa Ty
    0023       3fbb11bb11        0.001           1        [10] LS LS Ac Ac LS LS Ac Ac Wa Ty
    0024       3fbb111111        0.001           1        [10] LS LS LS LS LS LS Ac Ac Wa Ty
    0025        3ffb11111        0.001           1        [9 ] LS LS LS LS LS Ac Wa Wa Ty
    0026        3ffffb111        0.001           1        [9 ] LS LS LS Ac Wa Wa Wa Wa Ty
    0027           3ffbb1        0.001           1        [6 ] LS Ac Ac Wa Wa Ty
    0028           3fffb1        0.001           1        [6 ] LS Ac Wa Wa Wa Ty
    0029          3fffb11        0.001           1        [7 ] LS LS Ac Wa Wa Wa Ty
    .                               1363         1.00 



    In [9]: b.his                                                                                                                                                                                                                           
    Out[9]: 
    seqhis_ana
    .                     cfo:-  -1:g4live:tds3gun 
    .                               1363         1.00 
    0000             8cc2        0.329         448        [4 ] SI BT BT SA
    0001            8cc62        0.129         176        [5 ] SI SC BT BT SA
    0002            8cc52        0.098         133        [5 ] SI RE BT BT SA
    0003            8ccc2        0.081         110        [5 ] SI BT BT BT SA
    0004           8cc662        0.045          61        [6 ] SI SC SC BT BT SA
    0005           8cc652        0.038          52        [6 ] SI RE SC BT BT SA
    0006           8cc552        0.031          42        [6 ] SI RE RE BT BT SA
    0007           8ccc52        0.020          27        [6 ] SI RE BT BT BT SA
    0008          8cc6552        0.019          26        [7 ] SI RE RE SC BT BT SA
    0009           8ccc62        0.013          18        [6 ] SI SC BT BT BT SA
    0010          8cc6652        0.012          16        [7 ] SI RE SC SC BT BT SA
    0011           8cc562        0.011          15        [6 ] SI SC RE BT BT SA
    0012          8cc6662        0.011          15        [7 ] SI SC SC SC BT BT SA
    0013          8cc5552        0.010          14        [7 ] SI RE RE RE BT BT SA
    0014            8bcc2        0.007          10        [5 ] SI BT BT BR SA
    0015           8bcc62        0.007           9        [6 ] SI SC BT BT BR SA
    0016         8cc66652        0.007           9        [8 ] SI RE SC SC SC BT BT SA
    0017          8ccc552        0.006           8        [7 ] SI RE RE BT BT BT SA
    0018       8ccacccc62        0.005           7        [10] SI SC BT BT BT BT SR BT BT SA
    0019        8ccaccc62        0.005           7        [9 ] SI SC BT BT BT SR BT BT SA
    0020         8cc55552        0.004           6        [8 ] SI RE RE RE RE BT BT SA
    0021           8bcc52        0.004           6        [6 ] SI RE BT BT BR SA
    0022          8ccc652        0.004           6        [7 ] SI RE SC BT BT BT SA




    In [7]: a.selmat = "*Ty*"                                                                                                                                                                                                               

    In [8]: a.mat[:30]                                                                                                                                                                                                                      
    Out[8]: 
    seqmat_ana
    .                     cfo:-  1:g4live:tds3gun 
    .                               1320         1.00 
    0000          aaf3fb1        0.048          64        [7 ] LS Ac Wa Ty Wa Ll Ll
    0001         aaf3fb11        0.042          55        [8 ] LS LS Ac Wa Ty Wa Ll Ll
    0002         aaff3fb1        0.040          53        [8 ] LS Ac Wa Ty Wa Wa Ll Ll
    0003          fff3fb1        0.039          51        [7 ] LS Ac Wa Ty Wa Wa Wa
    0004       ffffff3fb1        0.037          49        [10] LS Ac Wa Ty Wa Wa Wa Wa Wa Wa
    0005           ff3fb1        0.027          36        [6 ] LS Ac Wa Ty Wa Wa
    0006       eeefbf3fb1        0.024          32        [10] LS Ac Wa Ty Wa Ac Wa Py Py Py
    0007       ffff3fb111        0.024          32        [10] LS LS LS Ac Wa Ty Wa Wa Wa Wa
    0008       fffff3fb11        0.023          31        [10] LS LS Ac Wa Ty Wa Wa Wa Wa Wa
    0009        aaf3fb111        0.023          30        [9 ] LS LS LS Ac Wa Ty Wa Ll Ll
    0010           993fb1        0.022          29        [6 ] LS Ac Wa Ty Ro Ro
    0011       fff3fb1111        0.020          27        [10] LS LS LS LS Ac Wa Ty Wa Wa Wa
    0012           3f3fb1        0.020          27        [6 ] LS Ac Wa Ty Wa Ty
    0013       eefbf3fb11        0.019          25        [10] LS LS Ac Wa Ty Wa Ac Wa Py Py
    0014       ff3fb11111        0.017          22        [10] LS LS LS LS LS Ac Wa Ty Wa Wa
    0015        aaff3fb11        0.017          22        [9 ] LS LS Ac Wa Ty Wa Wa Ll Ll
    0016        aafff3fb1        0.017          22        [9 ] LS Ac Wa Ty Wa Wa Wa Ll Ll
    0017       f3fb111111        0.015          20        [10] LS LS LS LS LS LS Ac Wa Ty Wa
    0018          ff3fb11        0.014          19        [7 ] LS LS Ac Wa Ty Wa Wa
    0019       aaf3fb1111        0.014          19        [10] LS LS LS LS Ac Wa Ty Wa Ll Ll
    0020       aafff3fb11        0.014          19        [10] LS LS Ac Wa Ty Wa Wa Wa Ll Ll
    0021          993fb11        0.012          16        [7 ] LS LS Ac Wa Ty Ro Ro
    0022         fff3fb11        0.012          16        [8 ] LS LS Ac Wa Ty Wa Wa Wa
    0023         ffff3fb1        0.011          14        [8 ] LS Ac Wa Ty Wa Wa Wa Wa
    0024        fff3fb111        0.010          13        [9 ] LS LS LS Ac Wa Ty Wa Wa Wa
    0025          99f3fb1        0.010          13        [7 ] LS Ac Wa Ty Wa Ro Ro
    0026       affff3fb11        0.010          13        [10] LS LS Ac Wa Ty Wa Wa Wa Wa Ll
    0027       defbff3fb1        0.010          13        [10] LS Ac Wa Ty Wa Wa Ac Wa Py Va
    0028       aaff3fb111        0.010          13        [10] LS LS LS Ac Wa Ty Wa Wa Ll Ll
    0029       aaffff3fb1        0.009          12        [10] LS Ac Wa Ty Wa Wa Wa Wa Ll Ll
    .                               1320         1.00 

    In [9]:                                                         
  



Hmm, I recall handling NoRINDEX but slapping down a SURFACE_ABSORB::


    272 unsigned int OpStatus::OpBoundaryFlag(const G4OpBoundaryProcessStatus status)
    273 {
    274     unsigned flag = 0 ;
    275     switch(status)
    276     {
    277         case FresnelRefraction:
    278         case SameMaterial:
    279                                flag=BOUNDARY_TRANSMIT;
    280                                break;
    281         case TotalInternalReflection:
    282         case       FresnelReflection:
    283                                flag=BOUNDARY_REFLECT;
    284                                break;
    285         case StepTooSmall:
    286                                flag=NAN_ABORT;
    287                                break;
    288         case Absorption:
    289                                flag=SURFACE_ABSORB ;
    290                                break;
    291         case Detection:
    292                                flag=SURFACE_DETECT ;
    293                                break;
    294         case SpikeReflection:
    295                                flag=SURFACE_SREFLECT ;
    296                                break;
    297         case LobeReflection:
    298         case LambertianReflection:
    299                                flag=SURFACE_DREFLECT ;
    300                                break;
    301         case NoRINDEX:
    302                                //flag=NAN_ABORT;
    303                                flag=SURFACE_ABSORB ;  // expt 
    304                                break;
    305         case Undefined:




TODO:

* study unmodified Geant4 handling of photons reaching the Tyvek 



g4-cls G4OpBoundaryProcess : does fStopAndKill at NoRINDEX : so its a terminal problem 
-----------------------------------------------------------------------------------------

::

     276         }
     277 
     278     G4MaterialPropertiesTable* aMaterialPropertiesTable;
     279         G4MaterialPropertyVector* Rindex;
     280 
     281     aMaterialPropertiesTable = Material1->GetMaterialPropertiesTable();
     282         if (aMaterialPropertiesTable) {
     283         Rindex = aMaterialPropertiesTable->GetProperty(kRINDEX);
     284     }
     285     else {
     286                 theStatus = NoRINDEX;
     287                 if ( verboseLevel > 0) BoundaryProcessVerbose();
     288                 aParticleChange.ProposeLocalEnergyDeposit(thePhotonMomentum);
     289                 aParticleChange.ProposeTrackStatus(fStopAndKill);
     290                 return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
     291     }
     292 
     293         if (Rindex) {
     294            Rindex1 = Rindex->Value(thePhotonMomentum);
     295         }
     296         else {
     297             theStatus = NoRINDEX;
     298                 if ( verboseLevel > 0) BoundaryProcessVerbose();
     299                 aParticleChange.ProposeLocalEnergyDeposit(thePhotonMomentum);
     300                 aParticleChange.ProposeTrackStatus(fStopAndKill);
     301                 return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
     302     }



Want to get the params of photon hitting the Tyvek so can rerun under debugger
and to try and see exactly which NoRINDEX is being tickled.
Select on material sequence and look at history of those::

    In [10]: b.selmat = "LS Ac Wa Ty"                                                                                                                                                                                                       

    In [11]: b.his                                                                                                                                                                                                                          
    Out[11]: 
    seqhis_ana
    .                     cfo:-  -1:g4live:tds3gun 
    .                                451         1.00 
    0000             8cc2        0.993         448        [4 ] SI BT BT SA
    0001             8cc1        0.007           3        [4 ] CK BT BT SA
    .                                451         1.00 

    In [12]: b.ox.shape
    Out[12]: (451, 4, 4)


    In [15]: pos = b.ox[:,0,:3]

    In [16]: np.sqrt(np.sum(pos*pos, axis=1))   ## all at the Tyvek radius
    Out[16]:
    A([20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.002, 20050.   , 20050.   , 20050.   ,
       20050.002, 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   ,
       20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   ,
       20050.   , 20050.002, 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   ,
       20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   ,
       20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   ,
       20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.002, 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   ,
       20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   ,
       20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   ,
       20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.002, 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   ,
       20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   ,
       20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.002, 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   ,
       20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.002, 20050.   , 20050.   , 20050.   , 20050.   ,
       20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.002, 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   ,
       20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   ,
       20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.002, 20050.   , 20050.   ,
       20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   ,
       20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   ,
       20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   ,
       20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   ,
       20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   ,
       20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   ,
       20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   ,
       20050.   , 20050.002, 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   ,
       20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   ,
       20050.   , 20050.   , 20050.   , 20050.002, 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.002, 20050.   , 20050.   , 20050.   , 20050.   ,
       20050.   , 20050.   , 20050.   , 20050.   , 20050.   , 20050.002, 20050.   , 20050.   , 20050.   ], dtype=float32)


    In [20]: np.save("/tmp/b_ox_Tyvek.npy", b.ox)  


    In [5]: dir = a[:,1,:3]

    In [6]: np.sqrt(np.sum(dir*dir,axis=1))
    Out[6]:
    array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,


Direction is normalized, so back up the final photons offseting the position by a negative multiple of the direction vector.::

    In [7]: a[:,0,:3] +=  -10.*a[:,1,:3]  


    In [8]: pos = a[:,0,:3]

    In [9]: np.sqrt(np.sum(pos*pos,axis=1))
    Out[9]:
    array([20040.002, 20040.004, 20040.   , 20040.   , 20040.002, 20040.   , 20040.002, 20040.002, 20040.   , 20040.   , 20040.   , 20040.   , 20040.   , 20040.   , 20040.   , 20040.   , 20040.   ,
           20040.002, 20040.   , 20040.   , 20040.   , 20039.998, 20040.   , 20040.002, 20040.   , 20040.   , 20040.   , 20040.002, 20040.002, 20040.   , 20040.   , 20040.002, 20040.   , 20040.002,
           20040.   , 20040.   , 20040.   , 20040.002, 20040.   , 20040.   , 20040.   , 20040.   , 20040.   , 20040.002, 20040.002, 20040.   , 20040.002, 20040.   , 20040.002, 20040.   , 20040.   ,

    In [10]: np.save("/tmp/b_ox_Tyvek_minus_10mm.npy", a )      

    epsilon:~ blyth$ scp /tmp/b_ox_Tyvek_minus_10mm.npy P:/tmp/


::

    P[blyth@localhost ~]$ jvi
    P[blyth@localhost ~]$ 
    P[blyth@localhost ~]$ jfu
    P[blyth@localhost ~]$ t tds3ip
    tds3ip () 
    { 
        local path=/tmp/b_ox_Tyvek_minus_10mm.npy;
        export OPTICKS_EVENT_PFX=tds3ip;
        export INPUT_PHOTON_PATH=$path;
        tds3 --dindex 0,1,2,3,4,5,6,7
    }




Dramatic history difference
--------------------------------

tds3ip.sh get
tds3ip.sh 1::


    AB(1,natural,g4live)  None 0     file_photons 451   load_slice 0:100k:   loaded_photons 451  
    A tds3ip/g4live/natural/  1 :  20210613-1458 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/tds3ip/evt/g4live/natural/1/fdom.npy () 
    B tds3ip/g4live/natural/ -1 :  20210613-1458 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/tds3ip/evt/g4live/natural/-1/fdom.npy (recstp) 
    .
    '#ab.__str__.ahis'
    ab.ahis
    .    all_seqhis_ana  cfo:sum  1:g4live:tds3ip   -1:g4live:tds3ip        c2        ab        ba 
    .                                451       451       646.00/4 = 161.50  (pval:0.000 prob:1.000)  
    0000               8d         0       451   -451           451.00        0.000 +- 0.000        0.000 +- 0.000  [2 ] TO SA
    0001             4ccd        56         0     56            56.00        0.000 +- 0.000        0.000 +- 0.000  [4 ] TO BT BT AB
    0002            4cccd        53         0     53            53.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] TO BT BT BT AB
    0003            49ccd        47         0     47            47.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] TO BT BT DR AB
    0004           4c9ccd        39         0     39            39.00        0.000 +- 0.000        0.000 +- 0.000  [6 ] TO BT BT DR BT AB
    0005       cccacccccd        29         0     29             0.00        0.000 +- 0.000        0.000 +- 0.000  [10] TO BT BT BT BT BT SR BT BT BT
    0006             8ccd        18         0     18             0.00        0.000 +- 0.000        0.000 +- 0.000  [4 ] TO BT BT SA
    0007         7ccc9ccd        16         0     16             0.00        0.000 +- 0.000        0.000 +- 0.000  [8 ] TO BT BT DR BT BT BT SD
    0008          4c99ccd        14         0     14             0.00        0.000 +- 0.000        0.000 +- 0.000  [7 ] TO BT BT DR DR BT AB
    0009            89ccd        12         0     12             0.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] TO BT BT DR SA
    0010       cccccccccd        10         0     10             0.00        0.000 +- 0.000        0.000 +- 0.000  [10] TO BT BT BT BT BT BT BT BT BT
    0011          4cb9ccd         7         0      7             0.00        0.000 +- 0.000        0.000 +- 0.000  [7 ] TO BT BT DR BR BT AB
    0012           499ccd         6         0      6             0.00        0.000 +- 0.000        0.000 +- 0.000  [6 ] TO BT BT DR DR AB
    0013           4b9ccd         5         0      5             0.00        0.000 +- 0.000        0.000 +- 0.000  [6 ] TO BT BT DR BR AB
    0014         8ccc9ccd         5         0      5             0.00        0.000 +- 0.000        0.000 +- 0.000  [8 ] TO BT BT DR BT BT BT SA
    0015         4c999ccd         5         0      5             0.00        0.000 +- 0.000        0.000 +- 0.000  [8 ] TO BT BT DR DR DR BT AB
    0016          4999ccd         4         0      4             0.00        0.000 +- 0.000        0.000 +- 0.000  [7 ] TO BT BT DR DR DR AB
    0017           899ccd         4         0      4             0.00        0.000 +- 0.000        0.000 +- 0.000  [6 ] TO BT BT DR DR SA
    0018         4ccccccd         4         0      4             0.00        0.000 +- 0.000        0.000 +- 0.000  [8 ] TO BT BT BT BT BT BT AB
    0019         49999ccd         4         0      4             0.00        0.000 +- 0.000        0.000 +- 0.000  [8 ] TO BT BT DR DR DR DR AB
    .                                451       451       646.00/4 = 161.50  (pval:0.000 prob:1.000)  
    '#ab.__str__.flg'



    In [6]: a.rpostr()[:20]                                                                                                                                                                                                                 
    Out[6]: 
    A([[20039.3388, 20050.2861, 20051.5839, 23193.1158,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ],
       [20039.1986, 20049.9672, 20052.4707, 20575.8718, 20585.2071, 20589.536 , 20595.9762, 20590.9699, 20586.6409, 20574.8025],
       [20041.0468, 20049.3645, 20051.8257, 24537.2156, 24391.3566,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ],
       [20040.6836, 20049.0477, 20050.6468, 20051.893 , 20051.893 ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ],
       [20040.7623, 20051.108 , 20051.108 , 20576.4905, 20586.836 , 20592.0087, 20615.3772, 20627.1062, 20641.515 , 21783.0406],
       [20039.1903, 20049.7732, 17820.8856, 17699.166 ,  3678.0252,  1907.9759,     0.    ,     0.    ,     0.    ,     0.    ],
       [20040.2316, 20050.2212, 20052.6192, 20576.4389, 20586.4286, 20592.1673, 20628.4205, 20638.4103, 20651.3754, 21854.1884],
       [20040.2089, 20049.1196, 20051.4921, 25153.2407, 20733.2044, 20727.8007, 20725.804 , 20721.6809,     0.    ,     0.    ],
       [20040.2092, 20049.9733, 20051.4272, 20053.903 , 20053.903 ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ],
       [20039.3957, 20049.2541, 20051.6555, 20576.1687, 20586.0271, 20590.83  , 20627.2124, 20635.7161, 20649.7282, 28467.9708],
       [20039.2714, 20050.0787, 20052.3088, 21471.8234,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ],
       [20040.9849, 20049.3752, 20051.7676, 20055.3733, 20055.3733,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ],
       [20039.8615, 20050.3488, 20052.9257, 21858.7335, 20052.1421, 21750.1594, 20051.9169, 22295.1602, 20051.3971, 22279.0026],
       [20040.2689, 20050.7394, 20051.5368, 23201.8686, 20052.7657,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ],
       [20040.6684, 20050.6729, 20052.7875, 20576.7248, 20587.1849, 20591.1302, 20597.1897, 20591.1302, 20586.7296, 20576.7248],
       [20041.1012, 20050.9524, 20052.2836, 30001.6096,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ],
       [20039.531 , 20050.1415, 20052.4103, 27325.2418, 20051.4752, 20528.5985, 20528.5985,     0.    ,     0.    ,     0.    ],
       [20039.1954, 20050.0059, 20051.6052, 20051.6052,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ],
       [20039.0745, 20049.8542, 20052.3332, 24818.4814, 20053.1513, 25730.236 , 20597.4118, 20597.4118,     0.    ,     0.    ],
       [20040.0603, 20049.5478, 20051.3264, 20070.6791, 20070.6791,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ]])

    In [7]: b.rpostr()[:20]                                                                                                                                                                                                                 
    Out[7]: 
    A([[20039.3388, 20050.2861,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ],
       [20039.1986, 20049.9672,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ],
       [20041.0468, 20049.3645,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ],
       [20040.6836, 20049.0477,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ],
       [20040.7623, 20051.108 ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ],
       [20039.1903, 20049.7732,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ],
       [20040.2316, 20050.2212,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ],
       [20040.2089, 20049.1196,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ],
       [20040.2092, 20049.9733,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ],
       [20039.3957, 20049.2541,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ],
       [20039.2714, 20050.0787,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ],
       [20040.9849, 20049.3752,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ],
       [20039.8615, 20050.3488,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ],
       [20040.2689, 20050.7394,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ],
       [20040.6684, 20050.6729,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ],
       [20041.1012, 20050.9524,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ],
       [20039.531 , 20050.1415,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ],
       [20039.1954, 20050.0059,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ],
       [20039.0745, 20049.8542,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ],
       [20040.0603, 20049.5478,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ,     0.    ]])

    In [8]:                                           



TODO some python that makes sense of the bndseq::


    In [6]: np.set_printoptions(edgeitems=10)                                                                                                                                                                                               
    In [7]: a.bn.view(np.int8).reshape(-1,16)                                                                                                                                                                                               
    Out[7]: 
    A([[ 16,  15,  13,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 16,  15, -17,  17, -23, -36,  24, -17,  17,   0,   0,   0,   0,   0,   0,   0],
       [ 16,  15,  13, -14,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 16, -14,  15,  14,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 16,  15, -17,  17, -23,  24, -17,  17,  13,   0,   0,   0,   0,   0,   0,   0],
       [ 16, -17, -18,  18,  18,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 16,  15, -17,  17, -23,  24, -17,  17,  13,   0,   0,   0,   0,   0,   0,   0],
       [ 16,  15,  13, -17,  17, -23, -35,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 16,  15, -14,  14,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 16,  15, -17,  17, -23,  24, -17,  17, -14,   0,   0,   0,   0,   0,   0,   0],
       ...,
       [ 16,  15,  13, -14, -14,  14,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 16,  15,  13, -14, -15, -14, -15,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 16,  15,  13, -15,  13, -15, -17,  17,  17,   0,   0,   0,   0,   0,   0,   0],
       [ 16,  15, -17,  17, -23, -36,  24, -17,  17,   0,   0,   0,   0,   0,   0,   0],
       [ 16,  15,  13,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 16,  15,  13, -14, -15,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 16,  15,  13,  13,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 16,  15,  13, -14,  14,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 16,  15,  13, -14,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 16,  15,  13, -14,  14,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]], dtype=int8)


::

    In [1]: from opticks.ana.blib import BLib                                                                                                                                                                                               

    In [2]: blib = BLib()                                                                                                                                                                                                                   

    In [3]: bn = [ 16,  15, -17,  17, -23, -36,  24, -17,  17,   0,   0,   0,   0,   0,   0,   0]                                                                                                                                           

    In [4]: print(blib.format(bn))                                                                                                                                                                                                          
     16 : Tyvek///Water
     15 : vetoWater/CDTyvekSurface//Tyvek
    -17 : Water///Acrylic
     17 : Water///Acrylic
    -23 : Water///Water
    -36 : Pyrex//PMT_20inch_veto_mirror_logsurf1/Vacuum
     24 : Water///Pyrex
    -17 : Water///Acrylic
     17 : Water///Acrylic




BP=G4OpBoundaryProcess::PostStepDoIt tds3ip
----------------------------------------------

* hmm, will probably need to rebuild Geant4 with debug symbols 

::

    P[blyth@localhost ~]$ BP=G4OpBoundaryProcess::PostStepDoIt tds3ip



    2021-06-13 22:07:57.284 INFO  [125040] [G4Opticks::setInputPhotons@1934]  input_photons 451,4,4
    Begin of Event --> 0

    Breakpoint 1, 0x00007fffcfa1ef40 in G4OpBoundaryProcess::PostStepDoIt(G4Track const&, G4Step const&) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4processes.so
    (gdb) bt
    #0  0x00007fffcfa1ef40 in G4OpBoundaryProcess::PostStepDoIt(G4Track const&, G4Step const&) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4processes.so
    #1  0x00007fffd04ad379 in G4SteppingManager::InvokePSDIP(unsigned long) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4tracking.so
    #2  0x00007fffd04ad7ff in G4SteppingManager::InvokePostStepDoItProcs() () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4tracking.so
    #3  0x00007fffd04aa8a5 in G4SteppingManager::Stepping() () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4tracking.so
    #4  0x00007fffd04b60fd in G4TrackingManager::ProcessOneTrack(G4Track*) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4tracking.so
    #5  0x00007fffd06edb53 in G4EventManager::DoProcessing(G4Event*) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4event.so
    #6  0x00007fffc289e760 in G4SvcRunManager::SimulateEvent(int) () from /home/blyth/junotop/offline/InstallArea/Linux-x86_64/lib/libG4Svc.so
    #7  0x00007fffc1dfea3c in DetSimAlg::execute (this=0x250d1c0) at ../src/DetSimAlg.cc:112
    #8  0x00007fffef13836d in Task::execute() () from /home/blyth/junotop/sniper/InstallArea/Linux-x86_64/lib/libSniperKernel.so
    #9  0x00007fffef13d568 in TaskWatchDog::run() () from /home/blyth/junotop/sniper/InstallArea/Linux-x86_64/lib/libSniperKernel.so


::

    (gdb) f 0
    #0  0x00007fffcfa1ef40 in G4OpBoundaryProcess::PostStepDoIt(G4Track const&, G4Step const&) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4processes.so
    (gdb) list
    11	{
    12		/* 754 requires that FP exceptions run in "no stop" mode by default,
    13		 * and until C vendors implement C99's ways to control FP exceptions,
    14		 * Python requires non-stop mode.  Alas, some platforms enable FP
    15		 * exceptions by default.  Here we disable them.
    16		 */
    17	#ifdef __FreeBSD__
    18		fedisableexcept(FE_OVERFLOW);
    19	#endif
    20		return Py_Main(argc, argv);
    (gdb) f 1
    #1  0x00007fffd04ad379 in G4SteppingManager::InvokePSDIP(unsigned long) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4tracking.so
    (gdb) list
    21	}
    (gdb) 





