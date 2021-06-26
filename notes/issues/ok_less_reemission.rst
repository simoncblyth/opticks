ok_less_reemission
=====================

* prior :doc:`ok_lacks_SI-4BT-SD`


After remove the sticks : the poppy becomes somewhat less OK reemission 
-------------------------------------------------------------------------------------

* less "SI RE .." and more of "SI BT .."

::

    In [4]: ab.his[:50]                                                                                                                                                                             
    Out[4]: 
    ab.his
    .       seqhis_ana  cfo:sum  1:g4live:tds3gun   -1:g4live:tds3gun        c2        ab        ba 
    .                              11684     11684       109.20/59 =  1.85  (pval:0.000 prob:1.000)  
       n             iseq         a         b    a-b       (a-b)^2/(a+b)         a/b                   b/a           [ns] label
    0000               42      1741      1721     20              0.12         1.012 +- 0.024        0.989 +- 0.024  [2 ] SI AB
    0001            7ccc2      1480      1406     74              1.90         1.053 +- 0.027        0.950 +- 0.025  [5 ] SI BT BT BT SD
    0002           7ccc62       737       666     71              3.59         1.107 +- 0.041        0.904 +- 0.035  [6 ] SI SC BT BT BT SD
    0003            8ccc2       660       597     63              3.16         1.106 +- 0.043        0.905 +- 0.037  [5 ] SI BT BT BT SA
    0004             8cc2       629       615     14              0.16         1.023 +- 0.041        0.978 +- 0.039  [4 ] SI BT BT SA
    0005              452       436       536   -100             10.29         0.813 +- 0.039        1.229 +- 0.053  [3 ] SI RE AB               ## LESS OK_RE 
    0006           7ccc52       424       438    -14              0.23         0.968 +- 0.047        1.033 +- 0.049  [6 ] SI RE BT BT BT SD
    0007              462       425       405     20              0.48         1.049 +- 0.051        0.953 +- 0.047  [3 ] SI SC AB
    0008           8ccc62       283       262     21              0.81         1.080 +- 0.064        0.926 +- 0.057  [6 ] SI SC BT BT BT SA
    0009          7ccc662       266       222     44              3.97         1.198 +- 0.073        0.835 +- 0.056  [7 ] SI SC SC BT BT BT SD
    0010            8cc62       209       212     -3              0.02         0.986 +- 0.068        1.014 +- 0.070  [5 ] SI SC BT BT SA
    0011          7ccc652       187       205    -18              0.83         0.912 +- 0.067        1.096 +- 0.077  [7 ] SI RE SC BT BT BT SD
    0012           8ccc52       189       201    -12              0.37         0.940 +- 0.068        1.063 +- 0.075  [6 ] SI RE BT BT BT SA
    0013            8cc52       151       192    -41              4.90         0.786 +- 0.064        1.272 +- 0.092  [5 ] SI RE BT BT SA         ### LESS OK:RE 
    0014               41       162       145     17              0.94         1.117 +- 0.088        0.895 +- 0.074  [2 ] CK AB
    0015          7ccc552       133       160    -27              2.49         0.831 +- 0.072        1.203 +- 0.095  [7 ] SI RE RE BT BT BT SD
    0016             4552       124       165    -41              5.82         0.752 +- 0.067        1.331 +- 0.104  [4 ] SI RE RE AB            ### LESS OK:RE
    0017             4cc2       133       115     18              1.31         1.157 +- 0.100        0.865 +- 0.081  [4 ] SI BT BT AB
    0018             4662       136       110     26              2.75         1.236 +- 0.106        0.809 +- 0.077  [4 ] SI SC SC AB
    0019             4652       121       117      4              0.07         1.034 +- 0.094        0.967 +- 0.089  [4 ] SI RE SC AB
    0020          8ccc662        86       108    -22              2.49         0.796 +- 0.086        1.256 +- 0.121  [7 ] SI SC SC BT BT BT SA
    0021         7ccc6662        87        91     -4              0.09         0.956 +- 0.102        1.046 +- 0.110  [8 ] SI SC SC SC BT BT BT SD
    0022          8ccc652        77        79     -2              0.03         0.975 +- 0.111        1.026 +- 0.115  [7 ] SI RE SC BT BT BT SA
    0023         7ccc6652        59        86    -27              5.03         0.686 +- 0.089        1.458 +- 0.157  [8 ] SI RE SC SC BT BT BT SD
    0024          7ccc562        76        51     25              4.92         1.490 +- 0.171        0.671 +- 0.094  [7 ] SI SC RE BT BT BT SD
    0025           8cc662        57        69    -12              1.14         0.826 +- 0.109        1.211 +- 0.146  [6 ] SI SC SC BT BT SA
    0026          8ccc552        62        63     -1              0.01         0.984 +- 0.125        1.016 +- 0.128  [7 ] SI RE RE BT BT BT SA
    0027         7ccc6552        53        71    -18              2.61         0.746 +- 0.103        1.340 +- 0.159  [8 ] SI RE RE SC BT BT BT SD
    0028             4562        49        66    -17              2.51         0.742 +- 0.106        1.347 +- 0.166  [4 ] SI SC RE AB
    0029           8cc552        38        70    -32              9.48         0.543 +- 0.088        1.842 +- 0.220  [6 ] SI RE RE BT BT SA
    0030            4cc62        55        52      3              0.08         1.058 +- 0.143        0.945 +- 0.131  [5 ] SI SC BT BT AB
    0031           8cc652        56        50      6              0.34         1.120 +- 0.150        0.893 +- 0.126  [6 ] SI RE SC BT BT SA
    0032           7cccc2        53        51      2              0.04         1.039 +- 0.143        0.962 +- 0.135  [6 ] SI BT BT BT BT SD
    0033              4c2        57        35     22              5.26         1.629 +- 0.216        0.614 +- 0.104  [3 ] SI BT AB
    0034            45552        29        49    -20              5.13         0.592 +- 0.110        1.690 +- 0.241  [5 ] SI RE RE RE AB
    0035            46662        42        34      8              0.84         1.235 +- 0.191        0.810 +- 0.139  [5 ] SI SC SC SC AB
    0036            4cc52        34        40     -6              0.49         0.850 +- 0.146        1.176 +- 0.186  [5 ] SI RE BT BT AB
    0037         8ccc6662        36        36      0              0.00         1.000 +- 0.167        1.000 +- 0.167  [8 ] SI SC SC SC BT BT BT SA
    0038         7ccc5552        36        35      1              0.01         1.029 +- 0.171        0.972 +- 0.164  [8 ] SI RE RE RE BT BT BT SD
    0039            46552        37        31      6              0.53         1.194 +- 0.196        0.838 +- 0.150  [5 ] SI RE RE SC AB
    0040            46652        38        29      9              1.21         1.310 +- 0.213        0.763 +- 0.142  [5 ] SI RE SC SC AB
    0041          8ccc562        33        33      0              0.00         1.000 +- 0.174        1.000 +- 0.174  [7 ] SI SC RE BT BT BT SA



Event 2 similar::






    In [1]: ab.his                                                                                                                                                                                  
    Out[1]: 
    ab.his
    .       seqhis_ana  cfo:sum  2:g4live:tds3gun   -2:g4live:tds3gun        c2        ab        ba 
    .                              11594     11594        98.62/61 =  1.62  (pval:0.002 prob:0.998)  
       n             iseq         a         b    a-b       (a-b)^2/(a+b)         a/b                   b/a           [ns] label
    0000               42      1741      1702     39              0.44         1.023 +- 0.025        0.978 +- 0.024  [2 ] SI AB
    0001            7ccc2      1499      1477     22              0.16         1.015 +- 0.026        0.985 +- 0.026  [5 ] SI BT BT BT SD
    0002           7ccc62       716       657     59              2.54         1.090 +- 0.041        0.918 +- 0.036  [6 ] SI SC BT BT BT SD
    0003            8ccc2       665       641     24              0.44         1.037 +- 0.040        0.964 +- 0.038  [5 ] SI BT BT BT SA
    0004             8cc2       605       630    -25              0.51         0.960 +- 0.039        1.041 +- 0.041  [4 ] SI BT BT SA
    0005              452       428       519    -91              8.74         0.825 +- 0.040        1.213 +- 0.053  [3 ] SI RE AB
    0006           7ccc52       406       437    -31              1.14         0.929 +- 0.046        1.076 +- 0.051  [6 ] SI RE BT BT BT SD
    0007              462       427       383     44              2.39         1.115 +- 0.054        0.897 +- 0.046  [3 ] SI SC AB
    0008           8ccc62       266       251     15              0.44         1.060 +- 0.065        0.944 +- 0.060  [6 ] SI SC BT BT BT SA
    0009          7ccc662       249       266    -17              0.56         0.936 +- 0.059        1.068 +- 0.066  [7 ] SI SC SC BT BT BT SD
    0010            8cc62       246       218     28              1.69         1.128 +- 0.072        0.886 +- 0.060  [5 ] SI SC BT BT SA
    0011           8ccc52       188       228    -40              3.85         0.825 +- 0.060        1.213 +- 0.080  [6 ] SI RE BT BT BT SA
    0012          7ccc652       186       186      0              0.00         1.000 +- 0.073        1.000 +- 0.073  [7 ] SI RE SC BT BT BT SD
    0013            8cc52       156       164     -8              0.20         0.951 +- 0.076        1.051 +- 0.082  [5 ] SI RE BT BT SA
    0014             4552       128       174    -46              7.01         0.736 +- 0.065        1.359 +- 0.103  [4 ] SI RE RE AB
    0015             4cc2       129       134     -5              0.10         0.963 +- 0.085        1.039 +- 0.090  [4 ] SI BT BT AB
    0016             4662       132       130      2              0.02         1.015 +- 0.088        0.985 +- 0.086  [4 ] SI SC SC AB
    0017               41       122       123     -1              0.00         0.992 +- 0.090        1.008 +- 0.091  [2 ] CK AB
    0018             4652       119       118      1              0.00         1.008 +- 0.092        0.992 +- 0.091  [4 ] SI RE SC AB
    .                              11594     11594        98.62/61 =  1.62  (pval:0.002 prob:0.998)  



Back to event 1::

    tds3gun.sh 1 


    In [2]: a.sel = "SI RE .."   

    In [4]: a.seqhis_ana.table[:20]                                                                                                                                                                 
    Out[4]: 
    seqhis_ana
    .                     cfo:-  1:g4live:tds3gun 
    .                               3080         1.00 
       n             iseq         frac           a    a-b      [ns] label
    0000              452        0.142         436        [3 ] SI RE AB
    0001           7ccc52        0.138         424        [6 ] SI RE BT BT BT SD
    0002           8ccc52        0.061         189        [6 ] SI RE BT BT BT SA
    0003          7ccc652        0.061         187        [7 ] SI RE SC BT BT BT SD
    0004            8cc52        0.049         151        [5 ] SI RE BT BT SA
    0005          7ccc552        0.043         133        [7 ] SI RE RE BT BT BT SD
    0006             4552        0.040         124        [4 ] SI RE RE AB
    0007             4652        0.039         121        [4 ] SI RE SC AB
    0008          8ccc652        0.025          77        [7 ] SI RE SC BT BT BT SA
    0009          8ccc552        0.020          62        [7 ] SI RE RE BT BT BT SA
    0010         7ccc6652        0.019          59        [8 ] SI RE SC SC BT BT BT SD
    0011           8cc652        0.018          56        [6 ] SI RE SC BT BT SA
    0012         7ccc6552        0.017          53        [8 ] SI RE RE SC BT BT BT SD
    0013            46652        0.012          38        [5 ] SI RE SC SC AB
    0014           8cc552        0.012          38        [6 ] SI RE RE BT BT SA
    0015            46552        0.012          37        [5 ] SI RE RE SC AB
    0016         7ccc5552        0.012          36        [8 ] SI RE RE RE BT BT BT SD
    0017            4cc52        0.011          34        [5 ] SI RE BT BT AB
    0018            45552        0.009          29        [5 ] SI RE RE RE AB
    .                               3080         1.00 


    In [5]: b.sel = "SI RE .."                                                                                                                                                                      

    In [6]: b.seqhis_ana.table[:20]                                                                                                                                                                 
    Out[6]: 
    seqhis_ana
    .                     cfo:-  -1:g4live:tds3gun 
    .                               3567         1.00 
       n             iseq         frac           a    a-b      [ns] label
    0000              452        0.150         536        [3 ] SI RE AB
    0001           7ccc52        0.123         438        [6 ] SI RE BT BT BT SD
    0002          7ccc652        0.057         205        [7 ] SI RE SC BT BT BT SD
    0003           8ccc52        0.056         201        [6 ] SI RE BT BT BT SA
    0004            8cc52        0.054         192        [5 ] SI RE BT BT SA
    0005             4552        0.046         165        [4 ] SI RE RE AB
    0006          7ccc552        0.045         160        [7 ] SI RE RE BT BT BT SD
    0007             4652        0.033         117        [4 ] SI RE SC AB
    0008         7ccc6652        0.024          86        [8 ] SI RE SC SC BT BT BT SD
    0009          8ccc652        0.022          79        [7 ] SI RE SC BT BT BT SA
    0010         7ccc6552        0.020          71        [8 ] SI RE RE SC BT BT BT SD
    0011           8cc552        0.020          70        [6 ] SI RE RE BT BT SA
    0012          8ccc552        0.018          63        [7 ] SI RE RE BT BT BT SA
    0013           8cc652        0.014          50        [6 ] SI RE SC BT BT SA
    0014            45552        0.014          49        [5 ] SI RE RE RE AB
    0015            4cc52        0.011          40        [5 ] SI RE BT BT AB
    0016         7ccc5552        0.010          35        [8 ] SI RE RE RE BT BT BT SD
    0017         8ccc6652        0.010          34        [8 ] SI RE SC SC BT BT BT SA
    0018            46552        0.009          31        [5 ] SI RE RE SC AB
    .                               3567         1.00 



15% more RE in G4::

    In [7]: 3567./3080.                                                                                                                                                                             
    Out[7]: 1.158116883116883


Boundary check::

    In [2]: a.sel = "SI RE AB"                                                                                                                                                                      

    In [3]: a.bn.shape                                                                                                                                                                              
    Out[3]: (436, 1, 4)

    In [4]: a.bn.view(np.int8).reshape(-1,16)                                                                                                                                                       
    Out[4]: 
    A([[18, 18,  0, ...,  0,  0,  0],
       [18, 18,  0, ...,  0,  0,  0],
       [18, 18,  0, ...,  0,  0,  0],
       ...,
       [18, 18,  0, ...,  0,  0,  0],
       [18, 18,  0, ...,  0,  0,  0],
       [18, 18,  0, ...,  0,  0,  0]], dtype=int8)


    In [6]: print(a.blib.format([18,18]))                                                                                                                                                           
     18 : Acrylic///LS
     18 : Acrylic///LS



::

    In [2]: run material.py                                                                                                                                                                         
    INFO:opticks.ana.proplib:names : None 
    INFO:opticks.ana.proplib:npath : /usr/local/opticks/geocache/OKX4Test_lWorld0x32a96e0_PV_g4live/g4ok_gltf/a3cbac8189a032341f76682cdb4f47b6/1/GItemList/GMaterialLib.txt 
    INFO:opticks.ana.proplib:names : ['LS', 'Steel', 'Tyvek', 'Air', 'Scintillator', 'TiO2Coating', 'Adhesive', 'Aluminium', 'Rock', 'LatticedShellSteel', 'Acrylic', 'PE_PA', 'Vacuum', 'Pyrex', 'Water', 'vetoWater', 'Galactic'] 
    INFO:opticks.ana.main:envvar OPTICKS_ANA_DEFAULTS -> defaults {'det': 'g4live', 'cat': 'g4live', 'src': 'torch', 'tag': '1', 'pfx': 'OKTest'} 
    WARNING:opticks.ana.env:legacy_init : OPTICKS_KEY envvar deleted for legacy running, unset IDPATH to use direct_init
    INFO:__main__:mat Water 
            wavelen      rindex      abslen     scatlen    reemprob    groupvel LS
    [[[   300.          1.5264      0.975    4887.5513      0.7214    177.2066]
      [   400.          1.5       195.5178  17976.7012      0.8004    189.7664]
      [   500.          1.4902 114196.2188  43987.5156      0.1231    195.3692]
      [   600.          1.4837  46056.8906 116999.7344      0.0483    198.683 ]]]




Add an input photon type "InwardsCubeCorners17699"
-----------------------------------------------------


::

    [2021-06-26 10:30:13,900] p77227 {/Users/blyth/opticks/ana/input_photons.py:294} INFO - load InwardsCubeCorners17699 from /Users/blyth/.opticks/InputPhotons/InwardsCubeCorners17699.npy /Users/blyth/.opticks/InputPhotons/InwardsCubeCorners17699.json 
    {'seed': 0, 'name': 'InwardsCubeCorners17699', 'creator': 'input_photons.py', 'num': 8}
    [[-10218.522 -10218.522 -10218.522      0.1        0.577      0.577      0.577      1.        -0.707      0.         0.707    440.         0.         0.         0.         0.   ]
     [ 10218.522 -10218.522 -10218.522      0.2       -0.577      0.577      0.577      1.        -0.707      0.        -0.707    440.         0.         0.         0.         0.   ]
     [-10218.522  10218.522 -10218.522      0.3        0.577     -0.577      0.577      1.        -0.707      0.         0.707    440.         0.         0.         0.         0.   ]
     [ 10218.522  10218.522 -10218.522      0.4       -0.577     -0.577      0.577      1.        -0.707      0.        -0.707    440.         0.         0.         0.         0.   ]
     [-10218.522 -10218.522  10218.522      0.5        0.577      0.577     -0.577      1.         0.707     -0.         0.707    440.         0.         0.         0.         0.   ]
     [ 10218.522 -10218.522  10218.522      0.6       -0.577      0.577     -0.577      1.         0.707      0.        -0.707    440.         0.         0.         0.         0.   ]
     [-10218.522  10218.522  10218.522      0.7        0.577     -0.577     -0.577      1.         0.707     -0.         0.707    440.         0.         0.         0.         0.   ]
     [ 10218.522  10218.522  10218.522      0.8       -0.577     -0.577     -0.577      1.         0.707      0.        -0.707    440.         0.         0.         0.         0.   ]]
    [2021-06-26 10:30:13,902] p77227 {/Users/blyth/opticks/ana/input_photons.py:294} INFO - load InwardsCubeCorners1 from /Users/blyth/.opticks/InputPhotons/InwardsCubeCorners1.npy /Users/blyth/.opticks/InputPhotons/InwardsCubeCorners1.json 
    {'seed': 0, 'name': 'InwardsCubeCorners1', 'creator': 'input_photons.py', 'num': 8}
    [[ -0.577  -0.577  -0.577   0.1     0.577   0.577   0.577   1.     -0.707   0.      0.707 440.      0.      0.      0.      0.   ]
     [  0.577  -0.577  -0.577   0.2    -0.577   0.577   0.577   1.     -0.707   0.     -0.707 440.      0.      0.      0.      0.   ]
     [ -0.577   0.577  -0.577   0.3     0.577  -0.577   0.577   1.     -0.707   0.      0.707 440.      0.      0.      0.      0.   ]
     [  0.577   0.577  -0.577   0.4    -0.577  -0.577   0.577   1.     -0.707   0.     -0.707 440.      0.      0.      0.      0.   ]
     [ -0.577  -0.577   0.577   0.5     0.577   0.577  -0.577   1.      0.707  -0.      0.707 440.      0.      0.      0.      0.   ]
     [  0.577  -0.577   0.577   0.6    -0.577   0.577  -0.577   1.      0.707   0.     -0.707 440.      0.      0.      0.      0.   ]
     [ -0.577   0.577   0.577   0.7     0.577  -0.577  -0.577   1.      0.707  -0.      0.707 440.      0.      0.      0.      0.   ]
     [  0.577   0.577   0.577   0.8    -0.577  -0.577  -0.577   1.      0.707   0.     -0.707 440.      0.      0.      0.      0.   ]]



    In [3]: p[:,0,:3]                                                                                                                                                                                       
    Out[3]: 
    array([[-10218.522, -10218.522, -10218.522],
           [ 10218.522, -10218.522, -10218.522],
           [-10218.522,  10218.522, -10218.522],
           [ 10218.522,  10218.522, -10218.522],
           [-10218.522, -10218.522,  10218.522],
           [ 10218.522, -10218.522,  10218.522],
           [-10218.522,  10218.522,  10218.522],
           [ 10218.522,  10218.522,  10218.522]], dtype=float32)

    In [4]: np.sqrt(np.sum(p[:,0,:3]*p[:,0,:3], axis=1 ))                                                                                                                                                   
    Out[4]: array([17699., 17699., 17699., 17699., 17699., 17699., 17699., 17699.], dtype=float32)

    In [5]: p[:,1,:3]                                                                                                                                                                                       
    Out[5]: 
    array([[ 0.577,  0.577,  0.577],
           [-0.577,  0.577,  0.577],
           [ 0.577, -0.577,  0.577],
           [-0.577, -0.577,  0.577],
           [ 0.577,  0.577, -0.577],
           [-0.577,  0.577, -0.577],
           [ 0.577, -0.577, -0.577],
           [-0.577, -0.577, -0.577]], dtype=float32)

    In [6]: 17699.*2                                                                                                                                                                                        
    Out[6]: 35398.0


