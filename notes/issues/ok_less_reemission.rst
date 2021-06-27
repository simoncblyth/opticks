ok_less_reemission
=====================

* prior :doc:`ok_lacks_SI-4BT-SD`
* next :doc:`tds3ip_InwardsCubeCorners17699_at_7_wavelengths`



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



Curious the input photons show no reemission in G4  : fixed by CPhotonInfo::Get switch 
-------------------------------------------------------------------------------------------

* rerun with tds3gun shows no such problem, it is specific to input photons 

::

    In [16]: ab.his[:40]                                                                                                                                                                                    
    Out[16]: 
    ab.his
    .       seqhis_ana  cfo:sum  1:g4live:tds3ip   -1:g4live:tds3ip        c2        ab        ba 
    .                              80000     80000      5442.81/126 = 43.20  (pval:0.000 prob:1.000)  
       n             iseq         a         b    a-b       (a-b)^2/(a+b)         a/b                   b/a           [ns] label
    0000               4d     12832     16531   -3699            465.98         0.776 +- 0.007        1.288 +- 0.010  [2 ] TO AB
    0001           7ccc6d     12583     12324    259              2.69         1.021 +- 0.009        0.979 +- 0.009  [6 ] TO SC BT BT BT SD
    0002            7cccd     10712     10911   -199              1.83         0.982 +- 0.009        1.019 +- 0.010  [5 ] TO BT BT BT SD
    0003              46d      4609      5969   -1360            174.85         0.772 +- 0.011        1.295 +- 0.017  [3 ] TO SC AB
    0004           8ccc6d      4512      4666   -154              2.58         0.967 +- 0.014        1.034 +- 0.015  [6 ] TO SC BT BT BT SA
    0005          7ccc66d      4527      4603    -76              0.63         0.983 +- 0.015        1.017 +- 0.015  [7 ] TO SC SC BT BT BT SD
    0006            8cccd      3814      3899    -85              0.94         0.978 +- 0.016        1.022 +- 0.016  [5 ] TO BT BT BT SA
    0007            8cc6d      3402      3365     37              0.20         1.011 +- 0.017        0.989 +- 0.017  [5 ] TO SC BT BT SA
    0008             466d      1618      2005   -387             41.34         0.807 +- 0.020        1.239 +- 0.028  [4 ] TO SC SC AB
    0009         7ccc666d      1717      1610    107              3.44         1.066 +- 0.026        0.938 +- 0.023  [8 ] TO SC SC SC BT BT BT SD
    0010          8ccc66d      1654      1665    -11              0.04         0.993 +- 0.024        1.007 +- 0.025  [7 ] TO SC SC BT BT BT SA
    0011           8cc66d      1248      1156     92              3.52         1.080 +- 0.031        0.926 +- 0.027  [6 ] TO SC SC BT BT SA
    0012            4cc6d      1090      1044     46              0.99         1.044 +- 0.032        0.958 +- 0.030  [5 ] TO SC BT BT AB
    0013            4666d       591       710   -119             10.88         0.832 +- 0.034        1.201 +- 0.045  [5 ] TO SC SC SC AB
    0014         8ccc666d       587       610    -23              0.44         0.962 +- 0.040        1.039 +- 0.042  [8 ] TO SC SC SC BT BT BT SA
    0015             4ccd       532       528      4              0.02         1.008 +- 0.044        0.992 +- 0.043  [4 ] TO BT BT AB
    0016        7ccc6666d       517       541    -24              0.54         0.956 +- 0.042        1.046 +- 0.045  [9 ] TO SC SC SC SC BT BT BT SD
    0017          7cccc6d       433       409     24              0.68         1.059 +- 0.051        0.945 +- 0.047  [7 ] TO SC BT BT BT BT SD
    0018             4c6d       385       435    -50              3.05         0.885 +- 0.045        1.130 +- 0.054  [4 ] TO SC BT AB
    0019          8cc666d       397       388      9              0.10         1.023 +- 0.051        0.977 +- 0.050  [7 ] TO SC SC SC BT BT SA
    0020           4cc66d       378       402    -24              0.74         0.940 +- 0.048        1.063 +- 0.053  [6 ] TO SC SC BT BT AB
    0021           7ccc5d       594         0    594            594.00         0.000 +- 0.000        0.000 +- 0.000  [6 ] TO RE BT BT BT SD
    0022       bccbccbc6d       283       276      7              0.09         1.025 +- 0.061        0.975 +- 0.059  [10] TO SC BT BR BT BT BR BT BT BR
    0023          8cccc6d       231       220     11              0.27         1.050 +- 0.069        0.952 +- 0.064  [7 ] TO SC BT BT BT BT SA
    0024              45d       446         0    446            446.00         0.000 +- 0.000        0.000 +- 0.000  [3 ] TO RE AB
    0025              4cd       197       245    -48              5.21         0.804 +- 0.057        1.244 +- 0.079  [3 ] TO BT AB
    0026           46666d       197       237    -40              3.69         0.831 +- 0.059        1.203 +- 0.078  [6 ] TO SC SC SC SC AB
    0027        8ccc6666d       200       223    -23              1.25         0.897 +- 0.063        1.115 +- 0.075  [9 ] TO SC SC SC SC BT BT BT SA
    0028        8ccaccc6d         0       387   -387            387.00         0.000 +- 0.000        0.000 +- 0.000  [9 ] TO SC BT BT BT SR BT BT SA
    0029       7ccc66666d       200       182     18              0.85         1.099 +- 0.078        0.910 +- 0.067  [10] TO SC SC SC SC SC BT BT BT SD
    0030         8caccc6d       297         0    297            297.00         0.000 +- 0.000        0.000 +- 0.000  [8 ] TO SC BT BT BT SR BT SA
    0031            4c66d       145       148     -3              0.03         0.980 +- 0.081        1.021 +- 0.084  [5 ] TO SC SC BT AB
    0032           8bcc6d       166       127     39              5.19         1.307 +- 0.101        0.765 +- 0.068  [6 ] TO SC BT BT BR SA
    0033          4cc666d       147       143      4              0.06         1.028 +- 0.085        0.973 +- 0.081  [7 ] TO SC SC SC BT BT AB
    0034         7cccc66d       135       137     -2              0.01         0.985 +- 0.085        1.015 +- 0.087  [8 ] TO SC SC BT BT BT BT SD
    0035         8cc6666d       136       131      5              0.09         1.038 +- 0.089        0.963 +- 0.084  [8 ] TO SC SC SC SC BT BT SA
    0036           8ccc5d       248         0    248            248.00         0.000 +- 0.000        0.000 +- 0.000  [6 ] TO RE BT BT BT SA
    0037           4cbc6d        96       124    -28              3.56         0.774 +- 0.079        1.292 +- 0.116  [6 ] TO SC BT BR BT AB
    0038          7ccc65d       215         0    215            215.00         0.000 +- 0.000        0.000 +- 0.000  [7 ] TO RE SC BT BT BT SD
    .                              80000     80000      5442.81/126 = 43.20  (pval:0.000 prob:1.000)  

    In [17]:                              
                              80000     80000      5442.81/126 = 43.20  (pval:0.000 prob:1.000)  

    In [2]: b.sel = "TO RE .."                                                                                                                                                                              
    [{_init_selection     :evt.py    :1312} WARNING  - _init_selection EMPTY nsel 0 len(psel) 80000 




    In [1]: a.sel = "TO RE .."   

    In [4]: a.his[:20]                                                                                                                                                                                      
    Out[4]: 
    seqhis_ana
    .                     cfo:-  1:g4live:tds3ip 
    .                               3628         1.00 
       n             iseq         frac           a    a-b      [ns] label
    0000           7ccc5d        0.164         594        [6 ] TO RE BT BT BT SD
    0001              45d        0.123         446        [3 ] TO RE AB
    0002           8ccc5d        0.068         248        [6 ] TO RE BT BT BT SA
    0003          7ccc65d        0.059         215        [7 ] TO RE SC BT BT BT SD
    0004            8cc5d        0.044         160        [5 ] TO RE BT BT SA
    0005          7ccc55d        0.041         147        [7 ] TO RE RE BT BT BT SD
    0006             455d        0.034         122        [4 ] TO RE RE AB
    0007             465d        0.029         105        [4 ] TO RE SC AB
    0008         7ccc665d        0.021          78        [8 ] TO RE SC SC BT BT BT SD
    0009          8ccc65d        0.020          73        [7 ] TO RE SC BT BT BT SA
    0010          8ccc55d        0.019          69        [7 ] TO RE RE BT BT BT SA
    0011         7ccc655d        0.017          62        [8 ] TO RE RE SC BT BT BT SD
    0012            4cc5d        0.017          61        [5 ] TO RE BT BT AB
    0013           8cc65d        0.015          53        [6 ] TO RE SC BT BT SA
    0014           8cc55d        0.012          43        [6 ] TO RE RE BT BT SA
    0015            4665d        0.010          35        [5 ] TO RE SC SC AB
    0016         7ccc555d        0.010          35        [8 ] TO RE RE RE BT BT BT SD
    0017        7ccc6665d        0.009          31        [9 ] TO RE SC SC SC BT BT BT SD
    0018             4c5d        0.008          29        [4 ] TO RE BT AB
    .                               3628         1.00 

    In [5]:                                              



Check the wavelength to energy for input photons
---------------------------------------------------

::

    2021-06-26 19:14:16.389 INFO  [6369] [GtOpticksTool::add_optical_photon@101]  idx 0 wavelength_nm   440 wavelength      0.000 energy      0.000 energy/eV      2.818
    2021-06-26 19:14:16.389 INFO  [6369] [GtOpticksTool::add_optical_photon@87]  m_pho  i       1 mski      -1 post (  10218.522-10218.522-10218.522       0.200) dirw (     -0.577     0.577     0.577       1.000) polw (     -0.707     0.000    -0.707     440.000) flgs       0       0       0       0



::

    082 #ifdef WITH_G4OPTICKS
     83 void GtOpticksTool::add_optical_photon(HepMC::GenEvent& event, unsigned idx, bool dump )
     84 {
     85     assert(m_pho);
     86 
     87     LOG(info)
     88         << " m_pho " << m_pho->desc(idx)
     89         ;
     90 
     91     glm::vec4 post = m_pho->getPositionTime(idx) ;
     92     glm::vec4 dirw = m_pho->getDirectionWeight(idx) ;
     93     glm::vec4 polw = m_pho->getPolarizationWavelength(idx) ;
     94 
     95     HepMC::GenVertex* vertex = new HepMC::GenVertex(HepMC::FourVector(post.x,post.y,post.z,post.w));
     96 
     97     G4double wavelength_nm = polw.w  ;
     98     G4double wavelength = wavelength_nm * nm ;
     99     G4double energy = h_Planck*c_light/wavelength ;
    100 
    101     LOG(info)
    102         << " idx " << idx
    103         << " wavelength_nm " << std::setw(5) << wavelength_nm
    104         << " wavelength " << std::fixed << std::setw(10) << std::setprecision(3) << wavelength
    105         << " energy " << std::fixed << std::setw(10) << std::setprecision(3) << energy
    106         << " energy/eV " << std::fixed << std::setw(10) << std::setprecision(3) << energy/eV
    107         ;
    108 
    109 
    110     HepMC::FourVector momentum(dirw.x*energy, dirw.y*energy, dirw.z*energy, energy );
    111 
    112     int pdg_id = 20022 ; // optical photon 
    113     int status = 1 ;  // ?
    114     HepMC::GenParticle* particle = new HepMC::GenParticle(momentum, pdg_id, status);
    115 
    116     HepMC::ThreeVector vec3in(polw.x, polw.y, polw.z);
    117     HepMC::Polarization polarization(vec3in);
    118 
    119     particle->set_polarization(polarization);
    120     vertex->add_particle_out(particle);
    121 
    122     event.add_vertex(vertex);
    123 }




BP=DsG4Scintillation::PostStepDoIt tds3ip
-------------------------------------------

Debugging shows the reemission is happening but the 
unusual situation of input photons is not properly handled
with regard to photon identity assignment and passing that identity between 
reemission generations. 
This messes up reemission bookkeeping. 


::

    BP=DsG4Scintillation::PostStepDoIt tds3ip


    234         if(doBothProcess) {
    235             flagReemission= doReemission
    236                 && aTrack.GetTrackStatus() == fStopAndKill
    237                 && aStep.GetPostStepPoint()->GetStepStatus() != fGeomBoundary;
    238         }


    (gdb) c
    Continuing.

    Breakpoint 10, DsG4Scintillation::PostStepDoIt (this=0x14d688e10, aTrack=..., aStep=...) at ../src/DsG4Scintillation.cc:258
    258	    if (TotalEnergyDeposit <= 0.0 && !flagReemission) {
    (gdb) p TotalEnergyDeposit
    $12 = 2.8178223935614284e-06
    (gdb) p TotalEnergyDeposit/eV
    $13 = 2.8178223935614284
    (gdb) p h_Planck
    $14 = 4.1356673336325146e-12
    (gdb) p h_Planck*c_light/TotalEnergyDeposit
    $15 = 0.00044000000789722201
    (gdb) p h_Planck*c_light/TotalEnergyDeposit/nm
    $16 = 440.00000789722191
    (gdb) 

    (gdb) c
    Continuing.

    Breakpoint 12, DsG4Scintillation::PostStepDoIt (this=0x14d688e10, aTrack=..., aStep=...) at ../src/DsG4Scintillation.cc:286
    286	    if (verboseLevel > 0 ) {
    (gdb) p Fast_Intensity
    $20 = (const G4MaterialPropertyVector *) 0x325b920
    (gdb) p Slow_Intensity
    $21 = (const G4MaterialPropertyVector *) 0x325b840
    (gdb) p Reemission_Prob
    $22 = (const G4MaterialPropertyVector *) 0x325dea0
    (gdb) 

    (gdb) c
    Continuing.

    Breakpoint 14, DsG4Scintillation::PostStepDoIt (this=0x14d688e10, aTrack=..., aStep=...) at ../src/DsG4Scintillation.cc:333
    333	            Reemission_Prob->Value(aTrack.GetKineticEnergy());
    (gdb) 

    (gdb) p aTrack.GetKineticEnergy()
    $26 = 2.8178223935614284e-06
    (gdb) p h_Planck*c_light/aTrack.GetKineticEnergy()
    $27 = 0.00044000000789722201
    (gdb) p h_Planck*c_light/aTrack.GetKineticEnergy()/nm
    $28 = 440.00000789722191
    (gdb) 

    (gdb) p p_reemission
    $29 = 0.22236851746465133


    (gdb) c
    Continuing.

    Breakpoint 17, DsG4Scintillation::PostStepDoIt (this=0x14d688e10, aTrack=..., aStep=...) at ../src/DsG4Scintillation.cc:478
    478	        (G4PhysicsOrderedFreeVector*)((*theReemissionIntegralTable)(materialIndex));
    (gdb) p materialIndex
    $30 = 1
    (gdb) 

    (gdb) c

    Breakpoint 23, DsG4Scintillation::PostStepDoIt (this=0x14d688e10, aTrack=..., aStep=...) at ../src/DsG4Scintillation.cc:599
    599	                if (verboseLevel>1) {
    (gdb) p sampledEnergy
    $40 = 2.8915839853784507e-06
    (gdb) p h_Planck*c_light/sampledEnergy/nm
    $41 = 428.77602092464434
    (gdb) 



    (gdb) p ancestor
    $44 = {static MISSING = 4294967295, gs = 53846304, ix = 0, id = 53846992, gn = 0}
    (gdb) b 528
    Breakpoint 28 at 0x7fffd09bba8f: file ../src/DsG4Scintillation.cc, line 528.
    (gdb) p ancestor_id
    $45 = 1



After creating a breakpoint, use "commands"::

    (gdb) b 512
    Breakpoint 2 at 0x7fffd09bba7a: file ../src/DsG4Scintillation.cc, line 512.
    (gdb) commands
    Type commands for breakpoint(s) 2, one per line.
    End with a line saying just "end".
    >silent
    >print ancestor
    >cont
    >end
    (gdb) 


::
    gdb) c
    Continuing.
    2021-06-26 21:00:09.653 INFO  [164455] [PMTEfficiencyCheck::addHitRecord@88]  m_eventID 0 m_record_count 0
    $1 = {static MISSING = 4294967295, gs = 4294967295, ix = 4294967295, id = 4294967295, gn = 4294967295}
    $2 = {static MISSING = 4294967295, gs = 4294967295, ix = 4294967295, id = 4294967295, gn = 4294967295}
    $3 = {static MISSING = 4294967295, gs = 0, ix = 0, id = 0, gn = 0}
    $4 = {static MISSING = 4294967295, gs = 4294967295, ix = 4294967295, id = 4294967295, gn = 4294967295}
    $5 = {static MISSING = 4294967295, gs = 4294967295, ix = 4294967295, id = 4294967295, gn = 4294967295}
    $6 = {static MISSING = 4294967295, gs = 0, ix = 0, id = 0, gn = 0}
    $7 = {static MISSING = 4294967295, gs = 4294967295, ix = 4294967295, id = 4294967295, gn = 4294967295}
    $8 = {static MISSING = 4294967295, gs = 4294967295, ix = 4294967295, id = 4294967295, gn = 4294967295}
    $9 = {static MISSING = 4294967295, gs = 4294967295, ix = 4294967295, id = 4294967295, gn = 4294967295}
    $10 = {static MISSING = 4294967295, gs = 4294967295, ix = 4294967295, id = 4294967295, gn = 4294967295}
    $11 = {static MISSING = 4294967295, gs = 4294967295, ix = 4294967295, id = 4294967295, gn = 4294967295}
    $12 = {static MISSING = 4294967295, gs = 0, ix = 0, id = 0, gn = 0}
    $13 = {static MISSING = 4294967295, gs = 4294967295, ix = 4294967295, id = 4294967295, gn = 4294967295}
    $14 = {static MISSING = 4294967295, gs = 0, ix = 0, id = 0, gn = 0}
    $15 = {static MISSING = 4294967295, gs = 4294967295, ix = 4294967295, id = 4294967295, gn = 4294967295}
    $16 = {static MISSING = 4294967295, gs = 0, ix = 0, id = 0, gn = 0}
    $17 = {static MISSING = 4294967295, gs = 4294967295, ix = 4294967295, id = 4294967295, gn = 4294967295}
    $18 = {static MISSING = 4294967295, gs = 4294967295, ix = 4294967295, id = 4294967295, gn = 4294967295}
    $19 = {static MISSING = 4294967295, gs = 0, ix = 0, id = 0, gn = 0}
    $20 = {static MISSING = 4294967295, gs = 0, ix = 0, id = 0, gn = 1}
    $21 = {static MISSING = 4294967295, gs = 4294967295, ix = 4294967295, id = 4294967295, gn = 4294967295}
    $22 = {static MISSING = 4294967295, gs = 0, ix = 0, id = 0, gn = 0}
    $23 = {static MISSING = 4294967295, gs = 4294967295, ix = 4294967295, id = 4294967295, gn = 4294967295}
    $24 = {static MISSING = 4294967295, gs = 4294967295, ix = 4294967295, id = 4294967295, gn = 4294967295}
    $25 = {static MISSING = 4294967295, gs = 0, ix = 0, id = 0, gn = 0}

    $410 = {static MISSING = 4294967295, gs = 4294967295, ix = 4294967295, id = 4294967295, gn = 4294967295}
    $411 = {static MISSING = 4294967295, gs = 0, ix = 0, id = 0, gn = 0}
    $412 = {static MISSING = 4294967295, gs = 0, ix = 0, id = 0, gn = 1}
    $413 = {static MISSING = 4294967295, gs = 0, ix = 0, id = 0, gn = 2}
    $414 = {static MISSING = 4294967295, gs = 4294967295, ix = 4294967295, id = 4294967295, gn = 4294967295}
    $415 = {static MISSING = 4294967295, gs = 4294967295, ix = 4294967295, id = 4294967295, gn = 4294967295}



Flip the switch **when_unlabelled_fabricate_trackid_photon = true** 
----------------------------------------------------------------------

* gets reemission bookkeeping to work with input photons

::

    .#ifdef WITH_G4OPTICKS
    -    CPho ancestor = CPhotonInfo::Get(&aTrack, false); 
    +    // fabrication only needed with input photons, but should have no impact 
    +    // with ordinary running as optical tracks should always be labelled
    +    // other than with input photons
    +    bool when_unlabelled_fabricate_trackid_photon = true ; 
    +    CPho ancestor = CPhotonInfo::Get(&aTrack, when_unlabelled_fabricate_trackid_photon ); 
         int ancestor_id = ancestor.get_id() ; 
         /**



::

    tds3ip.sh get
    tds3ip.sh 1

    In [2]: ab.his[:30]                                                                                                                                                                                     
    Out[2]: 
    ab.his
    .       seqhis_ana  cfo:sum  1:g4live:tds3ip   -1:g4live:tds3ip        c2        ab        ba 
    .                                800       800        18.56/10 =  1.86  (pval:0.046 prob:0.954)  
       n             iseq         a         b    a-b       (a-b)^2/(a+b)         a/b                   b/a           [ns] label
    0000               4d       122       123     -1              0.00         0.992 +- 0.090        1.008 +- 0.091  [2 ] TO AB
    0001            7cccd       101       130    -29              3.64         0.777 +- 0.077        1.287 +- 0.113  [5 ] TO BT BT BT SD
    0002           7ccc6d       122       103     19              1.60         1.184 +- 0.107        0.844 +- 0.083  [6 ] TO SC BT BT BT SD
    0003          7ccc66d        50        49      1              0.01         1.020 +- 0.144        0.980 +- 0.140  [7 ] TO SC SC BT BT BT SD
    0004              46d        47        52     -5              0.25         0.904 +- 0.132        1.106 +- 0.153  [3 ] TO SC AB
    0005           8ccc6d        36        60    -24              6.00         0.600 +- 0.100        1.667 +- 0.215  [6 ] TO SC BT BT BT SA
    0006            8cccd        45        41      4              0.19         1.098 +- 0.164        0.911 +- 0.142  [5 ] TO BT BT BT SA
    0007            8cc6d        29        39    -10              1.47         0.744 +- 0.138        1.345 +- 0.215  [5 ] TO SC BT BT SA
    0008          8ccc66d        20        19      1              0.03         1.053 +- 0.235        0.950 +- 0.218  [7 ] TO SC SC BT BT BT SA
    0009         7ccc666d        25        13     12              3.79         1.923 +- 0.385        0.520 +- 0.144  [8 ] TO SC SC SC BT BT BT SD
    0010             466d        19        12      7              1.58         1.583 +- 0.363        0.632 +- 0.182  [4 ] TO SC SC AB
    0011            4cc6d        15         5     10              0.00         3.000 +- 0.775        0.333 +- 0.149  [5 ] TO SC BT BT AB
    0012           8cc66d         7        10     -3              0.00         0.700 +- 0.265        1.429 +- 0.452  [6 ] TO SC SC BT BT SA
    0013           7ccc5d         8         5      3              0.00         1.600 +- 0.566        0.625 +- 0.280  [6 ] TO RE BT BT BT SD
    0014        7ccc6666d         4         7     -3              0.00         0.571 +- 0.286        1.750 +- 0.661  [9 ] TO SC SC SC SC BT BT BT SD
    0015             4ccd         4         7     -3              0.00         0.571 +- 0.286        1.750 +- 0.661  [4 ] TO BT BT AB
    0016             4c6d         6         4      2              0.00         1.500 +- 0.612        0.667 +- 0.333  [4 ] TO SC BT AB



tds3gun what wavelength to use for representative tds3ip 
----------------------------------------------------------------

::

    tds3gun.sh get
    tds3gun.sh 1

    In [1]: a.sel = ["SI AB", "SI BT BT BT SD", "SI BT BT BT SD", "SI BT BT BT SA", "SI BT BT SA" ]     ## select the most prolific, excluding RE


    In [2]: a.seqhis_ana.table                                                                                                                                                                             
    Out[2]: 
    seqhis_ana
    .                     cfo:-  1:g4live:tds3gun 
    .                               4510         1.00 
       n             iseq         frac           a    a-b      [ns] label
    0000               42        0.386        1741        [2 ] SI AB
    0001            7ccc2        0.328        1480        [5 ] SI BT BT BT SD
    0002            8ccc2        0.146         660        [5 ] SI BT BT BT SA
    0003             8cc2        0.139         629        [4 ] SI BT BT SA
       n             iseq         frac           a    a-b      [ns] label
    .                               4510         1.00 

    In [3]: a.ox.shape                                                                                                                                                                                     
    Out[3]: (4510, 4, 4)

    In [4]: a.ox[0]                                                                                                                                                                                        
    Out[4]: 
    A([[-17866.793 ,   7413.6465,    244.0195,    104.5536],
       [    -0.8806,      0.4732,      0.0257,      1.    ],
       [     0.4739,      0.8788,      0.0557,    425.9893],
       [        nan,      0.    ,      0.    ,      0.    ]], dtype=float32)

    In [8]: wl = a.ox[:,2,3]                                                                                          

    In [9]: wh = np.histogram(wl, bins=10)                                                                                                                                                                 
    In [12]: for i in range(len(wh[0])): print(" %10.3f  %5d " % ( wh[1][i], wh[0][i]))                                                                                                                    
        180.000      8 
        240.390     11 
        300.780     22 
        361.170   1481 
        421.561   2567 
        481.951    329 
        542.341     42 
        602.731     25 
        663.121     17 
        723.511      8 

    In [15]: wh = np.histogram(wl, bins=50)                                                                                                                                                                

    In [16]: for i in range(len(wh[0])): print(" %10.3f  %5d " % ( wh[1][i], wh[0][i]))                                                                                                                    
        180.000      2 
        192.078      1 
        204.156      0 
        216.234      2 
        228.312      3 
        240.390      0 
        252.468      7 
        264.546      2 
        276.624      1 
        288.702      1 
        300.780      3 
        312.858      4 
        324.936      5 
        337.014      4 
        349.092      6 
        361.170     12 
        373.249    114 
        385.327    206 
        397.405    405 
        409.483    744 
        421.561    790 
        433.639    680 
        445.717    508 
        457.795    351 
        469.873    238 
        481.951    120 
        494.029     99 
        506.107     59 
        518.185     31 
        530.263     20 
        542.341     13 
        554.419     14 
        566.497      6 
        578.575      4 
        590.653      5 
        602.731      4 
        614.809      3 
        626.887      5 
        638.965      7 
        651.043      6 
        663.121      5 
        675.199      5 
        687.277      1 
        699.355      2 
        711.433      4 
        723.511      3 
        735.589      1 
        747.667      0 
        759.745      0 
        771.824      4 


    In [20]: run ls.py                                                                                                                                                                                     
    [{__init__            :proplib.py:150} INFO     - names : None 
    [{__init__            :proplib.py:160} INFO     - npath : /usr/local/opticks/geocache/OKX4Test_lWorld0x32a96e0_PV_g4live/g4ok_gltf/a3cbac8189a032341f76682cdb4f47b6/1/GItemList/GMaterialLib.txt 
    [{__init__            :proplib.py:167} INFO     - names : ['LS', 'Steel', 'Tyvek', 'Air', 'Scintillator', 'TiO2Coating', 'Adhesive', 'Aluminium', 'Rock', 'LatticedShellSteel', 'Acrylic', 'PE_PA', 'Vacuum', 'Pyrex', 'Water', 'vetoWater', 'Galactic'] 
        wavelen      rindex      abslen     scatlen    reemprob    groupvel 
         60.000       1.454       0.003     546.429       0.400     206.241 
         80.000       1.454       0.003     546.429       0.400     206.241 
        100.000       1.454       0.003     546.429       0.400     206.241 
        120.000       1.454       0.003     546.429       0.400     192.299 
        140.000       1.664       0.003     546.429       0.400     173.446 
        160.000       1.793       0.003     546.429       0.400     118.988 
        180.000       1.527       0.003     546.429       0.410     139.949 
        200.000       1.618       0.003     547.535       0.420     177.249 
        220.000       1.600       0.198    1415.292       0.477     166.321 
        240.000       1.582       0.392    2283.049       0.538     166.320 
        260.000       1.563       0.586    3150.806       0.599     166.319 
        280.000       1.545       0.781    4018.563       0.660     166.319 
        300.000       1.526       0.975    4887.551       0.721     177.207 
        320.000       1.521       1.169    7505.381       0.782     186.734 
        340.000       1.516       1.364   10123.211       0.800     186.733 
        360.000       1.511       5.664   12741.041       0.800     186.733 
        380.000       1.505      12.239   15358.871       0.801     186.733     
        400.000       1.500     195.518   17976.701       0.800     189.766   ##  absorption very sensitive to wavelength in this range   
        420.000       1.497   40892.633   23161.414       0.497     193.682     
        440.000       1.495   84240.547   29164.996       0.222     195.357     
        460.000       1.494   78284.352   33453.633       0.169     195.915 
        480.000       1.492   92540.648   37742.270       0.135     195.684 
        500.000       1.490  114196.219   43987.516       0.123     195.369 
        520.000       1.488   88688.727   52136.293       0.106     195.275 
        540.000       1.487   91878.211   60285.070       0.089     196.430 
        560.000       1.485   93913.664   75733.656       0.072     198.024 
        580.000       1.485   67581.016   98222.445       0.057     198.572 
        600.000       1.484   46056.891  116999.734       0.048     198.683 
        620.000       1.483   44640.812  132183.031       0.040     198.732 
        640.000       1.482   15488.402  147366.312       0.031     198.733 
        660.000       1.481   20362.018  162549.594       0.023     198.733 
        680.000       1.480   20500.150  177732.875       0.014     199.247 
        700.000       1.480   13182.578  192957.234       0.005     200.349 
        720.000       1.479    7429.221  218677.828       0.000     200.931 
        740.000       1.479    5515.074  244398.406       0.000     200.931 
        760.000       1.479    2898.857  270119.000       0.000     200.931 
        780.000       1.478   10900.813  295839.562       0.000     200.936 
        800.000       1.478    9584.489  321429.000       0.000     201.905 
        820.000       1.478    5822.304  321429.000       0.000     202.823 

    In [21]:                                                                                        



tds3ip InwardsCubeCorners17699 repeat 10,000 using mono 440nm
------------------------------------------------------------------

* these input photons feature 35m of direct path length before hitting geometry 

* do not see any obvious reemission difference at this wavelength 
* TODO: look at wavelength distribution of tds3gun with discrepant RE, repeat tds3ip with various 
  wavelengths to try to reproduce the RE-discrepancy and make it worse in order to understand

::

    P[blyth@localhost cmt]$ t tds3ip
    tds3ip () 
    { 
        local name="InwardsCubeCorners17699";
        local path="$HOME/.opticks/InputPhotons/${name}.npy";
        local repeat=10000;
        export OPTICKS_EVENT_PFX=tds3ip;
        export INPUT_PHOTON_PATH=$path;
        export INPUT_PHOTON_REPEAT=$repeat;
        tds3
    }





    In [5]: ab.his[:100]                                                                                                                                                                                    
    Out[5]: 
    ab.his
    .       seqhis_ana  cfo:sum  1:g4live:tds3ip   -1:g4live:tds3ip        c2        ab        ba 
    .                              80000     80000      1285.99/150 =  8.57  (pval:0.000 prob:1.000)  
       n             iseq         a         b    a-b       (a-b)^2/(a+b)         a/b                   b/a           [ns] label
    0000               4d     12832     12872    -40              0.06         0.997 +- 0.009        1.003 +- 0.009  [2 ] TO AB
    0001           7ccc6d     12583     12324    259              2.69         1.021 +- 0.009        0.979 +- 0.009  [6 ] TO SC BT BT BT SD
    0002            7cccd     10712     10911   -199              1.83         0.982 +- 0.009        1.019 +- 0.010  [5 ] TO BT BT BT SD
    0003              46d      4609      4702    -93              0.93         0.980 +- 0.014        1.020 +- 0.015  [3 ] TO SC AB
    0004           8ccc6d      4512      4666   -154              2.58         0.967 +- 0.014        1.034 +- 0.015  [6 ] TO SC BT BT BT SA
    0005          7ccc66d      4527      4603    -76              0.63         0.983 +- 0.015        1.017 +- 0.015  [7 ] TO SC SC BT BT BT SD
    0006            8cccd      3814      3899    -85              0.94         0.978 +- 0.016        1.022 +- 0.016  [5 ] TO BT BT BT SA
    0007            8cc6d      3402      3365     37              0.20         1.011 +- 0.017        0.989 +- 0.017  [5 ] TO SC BT BT SA
    0008         7ccc666d      1717      1610    107              3.44         1.066 +- 0.026        0.938 +- 0.023  [8 ] TO SC SC SC BT BT BT SD
    0009          8ccc66d      1654      1665    -11              0.04         0.993 +- 0.024        1.007 +- 0.025  [7 ] TO SC SC BT BT BT SA
    0010             466d      1618      1581     37              0.43         1.023 +- 0.025        0.977 +- 0.025  [4 ] TO SC SC AB
    0011           8cc66d      1248      1156     92              3.52         1.080 +- 0.031        0.926 +- 0.027  [6 ] TO SC SC BT BT SA
    0012            4cc6d      1090      1044     46              0.99         1.044 +- 0.032        0.958 +- 0.030  [5 ] TO SC BT BT AB
    0013         8ccc666d       587       610    -23              0.44         0.962 +- 0.040        1.039 +- 0.042  [8 ] TO SC SC SC BT BT BT SA
    0014           7ccc5d       594       568     26              0.58         1.046 +- 0.043        0.956 +- 0.040  [6 ] TO RE BT BT BT SD
    0015            4666d       591       541     50              2.21         1.092 +- 0.045        0.915 +- 0.039  [5 ] TO SC SC SC AB
    0016             4ccd       532       528      4              0.02         1.008 +- 0.044        0.992 +- 0.043  [4 ] TO BT BT AB
    0017        7ccc6666d       517       541    -24              0.54         0.956 +- 0.042        1.046 +- 0.045  [9 ] TO SC SC SC SC BT BT BT SD
    0018              45d       446       495    -49              2.55         0.901 +- 0.043        1.110 +- 0.050  [3 ] TO RE AB
    0019          7cccc6d       433       409     24              0.68         1.059 +- 0.051        0.945 +- 0.047  [7 ] TO SC BT BT BT BT SD
    0020             4c6d       385       435    -50              3.05         0.885 +- 0.045        1.130 +- 0.054  [4 ] TO SC BT AB
    0021          8cc666d       397       388      9              0.10         1.023 +- 0.051        0.977 +- 0.050  [7 ] TO SC SC SC BT BT SA
    0022           4cc66d       378       402    -24              0.74         0.940 +- 0.048        1.063 +- 0.053  [6 ] TO SC SC BT BT AB
    0023       bccbccbc6d       283       276      7              0.09         1.025 +- 0.061        0.975 +- 0.059  [10] TO SC BT BR BT BT BR BT BT BR
    0024           8ccc5d       248       267    -19              0.70         0.929 +- 0.059        1.077 +- 0.066  [6 ] TO RE BT BT BT SA
    0025          8cccc6d       231       220     11              0.27         1.050 +- 0.069        0.952 +- 0.064  [7 ] TO SC BT BT BT BT SA
    0026              4cd       197       245    -48              5.21         0.804 +- 0.057        1.244 +- 0.079  [3 ] TO BT AB
    0027          7ccc56d       210       216     -6              0.08         0.972 +- 0.067        1.029 +- 0.070  [7 ] TO SC RE BT BT BT SD
    0028        8ccc6666d       200       223    -23              1.25         0.897 +- 0.063        1.115 +- 0.075  [9 ] TO SC SC SC SC BT BT BT SA
    0029          7ccc65d       215       202     13              0.41         1.064 +- 0.073        0.940 +- 0.066  [7 ] TO RE SC BT BT BT SD

    0030        8ccaccc6d         0       387   -387            387.00         0.000 +- 0.000        0.000 +- 0.000  [9 ] TO SC BT BT BT SR BT BT SA

    0031           46666d       197       190      7              0.13         1.037 +- 0.074        0.964 +- 0.070  [6 ] TO SC SC SC SC AB
    0032       7ccc66666d       200       182     18              0.85         1.099 +- 0.078        0.910 +- 0.067  [10] TO SC SC SC SC SC BT BT BT SD
    0033             456d       162       170     -8              0.19         0.953 +- 0.075        1.049 +- 0.080  [4 ] TO SC RE AB
    0034            8cc5d       160       159      1              0.00         1.006 +- 0.080        0.994 +- 0.079  [5 ] TO RE BT BT SA
    0035          7ccc55d       147       160    -13              0.55         0.919 +- 0.076        1.088 +- 0.086  [7 ] TO RE RE BT BT BT SD

    0036         8caccc6d       297         0    297            297.00         0.000 +- 0.000        0.000 +- 0.000  [8 ] TO SC BT BT BT SR BT SA

    0037           8bcc6d       166       127     39              5.19         1.307 +- 0.101        0.765 +- 0.068  [6 ] TO SC BT BT BR SA
    0038            4c66d       145       148     -3              0.03         0.980 +- 0.081        1.021 +- 0.084  [5 ] TO SC SC BT AB
    0039          4cc666d       147       143      4              0.06         1.028 +- 0.085        0.973 +- 0.081  [7 ] TO SC SC SC BT BT AB
    0040         7cccc66d       135       137     -2              0.01         0.985 +- 0.085        1.015 +- 0.087  [8 ] TO SC SC BT BT BT BT SD
    0041         8cc6666d       136       131      5              0.09         1.038 +- 0.089        0.963 +- 0.084  [8 ] TO SC SC SC SC BT BT SA
    0042             455d       122       141    -19              1.37         0.865 +- 0.078        1.156 +- 0.097  [4 ] TO RE RE AB
    0043           4ccc6d       135        77     58             15.87         1.753 +- 0.151        0.570 +- 0.065  [6 ] TO SC BT BT BT AB
    0044           4cbc6d        96       110    -14              0.95         0.873 +- 0.089        1.146 +- 0.109  [6 ] TO SC BT BR BT AB
    0045       ccc666666d       105       101      4              0.08         1.040 +- 0.101        0.962 +- 0.096  [10] TO SC SC SC SC SC SC BT BT BT
    0046       ccbccbc66d       107        98      9              0.40         1.092 +- 0.106        0.916 +- 0.093  [10] TO SC SC BT BR BT BT BR BT BT
    0047             465d       105        91     14              1.00         1.154 +- 0.113        0.867 +- 0.091  [4 ] TO RE SC AB
    0048       c6cbccbc6d        94        94      0              0.00         1.000 +- 0.103        1.000 +- 0.103  [10] TO SC BT BR BT BT BR BT SC BT
    0049       7ccc6cbc6d        83       101    -18              1.76         0.822 +- 0.090        1.217 +- 0.121  [10] TO SC BT BR BT SC BT BT BT SD
    0050         8cccc66d        80        80      0              0.00         1.000 +- 0.112        1.000 +- 0.112  [8 ] TO SC SC BT BT BT BT SA
    0051          8ccc56d        77        82     -5              0.16         0.939 +- 0.107        1.065 +- 0.118  [7 ] TO SC RE BT BT BT SA

    0052       8ccaccc66d         0       157   -157            157.00         0.000 +- 0.000        0.000 +- 0.000  [10] TO SC SC BT BT BT SR BT BT SA

    0053         7ccc566d        88        65     23              3.46         1.354 +- 0.144        0.739 +- 0.092  [8 ] TO SC SC RE BT BT BT SD
    0054       8ccc66666d        79        74      5              0.16         1.068 +- 0.120        0.937 +- 0.109  [10] TO SC SC SC SC SC BT BT BT SA
    0055       cbccbc666d        77        72      5              0.17         1.069 +- 0.122        0.935 +- 0.110  [10] TO SC SC SC BT BR BT BT BR BT
    0056         7ccc665d        78        67     11              0.83         1.164 +- 0.132        0.859 +- 0.105  [8 ] TO RE SC SC BT BT BT SD
    0057          8ccc55d        69        72     -3              0.06         0.958 +- 0.115        1.043 +- 0.123  [7 ] TO RE RE BT BT BT SA
    0058          7cbcc6d        62        78    -16              1.83         0.795 +- 0.101        1.258 +- 0.142  [7 ] TO SC BT BT BR BT SD
    0059          8ccc65d        73        62     11              0.90         1.177 +- 0.138        0.849 +- 0.108  [7 ] TO RE SC BT BT BT SA
    0060          466666d        69        55     14              1.58         1.255 +- 0.151        0.797 +- 0.107  [7 ] TO SC SC SC SC SC AB
    0061          7c6cc6d        56        66    -10              0.82         0.848 +- 0.113        1.179 +- 0.145  [7 ] TO SC BT BT SC BT SD
    0062         7ccc656d        59        63     -4              0.13         0.937 +- 0.122        1.068 +- 0.135  [8 ] TO SC RE SC BT BT BT SD
    0063         7ccc556d        59        62     -3              0.07         0.952 +- 0.124        1.051 +- 0.133  [8 ] TO SC RE RE BT BT BT SD
    0064            4566d        62        57      5              0.21         1.088 +- 0.138        0.919 +- 0.122  [5 ] TO SC SC RE AB
    0065            4cc5d        61        57      4              0.14         1.070 +- 0.137        0.934 +- 0.124  [5 ] TO RE BT BT AB
    0066         7ccc655d        62        55      7              0.42         1.127 +- 0.143        0.887 +- 0.120  [8 ] TO RE RE SC BT BT BT SD
    0067         4cc6666d        53        62     -9              0.70         0.855 +- 0.117        1.170 +- 0.149  [8 ] TO SC SC SC SC BT BT AB
    0068       ccc6cbc66d        65        47     18              2.89         1.383 +- 0.172        0.723 +- 0.105  [10] TO SC SC BT BR BT SC BT BT BT

    0069        8caccc66d       108         1    107            105.04       108.000 +- 10.392        0.009 +- 0.009  [9 ] TO SC SC BT BT BT SR BT SA

    0070           4c666d        57        51      6              0.33         1.118 +- 0.148        0.895 +- 0.125  [6 ] TO SC SC SC BT AB
    0071           8cc56d        60        47     13              1.58         1.277 +- 0.165        0.783 +- 0.114  [6 ] TO SC RE BT BT SA
    0072           8cc65d        53        52      1              0.01         1.019 +- 0.140        0.981 +- 0.136  [6 ] TO RE SC BT BT SA
    0073            4cccd        56        45     11              1.20         1.244 +- 0.166        0.804 +- 0.120  [5 ] TO BT BT BT AB
    0074           8cc55d        43        53    -10              1.04         0.811 +- 0.124        1.233 +- 0.169  [6 ] TO RE RE BT BT SA
    0075        7cccc666d        50        44      6              0.38         1.136 +- 0.161        0.880 +- 0.133  [9 ] TO SC SC SC BT BT BT BT SD
    0076          8bcc66d        59        32     27              8.01         1.844 +- 0.240        0.542 +- 0.096  [7 ] TO SC SC BT BT BR SA
    0077        4cbccbc6d        43        46     -3              0.10         0.935 +- 0.143        1.070 +- 0.158  [9 ] TO SC BT BR BT BT BR BT AB
    0078       6cbccbc66d        46        43      3              0.10         1.070 +- 0.158        0.935 +- 0.143  [10] TO SC SC BT BR BT BT BR BT SC
    0079            4556d        42        46     -4              0.18         0.913 +- 0.141        1.095 +- 0.161  [5 ] TO SC RE RE AB
    0080       8ccc6cbc6d        45        43      2              0.05         1.047 +- 0.156        0.956 +- 0.146  [10] TO SC BT BR BT SC BT BT BT SA
    0081        8cc66666d        50        37     13              1.94         1.351 +- 0.191        0.740 +- 0.122  [9 ] TO SC SC SC SC SC BT BT SA
    0082       ccc66cbc6d        45        40      5              0.29         1.125 +- 0.168        0.889 +- 0.141  [10] TO SC BT BR BT SC SC BT BT BT
    0083            4bc6d        35        44     -9              1.03         0.795 +- 0.134        1.257 +- 0.190  [5 ] TO SC BT BR AB
    0084        7ccccbc6d        34        44    -10              1.28         0.773 +- 0.133        1.294 +- 0.195  [9 ] TO SC BT BR BT BT BT BT SD
    0085         7ccc555d        35        41     -6              0.47         0.854 +- 0.144        1.171 +- 0.183  [8 ] TO RE RE RE BT BT BT SD
    0086            4665d        35        41     -6              0.47         0.854 +- 0.144        1.171 +- 0.183  [5 ] TO RE SC SC AB
    0087          4ccc66d        48        22     26              9.66         2.182 +- 0.315        0.458 +- 0.098  [7 ] TO SC SC BT BT BT AB
    0088          4cbc66d        36        33      3              0.13         1.091 +- 0.182        0.917 +- 0.160  [7 ] TO SC SC BT BR BT AB
    0089       cc6666666d        30        38     -8              0.94         0.789 +- 0.144        1.267 +- 0.205  [10] TO SC SC SC SC SC SC SC BT BT
    0090         8ccc566d        38        28     10              1.52         1.357 +- 0.220        0.737 +- 0.139  [8 ] TO SC SC RE BT BT BT SA
    0091            4555d        22        43    -21              6.78         0.512 +- 0.109        1.955 +- 0.298  [5 ] TO RE RE RE AB
    0092           7c6ccd        32        33     -1              0.02         0.970 +- 0.171        1.031 +- 0.180  [6 ] TO BT BT SC BT SD
    0093       bccbc6666d        36        28      8              1.00         1.286 +- 0.214        0.778 +- 0.147  [10] TO SC SC SC SC BT BR BT BT BR

    0094       ccaccc666d         2        62    -60             56.25         0.032 +- 0.023       31.000 +- 3.937  [10] TO SC SC SC BT BT BT SR BT BT

    0095         8ccc656d        33        30      3              0.14         1.100 +- 0.191        0.909 +- 0.166  [8 ] TO SC RE SC BT BT BT SA
    0096       66cbccbc6d        30        33     -3              0.14         0.909 +- 0.166        1.100 +- 0.191  [10] TO SC BT BR BT BT BR BT SC SC
    0097            4656d        34        26      8              1.07         1.308 +- 0.224        0.765 +- 0.150  [5 ] TO SC RE SC AB
    0098       cc6cbc666d        27        31     -4              0.28         0.871 +- 0.168        1.148 +- 0.206  [10] TO SC SC SC BT BR BT SC BT BT
    .                              80000     80000      1285.99/150 =  8.57  (pval:0.000 prob:1.000)  



    In [6]: a.sel = "*SR*"      

    In [13]: a.his[:10]                                                                                                                                                                                     
    Out[13]: 
    seqhis_ana
    .                     cfo:-  1:g4live:tds3ip 
    .                                812         1.00 
       n             iseq         frac           a    a-b      [ns] label
    0000         8caccc6d        0.366         297        [8 ] TO SC BT BT BT SR BT SA
    0001        8caccc66d        0.133         108        [9 ] TO SC SC BT BT BT SR BT SA
    0002       8caccc666d        0.057          46        [10] TO SC SC SC BT BT BT SR BT SA
    0003        8cacbcc6d        0.033          27        [9 ] TO SC BT BT BR BT SR BT SA
    0004       caccc6666d        0.027          22        [10] TO SC SC SC SC BT BT BT SR BT
    0005       8cabaccc6d        0.026          21        [10] TO SC BT BT BT SR BR SR BT SA
    0006         8caccc5d        0.018          15        [8 ] TO RE BT BT BT SR BT SA
    0007       8cacbcc66d        0.018          15        [10] TO SC SC BT BT BR BT SR BT SA
    0008       8caccccc6d        0.016          13        [10] TO SC BT BT BT BT BT SR BT SA
    .                                812         1.00 

    In [16]: a.seqmat_ana.table[:10]                                                                                                                                                                        
    Out[16]: 
    seqmat_ana
    .                     cfo:-  1:g4live:tds3ip 
    .                                812         1.00 
       n             iseq         frac           a    a-b      [ns] label
    0000         3eddeb11        0.385         313        [8 ] LS LS Ac Py Va Va Py Ty
    0001        3eddeb111        0.155         126        [9 ] LS LS LS Ac Py Va Va Py Ty
    0002       3eddeb1111        0.068          55        [10] LS LS LS LS Ac Py Va Va Py Ty
    0003        3eddeeb11        0.044          36        [9 ] LS LS Ac Py Py Va Va Py Ty
    0004       eddeb11111        0.043          35        [10] LS LS LS LS LS Ac Py Va Va Py
    0005       3eddddeb11        0.027          22        [10] LS LS Ac Py Va Va Va Va Py Ty
    0006       3eddeeb111        0.023          19        [10] LS LS LS Ac Py Py Va Va Py Ty
    0007       eddeddeb11        0.018          15        [10] LS LS Ac Py Va Va Py Va Va Py
    0008       eddddeb111        0.017          14        [10] LS LS LS Ac Py Va Va Va Va Py
    .                                812         1.00 




    In [11]: b.sel = "*SR*"     

    In [14]: b.his[:10]                                                                                                                                                                                     
    Out[14]: 
    seqhis_ana
    .                     cfo:-  -1:g4live:tds3ip 
    .                                932         1.00 
       n             iseq         frac           a    a-b      [ns] label
    0000        8ccaccc6d        0.415         387        [9 ] TO SC BT BT BT SR BT BT SA
    0001       8ccaccc66d        0.168         157        [10] TO SC SC BT BT BT SR BT BT SA
    0002       ccaccc666d        0.067          62        [10] TO SC SC SC BT BT BT SR BT BT
    0003       cabcaccc6d        0.050          47        [10] TO SC BT BT BT SR BT BR SR BT
    0004       8ccacbcc6d        0.039          36        [10] TO SC BT BT BR BT SR BT BT SA
    0005       caccc6666d        0.025          23        [10] TO SC SC SC SC BT BT BT SR BT
    0006        8ccaccc5d        0.023          21        [9 ] TO RE BT BT BT SR BT BT SA
    0007       acccaccc6d        0.020          19        [10] TO SC BT BT BT SR BT BT BT SR
    0008       abcaccc66d        0.019          18        [10] TO SC SC BT BT BT SR BT BR SR
    .                                932         1.00 

    In [17]: b.seqmat_ana.table[:10]                                                                                                                                                                        
    Out[17]: 
    seqmat_ana
    .                     cfo:-  -1:g4live:tds3ip 
    .                                932         1.00 
       n             iseq         frac           a    a-b      [ns] label
    0000        3edddeb11        0.438         408        [9 ] LS LS Ac Py Va Va Va Py Ty
    0001       3edddeb111        0.201         187        [10] LS LS LS Ac Py Va Va Va Py Ty
    0002       edddeb1111        0.080          75        [10] LS LS LS LS Ac Py Va Va Va Py
    0003       ddddddeb11        0.052          48        [10] LS LS Ac Py Va Va Va Va Va Va
    0004       3edddeeb11        0.043          40        [10] LS LS Ac Py Py Va Va Va Py Ty
    0005       dddeb11111        0.041          38        [10] LS LS LS LS LS Ac Py Va Va Va
    0006       ddedddeb11        0.023          21        [10] LS LS Ac Py Va Va Va Py Va Va
    0007       dddddeb111        0.021          20        [10] LS LS LS Ac Py Va Va Va Va Va
    0008       edddeeb111        0.015          14        [10] LS LS LS Ac Py Py Va Va Va Py
    .                                932         1.00 


* following SR:specular reflection see 2BT with G4 and 1BT with OK
* probably specific geometry issue : suspect XJfixtureConstruction

::

    In [1]: a.sel = "TO SC BT BT BT SR BT SA"                                                                                                                                                               

    In [2]: a.bn.shape                                                                                                                                                                                      
    Out[2]: (297, 1, 4)


    In [5]: np.set_printoptions(edgeitems=16)                                                                                                                                                               

    In [6]: a.bn.view(np.int8).reshape(-1,16)                                                                                                                                                               
    Out[6]: 
    A([[ 18,  18,  17, -19, -22,  19,  16,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  18,  17, -19, -22,  19,  16,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  18,  17, -19, -22,  19,  16,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  18,  17, -19, -22,  19,  16,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  18,  17, -19, -24,  19,  16,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  18,  17, -19, -22,  19,  16,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  18,  17, -19, -22,  19,  16,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  18,  17, -19, -22,  19,  16,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  18,  17, -19, -22,  19,  16,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  18,  17, -19, -22,  19,  16,   0,   0,   0,   0,   0,   0,   0,   0,   0],


    In [8]: print(a.blib.format(a.bn.view(np.int8).reshape(-1,16)[0]))                                                                                                                                      
     18 : Acrylic///LS        TO 
     18 : Acrylic///LS        SC
     17 : Water///Acrylic     BT
    -19 : LS///Acrylic        BT      /// huh: looks inconsistent border, should be Water here ??? 
    -22 : Water///PE_PA                            #### jcv XJfixtureConstruction
     19 : LS///Acrylic
     16 : Tyvek//Implicit_RINDEX_NoRINDEX_pInnerWater_pCentralDetector/Water


    epsilon:ana blyth$ ./bidx.py 
      0 :   1 :       1 : Galactic///Galactic 
      1 :   2 :       2 : Galactic///Rock 
      2 :   3 :       1 : Rock//Implicit_RINDEX_NoRINDEX_pExpHall_pTopRock/Air 
      3 :   4 :     191 : Air///Air 
      4 :   5 :       1 : Air///LS 
      5 :   6 :       1 : Air///Steel 
      6 :   7 :       1 : Air///Tyvek 
      7 :   8 :     504 : Air///Aluminium 
      8 :   9 :     504 : Aluminium///Adhesive 
      9 :  10 :   32256 : Adhesive///TiO2Coating 
     10 :  11 :   32256 : TiO2Coating///Scintillator 
     11 :  12 :       1 : Rock///Tyvek 
     12 :  13 :       1 : Tyvek//Implicit_RINDEX_NoRINDEX_pOuterWaterPool_pPoolLining/vetoWater 
     13 :  14 :    2120 : vetoWater///LatticedShellSteel 
     14 :  15 :       1 : vetoWater/CDTyvekSurface//Tyvek 
     15 :  16 :       1 : Tyvek//Implicit_RINDEX_NoRINDEX_pInnerWater_pCentralDetector/Water 

     16 :  17 :    3048 : Water///Acrylic 
     17 :  18 :       1 : Acrylic///LS 
     18 :  19 :      46 : LS///Acrylic 
     19 :  20 :       8 : LS///PE_PA 

     20 :  21 :   27960 : Water///Steel 

     21 :  22 :      56 : Water///PE_PA 

     22 :  23 :   45612 : Water///Pyrex 

     23 :  24 :   20012 : Pyrex///Pyrex 

     24 :  25 :   12612 : Pyrex/NNVTMCPPMT_photocathode_logsurf2/NNVTMCPPMT_photocathode_logsurf1/Vacuum 
     25 :  26 :   12612 : Pyrex//NNVTMCPPMT_mirror_logsurf1/Vacuum 
     26 :  27 :    5000 : Pyrex/HamamatsuR12860_photocathode_logsurf2/HamamatsuR12860_photocathode_logsurf1/Vacuum 
     27 :  28 :    5000 : Pyrex//HamamatsuR12860_mirror_logsurf1/Vacuum 
     28 :  29 :   25601 : Water///Water 
     29 :  30 :   25600 : Pyrex/PMT_3inch_photocathode_logsurf2/PMT_3inch_photocathode_logsurf1/Vacuum 
     30 :  31 :   25600 : Pyrex//PMT_3inch_absorb_logsurf1/Vacuum 
     31 :  32 :       1 : Water///LS 
     32 :  33 :       1 : Water/Steel_surface/Steel_surface/Steel 
     33 :  34 :    2400 : vetoWater///Water 
     34 :  35 :    2400 : Pyrex/PMT_20inch_veto_photocathode_logsurf2/PMT_20inch_veto_photocathode_logsurf1/Vacuum 
     35 :  36 :    2400 : Pyrex//PMT_20inch_veto_mirror_logsurf1/Vacuum 
    epsilon:ana blyth$ 


Use ggeo.sh to find where 0-based bidx 16,17,18,19 are used::

    In [2]: gg.bidx.shape                                                                                                                                                                                   
    Out[2]: (283812,)

    In [3]: np.where( gg.bidx == 17 )                                                                                                                                                                       
    Out[3]: (array([67843]),)

    In [4]: np.where( gg.bidx == 17 )[0]                                                                                                                                                                    
    Out[4]: array([67843])

    In [5]: gg.pv[67843]                                                                                                                                                                                    
    Out[5]: b'pTarget0x3358bb0'





    In [23]: w21 = np.where( gg.bidx == 21 )[0]                                                                                                                                                            

    In [24]: gg.pv[w21]                                                                                                                                                                                    
    Out[24]: 
    array([b'lXJfixture_phys0x349fa90', b'lXJfixture_phys0x349fb90', b'lXJfixture_phys0x349fc90', b'lXJfixture_phys0x349fd90', b'lXJfixture_phys0x349fe90', b'lXJfixture_phys0x349ff90',
           b'lXJfixture_phys0x34a0090', b'lXJfixture_phys0x34a0190', b'lXJfixture_phys0x34a0290', b'lXJfixture_phys0x34a0390', b'lXJfixture_phys0x349c620', b'lXJfixture_phys0x349c720',
           b'lXJfixture_phys0x349c820', b'lXJfixture_phys0x349c920', b'lXJfixture_phys0x349ca20', b'lXJfixture_phys0x349cb20', b'lXJfixture_phys0x349cc20', b'lXJfixture_phys0x349cd20',
           b'lXJfixture_phys0x349ce20', b'lXJfixture_phys0x349cf20', b'lXJfixture_phys0x349d020', b'lXJfixture_phys0x349d120', b'lXJfixture_phys0x349d220', b'lXJfixture_phys0x349d320',
           b'lXJfixture_phys0x349d420', b'lXJfixture_phys0x349d520', b'lXJfixture_phys0x349d620', b'lXJfixture_phys0x349d720', b'lXJfixture_phys0x349d820', b'lXJfixture_phys0x349d920',
           b'lXJfixture_phys0x349da20', b'lXJfixture_phys0x349db20', b'lXJfixture_phys0x349dc20', b'lXJfixture_phys0x349dd20', b'lXJfixture_phys0x349de20', b'lXJfixture_phys0x349df20',
           b'lXJfixture_phys0x349e020', b'lXJfixture_phys0x349e120', b'lXJfixture_phys0x349e220', b'lXJfixture_phys0x349e320', b'lXJfixture_phys0x349e420', b'lXJfixture_phys0x349e520',
           b'lXJfixture_phys0x349e620', b'lXJfixture_phys0x349e720', b'lXJfixture_phys0x34a1c90', b'lXJfixture_phys0x34a1d90', b'lXJfixture_phys0x34a1e90', b'lXJfixture_phys0x34a1f90',
           b'lXJfixture_phys0x34a2090', b'lXJfixture_phys0x34a2190', b'lXJfixture_phys0x34a2290', b'lXJfixture_phys0x34a2390', b'lXJfixture_phys0x34a2490', b'lXJfixture_phys0x34a2590',
           b'lXJfixture_phys0x34a2690', b'lXJfixture_phys0x34a2790'], dtype='|S100')


::

    jcv XJfixtureConstruction






    In [9]: a.seqmat_ana.table                                                                                                                                                                              
    Out[9]: 
    seqmat_ana
    .                     cfo:-  1:g4live:tds3ip 
    .                                297         1.00 
       n             iseq         frac           a    a-b      [ns] label
    0000         3eddeb11        1.000         297        [8 ] LS LS Ac Py Va Va Py Ty
       n             iseq         frac           a    a-b      [ns] label
    .                                297         1.00 

    epsilon:GItemList blyth$ cat.py GMaterialLib.txt 
    0    1    LS
    1    2    Steel
    2    3    Tyvek
    3    4    Air
    4    5    Scintillator
    5    6    TiO2Coating
    6    7    Adhesive
    7    8    Aluminium
    8    9    Rock
    9    10   LatticedShellSteel
    10   11   Acrylic
    11   12   PE_PA
    12   13   Vacuum
    13   14   Pyrex
    14   15   Water
    15   16   vetoWater
    16   17   Galactic
    epsilon:GItemList blyth$ 









jsc : flagReemission requires fStopAndKill track status not at fGeomBoundary step status
------------------------------------------------------------------------------------------

* G4OpAbsorption is principal way to fStopAndKill 

  * does that mean that process order must have scintillation after absorption ?



::

     223     if (aTrack.GetDefinition() == G4OpticalPhoton::OpticalPhoton()) {
     224         G4Track *track=aStep.GetTrack();
     225         //G4CompositeTrackInfo* composite=dynamic_cast<G4CompositeTrackInfo*>(track->GetUserInformation());
     226         //reemittedTI = composite?dynamic_cast<DsPhotonTrackInfo*>( composite->GetPhotonTrackInfo() ):0;
     227 
     228         const G4VProcess* process = track->GetCreatorProcess();
     229         if(process) pname = process->GetProcessName();
     230 
     231         if (verboseLevel > 0) {
     232           G4cout<<"Optical photon. Process name is " << pname<<G4endl;
     233         }
     234         if(doBothProcess) {
     235             flagReemission= doReemission
     236                 && aTrack.GetTrackStatus() == fStopAndKill
     237                 && aStep.GetPostStepPoint()->GetStepStatus() != fGeomBoundary;
     238         }








