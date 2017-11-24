tboolean-cubeplanes-many-stuck-tracks-drastic-difference
============================================================


* cubeplanes constructs a cube using a G4TesselatedSolid 


FIXED : the drastic problem, incomplete tris
------------------------------------------------

Following hint from Ivana Hrivnacova re G4TesselatedSolid

* http://hypernews.slac.stanford.edu/HyperNews/geant4/get/geometry/1257/1/1/1/1/1/1/1.html


PUMP UP Stats and use uvdomain emitconfig brings into line
------------------------------------------------------------

::

    [2017-11-24 16:52:28,788] p32647 {/Users/blyth/opticks/ana/ab.py:152} INFO - AB.init_point DONE
    AB(1,torch,tboolean-cubeplanes)  None 0 
    A tboolean-cubeplanes/torch/  1 :  20171124-1652 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-cubeplanes/torch/1/fdom.npy () 
    B tboolean-cubeplanes/torch/ -1 :  20171124-1652 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-cubeplanes/torch/-1/fdom.npy (recstp) 
    Rock//perfectAbsorbSurface/Vacuum,Vacuum///GlassSchottF2
    /tmp/blyth/opticks/tboolean-cubeplanes--
    .                seqhis_ana  1:tboolean-cubeplanes   -1:tboolean-cubeplanes        c2        ab        ba 
    .                             600000    600000         5.11/14 =  0.37  (pval:0.984 prob:0.016)  
    0000             8ccd    337842    337692             0.03        1.000 +- 0.002        1.000 +- 0.002  [4 ] TO BT BT SA
    0001               8d    215778    215777             0.00        1.000 +- 0.002        1.000 +- 0.002  [2 ] TO SA
    0002              8bd     23427     23472             0.04        0.998 +- 0.007        1.002 +- 0.007  [3 ] TO BR SA
    0003            8cbcd     20772     20919             0.52        0.993 +- 0.007        1.007 +- 0.007  [5 ] TO BT BR BT SA
    0004           8cbbcd      1319      1277             0.68        1.033 +- 0.028        0.968 +- 0.027  [6 ] TO BT BR BR BT SA
    0005              86d       255       241             0.40        1.058 +- 0.066        0.945 +- 0.061  [3 ] TO SC SA
    0006              4cd       144       133             0.44        1.083 +- 0.090        0.924 +- 0.080  [3 ] TO BT AB
    0007          8cbbbcd        77        80             0.06        0.963 +- 0.110        1.039 +- 0.116  [7 ] TO BT BR BR BR BT SA
    0008            86ccd        74        69             0.17        1.072 +- 0.125        0.932 +- 0.112  [5 ] TO BT BT SC SA
    0009       bbbbbbb6cd        62        64             0.03        0.969 +- 0.123        1.032 +- 0.129  [10] TO BT SC BR BR BR BR BR BR BR
    0010            8c6cd        52        61             0.72        0.852 +- 0.118        1.173 +- 0.150  [5 ] TO BT SC BT SA
    0011               4d        31        34             0.14        0.912 +- 0.164        1.097 +- 0.188  [2 ] TO AB
    0012           8cbc6d        29        26             0.16        1.115 +- 0.207        0.897 +- 0.176  [6 ] TO SC BT BR BT SA
    0013            8cc6d        24        34             1.72        0.706 +- 0.144        1.417 +- 0.243  [5 ] TO SC BT BT SA
    0014           8cb6cd        16        16             0.00        1.000 +- 0.250        1.000 +- 0.250  [6 ] TO BT SC BR BT SA
    0015         8cbc6ccd        10        10             0.00        1.000 +- 0.316        1.000 +- 0.316  [8 ] TO BT BT SC BT BR BT SA
    0016          8cc6ccd         9        10             0.00        0.900 +- 0.300        1.111 +- 0.351  [7 ] TO BT BT SC BT BT SA
    0017             8b6d         9         8             0.00        1.125 +- 0.375        0.889 +- 0.314  [4 ] TO SC BR SA
    0018          8cbbc6d         9         6             0.00        1.500 +- 0.500        0.667 +- 0.272  [7 ] TO SC BT BR BR BT SA
    0019             4ccd         8        17             0.00        0.471 +- 0.166        2.125 +- 0.515  [4 ] TO BT BT AB
    .                             600000    600000         5.11/14 =  0.37  (pval:0.984 prob:0.016)  
    .                pflags_ana  1:tboolean-cubeplanes   -1:tboolean-cubeplanes        c2        ab        ba 
    .                             600000    600000         2.28/9 =  0.25  (pval:0.986 prob:0.014)  
    0000             1880    337842    337692             0.03        1.000 +- 0.002        1.000 +- 0.002  [3 ] TO|BT|SA
    0001             1080    215778    215777             0.00        1.000 +- 0.002        1.000 +- 0.002  [2 ] TO|SA
    0002             1480     23427     23472             0.04        0.998 +- 0.007        1.002 +- 0.007  [3 ] TO|BR|SA
    0003             1c80     22170     22282             0.28        0.995 +- 0.007        1.005 +- 0.007  [4 ] TO|BT|BR|SA
    0004             10a0       255       241             0.40        1.058 +- 0.066        0.945 +- 0.061  [3 ] TO|SA|SC
    0005             18a0       160       174             0.59        0.920 +- 0.073        1.087 +- 0.082  [4 ] TO|BT|SA|SC
    0006             1808       152       150             0.01        1.013 +- 0.082        0.987 +- 0.081  [3 ] TO|BT|AB
    0007             1ca0       101        89             0.76        1.135 +- 0.113        0.881 +- 0.093  [5 ] TO|BT|BR|SA|SC
    0008             1c20        67        65             0.03        1.031 +- 0.126        0.970 +- 0.120  [4 ] TO|BT|BR|SC
    0009             1008        31        34             0.14        0.912 +- 0.164        1.097 +- 0.188  [2 ] TO|AB
    0010             14a0        10        11             0.00        0.909 +- 0.287        1.100 +- 0.332  [4 ] TO|BR|SA|SC
    0011             1c08         6        13             0.00        0.462 +- 0.188        2.167 +- 0.601  [4 ] TO|BT|BR|AB
    0012             1408         1         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [3 ] TO|BR|AB
    .                             600000    600000         2.28/9 =  0.25  (pval:0.986 prob:0.014)  
    .                seqmat_ana  1:tboolean-cubeplanes   -1:tboolean-cubeplanes        c2        ab        ba 
    .                             600000    600000         4.02/11 =  0.37  (pval:0.969 prob:0.031)  
    0000             1232    337842    337692             0.03        1.000 +- 0.002        1.000 +- 0.002  [4 ] Vm F2 Vm Rk
    0001               12    215778    215777             0.00        1.000 +- 0.002        1.000 +- 0.002  [2 ] Vm Rk
    0002              122     23682     23713             0.02        0.999 +- 0.006        1.001 +- 0.007  [3 ] Vm Vm Rk
    0003            12332     20824     20980             0.58        0.993 +- 0.007        1.007 +- 0.007  [5 ] Vm F2 F2 Vm Rk
    0004           123332      1338      1294             0.74        1.034 +- 0.028        0.967 +- 0.027  [6 ] Vm F2 F2 F2 Vm Rk
    0005              332       144       133             0.44        1.083 +- 0.090        0.924 +- 0.080  [3 ] Vm F2 F2
    0006          1233332        86        87             0.01        0.989 +- 0.107        1.012 +- 0.108  [7 ] Vm F2 F2 F2 F2 Vm Rk
    0007            12232        74        69             0.17        1.072 +- 0.125        0.932 +- 0.112  [5 ] Vm F2 Vm Vm Rk
    0008       3333333332        66        65             0.01        1.015 +- 0.125        0.985 +- 0.122  [10] Vm F2 F2 F2 F2 F2 F2 F2 F2 F2
    0009               22        31        34             0.14        0.912 +- 0.164        1.097 +- 0.188  [2 ] Vm Vm
    0010           123322        29        26             0.16        1.115 +- 0.207        0.897 +- 0.176  [6 ] Vm Vm F2 F2 Vm Rk
    0011            12322        24        34             1.72        0.706 +- 0.144        1.417 +- 0.243  [5 ] Vm Vm F2 Vm Rk
    0012         12332232        10        10             0.00        1.000 +- 0.316        1.000 +- 0.316  [8 ] Vm F2 Vm Vm F2 F2 Vm Rk
    0013             1222        10        11             0.00        0.909 +- 0.287        1.100 +- 0.332  [4 ] Vm Vm Vm Rk
    0014          1232232         9        10             0.00        0.900 +- 0.300        1.111 +- 0.351  [7 ] Vm F2 Vm Vm F2 Vm Rk
    0015          1233322         9         6             0.00        1.500 +- 0.500        0.667 +- 0.272  [7 ] Vm Vm F2 F2 F2 Vm Rk
    0016             2232         8        17             0.00        0.471 +- 0.166        2.125 +- 0.515  [4 ] Vm F2 Vm Vm
    0017           122332         5         5             0.00        1.000 +- 0.447        1.000 +- 0.447  [6 ] Vm F2 F2 Vm Vm Rk
    0018             3332         5        11             0.00        0.455 +- 0.203        2.200 +- 0.663  [4 ] Vm F2 F2 F2
    0019       1233333332         4         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [10] Vm F2 F2 F2 F2 F2 F2 F2 Vm Rk
    .                             600000    600000         4.02/11 =  0.37  (pval:0.969 prob:0.031)  




BUT low stats histories are increasing chisq above cut 
------------------------------------------------------------------------------

Low stats TO|BT|AB pushing above c2max ?

::

    AB(1,torch,tboolean-cubeplanes)  None 0 
    A tboolean-cubeplanes/torch/  1 :  20171124-1645 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-cubeplanes/torch/1/fdom.npy () 
    B tboolean-cubeplanes/torch/ -1 :  20171124-1645 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-cubeplanes/torch/-1/fdom.npy (recstp) 
    Rock//perfectAbsorbSurface/Vacuum,Vacuum///GlassSchottF2
    /tmp/blyth/opticks/tboolean-cubeplanes--
    .                seqhis_ana  1:tboolean-cubeplanes   -1:tboolean-cubeplanes        c2        ab        ba 
    .                             100000    100000         4.22/6 =  0.70  (pval:0.647 prob:0.353)  
    0000            8cccd     50221     50317             0.09        0.998 +- 0.004        1.002 +- 0.004  [5 ] TO BT BT BT SA
    0001              8cd     43287     43296             0.00        1.000 +- 0.005        1.000 +- 0.005  [3 ] TO BT SA
    0002             8bcd      3337      3253             1.07        1.026 +- 0.018        0.975 +- 0.017  [4 ] TO BT BR SA
    0003           8cbccd      2859      2822             0.24        1.013 +- 0.019        0.987 +- 0.019  [6 ] TO BT BT BR BT SA
    0004          8cbbccd       163       150             0.54        1.087 +- 0.085        0.920 +- 0.075  [7 ] TO BT BT BR BR BT SA
    0005             86cd        39        44             0.30        0.886 +- 0.142        1.128 +- 0.170  [4 ] TO BT SC SA
    0006             4ccd        16        25             1.98        0.640 +- 0.160        1.562 +- 0.312  [4 ] TO BT BT AB
    0007               4d        10        12             0.00        0.833 +- 0.264        1.200 +- 0.346  [2 ] TO AB
    0008         8cbbbccd         9         9             0.00        1.000 +- 0.333        1.000 +- 0.333  [8 ] TO BT BT BR BR BR BT SA
    0009           86cccd         9        11             0.00        0.818 +- 0.273        1.222 +- 0.369  [6 ] TO BT BT BT SC SA
    0010           8c6ccd         7         4             0.00        1.750 +- 0.661        0.571 +- 0.286  [6 ] TO BT BT SC BT SA
    0011          8cbc6cd         6         3             0.00        2.000 +- 0.816        0.500 +- 0.289  [7 ] TO BT SC BT BR BT SA
    0012       bbbbbb6ccd         6         5             0.00        1.200 +- 0.490        0.833 +- 0.373  [10] TO BT BT SC BR BR BR BR BR BR
    0013               3d         6         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [2 ] TO MI
    0014           8cc6cd         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [6 ] TO BT SC BT BT SA
    0015            4bccd         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [5 ] TO BT BT BR AB
    0016             8c6d         3         7             0.00        0.429 +- 0.247        2.333 +- 0.882  [4 ] TO SC BT SA
    0017              4cd         2         7             0.00        0.286 +- 0.202        3.500 +- 1.323  [3 ] TO BT AB
    0018            4cccd         2         6             0.00        0.333 +- 0.236        3.000 +- 1.225  [5 ] TO BT BT BT AB
    0019        8cbb6bccd         2         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [9 ] TO BT BT BR SC BR BR BT SA
    .                             100000    100000         4.22/6 =  0.70  (pval:0.647 prob:0.353)  
    .                pflags_ana  1:tboolean-cubeplanes   -1:tboolean-cubeplanes        c2        ab        ba 
    .                             100000    100000         7.34/3 = **2.45**  (pval:0.062 prob:0.938)  
    0000             1880     93508     93613             0.06        0.999 +- 0.003        1.001 +- 0.003  [3 ] TO|BT|SA
    0001             1c80      6368      6234             1.42        1.021 +- 0.013        0.979 +- 0.012  [4 ] TO|BT|BR|SA
    0002             18a0        63        69             0.27        0.913 +- 0.115        1.095 +- 0.132  [4 ] TO|BT|SA|SC
    0003             1808        20        38           **5.59**      0.526 +- 0.118        1.900 +- 0.308  [3 ] TO|BT|AB
    0004             1ca0        15        15             0.00        1.000 +- 0.258        1.000 +- 0.258  [5 ] TO|BT|BR|SA|SC
    0005             1008        10        12             0.00        0.833 +- 0.264        1.200 +- 0.346  [2 ] TO|AB
    0006             1c20         7         6             0.00        1.167 +- 0.441        0.857 +- 0.350  [4 ] TO|BT|BR|SC
    0007             1004         6         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [2 ] TO|MI
    0008             1c08         3         4             0.00        0.750 +- 0.433        1.333 +- 0.667  [4 ] TO|BT|BR|AB
    0009             1024         0         9             0.00        0.000 +- 0.000        0.000 +- 0.000  [3 ] TO|SC|MI
    .                             100000    100000         7.34/3 =  2.45  (pval:0.062 prob:0.938)  
    .                seqmat_ana  1:tboolean-cubeplanes   -1:tboolean-cubeplanes        c2        ab        ba 
    .                             100000    100000         3.74/5 =  0.75  (pval:0.588 prob:0.412)  
    0000            12321     50222     50317             0.09        0.998 +- 0.004        1.002 +- 0.004  [5 ] Rk Vm F2 Vm Rk
    0001              121     43287     43296             0.00        1.000 +- 0.005        1.000 +- 0.005  [3 ] Rk Vm Rk
    0002             1221      3376      3297             0.94        1.024 +- 0.018        0.977 +- 0.017  [4 ] Rk Vm Vm Rk
    0003           123321      2866      2826             0.28        1.014 +- 0.019        0.986 +- 0.019  [6 ] Rk Vm F2 F2 Vm Rk
    0004          1233321       164       152             0.46        1.079 +- 0.084        0.927 +- 0.075  [7 ] Rk Vm F2 F2 F2 Vm Rk
    0005             3321        16        25             1.98        0.640 +- 0.160        1.562 +- 0.312  [4 ] Rk Vm F2 F2
    0006               11        10        12             0.00        0.833 +- 0.264        1.200 +- 0.346  [2 ] Rk Rk
    0007           122321         9        11             0.00        0.818 +- 0.273        1.222 +- 0.369  [6 ] Rk Vm F2 Vm Vm Rk
    0008         12333321         9        11             0.00        0.818 +- 0.273        1.222 +- 0.369  [8 ] Rk Vm F2 F2 F2 F2 Vm Rk
    0009       3333333321         7         5             0.00        1.400 +- 0.529        0.714 +- 0.319  [10] Rk Vm F2 F2 F2 F2 F2 F2 F2 F2
    0010                1         6         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [1 ] Rk
    0011          1233221         6         3             0.00        2.000 +- 0.816        0.500 +- 0.289  [7 ] Rk Vm Vm F2 F2 Vm Rk
    0012            33321         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [5 ] Rk Vm F2 F2 F2
    0013           123221         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [6 ] Rk Vm Vm F2 Vm Rk
    0014             1211         3         7             0.00        0.429 +- 0.247        2.333 +- 0.882  [4 ] Rk Rk Vm Rk
    0015            22321         2         6             0.00        0.333 +- 0.236        3.000 +- 1.225  [5 ] Rk Vm F2 Vm Vm
    0016              221         2         7             0.00        0.286 +- 0.202        3.500 +- 1.323  [3 ] Rk Vm Vm
    0017        123333321         2         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [9 ] Rk Vm F2 F2 F2 F2 F2 Vm Rk
    0018            12221         2         3             0.00        0.667 +- 0.471        1.500 +- 0.866  [5 ] Rk Vm Vm Vm Rk
    0019          1232221         1         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [7 ] Rk Vm Vm Vm F2 Vm Rk
    .                             100000    100000         3.74/5 =  0.75  (pval:0.588 prob:0.412)  






::


    [2017-11-24 16:38:17,836] p31828 {/Users/blyth/opticks/ana/ab.py:152} INFO - AB.init_point DONE
    AB(1,torch,tboolean-cubeplanes)  None 0 
    A tboolean-cubeplanes/torch/  1 :  20171124-1638 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-cubeplanes/torch/1/fdom.npy () 
    B tboolean-cubeplanes/torch/ -1 :  20171124-1638 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-cubeplanes/torch/-1/fdom.npy (recstp) 
    Rock//perfectAbsorbSurface/Vacuum,Vacuum///GlassSchottF2
    /tmp/blyth/opticks/tboolean-cubeplanes--
    .                seqhis_ana  1:tboolean-cubeplanes   -1:tboolean-cubeplanes        c2        ab        ba 
    .                             100000    100000         4.14/2 =  2.07  (pval:0.126 prob:0.874)  
    0000               8d     83715     83731             0.00        1.000 +- 0.003        1.000 +- 0.003  [2 ] TO SA
    0001              8ad     16176     16177             0.00        1.000 +- 0.008        1.000 +- 0.008  [3 ] TO SR SA
    0002              86d        88        63             4.14        1.397 +- 0.149        0.716 +- 0.090  [3 ] TO SC SA
    0003               4d         9        13             0.00        0.692 +- 0.231        1.444 +- 0.401  [2 ] TO AB
    0004             8a6d         8        10             0.00        0.800 +- 0.283        1.250 +- 0.395  [4 ] TO SC SR SA
    0005             86ad         4         4             0.00        1.000 +- 0.500        1.000 +- 0.500  [4 ] TO SR SC SA
    0006              4ad         0         1             0.00        0.000 +- 0.000        0.000 +- 0.000  [3 ] TO SR AB
    0007            8a6ad         0         1             0.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] TO SR SC SR SA
    .                             100000    100000         4.14/2 =  2.07  (pval:0.126 prob:0.874)  
    .                pflags_ana  1:tboolean-cubeplanes   -1:tboolean-cubeplanes        c2        ab        ba 
    .                             100000    100000         4.14/2 =  2.07  (pval:0.126 prob:0.874)  
    0000             1080     83715     83731             0.00        1.000 +- 0.003        1.000 +- 0.003  [2 ] TO|SA
    0001             1280     16176     16177             0.00        1.000 +- 0.008        1.000 +- 0.008  [3 ] TO|SR|SA
    0002             10a0        88        63             4.14        1.397 +- 0.149        0.716 +- 0.090  [3 ] TO|SA|SC
    0003             12a0        12        15             0.00        0.800 +- 0.231        1.250 +- 0.323  [4 ] TO|SR|SA|SC
    0004             1008         9        13             0.00        0.692 +- 0.231        1.444 +- 0.401  [2 ] TO|AB
    0005             1208         0         1             0.00        0.000 +- 0.000        0.000 +- 0.000  [3 ] TO|SR|AB
    .                             100000    100000         4.14/2 =  2.07  (pval:0.126 prob:0.874)  
    .                seqmat_ana  1:tboolean-cubeplanes   -1:tboolean-cubeplanes        c2        ab        ba 
    .                             100000    100000         0.02/1 =  0.02  (pval:0.890 prob:0.110)  
    0000               12     83715     83731             0.00        1.000 +- 0.003        1.000 +- 0.003  [2 ] Vm Rk
    0001              122     16264     16240             0.02        1.001 +- 0.008        0.999 +- 0.008  [3 ] Vm Vm Rk
    0002             1222        12        14             0.00        0.857 +- 0.247        1.167 +- 0.312  [4 ] Vm Vm Vm Rk
    0003               22         9        13             0.00        0.692 +- 0.231        1.444 +- 0.401  [2 ] Vm Vm
    0004              222         0         1             0.00        0.000 +- 0.000        0.000 +- 0.000  [3 ] Vm Vm Vm
    0005            12222         0         1             0.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] Vm Vm Vm Vm Rk
    .                             100000    100000         0.02/1 =  0.02  (pval:0.890 prob:0.110)  
    ab.a.metadata           /tmp/blyth/opticks/evt/tboolean-cubeplanes/torch/1 1f9d4f67b8923f8c19db3344b63e111b 3394c3b127f73ebe








Below was due to incomplete G4TesselatedSolid
------------------------------------------------


::

    2017-11-20 18:58:39.071 INFO  [5999579] [CSteppingAction::setStep@111]  noZeroSteps 9 severity 4 ctx  record_id 90719 event_id 9 track_id 719 photon_id 719 parent_id -1 primary_id -2 reemtrack 0

    -------- WWWW ------- G4Exception-START -------- WWWW -------
    *** G4Exception : GeomNav1002
          issued by : G4Navigator::ComputeStep()
    Track stuck or not moving.
              Track stuck, not moving for 10 steps
              in volume -box_pv0_- at point (25.6317,196.246,200)
              direction: (0,0,-1).
              Potential geometry or navigation problem !
              Trying pushing it of 1e-07 mm ...Potential overlap in geometry!

    *** This is just a warning message. ***
    -------- WWWW -------- G4Exception-END --------- WWWW -------

    2017-11-20 18:58:39.071 INFO  [5999579] [CSteppingAction::setStep@111]  noZeroSteps 10 severity 5 ctx  record_id 90719 event_id 9 track_id 719 photon_id 719 parent_id -1 primary_id -2 reemtrack 0
    2017-11-20 18:58:39.071 INFO  [5999579] [CSteppingAction::setStep@111]  noZeroSteps 11 severity 0 ctx  record_id 90719 event_id 9 track_id 719 photon_id 719 parent_id -1 primary_id -2 reemtrack 0
    2017-11-20 18:58:39.071 INFO  [5999579] [CSteppingAction::setStep@111]  noZeroSteps 2 severity 1 ctx  record_id 90708 event_id 9 track_id 708 photon_id 708 parent_id -1 primary_id -2 reemtrack 0
    2017-11-20 18:58:39.071 INFO  [5999579] [CSteppingAction::setStep@111]  noZeroSteps 3 severity 1 ctx  record_id 90708 event_id 9 track_id 708 photon_id 708 parent_id -1 primary_id -2 reemtrack 0
    2017-11-20 18:58:39.072 INFO  [5999579] [CSteppingAction::setStep@111]  noZeroSteps 4 severity 2 ctx  record_id 90708 event_id 9 track_id 708 photon_id 708 parent_id -1 primary_id -2 reemtrack 0
    2017-11-20 18:58:39.072 INFO  [5999579] [CSteppingAction::setStep@111]  noZeroSteps 5 severity 2 ctx  record_id 90708 event_id 9 track_id 708 photon_id 708 parent_id -1 primary_id -2 reemtrack 0
    2017-11-20 18:58:39.072 INFO  [5999579] [CSteppingAction::setStep@111]  noZeroSteps 6 severity 3 ctx  record_id 90708 event_id 9 track_id 708 photon_id 708 parent_id -1 primary_id -2 reemtrack 0
    2017-11-20 18:58:39.072 INFO  [5999579] [CSteppingAction::setStep@111]  noZeroSteps 7 severity 3 ctx  record_id 90708 event_id 9 track_id 708 photon_id 708 parent_id -1 primary_id -2 reemtrack 0
    2017-11-20 18:58:39.072 INFO  [5999579] [CSteppingAction::setStep@111]  noZeroSteps 8 severity 4 ctx  record_id 90708 event_id 9 track_id 708 photon_id 708 parent_id -1 primary_id -2 reemtrack 0
    2017-11-20 18:58:39.072 INFO  [5999579] [CSteppingAction::setStep@111]  noZeroSteps 9 severity 4 ctx  record_id 90708 event_id 9 track_id 708 photon_id 708 parent_id -1 primary_id -2 reemtrack 0

    -------- WWWW ------- G4Exception-START -------- WWWW -------
    *** G4Exception : GeomNav1002
          issued by : G4Navigator::ComputeStep()
    Track stuck or not moving.
              Track stuck, not moving for 10 steps
              in volume -box_pv0_- at point (-50.476,113.388,200)
              direction: (0,0,-1).
              Potential geometry or navigation problem !
              Trying pushing it of 1e-07 mm ...Potential overlap in geometry!

    *** This is just a warning message. ***
    -------- WWWW -------- G4Exception-END --------- WWWW -------





::


    [2017-11-20 18:58:58,125] p80886 {/Users/blyth/opticks/ana/ab.py:152} INFO - AB.init_point DONE
    [2017-11-20 18:58:58,126] p80886 {/Users/blyth/opticks/ana/seq.py:160} WARNING - SeqType.code check [?0?] bad 1 
    [2017-11-20 18:58:58,128] p80886 {/Users/blyth/opticks/ana/seq.py:160} WARNING - SeqType.code check [?0?] bad 1 
    AB(1,torch,tboolean-cubeplanes)  None 0 
    A tboolean-cubeplanes/torch/  1 :  20171120-1858 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-cubeplanes/torch/1/fdom.npy () 
    B tboolean-cubeplanes/torch/ -1 :  20171120-1858 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-cubeplanes/torch/-1/fdom.npy (recstp) 
    Rock//perfectAbsorbSurface/Vacuum,Vacuum///GlassSchottF2
    /tmp/blyth/opticks/tboolean-cubeplanes--
    .                seqhis_ana  1:tboolean-cubeplanes   -1:tboolean-cubeplanes        c2        ab        ba 
    .                             100000    100000     43137.86/8 = 5392.23  (pval:0.000 prob:1.000)  
    0000            8cccd     50221     12612         22511.05        3.982 +- 0.018        0.251 +- 0.002  [5 ] TO BT BT BT SA
    0001              8cd     43287     71594          6974.92        0.605 +- 0.003        1.654 +- 0.006  [3 ] TO BT SA
    0002             8bcd      3337      1589           620.28        2.100 +- 0.036        0.476 +- 0.012  [4 ] TO BT BR SA
    0003           8cbccd      2859      1475           441.96        1.938 +- 0.036        0.516 +- 0.013  [6 ] TO BT BT BR BT SA
    0004          8cbbccd       163        56            52.28        2.911 +- 0.228        0.344 +- 0.046  [7 ] TO BT BT BR BR BT SA
    0005             86cd        39        75            11.37        0.520 +- 0.083        1.923 +- 0.222  [4 ] TO BT SC SA
    0006             4ccd        16         5             0.00        3.200 +- 0.800        0.312 +- 0.140  [4 ] TO BT BT AB
    0007               4d        10        14             0.00        0.714 +- 0.226        1.400 +- 0.374  [2 ] TO AB
    0008         8cbbbccd         9         2             0.00        4.500 +- 1.500        0.222 +- 0.157  [8 ] TO BT BT BR BR BR BT SA
    0009           86cccd         9         1             0.00        9.000 +- 3.000        0.111 +- 0.111  [6 ] TO BT BT BT SC SA
    0010           8c6ccd         7         3             0.00        2.333 +- 0.882        0.429 +- 0.247  [6 ] TO BT BT SC BT SA
    0011          8cbc6cd         6         4             0.00        1.500 +- 0.612        0.667 +- 0.333  [7 ] TO BT SC BT BR BT SA
    0012       bbbbbb6ccd         6         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [10] TO BT BT SC BR BR BR BR BR BR
    0013               3d         6         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [2 ] TO MI
    0014           8cc6cd         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [6 ] TO BT SC BT BT SA
    0015            4bccd         3         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] TO BT BT BR AB
    0016             8c6d         3         4             0.00        0.750 +- 0.433        1.333 +- 0.667  [4 ] TO SC BT SA
    0017              4cd         2         9             0.00        0.222 +- 0.157        4.500 +- 1.500  [3 ] TO BT AB
    0018            4cccd         2         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] TO BT BT BT AB
    0019        8cbb6bccd         2         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [9 ] TO BT BT BR SC BR BR BT SA
    .                             100000    100000     43137.86/8 = 5392.23  (pval:0.000 prob:1.000)  
    .                pflags_ana  1:tboolean-cubeplanes   -1:tboolean-cubeplanes        c2        ab        ba 
    .                             100000    100000       716.80/4 = 179.20  (pval:0.000 prob:1.000)  
    0000             1880     93508     95967            31.91        0.974 +- 0.003        1.026 +- 0.003  [3 ] TO|BT|SA
    0001             1c80      6368      3838           627.17        1.659 +- 0.021        0.603 +- 0.010  [4 ] TO|BT|BR|SA
    0002             18a0        63        99             8.00        0.636 +- 0.080        1.571 +- 0.158  [4 ] TO|BT|SA|SC
    0003             1808        20        15             0.71        1.333 +- 0.298        0.750 +- 0.194  [3 ] TO|BT|AB
    0004             1ca0        15        13             0.00        1.154 +- 0.298        0.867 +- 0.240  [5 ] TO|BT|BR|SA|SC
    0005             1008        10        14             0.00        0.714 +- 0.226        1.400 +- 0.374  [2 ] TO|AB
    0006             1c20         7         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [4 ] TO|BT|BR|SC
    0007             1004         6         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [2 ] TO|MI
    0008             1c08         3         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [4 ] TO|BT|BR|AB
    0009                0         0        49            49.00        0.000 +- 0.000        0.000 +- 0.000  [1 ]
    0010             1024         0         5             0.00        0.000 +- 0.000        0.000 +- 0.000  [3 ] TO|SC|MI
    .                             100000    100000       716.80/4 = 179.20  (pval:0.000 prob:1.000)  
    .                seqmat_ana  1:tboolean-cubeplanes   -1:tboolean-cubeplanes        c2        ab        ba 
    .                             100000    100000     44613.94/6 = 7435.66  (pval:0.000 prob:1.000)  
    0000            12321     50222     12612         22511.89        3.982 +- 0.018        0.251 +- 0.002  [5 ] Rk Vm F2 Vm Rk
    0001              121     43287     71594          6974.92        0.605 +- 0.003        1.654 +- 0.006  [3 ] Rk Vm Rk
    0002             1221      3376      1664           581.54        2.029 +- 0.035        0.493 +- 0.012  [4 ] Rk Vm Vm Rk
    0003           123321      2866       747          1242.78        3.837 +- 0.072        0.261 +- 0.010  [6 ] Rk Vm F2 F2 Vm Rk
    0004          1233321       164        57            51.81        2.877 +- 0.225        0.348 +- 0.046  [7 ] Rk Vm F2 F2 F2 Vm Rk
    0005             3321        16         5             0.00        3.200 +- 0.800        0.312 +- 0.140  [4 ] Rk Vm F2 F2
    0006               11        10        14             0.00        0.714 +- 0.226        1.400 +- 0.374  [2 ] Rk Rk
    0007           122321         9         1             0.00        9.000 +- 3.000        0.111 +- 0.111  [6 ] Rk Vm F2 Vm Vm Rk
    0008         12333321         9         2             0.00        4.500 +- 1.500        0.222 +- 0.157  [8 ] Rk Vm F2 F2 F2 F2 Vm Rk
    0009       3333333321         7         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [10] Rk Vm F2 F2 F2 F2 F2 F2 F2 F2
    0010                1         6         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [1 ] Rk
    0011          1233221         6         2             0.00        3.000 +- 1.225        0.333 +- 0.236  [7 ] Rk Vm Vm F2 F2 Vm Rk
    0012            33321         3         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] Rk Vm F2 F2 F2
    0013           123221         3     13211         13202.00        0.000 +- 0.000     4403.667 +- 38.313  [6 ] Rk Vm Vm F2 Vm Rk
    0014             1211         3         4             0.00        0.750 +- 0.433        1.333 +- 0.667  [4 ] Rk Rk Vm Rk
    0015            22321         2         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] Rk Vm F2 Vm Vm
    0016              221         2         9             0.00        0.222 +- 0.157        4.500 +- 1.500  [3 ] Rk Vm Vm
    0017        123333321         2         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [9 ] Rk Vm F2 F2 F2 F2 F2 Vm Rk
    0018            12221         2         2             0.00        1.000 +- 0.707        1.000 +- 0.707  [5 ] Rk Vm Vm Vm Rk
    0019          1232221         1         4             0.00        0.250 +- 0.250        4.000 +- 2.000  [7 ] Rk Vm Vm Vm F2 Vm Rk
    .                             100000    100000     44613.94/6 = 7435.66  (pval:0.000 prob:1.000)  
              /tmp/blyth/opticks/evt/tboolean-cubeplanes/torch/1 954f7a41ad772b7c066040935fcbf796 f4549f6a219ea89bae9eeaf2133ddb2e  100000    -1.0000 INTEROP_MODE 
    {u'verbosity': u'0', u'resolution': u'40', u'poly': u'IM', u'ctrl': u'0'}
    [2017-11-20 18:58:58,132] p80886 {/Users/blyth/opticks/ana/tboolean.py:25} INFO - early exit as non-interactive
    simon:issues blyth$ 

