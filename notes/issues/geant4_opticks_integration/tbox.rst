tbox
======

Pyrex cube in mineral oil, good agreement collectively and progressively::

    /Users/blyth/opticks/ana/tbox.py
    [2016-10-26 16:11:46,615] p69656 {/Users/blyth/opticks/ana/tbox.py:248} INFO -  A : BoxInBox/torch/  1 :  20161026-1608 /tmp/blyth/opticks/evt/BoxInBox/torch/1/fdom.npy 
    [2016-10-26 16:11:46,615] p69656 {/Users/blyth/opticks/ana/tbox.py:249} INFO -  B : BoxInBox/torch/ -1 :  20161026-1608 /tmp/blyth/opticks/evt/BoxInBox/torch/-1/fdom.npy 

          seqhis_ana  1:BoxInBox   -1:BoxInBox           c2           ab           ba 
                8ccd        384234       383919             0.13         1.00         1.00  [4 ] TO BT BT SA
                 4cd         72592        72441             0.16         1.00         1.00  [3 ] TO BT AB
                  4d         19692        19871             0.81         0.99         1.01  [2 ] TO AB
                4ccd         15942        16069             0.50         0.99         1.01  [4 ] TO BT BT AB
                 86d          3071         3280             6.88         0.94         1.07  [3 ] TO SC SA
               86ccd          2403         2617             9.12         0.92         1.09  [5 ] TO BT BT SC SA
               8cc6d           685          572            10.16         1.20         0.84  [5 ] TO SC BT BT SA
             8cc6ccd           594          447            20.76         1.33         0.75  [7 ] TO BT BT SC BT BT SA
                 46d           181          198             0.76         0.91         1.09  [3 ] TO SC AB
               46ccd           122          157             4.39         0.78         1.29  [5 ] TO BT BT SC AB
                4c6d            82           39            15.28         2.10         0.48  [4 ] TO SC BT AB
               8c6cd            71           80             0.54         0.89         1.13  [5 ] TO BT SC BT SA
              4c6ccd            68           45             4.68         1.51         0.66  [6 ] TO BT BT SC BT AB
               4cc6d            37           32             0.36         1.16         0.86  [5 ] TO SC BT BT AB
                 8bd            34           28             0.58         1.21         0.82  [3 ] TO BR SA
                866d            31           32             0.02         0.97         1.03  [4 ] TO SC SC SA
             4cc6ccd            31           28             0.15         1.11         0.90  [7 ] TO BT BT SC BT BT AB
               8cbcd            24           29             0.47         0.83         1.21  [5 ] TO BT BR BT SA
              866ccd            28           24             0.31         1.17         0.86  [6 ] TO BT BT SC SC SA
                8b6d            16           23             1.26         0.70         1.44  [4 ] TO SC BR SA
                          500000       500000         3.87 



::

        seqhis_ana_2  1:BoxInBox   -1:BoxInBox           c2           ab           ba 
                  cd        476156       475906             0.07         1.00         1.00  [2 ] TO BT
                  4d         19692        19871             0.81         0.99         1.01  [2 ] TO AB
                  6d          4117         4194             0.71         0.98         1.02  [2 ] TO SC
                  bd            35           29             0.56         1.21         0.83  [2 ] TO BR
                          500000       500000         0.54 
        seqhis_ana_3  1:BoxInBox   -1:BoxInBox           c2           ab           ba 
                 ccd        403450       403337             0.02         1.00         1.00  [3 ] TO BT BT
                 4cd         72592        72441             0.16         1.00         1.00  [3 ] TO BT AB
                  4d         19692        19871             0.81         0.99         1.01  [2 ] TO AB
                 86d          3071         3280             6.88         0.94         1.07  [3 ] TO SC SA
                 c6d           812          651            17.72         1.25         0.80  [3 ] TO SC BT
                 46d           181          198             0.76         0.91         1.09  [3 ] TO SC AB
                 6cd            82           92             0.57         0.89         1.12  [3 ] TO BT SC
                 66d            37           39             0.05         0.95         1.05  [3 ] TO SC SC
                 bcd            32           36             0.24         0.89         1.12  [3 ] TO BT BR
                 8bd            34           28             0.58         1.21         0.82  [3 ] TO BR SA
                 b6d            16           26             2.38         0.62         1.62  [3 ] TO SC BR
                 4bd             1            1             0.00         1.00         1.00  [3 ] TO BR AB
                          500000       500000         2.74 



Switching to pyrex cube in GdDopedLS get big discreps::

       tbox-;tbox-t     ## 380nm (AB/RE dominated)

          seqhis_ana  1:BoxInBox   -1:BoxInBox           c2           ab           ba 
                 85d        223952       159222         10934.91         1.41         0.71  [3 ] TO RE SA
                  4d         99490       100027             1.45         0.99         1.01  [2 ] TO AB
                855d         65565        82045          1839.92         0.80         1.25  [4 ] TO RE RE SA
               8555d         22181        42793          6538.84         0.52         1.93  [5 ] TO RE RE RE SA
                 45d         34225        15909          6691.58         2.15         0.46  [3 ] TO RE AB
              85555d          8116        22152          6508.83         0.37         2.73  [6 ] TO RE RE RE RE SA
                455d         12086         6968          1374.72         1.73         0.58  [4 ] TO RE RE AB
             855555d          3102        11703          4996.77         0.27         3.77  [7 ] TO RE RE RE RE RE SA
               8cc5d          9065         8580            13.33         1.06         0.95  [5 ] TO RE BT BT SA
            8555555d          1248         5832          2967.95         0.21         4.67  [8 ] TO RE RE RE RE RE RE SA
               4555d          4480         3422           141.66         1.31         0.76  [5 ] TO RE RE RE AB
              8cc55d          3249         3996            77.02         0.81         1.23  [6 ] TO RE RE BT BT SA
                  8d             0         3892          3892.00         0.00         0.00  [2 ] TO SA
           85555555d           515         3019          1774.20         0.17         5.86  [9 ] TO RE RE RE RE RE RE RE SA
              85cc5d           131         2884          2513.77         0.05        22.02  [6 ] TO RE BT BT RE SA
                  3d          2328            0          2328.00         0.00         0.00  [2 ] TO MI
          555555555d           203         1954          1421.42         0.10         9.63  [10] TO RE RE RE RE RE RE RE RE RE
              45555d          1886         1680            11.90         1.12         0.89  [6 ] TO RE RE RE RE AB
             8cc555d          1332         1789            66.92         0.74         1.34  [7 ] TO RE RE RE BT BT SA
          855555555d           198         1564          1059.00         0.13         7.90  [10] TO RE RE RE RE RE RE RE RE SA
                          500000       500000       614.07 






         op --mat 0

              domain    refractive_index   absorption_length   scattering_length     reemission_prob      group_velocity
                 380             1.50531             4.13273             23891.6            0.800544                 300
                 480             1.49198             27410.2             58710.2            0.135033                 300


        With torch photons at 380nm (absorption really dominates and reemission_prob 0.8) almost nothing gets to the Pyrex

        seqhis_ana_2  1:BoxInBox   -1:BoxInBox           c2           ab           ba 
                  5d        400432       396182            22.67         1.01         0.99  [2 ] TO RE
                  4d         99490        99731             0.29         1.00         1.00  [2 ] TO AB
                  8d             0         3610          3610.00         0.00         0.00  [2 ] TO SA    ## tbox-;tbox-t --dbgseqhis 8d  
                  cd             0          332           332.00         0.00         0.00  [2 ] TO BT
                  6d            78          140            17.63         0.56         1.79  [2 ] TO SC
                  bd             0            5             0.00         0.00         0.00  [2 ] TO BR


       With wavelengh pushed up to 480nm majority get thru to Pyrex 

       seqhis_ana_2  1:BoxInBox   -1:BoxInBox           c2           ab           ba 
                  cd        494585       494666             0.01         1.00         1.00  [2 ] TO BT
                  4d          3174         3208             0.18         0.99         1.01  [2 ] TO AB
                  6d          1713         1606             3.45         1.07         0.94  [2 ] TO SC
                  5d           466          454             0.16         1.03         0.97  [2 ] TO RE
                  bd            62           66             0.12         0.94         1.06  [2 ] TO BR
                          500000       500000         0.78 






                          500000       500000      1102.14 
        seqhis_ana_3  1:BoxInBox   -1:BoxInBox           c2           ab           ba 
                 85d        223952       159222         10934.91         1.41         0.71  [3 ] TO RE SA   ## SA misbehaving 
                 55d        128042       201563         16399.44         0.64         1.57  [3 ] TO RE RE
                  4d         99490       100027             1.45         0.99         1.01  [2 ] TO AB
                 45d         34225        15909          6691.58         2.15         0.46  [3 ] TO RE AB
                 c5d         10587        17252          1595.68         0.61         1.63  [3 ] TO RE BT
                  8d             0         3892          3892.00         0.00         0.00  [2 ] TO SA
                  3d          2328            0          2328.00         0.00         0.00  [2 ] TO MI
                 65d          1057         1627           121.05         0.65         1.54  [3 ] TO RE SC
                 ccd             0          337           337.00         0.00         0.00  [3 ] TO BT BT
                 35d           240            0           240.00         0.00         0.00  [3 ] TO RE MI
                 56d            48          102            19.44         0.47         2.12  [3 ] TO SC RE
                 4cd             0           25             0.00         0.00         0.00  [3 ] TO BT AB
                 86d            13           18             0.81         0.72         1.38  [3 ] TO SC SA
                 46d            17           10             0.00         1.70         0.59  [3 ] TO SC AB
                 bcd             0            7             0.00         0.00         0.00  [3 ] TO BT BR
                 b5d             1            3             0.00         0.33         3.00  [3 ] TO RE BR
                 c6d             0            3             0.00         0.00         0.00  [3 ] TO SC BT
                 8bd             0            2             0.00         0.00         0.00  [3 ] TO BR SA
                 6bd             0            1             0.00         0.00         0.00  [3 ] TO BR SC
                          500000       500000      3546.78 



Now switching off reemission::

       tbox-;tbox-t --nore


::

          seqhis_ana  1:BoxInBox   -1:BoxInBox           c2           ab           ba 
                8ccd        439393       380275          4263.85         1.16 +- 0.00         0.87 +- 0.00  [4 ] TO BT BT SA
                 4cd         50316        50208             0.12         1.00 +- 0.00         1.00 +- 0.00  [3 ] TO BT AB
               85ccd           168        24640         24140.55         0.01 +- 0.00       146.67 +- 0.93  [5 ] TO BT BT RE SA
              855ccd            61        12420         12238.19         0.00 +- 0.00       203.61 +- 1.83  [6 ] TO BT BT RE RE SA
             8555ccd            23         6435          6366.33         0.00 +- 0.00       279.78 +- 3.49  [7 ] TO BT BT RE RE RE SA
            85555ccd            15         3398          3353.26         0.00 +- 0.00       226.53 +- 3.89  [8 ] TO BT BT RE RE RE RE SA
                  4d          3174         3208             0.18         0.99 +- 0.02         1.01 +- 0.02  [2 ] TO AB
                4ccd          2772         2755             0.05         1.01 +- 0.02         0.99 +- 0.02  [4 ] TO BT BT AB
               45ccd            50         1892          1747.15         0.03 +- 0.00        37.84 +- 0.87  [5 ] TO BT BT RE AB
           855555ccd             5         1694          1679.06         0.00 +- 0.00       338.80 +- 8.23  [9 ] TO BT BT RE RE RE RE RE SA
             8cc5ccd            28         1326          1244.32         0.02 +- 0.00        47.36 +- 1.30  [7 ] TO BT BT RE BT BT SA
                 86d          1307         1187             5.77         1.10 +- 0.03         0.91 +- 0.03  [3 ] TO SC SA
               86ccd          1153         1058             4.08         1.09 +- 0.03         0.92 +- 0.03  [5 ] TO BT BT SC SA
          5555555ccd             3         1068          1059.03         0.00 +- 0.00       356.00 +- 10.89  [10] TO BT BT RE RE RE RE RE RE RE
              455ccd            24         1064           994.12         0.02 +- 0.00        44.33 +- 1.36  [6 ] TO BT BT RE RE AB
          8555555ccd             1          884           881.00         0.00 +- 0.00       884.00 +- 29.73  [10] TO BT BT RE RE RE RE RE RE SA
            8cc55ccd            14          599           558.28         0.02 +- 0.01        42.79 +- 1.75  [8 ] TO BT BT RE RE BT BT SA
             4555ccd             9          489           462.65         0.02 +- 0.01        54.33 +- 2.46  [7 ] TO BT BT RE RE RE AB
            85cc5ccd             5          425           410.23         0.01 +- 0.01        85.00 +- 4.12  [8 ] TO BT BT RE BT BT RE SA
               8cc6d           357          194            48.22         1.84 +- 0.10         0.54 +- 0.04  [5 ] TO SC BT BT SA
                          500000       500000      1024.65 


::

        seqhis_ana_2  1:BoxInBox   -1:BoxInBox           c2           ab           ba 
                  cd        494585       494666             0.01         1.00 +- 0.00         1.00 +- 0.00  [2 ] TO BT
                  4d          3174         3208             0.18         0.99 +- 0.02         1.01 +- 0.02  [2 ] TO AB
                  6d          1713         1606             3.45         1.07 +- 0.03         0.94 +- 0.02  [2 ] TO SC
                  5d           466          454             0.16         1.03 +- 0.05         0.97 +- 0.05  [2 ] TO RE
                  bd            62           66             0.12         0.94 +- 0.12         1.06 +- 0.13  [2 ] TO BR
                          500000       500000         0.78 
        seqhis_ana_3  1:BoxInBox   -1:BoxInBox           c2           ab           ba 
                 ccd        444113       444311             0.04         1.00 +- 0.00         1.00 +- 0.00  [3 ] TO BT BT
                 4cd         50316        50208             0.12         1.00 +- 0.00         1.00 +- 0.00  [3 ] TO BT AB
                  4d          3174         3208             0.18         0.99 +- 0.02         1.01 +- 0.02  [2 ] TO AB
                 86d          1307         1187             5.77         1.10 +- 0.03         0.91 +- 0.03  [3 ] TO SC SA   <<  more SA from Opticks
                 c6d           385          244            31.61         1.58 +- 0.08         0.63 +- 0.04  [3 ] TO SC BT
                 55d           200          221             1.05         0.90 +- 0.06         1.10 +- 0.07  [3 ] TO RE RE
                 85d           178          154             1.73         1.16 +- 0.09         0.87 +- 0.07  [3 ] TO RE SA
                 56d             0          154           154.00         0.00 +- 0.00         0.00 +- 0.00  [3 ] TO SC RE
                 6cd            90           80             0.59         1.12 +- 0.12         0.89 +- 0.10  [3 ] TO BT SC
                 bcd            66           67             0.01         0.99 +- 0.12         1.02 +- 0.12  [3 ] TO BT BR
                 8bd            62           62             0.00         1.00 +- 0.13         1.00 +- 0.13  [3 ] TO BR SA
                 c5d            45           58             1.64         0.78 +- 0.12         1.29 +- 0.17  [3 ] TO RE BT
                 45d            39           16             9.62         2.44 +- 0.39         0.41 +- 0.10  [3 ] TO RE AB
                 46d            11            9             0.00         1.22 +- 0.37         0.82 +- 0.27  [3 ] TO SC AB
                 b6d             5            8             0.00         0.62 +- 0.28         1.60 +- 0.57  [3 ] TO SC BR
                 66d             5            4             0.00         1.25 +- 0.56         0.80 +- 0.40  [3 ] TO SC SC
                 65d             2            3             0.00         0.67 +- 0.47         1.50 +- 0.87  [3 ] TO RE SC
                 4bd             0            3             0.00         0.00 +- 0.00         0.00 +- 0.00  [3 ] TO BR AB
                 b5d             2            2             0.00         1.00 +- 0.71         1.00 +- 0.71  [3 ] TO RE BR
                 cbd             0            1             0.00         0.00 +- 0.00         0.00 +- 0.00  [3 ] TO BR BT
                          500000       500000        15.87 
        seqhis_ana_4  1:BoxInBox   -1:BoxInBox           c2           ab           ba 
                8ccd        439393       380275          4263.85         1.16 +- 0.00         0.87 +- 0.00  [4 ] TO BT BT SA   << more SA from Opticks
                5ccd           445        59809         58487.15         0.01 +- 0.00       134.40 +- 0.55  [4 ] TO BT BT RE
                 4cd         50316        50208             0.12         1.00 +- 0.00         1.00 +- 0.00  [3 ] TO BT AB
                  4d          3174         3208             0.18         0.99 +- 0.02         1.01 +- 0.02  [2 ] TO AB
                4ccd          2772         2755             0.05         1.01 +- 0.02         0.99 +- 0.02  [4 ] TO BT BT AB
                6ccd          1502         1472             0.30         1.02 +- 0.03         0.98 +- 0.03  [4 ] TO BT BT SC
                 86d          1307         1187             5.77         1.10 +- 0.03         0.91 +- 0.03  [3 ] TO SC SA
 

::

    2016-10-26 18:13:55.217 WARN  [3747194] [GSurfaceLib::dump@658]         NearOutOutPiperSurface ( 42,  0,  3,100) 
    2016-10-26 18:13:55.217 WARN  [3747194] [GSurfaceLib::dump@658]            LegInDeadTubSurface ( 43,  0,  3,100) 
    2016-10-26 18:13:55.217 WARN  [3747194] [GSurfaceLib::dump@658]           perfectDetectSurface ( 44,  1,  1,100) 
    2016-10-26 18:13:55.217 WARN  [3747194] [GSurfaceLib::dump@658]           perfectAbsorbSurface ( 45,  1,  1,100) 
    2016-10-26 18:13:55.217 WARN  [3747194] [GSurfaceLib::dump@658]         perfectSpecularSurface ( 46,  1,  1,100) 
    2016-10-26 18:13:55.217 WARN  [3747194] [GSurfaceLib::dump@658]          perfectDiffuseSurface ( 47,  1,  1,100)  


