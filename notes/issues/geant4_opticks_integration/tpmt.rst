tpmt
=======

::

    simon:ana blyth$ tpmt.py 
    /Users/blyth/opticks/ana/tpmt.py
    writing opticks environment to /tmp/blyth/opticks/opticks_env.bash 
    [2016-10-26 15:56:52,005] p68777 {/Users/blyth/opticks/ana/tpmt.py:130} INFO - tag 10 src torch det PmtInBox c2max 2.0  
    [2016-10-26 15:56:52,354] p68777 {/Users/blyth/opticks/ana/tpmt.py:146} INFO -  a : PmtInBox/torch/ 10 :  20161026-1555 /tmp/blyth/opticks/evt/PmtInBox/torch/10/fdom.npy 
    [2016-10-26 15:56:52,354] p68777 {/Users/blyth/opticks/ana/tpmt.py:147} INFO -  b : PmtInBox/torch/-10 :  20161026-1555 /tmp/blyth/opticks/evt/PmtInBox/torch/-10/fdom.npy 
    ...

         seqhis_ana  10:PmtInBox   -10:PmtInBox           c2           ab           ba 
                 8cd         67948        68077             0.12         1.00         1.00  [3 ] TO BT SA
                 7cd         21648        21343             2.16         1.01         0.99  [3 ] TO BT SD
                8ccd          4581         4649             0.50         0.99         1.01  [4 ] TO BT BT SA
                  4d          3794         3882             1.01         0.98         1.02  [2 ] TO AB
                 86d           640          678             1.10         0.94         1.06  [3 ] TO SC SA
                 4cd           444          433             0.14         1.03         0.98  [3 ] TO BT AB
                4ccd           350          354             0.02         0.99         1.01  [4 ] TO BT BT AB
                 8bd           283          317             1.93         0.89         1.12  [3 ] TO BR SA
                8c6d            81           55             4.97         1.47         0.68  [4 ] TO SC BT SA
               86ccd            51           61             0.89         0.84         1.20  [5 ] TO BT BT SC SA
              8cbbcd            36           43             0.62         0.84         1.19  [6 ] TO BT BR BR BT SA
                 46d            40           28             2.12         1.43         0.70  [3 ] TO SC AB
                 4bd            28           17             2.69         1.65         0.61  [3 ] TO BR AB
                7c6d            20           14             1.06         1.43         0.70  [4 ] TO SC BT SD
            8cbc6ccd             9            4             0.00         2.25         0.44  [8 ] TO BT BT SC BT BR BT SA
                866d             8            7             0.00         1.14         0.88  [4 ] TO SC SC SA
               8cc6d             7            7             0.00         1.00         1.00  [5 ] TO SC BT BT SA
               46ccd             1            7             0.00         0.14         7.00  [5 ] TO BT BT SC AB
                86bd             6            1             0.00         6.00         0.17  [4 ] TO BR SC SA
          cbccbbbbcd             4            1             0.00         4.00         0.25  [10] TO BT BR BR BR BR BT BT BR BT
                          100000       100000         1.38 
        seqhis_ana_2  10:PmtInBox   -10:PmtInBox           c2           ab           ba 
                  cd         95086        94986             0.05         1.00         1.00  [2 ] TO BT            ## excellent progressive mask agreement too
                  4d          3794         3882             1.01         0.98         1.02  [2 ] TO AB
                  6d           802          797             0.02         1.01         0.99  [2 ] TO SC
                  bd           318          335             0.44         0.95         1.05  [2 ] TO BR
                          100000       100000         0.38 
        seqhis_ana_3  10:PmtInBox   -10:PmtInBox           c2           ab           ba 
                 8cd         67948        68077             0.12         1.00         1.00  [3 ] TO BT SA
                 7cd         21648        21343             2.16         1.01         0.99  [3 ] TO BT SD
                 ccd          5000         5082             0.67         0.98         1.02  [3 ] TO BT BT
                  4d          3794         3882             1.01         0.98         1.02  [2 ] TO AB
                 86d           640          678             1.10         0.94         1.06  [3 ] TO SC SA
                 4cd           444          433             0.14         1.03         0.98  [3 ] TO BT AB
                 8bd           283          317             1.93         0.89         1.12  [3 ] TO BR SA
                 c6d           109           76             5.89         1.43         0.70  [3 ] TO SC BT
                 bcd            46           51             0.26         0.90         1.11  [3 ] TO BT BR
                 46d            40           28             2.12         1.43         0.70  [3 ] TO SC AB
                 4bd            28           17             2.69         1.65         0.61  [3 ] TO BR AB
                 66d            10           10             0.00         1.00         1.00  [3 ] TO SC SC
                 6bd             7            1             0.00         7.00         0.14  [3 ] TO BR SC
                 b6d             3            5             0.00         0.60         1.67  [3 ] TO SC BR






