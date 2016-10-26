tboxlaser
=============

fixpol beam down onto a cube of scintillator within mineral oil box.

::

     27 tboxlaser--(){
     28     local cmdline=$*
     29     local tag=$(tboxlaser-tag)
     30     local photons=500000
     31     
     32     #local nm=380
     33     local nm=480
     34     
     35     local m1=MineralOil
     36     #local m2=Pyrex
     37     local m2=GdDopedLS
     38     
     39     ## beam in -Z direction, fixpol +Y
     40     
     41     local torch_config=(
     42                  type=point
     43                  mode=fixpol
     44                  polarization=0,1,0
     45                  source=0,0,299
     46                  target=0,0,0
     47                  photons=$photons
     48                  material=$m1
     49                  wavelength=$nm
     50                  weight=1.0
     51                  time=0.1
     52                  zenithazimuth=0,1,0,1
     53                  radius=0
     54                )   
     55     local test_config=(
     56                  mode=BoxInBox
     57                  analytic=1
     58                  
     59                  shape=box
     60                  boundary=Rock//perfectAbsorbSurface/$m1
     61                  parameters=0,0,0,300
     62                  
     63                  shape=box
     64                  boundary=$m1///$m2
     65                  parameters=0,0,0,100
     66                    ) 
     67                    
     68     op.sh \
     69        --test --testconfig "$(join _ ${test_config[@]})" \
     70        --torch --torchconfig "$(join _ ${torch_config[@]})" \
     71        --animtimemax 10 \
     72        --timemax 10 \
     73        --cat boxlaser --tag $tag --save  \
     74        --eye 0.5,0.5,0.0 \
     75        $* 
     76 }      
     77 tboxlaser-args(){  echo  --tag $(tboxlaser-tag) --det boxlaser --src torch ; }
     78 tboxlaser-a(){     tbox.py  $(tboxlaser-args) $* ; } 
     79 tboxlaser-i(){     ipython -i $(which tbox.py) --  $(tboxlaser-args) $* ; }
     80 tboxlaser-t()
     81 {
     82     tboxlaser-
     83     tboxlaser-- --okg4 --compute $*
     84 }






check fx machinery
-----------------------

Check with constants for properties::

    tboxlaser-tfx()
    {
        tboxlaser-t  --fxabconfig 10000 --fxab --fxscconfig 10000 --fxsc --fxreconfig 0.5 --fxre $*
    }


::

    [2016-10-26 20:28:48,331] p85486 {/Users/blyth/opticks/ana/tboxlaser.py:40} INFO -  a : boxlaser/torch/  1 :  20161026-2028 /tmp/blyth/opticks/evt/boxlaser/torch/1/fdom.npy 
    [2016-10-26 20:28:48,331] p85486 {/Users/blyth/opticks/ana/tboxlaser.py:41} INFO -  b : boxlaser/torch/ -1 :  20161026-2028 /tmp/blyth/opticks/evt/boxlaser/torch/-1/fdom.npy 
              seqhis_ana  1:boxlaser   -1:boxlaser           c2           ab           ba 
                    8ccd        443899       443465             0.21         1.00 +- 0.00         1.00 +- 0.00  [4 ] TO BT BT SA
                      4d          9637         9777             1.01         0.99 +- 0.01         1.01 +- 0.01  [2 ] TO AB
                    4ccd          8934         8976             0.10         1.00 +- 0.01         1.00 +- 0.01  [4 ] TO BT BT AB
                   8c6cd          8610         8410             2.35         1.02 +- 0.01         0.98 +- 0.01  [5 ] TO BT SC BT SA
                     86d          7090         7478            10.33         0.95 +- 0.01         1.05 +- 0.01  [3 ] TO SC SA
                   86ccd          6426         6918            18.14         0.93 +- 0.01         1.08 +- 0.01  [5 ] TO BT BT SC SA
                     4cd          4762         4767             0.00         1.00 +- 0.01         1.00 +- 0.01  [3 ] TO BT AB
                   8c5cd          4383         4381             0.00         1.00 +- 0.02         1.00 +- 0.02  [5 ] TO BT RE BT SA
                   8cc6d          2051         1728            27.61         1.19 +- 0.03         0.84 +- 0.02  [5 ] TO SC BT BT SA
                 8cc6ccd          1968         1603            37.31         1.23 +- 0.03         0.81 +- 0.02  [7 ] TO BT BT SC BT BT SA
                   46ccd           159          261            24.77         0.61 +- 0.05         1.64 +- 0.10  [5 ] TO BT BT SC AB
                   4c6cd           226          249             1.11         0.91 +- 0.06         1.10 +- 0.07  [5 ] TO BT SC BT AB
                     46d           195          222             1.75         0.88 +- 0.06         1.14 +- 0.08  [3 ] TO SC AB
                  86c6cd           172          211             3.97         0.82 +- 0.06         1.23 +- 0.08  [6 ] TO BT SC BT SC SA
                  866ccd           160          197             3.83         0.81 +- 0.06         1.23 +- 0.09  [6 ] TO BT BT SC SC SA
                    866d           156          196             4.55         0.80 +- 0.06         1.26 +- 0.09  [4 ] TO SC SC SA
                   4c5cd            99          109             0.48         0.91 +- 0.09         1.10 +- 0.11  [5 ] TO BT RE BT AB
                  8cb6cd            64          102             8.70         0.63 +- 0.08         1.59 +- 0.16  [6 ] TO BT SC BR BT SA
                  8c66cd            93           96             0.05         0.97 +- 0.10         1.03 +- 0.11  [6 ] TO BT SC SC BT SA
                  86c5cd            81           95             1.11         0.85 +- 0.09         1.17 +- 0.12  [6 ] TO BT RE BT SC SA
                              500000       500000         4.78 





Pretty good, after adopt fixpol::


    [2016-10-26 19:20:50,882] p77606 {/Users/blyth/opticks/ana/tboxlaser.py:40} INFO -  a : boxlaser/torch/  1 :  20161026-1920 /tmp/blyth/opticks/evt/boxlaser/torch/1/fdom.npy 
    [2016-10-26 19:20:50,882] p77606 {/Users/blyth/opticks/ana/tboxlaser.py:41} INFO -  b : boxlaser/torch/ -1 :  20161026-1920 /tmp/blyth/opticks/evt/boxlaser/torch/-1/fdom.npy 
              seqhis_ana  1:boxlaser   -1:boxlaser           c2           ab           ba 
                    8ccd        483237       483157             0.01         1.00 +- 0.00         1.00 +- 0.00  [4 ] TO BT BT SA
                      4d          4070         4143             0.65         0.98 +- 0.02         1.02 +- 0.02  [2 ] TO AB
                    4ccd          3953         3964             0.02         1.00 +- 0.02         1.00 +- 0.02  [4 ] TO BT BT AB
                     4cd          3121         3080             0.27         1.01 +- 0.02         0.99 +- 0.02  [3 ] TO BT AB
                   8c6cd          1622         1608             0.06         1.01 +- 0.03         0.99 +- 0.02  [5 ] TO BT SC BT SA
                   86ccd          1264         1375             4.67         0.92 +- 0.03         1.09 +- 0.03  [5 ] TO BT BT SC SA
                     86d          1303         1327             0.22         0.98 +- 0.03         1.02 +- 0.03  [3 ] TO SC SA

                   8cc6d           392          338             3.99         1.16 +- 0.06         0.86 +- 0.05  [5 ] TO SC BT BT SA
                 8cc6ccd           375          313             5.59         1.20 +- 0.06         0.83 +- 0.05  [7 ] TO BT BT SC BT BT SA
                   8c5cd           245          369            25.04         0.66 +- 0.04         1.51 +- 0.08  [5 ] TO BT RE BT SA
                  8c55cd            72           82             0.65         0.88 +- 0.10         1.14 +- 0.13  [6 ] TO BT RE RE BT SA
                    45cd            56           20            17.05         2.80 +- 0.37         0.36 +- 0.08  [4 ] TO BT RE AB
                 8c555cd            52            9            30.31         5.78 +- 0.80         0.17 +- 0.06  [7 ] TO BT RE RE RE BT SA
                     8bd            32           40             0.89         0.80 +- 0.14         1.25 +- 0.20  [3 ] TO BR SA
                   8cbcd            35           25             1.67         1.40 +- 0.24         0.71 +- 0.14  [5 ] TO BT BR BT SA
                   46ccd            12           24             4.00         0.50 +- 0.14         2.00 +- 0.41  [5 ] TO BT BT SC AB
                8c5555cd            21            1             0.00        21.00 +- 4.58         0.05 +- 0.05  [8 ] TO BT RE RE RE RE BT SA
                   4c6cd            19           20             0.03         0.95 +- 0.22         1.05 +- 0.24  [5 ] TO BT SC BT AB
                  8cb6cd            14           18             0.50         0.78 +- 0.21         1.29 +- 0.30  [6 ] TO BT SC BR BT SA
                   455cd            17            4             0.00         4.25 +- 1.03         0.24 +- 0.12  [5 ] TO BT RE RE AB
                              500000       500000         5.31 


        seqhis_ana_1  1:boxlaser   -1:boxlaser           c2           ab           ba 
                   d        500000       500000             0.00         1.00 +- 0.00         1.00 +- 0.00  [1 ] TO
                          500000       500000         0.00 
        seqhis_ana_2  1:boxlaser   -1:boxlaser           c2           ab           ba 
                  cd        494185       494126             0.00         1.00 +- 0.00         1.00 +- 0.00  [2 ] TO BT
                  4d          4070         4143             0.65         0.98 +- 0.02         1.02 +- 0.02  [2 ] TO AB
                  6d          1713         1691             0.14         1.01 +- 0.02         0.99 +- 0.02  [2 ] TO SC
                  bd            32           40             0.89         0.80 +- 0.14         1.25 +- 0.20  [2 ] TO BR
                          500000       500000         0.42 
        seqhis_ana_3  1:boxlaser   -1:boxlaser           c2           ab           ba 
                 ccd        488862       488848             0.00         1.00 +- 0.00         1.00 +- 0.00  [3 ] TO BT BT
                  4d          4070         4143             0.65         0.98 +- 0.02         1.02 +- 0.02  [2 ] TO AB
                 4cd          3121         3080             0.27         1.01 +- 0.02         0.99 +- 0.02  [3 ] TO BT AB
                 6cd          1667         1670             0.00         1.00 +- 0.02         1.00 +- 0.02  [3 ] TO BT SC
                 86d          1303         1327             0.22         0.98 +- 0.03         1.02 +- 0.03  [3 ] TO SC SA
                 5cd           499          503             0.02         0.99 +- 0.04         1.01 +- 0.04  [3 ] TO BT RE
                 c6d           401          345             4.20         1.16 +- 0.06         0.86 +- 0.05  [3 ] TO SC BT
                 8bd            32           40             0.89         0.80 +- 0.14         1.25 +- 0.20  [3 ] TO BR SA
                 bcd            36           25             1.98         1.44 +- 0.24         0.69 +- 0.14  [3 ] TO BT BR
                 46d             7           12             0.00         0.58 +- 0.22         1.71 +- 0.49  [3 ] TO SC AB
                 66d             2            7             0.00         0.29 +- 0.20         3.50 +- 1.32  [3 ] TO SC SC
                          500000       500000         0.91 

        seqhis_ana_4  1:boxlaser   -1:boxlaser           c2           ab           ba 
                8ccd        483237       483157             0.01         1.00 +- 0.00         1.00 +- 0.00  [4 ] TO BT BT SA
                  4d          4070         4143             0.65         0.98 +- 0.02         1.02 +- 0.02  [2 ] TO AB
                4ccd          3953         3964             0.02         1.00 +- 0.02         1.00 +- 0.02  [4 ] TO BT BT AB
                 4cd          3121         3080             0.27         1.01 +- 0.02         0.99 +- 0.02  [3 ] TO BT AB
                6ccd          1671         1727             0.92         0.97 +- 0.02         1.03 +- 0.02  [4 ] TO BT BT SC
                c6cd          1646         1638             0.02         1.00 +- 0.02         1.00 +- 0.02  [4 ] TO BT SC BT
                 86d          1303         1327             0.22         0.98 +- 0.03         1.02 +- 0.03  [3 ] TO SC SA
                cc6d           398          342             4.24         1.16 +- 0.06         0.86 +- 0.05  [4 ] TO SC BT BT
                c5cd           247          377            27.08         0.66 +- 0.04         1.53 +- 0.08  [4 ] TO BT RE BT
                55cd           192          101            28.26         1.90 +- 0.14         0.53 +- 0.05  [4 ] TO BT RE RE
                45cd            56           20            17.05         2.80 +- 0.37         0.36 +- 0.08  [4 ] TO BT RE AB
                 8bd            32           40             0.89         0.80 +- 0.14         1.25 +- 0.20  [3 ] TO BR SA
                cbcd            35           25             1.67         1.40 +- 0.24         0.71 +- 0.14  [4 ] TO BT BR BT
                b6cd            14           18             0.50         0.78 +- 0.21         1.29 +- 0.30  [4 ] TO BT SC BR
                 46d             7           12             0.00         0.58 +- 0.22         1.71 +- 0.49  [3 ] TO SC AB
                46cd             4           11             0.00         0.36 +- 0.18         2.75 +- 0.83  [4 ] TO BT SC AB
                b5cd             2            5             0.00         0.40 +- 0.28         2.50 +- 1.12  [4 ] TO BT RE BR
                866d             2            5             0.00         0.40 +- 0.28         2.50 +- 1.12  [4 ] TO SC SC SA
                66cd             2            3             0.00         0.67 +- 0.47         1.50 +- 0.87  [4 ] TO BT SC SC
                4c6d             0            3             0.00         0.00 +- 0.00         0.00 +- 0.00  [4 ] TO SC BT AB
                          500000       500000         5.84 
        seqhis_ana_5  1:boxlaser   -1:boxlaser           c2           ab           ba 
                8ccd        483237       483157             0.01         1.00 +- 0.00         1.00 +- 0.00  [4 ] TO BT BT SA
                  4d          4070         4143             0.65         0.98 +- 0.02         1.02 +- 0.02  [2 ] TO AB
                4ccd          3953         3964             0.02         1.00 +- 0.02         1.00 +- 0.02  [4 ] TO BT BT AB
                 4cd          3121         3080             0.27         1.01 +- 0.02         0.99 +- 0.02  [3 ] TO BT AB
               8c6cd          1622         1608             0.06         1.01 +- 0.03         0.99 +- 0.02  [5 ] TO BT SC BT SA
               86ccd          1264         1375             4.67         0.92 +- 0.03         1.09 +- 0.03  [5 ] TO BT BT SC SA
                 86d          1303         1327             0.22         0.98 +- 0.03         1.02 +- 0.03  [3 ] TO SC SA
               8cc6d           392          338             3.99         1.16 +- 0.06         0.86 +- 0.05  [5 ] TO SC BT BT SA
               c6ccd           384          320             5.82         1.20 +- 0.06         0.83 +- 0.05  [5 ] TO BT BT SC BT
               8c5cd           245          369            25.04         0.66 +- 0.04         1.51 +- 0.08  [5 ] TO BT RE BT SA
               555cd            98           13            65.09         7.54 +- 0.76         0.13 +- 0.04  [5 ] TO BT RE RE RE
               c55cd            76           83             0.31         0.92 +- 0.11         1.09 +- 0.12  [5 ] TO BT RE RE BT
                45cd            56           20            17.05         2.80 +- 0.37         0.36 +- 0.08  [4 ] TO BT RE AB
                 8bd            32           40             0.89         0.80 +- 0.14         1.25 +- 0.20  [3 ] TO BR SA
               8cbcd            35           25             1.67         1.40 +- 0.24         0.71 +- 0.14  [5 ] TO BT BR BT SA
               46ccd            12           24             4.00         0.50 +- 0.14         2.00 +- 0.41  [5 ] TO BT BT SC AB
               4c6cd            19           20             0.03         0.95 +- 0.22         1.05 +- 0.24  [5 ] TO BT SC BT AB
               cb6cd            14           18             0.50         0.78 +- 0.21         1.29 +- 0.30  [5 ] TO BT SC BR BT
               455cd            17            4             0.00         4.25 +- 1.03         0.24 +- 0.12  [5 ] TO BT RE RE AB
                 46d             7           12             0.00         0.58 +- 0.22         1.71 +- 0.49  [3 ] TO SC AB
                          500000       500000         7.24 





