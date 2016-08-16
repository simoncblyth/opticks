PMT Test
==========

Shooting PMT and checking numbers in ballpark...

::

    ggv-pmt-test(){
       type $FUNCNAME

       local torch_config=(
                     type=disclin
                     photons=500000
                     wavelength=380 
                     frame=1
                     source=0,0,300
                     target=0,0,0
                     radius=100
                     zenithazimuth=0,1,0,1
                     material=Vacuum
                   )    
         

       #  slice=2:3  PMT front face only

       local test_config=(
                     mode=PmtInBox
                     boundary=Rock//perfectAbsorbSurface/MineralOil
                     dimensions=300,0,0,0
                     shape=B
                     analytic=1
                       )    

       ggv \
           --test --testconfig "$(join _ ${test_config[@]})" \
           --torch --torchconfig "$(join _ ${torch_config[@]})" \
           --animtimemax 10 \ 
           --eye 0.5,0.5,0.0 \
           $*   

    }



With Torch default 500nm (qe 0.105)::

    [2015-Nov-17 19:49:32.290014]:info: App::indexSequence m_seqhis
    [2015-Nov-17 19:49:32.290553]:info: 
        0    431969     0.864                      8cd                               TO BT SA 
        1     50982     0.102                      7cd                               TO BT SD   # 0.102/(0.864+0.102) = 0.10559
        2     10416     0.021                     8ccd                            TO BT BT SA   # edge grazer
        3      3905     0.008                       4d                                  TO AB 
        4      1139     0.002                      86d                               TO SC SA 
        5      1063     0.002                      4cd                               TO BT AB 
        6       155     0.000                     8c6d                            TO SC BT SA 
        7       140     0.000                     4ccd                            TO BT BT AB 
        8        95     0.000                      8bd                               TO BR SA 
        9        45     0.000                    86ccd                         TO BT BT SC SA 
       10        24     0.000                     7c6d                            TO SC BT SD 
       11        13     0.000                      46d                               TO SC AB 
       12        13     0.000                    8cc6d                         TO SC BT BT SA 
       13        13     0.000                 8cbccbcd                TO BT BR BT BT BR BT SA 
       14         6     0.000                8cbccbbcd             TO BT BR BR BT BT BR BT SA 
       15         3     0.000                     866d                            TO SC SC SA 
       16         3     0.000                 8cbc6ccd                TO BT BT SC BT BR BT SA 
       17         2     0.000                     4c6d                            TO SC BT AB 
       18         2     0.000                     8b6d                            TO SC BR SA 
       19         2     0.000                   8cbbcd                      TO BT BR BR BT SA 
       20         2     0.000                  8cc6ccd                   TO BT BT SC BT BT SA 
       21         2     0.000                 8cccbbcd                TO BT BR BR BT BT BT SA 
       22         1     0.000                    46ccd                         TO BT BT SC AB 
       23         1     0.000               86cbccbbcd          TO BT BR BR BT BT BR BT SC SA 
       24         1     0.000               8ccccc6ccd          TO BT BT SC BT BT BT BT BT SA 
       25         1     0.000               cbccbc6ccd          TO BT BT SC BT BR BT BT BR BT 
       26         1     0.000               cbcccc6ccd          TO BT BT SC BT BT BT BT BR BT 
       27         1     0.000               cccccc6ccd          TO BT BT SC BT BT BT BT BT BT 
      TOT    500000

After change wavelength to 380nm (qe 0.24)::

    [2015-Nov-17 20:34:04.988286]:info: 
        0    351525     0.703                      8cd                               TO BT SA 
        1    111249     0.223                      7cd                               TO BT SD  #  0.223/(0.703+0.223) = 0.2408
        2     18047     0.036                       4d                                  TO AB 
        3     11686     0.023                     8ccd                            TO BT BT SA   # edge grazer
        4      3040     0.006                      86d                               TO SC SA 
        5      1727     0.003                      4cd                               TO BT AB 
        6       856     0.002                     4ccd                            TO BT BT AB 
        7       749     0.001                      8bd                               TO BR SA 
        8       374     0.001                     8c6d                            TO SC BT SA 
        9       187     0.000                      46d                               TO SC AB 
       10       139     0.000                    86ccd                         TO BT BT SC SA 
       11       107     0.000                   8cbbcd                      TO BT BR BR BT SA 
       12        99     0.000                     7c6d                            TO SC BT SD 
       13        53     0.000                      4bd                               TO BR AB 
       14        33     0.000                     866d                            TO SC SC SA 
       15        30     0.000                    8cc6d                         TO SC BT BT SA 
       16        22     0.000                     8b6d                            TO SC BR SA 
       17        12     0.000                 8cbc6ccd                TO BT BT SC BT BR BT SA 
       18        11     0.000                   4cbbcd                      TO BT BR BR BT AB 
       19        10     0.000                     86bd                            TO BR SC SA 
       20         7     0.000                    46ccd                         TO BT BT SC AB 
       21         6     0.000                     4c6d                            TO SC BT AB 
       22         5     0.000                    4bbcd                         TO BT BR BR AB 
       23         2     0.000                     4bcd                            TO BT BR AB 
       24         2     0.000                    4cc6d                         TO SC BT BT AB 
       25         2     0.000                    7c66d                         TO SC SC BT SD 
       26         2     0.000                    8c66d                         TO SC SC BT SA 
       27         2     0.000                   8b6ccd                      TO BT BT SC BR SA 
       28         2     0.000                   8cbc6d                      TO SC BT BR BT SA 
       29         2     0.000                  86cbbcd                   TO BT BR BR BT SC SA 
       30         2     0.000                  8cc6ccd                   TO BT BT SC BT BT SA 
       31         2     0.000               8ccccc6ccd          TO BT BT SC BT BT BT BT BT SA 
      TOT    499992





Dump the sensor surface::

    ggv --surf 6

    [2015-Nov-17 20:29:49.173433]:info:  (  6,  0,  3,100) GPropertyMap<T>::  6        surface s: GOpticalSurface  type 0 model 1 finish 3 value     1 lvPmtHemiCathodeSensorSurface k:detect absorb reflect_specular reflect_diffuse lvPmtHemiCathodeSensorSurface
                  domain              detect              absorb    reflect_specular     reflect_diffuse
                      60                   0                   1                   0                   0
                      80                   0                   1                   0                   0
                     100                   0                   1                   0                   0
                     120                   0                   1                   0                   0
                     140         2.49859e-05            0.999975                   0                   0
                     160         5.00012e-05             0.99995                   0                   0
                     180         7.50165e-05            0.999925                   0                   0
                     200         0.000100477              0.9999                   0                   0
                     220         0.000475524            0.999524                   0                   0
                     240         0.000850572            0.999149                   0                   0
                     260          0.00242326            0.997577                   0                   0
                     280          0.00478916            0.995211                   0                   0
                     300           0.0315071            0.968493                   0                   0
                     320            0.154862            0.845138                   0                   0
                     340             0.21478             0.78522                   0                   0
                     360            0.235045            0.764955                   0                   0
                     380                0.24                0.76                   0                   0
                     400            0.225015            0.774985                   0                   0
                     420            0.214631            0.785369                   0                   0
                     440             0.19469             0.80531                   0                   0
                     460            0.170217            0.829783                   0                   0
                     480            0.139298            0.860702                   0                   0
                     500            0.104962            0.895038                   0                   0
                     520           0.0849148            0.915085                   0                   0
                     540           0.0652552            0.934745                   0                   0
                     560           0.0460444            0.953956                   0                   0
                     580            0.027468            0.972532                   0                   0
                     600           0.0190265            0.980974                   0                   0
                     620          0.00998792            0.990012                   0                   0




