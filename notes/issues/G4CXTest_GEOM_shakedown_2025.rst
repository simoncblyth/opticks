G4CXTest_GEOM_shakedown_2025
===============================



Issue 2 : FIXED : where is the C++ chi2 detail ? script level updates needed
------------------------------------------------------------------------------

::

    P[blyth@localhost opticks]$ ~/o/G4CXTest_GEOM.sh chi2
                       BASH_SOURCE : /data/blyth/junotop/opticks/g4cx/tests/../../sysrap/tests/sseq_index_test.sh 
                              SDIR : /data/blyth/junotop/opticks/sysrap/tests 
                               TMP : /data/blyth/opticks 
                        EXECUTABLE : G4CXTest 
                           VERSION : 98 
                              BASE : /data/blyth/opticks/GEOM/J_2025jan08 
                              GEOM : J_2025jan08 
                            LOGDIR : /data/blyth/opticks/GEOM/J_2025jan08/G4CXTest/ALL98 
                             AFOLD : /data/blyth/opticks/GEOM/J_2025jan08/G4CXTest/ALL98/A000 
                             BFOLD : /data/blyth/opticks/GEOM/J_2025jan08/G4CXTest/ALL98/B000 

                             ^^^^^^^^^^ WRONG FOLD :THE SUB-SCRIPT IS NOW AFOLD/BFOLD STEERED FROM ABOVE WHEN AFOLD/BFOLD DEFINED ^^^^^^

                              FOLD : /data/blyth/opticks/sseq_index_test 
    a_path $AFOLD/seq.npy - a_seq -
    b_path $BFOLD/seq.npy - b_seq -
    f

    CMDLINE:/data/blyth/junotop/opticks/sysrap/tests/sseq_index_test.py
    f.base:/data/blyth/opticks/sseq_index_test

      : f.sseq_index_ab_chi2                               :                 (4,) : 54 days, 0:41:45.773076 

     min_stamp : 2024-11-27 15:59:11.517608 
     max_stamp : 2024-11-27 15:59:11.517608 
     dif_stamp : 0:00:00 
     age_stamp : 54 days, 0:41:45.773076 
    [1970.03 1823.     40.      0.  ]
    c2sum/c2n:c2per(C2CUT)  1970.03/1823:1.081 (40) pv[1.000,> 0.05 : null-hyp ] 
    c2sum :  1970.0301 c2n :  1823.0000 c2per:     1.0807  C2CUT:   40 
    P[blyth@localhost opticks]$ 



The old SAB python way works but its slow (limited to lowish stats), how to enable the faster the C++ way ?:: 

    SAB=1 ~/o/G4CXTest_GEOM.sh pdb


    QCF qcf :  
    a.q 1000000 b.q 1000000 lim slice(None, None, None) 
    c2sum :  2362.9392 c2n :  2287.0000 c2per:     1.0332  C2CUT:   30 
    c2sum/c2n:c2per(C2CUT)  2362.94/2287:1.033 (30) pv[1.000,> 0.05 : null-hyp ] 

    np.c_[siq,_quo,siq,sabo2,sc2,sabo1][0:40]  ## A-B history frequency chi2 comparison 
    [[' 0' 'TO AB                                                                                          ' ' 0' '127116 127238' ' 0.0585' '     3      0']
     [' 1' 'TO BT BT BT BT BT BT SD                                                                        ' ' 1' ' 70015  70420' ' 1.1680' '     4      1']
     [' 2' 'TO BT BT BT BT BT BT SA                                                                        ' ' 2' ' 56878  56955' ' 0.0521' '     9      9']
     [' 3' 'TO SC AB                                                                                       ' ' 3' ' 51543  51096' ' 1.9467' '    33     49']
     [' 4' 'TO SC BT BT BT BT BT BT SD                                                                     ' ' 4' ' 36002  36125' ' 0.2098' '     7    104']
     [' 5' 'TO SC BT BT BT BT BT BT SA                                                                     ' ' 5' ' 30112  29855' ' 1.1014' '    22     25']
     [' 6' 'TO SC SC AB                                                                                    ' ' 6' ' 19790  19993' ' 1.0358' '    58     40']
     [' 7' 'TO RE AB                                                                                       ' ' 7' ' 18254  18320' ' 0.1191' '    55     18']
     [' 8' 'TO BT BT SA                                                                                    ' ' 8' ' 15651  15716' ' 0.1347' '    94     79']
     [' 9' 'TO SC SC BT BT BT BT BT BT SD                                                                  ' ' 9' ' 15539  15354' ' 1.1079' '    40     43']
     ['10' 'TO SC SC BT BT BT BT BT BT SA                                                                  ' '10' ' 12871  12801' ' 0.1909' '   129     26']
     ['11' 'TO BT BT AB                                                                                    ' '11' ' 10911  10899' ' 0.0066' '     8     71']
     ['12' 'TO BT AB                                                                                       ' '12' '  9071   9402' ' 5.9309' '    34     19']
     ['13' 'TO BT BT BT SA                                                                                 ' '13' '  9023   9020' ' 0.0005' '   155    747']
     ['14' 'TO BT BT BT BT BT BT BT SA                                                                     ' '14' '  7387   7642' ' 4.3266' '    26    265']
     ['15' 'TO SC SC SC AB                                                                                 ' '15' '  7372   7413' ' 0.1137' '    46    307']
     ['16' 'TO RE BT BT BT BT BT BT SD                                                                     ' '16' '  7316   7376' ' 0.2450' '    96     10']
     ['17' 'TO SC RE AB                                                                                    ' '17' '  7148   7216' ' 0.3219' '    21    209']
     ['18' 'TO RE BT BT BT BT BT BT SA                                                                     ' '18' '  6968   6974' ' 0.0026' '   316    220']
     ['19' 'TO SC BT BT AB                                                                                 ' '19' '  6423   6494' ' 0.3903' '    68     33']
     ['20' 'TO BT BT BT BT BT BT BT SR SA                                                                  ' '20' '  6405   6430' ' 0.0487' '   349     73']
     ['21' 'TO SC SC SC BT BT BT BT BT BT SD                                                               ' '21' '  6104   6302' ' 3.1601' '   146     17']
     ['22' 'TO BT BT BT BT SD                                                                              ' '22' '  6178   5989' ' 2.9359' '   238    285']
     ['23' 'TO SC BT AB                                                                                    ' '23' '  5555   5762' ' 3.7863' '   325    329']
     ['24' 'TO BT BT DR BT SA                                                                              ' '24' '  5517   5558' ' 0.1518' '    13     78']
     ['25' 'TO RE RE AB                                                                                    ' '25' '  5497   5390' ' 1.0516' '   152    214']
     ['26' 'TO SC SC SC BT BT BT BT BT BT SA                                                               ' '26' '  5109   5166' ' 0.3162' '     6    240']
     ['27' 'TO SC BT BT SA                                                                                 ' '27' '  4746   4886' ' 2.0349' '   222     97']
     ['28' 'TO SC BT BT BT BT BT BT BT SA                                                                  ' '28' '  4480   4425' ' 0.3397' '    85    256']
     ['29' 'TO BT BT BT BT BT BT BR BT BT BT BT BT BT BT BT SD                                             ' '29' '  3839   3825' ' 0.0256' '   455    345']
     ['30' 'TO RE SC AB                                                                                    ' '30' '  3548   3493' ' 0.4296' '   268     93']
     ['31' 'TO SC RE BT BT BT BT BT BT SD                                                                  ' '31' '  3115   3200' ' 1.1441' '   704    110']
     ['32' 'TO SC BT BT BT BT BT BT BT SR SA                                                               ' '32' '  3111   3176' ' 0.6720' '   584    139']
     ['33' 'TO BT BT BT BT BT BT BT SD                                                                     ' '33' '  3175   3136' ' 0.2410' '   862     74']
     ['34' 'TO SC BT BT BT SA                                                                              ' '34' '  3106   3134' ' 0.1256' '   922    135']
     ['35' 'TO BT BT BT BT BT BT BR BT BT BT BT BT BT BT BT SA                                             ' '35' '  3124   3082' ' 0.2842' '   615    531']
     ['36' 'TO BT BT BT BT BT BT BT SR SR SA                                                               ' '36' '  3089   3058' ' 0.1563' '   171     57']
     ['37' 'TO SC SC BT BT AB                                                                              ' '37' '  2984   2930' ' 0.4931' '   802     90']
     ['38' 'TO BT BT BT BT AB                                                                              ' '38' '  2967   2848' ' 2.4353' '    73    460']
     ['39' 'TO SC RE BT BT BT BT BT BT SA                                                                  ' '39' '  2961   2900' ' 0.6349' '   661      3']]

    np.c_[siq,_quo,siq,sabo2,sc2,sabo1][bzero]  ## in A but not B 
    [['1412' 'TO BT BT DR BT BT BT SD                                                                        ' '1412' '    30      0' ' 0.0000' '  7782     -1']
     ['2581' 'TO BT BT BT BT BT BT BR BT BT BT BT BT BT BT SR SD                                             ' '2581' '    15      0' ' 0.0000' ' 39967     -1']
     ['2926' 'TO RE RE RE SC SC SC BT BT AB                                                                  ' '2926' '    13      0' ' 0.0000' ' 20118     -1']
     ['2975' 'TO BT BT BT DR BT BT BT SD                                                                     ' '2975' '    13      0' ' 0.0000' ' 19038     -1']
     ['3130' 'TO BT BT DR BT BT BT BT SR BT BT BT BT BT BT BT SD                                             ' '3130' '    12      0' ' 0.0000' ' 43495     -1']
     ['3132' 'TO SC BT BT BT BT BT BT BR BT BT BT BT BT BT BT SR SD                                          ' '3132' '    12      0' ' 0.0000' '111913     -1']
     ['3602' 'TO SC BT BT BT BT BT BT BR BT BT BT BT BT BT BT BT BT SD                                       ' '3602' '    11      0' ' 0.0000' ' 16191     -1']
     ['3605' 'TO SC BT BT DR BT BT BT SD                                                                     ' '3605' '    11      0' ' 0.0000' ' 95956     -1']]

    np.c_[siq,_quo,siq,sabo2,sc2,sabo1][azero]  ## in B but not A 
    [['2448' 'TO SC BT BT BT BT BT BT BR BT BT BT BT BT BT BT SR BT SD                                       ' '2448' '     0     16' ' 0.0000' '    -1  24615']
     ['3023' 'TO BT BT BT BT BT BT BR BT BT BT BT BT BT BT SR BT SD                                          ' '3023' '     0     13' ' 0.0000' '    -1  73425']
     ['3213' 'TO SC BT BT BT BT BR BR BR DR AB                                                               ' '3213' '     0     12' ' 0.0000' '    -1 192811']
     ['3404' 'TO BT BT BT BT BT BT BR BT BT BT BT BT BT BT SR SR BT SD                                       ' '3404' '     0     11' ' 0.0000' '    -1  71722']]
    ]----- repr(ab) 
    PICK=A MODE=3  ~/opticks/g4cx/tests/G4CXTest_GEOM.sh 

    In [1]: 


Had to change the script to find the seq in new layout, then the C++ way works (very fast)::

    P[blyth@localhost opticks]$ ~/opticks/sysrap/tests/sseq_index_test.sh run
    a_path $AFOLD/seq.npy /data/blyth/opticks/GEOM/J_2025jan08/G4CXTest/ALL98_Debug_Philox/A000/seq.npy a_seq (1000000, 2, 2, )
    b_path $BFOLD/seq.npy /data/blyth/opticks/GEOM/J_2025jan08/G4CXTest/ALL98_Debug_Philox/B000/seq.npy b_seq (1000000, 2, 2, )
    sseq_index_test__DEBUG:0
    AB
    [sseq_index_ab::desc u.size 152198 opt BRIEF mode 6sseq_index_ab_chi2::desc sum  1893.1207 ndf 1810.0000 sum/ndf     1.0459 sseq_index_ab_chi2_ABSUM_MIN:40.0000
        TO AB                                                                                            :  127116 127238 :     0.0585 : Y :       3      0 :   
        TO BT BT BT BT BT BT SD                                                                          :   70015  70420 :     1.1680 : Y :       4      1 :   
        TO BT BT BT BT BT BT SA                                                                          :   56878  56955 :     0.0521 : Y :       9      9 :   
        TO SC AB                                                                                         :   51543  51096 :     1.9467 : Y :      33     49 :   
        TO SC BT BT BT BT BT BT SD                                                                       :   36002  36125 :     0.2098 : Y :       7    104 :   
        TO SC BT BT BT BT BT BT SA                                                                       :   30112  29855 :     1.1014 : Y :      22     25 :   
        TO SC SC AB                                                                                      :   19790  19993 :     1.0358 : Y :      58     40 :   
        TO RE AB                                                                                         :   18254  18320 :     0.1191 : Y :      55     18 :   
        TO BT BT SA                                                                                      :   15651  15716 :     0.1347 : Y :      94     79 :   
        TO SC SC BT BT BT BT BT BT SD                                                                    :   15539  15354 :     1.1079 : Y :      40     43 :   
        TO SC SC BT BT BT BT BT BT SA                                                                    :   12871  12801 :     0.1909 : Y :     129     26 :   
        TO BT BT AB                                                                                      :   10911  10899 :     0.0066 : Y :       8     71 :   
        TO BT AB                                                                                         :    9071   9402 :     5.9309 : Y :      34     19 :   
        TO BT BT BT SA                                                                                   :    9023   9020 :     0.0005 : Y :     155    747 :   
        TO BT BT BT BT BT BT BT SA                                                                       :    7387   7642 :     4.3266 : Y :      26    265 :   
        TO SC SC SC AB                                                                                   :    7372   7413 :     0.1137 : Y :      46    307 :   
        TO RE BT BT BT BT BT BT SD                                                                       :    7316   7376 :     0.2450 : Y :      96     10 :   
        TO SC RE AB                                                                                      :    7148   7216 :     0.3219 : Y :      21    209 :   
        TO RE BT BT BT BT BT BT SA                                                                       :    6968   6974 :     0.0026 : Y :     316    220 :   
        TO SC BT BT AB                                                                                   :    6423   6494 :     0.3903 : Y :      68     33 :   
        TO BT BT BT BT BT BT BT SR SA                                                                    :    6405   6430 :     0.0487 : Y :     349     73 :   
        TO SC SC SC BT BT BT BT BT BT SD                                                                 :    6104   6302 :     3.1601 : Y :     146     17 :   
        TO BT BT BT BT SD                                                                                :    6178   5989 :     2.9359 : Y :     238    285 :   
        TO SC BT AB                                                                                      :    5555   5762 :     3.7863 : Y :     325    329 :   
        TO BT BT DR BT SA                                                                                :    5517   5558 :     0.1518 : Y :      13     78 :   
        TO RE RE AB                                                                                      :    5497   5390 :     1.0516 : Y :     152    214 :   
        TO SC SC SC BT BT BT BT BT BT SA                                                                 :    5109   5166 :     0.3162 : Y :       6    240 :   
        TO SC BT BT SA                                                                                   :    4746   4886 :     2.0349 : Y :     222     97 :   
        TO SC BT BT BT BT BT BT BT SA                                                                    :    4480   4425 :     0.3397 : Y :      85    256 :   
        TO BT BT BT BT BT BT BR BT BT BT BT BT BT BT BT SD                                               :    3839   3825 :     0.0256 : Y :     455    345 :   
        TO RE SC AB                                                                                      :    3548   3493 :     0.4296 : Y :     268     93 :   
        TO SC RE BT BT BT BT BT BT SD                                                                    :    3115   3200 :     1.1441 : Y :     704    110 :   
        TO SC BT BT BT BT BT BT BT SR SA                                                                 :    3111   3176 :     0.6720 : Y :     584    139 :   
        TO BT BT BT BT BT BT BT SD                                                                       :    3175   3136 :     0.2410 : Y :     862     74 :   
        TO SC BT BT BT SA                                                                                :    3106   3134 :     0.1256 : Y :     922    135 :   
        TO BT BT BT BT BT BT BR BT BT BT BT BT BT BT BT SA                                               :    3124   3082 :     0.2842 : Y :     615    531 :   
        TO BT BT BT BT BT BT BT SR SR SA                                                                 :    3089   3058 :     0.1563 : Y :     171     57 :   
        TO SC SC BT BT AB                                                                                :    2984   2930 :     0.4931 : Y :     802     90 :   
        TO BT BT BT BT AB                                                                                :    2967   2848 :     2.4353 : Y :      73    460 :   
        TO SC RE BT BT BT BT BT BT SA                                                                    :    2961   2900 :     0.6349 : Y :     661      3 :   
        TO SC BT BT BT BT SD                                                                             :    2807   2831 :     0.1022 : Y :    1692    696 :   
        TO RE SC BT BT BT BT BT BT SD                                                                    :    2754   2813 :     0.6253 : Y :     214    481 :   
        TO SC SC SC SC AB                                                                                :    2736   2781 :     0.3670 : Y :     913    431 :   
        TO SC SC BT AB                                                                                   :    2584   2761 :     5.8614 : Y :     224    616 :   
        TO SC SC RE AB                                                                                   :    2699   2675 :     0.1072 : Y :     258     23 :   
        TO RE SC BT BT BT BT BT BT SA                                                                    :    2633   2562 :     0.9704 : Y :     228    601 :   
        TO SC SC SC SC BT BT BT BT BT BT SD                                                              :    2287   2354 :     0.9672 : Y :     205    101 :   
        TO RE RE BT BT BT BT BT BT SD                                                                    :    2205   2238 :     0.2451 : Y :     121    574 :   
        TO RE RE BT BT BT BT BT BT SA                                                                    :    2085   2132 :     0.5238 : Y :     843     98 :   
        TO BT BT BT BT BT BT BT SR SR SR SA                                                              :    2114   2026 :     1.8705 : Y :    1292    314 :   
        TO SC RE RE AB                                                                                   :    2052   2103 :     0.6260 : Y :     697    286 :   
        TO SC BT BT BT BT BT BT BT SD                                                                    :    2085   2048 :     0.3312 : Y :      81    213 :   
        TO SC SC BT BT BT BT BT BT BT SA                                                                 :    1983   1989 :     0.0091 : Y :    1736    911 :   
        TO SC SC BT BT SA                                                                                :    1927   1962 :     0.3150 : Y :     709    501 :   
        TO SC BT BT BT BT SA                                                                             :    1961   1940 :     0.1130 : Y :     396   1082 :   
        TO SC SC SC SC BT BT BT BT BT BT SA                                                              :    1934   1893 :     0.4392 : Y :      72     95 :   
        TO SC BT BT BT BT BT BT BR BT BT BT BT BT BT BT BT SD                                            :    1763   1762 :     0.0003 : Y :     915    510 :   
        TO RE RE RE AB                                                                                   :    1673   1622 :     0.7894 : Y :     470    406 :   
        TO BT BT BT BT BR BT BT BT BT SA                                                                 :    1536   1479 :     1.0776 : Y :     163   1433 :   
        TO SC BT BT BT BT BT BT BR BT BT BT BT BT BT BT BT SA                                            :    1513   1485 :     0.2615 : Y :     500    762 :   
    ]sseq_index_ab::desc

    AB
    [sseq_index_ab::desc u.size 152198 opt AZERO mode 1
        TO SC BT BT BT BT BT BT BR BT BT BT BT BT BT BT SR BT SD                                         :      -1     16 :     0.0000 : N :      -1  24615 : AZERO C2EXC  
        TO BT BT BT BT BT BT BR BT BT BT BT BT BT BT SR BT SD                                            :      -1     13 :     0.0000 : N :      -1  73425 : AZERO C2EXC  
        TO SC BT BT BT BT BR BR BR DR AB                                                                 :      -1     12 :     0.0000 : N :      -1 192811 : AZERO C2EXC  
        TO BT BT BT BT BT BT BR BT BT BT BT BT BT BT SR SR BT SD                                         :      -1     11 :     0.0000 : N :      -1  71722 : AZERO C2EXC  
    ]sseq_index_ab::desc

    AB
    [sseq_index_ab::desc u.size 152198 opt BZERO mode 2
        TO BT BT DR BT BT BT SD                                                                          :      30     -1 :     0.0000 : N :    7782     -1 : BZERO C2EXC  
        TO BT BT BT BT BT BT BR BT BT BT BT BT BT BT SR SD                                               :      15     -1 :     0.0000 : N :   39967     -1 : BZERO C2EXC  
        TO BT BT BT DR BT BT BT SD                                                                       :      13     -1 :     0.0000 : N :   19038     -1 : BZERO C2EXC  
        TO RE RE RE SC SC SC BT BT AB                                                                    :      13     -1 :     0.0000 : N :   20118     -1 : BZERO C2EXC  
        TO BT BT DR BT BT BT BT SR BT BT BT BT BT BT BT SD                                               :      12     -1 :     0.0000 : N :   43495     -1 : BZERO C2EXC  
        TO SC BT BT BT BT BT BT BR BT BT BT BT BT BT BT SR SD                                            :      12     -1 :     0.0000 : N :  111913     -1 : BZERO C2EXC  
        TO SC BT BT BT BT BT BT BR BT BT BT BT BT BT BT BT BT SD                                         :      11     -1 :     0.0000 : N :   16191     -1 : BZERO C2EXC  
        TO SC BT BT DR BT BT BT SD                                                                       :      11     -1 :     0.0000 : N :   95956     -1 : BZERO C2EXC  
    ]sseq_index_ab::desc

    AB
    [sseq_index_ab::desc u.size 152198 opt DEVIANT mode 5
    :r:`TO RE RE BT BT BT BT BT BT BT BT SA                                                              :      89     50 :    10.9424 : Y :   10000   4835 : DEVIANT  `
    :r:`TO BT BT BT BT BT BT BT BT SD                                                                    :      47      1 :    44.0833 : Y :   56216 814025 : DEVIANT  `
    :r:`TO BT BT BT BT BR BR BR DR AB                                                                    :       7     36 :    19.5581 : Y :   54913  19130 : DEVIANT  `
    ]sseq_index_ab::desc







Issue 1 : FIXED : torch genstep config 
-----------------------------------------

* AHHA : probably now needs a comma delimited list ?  NOPE : MORE THAN THAT.

* comparing ~/o/cxs_min.sh ~/o/G4CXTest_GEOM.sh shows big change to 
  more flexible genstep config that has not yet been adopted in ~/o/G4CXTest_GEOM.sh  ?

* this change happenend following standard support for very large numbers of photons,
  the old assumption than can get away with test torch running with a single genstep is
  not valid anymore : have to split into multiple gensteps for the multi-launch 
  to be able to slice the gensteps as needed for VRAM



::

    ~/o/G4CXTest_GEOM.sh dbg


    2025-01-20 15:52:56.277 INFO  [29024] [G4CXApp::BeamOn@343] [ OPTICKS_NUM_EVENT=1
    2025-01-20 15:54:32.867 INFO  [29024] [G4CXApp::GeneratePrimaries@223] [ SEventConfig::RunningModeLabel SRM_TORCH eventID 0
    G4CXTest: /home/blyth/opticks/sysrap/SEvent.cc:179: static NP* SEvent::MakeGenstep(int, int): Assertion `num_gs > 0' failed.

    Thread 1 "G4CXTest" received signal SIGABRT, Aborted.
    0x00007ffff23ab387 in raise () from /lib64/libc.so.6
    (gdb) bt
    #0  0x00007ffff23ab387 in raise () from /lib64/libc.so.6
    #1  0x00007ffff23aca78 in abort () from /lib64/libc.so.6
    #2  0x00007ffff23a41a6 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007ffff23a4252 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007ffff3a8058e in SEvent::MakeGenstep (gentype=6, index_arg=0) at /home/blyth/opticks/sysrap/SEvent.cc:179
    #5  0x00007ffff3a8022c in SEvent::MakeTorchGenstep (idx_arg=0) at /home/blyth/opticks/sysrap/SEvent.cc:143
    #6  0x000000000040a1e0 in G4CXApp::GeneratePrimaries (this=0x6c6400, event=0x24303020) at /home/blyth/opticks/g4cx/tests/G4CXApp.h:236
    #7  0x00007ffff7059c4a in G4RunManager::GenerateEvent(int) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4run.so
    #8  0x00007ffff705795c in G4RunManager::DoEventLoop(int, char const*, int) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4run.so
    #9  0x00007ffff70553ae in G4RunManager::BeamOn(int, char const*, int) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4run.so
    #10 0x000000000040aa25 in G4CXApp::BeamOn (this=0x6c6400) at /home/blyth/opticks/g4cx/tests/G4CXApp.h:344
    #11 0x000000000040ab31 in G4CXApp::Main () at /home/blyth/opticks/g4cx/tests/G4CXApp.h:351
    #12 0x000000000040acbf in main (argc=1, argv=0x7fffffff4338) at /home/blyth/opticks/g4cx/tests/G4CXTest.cc:13
    (gdb) 

    (gdb) p with_index
    $1 = true

    (gdb) p index_arg
    $2 = 0

    (gdb) p num_ph
    $3 = 1000000

    (gdb) p num_gs
    $4 = 0



The change to SEventConfig::_GetNumGenstepPerEvent (accepting the comma delimited list) is recent : script config not handling it yet::


     106 std::vector<int>* SEventConfig::_GetNumGenstepPerEvent()
     107 {
     108     const char* spec = ssys::getenvvar(kNumGenstep,  _NumGenstepDefault );
     109     return sstr::ParseIntSpecList<int>( spec, ',' );
     110 }
     111 std::vector<int>* SEventConfig::_NumGenstepPerEvent = _GetNumGenstepPerEvent() ;
     112 

::

    149 /**
    150 SEvent::MakeGenstep
    151 ---------------------
    152 
    153 NB index_arg is userspace 0-based index, that is not the same as the internal SEvt::index 
    154 which may be offset by OPTICKS_START_INDEX
    155 
    156 **/
    157 
    158 
    159 NP* SEvent::MakeGenstep( int gentype, int index_arg )
    160 {
    161     bool with_index = index_arg != -1 ;
    162     if(with_index) assert( index_arg >= 0 );  // index_arg is 0-based 
    163     int num_ph = with_index ? SEventConfig::NumPhoton(index_arg)  : ssys::getenvint("SEvent__MakeGenstep_num_ph", 100 ) ;
    164     int num_gs = with_index ? SEventConfig::NumGenstep(index_arg) : ssys::getenvint("SEvent__MakeGenstep_num_gs", 1   ) ;
    165 
    166     bool dump = ssys::getenvbool("SEvent_MakeGenstep_dump");
    167     const int M = 1000000 ;
    168 
    169     LOG(LEVEL)
    170         << " gentype " << gentype
    171         << " index_arg " << index_arg
    172         << " with_index " << ( with_index ? "YES" : "NO " )
    173         << " num_ph " << num_ph
    174         << " num_ph/M " << num_ph/M
    175         << " num_gs " << num_gs
    176         << " dump " << dump
    177         ;
    178 
    179     assert( num_gs > 0 );
    180 
    181     NP* gs = NP::Make<float>(num_gs, 6, 4 );
    182     gs->set_meta<std::string>("creator", "SEvent::MakeGenstep" );
    183     gs->set_meta<int>("num_ph", num_ph );
    184     gs->set_meta<int>("num_gs", num_gs );
    185     gs->set_meta<int>("index_arg",  index_arg );
    186 
    187 
    188     int gs_start = 0 ;
    189     int gs_stop = num_gs ;
    190     int gs_ph   = num_ph/num_gs ; // divide the num_ph equally between the num_gs   
    191 
    192     switch(gentype)
    193     {
    194         case  OpticksGenstep_TORCH:         FillGenstep<storch>(   gs, gs_start, gs_stop, gs_ph, dump) ; break ;
    195         case  OpticksGenstep_CERENKOV:      FillGenstep<scerenkov>(gs, gs_start, gs_stop, gs_ph, dump) ; break ;
    196         case  OpticksGenstep_SCINTILLATION: FillGenstep<sscint>(   gs, gs_start, gs_stop, gs_ph, dump) ; break ;
    197         case  OpticksGenstep_CARRIER:       FillGenstep<scarrier>( gs, gs_start, gs_stop, gs_ph, dump) ; break ;
    198     }
    199     return gs ;
    200 }



