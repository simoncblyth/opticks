tboolean-cubeplanes-many-stuck-tracks-drastic-difference
============================================================

* cubeplanes constructs a cube using a G4TesselatedSolid 


TODO:

* stats on stuck tracks


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

