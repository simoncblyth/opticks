tboolean-proxy-scan-LV10-history-misaligned-big-bouncer
==========================================================


Context
----------

* :doc:`tboolean-proxy-scan`


Command shortcuts
---------------------

::

    lv(){ echo 21 ; }
    # default geometry LV index to test 

    ts(){  PROXYLV=${1:-$(lv)} tboolean.sh --align --dbgskipclearzero --dbgnojumpzero --dbgkludgeflatzero $* ; } 
    # **simulate** : aligned bi-simulation creating OK+G4 events 

    tv(){  PROXYLV=${1:-$(lv)} tboolean.sh --load $* ; } 
    # **visualize** : load events and visualize the propagation

    tv4(){ tv --vizg4 $* ; }
    # **visualize** the geant4 propagation 

    ta(){  tboolean-;PROXYLV=${1:-$(lv)} tboolean-proxy-ip ; } 
    # **analyse** : load events and analyse the propagation



ISSUE : Lost history alignment for big bouncers
-------------------------------------------------

ta 10::

    ab.his
    .                seqhis_ana  1:tboolean-proxy-10:tboolean-proxy-10   -1:tboolean-proxy-10:tboolean-proxy-10        c2        ab        ba 
    .                              10000     10000         0.03/13 =  0.00  (pval:1.000 prob:0.000)  
    0000             8ccd      7610      7610      0             0.00        1.000 +- 0.011        1.000 +- 0.011  [4 ] TO BT BT SA
    0001              8bd       510       510      0             0.00        1.000 +- 0.044        1.000 +- 0.044  [3 ] TO BR SA
    0002            8cbcd       492       492      0             0.00        1.000 +- 0.045        1.000 +- 0.045  [5 ] TO BT BR BT SA
    0003              86d       467       467      0             0.00        1.000 +- 0.046        1.000 +- 0.046  [3 ] TO SC SA
    0004            86ccd       449       449      0             0.00        1.000 +- 0.047        1.000 +- 0.047  [5 ] TO BT BT SC SA
    0005            8cc6d        75        75      0             0.00        1.000 +- 0.115        1.000 +- 0.115  [5 ] TO SC BT BT SA
    0006          8cc6ccd        67        67      0             0.00        1.000 +- 0.122        1.000 +- 0.122  [7 ] TO BT BT SC BT BT SA
    0007              4cd        44        44      0             0.00        1.000 +- 0.151        1.000 +- 0.151  [3 ] TO BT AB
    0008           866ccd        35        35      0             0.00        1.000 +- 0.169        1.000 +- 0.169  [6 ] TO BT BT SC SC SA
    0009             866d        30        30      0             0.00        1.000 +- 0.183        1.000 +- 0.183  [4 ] TO SC SC SA
    0010           8cbbcd        26        26      0             0.00        1.000 +- 0.196        1.000 +- 0.196  [6 ] TO BT BR BR BT SA
    0011             86bd        25        25      0             0.00        1.000 +- 0.200        1.000 +- 0.200  [4 ] TO BR SC SA
    0012           86cbcd        20        20      0             0.00        1.000 +- 0.224        1.000 +- 0.224  [6 ] TO BT BR BT SC SA
    0013       bbbbbbb6cd        16        15      1             0.03        1.067 +- 0.267        0.938 +- 0.242  [10] TO BT SC BR BR BR BR BR BR BR
    0014             8b6d        14        14      0             0.00        1.000 +- 0.267        1.000 +- 0.267  [4 ] TO SC BR SA
    0015           8b6ccd        11        11      0             0.00        1.000 +- 0.302        1.000 +- 0.302  [6 ] TO BT BT SC BR SA
    0016            8c6cd         9         9      0             0.00        1.000 +- 0.333        1.000 +- 0.333  [5 ] TO BT SC BT SA
    0017           8cbc6d         8         8      0             0.00        1.000 +- 0.354        1.000 +- 0.354  [6 ] TO SC BT BR BT SA
    0018         8cbc6ccd         8         8      0             0.00        1.000 +- 0.354        1.000 +- 0.354  [8 ] TO BT BT SC BT BR BT SA
    0019           86cc6d         7         7      0             0.00        1.000 +- 0.378        1.000 +- 0.378  [6 ] TO SC BT BT SC SA
    .                              10000     10000         0.03/13 =  0.00  (pval:1.000 prob:0.000)  



Up the slice to see full table, then remove the matched histories::

    In [2]: ab.his[:100]
    Out[2]: 
    ab.his
    .                seqhis_ana  1:tboolean-proxy-10:tboolean-proxy-10   -1:tboolean-proxy-10:tboolean-proxy-10        c2        ab        ba 
    .                              10000     10000         0.03/13 =  0.00  (pval:1.000 prob:0.000)  
    0013       bbbbbbb6cd        16        15      1             0.03        1.067 +- 0.267        0.938 +- 0.242  [10] TO BT SC BR BR BR BR BR BR BR
    0048       6bbbbbb6cd         1         0      1             0.00        0.000 +- 0.000        0.000 +- 0.000  [10] TO BT SC BR BR BR BR BR BR SC
    0061         8cb6b6cd         1         0      1             0.00        0.000 +- 0.000        0.000 +- 0.000  [8 ] TO BT SC BR SC BR BT SA
    0064       4bbbbbb6cd         0         1     -1             0.00        0.000 +- 0.000        0.000 +- 0.000  [10] TO BT SC BR BR BR BR BR BR AB
    0065        8cb6bb6cd         0         1     -1             0.00        0.000 +- 0.000        0.000 +- 0.000  [9 ] TO BT SC BR BR SC BR BT SA
    0066            4b6cd         0         1     -1             0.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] TO BT SC BR AB
    .                              10000     10000         0.03/13 =  0.00  (pval:1.000 prob:0.000)  



Only three maligned as they come in pairs::

    In [3]: ab.maligned
    Out[3]: array([ 436, 1676, 5207])

    In [7]: ab.maligned??
    ..
    def _get_maligned(self):
        return np.where(self.a.seqhis != self.b.seqhis)[0]


::

    In [10]: ab.dumpline(ab.maligned)
          0    436 : * :                      TO BT SC BR BR BR BR BR BR SC                      TO BT SC BR BR BR BR BR BR AB 
          1   1676 : * :                      TO BT SC BR BR BR BR BR BR BR                         TO BT SC BR BR SC BR BT SA 
          2   5207 : * :                            TO BT SC BR SC BR BT SA                                     TO BT SC BR AB 





* hmm not just truncation, 



ts 10:436 : OK SC where G4 AB at last point : SC vs AB
----------------------------------------------------

436 : no rpost difference just different (truncated) flag for the last point SC vs AB::

    In [11]: a.rposti(436)
    Out[11]: 
    A()sliced
    A([[ -2155.6133,  -5961.4464, -71998.8026,      0.    ],
       [ -2155.6133,  -5961.4464,  -2500.5993,    231.8218],
       [ -2155.6133,  -5961.4464,  -2100.6792,    234.2389],
       [ -9266.2806,  -8749.9003,   2500.5993,    288.2721],
       [-16994.407 , -11780.0641,  -2500.5993,    346.9856],
       [-23999.6009, -14524.5708,   2032.561 ,    400.2277],
       [-23278.8657, -14808.0306,   2500.5993,    405.7211],
       [-15550.7394, -17838.1945,  -2500.5993,    464.4347],
       [ -7822.613 , -20866.161 ,   2500.5993,    523.1482],
       [  -428.4858, -23764.483 ,  -2283.0604,    579.3128]])

    In [12]: b.rposti(436)
    Out[12]: 
    A()sliced
    A([[ -2155.6133,  -5961.4464, -71998.8026,      0.    ],
       [ -2155.6133,  -5961.4464,  -2500.5993,    231.8218],
       [ -2155.6133,  -5961.4464,  -2100.6792,    234.2389],
       [ -9266.2806,  -8749.9003,   2500.5993,    288.2721],
       [-16994.407 , -11780.0641,  -2500.5993,    346.9856],
       [-23999.6009, -14524.5708,   2032.561 ,    400.2277],
       [-23278.8657, -14808.0306,   2500.5993,    405.7211],
       [-15550.7394, -17838.1945,  -2500.5993,    464.4347],
       [ -7822.613 , -20866.161 ,   2500.5993,    523.1482],
       [  -428.4858, -23764.483 ,  -2283.0604,    579.3128]])

    In [16]: (a.rposti(436) - b.rposti(436))*1e9
    Out[16]: 
    A()sliced
    A([[0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.]])





ts 10:1676 : OK sails where G4 scatters : sail vs SC
--------------------------------------------------------

::

    In [10]: ab.dumpline(ab.maligned)
          0    436 : * :                      TO BT SC BR BR BR BR BR BR SC                      TO BT SC BR BR BR BR BR BR AB 
          1   1676 : * :                      TO BT SC BR BR BR BR BR BR BR                         TO BT SC BR BR SC BR BT SA 
          2   5207 : * :                            TO BT SC BR SC BR BT SA                                     TO BT SC BR AB 



    In [18]: np.set_printoptions(suppress=True)

    In [19]: a.rposti(1676)
    Out[19]: 
    A()sliced
    A([[ -1160.2078,  -5902.1176, -71998.8026,      0.    ],      TO
       [ -1160.2078,  -5902.1176,  -2500.5993,    231.8218],      BT
       [ -1160.2078,  -5902.1176,     70.3156,    247.4011],      SC 
       [   788.8534,  -5831.8019,   2500.5993,    266.2765],      BR 
       [  4803.436 ,  -5682.3812,  -2500.5993,    305.1478],      BR
       [  8815.8212,  -5532.9605,   2500.5993,   *343.9972*],     BR
       [ 12828.2064,  -5383.5398,  -2500.5993,    382.8686],      BR   
       [ 16842.7889,  -5234.1191,   2500.5993,    421.718 ],      BR
       [ 20855.1741,  -5084.6984,  -2500.5993,    460.5893],      BR
       [ 23999.6009,  -4968.2382,   1417.2993,    491.0448]])     BR

    In [20]: b.rposti(1676)
    Out[20]: 
    A()sliced
    A([[ -1160.2078,  -5902.1176, -71998.8026,      0.    ],      TO
       [ -1160.2078,  -5902.1176,  -2500.5993,    231.8218],      BT
       [ -1160.2078,  -5902.1176,     70.3156,    247.4011],      SC
       [   788.8534,  -5831.8019,   2500.5993,    266.2765],      BR   
       [  4803.436 ,  -5682.3812,  -2500.5993,    305.1478],      BR
       [  8534.5587,  -5543.9474,   2151.2186,   *341.2945*],     SC  <------   scatters 2.7ns before other history reflects again 
       [ 15647.4234,  -1852.3772,   2500.5993,    389.9001],      BR
       [ 23999.6009,   2485.2178,   2089.6924,    446.9876],      BT 
       [ 72001.    ,  59678.1872,  -3315.8211,    696.696 ]])     SA


    In [26]: a.rposti(1676)[:9] - b.rposti(1676)
    Out[26]: 
    A()sliced
    A([[     0.    ,      0.    ,      0.    ,      0.    ],
       [     0.    ,      0.    ,      0.    ,      0.    ],
       [     0.    ,      0.    ,      0.    ,      0.    ],
       [     0.    ,      0.    ,      0.    ,      0.    ],
       [     0.    ,      0.    ,      0.    ,      0.    ],
       [   281.2625,     10.9868,    349.3807,      2.7028],
       [ -2819.217 ,  -3531.1627,  -5001.1986,     -7.0316],
       [ -7156.8119,  -7719.3369,    410.9069,    -25.2697],
       [-51145.8259, -64762.8856,    815.2217,   -236.1067]])




ts 10:5207 : OK SC where G4 AB : SC vs AB
--------------------------------------------

Common points exactly match, but where G4 aborbs Opticks scatters::

    In [10]: ab.dumpline(ab.maligned)
          0    436 : * :                      TO BT SC BR BR BR BR BR BR SC                      TO BT SC BR BR BR BR BR BR AB 
          1   1676 : * :                      TO BT SC BR BR BR BR BR BR BR                         TO BT SC BR BR SC BR BT SA 
          2   5207 : * :                            TO BT SC BR SC BR BT SA                                     TO BT SC BR AB 


    In [27]: a.rposti(5207)
    Out[27]: 
    A()sliced
    A([[  5719.7364,  -3812.4252, -71998.8026,      0.    ],      TO
       [  5719.7364,  -3812.4252,  -2500.5993,    231.8218],      BT
       [  5719.7364,  -3812.4252,   1070.1159,    253.4658],      SC 
       [ -4111.2665,  -4667.1994,  -2500.5993,    317.0575],      BR
       [-15590.2919,  -5664.8023,   1667.7987,    391.3064],      SC   <--- Opticks scatters
       [-18840.1921,  -6376.748 ,   2500.5993,    412.0715],      BR
       [-23999.6009,  -7510.5874,   1177.7867,    445.0759],      BT 
       [-72001.    , -27056.1331, -21655.0143,    633.9832]])     SA


    In [28]: b.rposti(5207)
    Out[28]: 
    A()sliced
    A([[  5719.7364,  -3812.4252, -71998.8026,      0.    ],      TO
       [  5719.7364,  -3812.4252,  -2500.5993,    231.8218],      BT
       [  5719.7364,  -3812.4252,   1070.1159,    253.4658],      SC
       [ -4111.2665,  -4667.1994,  -2500.5993,    317.0575],      BR
       [-15590.2919,  -5664.8023,   1667.7987,    391.3064]])     AB   <--- G4 absorbs : at exact same point 

    In [30]: (a.rposti(5207)[:5] -  b.rposti(5207))*1e9
    Out[30]: 
    A()sliced
    A([[0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.]])



Need to dump the scatter/absorb decision and randoms.


::

    549         slot++ ;
    550 
    551         command = propagate_to_boundary( p, s, rng );
    552         if(command == BREAK)    break ;           // BULK_ABSORB
    553         if(command == CONTINUE) continue ;        // BULK_REEMIT/BULK_SCATTER
    554         // PASS : survivors will go on to pick up one of the below flags, 
    555 
    556


* :doc:`masked-running-revived`




