check_innerwater_bulk_absorb
===============================



Issue : looks to be substantially more bulk absorption in the water in OK than in G4, investigating why
---------------------------------------------------------------------------------------------------------

::

    epsilon:offline blyth$ tds3gun.sh 1

    In [3]: ab.his
    Out[3]:
    ab.his
    .       seqhis_ana  cfo:sum  1:g4live:tds3gun   -1:g4live:tds3gun        c2        ab        ba
    .                              11278     11278       513.94/67 =  7.67  (pval:0.000 prob:1.000)
    0000               42      1653      1665    -12             0.04        0.993 +- 0.024        1.007 +- 0.025  [2 ] SI AB
    0001            7ccc2      1292      1230     62             1.52        1.050 +- 0.029        0.952 +- 0.027  [5 ] SI BT BT BT SD
    0002            8ccc2       590       674    -84             5.58        0.875 +- 0.036        1.142 +- 0.044  [5 ] SI BT BT BT SA
    0003           7ccc62       581       552     29             0.74        1.053 +- 0.044        0.950 +- 0.040  [6 ] SI SC BT BT BT SD
    0004             8cc2       564       464    100             9.73        1.216 +- 0.051        0.823 +- 0.038  [4 ] SI BT BT SA
    0005              452       422       534   -112            13.12        0.790 +- 0.038        1.265 +- 0.055  [3 ] SI RE AB
    0006           7ccc52       380       397    -17             0.37        0.957 +- 0.049        1.045 +- 0.052  [6 ] SI RE BT BT BT SD
    0007              462       392       367     25             0.82        1.068 +- 0.054        0.936 +- 0.049  [3 ] SI SC AB
    0008           8ccc62       251       267    -16             0.49        0.940 +- 0.059        1.064 +- 0.065  [6 ] SI SC BT BT BT SA
    0009          7ccc662       219       213      6             0.08        1.028 +- 0.069        0.973 +- 0.067  [7 ] SI SC SC BT BT BT SD
    0010             4cc2       278       127    151            56.30        2.189 +- 0.131        0.457 +- 0.041  [4 ] SI BT BT AB
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    0011            8cc62       206       186     20             1.02        1.108 +- 0.077        0.903 +- 0.066  [5 ] SI SC BT BT SA
    0012           8ccc52       154       188    -34             3.38        0.819 +- 0.066        1.221 +- 0.089  [6 ] SI RE BT BT BT SA
    0013          7ccc652       157       159     -2             0.01        0.987 +- 0.079        1.013 +- 0.080  [7 ] SI RE SC BT BT BT SD
    0014               41       142       144     -2             0.01        0.986 +- 0.083        1.014 +- 0.085  [2 ] CK AB
    0015            4cc62       197        71    126            59.24        2.775 +- 0.198        0.360 +- 0.043  [5 ] SI SC BT BT AB
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    0016             4552       124       142    -18             1.22        0.873 +- 0.078        1.145 +- 0.096  [4 ] SI RE RE AB
    0017            8cc52       126       138    -12             0.55        0.913 +- 0.081        1.095 +- 0.093  [5 ] SI RE BT BT SA
    0018             4662       137       121     16             0.99        1.132 +- 0.097        0.883 +- 0.080  [4 ] SI SC SC AB
    0019             4652       121       112      9             0.35        1.080 +- 0.098        0.926 +- 0.087  [4 ] SI RE SC AB
    .                              11278     11278       513.94/67 =  7.67  (pval:0.000 prob:1.000)




evt.py  rpostd01(slice(0,2))  first step distance
-----------------------------------------------------

Testing with tds3ip which is expected to give distance of 10mm::

    In [4]: b.rpostd01(slice(0,2))
    Out[4]:
    A([10.9867, 10.833 ,  8.3912,  8.3912, 10.3583, 10.6771, 10.0294,  8.9706,  9.8608,  9.8608, 10.9867,  8.3912, 10.5189, 10.5189, 10.0294,  9.8608, 10.6771, 10.833 , 10.833 ,  9.5147, 10.3583,



Variation from 10 is due to the domain compression.  Use the deluxe double precision steps for b only, gets close to 10::


    In [9]: dxd = b.dx[:,1,0,:3] - b.dx[:,0,0,:3]

    In [12]: np.sqrt(np.sum(dxd*dxd, axis=1))
    Out[12]:
    A([10.0001, 10.    , 10.0006, 10.0003,  9.9998,  9.9995,  9.9992,  9.9993,  9.9995, 10.0007, 10.0004, 10.0012, 10.0002,  9.9994, 10.0013,  9.9999,  9.9997,  9.999 , 10.0012, 10.0009, 10.    ,
       10.0017, 10.0007,  9.9999, 10.0001, 10.0001, 10.0003,  9.9993,  9.9993, 10.0003, 10.0008,  9.9993,  9.9991,  9.9992, 10.    , 10.0005,  9.9995,  9.9997,  9.9998, 10.0001, 10.0002, 10.    ,
       10.0005,  9.9998,  9.9989, 10.0004,  9.9987, 10.0012,  9.9992, 10.0008, 10.0004,  9.9993, 10.0001, 10.0001, 10.0001,  9.9996, 10.0001, 10.0004,  9.9995,  9.9998,  9.9997, 10.    ,  9.999 ,
       10.0007,  9.9998,  9.9999, 10.0003,  9.9999,  9.9986, 10.0002,  9.9994, 10.0004, 10.0004,  9.9997, 10.0005,  9.9999, 10.0013,  9.9998,  9.9997, 10.0006, 10.0009, 10.0012,  9.9998,  9.9999,




Select a suitable photon to repeat many times to compare absorption distance
----------------------------------------------------------------------------------



G4 selection of photons that mostly just go thru the acrylic and then direct to tyvek::

    epsilon:offline blyth$ tds3gun.sh 1

    In [1]: b.sel = "SI BT BT SA"

    In [2]: b.dx.shape         # the arrays honour the selection, here the deluxe g4only doubles step array 
    Out[2]: (464, 10, 2, 4)


Most but not all at the Tyvek radius::

    In [9]: p3 = b.dx[:,3,0,:3]

    In [10]: np.sqrt(np.sum(p3*p3,axis=1))
    Out[10]:
    A([20050.    , 20050.    , 20050.    , 20050.    , 20050.    , 20050.    , 20050.    , 20050.    , 20050.    , 20050.    , 20050.    , 20050.    , 20050.    , 20050.    , 20050.    , 20050.    ,
       20050.    , 20050.    , 20050.    , 20050.    , 20050.    , 20050.    , 20050.    , 20050.    , 20050.    , 20050.    , 20050.    , 20050.    , 20050.    , 20050.    , 20050.    , 20050.    ,
       20050.    , 20050.    , 20050.    , 20050.    , 20050.    , 20050.    , 20050.    , 20050.    , 20050.    , 20050.    , 20050.    , 20050.    , 20050.    , 20050.    , 20050.    , 20050.    ,
       20050.    , 20050.    , 20050.    , 20050.    , 20050.    , 20050.    , 20050.    , 20050.    , 17960.2199,


All with almost the same distance in the water, must be quite radial::

    In [14]: np.sqrt(np.sum(d23*d23, axis=1))
    Out[14]:
    A([2230.2825, 2230.6623, 2230.1289, 2230.0021, 2230.2428, 2230.0228, 2230.2414, 2230.0872, 2230.0179, 2230.1445, 2230.2377, 2230.1683, 2230.2374, 2230.1251, 2230.2369, 2230.1349, 2230.0575,
       2230.02  , 2230.1232, 2230.209 , 2230.0598, 2230.0281, 2230.2156, 2230.1955, 2230.0016, 2230.2151, 2230.159 , 2230.1254, 2230.0932, 2230.1678, 2230.125 , 2230.1332, 2230.0187, 2230.2098,
       2230.177 , 2230.1518, 2230.207 , 2230.2092, 2230.0282, 2230.0631, 2230.0306, 2230.1727, 2230.1675, 2230.159 , 2230.2057, 2230.0961, 2230.0306, 2230.0795, 2230.2116, 2230.1711, 2230.2127,
       2230.0608, 2230.2066, 2230.0243, 2230.2138, 2230.1138,  140.2281, 2230.0364, 2230.2136, 2230.1919, 2230.0731, 2230.0385, 2230.1536, 2230.0063, 2230.029 , 2230.1392, 2230.2142, 2230.0647,


Yep, all radial::

    In [15]: 20050 - 17820
    Out[15]: 2230


Pick the first photon in the selection, and back it up by 2220mm so it starts from 10mm into the water::

    In [7]: b.ox.shape
    Out[7]: (464, 4, 4)

    In [8]: b.ox[0:1]
    Out[8]:
    A([[[ 14210.083 ,   5228.8896, -13142.859 ,    110.41  ],
        [     0.7191,      0.2595,     -0.6447,      1.    ],
        [    -0.4468,     -0.538 ,     -0.7148,    415.3976],
        [     0.    ,      0.    ,      0.    ,      0.    ]]], dtype=float32)

    In [10]: ph1 = b.ox[0:1]

    In [11]: ph1.shape
    Out[11]: (1, 4, 4)

    In [12]: ph1[0,0,:3]
    Out[12]: A([ 14210.083 ,   5228.8896, -13142.859 ], dtype=float32)

    In [14]: ph1[0,0,:3] -= ph1[0,1,:3]*2220.

    In [15]:

    In [17]: m0_ = lambda p:np.sqrt(np.sum(p*p, axis=0))  # magnitude axis 0

    In [18]: m0_(ph1[0,0,:3])
    Out[18]: A(17830.281, dtype=float32)


    In [21]: np.save("/tmp/check_innerwater_bulk_absorb.npy", ph1 )


::

    scp /tmp/check_innerwater_bulk_absorb.npy P:/tmp/


::

    1107 tds3ip(){
    1108    #local name="RandomSpherical10"
    1109    #local name="CubeCorners"
    1110    #local name="CubeCorners10x10"
    1111    #local name="CubeCorners100x100"
    1112    #local path="$HOME/.opticks/InputPhotons/${name}.npy"
    1113
    1114    local path=/tmp/check_innerwater_bulk_absorb.npy
    1115
    1116    export OPTICKS_EVENT_PFX=tds3ip
    1117    export INPUT_PHOTON_PATH=$path
    1118    export INPUT_PHOTON_REPEAT=10
    1119
    1120    #tds3 --dbgseqhis 0x7ccccd   # "TO BT BT BT BT SD"
    1121    #tds3 --dindex 0,1,2,3,4,5
    1122
    1123    tds3
    1124
    1125 }





Runs fine but event loading "tds3ip.sh 1" gives error from compare shapes.::

    ~/opticks/ana/ab.py in compare_shapes(self)
        483 
        484     def compare_shapes(self):
    --> 485         assert self.a.dshape == self.b.dshape, (self.a.dshape, self.b.dshape)
        486         self.dshape = self.a.dshape
        487 

    AssertionError: (' file_photons 1   load_slice 0:100k:   loaded_photons 1 ', ' file_photons 10   load_slice 0:100k:   loaded_photons 10 ')
    > /Users/blyth/opticks/ana/ab.py(485)compare_shapes()
        483 
        484     def compare_shapes(self):
    --> 485         assert self.a.dshape == self.b.dshape, (self.a.dshape, self.b.dshape)
        486         self.dshape = self.a.dshape
        487 

    ipdb>                                                               
     


Booting without compare reveals why::

    epsilon:offline blyth$ tds3ip.sh 1 -C

    a.valid:True
    b.valid:True
    ab.valid:True
    als[:10]
    TO SA
    bls[:10]
    TO AB
    TO SA
    TO SA
    TO SA
    TO SA
    TO SA
    TO SA
    TO SA
    TO SA
    TO SA

    In [1]: a.ox.shape                                                                                                                                                             
    Out[1]: (1, 4, 4)

    In [2]: b.ox.shape                                                                                                                                                             
    Out[2]: (10, 4, 4)

    In [3]:                                


The input photon repeat instruction has no effect on the OK running. 
How do the input photons get passed to GPU propagation ?

    epsilon:offline blyth$ jgr GtOpticksTool::Get
    ./Simulation/GenTools/src/GtOpticksTool.cc:const GtOpticksTool* GtOpticksTool::Get()
    ./Simulation/DetSimV2/PMTSim/src/junoSD_PMT_v2_Opticks.cc:    const GtOpticksTool* tool = GtOpticksTool::Get(); 


::

     65 #ifdef WITH_G4OPTICKS
     66 /**
     67 junoSD_PMT_v2_Opticks::Initialize
     68 -----------------------------------
     69 
     70 HMM: this grabbing from the input is kinda cheating, 
     71 should really re-constitute from the G4Event  primaries
     72 but input_photons.py is just for debugging, so I judge this
     73 to be accepatble.
     74 
     75 **/
     76 
     77 void junoSD_PMT_v2_Opticks::Initialize(G4HCofThisEvent* /*HCE*/)
     78 {
     79     const GtOpticksTool* tool = GtOpticksTool::Get();
     80     NPY<float>* input_photons = tool ? tool->getInputPhotons() : nullptr ;
     81     G4Opticks* g4ok = G4Opticks::Get() ;
     82 
     83     LOG(info) 
     84         << " tool " << tool
     85         << " input_photons " << input_photons
     86         << " g4ok " << g4ok 
     87         ;
     88 
     89     if(input_photons)
     90     {
     91         g4ok->setInputPhotons(input_photons);
     92     }   
     93 }       


Pass the repeat along with the photons back into g4ok::

     77 void junoSD_PMT_v2_Opticks::Initialize(G4HCofThisEvent* /*HCE*/)
     78 {
     79     const GtOpticksTool* tool = GtOpticksTool::Get();
     80     NPY<float>* input_photons = tool ? tool->getInputPhotons() : nullptr ;
     81     int input_photon_repeat = tool ? tool->getInputPhotonRepeat() : 0 ; 
     82     G4Opticks* g4ok = G4Opticks::Get() ;
     83     
     84     LOG(info) 
     85         << " tool " << tool  
     86         << " input_photons " << input_photons
     87         << " input_photon_repeat " << input_photon_repeat
     88         << " g4ok " << g4ok
     89         ;
     90     
     91     if(input_photons)
     92     {   
     93         g4ok->setInputPhotons(input_photons, input_photon_repeat );
     94     }
     95 }
     96 


And act on the repeat in the carrier::

    404 OpticksGenstep* OpticksGenstep::MakeInputPhotonCarrier(NPY<float>* ip, unsigned tagoffset, int repeat ) // static
    405 {
    406     unsigned ip_num = ip->getNumItems();             
    407     NPY<float>* ipr = repeat == 0 ? ip : NPY<float>::make_repeat( ip, repeat );
    408     unsigned ipr_num = ipr->getNumItems();
    409     
    410     LOG(LEVEL)
    411         << " tagoffset " << tagoffset
    412         << " repeat " << repeat 
    413         << " ip_num " << ip_num
    414         << " ip " << ip->getShapeString()
    415         << " ipr_num " << ipr_num
    416         << " ipr " << ipr->getShapeString()
    417         ;  
    418         
    419     NStep onestep ;
    420     onestep.setGenstepType( OpticksGenstep_EMITSOURCE );
    421     onestep.setNumPhotons(  ipr_num );
    422     onestep.fillArray(); 
    423     NPY<float>* gs = onestep.getArray();
    424     
    425     
    426     bool compute = true ;
    427     ipr->setBufferSpec(OpticksEvent::SourceSpec(compute));
    428     ipr->setArrayContentIndex( tagoffset );
    429     
    430     gs->setBufferSpec(OpticksEvent::GenstepSpec(compute));
    431     gs->setArrayContentIndex( tagoffset );
    432 
    433     OpticksActionControl oac(gs->getActionControlPtr());
    434     oac.add(OpticksActionControl::GS_EMITSOURCE_);       // needed ?
    435     LOG(LEVEL) 
    436         << " gs " << gs 
    437         << " oac.desc " << oac.desc("gs")
    438         << " oac.numSet " << oac.numSet()
    439         ; 
    440 
    441     gs->setAux((void*)ipr);  // under-radar association of input photons with the fabricated genstep 
    442 
    443     OpticksGenstep* ogs = new OpticksGenstep(gs);
    444     return ogs ;
    445 }
    446 



tds3ip.sh 1::


    [{dump                :ab.py     :325} INFO     - ]
    als[:10]
    TO SA
    TO SA
    TO SA
    TO SA
    TO SA
    TO SA
    TO SA
    *TO AB
    TO SA
    *TO AB
    bls[:10]
    *TO AB
    TO SA
    TO SA
    TO SA
    TO SA
    TO SA
    TO SA
    TO SA
    TO SA
    TO SA

    In [1]: ab.his                                                                                                                                                                 
    Out[1]: 
    ab.his
    .       seqhis_ana  cfo:sum  1:g4live:tds3ip   -1:g4live:tds3ip        c2        ab        ba 
    .                                 10        10         0.00/-1 =  0.00  (pval:nan prob:nan)  
    0000               8d         8         9     -1             0.00        0.889 +- 0.314        1.125 +- 0.375  [2 ] TO SA
    0001               4d         2         1      1             0.00        2.000 +- 1.414        0.500 +- 0.500  [2 ] TO AB
    .                                 10        10         0.00/-1 =  0.00  (pval:nan prob:nan)  


    In [3]: a.rpost_(slice(0,2))                                                                                                                                                   
    Out[3]: 
    A([[[ 12614.5207,   4652.852 , -11711.7832,    110.416 ],
        [ 14209.418 ,   5229.6518, -13143.7117,    120.5969]],

       [[ 12614.5207,   4652.852 , -11711.7832,    110.416 ],
        [ 14209.418 ,   5229.6518, -13143.7117,    120.5969]],

       [[ 12614.5207,   4652.852 , -11711.7832,    110.416 ],
        [ 14209.418 ,   5229.6518, -13143.7117,    120.5969]],

       [[ 12614.5207,   4652.852 , -11711.7832,    110.416 ],
        [ 14209.418 ,   5229.6518, -13143.7117,    120.5969]],

       [[ 12614.5207,   4652.852 , -11711.7832,    110.416 ],
        [ 14209.418 ,   5229.6518, -13143.7117,    120.5969]],

       [[ 12614.5207,   4652.852 , -11711.7832,    110.416 ],
        [ 14209.418 ,   5229.6518, -13143.7117,    120.5969]],

       [[ 12614.5207,   4652.852 , -11711.7832,    110.416 ],
        [ 14209.418 ,   5229.6518, -13143.7117,    120.5969]],

       [[ 12614.5207,   4652.852 , -11711.7832,    110.416 ],
        [ 13110.7517,   4832.3008, -12156.7431,    113.5655]],

       [[ 12614.5207,   4652.852 , -11711.7832,    110.416 ],
        [ 14209.418 ,   5229.6518, -13143.7117,    120.5969]],

       [[ 12614.5207,   4652.852 , -11711.7832,    110.416 ],
        [ 13874.3248,   5108.7985, -12841.5784,    118.4729]]])

    In [4]: b.rpost_(slice(0,2))                                                                                                                                                   
    Out[4]: 
    A([[[ 12614.5207,   4652.852 , -11711.7832,    110.416 ],
        [ 13575.8538,   5000.763 , -12574.2363,    116.5319]],

       [[ 12614.5207,   4652.852 , -11711.7832,    110.416 ],
        [ 14209.418 ,   5229.6518, -13143.7117,    120.5969]],

       [[ 12614.5207,   4652.852 , -11711.7832,    110.416 ],
        [ 14209.418 ,   5229.6518, -13143.7117,    120.5969]],

       [[ 12614.5207,   4652.852 , -11711.7832,    110.416 ],
        [ 14209.418 ,   5229.6518, -13143.7117,    120.5969]],

       [[ 12614.5207,   4652.852 , -11711.7832,    110.416 ],
        [ 14209.418 ,   5229.6518, -13143.7117,    120.5969]],

       [[ 12614.5207,   4652.852 , -11711.7832,    110.416 ],
        [ 14209.418 ,   5229.6518, -13143.7117,    120.5969]],

       [[ 12614.5207,   4652.852 , -11711.7832,    110.416 ],
        [ 14209.418 ,   5229.6518, -13143.7117,    120.5969]],

       [[ 12614.5207,   4652.852 , -11711.7832,    110.416 ],
        [ 14209.418 ,   5229.6518, -13143.7117,    120.5969]],

       [[ 12614.5207,   4652.852 , -11711.7832,    110.416 ],
        [ 14209.418 ,   5229.6518, -13143.7117,    120.5969]],

       [[ 12614.5207,   4652.852 , -11711.7832,    110.416 ],
        [ 14209.418 ,   5229.6518, -13143.7117,    120.5969]]])

    In [5]: from opticks.ana.evt import m1_, m2_                        


Expected start and end radii::

    In [10]: m1_(ar.reshape(-1,4)[:,:3])                                                                                                                                           
    Out[10]: 
    A([17830.901 , 20050.2861, 17830.901 , 20050.2861, 17830.901 , 20050.2861, 17830.901 , 20050.2861, 17830.901 , 20050.2861, 17830.901 , 20050.2861, 17830.901 , 20050.2861, 17830.901 , 18521.0513,
       17830.901 , 20050.2861, 17830.901 , 19583.2287])

    In [11]: br = b.rpost_(slice(0,2))                                                                                                                                             

    In [12]: m1_(br.reshape(-1,4)[:,:3])                                                                                                                                           
    Out[12]: 
    A([17830.901 , 19168.2773, 17830.901 , 20050.2861, 17830.901 , 20050.2861, 17830.901 , 20050.2861, 17830.901 , 20050.2861, 17830.901 , 20050.2861, 17830.901 , 20050.2861, 17830.901 , 20050.2861,
       17830.901 , 20050.2861, 17830.901 , 20050.2861])

    In [13]:                                          



Increase repeat factor to 100,000 in order to compare absorption fractions::

    P[blyth@localhost cmt]$ jvi
    P[blyth@localhost cmt]$ jfu
    P[blyth@localhost cmt]$ t tds3ip
    tds3ip () 
    { 
        local path=/tmp/check_innerwater_bulk_absorb.npy;
        export OPTICKS_EVENT_PFX=tds3ip;
        export INPUT_PHOTON_PATH=$path;
        export INPUT_PHOTON_REPEAT=100000;
        tds3
    }



trips assert::

    2021-06-16 22:05:46.321 INFO  [177147] [junoSD_PMT_v2_Opticks::Initialize@84]  tool 0x1e3b340 input_photons 0x2fe4590 input_photon_repeat 100000 g4ok 0x4cdcaf0
    2021-06-16 22:05:46.321 INFO  [177147] [G4Opticks::setInputPhotons@1934]  input_photons 1,4,4 repeat 100000
    Begin of Event --> 0
    2021-06-16 22:06:43.775 INFO  [177147] [PMTEfficiencyCheck::addHitRecord@88]  m_eventID 0 m_record_count 0
    2021-06-16 22:06:43.776 FATAL [177147] [CCtx::ProcessHits@592]  _pho not equal to hit   _pho.desc CPho gs 0 ix 99766 id 99766 gn 0 hit.desc CPho (missing) 
    python: /home/blyth/opticks/cfg4/CCtx.cc:597: void CCtx::ProcessHits(const G4Step*, bool): Assertion `0' failed.

    Program received signal SIGABRT, Aborted.
    0x00007ffff6cf9387 in raise () from /lib64/libc.so.6
    (gdb) bt
    #3  0x00007ffff6cf2252 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007fffcddad75d in CCtx::ProcessHits (this=0x153ad8b90, step=0x24c9a70, efficiency_collect=false) at /home/blyth/opticks/cfg4/CCtx.cc:597
    #5  0x00007fffcddb32e4 in CManager::ProcessHits (this=0x153ad8b30, step=0x24c9a70, efficiency_collect=false) at /home/blyth/opticks/cfg4/CManager.cc:619
    #6  0x00007fffce07744c in G4OpticksRecorder::ProcessHits (this=0x2526010, step=0x24c9a70, efficiency_collect=false) at /home/blyth/opticks/g4ok/G4OpticksRecorder.cc:154
    #7  0x00007fffc233f992 in junoSD_PMT_v2::ProcessHits (this=0x34ae410, step=0x24c9a70) at ../src/junoSD_PMT_v2.cc:466
    #8  0x00007fffd04aa98c in G4SteppingManager::Stepping() () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4tracking.so
    #9  0x00007fffd04b60fd in G4TrackingManager::ProcessOneTrack(G4Track*) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4tracking.so
    #10 0x00007fffd06edb53 in G4EventManager::DoProcessing(G4Event*) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4event.so
    #11 0x00007fffc2897760 in G4SvcRunManager::SimulateEvent(int) () from /home/blyth/junotop/offline/InstallArea/Linux-x86_64/lib/libG4Svc.so


Unsure how, but for now exclude comparison for missings::

    584 void CCtx::ProcessHits( const G4Step* step, bool efficiency_collect )
    585 {   
    586     const G4Track* track = step->GetTrack();    
    587     bool fabricate_unlabelled = false ;
    588     CPho hit = CPhotonInfo::Get(track, fabricate_unlabelled); 
    589     
    590     if(!hit.is_missing())
    591     {
    592         if(!_pho.isEqual(hit))
    593         {
    594             LOG(fatal)
    595                 << " _pho not equal to hit "
    596                 << "  _pho.desc " << _pho.desc()
    597                 << " hit.desc " << hit.desc()
    598                 ;
    599             assert(0);
    600         }   
    601     }   





* comparison does not show much of an AB difference
* but does show lots more SC in G4 that in OK

  * OK has "TO SA" sail to boundary excess of 989/100,000 (1%) 
  * for G4 these are spread across various "TO SC .." histories  


tds3ip.sh::

    In [2]: ab.his[:30]                                                                                                                                                                                 
    Out[2]: 
    ab.his
    .       seqhis_ana  cfo:sum  1:g4live:tds3ip   -1:g4live:tds3ip        c2        ab        ba 
    .                             100000    100000       739.41/9 = 82.16  (pval:0.000 prob:1.000)  
    0000               8d     93766     92777    989             5.24        1.011 +- 0.003        0.989 +- 0.003  [2 ] TO SA
    0001               4d      6031      5918    113             1.07        1.019 +- 0.013        0.981 +- 0.013  [2 ] TO AB
    0002             7c6d        38       311   -273           213.55        0.122 +- 0.020        8.184 +- 0.464  [4 ] TO SC BT SD
    0003              86d        33       236   -203           153.19        0.140 +- 0.024        7.152 +- 0.466  [3 ] TO SC SA
    0004            4cc6d        10       212   -202           183.80        0.047 +- 0.015       21.200 +- 1.456  [5 ] TO SC BT BT AB
    0005             8c6d        13        80    -67            48.27        0.163 +- 0.045        6.154 +- 0.688  [4 ] TO SC BT SA
    0006              46d        20        63    -43            22.28        0.317 +- 0.071        3.150 +- 0.397  [3 ] TO SC AB
    0007          8ccac6d         0        72    -72            72.00        0.000 +- 0.000        0.000 +- 0.000  [7 ] TO SC BT SR BT BT SA
    0008           46cc6d         1        39    -38            36.10        0.026 +- 0.026       39.000 +- 6.245  [6 ] TO SC BT BT SC AB
    0009             4c6d        10        21    -11             3.90        0.476 +- 0.151        2.100 +- 0.458  [4 ] TO SC BT AB
    0010            7cc6d         2        27    -25             0.00        0.074 +- 0.052       13.500 +- 2.598  [5 ] TO SC BT BT SD
    0011       ccacccac6d         0        26    -26             0.00        0.000 +- 0.000        0.000 +- 0.000  [10] TO SC BT SR BT BT BT SR BT BT
    0012           4ccc6d         0        19    -19             0.00        0.000 +- 0.000        0.000 +- 0.000  [6 ] TO SC BT BT BT AB
    0013         7ccccc6d         9         9      0             0.00        1.000 +- 0.333        1.000 +- 0.333  [8 ] TO SC BT BT BT BT BT SD
    0014          466cc6d         1        17    -16             0.00        0.059 +- 0.059       17.000 +- 4.123  [7 ] TO SC BT BT SC SC AB
    0015            8cc6d         0        16    -16             0.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] TO SC BT BT SA
    0016        7ccc6cc6d         5        10     -5             0.00        0.500 +- 0.224        2.000 +- 0.632  [9 ] TO SC BT BT SC BT BT BT SD
    0017           8cac6d        13         0     13             0.00        0.000 +- 0.000        0.000 +- 0.000  [6 ] TO SC BT SR BT SA
    0018            7cb6d         0        10    -10             0.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] TO SC BR BT SD
    0019          46ccc6d         0         9     -9             0.00        0.000 +- 0.000        0.000 +- 0.000  [7 ] TO SC BT BT BT SC AB
    0020       7ccc66cc6d         3         4     -1             0.00        0.750 +- 0.433        1.333 +- 0.667  [10] TO SC BT BT SC SC BT BT BT SD
    0021        8ccc6cc6d         2         5     -3             0.00        0.400 +- 0.283        2.500 +- 1.118  [9 ] TO SC BT BT SC BT BT BT SA
    0022         466ccc6d         0         6     -6             0.00        0.000 +- 0.000        0.000 +- 0.000  [8 ] TO SC BT BT BT SC SC AB
    0023         4cc6cc6d         1         4     -3             0.00        0.250 +- 0.250        4.000 +- 2.000  [8 ] TO SC BT BT SC BT BT AB
    0024             866d         0         5     -5             0.00        0.000 +- 0.000        0.000 +- 0.000  [4 ] TO SC SC SA
    0025         4666cc6d         1         4     -3             0.00        0.250 +- 0.250        4.000 +- 2.000  [8 ] TO SC BT BT SC SC SC AB
    0026       7cccc6cc6d         0         5     -5             0.00        0.000 +- 0.000        0.000 +- 0.000  [10] TO SC BT BT SC BT BT BT BT SD
    0027            7c66d         0         4     -4             0.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] TO SC SC BT SD
    0028          4cccc6d         2         2      0             0.00        1.000 +- 0.707        1.000 +- 0.707  [7 ] TO SC BT BT BT BT AB
    0029          4ccac6d         0         4     -4             0.00        0.000 +- 0.000        0.000 +- 0.000  [7 ] TO SC BT SR BT BT AB
    .                             100000    100000       739.41/9 = 82.16  (pval:0.000 prob:1.000)  




Wildcard selection fails for a from dx::

    In [3]: a.sel = "*SC*"                                                                                                                                                                              
    ---------------------------------------------------------------------------
    IndexError                                Traceback (most recent call last)
    <ipython-input-3-5f83b3fe4981> in <module>
    ----> 1 a.sel = "*SC*"

    ~/opticks/ana/evt.py in _set_sel(self, arg)
       1409 
       1410         psel = self.make_selection(sel, False)
    -> 1411         self._init_selection(psel)
       1412     sel = property(_get_sel, _set_sel)
       1413 

    ~/opticks/ana/evt.py in _init_selection(self, psel)
       1312         self.wl = self.wl_[psel]
       1313         self.rx = self.rx_[psel]
    -> 1314         self.dx = self.dx_[psel]
       1315 
       1316         if not self.so_.missing and len(self.so_)>0:

    IndexError: too many indices for array: array is 0-dimensional, but 1 were indexed
    > /Users/blyth/opticks/ana/evt.py(1314)_init_selection()
       1312         self.wl = self.wl_[psel]
       1313         self.rx = self.rx_[psel]
    -> 1314         self.dx = self.dx_[psel]
       1315 
       1316         if not self.so_.missing and len(self.so_)>0:

    ipdb>                                                                                                                                                                                               




    In [1]: a.sel = "*SC*"                                                                                                                                                                              

    In [2]: a.his[:30]                                                                                                                                                                                  
    Out[2]: 
    seqhis_ana
    .                     cfo:-  1:g4live:tds3ip 
    .                                203         1.00 
    0000             7c6d        0.187          38        [4 ] TO SC BT SD
    0001              86d        0.163          33        [3 ] TO SC SA
    0002              46d        0.099          20        [3 ] TO SC AB
    0003           8cac6d        0.064          13        [6 ] TO SC BT SR BT SA
    0004             8c6d        0.064          13        [4 ] TO SC BT SA
    0005             4c6d        0.049          10        [4 ] TO SC BT AB
    0006            4cc6d        0.049          10        [5 ] TO SC BT BT AB
    0007         7ccccc6d        0.044           9        [8 ] TO SC BT BT BT BT BT SD
    0008        7ccc6cc6d        0.025           5        [9 ] TO SC BT BT SC BT BT BT SD
    0009           45cc6d        0.015           3        [6 ] TO SC BT BT RE AB
    0010       7ccc66cc6d        0.015           3        [10] TO SC BT BT SC SC BT BT BT SD
    0011          4cccc6d        0.010           2        [7 ] TO SC BT BT BT BT AB
    0012        7ccc5cc6d        0.010           2        [9 ] TO SC BT BT RE BT BT BT SD
    0013        8ccc6cc6d        0.010           2        [9 ] TO SC BT BT SC BT BT BT SA
    0014            7cc6d        0.010           2        [5 ] TO SC BT BT SD
    0015       8ccc65cc6d        0.010           2        [10] TO SC BT BT RE SC BT BT BT SA
    0016          7cccb6d        0.005           1        [7 ] TO SC BR BT BT BT SD
    0017           46cc6d        0.005           1        [6 ] TO SC BT BT SC AB
    0018           4cac6d        0.005           1        [6 ] TO SC BT SR BT AB
    0019           4ccb6d        0.005           1        [6 ] TO SC BR BT BT AB
    0020            4cb6d        0.005           1        [5 ] TO SC BR BT AB
    0021           7ccc6d        0.005           1        [6 ] TO SC BT BT BT SD
    0022             8b6d        0.005           1        [4 ] TO SC BR SA
    0023          465cc6d        0.005           1        [7 ] TO SC BT BT RE SC AB
    0024          466cc6d        0.005           1        [7 ] TO SC BT BT SC SC AB
    0025          4bcac6d        0.005           1        [7 ] TO SC BT SR BT BR AB
    0026          4c5cc6d        0.005           1        [7 ] TO SC BT BT RE BT AB
    0027             4b6d        0.005           1        [4 ] TO SC BR AB
    0028          7ccac6d        0.005           1        [7 ] TO SC BT SR BT BT SD
    0029       cc5566cc6d        0.005           1        [10] TO SC BT BT SC SC RE RE BT BT
    .                                203         1.00 




    In [4]: b.sel = "*SC*"                                       

    In [7]: b.his[:30]                                                                                                                                                                                  
    Out[7]: 
    seqhis_ana
    .                     cfo:-  -1:g4live:tds3ip 
    .                               1305         1.00 
    0000             7c6d        0.238         311        [4 ] TO SC BT SD
    0001              86d        0.181         236        [3 ] TO SC SA
    0002            4cc6d        0.162         212        [5 ] TO SC BT BT AB
    0003             8c6d        0.061          80        [4 ] TO SC BT SA
    0004          8ccac6d        0.055          72        [7 ] TO SC BT SR BT BT SA
    0005              46d        0.048          63        [3 ] TO SC AB
    0006           46cc6d        0.030          39        [6 ] TO SC BT BT SC AB
    0007            7cc6d        0.021          27        [5 ] TO SC BT BT SD
    0008       ccacccac6d        0.020          26        [10] TO SC BT SR BT BT BT SR BT BT
    0009             4c6d        0.016          21        [4 ] TO SC BT AB
    0010           4ccc6d        0.015          19        [6 ] TO SC BT BT BT AB
    0011          466cc6d        0.013          17        [7 ] TO SC BT BT SC SC AB
    0012            8cc6d        0.012          16        [5 ] TO SC BT BT SA
    0013        7ccc6cc6d        0.008          10        [9 ] TO SC BT BT SC BT BT BT SD
    0014            7cb6d        0.008          10        [5 ] TO SC BR BT SD
    0015         7ccccc6d        0.007           9        [8 ] TO SC BT BT BT BT BT SD
    0016          46ccc6d        0.007           9        [7 ] TO SC BT BT BT SC AB
    0017         466ccc6d        0.005           6        [8 ] TO SC BT BT BT SC SC AB
    0018       7cccc6cc6d        0.004           5        [10] TO SC BT BT SC BT BT BT BT SD
    0019        8ccc6cc6d        0.004           5        [9 ] TO SC BT BT SC BT BT BT SA
    0020             866d        0.004           5        [4 ] TO SC SC SA
    0021         4666cc6d        0.003           4        [8 ] TO SC BT BT SC SC SC AB
    0022       caccccac6d        0.003           4        [10] TO SC BT SR BT BT BT BT SR BT
    0023         4cc6cc6d        0.003           4        [8 ] TO SC BT BT SC BT BT AB
    0024          4ccac6d        0.003           4        [7 ] TO SC BT SR BT BT AB
    0025       7ccc66cc6d        0.003           4        [10] TO SC BT BT SC SC BT BT BT SD
    0026            7c66d        0.003           4        [5 ] TO SC SC BT SD
    0027           7c6b6d        0.002           3        [6 ] TO SC BR SC BT SD
    0028         8ccacc6d        0.002           3        [8 ] TO SC BT BT SR BT BT SA
    0029             8b6d        0.002           3        [4 ] TO SC BR SA
    .                               1305         1.00 



Wow, drastically more water SC in G4:1305 than OK:203 ?
---------------------------------------------------------


::

    087 __device__ int propagate_to_boundary( Photon& p, State& s, curandState &rng)
     88 {           
     89     //float speed = SPEED_OF_LIGHT/s.material1.x ;    // .x:refractive_index    (phase velocity of light in medium)
     90     float speed = s.m1group2.x ;  // .x:group_velocity  (group velocity of light in the material) see: opticks-find GROUPVEL
     91 
     92 #ifdef WITH_ALIGN_DEV
     93 #ifdef WITH_LOGDOUBLE
     94             
     95     float u_boundary_burn = curand_uniform(&rng) ;
     96     float u_scattering = curand_uniform(&rng) ;   
     97     float u_absorption = curand_uniform(&rng) ;
     98         
     99     //  these two doubles brings about 100 lines of PTX with .f64
    100     //  see notes/issues/AB_SC_Position_Time_mismatch.rst      
    101     float scattering_distance = -s.material1.z*log(double(u_scattering)) ;   // .z:scattering_length
    102     float absorption_distance = -s.material1.y*log(double(u_absorption)) ;   // .y:absorption_length 
    103 


::

      63 const char* GMaterialLib::keyspec =
      64 "refractive_index:RINDEX,"
      65 "absorption_length:ABSLENGTH,"
      66 "scattering_length:RAYLEIGH,"
      67 "reemission_prob:REEMISSIONPROB,"
      68 "group_velocity:GROUPVEL,"
      69 "extra_y:EXTRA_Y,"
      70 "extra_z:EXTRA_Z,"
      71 "extra_w:EXTRA_W,"
      72 "detect:EFFICIENCY,"
      73 ;




Check the boundary array::

    In [5]: a.bn.view(np.int8)                                                                                                                                                                          
    Out[5]: 
    A([[[16,  0,  0, ...,  0,  0,  0]],

       [[16,  0,  0, ...,  0,  0,  0]],

       [[16,  0,  0, ...,  0,  0,  0]],

       ...,

       [[16,  0,  0, ...,  0,  0,  0]],

       [[16,  0,  0, ...,  0,  0,  0]],

       [[16,  0,  0, ...,  0,  0,  0]]], dtype=int8)


    In [17]: blib.format([16])                                                                                                                                                                          
    Out[17]: ' 16 : Tyvek//Implicit_RINDEX_NoRINDEX_pInnerWater_pCentralDetector/Water'



::


     25 enum {  
     26     OMAT,
     27     OSUR,
     28     ISUR, 
     29     IMAT  
     30 };


     32 __device__ void fill_state( State& s, int boundary, uint4 identity, float wavelength )
     33 {   
     34     // boundary : 1 based code, signed by cos_theta of photon direction to outward geometric normal
     35     // >0 outward going photon
     36     // <0 inward going photon 
     37     //
     38     // NB the line is above the details of the payload (ie how many float4 per matsur) 
     39     //    it is just 
     40     //                boundaryIndex*4  + 0/1/2/3     for OMAT/OSUR/ISUR/IMAT 
     41     //
     42     
     43     int line = boundary > 0 ? (boundary - 1)*BOUNDARY_NUM_MATSUR : (-boundary - 1)*BOUNDARY_NUM_MATSUR  ;

     ///
     /// for boundary 16   ' 16 : Tyvek//Implicit_RINDEX_NoRINDEX_pInnerWater_pCentralDetector/Water'
     /// 
     ///          line = (16-1)*4 = 60
     ///          m1_line = 60 + IMAT = 63 
     ///          m2_line = 60 + OMAT = 60 
     ///
     ///  +ve boundary means photons are travelling in same direction as the outward going normal to the geometry 
     ///   so IMAT comes first and is m1 and OMAT is m2 
     ///

     44     
     45     // pick relevant lines depening on boundary sign, ie photon direction relative to normal
     46     // 
     47     int m1_line = boundary > 0 ? line + IMAT : line + OMAT ;
     48     int m2_line = boundary > 0 ? line + OMAT : line + IMAT ;   
     49     int su_line = boundary > 0 ? line + ISUR : line + OSUR ;   
     50 
     51     //  consider photons arriving at PMT cathode surface
     52     //  geometry normals are expected to be out of the PMT 
     53     //
     54     //  boundary sign will be -ve : so line+3 outer-surface is the relevant one
     55 
     56     s.material1 = boundary_lookup( wavelength, m1_line, 0);
     57     s.m1group2  = boundary_lookup( wavelength, m1_line, 1);
     58 
     59     s.material2 = boundary_lookup( wavelength, m2_line, 0);
     60     s.surface   = boundary_lookup( wavelength, su_line, 0);
     61 
     62     s.optical = optical_buffer[su_line] ;   // index/type/finish/value
     63 
     64     s.index.x = optical_buffer[m1_line].x ; // m1 index
     65     s.index.y = optical_buffer[m2_line].x ; // m2 index 
     66     s.index.z = optical_buffer[su_line].x ; // su index
     67     s.index.w = identity.w   ;
     68 
     69     s.identity = identity ;
     70 
     71 }


::

    epsilon:optickscore blyth$ opticks-f BOUNDARY_NUM_MATSUR
    ./ggeo/GPropertyLib.cc:unsigned int GPropertyLib::NUM_MATSUR = BOUNDARY_NUM_MATSUR  ;    // 4 material/surfaces that comprise a boundary om-os-is-im 
    ./ggeo/GPropertyLib.hh:#define BOUNDARY_NUM_MATSUR 4
    ./optixrap/cu/wavelength_lookup.h:     unsigned int line = ibnd*BOUNDARY_NUM_MATSUR + jqwn ; 
    ./optixrap/cu/boundary_lookup.h:    unsigned nj = BOUNDARY_NUM_MATSUR ;     
    ./optixrap/cu/state.h:    int line = boundary > 0 ? (boundary - 1)*BOUNDARY_NUM_MATSUR : (-boundary - 1)*BOUNDARY_NUM_MATSUR  ; 
    ./optixrap/tests/boundaryLookupTest.cc:    unsigned eight = BOUNDARY_NUM_MATSUR*BOUNDARY_NUM_FLOAT4 ; 
    ./optixrap/tests/cu/boundaryLookupTest.cu:    for(unsigned j=0 ; j < BOUNDARY_NUM_MATSUR ; j++){
    ./optixrap/tests/cu/boundaryLookupTest.cu:    unsigned nj = BOUNDARY_NUM_MATSUR ;
    ./optixrap/tests/cu/interpolationTest.cu:    unsigned nj = BOUNDARY_NUM_MATSUR ;
    ./optixrap/tests/cu/interpolationTest.cu:    unsigned nj = BOUNDARY_NUM_MATSUR ;
    epsilon:opticks blyth$ 




::

    epsilon:ggeo blyth$ jgr RAYLEIGH
    ./Simulation/DetSimV2/PhysiSim/src/DsG4OpRayleigh.cc:                            aMaterialPropertiesTable->GetProperty("RAYLEIGH");
    ./Simulation/DetSimV2/PhysiSim/src/DsG4OpRayleigh.cc:                   aMaterialPropertyTable->GetProperty("RAYLEIGH");
    ./Simulation/DetSimV2/MCParamsSvc/share/filldb.C:        " FASTCOMPONENT, REEMISSIONPROB, RAYLEIGH, "
    ./Simulation/DetSimV2/MCParamsSvc/share/filldb.C:        " '%s', '%s', '%s', " // FASTCOMPONENT, REEMISSIONPROB, RAYLEIGH,
    ./Simulation/DetSimV2/MCParamsSvc/share/filldb.C:    TString rayleigh = load("Material.LS.RAYLEIGH");
    ./Simulation/DetSimV2/MCParamsSvc/share/filldb.C:           fastc.Data(), reem.Data(), rayleigh.Data(), // FASTCOMPONENT, REEMISSIONPROB, RAYLEIGH,
    ./Simulation/DetSimV2/MCParamsSvc/share/gen_all.py:    ("Material.LS.RAYLEIGH", "vec_d2d"),
    ./Simulation/DetSimV2/MCParamsSvc/share/mc.json:    "objectType": "Material.LS.RAYLEIGH",
    ./Simulation/DetSimV2/MCParamsSvc/share/create.sql:  `RAYLEIGH` longblob COMMENT '',
    ./Simulation/DetSimV2/MCParamsSvc/src/MCParamsDBSvc.cc:    {"Material.LS.RAYLEIGH", "vec_d2d"},
    ./Simulation/DetSimV2/MCParamsSvc/src/test/TestAlg.cc:    st = m_params_svc->Get("Material.LS.RAYLEIGH", LS_rayleigh);
    ./Simulation/DetSimV2/MCParamsSvc/src/test/TestAlg.cc:    if (st) { LogInfo << "LS.RAYLEIGH: " << LS_rayleigh.size() << std::endl; }
    ./Simulation/DetSimV2/MCParamsSvc/src/test/TestAlg.cc:    save_it("LS_RAYLEIGH", LS_rayleigh);
    ./Simulation/DetSimV2/CalibUnit/share/LS.gdml:    <matrix coldim="2" name="RAYLEIGH0x252d220" values="1.55e-06 357143 
    ./Simulation/DetSimV2/CalibUnit/share/LS.gdml:      <property name="RAYLEIGH" ref="RAYLEIGH0x252d220"/>
    ./Simulation/DetSimV2/DetSimOptions/src/LSExpDetectorConstructionMaterial.icc:        LSMPT->AddProperty("RAYLEIGH", GdLSRayEnergy, GdLSRayLength, 11);
    ./Simulation/DetSimV2/DetSimOptions/src/LSExpDetectorConstructionMaterial.icc:                G4cout << "Scale RAYLEIGH from " << LS_scales_map["RayleighLenBefore"]
    ./Simulation/DetSimV2/DetSimOptions/src/LSExpDetectorConstructionMaterial.icc:            helper_mpt(LSMPT, "RAYLEIGH",                   mcgt.data(), "Material.LS.RAYLEIGH", scale_rayleigh);
    ./Simulation/DetSimV2/DetSimOptions/src/LSExpDetectorConstructionMaterial.icc:        LABMPT->AddProperty("RAYLEIGH", GdLSRayEnergy, GdLSRayLength, 11);
    ./Simulation/DetSimV2/DetSimOptions/src/LSExpDetectorConstructionMaterial.icc:       // AcrylicMPT->AddProperty("RAYLEIGH", AcrylicRayEnergy, AcrylicRayLength, 11);
    ./Simulation/DetSimV2/DetSimOptions/src/LSExpDetectorConstructionMaterial.icc:       // AcrylicMaskMPT->AddProperty("RAYLEIGH", AcrylicRayEnergy, AcrylicRayLength, 11);
    ./Simulation/DetSimV2/DetSimOptions/src/LSExpDetectorConstructionMaterial.icc:        MylarMPT->AddProperty("RAYLEIGH",AcrylicRayEnergy,RayleighLengthMylar,11);
    ./Simulation/DetSimV2/AnalysisCode/src/OpticalParameterAnaMgr.cc:        // RAYLEIGH
    ./Simulation/DetSimV2/AnalysisCode/src/OpticalParameterAnaMgr.cc:        get_matprop(tbl_LS, "RAYLEIGH", LS_Rayleigh_n, LS_Rayleigh_energy, LS_Rayleigh_len);
    epsilon:offline blyth$ jcv DsG4OpRayleigh
    2 files to edit
    ./Simulation/DetSimV2/PhysiSim/include/DsG4OpRayleigh.h
    ./Simulation/DetSimV2/PhysiSim/src/DsG4OpRayleigh.cc
    epsilon:offline blyth$ 




Notice special casing of material named "Water", that smells a bit fishy. Because it is on-the-fly 
changing properties without changing the material.::

    223 void DsG4OpRayleigh::BuildThePhysicsTable()
    224 {
    225 //      Builds a table of scattering lengths for each material
    226         
    227         if (thePhysicsTable) return;
    228         
    229         const G4MaterialTable* theMaterialTable=
    230                                G4Material::GetMaterialTable();
    231         G4int numOfMaterials = G4Material::GetNumberOfMaterials();
    232         
    233         // create a new physics table
    234         
    235         thePhysicsTable = new G4PhysicsTable(numOfMaterials);
    236         
    237         // loop for materials
    238         
    239         for (G4int i=0 ; i < numOfMaterials; i++)
    240         {   
    241             G4PhysicsOrderedFreeVector* ScatteringLengths = NULL;
    242             
    243             G4MaterialPropertiesTable *aMaterialPropertiesTable =
    244                          (*theMaterialTable)[i]->GetMaterialPropertiesTable();
    245                                          
    246             if(aMaterialPropertiesTable){
    247               
    248               G4MaterialPropertyVector* AttenuationLengthVector =
    249                             aMaterialPropertiesTable->GetProperty("RAYLEIGH");
    250               
    251               if(!AttenuationLengthVector){
    252                 
    253                 if ((*theMaterialTable)[i]->GetName() == "Water")
    254                 {  
    255                    // Call utility routine to Generate
    256                    // Rayleigh Scattering Lengths
    257                    
    258                    DefaultWater = true;
    259                    
    260                    ScatteringLengths =
    261                    RayleighAttenuationLengthGenerator(aMaterialPropertiesTable);
    262                 }
    263               }
    264             }
    265             
    266             thePhysicsTable->insertAt(i,ScatteringLengths);
    267         }
    268 }
    269 


This looks most odd on several counts:

1. AttenuationLengthVector from RAYLEIGH property not used, other than existance
2. ScatteringLengths is repeatedly used across multiple materials
3. what you get will depend on the ordering of the Water material wrt to the others


More water special casing::

    273 G4double DsG4OpRayleigh::GetMeanFreePath(const G4Track& aTrack,
    274                                      G4double ,
    275                                      G4ForceCondition* )
    276 {
    277         const G4DynamicParticle* aParticle = aTrack.GetDynamicParticle();
    278         const G4Material* aMaterial = aTrack.GetMaterial();
    279 
    280         G4double thePhotonEnergy = aParticle->GetTotalEnergy();
    281 
    282         G4double AttenuationLength = DBL_MAX;
    283 
    284         if ((strcmp(aMaterial->GetName(), "Water") == 0 )
    285             && DefaultWater){
    286 
    287            G4bool isOutRange;
    288 
    289            AttenuationLength =
    290                 (*thePhysicsTable)(aMaterial->GetIndex())->
    291                            GetValue(thePhotonEnergy, isOutRange);

    ////    for "Water" get AttenuationLength from thePhysicsTable 
    ////    otherwise do the more standard lookup from material properties

    292         }
    293         else {
    294 
    295            G4MaterialPropertiesTable* aMaterialPropertyTable =
    296                            aMaterial->GetMaterialPropertiesTable();
    297 
    298            if(aMaterialPropertyTable){
    299              G4MaterialPropertyVector* AttenuationLengthVector =
    300                    aMaterialPropertyTable->GetProperty("RAYLEIGH");
    301              if(AttenuationLengthVector){
    302                AttenuationLength = AttenuationLengthVector ->
    303                                     GetProperty(thePhotonEnergy);
    304              }
    305              else{
    306 //               G4cout << "No Rayleigh scattering length specified" << G4endl;
    307              }
    308            }
    309            else{
    310 //             G4cout << "No Rayleigh scattering length specified" << G4endl; 
    311            }
    312         }
    313 
    314         return AttenuationLength;
    315 }



But above not used, are using standard G4::

    jcv DsPhysConsOptical

    221     G4OpRayleigh* rayleigh = 0;
    222     if (m_useRayleigh) {
    223         rayleigh = new G4OpRayleigh();
    224     //        rayleigh->SetVerboseLevel(2);
    225     }
    226 


g4-cls G4OpRayleigh::


    215 // BuildPhysicsTable for the Rayleigh Scattering process
    216 // --------------------------------------------------------
    217 void G4OpRayleigh::BuildPhysicsTable(const G4ParticleDefinition&)
    218 {
    219   if (thePhysicsTable) {
    220      thePhysicsTable->clearAndDestroy();
    221      delete thePhysicsTable;
    222      thePhysicsTable = NULL;
    223   }
    224 
    225   const G4MaterialTable* theMaterialTable = G4Material::GetMaterialTable();
    226   const G4int numOfMaterials = G4Material::GetNumberOfMaterials();
    227 
    228   thePhysicsTable = new G4PhysicsTable( numOfMaterials );
    229 
    230   for( G4int iMaterial = 0; iMaterial < numOfMaterials; iMaterial++ )
    231   {
    232       G4Material* material = (*theMaterialTable)[iMaterial];
    233       G4MaterialPropertiesTable* materialProperties =
    234                                        material->GetMaterialPropertiesTable();
    235       G4PhysicsOrderedFreeVector* rayleigh = NULL;
    236       if ( materialProperties != NULL ) {
    237          rayleigh = materialProperties->GetProperty( kRAYLEIGH );
    238          if ( rayleigh == NULL ) rayleigh =
    239                                    CalculateRayleighMeanFreePaths( material );
    240       }
    241       thePhysicsTable->insertAt( iMaterial, rayleigh );
    242   }
    243 }

::

    epsilon:offline blyth$ g4-hh kRAYLEIGH
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/materials/include/G4MaterialPropertiesIndex.hh:  kRAYLEIGH,                   // Rayleigh scattering attenuation length
    epsilon:offline blyth$ g4-cc kRAYLEIGH
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/processes/optical/src/G4OpRayleigh.cc:         rayleigh = materialProperties->GetProperty( kRAYLEIGH );
    epsilon:offline blyth$ 

    epsilon:offline blyth$ g4-cc RAYLEIGH
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/materials/src/G4MaterialPropertiesTable.cc:  G4MaterialPropertyName.push_back(G4String("RAYLEIGH"));
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/processes/optical/src/G4OpRayleigh.cc:         rayleigh = materialProperties->GetProperty( kRAYLEIGH );




::

     39 
     40 enum G4MaterialPropertyIndex   {
     41   kNullPropertyIndex = -1,     // the number of G4MaterialPropertyIndex
     42   kRINDEX,                     // index of refraction                  
     43   kREFLECTIVITY,               // reflectivity         
     44   kREALRINDEX,                 // real part of the refractive index
     45   kIMAGINARYRINDEX,            // imaginary part of the refractive index
     46   kEFFICIENCY,                 // efficiency 
     47   kTRANSMITTANCE,              // transmittance of a dielectric surface
     48   kSPECULARLOBECONSTANT,       // reflection probability about the normal of a micro facet. 
     49   kSPECULARSPIKECONSTANT,      // reflection probability about the average surface normal
     50   kBACKSCATTERCONSTANT,        // for the case of several reflections within a deep groove
     51   kGROUPVEL,                   // group velocity
     52   kMIEHG,                      // Mie scattering length
     53   kRAYLEIGH,                   // Rayleigh scattering attenuation length
     54   kWLSCOMPONENT,               // the relative emission spectrum of the material as a function of the photon's momentum
     55   kWLSABSLENGTH,               // the absorption length of the material as a function of the photon's momentum
     56   kABSLENGTH,                  // the absorption length
     57   kFASTCOMPONENT,              // fast component of scintillation
     58   kSLOWCOMPONENT,              // slow component of scintillation
     59   kPROTONSCINTILLATIONYIELD,   // scintillation light yield by protons  
     60   kDEUTERONSCINTILLATIONYIELD, // scintillation light yield by deuterons
     61   kTRITONSCINTILLATIONYIELD,   // scintillation light yield by tritons
     62   kALPHASCINTILLATIONYIELD,    // scintillation light yield by alphas
     63   kIONSCINTILLATIONYIELD,      // scintillation light yield by ions
     64   kELECTRONSCINTILLATIONYIELD, // scintillation light yield by electrons
     65   kNumberOfPropertyIndex       // the number of G4MaterialPropertyIndex
     66 } ;


     60 G4MaterialPropertiesTable::G4MaterialPropertiesTable()
     61 {
     62   // elements of these 2 vectors must be in same order as
     63   // the corresponding enums in G4MaterialPropertiesIndex.hh
     64   G4MaterialPropertyName.push_back(G4String("RINDEX"));
     65   G4MaterialPropertyName.push_back(G4String("REFLECTIVITY"));
     66   G4MaterialPropertyName.push_back(G4String("REALRINDEX"));
     67   G4MaterialPropertyName.push_back(G4String("IMAGINARYRINDEX"));
     68   G4MaterialPropertyName.push_back(G4String("EFFICIENCY"));
     69   G4MaterialPropertyName.push_back(G4String("TRANSMITTANCE"));
     70   G4MaterialPropertyName.push_back(G4String("SPECULARLOBECONSTANT"));
     71   G4MaterialPropertyName.push_back(G4String("SPECULARSPIKECONSTANT"));
     72   G4MaterialPropertyName.push_back(G4String("BACKSCATTERCONSTANT"));
     73   G4MaterialPropertyName.push_back(G4String("GROUPVEL"));
     74   G4MaterialPropertyName.push_back(G4String("MIEHG"));
     75   G4MaterialPropertyName.push_back(G4String("RAYLEIGH"));
     76   G4MaterialPropertyName.push_back(G4String("WLSCOMPONENT"));
     77   G4MaterialPropertyName.push_back(G4String("WLSABSLENGTH"));
     78   G4MaterialPropertyName.push_back(G4String("ABSLENGTH"));
     79   G4MaterialPropertyName.push_back(G4String("FASTCOMPONENT"));
     80   G4MaterialPropertyName.push_back(G4String("SLOWCOMPONENT"));
     81   G4MaterialPropertyName.push_back(G4String("PROTONSCINTILLATIONYIELD"));
     82   G4MaterialPropertyName.push_back(G4String("DEUTERONSCINTILLATIONYIELD"));
     83   G4MaterialPropertyName.push_back(G4String("TRITONSCINTILLATIONYIELD"));
     84   G4MaterialPropertyName.push_back(G4String("ALPHASCINTILLATIONYIELD"));
     85   G4MaterialPropertyName.push_back(G4String("IONSCINTILLATIONYIELD"));
     86   G4MaterialPropertyName.push_back(G4String("ELECTRONSCINTILLATIONYIELD"));

::

    223 G4MaterialPropertyVector*
    224 G4MaterialPropertiesTable::GetProperty(const char *key, G4bool warning)
    225 {
    226   // Returns a Material Property Vector corresponding to a key
    227   const G4int index = GetPropertyIndex(G4String(key), warning);
    228   return GetProperty(index);
    229 }



::

    epsilon:offline blyth$ jgr RAYLEIGH
    ./Simulation/DetSimV2/PhysiSim/src/DsG4OpRayleigh.cc:                            aMaterialPropertiesTable->GetProperty("RAYLEIGH");
    ./Simulation/DetSimV2/PhysiSim/src/DsG4OpRayleigh.cc:                   aMaterialPropertyTable->GetProperty("RAYLEIGH");
    These are not used currently it seems  

    ./Simulation/DetSimV2/MCParamsSvc/share/filldb.C:        " FASTCOMPONENT, REEMISSIONPROB, RAYLEIGH, "
    ./Simulation/DetSimV2/MCParamsSvc/share/filldb.C:        " '%s', '%s', '%s', " // FASTCOMPONENT, REEMISSIONPROB, RAYLEIGH,
    ./Simulation/DetSimV2/MCParamsSvc/share/filldb.C:    TString rayleigh = load("Material.LS.RAYLEIGH");
    ./Simulation/DetSimV2/MCParamsSvc/share/filldb.C:           fastc.Data(), reem.Data(), rayleigh.Data(), // FASTCOMPONENT, REEMISSIONPROB, RAYLEIGH,
    ./Simulation/DetSimV2/MCParamsSvc/share/gen_all.py:    ("Material.LS.RAYLEIGH", "vec_d2d"),
    ./Simulation/DetSimV2/MCParamsSvc/share/mc.json:    "objectType": "Material.LS.RAYLEIGH",
    ./Simulation/DetSimV2/MCParamsSvc/share/create.sql:  `RAYLEIGH` longblob COMMENT '',
    ./Simulation/DetSimV2/MCParamsSvc/src/MCParamsDBSvc.cc:    {"Material.LS.RAYLEIGH", "vec_d2d"},
    ./Simulation/DetSimV2/MCParamsSvc/src/test/TestAlg.cc:    st = m_params_svc->Get("Material.LS.RAYLEIGH", LS_rayleigh);
    ./Simulation/DetSimV2/MCParamsSvc/src/test/TestAlg.cc:    if (st) { LogInfo << "LS.RAYLEIGH: " << LS_rayleigh.size() << std::endl; }
    ./Simulation/DetSimV2/MCParamsSvc/src/test/TestAlg.cc:    save_it("LS_RAYLEIGH", LS_rayleigh);
    ./Simulation/DetSimV2/CalibUnit/share/LS.gdml:    <matrix coldim="2" name="RAYLEIGH0x252d220" values="1.55e-06 357143 
    ./Simulation/DetSimV2/CalibUnit/share/LS.gdml:      <property name="RAYLEIGH" ref="RAYLEIGH0x252d220"/>
    ./Simulation/DetSimV2/DetSimOptions/src/LSExpDetectorConstructionMaterial.icc:        LSMPT->AddProperty("RAYLEIGH", GdLSRayEnergy, GdLSRayLength, 11);
    ./Simulation/DetSimV2/DetSimOptions/src/LSExpDetectorConstructionMaterial.icc:                G4cout << "Scale RAYLEIGH from " << LS_scales_map["RayleighLenBefore"]
    ./Simulation/DetSimV2/DetSimOptions/src/LSExpDetectorConstructionMaterial.icc:            helper_mpt(LSMPT, "RAYLEIGH",                   mcgt.data(), "Material.LS.RAYLEIGH", scale_rayleigh);
    ./Simulation/DetSimV2/DetSimOptions/src/LSExpDetectorConstructionMaterial.icc:        LABMPT->AddProperty("RAYLEIGH", GdLSRayEnergy, GdLSRayLength, 11);
    ./Simulation/DetSimV2/DetSimOptions/src/LSExpDetectorConstructionMaterial.icc:       // AcrylicMPT->AddProperty("RAYLEIGH", AcrylicRayEnergy, AcrylicRayLength, 11);
    ./Simulation/DetSimV2/DetSimOptions/src/LSExpDetectorConstructionMaterial.icc:       // AcrylicMaskMPT->AddProperty("RAYLEIGH", AcrylicRayEnergy, AcrylicRayLength, 11);
    ./Simulation/DetSimV2/DetSimOptions/src/LSExpDetectorConstructionMaterial.icc:        MylarMPT->AddProperty("RAYLEIGH",AcrylicRayEnergy,RayleighLengthMylar,11);
    ./Simulation/DetSimV2/AnalysisCode/src/OpticalParameterAnaMgr.cc:        // RAYLEIGH
    ./Simulation/DetSimV2/AnalysisCode/src/OpticalParameterAnaMgr.cc:        get_matprop(tbl_LS, "RAYLEIGH", LS_Rayleigh_n, LS_Rayleigh_energy, LS_Rayleigh_len);
    epsilon:offline blyth$ 


Looks like water RAYLEIGH never gets set::

    epsilon:offline blyth$ grep RAYLEIGH Simulation/DetSimV2/DetSimOptions/src/LSExpDetectorConstructionMaterial.icc
            LSMPT->AddProperty("RAYLEIGH", GdLSRayEnergy, GdLSRayLength, 11);
                    G4cout << "Scale RAYLEIGH from " << LS_scales_map["RayleighLenBefore"]
                helper_mpt(LSMPT, "RAYLEIGH",                   mcgt.data(), "Material.LS.RAYLEIGH", scale_rayleigh);
            LABMPT->AddProperty("RAYLEIGH", GdLSRayEnergy, GdLSRayLength, 11);
           // AcrylicMPT->AddProperty("RAYLEIGH", AcrylicRayEnergy, AcrylicRayLength, 11);
           // AcrylicMaskMPT->AddProperty("RAYLEIGH", AcrylicRayEnergy, AcrylicRayLength, 11);
            MylarMPT->AddProperty("RAYLEIGH",AcrylicRayEnergy,RayleighLengthMylar,11);
    epsilon:offline blyth$ 



::

    104 void X4MaterialTable::init()
    105 {
    106     unsigned num_input_materials = m_input_materials.size() ;
    107 
    108     LOG(LEVEL) << ". G4 nmat " << num_input_materials ;
    109 
    110     for(unsigned i=0 ; i < num_input_materials ; i++)
    111     {
    112         G4Material* material = m_input_materials[i] ;
    113         G4MaterialPropertiesTable* mpt = material->GetMaterialPropertiesTable();
    114 
    115         if( mpt == NULL )
    116         {
    117             LOG(error) << "PROCEEDING TO convert material with no mpt " << material->GetName() ;
    118             // continue ;  
    119         }
    120         else
    121         {
    122             LOG(LEVEL) << " converting material with mpt " <<  material->GetName() ;
    123         }
    124 
    125 
    126         GMaterial* mat = X4Material::Convert( material );
    127         if(mat->hasProperty("EFFICIENCY"))
    128         {
    129              m_materials_with_efficiency.push_back(material);
    130         }
    131 
    132         //assert( mat->getIndex() == i ); // this is not the lib, no danger of triggering a close
    133 
    134         m_mlib->add(mat) ;    // creates standardized material
    135         m_mlib->addRaw(mat) ; // stores as-is
    136     }
    137 }



Suspect that lack of RAYLEIGH property for "Water" means that 
G4 is using a calculation from the RINDEX and some constants and 
OK is using an arbitrary and very small default.::

    268 G4PhysicsOrderedFreeVector*
    269 G4OpRayleigh::CalculateRayleighMeanFreePaths( const G4Material* material ) const
    270 {
    271   G4MaterialPropertiesTable* materialProperties =
    272                                        material->GetMaterialPropertiesTable();
    273 
    274   // Retrieve the beta_T or isothermal compressibility value. For backwards
    275   // compatibility use a constant if the material is "Water". If the material
    276   // doesn't have an ISOTHERMAL_COMPRESSIBILITY constant then return
    277   G4double betat;
    278   if ( material->GetName() == "Water" )
    279     betat = 7.658e-23*m3/MeV;
    280   else if(materialProperties->ConstPropertyExists("ISOTHERMAL_COMPRESSIBILITY"))
    281     betat = materialProperties->GetConstProperty(kISOTHERMAL_COMPRESSIBILITY);
    282   else
    283     return NULL;
    284         


TODO:

1. confirm these by introspecting GMaterialLib "Water" properties and doing some G4 dumping 
2. try to grab the result of the G4 calculation and get it into GMaterialLib for use on GPU  

Note that other materials that lack properties can have similar problems.::

    077 class G4OpRayleigh : public G4VDiscreteProcess
    ...
    122         G4PhysicsTable* GetPhysicsTable() const;





