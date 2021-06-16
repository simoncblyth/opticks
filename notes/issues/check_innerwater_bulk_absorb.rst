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





Ran fine but "tds3ip.sh 1" gives error from compare shapes.::

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


Pass the repeat along with the photons backinto g4ok::

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



