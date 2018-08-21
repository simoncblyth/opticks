OKG4Test_no_OK_hits_again
===========================

Following recent changes are not getting no OK event hits any more.
Actually on G4 side are getting hits, but thats not yet reflected 
within the OpticksEvent for G4.

* this was working very recently (or was that only precache) ? 
* checking precache with ckm-- also see no OK hits 


After adopt X4::GDMLName in X4PhysicalVolume::addBoundary
-------------------------------------------------------------

Have regained sensor surfaces::

    epsilon:issues blyth$ dbgtex.py 

    [[[  1   0   0   0]
      [  0   0   0   0]
      [  0   0   0   0]
      [  1   0   0   0]]

     [[  1   0   0   0]
      [  0   0   0   0]
      [  0   0   0   0]
      [  2   0   0   0]]

     [[  2   0   0   0]
      [  5   0   3 100]
      [  5   0   3 100]
      [  3   0   0   0]]]
    epsilon:issues blyth$ 

::

    epsilon:1 blyth$ cd dbgtex/
    epsilon:dbgtex blyth$ l
    total 48
    -rw-r--r--  1 blyth  staff     90 Aug 21 23:04 bnd.txt
    -rw-r--r--  1 blyth  staff    272 Aug 21 23:04 obuf.npy
    -rw-r--r--  1 blyth  staff  15072 Aug 21 23:04 buf.npy
    epsilon:dbgtex blyth$ cat bnd.txt 
    Air///Air
    Air///Water
    Water/Det0x110ebadb0SensorSurface/Det0x110ebadb0SensorSurface/Glass
    epsilon:dbgtex blyth$ 


And in ckm (precache) have some hits::

    2018-08-21 23:04:13.811 ERROR [185967] [OEvent::downloadHits@398] OEvent::downloadHits.cpho
    2018-08-21 23:04:13.812 ERROR [185967] [OEvent::downloadHits@400] OEvent::downloadHits.cpho DONE 
    TBuf::TBuf.m_spec : dev_ptr 0x700943000 size 75 num_bytes 4800 hexdump 0 
    TBuf::TBuf.m_spec : dev_ptr 0x700242000 size 9 num_bytes 576 hexdump 0 
    2018-08-21 23:04:13.812 FATAL [185967] [OpPropagator::propagate@67] OpPropagator::propagate(1) DONE nhit: 9
    2018-08-21 23:04:13.812 ERROR [185967] [OpticksEvent::save@1582] skip as CanAnalyse returns false 
    2018-08-21 23:04:13.812 INFO  [185967] [OpticksEvent::save@1593] OpticksEvent::save  id: 1 typ: natural tag: 1 det: g4live cat:  udet: g4live num_photons: 75 num_source : 0 genstep 4,6,4 nopstep 0,4,4 photon 75,4,4 source NULL record 75,10,2,4 phosel 75,1,4 recsel 75,10,1,4 sequence 75,1,2 seed 75,1,1 hit 9,4,4 dir /tmp/blyth/opticks/evt/g4live/natural/1
    2018-08-21 23:04:13.812 INFO  [185967] [OpticksEvent::saveNopstepData@1652] saveNopstepData zero nop 
    2018-08-21 23:04:13.812 INFO  [185967] [NPY<float>::dump@1687] OpticksEvent::save (nopstep) (0,4,4) 

ckm saves into /tmp/blyth/opticks/evt/g4live/natural/1::

    epsilon:1 blyth$ np.py 
    /private/tmp/blyth/opticks/evt/g4live/natural/1
            ./report.txt : 38 
                ./ps.npy : (75, 1, 4) 
                ./ht.npy : (9, 4, 4) 
                ./rx.npy : (75, 10, 2, 4) 
              ./fdom.npy : (3, 1, 4) 
                ./ox.npy : (75, 4, 4) 
                ./gs.npy : (4, 6, 4) 
                ./rs.npy : (75, 10, 1, 4) 
                ./ph.npy : (75, 1, 2) 
              ./idom.npy : (1, 1, 4) 
    ./20180821_230413/report.txt : 38 
    epsilon:1 blyth$ 


Postcache too::

    epsilon:torch blyth$ np.py 
    /private/tmp/blyth/opticks/evt/g4live/torch
           ./Opticks.npy : (34, 1, 4) 
         ./-1/report.txt : 35 
             ./-1/ps.npy : (54, 1, 4) 
             ./-1/ht.npy : (0, 4, 4)    #### HAVE G4 HITS THAT NEED TO BE REFLECTED INTO THE OPTICKS EVENT  
             ./-1/rx.npy : (54, 10, 2, 4) 
           ./-1/fdom.npy : (3, 1, 4) 
             ./-1/ox.npy : (54, 4, 4) 
             ./-1/no.npy : (94, 4, 4) 
             ./-1/gs.npy : (3, 6, 4) 
             ./-1/rs.npy : (54, 10, 1, 4) 
             ./-1/ph.npy : (54, 1, 2) 
           ./-1/idom.npy : (1, 1, 4) 
    ./-1/20180821_232057/report.txt : 35 
          ./1/report.txt : 38 
              ./1/ps.npy : (54, 1, 4) 
              ./1/ht.npy : (9, 4, 4) 
              ./1/rx.npy : (54, 10, 2, 4) 
            ./1/fdom.npy : (3, 1, 4) 
              ./1/ox.npy : (54, 4, 4) 
              ./1/no.npy : (94, 4, 4) 
              ./1/gs.npy : (3, 6, 4) 
              ./1/rs.npy : (54, 10, 1, 4) 
              ./1/ph.npy : (54, 1, 2) 
            ./1/idom.npy : (1, 1, 4) 
    ./1/20180821_232057/report.txt : 38 
    epsilon:torch blyth$ 


* notice different photon counts before and after cache 

  * TODO: check the primary transport

  



Manual ipython dumping photon seqhis labels : reveals no SD (surface detect)
--------------------------------------------------------------------------------

OK event correctly CK but no SD what happened to SensorSurfaces ?::

    epsilon:-1 blyth$ cd ../1
    epsilon:1 blyth$ ip
    args: /opt/local/bin/ipython

    In [1]: from opticks.ana.histype import HisType

    In [2]: af = HisType()

    In [3]: ph = np.load("ph.npy")

    In [4]: print "\n".join(map(lambda _:af.label(_), ph[:,0,0] ))
    CK BT BT BT MI
    CK BT BT BT MI
    CK BT MI
    CK BT BT BT MI
    CK BT MI
    CK BT MI
    CK BT MI
    CK BT MI
    CK BT BT BT MI
    CK BT MI
    ...

::

    In [14]: ph = np.load("ph.npy")

    In [16]: ph.shape
    Out[16]: (54, 1, 2)

    In [17]: from opticks.ana.histype import HisType

    In [18]: af = HisType()

    In [25]: print "\n".join(map(lambda _:af.label(_), ph[:,0,0] ))
    TO BT MI
    TO BT BT BT MI
    TO BT BT BT MI
    TO BR BR BT BR BR BR BR BR BT
    TO BR BR BR BT MI
    TO BT BT BT MI
    TO BT BT BT MI
    TO BT MI
    TO BT MI
    ...
    ## G4 event coming in as torch, when should be cerenkov 


Cumulative totals with histype.py
---------------------------------------

Can see this cumulatively with *histype.py*::

    epsilon:1 blyth$ histype.py --tag 1 --det g4live
    args: /Users/blyth/opticks/ana/histype.py --tag 1 --det g4live
    [2018-08-21 16:26:42,340] p29529 {/Users/blyth/opticks/ana/histype.py:62} INFO - loaded ph /tmp/blyth/opticks/evt/g4live/torch/1/ph.npy 20180821-1532 shape (54, 1, 2) 
    [2018-08-21 16:26:42,341] p29529 {/Users/blyth/opticks/ana/histype.py:23} INFO - test_HistoryTable
            36 CK BT MI 
            15 CK BT BT BT MI 
             2 CK BT BT BT BT MI 
             1 CK BT BT MI

::

    epsilon:1 blyth$ histype.py --tag -1 --det g4live
    args: /Users/blyth/opticks/ana/histype.py --tag -1 --det g4live
    [2018-08-21 16:26:28,087] p29522 {/Users/blyth/opticks/ana/histype.py:62} INFO - loaded ph /tmp/blyth/opticks/evt/g4live/torch/-1/ph.npy 20180821-1532 shape (54, 1, 2) 
    [2018-08-21 16:26:28,088] p29522 {/Users/blyth/opticks/ana/histype.py:23} INFO - test_HistoryTable
            28 TO BT MI 
            13 TO BT BT BT MI 
             2 TO BR BR BR BR BR BR BR BR BT 
             2 TO BR BR BR BR BR BT BT BR BR 
             1 TO BR BR BR BT BT BR BR BT BT 
             1 TO BT BT BR BR BR BR BR BT BT 
             1 TO BR BR BR BT BT BR BR BR BT 
             1 TO BR BR BT BR BR BR BR BR BT 
             1 TO BT BT BR BR BR BT BT BR BR 
             1 TO BR BR BT BT BT MI 
             1 TO BT BR BT BT MI 
             1 TO BR BR BR BT MI 
             1 TO BR BT MI 




ckm- checking the SD got converted to sensors::

    2018-08-21 16:45:41.822 FATAL [775455] [X4PhysicalVolume::convertSensors@144] [
    2018-08-21 16:45:41.822 INFO  [775455] [X4PhysicalVolume::convertSensors_r@206]  is_lvsdname 0 is_sd 1 name Det nameref Det0x110d97a80
    2018-08-21 16:45:41.822 ERROR [775455] [X4PhysicalVolume::convertSensors@149]  m_lvsdname (null) num_clv 1
    2018-08-21 16:45:41.822 INFO  [775455] [GGeoSensor::AddSensorSurfaces@54] GGeoSensor::AddSensorSurfaces i 0 sslv Det0x110d97a80 index 5
    2018-08-21 16:45:41.822 FATAL [775455] [*GGeoSensor::MakeOpticalSurface@95]  sslv Det0x110d97a80 name Det0x110d97a80SensorSurface
    2018-08-21 16:45:41.822 ERROR [775455] [GPropertyMap<float>::setStandardDomain@278]  setStandardDomain(NULL) -> default_domain  GDomain  low 60 high 820 step 20 length 39
    2018-08-21 16:45:41.822 INFO  [775455] [GGeoSensor::AddSensorSurfaces@65]  gss GSS:: GPropertyMap<T>::  5    skinsurface s: GOpticalSurface  type 0 model 1 finish 3 value     1   Det0x110d97a80SensorSurface k:EFFICIENCY GROUPVEL RINDEX
    2018-08-21 16:45:41.822 FATAL [775455] [*GGeoSensor::MakeOpticalSurface@95]  sslv Det0x110d97a80 name Det0x110d97a80SensorSurface
    2018-08-21 16:45:41.822 ERROR [775455] [X4PhysicalVolume::convertSensors@162]  num_bds 0 num_sks0 0 num_sks1 1
    2018-08-21 16:45:41.822 FATAL [775455] [X4PhysicalVolume::convertSensors@168] ]


Hmm booting from a prior runs primaries::

    2018-08-21 16:45:41.858 INFO  [775455] [OpticksHub::configureGeometryTri@558] OpticksHub::configureGeometryTri restrict_mesh -1 nmm 1
    2018-08-21 16:45:41.858 FATAL [775455] [OpticksGen::initFromPrimaries@101] booting from input_primaries /usr/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/c250d41454fba7cb19f3b83815b132c2/1/primaries.npy
    2018-08-21 16:45:41.858 FATAL [775455] [OpticksHub::init@189] ]
    2018-08-21 16:45:41.858 INFO  [775455] [SLog::operator@21] OpticksHub::OpticksHub  DONE



For Opticks to get SURFACE_DETECT the surface EFFICIENCY property must be which feeds into the detect 

::

    605 __device__ int
    606 propagate_at_surface(Photon &p, State &s, curandState &rng)
    607 {
    608     float u_surface = curand_uniform(&rng);
    609 #ifdef WITH_ALIGN_DEV
    610     float u_surface_burn = curand_uniform(&rng);
    611 #endif
    612 #ifdef WITH_ALIGN_DEV_DEBUG
    613     rtPrintf("propagate_at_surface   u_OpBoundary_DiDiReflectOrTransmit:        %.9g \n", u_surface);
    614     rtPrintf("propagate_at_surface   u_OpBoundary_DoAbsorption:   %.9g \n", u_surface_burn);
    615 #endif
    616 
    617     if( u_surface < s.surface.y )   // absorb   
    618     {
    619         s.flag = SURFACE_ABSORB ;
    620         s.index.x = s.index.y ;   // kludge to get m2 into seqmat for BREAKERs
    621         return BREAK ;
    622     }
    623     else if ( u_surface < s.surface.y + s.surface.x )  // absorb + detect
    624     {
    625         s.flag = SURFACE_DETECT ;
    626         s.index.x = s.index.y ;   // kludge to get m2 into seqmat for BREAKERs
    627         return BREAK ;
    628     }
    629     else if (u_surface  < s.surface.y + s.surface.x + s.surface.w )  // absorb + detect + reflect_diffuse 
    630     {
    631         s.flag = SURFACE_DREFLECT ;
    632         propagate_at_diffuse_reflector_geant4_style(p, s, rng);
    633         return CONTINUE;
    634     }
    635     else
    636     {


::

     04 struct State
      5 {
      6    unsigned int flag ;
      7    float4 material1 ;    // refractive_index/absorption_length/scattering_length/reemission_prob
      8    float4 m1group2  ;    // group_velocity/spare1/spare2/spare3
      9    float4 material2 ;  
     10    float4 surface    ;   //  detect/absorb/reflect_specular/reflect_diffuse
     11    float3 surface_normal ; 
     12    float cos_theta ;
     13    float distance_to_boundary ;
     14    uint4 optical ;   // x/y/z/w index/type/finish/value  
     15    uint4 index ;     // indices of m1/m2/surf/sensor
     16    uint4 identity ;  //  node/mesh/boundary/sensor indices of last intersection
     17    float ureflectcheat ;
     18 };
     19 


Surface float4 comes from boundary lookup into texture::

     29 __device__ void fill_state( State& s, int boundary, uint4 identity, float wavelength )
     30 {   
     31     // boundary : 1 based code, signed by cos_theta of photon direction to outward geometric normal
     32     // >0 outward going photon
     33     // <0 inward going photon
     34     //
     35     // NB the line is above the details of the payload (ie how many float4 per matsur) 
     36     //    it is just 
     37     //                boundaryIndex*4  + 0/1/2/3     for OMAT/OSUR/ISUR/IMAT 
     38     //
     39     
     40     int line = boundary > 0 ? (boundary - 1)*BOUNDARY_NUM_MATSUR : (-boundary - 1)*BOUNDARY_NUM_MATSUR  ;
     41     
     42     // pick relevant lines depening on boundary sign, ie photon direction relative to normal
     43     //  
     44     int m1_line = boundary > 0 ? line + IMAT : line + OMAT ;
     45     int m2_line = boundary > 0 ? line + OMAT : line + IMAT ;
     46     int su_line = boundary > 0 ? line + ISUR : line + OSUR ;
     47     
     48     //  consider photons arriving at PMT cathode surface
     49     //  geometry normals are expected to be out of the PMT 
     50     //
     51     //  boundary sign will be -ve : so line+3 outer-surface is the relevant one
     52     
     53     s.material1 = boundary_lookup( wavelength, m1_line, 0);
     54     s.m1group2  = boundary_lookup( wavelength, m1_line, 1);
     55     
     56     s.material2 = boundary_lookup( wavelength, m2_line, 0);
     57     s.surface   = boundary_lookup( wavelength, su_line, 0);
     58     
     59     s.optical = optical_buffer[su_line] ;   // index/type/finish/value
     60     
     61     s.index.x = optical_buffer[m1_line].x ; // m1 index
     62     s.index.y = optical_buffer[m2_line].x ; // m2 index 
     63     s.index.z = optical_buffer[su_line].x ; // su index
     64     s.index.w = identity.w   ;
     65     
     66     s.identity = identity ;
     67 




Added --dbgtex to dump OBndLib::convert dumping to "$OPTICKS_KEYDIR/dbgtex/buf.npy"  and "obuf.npy"

::


    In [19]: t.shape
    Out[19]: (3, 4, 2, 39, 4)


    In [19]: t.shape             ## 3 boundaries
    Out[19]: (3, 4, 2, 39, 4)

    In [20]: t[0].shape          ## 4 qty : omat/osur/isur/imat
    Out[20]: (4, 2, 39, 4)

    In [21]: t[0,1].shape        ## pick osur 
    Out[21]: (2, 39, 4)

    In [22]: t[0,1,0].shape      ## 1st group of osur 
    Out[22]: (39, 4)

    In [23]: t[0,1,0]            ## all boundaries look the same, unset 
    Out[23]: 
    array([[-1., -1., -1., -1.],
           [-1., -1., -1., -1.],
           ...
           [-1., -1., -1., -1.],
           [-1., -1., -1., -1.]], dtype=float32)


Actually can see this from top level dump, all osur/isur surfaces are unset::

    In [28]: t[1]
    Out[28]: 
    array([[[[      1.    , 1000000.    , 1000000.    ,       0.    ],
             [      1.    , 1000000.    , 1000000.    ,       0.    ],
             [      1.    , 1000000.    , 1000000.    ,       0.    ],
             ...,
             [      1.    , 1000000.    , 1000000.    ,       0.    ],
             [      1.    , 1000000.    , 1000000.    ,       0.    ],
             [      1.    , 1000000.    , 1000000.    ,       0.    ]],

            [[    299.7924,       0.    ,       0.    ,       0.    ],
             [    299.7924,       0.    ,       0.    ,       0.    ],
             [    299.7924,       0.    ,       0.    ,       0.    ],
             ...,
             [    299.7924,       0.    ,       0.    ,       0.    ],
             [    299.7924,       0.    ,       0.    ,       0.    ],
             [    299.7924,       0.    ,       0.    ,       0.    ]]],


           [[[     -1.    ,      -1.    ,      -1.    ,      -1.    ],
             [     -1.    ,      -1.    ,      -1.    ,      -1.    ],
             [     -1.    ,      -1.    ,      -1.    ,      -1.    ],
             ...,
             [     -1.    ,      -1.    ,      -1.    ,      -1.    ],
             [     -1.    ,      -1.    ,      -1.    ,      -1.    ],
             [     -1.    ,      -1.    ,      -1.    ,      -1.    ]],

            [[     -1.    ,      -1.    ,      -1.    ,      -1.    ],
             [     -1.    ,      -1.    ,      -1.    ,      -1.    ],
             [     -1.    ,      -1.    ,      -1.    ,      -1.    ],
             ...,
             [     -1.    ,      -1.    ,      -1.    ,      -1.    ],
             [     -1.    ,      -1.    ,      -1.    ,      -1.    ],
             [     -1.    ,      -1.    ,      -1.    ,      -1.    ]]],


           [[[     -1.    ,      -1.    ,      -1.    ,      -1.    ],
             [     -1.    ,      -1.    ,      -1.    ,      -1.    ],
             [     -1.    ,      -1.    ,      -1.    ,      -1.    ],
             ...,
             [     -1.    ,      -1.    ,      -1.    ,      -1.    ],
             [     -1.    ,      -1.    ,      -1.    ,      -1.    ],
             [     -1.    ,      -1.    ,      -1.    ,      -1.    ]],

            [[     -1.    ,      -1.    ,      -1.    ,      -1.    ],
             [     -1.    ,      -1.    ,      -1.    ,      -1.    ],
             [     -1.    ,      -1.    ,      -1.    ,      -1.    ],
             ...,
             [     -1.    ,      -1.    ,      -1.    ,      -1.    ],
             [     -1.    ,      -1.    ,      -1.    ,      -1.    ],
             [     -1.    ,      -1.    ,      -1.    ,      -1.    ]]],


           [[[      1.3608, 1000000.    , 1000000.    ,       0.    ],
             [      1.3608, 1000000.    , 1000000.    ,       0.    ],
             [      1.3608, 1000000.    , 1000000.    ,       0.    ],
             ...,
             [      1.3435, 1000000.    , 1000000.    ,       0.    ],
             [      1.3435, 1000000.    , 1000000.    ,       0.    ],
             [      1.3435, 1000000.    , 1000000.    ,       0.    ]],

            [[    220.306 ,       0.    ,       0.    ,       0.    ],
             [    220.306 ,       0.    ,       0.    ,       0.    ],
             [    220.306 ,       0.    ,       0.    ,       0.    ],
             ...,
             [    223.1429,       0.    ,       0.    ,       0.    ],
             [    223.1429,       0.    ,       0.    ,       0.    ],
             [    223.1429,       0.    ,       0.    ,       0.    ]]]], dtype=float32)



dbgtex.py optical buffer for 3 boundaries with just omat/imat set::

    Out[1]: 
    array([[[1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0]],

           [[1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [2, 0, 0, 0]],

           [[2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [3, 0, 0, 0]]], dtype=uint32)

Added bnd.txt to the dbgtex dir, makes it very clear that are missing the SensorSurfaces::

    epsilon:~ blyth$ cd /usr/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/c250d41454fba7cb19f3b83815b132c2/1/dbgtex
    epsilon:dbgtex blyth$ l
    total 48
    -rw-r--r--  1 blyth  staff     36 Aug 21 19:25 bnd.txt
    -rw-r--r--  1 blyth  staff    272 Aug 21 19:25 obuf.npy
    -rw-r--r--  1 blyth  staff  15072 Aug 21 19:25 buf.npy

    epsilon:dbgtex blyth$ cat bnd.txt 
    Air///Air
    Air///Water
    Water///Glass



Review what CerenkovMinimal.cc does, aims to be a bog standard Geant4 example, with minimal addition of Opticks::


     01 #include "OPTICKS_LOG.hh"
      2 #include "G4.hh"
      3 
      4 int main(int argc, char** argv)
      5 {
      6     OPTICKS_LOG(argc, argv);
      7     G4 g(1) ;
      8     return 0 ;
      9 }
     10 

     18 G4::G4(int nev)
     19     :
     20     ctx(new Ctx),
     21     rm(new G4RunManager),
     22     sdn("SD0"),
     23     sd(new SensitiveDetector(sdn)),
     24     dc(new DetectorConstruction(sdn)),
     25     pl(new PhysicsList<L4Cerenkov>()),
     26     ga(NULL),
     27     ra(NULL),
     28     ea(NULL),
     29     ta(NULL),
     30     sa(NULL)
     31 {
     32     rm->SetUserInitialization(dc);
     33     rm->SetUserInitialization(pl);
     34 
     35     ga = new PrimaryGeneratorAction(ctx);
     36     ra = new RunAction(ctx) ;
     37     ea = new EventAction(ctx) ;
     38     ta = new TrackingAction(ctx) ;
     39     sa = new SteppingAction(ctx) ;
     40 
     41     rm->SetUserAction(ga);
     42     rm->SetUserAction(ra);
     43     rm->SetUserAction(ea);
     44     rm->SetUserAction(ta);
     45     rm->SetUserAction(sa);
     46 
     47     rm->Initialize();
     48 
     49     beamOn(nev);
     50 }
     

BeginOfRunAction hand over the world::

     18 void RunAction::BeginOfRunAction(const G4Run*)
     19 {
     20     LOG(info) << "." ;
     21 #ifdef WITH_OPTICKS
     22     G4VPhysicalVolume* world = G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking()->GetWorldVolume() ;
     23     assert( world ) ;
     24     G4Opticks::GetOpticks()->setGeometry(world);
     25 #endif
     26 }


Critical aspect::

    114 GGeo* G4Opticks::translateGeometry( const G4VPhysicalVolume* top )
    115 {
    116     const char* keyspec = X4PhysicalVolume::Key(top) ;
    117     BOpticksKey::SetKey(keyspec);
    118     LOG(error) << " SetKey " << keyspec  ;
    119     
    120     Opticks* ok = new Opticks(0,0, fEmbeddedCommandLine);  // Opticks instanciation must be after BOpticksKey::SetKey
    121     
    122     const char* gdmlpath = ok->getGDMLPath();   // inside geocache, not SrcGDMLPath from opticksdata
    123     CGDML::Export( gdmlpath, top ); 
    124     
    125     GGeo* gg = new GGeo(ok) ;
    126     X4PhysicalVolume xtop(gg, top) ;   // <-- populates gg 
    127     gg->postDirectTranslation(); 
    128     
    129     int root = 0 ;
    130     const char* gltfpath = ok->getGLTFPath();   // inside geocache
    131     GGeoGLTF::Save(gg, gltfpath, root );
    132     
    133     return gg ;
    134 }   


::

    120 void X4PhysicalVolume::init()
    121 {
    122     LOG(info) << "query : " << m_query->desc() ;
    123 
    124     convertMaterials();
    125     convertSurfaces();
    126     convertSensors();  // before closeSurfaces as may add some SensorSurfaces
    127     closeSurfaces();
    128     convertSolids();
    129     convertStructure();
    130     convertCheck();
    131 }



::

    142 void X4PhysicalVolume::convertSensors()
    143 {
    144     LOG(fatal) << "[" ;
    145 
    146     convertSensors_r(m_top, 0);
    147 
    148     unsigned num_clv = m_ggeo->getNumCathodeLV();
    149     LOG(error)
    150          << " m_lvsdname " << m_lvsdname
    151          << " num_clv " << num_clv
    152          ;
    153 
    154     unsigned num_bds = m_ggeo->getNumBorderSurfaces() ;
    155     unsigned num_sks0 = m_ggeo->getNumSkinSurfaces() ;
    156 
    157     GGeoSensor::AddSensorSurfaces(m_ggeo) ;
    158 
    159     unsigned num_sks1 = m_ggeo->getNumSkinSurfaces() ;
    160     assert( num_bds == m_ggeo->getNumBorderSurfaces()  );
    161 
    162     LOG(error)
    163          << " num_bds " << num_bds
    164          << " num_sks0 " << num_sks0
    165          << " num_sks1 " << num_sks1
    166          ;
    167 
    168     LOG(fatal) << "]" ;
    169 }

::

     36 void GGeoSensor::AddSensorSurfaces( GGeo* gg )
     37 {
     38     GMaterial* cathode_props = gg->getCathode() ;
     39     if(!cathode_props)
     40     {
     41         LOG(fatal) << " require a cathode material to AddSensorSurfaces " ;
     42         return ;
     43     }
     44 
     45     unsigned nclv = gg->getNumCathodeLV();
     46 
     47     for(unsigned i=0 ; i < nclv ; i++)
     48     {
     49         const char* sslv = gg->getCathodeLV(i);
     50         unsigned index = gg->getNumMaterials() + gg->getNumSkinSurfaces() + gg->getNumBorderSurfaces() ;
     51         // standard materials/surfaces use the originating aiMaterial index, 
     52         // extend that for fake SensorSurface by toting up all 
     53 
     54         LOG(info) << "GGeoSensor::AddSensorSurfaces"
     55                   << " i " << i
     56                   << " sslv " << sslv
     57                   << " index " << index
     58                   ;
     59 
     60         GSkinSurface* gss = MakeSensorSurface(sslv, index);
     61         gss->setStandardDomain();  // default domain 
     62         gss->setSensor();
     63         gss->add(cathode_props);
     64 
     65         LOG(info) << " gss " << gss->description();
     66 
     67         gg->add(gss);
     68 
     69         {
     70             // not setting sensor or domain : only the standardized need those
     71             GSkinSurface* gss_raw = MakeSensorSurface(sslv, index);
     72             gss_raw->add(cathode_props);
     73             gg->addRaw(gss_raw);
     74         }
     75     }
     76 }


::

    2018-08-21 19:45:23.926 ERROR [31834] [X4PhysicalVolume::convertSensors@149]  m_lvsdname (null) num_clv 1
    2018-08-21 19:45:23.926 INFO  [31834] [GGeoSensor::AddSensorSurfaces@54] GGeoSensor::AddSensorSurfaces i 0 sslv Det0x110d97c30 index 5
    2018-08-21 19:45:23.926 FATAL [31834] [*GGeoSensor::MakeOpticalSurface@95]  sslv Det0x110d97c30 name Det0x110d97c30SensorSurface

    ## this name looks problematic : Det0x110d97c30SensorSurface



Surprised regards the ordering here::

    epsilon:GItemList blyth$ cat GSurfaceLib.txt
    perfectDetectSurface
    perfectAbsorbSurface
    perfectSpecularSurface
    perfectDiffuseSurface
    Det0x110d97c30SensorSurface

Giving 1-based index of 5 to the SensorSurface::

    GGeoSensor::AddSensorSurfaces i 0 sslv Det0x110d97c30 index 5



::

    191 void X4PhysicalVolume::convertSensors_r(const G4VPhysicalVolume* const pv, int depth)
    192 {
    193     const G4LogicalVolume* const lv = pv->GetLogicalVolume();
    194     const char* lvname = lv->GetName().c_str();
    195     G4VSensitiveDetector* sd = lv->GetSensitiveDetector() ;
    196 
    197     bool is_lvsdname = m_lvsdname && BStr::Contains(lvname, m_lvsdname, ',' ) ;
    198     bool is_sd = sd != NULL ;
    199 
    200     const std::string sdn = sd ? sd->GetName() : "SD?" ;   // perhaps GetFullPathName() 
    201 
    202     if( is_lvsdname || is_sd )
    203     {
    204         std::string name = BFile::Name(lvname);
    205         std::string nameref = SGDML::GenerateName( name.c_str() , lv , true );
    206         LOG(info)
    207             << " is_lvsdname " << is_lvsdname
    208             << " is_sd " << is_sd
    209             << " name " << name
    210             << " nameref " << nameref
    211             ;
    212 
    213         m_ggeo->addLVSD(nameref.c_str(), sdn.c_str()) ;
    214     } 
    215 
    216     for (int i=0 ; i < lv->GetNoDaughters() ;i++ )
    217     {
    218         const G4VPhysicalVolume* const child_pv = lv->GetDaughter(i);
    219         convertSensors_r(child_pv, depth+1 );
    220     }
    221 }


The LVSD goes into cachemeta::

    epsilon:1 blyth$ cat cachemeta.json 
    {"answer":42,"argline":" /usr/local/opticks/lib/CerenkovMinimal","lv2sd":{"Det0x110d9a820":"SD0"},"question":"huh?"}epsilon:1 blyth$ 


Probable source of bug is LV name resolution fail in X4PhysicalVolume::addBoundary::

    604 /**
    605 X4PhysicalVolume::addBoundary
    606 ------------------------------
    607 
    608 See notes/issues/ab-blib.rst
    609 
    610 **/
    611 
    612 unsigned X4PhysicalVolume::addBoundary(const G4VPhysicalVolume* const pv, const G4VPhysicalVolume* const pv_p )
    613 {
    614     const G4LogicalVolume* const lv   = pv->GetLogicalVolume() ;
    615     const G4LogicalVolume* const lv_p = pv_p ? pv_p->GetLogicalVolume() : NULL ;
    616 
    617     const G4Material* const imat_ = lv->GetMaterial() ;
    618     const G4Material* const omat_ = lv_p ? lv_p->GetMaterial() : imat_ ;  // top omat -> imat 
    619 
    620     const char* omat = X4::BaseName(omat_) ;
    621     const char* imat = X4::BaseName(imat_) ;
    622 
    623     // Why do boundaries with this material pair have surface finding problem for the old route ?
    624     bool problem_pair  = strcmp(omat, "UnstStainlessSteel") == 0 && strcmp(imat, "BPE") == 0 ;
    625 
    626     bool first_priority = true ;
    627     const G4LogicalSurface* const isur_ = findSurface( pv  , pv_p , first_priority );
    628     const G4LogicalSurface* const osur_ = findSurface( pv_p, pv   , first_priority );
    629 
    630     // doubtful of findSurface priority with double skin surfaces, see g4op-
    631 
    632 
    633     // the above will not find Opticks SensorSurfaces ... so look for those with GGeo
    634 
    635     const char* _lv = X4::BaseNameAsis(lv) ;
    636     const char* _lv_p = X4::BaseNameAsis(lv_p) ;   // NULL when no lv_p   
    637 
    638     const GSkinSurface* g_sslv = m_ggeo->findSkinSurface(_lv) ;
    639     const GSkinSurface* g_sslv_p = _lv_p ? m_ggeo->findSkinSurface(_lv_p) : NULL ;
    640 
    641     if( g_sslv_p )
    642         LOG(debug) << " node_count " << m_node_count
    643                    << " _lv_p   " << _lv_p
    644                    << " g_sslv_p " << g_sslv_p->getName()
    645                    ;
    ...
    675     unsigned boundary = 0 ;
    676     if( g_sslv == NULL && g_sslv_p == NULL  )
    677     {
    678         const char* osur = X4::BaseName( osur_ );
    679         const char* isur = X4::BaseName( isur_ );
    680         boundary = m_blib->addBoundary( omat, osur, isur, imat );
    681     }
    682     else if( g_sslv && !g_sslv_p )
    683     {
    684         const char* osur = g_sslv->getName();
    685         const char* isur = osur ;
    686         boundary = m_blib->addBoundary( omat, osur, isur, imat );
    687     }
    688     else if( g_sslv_p && !g_sslv )
    689     {
    690         const char* osur = g_sslv_p->getName();
    691         const char* isur = osur ;
    692         boundary = m_blib->addBoundary( omat, osur, isur, imat );
    693     }
    694     else if( g_sslv_p && g_sslv )
    695     {
    696         assert( 0 && "fabled double skin found : see notes/issues/ab-blib.rst  " );
    697     }
    698 
    699     return boundary ;
    700 }



