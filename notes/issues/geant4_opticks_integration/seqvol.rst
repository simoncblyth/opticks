seqvol : volume sequence indexing
===================================

Problem with volume sequencing is the large number of volumes and expensive storage of int32*10 sequence
but actually the number relevant to critical optical path is not so big, 
so judicious favoriting of 15 volumes 0x0->0xE specific to targetted AD and overflow 0xF for all others
may be sufficient.

tlaser node index dumping
---------------------------

::

     tlaser- ; tlaser-- --compute --pindex 0
     ...
     photon_id 0 slot 0 s.identity.x 3159 
     photon_id 0 slot 1 s.identity.x 3158 
     photon_id 0 slot 2 s.identity.x 3157 
     photon_id 0 slot 3 s.identity.x 3156 
     photon_id 0 slot 4 s.identity.x 4413 
     2016-10-02 13:50:36.831 INFO  [1363053] [OPropagator::launch@218] 1 : (0;10000,1) launch_times vali,comp,prel,lnch  0.0000 0.0000 0.0000 0.0335


     tlaser- ; tlaser-- --compute --pindex 1

     photon_id 1 slot 0 s.identity.x 3159 
     photon_id 1 slot 1 s.identity.x 3158 
     photon_id 1 slot 2 s.identity.x 3157 
     photon_id 1 slot 3 s.identity.x 3156 
     photon_id 1 slot 4 s.identity.x 4413 
     2016-10-02 13:51:56.063 INFO  [1363607] [OPropagator::launch@218] 1 : (0;10000,1) launch_times vali,comp,prel,lnch  0.0000 0.0000 0.0000 0.0334

     tlaser- ; tlaser-- --compute --pindex 2

     photon_id 2 slot 0 s.identity.x 3159 
     photon_id 2 slot 1 s.identity.x 3159 
     photon_id 2 slot 2 s.identity.x 3158 
     photon_id 2 slot 3 s.identity.x 3157 
     photon_id 2 slot 4 s.identity.x 3156 
     photon_id 2 slot 5 s.identity.x 4412 
     2016-10-02 13:52:42.050 INFO  [1364078] [OPropagator::launch@218] 1 : (0;10000,1) launch_times vali,comp,prel,lnch  0.0000 0.0000 0.0000 0.0235

     tlaser- ; tlaser-- --compute --pindex 3

     photon_id 3 slot 0 s.identity.x 3159 
     photon_id 3 slot 1 s.identity.x 3158 
     photon_id 3 slot 2 s.identity.x 3157 
     photon_id 3 slot 3 s.identity.x 3156 
     photon_id 3 slot 4 s.identity.x 4413 
     2016-10-02 13:54:25.898 INFO  [1364844] [OPropagator::launch@218] 1 : (0;10000,1) launch_times vali,comp,prel,lnch  0.0000 0.0000 0.0000 0.0544



Above identity.x is probably a zero based index, but below list is 1-based::

    delta:GItemList blyth$ idp
    delta:g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae blyth$ vi GItemList/PVNames.txt 

    00001 top
     0002 __dd__Structure__Sites__db-rock0xc15d358
     0003 __dd__Geometry__Sites__lvNearSiteRock--pvNearHallTop0xbf89820
     0004 __dd__Geometry__Sites__lvNearHallTop--pvNearTopCover0xc23f9b8
     ....
     3147 __dd__Geometry__RPCSupport__lvNearHbeamBigUnit--pvNearRightDiagSILeftY40xbf89630
     3148 __dd__Geometry__Sites__lvNearSiteRock--pvNearHallBot0xcd2fa58
     3149 __dd__Geometry__Sites__lvNearHallBot--pvNearPoolDead0xc13c018
     3150 __dd__Geometry__Pool__lvNearPoolDead--pvNearPoolLiner0xbf4b270
     3151 __dd__Geometry__Pool__lvNearPoolLiner--pvNearPoolOWS0xbf55b10
     3152 __dd__Geometry__Pool__lvNearPoolOWS--pvNearPoolCurtain0xc5c5f20
     3153 __dd__Geometry__Pool__lvNearPoolCurtain--pvNearPoolIWS0xc15a498
     3154 __dd__Geometry__Pool__lvNearPoolIWS--pvNearADE10xc2cf528
     3155 __dd__Geometry__AD__lvADE--pvSST0xc128d90
     3156 __dd__Geometry__AD__lvSST--pvOIL0xc241510
     3157 __dd__Geometry__AD__lvOIL--pvOAV0xbf8f638
     3158 __dd__Geometry__AD__lvOAV--pvLSO0xbf8e120
     3159 __dd__Geometry__AD__lvLSO--pvIAV0xc2d0348
     3160 __dd__Geometry__AD__lvIAV--pvGDS0xbf6ab00
     3161 __dd__Geometry__AD__lvIAV--pvOcrGdsInIAV0xbf6b0e0
     ....
     4409 __dd__Geometry__AD__lvOIL--pvRadialShield..150xc113258
     4410 __dd__Geometry__AD__lvOIL--pvRadialShield..160xc3ccdb8
     4411 __dd__Geometry__AD__lvOIL--pvRadialShield..170xc3cce80
     4412 __dd__Geometry__AD__lvOIL--pvRadialShield..180xc3d6b88
     4413 __dd__Geometry__AD__lvOIL--pvRadialShield..190xc3d6c50
     4414 __dd__Geometry__AD__lvOIL--pvRadialShield..200xc3d6d18
     4415 __dd__Geometry__AD__lvOIL--pvRadialShield..210xc3d6de0
    12228 __dd__Geometry__Sites__lvNearHallBot--pvNearHallRadSlabs--pvNearHallRadSlab70xc15ccb0
    12229 __dd__Geometry__Sites__lvNearHallBot--pvNearHallRadSlabs--pvNearHallRadSlab80xc15cdb8
    12230 __dd__Geometry__Sites__lvNearHallBot--pvNearHallRadSlabs--pvNearHallRadSlab90xc15cf08


    delta:g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae blyth$ vi GItemList/LVNames.txt 
    00001 World0xc15cfc0
        2 __dd__Geometry__Sites__lvNearSiteRock0xc030350
        3 __dd__Geometry__Sites__lvNearHallTop0xc136890
        4 __dd__Geometry__PoolDetails__lvNearTopCover0xc137060
        5 __dd__Geometry__RPC__lvRPCMod0xbf54e60
    .....
    03148 __dd__Geometry__Sites__lvNearHallBot0xbf89c60
     3149 __dd__Geometry__Pool__lvNearPoolDead0xc2dc490
     3150 __dd__Geometry__Pool__lvNearPoolLiner0xc21e9d0
     3151 __dd__Geometry__Pool__lvNearPoolOWS0xbf93840
     3152 __dd__Geometry__Pool__lvNearPoolCurtain0xc2ceef0
     3153 __dd__Geometry__Pool__lvNearPoolIWS0xc28bc60
     3154 __dd__Geometry__AD__lvADE0xc2a78c0
     3155 __dd__Geometry__AD__lvSST0xc234cd0
     3156 __dd__Geometry__AD__lvOIL0xbf5e0b8
     3157 __dd__Geometry__AD__lvOAV0xbf1c760
     3158 __dd__Geometry__AD__lvLSO0xc403e40
     3159 __dd__Geometry__AD__lvIAV0xc404ee8
     3160 __dd__Geometry__AD__lvGDS0xbf6cbb8
     3161 __dd__Geometry__AdDetails__lvOcrGdsInIav0xbf6dd58
     3162 __dd__Geometry__AdDetails__lvIavTopHub0xc129d88
     3163 __dd__Geometry__AdDetails__lvCtrGdsOflBotClp0xc407eb0



dbgseqhis
------------

Hmm looks like in Opticks gets SURFACE_ABSORB on radial shield, but with 
CG4 that happens on SST ?



::

   tlaser- ; tlaser-- --okg4 --compute --dbgseqhis 8ccccccd



    ----CSteppingAction----
    2016-10-02 14:22:14.619 INFO  [1371115] [CRecorder::Dump@670] CSteppingAction::UserSteppingAction DONE record_id    6717
    2016-10-02 14:22:14.619 INFO  [1371115] [CRecorder::Dump@673]  seqhis 8ccccccd TORCH BOUNDARY_TRANSMIT BOUNDARY_TRANSMIT BOUNDARY_TRANSMIT BOUNDARY_TRANSMIT BOUNDARY_TRANSMIT BOUNDARY_TRANSMIT SURFACE_ABSORB . . . . . . . . 
    2016-10-02 14:22:14.619 INFO  [1371115] [CRecorder::Dump@677]  seqmat 44343231 GdDopedLS Acrylic LiquidScintillator Acrylic MineralOil Acrylic MineralOil MineralOil - - - - - - - - 
    2016-10-02 14:22:14.619 INFO  [1371115] [Rec::Dump@226] CSteppingAction::UserSteppingAction (Rec)DONE nstates 7
    [  0/  7]
               stepStatus [           Undefined/        GeomBoundary]
                     flag [               TORCH/   BOUNDARY_TRANSMIT]
               bs pri/cur [                 Und/                 FrT]
                 material [           GdDopedLS/             Acrylic]
    (rec state ;opticalphoton stepNum    7(tk ;opticalphoton tid 6718 pid 0 nm    430 mm  ori[ -1.81e+04  -8e+05-6.60e+03]  pos[ 1.38e+03-2.07e+03       0]  )
      pre d/Geometry/AD/lvIAV#pvGDS rials/GdDopedLS          noProc           Undefined pos[        0       0       0]  dir[    0.556  -0.831       0]  pol[       -1  0.0226       0]  ns  0.100 nm 430.000
     post d/Geometry/AD/lvLSO#pvIAV terials/Acrylic  Transportation        GeomBoundary pos[      861-1.29e+03       0]  dir[    0.556  -0.831       0]  pol[       -1  0.0226       0]  ns  8.059 nm 430.000
     )
    [  1/  7]
               stepStatus [        GeomBoundary/        GeomBoundary]
                     flag [   BOUNDARY_TRANSMIT/   BOUNDARY_TRANSMIT]
               bs pri/cur [                 FrT/                 FrT]
                 material [             Acrylic/  LiquidScintillator]
    (rec state ;opticalphoton stepNum    7(tk ;opticalphoton tid 6718 pid 0 nm    430 mm  ori[ -1.81e+04  -8e+05-6.60e+03]  pos[ 1.38e+03-2.07e+03       0]  )
      pre d/Geometry/AD/lvLSO#pvIAV terials/Acrylic  Transportation        GeomBoundary pos[      861-1.29e+03       0]  dir[    0.556  -0.831       0]  pol[       -1  0.0226       0]  ns  8.059 nm 430.000
     post d/Geometry/AD/lvOAV#pvLSO uidScintillator  Transportation        GeomBoundary pos[      867-1.3e+03       0]  dir[    0.556  -0.831       0]  pol[       -1  0.0226       0]  ns  8.110 nm 430.000
     )
    [  2/  7]
               stepStatus [        GeomBoundary/        GeomBoundary]
                     flag [   BOUNDARY_TRANSMIT/   BOUNDARY_TRANSMIT]
               bs pri/cur [                 FrT/                 FrT]
                 material [  LiquidScintillator/             Acrylic]
    (rec state ;opticalphoton stepNum    7(tk ;opticalphoton tid 6718 pid 0 nm    430 mm  ori[ -1.81e+04  -8e+05-6.60e+03]  pos[ 1.38e+03-2.07e+03       0]  )
      pre d/Geometry/AD/lvOAV#pvLSO uidScintillator  Transportation        GeomBoundary pos[      867-1.3e+03       0]  dir[    0.556  -0.831       0]  pol[       -1  0.0226       0]  ns  8.110 nm 430.000
     post d/Geometry/AD/lvOIL#pvOAV terials/Acrylic  Transportation        GeomBoundary pos[  1.1e+03-1.65e+03       0]  dir[    0.556  -0.831       0]  pol[       -1  0.0226       0]  ns 10.277 nm 430.000
     )
    [  3/  7]
               stepStatus [        GeomBoundary/        GeomBoundary]
                     flag [   BOUNDARY_TRANSMIT/   BOUNDARY_TRANSMIT]
               bs pri/cur [                 FrT/                 FrT]
                 material [             Acrylic/          MineralOil]
    (rec state ;opticalphoton stepNum    7(tk ;opticalphoton tid 6718 pid 0 nm    430 mm  ori[ -1.81e+04  -8e+05-6.60e+03]  pos[ 1.38e+03-2.07e+03       0]  )
      pre d/Geometry/AD/lvOIL#pvOAV terials/Acrylic  Transportation        GeomBoundary pos[  1.1e+03-1.65e+03       0]  dir[    0.556  -0.831       0]  pol[       -1  0.0226       0]  ns 10.277 nm 430.000
     post d/Geometry/AD/lvSST#pvOIL ials/MineralOil  Transportation        GeomBoundary pos[ 1.11e+03-1.66e+03       0]  dir[    0.556  -0.831       0]  pol[       -1  0.0226       0]  ns 10.371 nm 430.000
     )
    [  4/  7]
               stepStatus [        GeomBoundary/        GeomBoundary]
                     flag [   BOUNDARY_TRANSMIT/   BOUNDARY_TRANSMIT]
               bs pri/cur [                 FrT/                 FrT]
                 material [          MineralOil/             Acrylic]
    (rec state ;opticalphoton stepNum    7(tk ;opticalphoton tid 6718 pid 0 nm    430 mm  ori[ -1.81e+04  -8e+05-6.60e+03]  pos[ 1.38e+03-2.07e+03       0]  )
      pre d/Geometry/AD/lvSST#pvOIL ials/MineralOil  Transportation        GeomBoundary pos[ 1.11e+03-1.66e+03       0]  dir[    0.556  -0.831       0]  pol[       -1  0.0226       0]  ns 10.371 nm 430.000
     post D/lvOIL#pvRadialShield:20 terials/Acrylic  Transportation        GeomBoundary pos[ 1.26e+03-1.88e+03       0]  dir[    0.556  -0.831       0]  pol[       -1  0.0226       0]  ns 11.683 nm 430.000
     )
    [  5/  7]
               stepStatus [        GeomBoundary/        GeomBoundary]
                     flag [   BOUNDARY_TRANSMIT/   BOUNDARY_TRANSMIT]
               bs pri/cur [                 FrT/                 FrT]
                 material [             Acrylic/          MineralOil]
    (rec state ;opticalphoton stepNum    7(tk ;opticalphoton tid 6718 pid 0 nm    430 mm  ori[ -1.81e+04  -8e+05-6.60e+03]  pos[ 1.38e+03-2.07e+03       0]  )
      pre D/lvOIL#pvRadialShield:20 terials/Acrylic  Transportation        GeomBoundary pos[ 1.26e+03-1.88e+03       0]  dir[    0.556  -0.831       0]  pol[       -1  0.0226       0]  ns 11.683 nm 430.000
     post d/Geometry/AD/lvSST#pvOIL ials/MineralOil  Transportation        GeomBoundary pos[ 1.26e+03-1.88e+03       0]  dir[    0.556  -0.831       0]  pol[       -1  0.0226       0]  ns 11.698 nm 430.000
     )
    [  6/  7]
               stepStatus [        GeomBoundary/        GeomBoundary]
                     flag [   BOUNDARY_TRANSMIT/      SURFACE_ABSORB]
               bs pri/cur [                 FrT/                 Abs]
                 material [          MineralOil/      StainlessSteel]




Where did the surface handling come from in Opticks...
----------------------------------------------------------


cu/generate.cu::

    418         if(s.optical.x > 0 )       // x/y/z/w:index/type/finish/value
    419         {
    420             command = propagate_at_surface(p, s, rng);
    421             if(command == BREAK)    break ;       // SURFACE_DETECT/SURFACE_ABSORB
    422             if(command == CONTINUE) continue ;    // SURFACE_DREFLECT/SURFACE_SREFLECT
    423         }
    424         else
    425         {
    426             //propagate_at_boundary(p, s, rng);     // BOUNDARY_RELECT/BOUNDARY_TRANSMIT
    427             propagate_at_boundary_geant4_style(p, s, rng);     // BOUNDARY_RELECT/BOUNDARY_TRANSMIT
    428             // tacit CONTINUE
    429         }


cu/state.h::

     27 __device__ void fill_state( State& s, int boundary, uint4 identity, float wavelength )
     28 {           
     29     // boundary : 1 based code, signed by cos_theta of photon direction to outward geometric normal
     30     // >0 outward going photon
     31     // <0 inward going photon
     32     //  
     33     // NB the line is above the details of the payload (ie how many float4 per matsur) 
     34     //    it is just 
     35     //                boundaryIndex*4  + 0/1/2/3     for OMAT/OSUR/ISUR/IMAT 
     36     //      
     37         
     38     int line = boundary > 0 ? (boundary - 1)*BOUNDARY_NUM_MATSUR : (-boundary - 1)*BOUNDARY_NUM_MATSUR  ;
     39     
     40     // pick relevant lines depening on boundary sign, ie photon direction relative to normal
     41     // 
     42     int m1_line = boundary > 0 ? line + IMAT : line + OMAT ;
     43     int m2_line = boundary > 0 ? line + OMAT : line + IMAT ;
     44     int su_line = boundary > 0 ? line + ISUR : line + OSUR ;  
     ///
     ///      *su_line*
     ///         of the inner/outer surface for this boundary depending on photon direction
     ///         hmm THAT means logical skin surfaces which have no directionality 
     ///         (as opposed to border surfaces that do)
     ///         would need to be duplicated into ISUR and OSUR ?? is that the case ??
     ///
     ///         this feeds directly into surface or boundary treatment via s.optical = optical_buffer[su_line]
     ///
     45     
     46     //  consider photons arriving at PMT cathode surface
     47     //  geometry normals are expected to be out of the PMT 
     48     //
     49     //  boundary sign will be -ve : so line+3 outer-surface is the relevant one
     50     
     51     s.material1 = boundary_lookup( wavelength, m1_line, 0);  
     52     s.material2 = boundary_lookup( wavelength, m2_line, 0);
     53     s.surface   = boundary_lookup( wavelength, su_line, 0);
     54     
     55     s.optical = optical_buffer[su_line] ;   // index/type/finish/value
     ..




op --bnd : shows no isur/osur duping but there are more skin surfs that border surfs, SO THIS IS A BUG
-----------------------------------------------------------------------------------------------------------

But this is in wrong direction ? Unless equivalent CSur issue ?

::

    delta:ggeo blyth$ op --bnd
    === op-cmdline-binary-match : finds 1st argument with associated binary : --bnd
    248 -rwxr-xr-x  1 blyth  staff  126436 Oct  2 15:49 /usr/local/opticks/lib/GBndLibTest
    proceeding : /usr/local/opticks/lib/GBndLibTest --bnd
    2016-10-02 15:49:56.271 INFO  [1395790] [main@28] /usr/local/opticks/lib/GBndLibTest
    2016-10-02 15:49:56.273 INFO  [1395790] [main@32]  ok 
    2016-10-02 15:49:56.273 INFO  [1395790] [main@36]  loaded blib 
    2016-10-02 15:49:56.278 INFO  [1395790] [main@40]  loaded all  blib 0x7fbc19e09e30 mlib 0x7fbc19e0afd0 slib 0x7fbc19e46090
    2016-10-02 15:49:56.278 INFO  [1395790] [GBndLib::dump@836] GBndLib::dump
    2016-10-02 15:49:56.278 INFO  [1395790] [GBndLib::dump@838] GBndLib::dump ni 123
     (  0) om:                   Vacuum os:                          is:                          im:                   Vacuum
     (  1) om:                   Vacuum os:                          is:                          im:                     Rock
     (  2) om:                     Rock os:                          is:                          im:                      Air
     (  3) om:                      Air os:     NearPoolCoverSurface is:                          im:                      PPE
     (  4) om:                      Air os:                          is:                          im:                Aluminium
     (  5) om:                Aluminium os:                          is:                          im:                     Foam
     (  6) om:                     Foam os:                          is:                          im:                 Bakelite
     (  7) om:                 Bakelite os:                          is:                          im:                      Air
     (  8) om:                      Air os:                          is:                          im:                   MixGas
     (  9) om:                      Air os:                          is:                          im:                      Air
     ( 10) om:                      Air os:                          is:                          im:                     Iron
     ( 11) om:                     Rock os:                          is:                          im:                     Rock
     ( 12) om:                     Rock os:                          is:                          im:                DeadWater
     ( 13) om:                DeadWater os:     NearDeadLinerSurface is:                          im:                    Tyvek
     ( 14) om:                    Tyvek os:                          is:      NearOWSLinerSurface im:                 OwsWater
     ( 15) om:                 OwsWater os:                          is:                          im:                    Tyvek
     ( 16) om:                    Tyvek os:                          is:    NearIWSCurtainSurface im:                 IwsWater
     ( 17) om:                 IwsWater os:                          is:                          im:                 IwsWater
     ( 18) om:                 IwsWater os:     SSTWaterSurfaceNear1 is:                          im:           StainlessSteel
     ( 19) om:           StainlessSteel os:                          is:            SSTOilSurface im:               MineralOil
     ( 20) om:               MineralOil os:                          is:                          im:                  Acrylic
     ( 21) om:                  Acrylic os:                          is:                          im:       LiquidScintillator
     ( 22) om:       LiquidScintillator os:                          is:                          im:                  Acrylic
     ( 23) om:                  Acrylic os:                          is:                          im:                GdDopedLS
     ( 24) om:       LiquidScintillator os:                          is:                          im:                   Teflon
     ( 25) om:       LiquidScintillator os:                          is:                          im:                GdDopedLS
     ( 26) om:                   Teflon os:                          is:                          im:                GdDopedLS
     ( 27) om:               MineralOil os:                          is:                          im:                    Pyrex
     ( 28) om:                    Pyrex os:                          is:                          im:                   Vacuum
     ( 29) om:                   Vacuum os:lvPmtHemiCathodeSensorSurface is:                          im:                 Bialkali
     ( 30) om:                   Vacuum os:                          is:                          im:             OpaqueVacuum
     ( 31) om:               MineralOil os:                          is:                          im:       UnstStainlessSteel
     ( 32) om:               MineralOil os:                          is:                          im:                   Vacuum
     ( 33) om:                   Vacuum os:                          is:                          im:                    Pyrex
     ( 34) om:                   Vacuum os:lvHeadonPmtCathodeSensorSurface is:                          im:                 Bialkali
     ( 35) om:                   Vacuum os:                          is:                          im:                      PVC
     ( 36) om:               MineralOil os:                          is:                          im:           StainlessSteel
     ( 37) om:               MineralOil os:             RSOilSurface is:                          im:                  Acrylic
     ( 38) om:                  Acrylic os:                          is:                          im:                      Air
     ( 39) om:                      Air os:         ESRAirSurfaceTop is:                          im:                      ESR
     ( 40) om:                      Air os:         ESRAirSurfaceBot is:                          im:                      ESR
     ( 41) om:               MineralOil os:                          is:                          im:                   Teflon
     ( 42) om:               MineralOil os:                          is:                          im:       LiquidScintillator
     ( 43) om:                   Vacuum os:                          is:                          im:                    Nylon
     ( 44) om:                   Vacuum os:                          is:                          im:                  Acrylic
     ( 45) om:           StainlessSteel os:                          is:                          im:                GdDopedLS
     ( 46) om:           StainlessSteel os:                          is:                          im:       LiquidScintillator
     ( 47) om:                 IwsWater os:                          is:                          im:                    Water
     ( 48) om:                    Water os:                          is:                          im:           StainlessSteel
     ( 49) om:           StainlessSteel os:                          is:                          im:                 Nitrogen
     ( 50) om:                 Nitrogen os:                          is:                          im:                      BPE
     ( 51) om:                 Nitrogen os:                          is:                          im:           StainlessSteel
     ( 52) om:                 Nitrogen os:                          is:                          im:                   Vacuum
     ( 53) om:                  Acrylic os:                          is:                          im:                    Nylon
     ( 54) om:                  Acrylic os:                          is:                          im:           StainlessSteel
     ( 55) om:                   Vacuum os:                          is:                          im:           StainlessSteel
     ( 56) om:           StainlessSteel os:                          is:                          im:                Aluminium
     ( 57) om:                Aluminium os:                          is:                          im:                    Ge_68
     ( 58) om:                      Air os:                          is:                          im:           StainlessSteel
     ( 59) om:           StainlessSteel os:                          is:                          im:                      Air
     ( 60) om:                      Air os:                          is:                          im:                  Acrylic
     ( 61) om:                  Acrylic os:                          is:                          im:                Aluminium
     ( 62) om:                Aluminium os:                          is:                          im:                    Co_60
     ( 63) om:                  Acrylic os:                          is:                          im:                   Vacuum
     ( 64) om:           StainlessSteel os:                          is:                          im:                   Vacuum
     ( 65) om:                   Vacuum os:                          is:                          im:                     C_13
     ( 66) om:                   Vacuum os:                          is:                          im:                   Silver
     ( 67) om:                 Nitrogen os:                          is:                          im:                  Acrylic
     ( 68) om:                 IwsWater os:                          is:                          im:           StainlessSteel
     ( 69) om:           StainlessSteel os:                          is:                          im:              NitrogenGas
     ( 70) om:              NitrogenGas os:                          is:                          im:                  Acrylic
     ( 71) om:              NitrogenGas os:                          is:                          im:       LiquidScintillator
     ( 72) om:              NitrogenGas os:                          is:                          im:                GdDopedLS
     ( 73) om:                 Nitrogen os:                          is:                          im:                 Nitrogen
     ( 74) om:                 Nitrogen os:                          is:                          im:                GdDopedLS
     ( 75) om:                 Nitrogen os:                          is:                          im:       LiquidScintillator
     ( 76) om:                 IwsWater os:       AdCableTraySurface is:                          im:       UnstStainlessSteel
     ( 77) om:       UnstStainlessSteel os:                          is:                          im:                      BPE
     ( 78) om:                    Water os:                          is:                          im:                 Nitrogen
     ( 79) om:                 Nitrogen os:                          is:                          im:               MineralOil
     ( 80) om:                 IwsWater os:     SSTWaterSurfaceNear2 is:                          im:           StainlessSteel
     ( 81) om:                 IwsWater os:                          is:                          im:                    Pyrex
     ( 82) om:                 IwsWater os:      PmtMtTopRingSurface is:                          im:       UnstStainlessSteel
     ( 83) om:                 IwsWater os:     PmtMtBaseRingSurface is:                          im:       UnstStainlessSteel
     ( 84) om:                 IwsWater os:         PmtMtRib1Surface is:                          im:       UnstStainlessSteel
     ( 85) om:                 IwsWater os:                          is:                          im:       UnstStainlessSteel
     ( 86) om:                 IwsWater os:         PmtMtRib2Surface is:                          im:       UnstStainlessSteel
     ( 87) om:                 IwsWater os:         PmtMtRib3Surface is:                          im:       UnstStainlessSteel
     ( 88) om:                 IwsWater os:       LegInIWSTubSurface is:                          im:    ADTableStainlessSteel
     ( 89) om:                 IwsWater os:        TablePanelSurface is:                          im:    ADTableStainlessSteel
     ( 90) om:                 IwsWater os:       SupportRib1Surface is:                          im:    ADTableStainlessSteel
     ( 91) om:                 IwsWater os:       SupportRib5Surface is:                          im:    ADTableStainlessSteel
     ( 92) om:                 IwsWater os:         SlopeRib1Surface is:                          im:    ADTableStainlessSteel
     ( 93) om:                 IwsWater os:         SlopeRib5Surface is:                          im:    ADTableStainlessSteel
     ( 94) om:                 IwsWater os:  ADVertiCableTraySurface is:                          im:       UnstStainlessSteel
     ( 95) om:                 IwsWater os: ShortParCableTraySurface is:                          im:       UnstStainlessSteel
     ( 96) om:                 IwsWater os:    NearInnInPiperSurface is:                          im:                      PVC
     ( 97) om:                 IwsWater os:   NearInnOutPiperSurface is:                          im:                      PVC
     ( 98) om:                    Tyvek os:                          is:                          im:    ADTableStainlessSteel
     ( 99) om:                 OwsWater os:                          is:                          im:                    Pyrex
     (100) om:                 OwsWater os:      PmtMtTopRingSurface is:                          im:       UnstStainlessSteel
     (101) om:                 OwsWater os:     PmtMtBaseRingSurface is:                          im:       UnstStainlessSteel
     (102) om:                 OwsWater os:         PmtMtRib1Surface is:                          im:       UnstStainlessSteel
     (103) om:                 OwsWater os:                          is:                          im:       UnstStainlessSteel
     (104) om:                 OwsWater os:         PmtMtRib2Surface is:                          im:       UnstStainlessSteel
     (105) om:                 OwsWater os:         PmtMtRib3Surface is:                          im:       UnstStainlessSteel
     (106) om:                 OwsWater os:       LegInOWSTubSurface is:                          im:    ADTableStainlessSteel
     (107) om:                 OwsWater os:      UnistrutRib6Surface is:                          im:       UnstStainlessSteel
     (108) om:                 OwsWater os:      UnistrutRib7Surface is:                          im:       UnstStainlessSteel
     (109) om:                 OwsWater os:      UnistrutRib3Surface is:                          im:       UnstStainlessSteel
     (110) om:                 OwsWater os:      UnistrutRib5Surface is:                          im:       UnstStainlessSteel
     (111) om:                 OwsWater os:      UnistrutRib4Surface is:                          im:       UnstStainlessSteel
     (112) om:                 OwsWater os:      UnistrutRib1Surface is:                          im:       UnstStainlessSteel
     (113) om:                 OwsWater os:      UnistrutRib2Surface is:                          im:       UnstStainlessSteel
     (114) om:                 OwsWater os:      UnistrutRib8Surface is:                          im:       UnstStainlessSteel
     (115) om:                 OwsWater os:      UnistrutRib9Surface is:                          im:       UnstStainlessSteel
     (116) om:                 OwsWater os: TopShortCableTraySurface is:                          im:       UnstStainlessSteel
     (117) om:                 OwsWater os:TopCornerCableTraySurface is:                          im:       UnstStainlessSteel
     (118) om:                 OwsWater os:    VertiCableTraySurface is:                          im:       UnstStainlessSteel
     (119) om:                 OwsWater os:    NearOutInPiperSurface is:                          im:                      PVC
     (120) om:                 OwsWater os:   NearOutOutPiperSurface is:                          im:                      PVC
     (121) om:                DeadWater os:      LegInDeadTubSurface is:                          im:    ADTableStainlessSteel
     (122) om:                     Rock os:                          is:                          im:                  RadRock
    2016-10-02 15:49:56.282 INFO  [1395790] [GPropertyLib::close@318] GPropertyLib::close type GBndLib buf 123,4,2,39,4
    2016-10-02 15:49:56.284 INFO  [1395790] [GItemList::save@114] GItemList::save writing to /tmp/blyth/opticks/GItemList/GBndLib.txt
    2016-10-02 15:49:56.285 INFO  [1395790] [main@59]  after blib saveToCache 
    2016-10-02 15:49:56.285 INFO  [1395790] [main@61]  after blib saveOpticalBuffer 
    /Users/blyth/opticks/bin/op.sh RC 0
    delta:ggeo blyth$ 







optical_buffer
-----------------


::

    delta:optixrap blyth$ opticks-find optical_buffer 
    ./bin/oks.bash:    rtBuffer<uint4>                optical_buffer;   // INPUT 
    ./ggeo/ggeodev.bash:    simon:ggeo blyth$ ./optical_buffer.py 
    ./optixrap/cu/generate.cu:rtBuffer<uint4>                optical_buffer; 
    ./optixrap/cu/generate.cu:             slot == 0 ? optical_buffer[MaterialIndex].x : s.index.z, \
    ./ggeo/GBndLib.cc:    NPY<unsigned int>* optical_buffer = createOpticalBuffer();
    ./ggeo/GBndLib.cc:    setOpticalBuffer(optical_buffer);
    ./ggeo/GBndLib.cc:    saveToCache(optical_buffer, "Optical") ; 
    ./ggeo/GBndLib.cc:    NPY<unsigned int>* optical_buffer = createOpticalBuffer();
    ./ggeo/GBndLib.cc:    setOpticalBuffer(optical_buffer);
    ./ggeo/GBndLib.cc:              << " optical_buffer  " << optical_buffer->getShapeString()
    ./ggeo/GBndLib.cc:    m_optical_buffer(NULL)
    ./ggeo/GBndLib.cc:    return m_optical_buffer ;
    ./ggeo/GBndLib.cc:void GBndLib::setOpticalBuffer(NPY<unsigned int>* optical_buffer)
    ./ggeo/GBndLib.cc:    m_optical_buffer = optical_buffer ;
    ./ggeo/GSurfaceLib.cc:    m_optical_buffer(NULL)
    ./ggeo/GSurfaceLib.cc:    m_optical_buffer = ibuf ; 
    ./ggeo/GSurfaceLib.cc:    return m_optical_buffer ;
    ./optixrap/OBndLib.cc:    optix::Buffer optical_buffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT4, nx );
    ./optixrap/OBndLib.cc:    memcpy( optical_buffer->map(), obuf->getBytes(), numBytes );
    ./optixrap/OBndLib.cc:    optical_buffer->unmap();
    ./optixrap/OBndLib.cc:    m_context["optical_buffer"]->setBuffer(optical_buffer);
    ./ggeo/GBndLib.hh:// Former *GBoundaryLib* encompassed uint4 optical_buffer that 
    ./ggeo/GBndLib.hh:       void setOpticalBuffer(NPY<unsigned int>* optical_buffer);
    ./ggeo/GBndLib.hh:       NPY<unsigned int>*   m_optical_buffer ;  
    ./ggeo/GSurfaceLib.hh:       NPY<unsigned int>*                      m_optical_buffer ; 
    ./optixrap/cu/state.h:    s.optical = optical_buffer[su_line] ;   // index/type/finish/value
    ./optixrap/cu/state.h:    s.index.x = optical_buffer[m1_line].x ; // m1 index
    ./optixrap/cu/state.h:    s.index.y = optical_buffer[m2_line].x ; // m2 index 
    ./optixrap/cu/state.h:    s.index.z = optical_buffer[su_line].x ; // su index
    delta:opticks blyth$ 


::

    delta:geant4_opticks_integration blyth$ op --surf 
    === op-cmdline-binary-match : finds 1st argument with associated binary : --surf
    224 -rwxr-xr-x  1 blyth  staff  112772 Oct  2 15:10 /usr/local/opticks/lib/GSurfaceLibTest
    proceeding : /usr/local/opticks/lib/GSurfaceLibTest
    2016-10-02 15:41:35.411 INFO  [1393288] [GSurfaceLib::Summary@137] GSurfaceLib::dump NumSurfaces 48 NumFloat4 2
    2016-10-02 15:41:35.411 INFO  [1393288] [GSurfaceLib::dump@651]  (index,type,finish,value) 
    2016-10-02 15:41:35.411 WARN  [1393288] [GSurfaceLib::dump@658]           NearPoolCoverSurface (  0,  0,  3,100) 
    2016-10-02 15:41:35.411 WARN  [1393288] [GSurfaceLib::dump@658]           NearDeadLinerSurface (  1,  0,  3, 20) 
    2016-10-02 15:41:35.411 WARN  [1393288] [GSurfaceLib::dump@658]            NearOWSLinerSurface (  2,  0,  3, 20) 
    2016-10-02 15:41:35.411 WARN  [1393288] [GSurfaceLib::dump@658]          NearIWSCurtainSurface (  3,  0,  3, 20) 
    2016-10-02 15:41:35.411 WARN  [1393288] [GSurfaceLib::dump@658]           SSTWaterSurfaceNear1 (  4,  0,  3,100) 
    2016-10-02 15:41:35.411 WARN  [1393288] [GSurfaceLib::dump@658]                  SSTOilSurface (  5,  0,  3,100) 
    2016-10-02 15:41:35.411 WARN  [1393288] [GSurfaceLib::dump@658]  lvPmtHemiCathodeSensorSurface (  6,  0,  3,100) 
    2016-10-02 15:41:35.411 WARN  [1393288] [GSurfaceLib::dump@658] lvHeadonPmtCathodeSensorSurface (  7,  0,  3,100) 
    2016-10-02 15:41:35.411 WARN  [1393288] [GSurfaceLib::dump@658]                   RSOilSurface (  8,  0,  3,100) 
    2016-10-02 15:41:35.411 WARN  [1393288] [GSurfaceLib::dump@658]               ESRAirSurfaceTop (  9,  0,  0,  0) 
    2016-10-02 15:41:35.412 WARN  [1393288] [GSurfaceLib::dump@658]               ESRAirSurfaceBot ( 10,  0,  0,  0) 
    2016-10-02 15:41:35.412 WARN  [1393288] [GSurfaceLib::dump@658]             AdCableTraySurface ( 11,  0,  3,100) 
    2016-10-02 15:41:35.412 WARN  [1393288] [GSurfaceLib::dump@658]           SSTWaterSurfaceNear2 ( 12,  0,  3,100) 
    2016-10-02 15:41:35.412 WARN  [1393288] [GSurfaceLib::dump@658]            PmtMtTopRingSurface ( 13,  0,  3,100) 
    2016-10-02 15:41:35.412 WARN  [1393288] [GSurfaceLib::dump@658]           PmtMtBaseRingSurface ( 14,  0,  3,100) 
    2016-10-02 15:41:35.412 WARN  [1393288] [GSurfaceLib::dump@658]               PmtMtRib1Surface ( 15,  0,  3,100) 
    2016-10-02 15:41:35.412 WARN  [1393288] [GSurfaceLib::dump@658]               PmtMtRib2Surface ( 16,  0,  3,100) 
    2016-10-02 15:41:35.412 WARN  [1393288] [GSurfaceLib::dump@658]               PmtMtRib3Surface ( 17,  0,  3,100) 
    2016-10-02 15:41:35.412 WARN  [1393288] [GSurfaceLib::dump@658]             LegInIWSTubSurface ( 18,  0,  3,100) 
    2016-10-02 15:41:35.412 WARN  [1393288] [GSurfaceLib::dump@658]              TablePanelSurface ( 19,  0,  3,100) 
    2016-10-02 15:41:35.412 WARN  [1393288] [GSurfaceLib::dump@658]             SupportRib1Surface ( 20,  0,  3,100) 
    2016-10-02 15:41:35.412 WARN  [1393288] [GSurfaceLib::dump@658]             SupportRib5Surface ( 21,  0,  3,100) 
    2016-10-02 15:41:35.412 WARN  [1393288] [GSurfaceLib::dump@658]               SlopeRib1Surface ( 22,  0,  3,100) 
    2016-10-02 15:41:35.412 WARN  [1393288] [GSurfaceLib::dump@658]               SlopeRib5Surface ( 23,  0,  3,100) 
    2016-10-02 15:41:35.412 WARN  [1393288] [GSurfaceLib::dump@658]        ADVertiCableTraySurface ( 24,  0,  3,100) 
    2016-10-02 15:41:35.412 WARN  [1393288] [GSurfaceLib::dump@658]       ShortParCableTraySurface ( 25,  0,  3,100) 
    2016-10-02 15:41:35.412 WARN  [1393288] [GSurfaceLib::dump@658]          NearInnInPiperSurface ( 26,  0,  3,100) 
    2016-10-02 15:41:35.412 WARN  [1393288] [GSurfaceLib::dump@658]         NearInnOutPiperSurface ( 27,  0,  3,100) 
    2016-10-02 15:41:35.412 WARN  [1393288] [GSurfaceLib::dump@658]             LegInOWSTubSurface ( 28,  0,  3,100) 
    2016-10-02 15:41:35.412 WARN  [1393288] [GSurfaceLib::dump@658]            UnistrutRib6Surface ( 29,  0,  3,100) 
    2016-10-02 15:41:35.412 WARN  [1393288] [GSurfaceLib::dump@658]            UnistrutRib7Surface ( 30,  0,  3,100) 
    2016-10-02 15:41:35.412 WARN  [1393288] [GSurfaceLib::dump@658]            UnistrutRib3Surface ( 31,  0,  3,100) 
    2016-10-02 15:41:35.412 WARN  [1393288] [GSurfaceLib::dump@658]            UnistrutRib5Surface ( 32,  0,  3,100) 
    2016-10-02 15:41:35.412 WARN  [1393288] [GSurfaceLib::dump@658]            UnistrutRib4Surface ( 33,  0,  3,100) 
    2016-10-02 15:41:35.412 WARN  [1393288] [GSurfaceLib::dump@658]            UnistrutRib1Surface ( 34,  0,  3,100) 
    2016-10-02 15:41:35.412 WARN  [1393288] [GSurfaceLib::dump@658]            UnistrutRib2Surface ( 35,  0,  3,100) 
    2016-10-02 15:41:35.412 WARN  [1393288] [GSurfaceLib::dump@658]            UnistrutRib8Surface ( 36,  0,  3,100) 
    2016-10-02 15:41:35.412 WARN  [1393288] [GSurfaceLib::dump@658]            UnistrutRib9Surface ( 37,  0,  3,100) 
    2016-10-02 15:41:35.412 WARN  [1393288] [GSurfaceLib::dump@658]       TopShortCableTraySurface ( 38,  0,  3,100) 
    2016-10-02 15:41:35.412 WARN  [1393288] [GSurfaceLib::dump@658]      TopCornerCableTraySurface ( 39,  0,  3,100) 
    2016-10-02 15:41:35.412 WARN  [1393288] [GSurfaceLib::dump@658]          VertiCableTraySurface ( 40,  0,  3,100) 
    2016-10-02 15:41:35.412 WARN  [1393288] [GSurfaceLib::dump@658]          NearOutInPiperSurface ( 41,  0,  3,100) 
    2016-10-02 15:41:35.412 WARN  [1393288] [GSurfaceLib::dump@658]         NearOutOutPiperSurface ( 42,  0,  3,100) 
    2016-10-02 15:41:35.413 WARN  [1393288] [GSurfaceLib::dump@658]            LegInDeadTubSurface ( 43,  0,  3,100) 
    2016-10-02 15:41:35.413 WARN  [1393288] [GSurfaceLib::dump@658]           perfectDetectSurface ( 44,  1,  1,100) 
    2016-10-02 15:41:35.413 WARN  [1393288] [GSurfaceLib::dump@658]           perfectAbsorbSurface ( 45,  1,  1,100) 
    2016-10-02 15:41:35.413 WARN  [1393288] [GSurfaceLib::dump@658]         perfectSpecularSurface ( 46,  1,  1,100) 
    2016-10-02 15:41:35.413 WARN  [1393288] [GSurfaceLib::dump@658]          perfectDiffuseSurface ( 47,  1,  1,100) 



closest_hit_propagate
------------------------

oxrap/cu/material1_propagate.cu::

     01 #include <optix.h>
      2 #include "PerRayData_propagate.h"
      3 #include "wavelength_lookup.h"
      4 
      5 //attributes set by TriangleMesh.cu:mesh_intersect 
      6 
      7 rtDeclareVariable(float3,  geometricNormal, attribute geometric_normal, );
      8 rtDeclareVariable(uint4,  instanceIdentity, attribute instance_identity, );
      9 
     10 rtDeclareVariable(PerRayData_propagate, prd, rtPayload, );
     11 rtDeclareVariable(optix::Ray,           ray, rtCurrentRay, );
     12 rtDeclareVariable(float,                  t, rtIntersectionDistance, );
     13 
     14 
     15 RT_PROGRAM void closest_hit_propagate()
     16 {
     17      const float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometricNormal)) ;
     18 
     19      float cos_theta = dot(n,ray.direction);
     20 
     21      prd.cos_theta = cos_theta ;
     22 
     23      prd.distance_to_boundary = t ;
     24 
     25      unsigned int boundaryIndex = instanceIdentity.z ;
     26 
     27      prd.boundary = cos_theta < 0.f ? -(boundaryIndex + 1) : boundaryIndex + 1 ;
     28 
     29      prd.identity = instanceIdentity ;
     30 
     31      prd.surface_normal = cos_theta > 0.f ? -n : n ;
     32 
     33 }


instance_identity comes from the intersects
---------------------------------------------


::

    delta:cu blyth$ grep instance_identity *.*
    TriangleMesh.cu:rtDeclareVariable(uint4, instanceIdentity,   attribute instance_identity,);
    hemi-pmt.cu:rtDeclareVariable(uint4, instanceIdentity,   attribute instance_identity,);
    material1_propagate.cu:rtDeclareVariable(uint4,  instanceIdentity, attribute instance_identity, );
    material1_radiance.cu:rtDeclareVariable(uint4,  instanceIdentity, attribute instance_identity, );
    sphere.cu:rtDeclareVariable(uint4, instanceIdentity,   attribute instance_identity,);


mesh_intersect
----------------

::

    010 // inputs from OGeo
     11 rtBuffer<int3>   indexBuffer;
     12 rtBuffer<float3> vertexBuffer;
     13 rtBuffer<uint4>  identityBuffer;
     14 rtDeclareVariable(unsigned int, instance_index,  ,);
     15 rtDeclareVariable(unsigned int, primitive_count, ,);
     16 
     17 // attribute variables communicating from intersection program to closest hit program
     18 // (must be set between rtPotentialIntersection and rtReportIntersection)
     19 rtDeclareVariable(uint4, instanceIdentity,   attribute instance_identity,);
     20 rtDeclareVariable(float3, geometricNormal, attribute geometric_normal, );
     21 rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
     22 
     23 
     24 
     25 RT_PROGRAM void mesh_intersect(int primIdx)
     26 {
     27     int3 index = indexBuffer[primIdx];
     28 
     29     float3 p0 = vertexBuffer[index.x];
     30     float3 p1 = vertexBuffer[index.y];
     31     float3 p2 = vertexBuffer[index.z];
     32 
     33     uint4 identity = identityBuffer[instance_index*primitive_count+primIdx] ;  // index just primIdx for non-instanced
     34 
     35     float3 n;
     36     float  t, beta, gamma;
     37     if(intersect_triangle(ray, p0, p1, p2, n, t, beta, gamma))
     38     {
     39         if(rtPotentialIntersection( t ))
     40         {
     41             geometricNormal = normalize(n);
     42             instanceIdentity = identity ;
     .. 
     53             rtReportIntersection(0);    // material index 0 
     54         }
     55     }
     56 }



oxrap/cu/hemi-pmt.cu::

    1248 RT_PROGRAM void intersect(int primIdx)
    1249 {
    1250   const uint4& solid    = solidBuffer[primIdx];
    1251   unsigned int numParts = solid.y ;
    1252 
    1253   //const uint4& identity = identityBuffer[primIdx] ; 
    1254   //const uint4 identity = identityBuffer[instance_index*primitive_count+primIdx] ;  // just primIdx for non-instanced
    1255 
    1256   // try with just one identity per-instance 
    1257   uint4 identity = identityBuffer[instance_index] ;
    1258 
    1259 
    1260   for(unsigned int p=0 ; p < numParts ; p++)
    1261   {
    1262       unsigned int partIdx = solid.x + p ;
    1263 
    1264       quad q0, q1, q2, q3 ;
    1265 
    1266       q0.f = partBuffer[4*partIdx+0];
    1267       q1.f = partBuffer[4*partIdx+1];
    1268       q2.f = partBuffer[4*partIdx+2] ;
    1269       q3.f = partBuffer[4*partIdx+3];
    1270 
    1271       identity.z = q1.u.z ;  // boundary from partBuffer (see ggeo-/GPmt)
    1272 
    1273       int partType = q2.i.w ;
    1274 
    1275       // TODO: use enum
    1276       switch(partType)
    1277       {
    1278           case 0:
    1279                 intersect_aabb(q2, q3, identity);
    1280                 break ;
    1281           case 1:
    1282                 intersect_zsphere<false>(q0,q1,q2,q3,identity);
    1283                 break ;



identityBuffer
----------------

::

    delta:cfg4 blyth$ opticks-find identityBuffer
    ./optixrap/cu/hemi-pmt.cu:rtBuffer<uint4>  identityBuffer; 
    ./optixrap/cu/hemi-pmt.cu:  uint4 identity = identityBuffer[instance_index] ; 
    ./optixrap/cu/hemi-pmt.cu:  //const uint4& identity = identityBuffer[primIdx] ; 
    ./optixrap/cu/hemi-pmt.cu:  //const uint4 identity = identityBuffer[instance_index*primitive_count+primIdx] ;  // just primIdx for non-instanced
    ./optixrap/cu/hemi-pmt.cu:  uint4 identity = identityBuffer[instance_index] ; 
    ./optixrap/cu/sphere.cu:rtBuffer<uint4>  identityBuffer; 
    ./optixrap/cu/sphere.cu:  uint4 identity = identityBuffer[instance_index*primitive_count+primIdx] ;  // just primIdx for non-instanced
    ./optixrap/cu/TriangleMesh.cu:rtBuffer<uint4>  identityBuffer; 
    ./optixrap/cu/TriangleMesh.cu:    uint4 identity = identityBuffer[instance_index*primitive_count+primIdx] ;  // index just primIdx for non-instanced
    ./ggeo/GPmt.cc:    792   const uint4& identity = identityBuffer[primIdx] ;
    ./optixrap/OGeo.cc:    optix::Buffer identityBuffer = createInputBuffer<optix::uint4, unsigned int>( idBuf, RT_FORMAT_UNSIGNED_INT4, 1 , "identityBuffer"); 
    ./optixrap/OGeo.cc:    geometry["identityBuffer"]->setBuffer(identityBuffer);
    ./optixrap/OGeo.cc:   optix::Buffer identityBuffer = createInputBuffer<optix::uint4>( id, RT_FORMAT_UNSIGNED_INT4, 1 , "identityBuffer"); 
    ./optixrap/OGeo.cc:   geometry["identityBuffer"]->setBuffer(identityBuffer);


OGeo.cc::

    537     optix::Geometry geometry = m_context->createGeometry();
    538     geometry->setIntersectionProgram(m_ocontext->createProgram("TriangleMesh.cu.ptx", "mesh_intersect"));
    539     geometry->setBoundingBoxProgram(m_ocontext->createProgram("TriangleMesh.cu.ptx", "mesh_bounds"));
    540 
    541     unsigned int numSolids = mm->getNumSolids();
    542     unsigned int numFaces = mm->getNumFaces();
    543     unsigned int numITransforms = mm->getNumITransforms();
    544 
    545     geometry->setPrimitiveCount(numFaces);
    546     assert(geometry->getPrimitiveCount() == numFaces);
    547     geometry["primitive_count"]->setUint( geometry->getPrimitiveCount() );  // needed for instanced offsets 
    548 
    549     LOG(trace) << "OGeo::makeTriangulatedGeometry "
    550               << " mmIndex " << mm->getIndex()
    551               << " numFaces (PrimitiveCount) " << numFaces
    552               << " numSolids " << numSolids
    553               << " numITransforms " << numITransforms
    554               ;
    555 
    556 
    557     GBuffer* id = NULL ;
    558     if(numITransforms > 0)
    559     {
    560         id = mm->getFaceRepeatedInstancedIdentityBuffer();
    561         assert(id);
    562         LOG(trace) << "OGeo::makeTriangulatedGeometry using FaceRepeatedInstancedIdentityBuffer"
    563                   << " friid items " << id->getNumItems()
    564                   << " numITransforms*numFaces " << numITransforms*numFaces
    565                   ;
    566 
    567         assert( id->getNumItems() == numITransforms*numFaces );
    568    }
    569    else
    570    {
    571         id = mm->getFaceRepeatedIdentityBuffer();
    572         assert(id);
    573         LOG(trace) << "OGeo::makeTriangulatedGeometry using FaceRepeatedIdentityBuffer"
    574                   << " frid items " << id->getNumItems()
    575                   << " numFaces " << numFaces
    576                   ;
    577         assert( id->getNumItems() == numFaces );
    578    }
    579 
    580    optix::Buffer identityBuffer = createInputBuffer<optix::uint4>( id, RT_FORMAT_UNSIGNED_INT4, 1 , "identityBuffer");
    581    geometry["identityBuffer"]->setBuffer(identityBuffer);



FaceRepeatedIdentityBuffer
-----------------------------

::

    delta:opticks blyth$ opticks-find FaceRepeatedIdentityBuffer
    ./ggeo/GMesh.cc:GBuffer* GMesh::makeFaceRepeatedIdentityBuffer()
    ./ggeo/GMesh.cc:        LOG(warning) << "GMesh::makeFaceRepeatedIdentityBuffer only relevant to non-instanced meshes " ;
    ./ggeo/GMesh.cc:    LOG(info) << "GMesh::makeFaceRepeatedIdentityBuffer"
    ./ggeo/GMesh.cc:GBuffer*  GMesh::getFaceRepeatedIdentityBuffer()
    ./ggeo/GMesh.cc:         m_facerepeated_identity_buffer = makeFaceRepeatedIdentityBuffer() ;  
    ./ggeo/tests/GGeoTest.cc:        GBuffer* frid = mm->getFaceRepeatedIdentityBuffer();
    ./optixrap/OGeo.cc:        id = mm->getFaceRepeatedIdentityBuffer();
    ./optixrap/OGeo.cc:        LOG(trace) << "OGeo::makeTriangulatedGeometry using FaceRepeatedIdentityBuffer"
    ./ggeo/GMesh.hh:      GBuffer* getFaceRepeatedIdentityBuffer(); 
    ./ggeo/GMesh.hh:      GBuffer* makeFaceRepeatedIdentityBuffer();




Face repeated from the solid level m_identity::

    1884 GBuffer* GMesh::makeFaceRepeatedIdentityBuffer()
    1885 {
    ....
    1902     guint4* nodeinfo = getNodeInfo();
    ....
    1916     // duplicate nodeinfo for each solid out to each face
    1917     unsigned int offset(0);
    1918     guint4* rid = new guint4[numFaces] ;
    1919     for(unsigned int s=0 ; s < numSolids ; s++)
    1920     {  
    1921         guint4 sid = m_identity[s]  ;
    1922         unsigned int nf = (nodeinfo + s)->x ;
    1923         for(unsigned int f=0 ; f < nf ; ++f) rid[offset+f] = sid ;
    1924         offset += nf ;
    1925     } 
    1926    
    1927     unsigned int size = sizeof(guint4) ;
    1928     GBuffer* buffer = new GBuffer( size*numFaces, (void*)rid, size, 4 );
    1929     return buffer ;
    1930 }


    1935 GBuffer*  GMesh::getFaceRepeatedIdentityBuffer()
    1936 {
    1937     if(m_facerepeated_identity_buffer == NULL)
    1938     {
    1939          m_facerepeated_identity_buffer = makeFaceRepeatedIdentityBuffer() ;
    1940     }
    1941     return m_facerepeated_identity_buffer ;
    1942 }
    1943 

    delta:optixrap blyth$ opticks-find getFaceRepeatedIdentityBuffer 
    ./ggeo/GMesh.cc:GBuffer*  GMesh::getFaceRepeatedIdentityBuffer()
    ./ggeo/tests/GGeoTest.cc:        GBuffer* frid = mm->getFaceRepeatedIdentityBuffer();
    ./optixrap/OGeo.cc:        id = mm->getFaceRepeatedIdentityBuffer();
    ./ggeo/GMesh.hh:      GBuffer* getFaceRepeatedIdentityBuffer(); 


Solid level identity are merged into m_identity within GMergedMesh methods such as::

    398 void GMergedMesh::mergeSolid( GSolid* solid, bool selected )
    399 {
    400     GMesh* mesh = solid->getMesh();
    401     unsigned int nvert = mesh->getNumVertices();
    402     unsigned int nface = mesh->getNumFaces();
    403     guint4 _identity = solid->getIdentity();
    ...
    411 
    412    if(m_verbosity > 1)
    413    {
    414 
    415         const char* pvn = solid->getPVName() ;
    416         const char* lvn = solid->getLVName() ;
    417 
    418         LOG(info) << "GMergedMesh::mergeSolid"
    419                   << " m_cur_solid " << m_cur_solid
    420                   << " idx " << solid->getIndex()
    421                   << " id " << _identity.description()
    422                   << " pv " << ( pvn ? pvn : "-" )
    423                   << " lv " << ( lvn ? lvn : "-" )
    424                   << " bb " << bb.description()
    425                   ;
    426         transform->Summary("GMergedMesh::mergeSolid transform");
    427    }
    428 
    429 
    430     unsigned int boundary = solid->getBoundary();
    431     NSensor* sensor = solid->getSensor();
    432 
    433     unsigned int nodeIndex = solid->getIndex();
    434     unsigned int meshIndex = mesh->getIndex();
    435     unsigned int sensorIndex = NSensor::RefIndex(sensor) ;
    436     assert(_identity.x == nodeIndex);
    437     assert(_identity.y == meshIndex);
    438     assert(_identity.z == boundary);
    439     //assert(_identity.w == sensorIndex);   this is no longer the case, now require SensorSurface in the identity
    440    


::

     920 void GMesh::setIdentity(guint4* identity)
     921 {
     922     m_identity = identity ;
     923     assert(m_num_solids > 0);
     924     unsigned int size = sizeof(guint4);
     925     assert(size == sizeof(unsigned int)*4 );
     926     m_identity_buffer = new GBuffer( size*m_num_solids, (void*)m_identity, size, 4 );
     927 }

::

    delta:ggeo blyth$ opticks-find setIdentity
    ./ggeo/GMesh.cc:    setIdentity(new guint4[numSolids]);
    ./ggeo/GMesh.cc:    if(strcmp(name, identity_) == 0)        setIdentityBuffer(buffer) ; 
    ./ggeo/GMesh.cc:void GMesh::setIdentity(guint4* identity)  
    ./ggeo/GMesh.cc:void GMesh::setIdentityBuffer(GBuffer* buffer) 
    ./ggeo/GTreeCheck.cc:     // cf GMesh::setIdentity
    ./ggeo/GMesh.hh:      void setIdentityBuffer(GBuffer* buffer);
    ./ggeo/GMesh.hh:      void setIdentity(guint4* identity);
    delta:opticks blyth$ 







From cache, see only node level identity, vaguely recall that face repeating is done dynamically and not persisted::

    In [1]: import numpy as np

    In [2]: i = np.load("identity.npy")

    In [3]: i
    Out[3]: 
    array([[    0,   248,     0,     0],
           [    1,   247,     1,     0],
           [    2,    21,     2,     0],
           ..., 
           [12227,   243,   122,     0],
           [12228,   244,   122,     0],
           [12229,   245,   122,     0]], dtype=uint32)

    In [4]: i.shape
    Out[4]: (12230, 4)

    In [5]: ii = np.load("iidentity.npy")

    In [6]: ii.shape
    Out[6]: (12230, 4)

    In [7]: ii
    Out[7]: 
    array([[    0,   248,     0,     0],
           [    1,   247,     1,     0],
           [    2,    21,     2,     0],
           ..., 
           [12227,   243,   122,     0],
           [12228,   244,   122,     0],
           [12229,   245,   122,     0]], dtype=uint32)



