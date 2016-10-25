tlaser
========

Shoot horizontal laser in X direction (vertical geometry too involved)::

     19 tlaser--(){
     21     local msg="=== $FUNCNAME :"
     23     local det=$(tlaser-det)
     24     local tag=$(tlaser-tag)
     25 
     26     local torch_config=(
     27                  type=point
     28                  frame=3153
     29                  source=0,0,0
     30                  target=1,0,0
     31                  photons=10000
     32                  material=GdDopedLS
     33                  wavelength=430
     34                  weight=1.0
     35                  time=0.1
     36                  zenithazimuth=0,1,0,1
     37                  radius=0
     38                )
     40     op.sh  \
     41             $* \
     42             --animtimemax 15 \
     43             --timemax 15 \
     44             --eye 0,1,0 \
     45             --torch --torchconfig "$(join _ ${torch_config[@]})" \
     46             --torchdbg \
     47             --save --tag $tag --cat $det
     51 }

::

    tlaser- ; tlaser-- --okg4 --compute





After REJOIN fix still large discreps, eg top line SA
---------------------------------------------------------

::

    tlaser-;tlaser-t
    tlaser.py 

         seqhis_ana     1:laser     -1:laser           c2 
              8ccccd        763501       813497          1585.04  [6 ] TO BT BT BT BT SA     
          cccc9ccccd         25263        26200            17.06  [10] TO BT BT BT BT DR BT BT BT BT
                            
    In [2]: 25263./(763501.+25263.) 
    Out[2]: 0.03202859156858072

    In [3]: 26200./(813497.+26200.)
    Out[3]: 0.0312017311006232


    In [1]: 813497./763501.     TODO: include the ratio in the output  (expected reflectivity is ballpark 4%)
    Out[1]: 1.0654825599442568


                  4d         55825        47634           648.49  [2 ] TO AB
             8cccc6d         19707        18533            36.04  [7 ] TO SC BT BT BT BT SA
                4ccd         12576        11563            42.51  [4 ] TO BT BT AB
             8cccc5d         11183         7742           625.65  [7 ] TO RE BT BT BT BT SA
              4ccccd          8554         8756             2.36  [6 ] TO BT BT BT BT AB
                 45d          7531         2208          2909.37  [3 ] TO RE AB
            8cccc55d          5362         2116          1409.00  [8 ] TO RE RE BT BT BT BT SA
             8cc6ccd          4109         4155             0.26  [7 ] TO BT BT SC BT BT SA
                455d          3588          621          2091.49  [4 ] TO RE RE AB
             86ccccd          2836         2743             1.55  [7 ] TO BT BT BT BT SC SA
          cccccc6ccd          2674         1919           124.11  [10] TO BT BT SC BT BT BT BT BT BT
           8cccc555d          2524          610          1168.92  [9 ] TO RE RE RE BT BT BT BT SA
             8cc5ccd          2359         1866            57.53  [7 ] TO BT BT RE BT BT SA
             89ccccd          1880         2221            28.35  [7 ] TO BT BT BT BT DR SA
          cacccccc6d          2210         2127             1.59  [10] TO SC BT BT BT BT BT BT SR BT
                 46d          2118         1569            81.75  [3 ] TO SC AB
          cccc6ccccd          2060         1752            24.89  [10] TO BT BT BT BT SC BT BT BT BT
               4cccd          1940         1981             0.43  [5 ] TO BT BT BT AB
                         1000000      1000000       106.82 
 

Dump top line, RSOilSurface as dielectric_metal when its MO/Ac ?::

    tlaser-;tlaser-t --dbgseqhis 8ccccd 


    ----CRecorder::compare---- record_id        5 --dindex 
    2016-10-25 20:11:36.056 INFO  [3525262] [CRecorder::Dump@847] CRecorder::compare (rdr-dump)DONE record_id       5
    2016-10-25 20:11:36.056 INFO  [3525262] [CRecorder::Dump@850]  seqhis 8ccccd TORCH BOUNDARY_TRANSMIT BOUNDARY_TRANSMIT BOUNDARY_TRANSMIT BOUNDARY_TRANSMIT SURFACE_ABSORB . . . . . . . . . . 
    2016-10-25 20:11:36.056 INFO  [3525262] [CRecorder::Dump@854]  seqmat 343231 GdDopedLS Acrylic LiquidScintillator Acrylic MineralOil Acrylic - - - - - - - - - - 
    2016-10-25 20:11:36.056 INFO  [3525262] [CRec::dump@40] crec record_id 5 nstp 5  Ori[ -18079.453-799699.438-6605.000] 
        0[   0](Stp ;opticalphoton stepNum 1513010768(tk ;opticalphoton tid 6 pid 0 nm    430 mm  ori[ -18079.453-799699.438-6605.000]  pos[ 1255.240-1878.345   0.000]  )
      pre d/Geometry/AD/lvIAV#pvGDS rials/GdDopedLS          noProc           Undefined pos[      0.000     0.000     0.000]  dir[    0.556  -0.831   0.000]  pol[   -1.000   0.023   0.000]  ns  0.100 nm 430.000
     post d/Geometry/AD/lvLSO#pvIAV terials/Acrylic  Transportation        GeomBoundary pos[    861.221 -1288.733     0.000]  dir[    0.556  -0.831   0.000]  pol[   -1.000   0.023   0.000]  ns  8.059 nm 430.000
     )
        1[   1](Stp ;opticalphoton stepNum 1513010768(tk ;opticalphoton tid 6 pid 0 nm    430 mm  ori[ -18079.453-799699.438-6605.000]  pos[ 1255.240-1878.345   0.000]  )
      pre d/Geometry/AD/lvLSO#pvIAV terials/Acrylic  Transportation        GeomBoundary pos[    861.221 -1288.733     0.000]  dir[    0.556  -0.831   0.000]  pol[   -1.000   0.023   0.000]  ns  8.059 nm 430.000
     post d/Geometry/AD/lvOAV#pvLSO uidScintillator  Transportation        GeomBoundary pos[    866.777 -1297.048     0.000]  dir[    0.556  -0.831   0.000]  pol[   -1.000   0.023   0.000]  ns  8.110 nm 430.000
     )
        2[   2](Stp ;opticalphoton stepNum 1513010768(tk ;opticalphoton tid 6 pid 0 nm    430 mm  ori[ -18079.453-799699.438-6605.000]  pos[ 1255.240-1878.345   0.000]  )
      pre d/Geometry/AD/lvOAV#pvLSO uidScintillator  Transportation        GeomBoundary pos[    866.777 -1297.048     0.000]  dir[    0.556  -0.831   0.000]  pol[   -1.000   0.023   0.000]  ns  8.110 nm 430.000
     post d/Geometry/AD/lvOIL#pvOAV terials/Acrylic  Transportation        GeomBoundary pos[   1101.250 -1647.913     0.000]  dir[    0.556  -0.831   0.000]  pol[   -1.000   0.023   0.000]  ns 10.301 nm 430.000
     )
        3[   3](Stp ;opticalphoton stepNum 1513010768(tk ;opticalphoton tid 6 pid 0 nm    430 mm  ori[ -18079.453-799699.438-6605.000]  pos[ 1255.240-1878.345   0.000]  )
      pre d/Geometry/AD/lvOIL#pvOAV terials/Acrylic  Transportation        GeomBoundary pos[   1101.250 -1647.913     0.000]  dir[    0.556  -0.831   0.000]  pol[   -1.000   0.023   0.000]  ns 10.301 nm 430.000
     post d/Geometry/AD/lvSST#pvOIL ials/MineralOil  Transportation        GeomBoundary pos[   1111.251 -1662.879     0.000]  dir[    0.556  -0.831   0.000]  pol[   -1.000   0.023   0.000]  ns 10.393 nm 430.000
     )
        4[   4](Stp ;opticalphoton stepNum 1513010768(tk ;opticalphoton tid 6 pid 0 nm    430 mm  ori[ -18079.453-799699.438-6605.000]  pos[ 1255.240-1878.345   0.000]  )
      pre d/Geometry/AD/lvSST#pvOIL ials/MineralOil  Transportation        GeomBoundary pos[   1111.251 -1662.879     0.000]  dir[    0.556  -0.831   0.000]  pol[   -1.000   0.023   0.000]  ns 10.393 nm 430.000
     post D/lvOIL#pvRadialShield:20 terials/Acrylic  Transportation        GeomBoundary pos[   1255.240 -1878.345     0.000]  dir[    0.556  -0.831   0.000]  pol[   -1.000   0.023   0.000]  ns 11.738 nm 430.000
     )
    2016-10-25 20:11:36.057 INFO  [3525262] [*DsG4OpBoundaryProcess::PostStepDoIt@442] OpticalSurface  name RSOilSurface thePhotonMomentum (eV) 2.88335 theReflectivity 0.0409174 theEfficiency 0. dielectric_metal  ground - m1 /dd/Materials/MineralOil m2 /dd/Materials/Acrylic
    2016-10-25 20:11:36.057 INFO  [3525262] [*DsG4OpBoundaryProcess::PostStepDoIt@442] OpticalSurface  name RSOilSurface thePhotonMomentum (eV) 2.88335 theReflectivity 0.0409174 theEfficiency 0. dielectric_metal  ground - m1 /dd/Materials/MineralOil m2 /dd/Materials/Acrylic





::

    2016-10-25 20:11:31.336 INFO  [3525262] [GSurLib::dump@196] GGeo::loadFromCache GSurLib::dump
        0 S(   0                NearPoolCoverSurface)  nlv   1 npvp   1  [ obnd    3:Air/NearPoolCoverSurface//PPE] 
        1 B(   1                NearDeadLinerSurface)  nlv   1 npvp   1  [ obnd   13:DeadWater/NearDeadLinerSurface//Tyvek] 
        2 B(   2                 NearOWSLinerSurface)  nlv   1 npvp   1  [ ibnd   14:Tyvek//NearOWSLinerSurface/OwsWater] 
        3 B(   3               NearIWSCurtainSurface)  nlv   1 npvp   1  [ ibnd   16:Tyvek//NearIWSCurtainSurface/IwsWater] 
        4 B(   4                SSTWaterSurfaceNear1)  nlv   1 npvp   1  [ obnd   18:IwsWater/SSTWaterSurfaceNear1//StainlessSteel] 
        5 B(   5                       SSTOilSurface)  nlv   1 npvp   2  [ ibnd   19:StainlessSteel//SSTOilSurface/MineralOil] 
        6 S(   6       lvPmtHemiCathodeSensorSurface)  nlv   1 npvp 672  [ obnd   29:Vacuum/lvPmtHemiCathodeSensorSurface//Bialkali] 
        7 S(   7     lvHeadonPmtCathodeSensorSurface)  nlv   1 npvp  12  [ obnd   34:Vacuum/lvHeadonPmtCathodeSensorSurface//Bialkali] 
        8 S(   8                        RSOilSurface)  nlv   1 npvp  64  [ obnd   37:MineralOil/RSOilSurface//Acrylic]                <-- FLIPPED ???
        9 B(   9                    ESRAirSurfaceTop)  nlv   1 npvp   2  [ obnd   39:Air/ESRAirSurfaceTop//ESR] 
       10 B(  10                    ESRAirSurfaceBot)  nlv   1 npvp   2  [ obnd   40:Air/ESRAirSurfaceBot//ESR] 
       11 S(  11                  AdCableTraySurface)  nlv   1 npvp   2  [ obnd   76:IwsWater/AdCableTraySurface//UnstStainlessSteel] 
       12 B(  12                SSTWaterSurfaceNear2)  nlv   1 npvp   1  [ obnd   80:IwsWater/SSTWaterSurfaceNear2//StainlessSteel] 


::

    op --surf 8    ## type 0, is dielectric_metal ... TODO: trace this 


    2016-10-25 20:29:13.727 INFO  [3530462] [GSurfaceLib::dump@717]  (  8,  0,  3,100) GPropertyMap<T>::  8        surface s: GOpticalSurface  type 0 model 1 finish 3 value     1                  RSOilSurface k:detect absorb reflect_specular reflect_diffuse extra_x extra_y extra_z extra_w RSOilSurface
                  domain              detect              absorb    reflect_specular     reflect_diffuse             extra_x
                      60                   0               0.827                   0               0.173                  -1
                      80                   0            0.827015                   0            0.172985                  -1
                     100                   0             0.85649                   0             0.14351                  -1
                     120                   0            0.885965                   0            0.114035                  -1
                     140                   0            0.897743                   0            0.102257                  -1
                     160                   0            0.909501                   0           0.0904994                  -1
                     180                   0            0.921258                   0           0.0787423                  -1
                     200                   0            0.933007                   0           0.0669933                  -1
                     220                   0            0.938282                   0           0.0617179                  -1
                     240                   0            0.943557                   0           0.0564426                  -1
                     260                   0            0.947648                   0           0.0523518                  -1
                     280                   0             0.95055                   0           0.0494499                  -1
                     300                   0            0.953451                   0           0.0465491                  -1
                     320                   0            0.954789                   0           0.0452105                  -1
                     340                   0            0.956128                   0            0.043872                  -1



Optical Surface Trace
------------------------

Other than perfect additions all surfaces are type=dielectric_metal with finish ground 
(other than ESRAir.. which is polished)

Looks to be a surface type bug.

Hmm the perfect surfaces listed as finish: polishedfrontpainted

::

     61 enum G4OpticalSurfaceFinish
     62 {
     63    polished,                    // smooth perfectly polished surface
     64    polishedfrontpainted,        // smooth top-layer (front) paint
     65    polishedbackpainted,         // same is 'polished' but with a back-paint
     66 
     67    ground,                      // rough surface
     68    groundfrontpainted,          // rough top-layer (front) paint
     69    groundbackpainted,           // same as 'ground' but with a back-paint

::

     65 enum G4SurfaceType
     66 {
     67    dielectric_metal,            // dielectric-metal interface
     68    dielectric_dielectric,       // dielectric-dielectric interface
     69    dielectric_LUT,              // dielectric-Look-Up-Table interface
     70    dielectric_dichroic,         // dichroic filter interface
     71    firsov,                      // for Firsov Process
     72    x_ray                        // for x-ray mirror process
     73 };
     74 
     75 /////////////////////
     76 // Class Definition
     77 /////////////////////
     78 
     79 class G4SurfaceProperty
     80 {

::

    op --surf

    2016-10-25 20:54:23.188 INFO  [3537695] [GSurfaceLib::Summary@137] GSurfaceLib::dump NumSurfaces 48 NumFloat4 2
    2016-10-25 20:54:23.189 INFO  [3537695] [GSurfaceLib::dump@651]  (index,type,finish,value) 
    2016-10-25 20:54:23.189 WARN  [3537695] [GSurfaceLib::dump@658]           NearPoolCoverSurface (  0,  0,  3,100) 
    2016-10-25 20:54:23.189 WARN  [3537695] [GSurfaceLib::dump@658]           NearDeadLinerSurface (  1,  0,  3, 20) 
    2016-10-25 20:54:23.189 WARN  [3537695] [GSurfaceLib::dump@658]            NearOWSLinerSurface (  2,  0,  3, 20) 
    2016-10-25 20:54:23.189 WARN  [3537695] [GSurfaceLib::dump@658]          NearIWSCurtainSurface (  3,  0,  3, 20) 
    2016-10-25 20:54:23.189 WARN  [3537695] [GSurfaceLib::dump@658]           SSTWaterSurfaceNear1 (  4,  0,  3,100) 
    2016-10-25 20:54:23.189 WARN  [3537695] [GSurfaceLib::dump@658]                  SSTOilSurface (  5,  0,  3,100) 
    2016-10-25 20:54:23.189 WARN  [3537695] [GSurfaceLib::dump@658]  lvPmtHemiCathodeSensorSurface (  6,  0,  3,100) 
    2016-10-25 20:54:23.189 WARN  [3537695] [GSurfaceLib::dump@658] lvHeadonPmtCathodeSensorSurface (  7,  0,  3,100) 
    2016-10-25 20:54:23.189 WARN  [3537695] [GSurfaceLib::dump@658]                   RSOilSurface (  8,  0,  3,100) 
    2016-10-25 20:54:23.189 WARN  [3537695] [GSurfaceLib::dump@658]               ESRAirSurfaceTop (  9,  0,  0,  0) 
    2016-10-25 20:54:23.189 WARN  [3537695] [GSurfaceLib::dump@658]               ESRAirSurfaceBot ( 10,  0,  0,  0) 
    2016-10-25 20:54:23.189 WARN  [3537695] [GSurfaceLib::dump@658]             AdCableTraySurface ( 11,  0,  3,100) 
    2016-10-25 20:54:23.189 WARN  [3537695] [GSurfaceLib::dump@658]           SSTWaterSurfaceNear2 ( 12,  0,  3,100) 
    2016-10-25 20:54:23.189 WARN  [3537695] [GSurfaceLib::dump@658]            PmtMtTopRingSurface ( 13,  0,  3,100) 
    2016-10-25 20:54:23.189 WARN  [3537695] [GSurfaceLib::dump@658]           PmtMtBaseRingSurface ( 14,  0,  3,100) 
    2016-10-25 20:54:23.189 WARN  [3537695] [GSurfaceLib::dump@658]               PmtMtRib1Surface ( 15,  0,  3,100) 
    2016-10-25 20:54:23.189 WARN  [3537695] [GSurfaceLib::dump@658]               PmtMtRib2Surface ( 16,  0,  3,100) 
    2016-10-25 20:54:23.189 WARN  [3537695] [GSurfaceLib::dump@658]               PmtMtRib3Surface ( 17,  0,  3,100) 
    2016-10-25 20:54:23.189 WARN  [3537695] [GSurfaceLib::dump@658]             LegInIWSTubSurface ( 18,  0,  3,100) 
    2016-10-25 20:54:23.189 WARN  [3537695] [GSurfaceLib::dump@658]              TablePanelSurface ( 19,  0,  3,100) 
    2016-10-25 20:54:23.189 WARN  [3537695] [GSurfaceLib::dump@658]             SupportRib1Surface ( 20,  0,  3,100) 
    2016-10-25 20:54:23.189 WARN  [3537695] [GSurfaceLib::dump@658]             SupportRib5Surface ( 21,  0,  3,100) 
    2016-10-25 20:54:23.189 WARN  [3537695] [GSurfaceLib::dump@658]               SlopeRib1Surface ( 22,  0,  3,100) 
    2016-10-25 20:54:23.189 WARN  [3537695] [GSurfaceLib::dump@658]               SlopeRib5Surface ( 23,  0,  3,100) 
    2016-10-25 20:54:23.189 WARN  [3537695] [GSurfaceLib::dump@658]        ADVertiCableTraySurface ( 24,  0,  3,100) 
    2016-10-25 20:54:23.189 WARN  [3537695] [GSurfaceLib::dump@658]       ShortParCableTraySurface ( 25,  0,  3,100) 
    2016-10-25 20:54:23.189 WARN  [3537695] [GSurfaceLib::dump@658]          NearInnInPiperSurface ( 26,  0,  3,100) 
    2016-10-25 20:54:23.189 WARN  [3537695] [GSurfaceLib::dump@658]         NearInnOutPiperSurface ( 27,  0,  3,100) 
    2016-10-25 20:54:23.189 WARN  [3537695] [GSurfaceLib::dump@658]             LegInOWSTubSurface ( 28,  0,  3,100) 
    2016-10-25 20:54:23.189 WARN  [3537695] [GSurfaceLib::dump@658]            UnistrutRib6Surface ( 29,  0,  3,100) 
    2016-10-25 20:54:23.189 WARN  [3537695] [GSurfaceLib::dump@658]            UnistrutRib7Surface ( 30,  0,  3,100) 
    2016-10-25 20:54:23.189 WARN  [3537695] [GSurfaceLib::dump@658]            UnistrutRib3Surface ( 31,  0,  3,100) 
    2016-10-25 20:54:23.190 WARN  [3537695] [GSurfaceLib::dump@658]            UnistrutRib5Surface ( 32,  0,  3,100) 
    2016-10-25 20:54:23.190 WARN  [3537695] [GSurfaceLib::dump@658]            UnistrutRib4Surface ( 33,  0,  3,100) 
    2016-10-25 20:54:23.190 WARN  [3537695] [GSurfaceLib::dump@658]            UnistrutRib1Surface ( 34,  0,  3,100) 
    2016-10-25 20:54:23.190 WARN  [3537695] [GSurfaceLib::dump@658]            UnistrutRib2Surface ( 35,  0,  3,100) 
    2016-10-25 20:54:23.190 WARN  [3537695] [GSurfaceLib::dump@658]            UnistrutRib8Surface ( 36,  0,  3,100) 
    2016-10-25 20:54:23.190 WARN  [3537695] [GSurfaceLib::dump@658]            UnistrutRib9Surface ( 37,  0,  3,100) 
    2016-10-25 20:54:23.190 WARN  [3537695] [GSurfaceLib::dump@658]       TopShortCableTraySurface ( 38,  0,  3,100) 
    2016-10-25 20:54:23.190 WARN  [3537695] [GSurfaceLib::dump@658]      TopCornerCableTraySurface ( 39,  0,  3,100) 
    2016-10-25 20:54:23.190 WARN  [3537695] [GSurfaceLib::dump@658]          VertiCableTraySurface ( 40,  0,  3,100) 
    2016-10-25 20:54:23.190 WARN  [3537695] [GSurfaceLib::dump@658]          NearOutInPiperSurface ( 41,  0,  3,100) 
    2016-10-25 20:54:23.190 WARN  [3537695] [GSurfaceLib::dump@658]         NearOutOutPiperSurface ( 42,  0,  3,100) 
    2016-10-25 20:54:23.190 WARN  [3537695] [GSurfaceLib::dump@658]            LegInDeadTubSurface ( 43,  0,  3,100) 
    2016-10-25 20:54:23.190 WARN  [3537695] [GSurfaceLib::dump@658]           perfectDetectSurface ( 44,  1,  1,100) 
    2016-10-25 20:54:23.190 WARN  [3537695] [GSurfaceLib::dump@658]           perfectAbsorbSurface ( 45,  1,  1,100) 
    2016-10-25 20:54:23.190 WARN  [3537695] [GSurfaceLib::dump@658]         perfectSpecularSurface ( 46,  1,  1,100) 
    2016-10-25 20:54:23.190 WARN  [3537695] [GSurfaceLib::dump@658]          perfectDiffuseSurface ( 47,  1,  1,100) 



::

    248 void G4DAEWriteStructure::
    249 OpticalSurfaceWrite(xercesc::DOMElement* targetElement,
    250                     const G4OpticalSurface* const surf)
    251 {
    252    xercesc::DOMElement* optElement = NewElement("opticalsurface");
    253    G4OpticalSurfaceModel smodel = surf->GetModel();
    254    G4double sval = (smodel==glisur) ? surf->GetPolish() : surf->GetSigmaAlpha();
    255 
    256    optElement->setAttributeNode(NewNCNameAttribute("name", surf->GetName()));
    257    optElement->setAttributeNode(NewAttribute("model", smodel));
    258    optElement->setAttributeNode(NewAttribute("finish", surf->GetFinish()));
    259    optElement->setAttributeNode(NewAttribute("type", surf->GetType()));
    260    optElement->setAttributeNode(NewAttribute("value", sval));
    261 
    262    G4MaterialPropertiesTable* ptable = surf->GetMaterialPropertiesTable();
    263    PropertyWrite( optElement, ptable );
    264 
    265    targetElement->appendChild(optElement);
    266 }





Prior to fixing aim
----------------------


::
    delta:ana blyth$ tlaser.py  ## apply seqhis selection to pick the most common seqs for A and B

      A:seqhis_ana       noname 
              8ccccd        1.000           7673       [6 ] TO BT BT BT BT SA
                            7673         1.00 
       B:seqhis_ana       noname 
            8c0cc0cd        1.000           7030       [8 ] TO BT ?0? BT BT ?0? BT SA
                            7030         1.00 



Laser aim issue
-------------------

Huh looks like laser going in different directions::

    In [6]: a.rpost_(slice(0,6))     ## heading in some combination of X and Y direction
    Out[6]: 
    A()sliced
    A([[[ -18079.4443, -799699.4149,   -6604.9499,       0.0998],
            [ -17219.8321, -800985.8917,   -6604.9499,       7.8266],
            [ -17214.1845, -800994.1278,   -6604.9499,       7.8765],
            [ -16980.2796, -801344.2792,   -6604.9499,       9.98  ],
            [ -16970.161 , -801359.3395,   -6604.9499,      10.0702],
            [ -16826.3825, -801575.3603,   -6604.9499,      11.3474]],

       In [13]: b.rpost_(slice(0,6))   ## huh heading in -Z direction
    Out[13]: 
    A()sliced
    A([[[ -18079.4443, -799699.4149,   -6604.9499,       0.0998],
            [ -18079.4443, -799699.4149,   -8635.0278,      10.5229],
            [ -18079.4443, -799699.4149,   -8650.0881,      10.6008],
            [ -18079.4443, -799699.4149,   -8850.1073,      11.639 ],
            [ -18079.4443, -799699.4149,   -8895.0528,      11.8702],
            [ -18079.4443, -799699.4149,   -9092.013 ,      12.8928]],

::

    OKTest --load --vizg4 --cat laser
    OKG4Test --load --vizg4 --cat laser
    

Gensteps are same by construction, suspect CTorchSource not reading it::

    In [3]: a.gs
    Out[3]: 
    A(torch,1,laser)-
    A([[[      0.    ,       0.    ,       0.    ,       0.    ],
            [ -18079.4531, -799699.4375,   -6605.    ,       0.1   ],
            [      0.5556,      -0.8314,       0.    ,       1.    ],
            [      0.    ,       0.    ,       0.    ,     430.    ],
            [      0.    ,       1.    ,       0.    ,       1.    ],
            [      0.    ,       0.    ,       0.    ,       0.    ]]], dtype=float32)

    In [4]: b.gs
    Out[4]: 
    A(torch,-1,laser)-
    A([[[      0.    ,       0.    ,       0.    ,       0.    ],
            [ -18079.4531, -799699.4375,   -6605.    ,       0.1   ],
            [      0.5556,      -0.8314,       0.    ,       1.    ],
            [      0.    ,       0.    ,       0.    ,     430.    ],
            [      0.    ,       1.    ,       0.    ,       1.    ],
            [      0.    ,       0.    ,       0.    ,       0.    ]]], dtype=float32)



after fix aiming, restricted to top seq
--------------------------------------------

Restricting to top seq::

      A:seqhis_ana       noname 
              8ccccd        1.000           7673       [6 ] TO BT BT BT BT SA
                            7673         1.00 
       B:seqhis_ana       noname 
            8ccccccd        1.000           7500       [8 ] TO BT BT BT BT BT BT SA
                            7500         1.00 


       tlaser- ; tlaser-- --okg4 --compute --dbgseqhis 8ccccccd


::

    In [8]: a.rpost_(slice(0,9))[0]
    Out[8]: 
    A()sliced
    A([[     -18079.4443, -799699.4149,   -6604.9499,       0.0998],
           [ -17219.8321, -800985.8917,   -6604.9499,       7.8266],
           [ -17214.1845, -800994.1278,   -6604.9499,       7.8765],
           [ -16980.2796, -801344.2792,   -6604.9499,       9.98  ],
           [ -16970.161 , -801359.3395,   -6604.9499,      10.0702],
           [ -16826.3825, -801575.3603,   -6604.9499,      11.3474],
           [ -16520.    , -802110.    ,   -7125.    ,       0.    ],   << decompression dummies
           [ -16520.    , -802110.    ,   -7125.    ,       0.    ],
           [ -16520.    , -802110.    ,   -7125.    ,       0.    ]])


    In [14]: a.ox[:,0]    # final position photons, no compression
    Out[14]: 
    A()sliced
    A([[ -16826.3945, -801575.375 ,   -6605.    ,      11.3472],
           [ -16826.3945, -801575.375 ,   -6605.    ,      11.3472],
           [ -16826.3945, -801575.375 ,   -6605.    ,      11.3472],
           ..., 
           [ -16826.3945, -801575.375 ,   -6605.    ,      11.3472],
           [ -16826.3945, -801575.375 ,   -6605.    ,      11.3472],
           [ -16826.3945, -801575.375 ,   -6605.    ,      11.3472]], dtype=float32)



    In [9]: b.rpost_(slice(0,9))[0]
    Out[9]: 
    A()sliced
    A([[     -18079.4443, -799699.4149,   -6604.9499,       0.0998],
           [ -17218.1849, -800988.2449,   -6604.9499,       8.0587],
           [ -17212.7726, -800996.481 ,   -6604.9499,       8.1104],
           [ -16978.1618, -801347.3383,   -6604.9499,      10.2771],
           [ -16968.2785, -801362.3986,   -6604.9499,      10.3705],
           [ -16824.2646, -801577.7134,   -6604.9499,      11.6829],
           [ -16822.6174, -801580.3019,   -6604.9499,      11.6985],
           [ -16696.9582, -801768.0847,   -6604.9499,      12.842 ],
           [ -16520.    , -802110.    ,   -7125.    ,       0.    ]])

    In [15]: b.ox[:,0]
    Out[15]: 
    A()sliced
    A([[ -16697.0586, -801768.0625,   -6605.    ,      12.842 ],
           [ -16697.0586, -801768.0625,   -6605.    ,      12.842 ],
           [ -16697.0586, -801768.0625,   -6605.    ,      12.842 ],
           ..., 
           [ -16697.0586, -801768.0625,   -6605.    ,      12.842 ],
           [ -16697.0586, -801768.0625,   -6605.    ,      12.842 ],
           [ -16697.0586, -801768.0625,   -6605.    ,      12.842 ]], dtype=float32)

    In [17]: a.ox[:7500,0] - b.ox[:,0]
    Out[17]: 
    A()sliced
    A([[-129.3359,  192.6875,    0.    ,   -1.4948],
           [-129.3359,  192.6875,    0.    ,   -1.4948],
           [-129.3359,  192.6875,    0.    ,   -1.4948],
           ..., 
           [-129.3359,  192.6875,    0.    ,   -1.4948],
           [-129.3359,  192.6875,    0.    ,   -1.4948],
           [-129.3359,  192.6875,    0.    ,   -1.4948]], dtype=float32)


After fix CG4 skin surfaces
----------------------------

::

    In [1]: a.rpost_(slice(0,9))[0]
    Out[1]: 
    A()sliced
    A([[ -18079.4443, -799699.4149,   -6604.9499,       0.0998],
           [ -17219.8321, -800985.8917,   -6604.9499,       7.8266],
           [ -17214.1845, -800994.1278,   -6604.9499,       7.8765],
           [ -16980.2796, -801344.2792,   -6604.9499,       9.98  ],
           [ -16970.161 , -801359.3395,   -6604.9499,      10.0702],
           [ -16826.3825, -801575.3603,   -6604.9499,      11.3474],
           [ -16520.    , -802110.    ,   -7125.    ,       0.    ],
           [ -16520.    , -802110.    ,   -7125.    ,       0.    ],
           [ -16520.    , -802110.    ,   -7125.    ,       0.    ]])

    In [2]: b.rpost_(slice(0,9))[0]
    Out[2]: 
    A()sliced
    A([[ -18079.4443, -799699.4149,   -6604.9499,       0.0998],
           [ -17218.1849, -800988.2449,   -6604.9499,       8.0587],
           [ -17212.7726, -800996.481 ,   -6604.9499,       8.1104],
           [ -16978.1618, -801347.3383,   -6604.9499,      10.2771],
           [ -16968.2785, -801362.3986,   -6604.9499,      10.3705],
           [ -16824.2646, -801577.7134,   -6604.9499,      11.6829],
           [ -16520.    , -802110.    ,   -7125.    ,       0.    ],
           [ -16520.    , -802110.    ,   -7125.    ,       0.    ],
           [ -16520.    , -802110.    ,   -7125.    ,       0.    ]])


    In [4]: a.ox[:,0]
    Out[4]: 
    A()sliced
    A([[ -16826.3945, -801575.375 ,   -6605.    ,      11.3472],
           [ -16826.3945, -801575.375 ,   -6605.    ,      11.3472],
           [ -16826.3945, -801575.375 ,   -6605.    ,      11.3472],
           ..., 
           [ -16826.3945, -801575.375 ,   -6605.    ,      11.3472],
           [ -16826.3945, -801575.375 ,   -6605.    ,      11.3472],
           [ -16826.3945, -801575.375 ,   -6605.    ,      11.3472]], dtype=float32)

    In [5]: b.ox[:,0]
    Out[5]: 
    A()sliced
    A([[ -16824.2129, -801577.8125,   -6605.    ,      11.6829],
           [ -16824.2129, -801577.8125,   -6605.    ,      11.6829],
           [ -16824.2129, -801577.8125,   -6605.    ,      11.6829],
           ..., 
           [ -16824.2129, -801577.8125,   -6605.    ,      11.6829],
           [ -16824.2129, -801577.8125,   -6605.    ,      11.6829],
           [ -16824.2129, -801577.8125,   -6605.    ,      11.6829]], dtype=float32)

    In [8]: a.ox[:,0] - b.ox[:763501,0]    ## few mm presumably tesselation effect
    Out[8]: 
    A()sliced
    A([[-2.1816,  2.4375,  0.    , -0.3357],
           [-2.1816,  2.4375,  0.    , -0.3357],
           [-2.1816,  2.4375,  0.    , -0.3357],
           ..., 
           [-2.1816,  2.4375,  0.    , -0.3357],
           [-2.1816,  2.4375,  0.    , -0.3357],
           [-2.1816,  2.4375,  0.    , -0.3357]], dtype=float32)


Time shift is smaller than I recall the groupvel issue being::

    In [30]: 0.33/11.
    Out[30]: 0.030




Termination boundaries
------------------------

::

    134 #define FLAGS(p, s, prd) \
    135 { \
    136     p.flags.i.x = prd.boundary ;  \
    137     p.flags.u.y = s.identity.w ;  \
    138     p.flags.u.w |= s.flag ; \
    139 } \


::

    ( 37) om:               MineralOil os:             RSOilSurface is:                          im:                  Acrylic

    (signed boundaries are 1-based, as 0 means miss : so subtract 1 for the 0-based op --bnd)

    GSurLib::pushBorderSurfaces does not list it, so it should be isur/osur duped in order to be relevant in both directions ???

    WHAT IS THE CG4 8? just the slot 

    HUH : -ve boundary corresponds to inward going photons  ???


    In [21]: a.ox[:,3].view(np.int32)
    Out[21]: 
    A()sliced
    A([[     -38,        0, 67305984,     6272],
           [     -38,        0, 67305984,     6272],
           [     -38,        0, 67305984,     6272],
           ..., 
           [     -38,        0, 67305984,     6272],
           [     -38,        0, 67305984,     6272],
           [     -38,        0, 67305984,     6272]], dtype=int32)

    In [22]: b.ox[:,3].view(np.int32)
    Out[22]: 
    A()sliced
    A([[       8,        0, 67305984,     6272],
           [       8,        0, 67305984,     6272],
           [       8,        0, 67305984,     6272],
           ..., 
           [       8,        0, 67305984,     6272],
           [       8,        0, 67305984,     6272],
           [       8,        0, 67305984,     6272]], dtype=int32)


::

    586 void CRecorder::RecordPhoton(const G4Step* step)
    587 {
    588     // gets called at last step (eg absorption) or when truncated
    ...
    609     target->setUInt(target_record_id, 3, 0, 0, m_slot );
    610     target->setUInt(target_record_id, 3, 0, 1, 0u );
    611     target->setUInt(target_record_id, 3, 0, 2, m_c4.u );
    612     target->setUInt(target_record_id, 3, 0, 3, m_mskhis );
    613 


z is c4::

    309     // initial quadrant 
    310     uifchar4 c4 ;
    311     c4.uchar_.x =
    312                   (  p.position.x > 0.f ? QX : 0u )
    313                    |
    314                   (  p.position.y > 0.f ? QY : 0u )
    315                    |
    316                   (  p.position.z > 0.f ? QZ : 0u )
    317                   ;
    318 
    319     c4.uchar_.y = 2u ;   // 3-bytes up for grabs
    320     c4.uchar_.z = 3u ;
    321     c4.uchar_.w = 4u ;
    322 
    323     p.flags.f.z = c4.f ;


    In [28]: a.c4
    Out[28]: 
    rec.array([(0, 2, 3, 4), (0, 2, 3, 4), (0, 2, 3, 4), ..., (0, 2, 3, 4), (0, 2, 3, 4), (0, 2, 3, 4)], 
          dtype=[('x', 'u1'), ('y', 'u1'), ('z', 'u1'), ('w', 'u1')])

    In [29]: b.c4
    Out[29]: 
    rec.array([(0, 2, 3, 4), (0, 2, 3, 4), (0, 2, 3, 4), ..., (0, 2, 3, 4), (0, 2, 3, 4), (0, 2, 3, 4)], 
          dtype=[('x', 'u1'), ('y', 'u1'), ('z', 'u1'), ('w', 'u1')])




* old groupvel timing issue apparent, fixing that will help with this
* looks like CG4 is taking a few steps more prior to SA



probable cause CG4 logical skin surfaces lacking lv
-----------------------------------------------------

::

    2016-10-02 16:51:37.006 INFO  [1411044] [CBorderSurfaceTable::init@21] CBorderSurfaceTable::init nsurf 11
        0               NearDeadLinerSurface pv1 /dd/Geometry/Sites/lvNearHallBot#pvNearPoolDead #0 pv2 /dd/Geometry/Pool/lvNearPoolDead#pvNearPoolLiner #0
        1                NearOWSLinerSurface pv1 /dd/Geometry/Pool/lvNearPoolLiner#pvNearPoolOWS #0 pv2 /dd/Geometry/Pool/lvNearPoolDead#pvNearPoolLiner #0
        2              NearIWSCurtainSurface pv1 /dd/Geometry/Pool/lvNearPoolCurtain#pvNearPoolIWS #0 pv2 /dd/Geometry/Pool/lvNearPoolOWS#pvNearPoolCurtain #0
        3               SSTWaterSurfaceNear1 pv1 /dd/Geometry/Pool/lvNearPoolIWS#pvNearADE1 #0 pv2 /dd/Geometry/AD/lvADE#pvSST #0
        4                      SSTOilSurface pv1 /dd/Geometry/AD/lvSST#pvOIL #0 pv2 /dd/Geometry/AD/lvADE#pvSST #0
        5                      SSTOilSurface pv1 /dd/Geometry/AD/lvSST#pvOIL #0 pv2 /dd/Geometry/AD/lvADE#pvSST #0
        6                   ESRAirSurfaceTop pv1 /dd/Geometry/AdDetails/lvTopReflector#pvTopRefGap #0 pv2 /dd/Geometry/AdDetails/lvTopRefGap#pvTopESR #0
        7                   ESRAirSurfaceTop pv1 /dd/Geometry/AdDetails/lvTopReflector#pvTopRefGap #0 pv2 /dd/Geometry/AdDetails/lvTopRefGap#pvTopESR #0
        8                   ESRAirSurfaceBot pv1 /dd/Geometry/AdDetails/lvBotReflector#pvBotRefGap #0 pv2 /dd/Geometry/AdDetails/lvBotRefGap#pvBotESR #0
        9                   ESRAirSurfaceBot pv1 /dd/Geometry/AdDetails/lvBotReflector#pvBotRefGap #0 pv2 /dd/Geometry/AdDetails/lvBotRefGap#pvBotESR #0
       10               SSTWaterSurfaceNear2 pv1 /dd/Geometry/Pool/lvNearPoolIWS#pvNearADE2 #0 pv2 /dd/Geometry/AD/lvADE#pvSST #0

    2016-10-02 16:51:37.006 INFO  [1411044] [CBorderSurfaceTable::dump@47] CGeometryTest CBorderSurfaceTable
    2016-10-02 16:51:37.006 INFO  [1411044] [CSkinSurfaceTable::init@22] CSkinSurfaceTable::init nsurf 36
        0               NearPoolCoverSurface lv NULL
        1      lvPmtHemiCathodeSensorSurface lv NULL
        2    lvHeadonPmtCathodeSensorSurface lv NULL
        3                       RSOilSurface lv NULL
        4                 AdCableTraySurface lv NULL
        5                PmtMtTopRingSurface lv NULL
        6               PmtMtBaseRingSurface lv NULL
        7                   PmtMtRib1Surface lv NULL
        8                   PmtMtRib2Surface lv NULL
        9                   PmtMtRib3Surface lv NULL
       10                 LegInIWSTubSurface lv NULL
       11                  TablePanelSurface lv NULL
       12                 SupportRib1Surface lv NULL
       13                 SupportRib5Surface lv NULL
       14                   SlopeRib1Surface lv NULL
       15                   SlopeRib5Surface lv NULL
       16            ADVertiCableTraySurface lv NULL
       17           ShortParCableTraySurface lv NULL
       18              NearInnInPiperSurface lv NULL
       19             NearInnOutPiperSurface lv NULL
       20                 LegInOWSTubSurface lv NULL
       21                UnistrutRib6Surface lv NULL
       22                UnistrutRib7Surface lv NULL
       23                UnistrutRib3Surface lv NULL
       24                UnistrutRib5Surface lv NULL
       25                UnistrutRib4Surface lv NULL
       26                UnistrutRib1Surface lv NULL
       27                UnistrutRib2Surface lv NULL
       28                UnistrutRib8Surface lv NULL
       29                UnistrutRib9Surface lv NULL
       30           TopShortCableTraySurface lv NULL
       31          TopCornerCableTraySurface lv NULL
       32              VertiCableTraySurface lv NULL
       33              NearOutInPiperSurface lv NULL
       34             NearOutOutPiperSurface lv NULL
       35                LegInDeadTubSurface lv NULL


After fix CG4 logical skin surfaces 
--------------------------------------

Steps looking rather similar now, next issue more  BULK_ABSORB AB in CG4 than OK.

::

       A:seqhis_ana      1:laser 
              8ccccd        0.764         763501       [6 ] TO BT BT BT BT SA
                  4d        0.056          55825       [2 ] TO AB
          cccc9ccccd        0.025          25263       [10] TO BT BT BT BT DR BT BT BT BT
             8cccc6d        0.020          19707       [7 ] TO SC BT BT BT BT SA
                4ccd        0.013          12576       [4 ] TO BT BT AB
             8cccc5d        0.011          11183       [7 ] TO RE BT BT BT BT SA
              4ccccd        0.009           8554       [6 ] TO BT BT BT BT AB
                 45d        0.008           7531       [3 ] TO RE AB
            8cccc55d        0.005           5362       [8 ] TO RE RE BT BT BT BT SA
             8cc6ccd        0.004           4109       [7 ] TO BT BT SC BT BT SA
                455d        0.004           3588       [4 ] TO RE RE AB
             86ccccd        0.003           2836       [7 ] TO BT BT BT BT SC SA
          cccccc6ccd        0.003           2674       [10] TO BT BT SC BT BT BT BT BT BT
           8cccc555d        0.003           2524       [9 ] TO RE RE RE BT BT BT BT SA
             8cc5ccd        0.002           2359       [7 ] TO BT BT RE BT BT SA
          cacccccc6d        0.002           2210       [10] TO SC BT BT BT BT BT BT SR BT
                 46d        0.002           2118       [3 ] TO SC AB
          cccc6ccccd        0.002           2060       [10] TO BT BT BT BT SC BT BT BT BT
               4cccd        0.002           1940       [5 ] TO BT BT BT AB
             89ccccd        0.002           1880       [7 ] TO BT BT BT BT DR SA
                         1000000         1.00 
       B:seqhis_ana     -1:laser 
              8ccccd        0.813         813472       [6 ] TO BT BT BT BT SA
                  4d        0.072          71523       [2 ] TO AB
          cccc9ccccd        0.027          27170       [10] TO BT BT BT BT DR BT BT BT BT
                4ccd        0.017          17386       [4 ] TO BT BT AB
             8cccc6d        0.015          15107       [7 ] TO SC BT BT BT BT SA
              4ccccd        0.009           8842       [6 ] TO BT BT BT BT AB
          cacccccc6d        0.004           3577       [10] TO SC BT BT BT BT BT BT SR BT
             8cc6ccd        0.003           3466       [7 ] TO BT BT SC BT BT SA
                 46d        0.003           2515       [3 ] TO SC AB
             86ccccd        0.002           2476       [7 ] TO BT BT BT BT SC SA
           cac0ccc6d        0.002           2356       [9 ] TO SC BT BT BT ?0? BT SR BT
          cccccc6ccd        0.002           2157       [10] TO BT BT SC BT BT BT BT BT BT
             89ccccd        0.002           2127       [7 ] TO BT BT BT BT DR SA
               4cccd        0.002           1977       [5 ] TO BT BT BT AB
          cccc6ccccd        0.002           1949       [10] TO BT BT BT BT SC BT BT BT BT
            8ccccc6d        0.002           1515       [8 ] TO SC BT BT BT BT BT SA
          ccbccccc6d        0.001           1429       [10] TO SC BT BT BT BT BT BR BT BT
           4cc9ccccd        0.001           1215       [9 ] TO BT BT BT BT DR BT BT AB
                 4cd        0.001           1077       [3 ] TO BT AB
               4cc6d        0.001            802       [5 ] TO SC BT BT AB
                         1000000         1.00 






full seq following fixed aim
--------------------------------

::

      A:seqhis_ana      1:laser 
              8ccccd        0.767           7673       [6 ] TO BT BT BT BT SA
                  4d        0.055            553       [2 ] TO AB
          cccc9ccccd        0.024            242       [10] TO BT BT BT BT DR BT BT BT BT
             8cccc6d        0.019            188       [7 ] TO SC BT BT BT BT SA
                4ccd        0.012            122       [4 ] TO BT BT AB
             8cccc5d        0.012            121       [7 ] TO RE BT BT BT BT SA
                 45d        0.006             65       [3 ] TO RE AB
              4ccccd        0.006             63       [6 ] TO BT BT BT BT AB
            8cccc55d        0.005             52       [8 ] TO RE RE BT BT BT BT SA
             8cc6ccd        0.004             39       [7 ] TO BT BT SC BT BT SA
                455d        0.003             34       [4 ] TO RE RE AB
          cccccc6ccd        0.003             34       [10] TO BT BT SC BT BT BT BT BT BT
             8cc5ccd        0.003             27       [7 ] TO BT BT RE BT BT SA
             86ccccd        0.003             27       [7 ] TO BT BT BT BT SC SA
           8cccc555d        0.003             26       [9 ] TO RE RE RE BT BT BT BT SA
               4cccd        0.003             25       [5 ] TO BT BT BT AB
          cacccccc5d        0.002             22       [10] TO RE BT BT BT BT BT BT SR BT
                 46d        0.002             21       [3 ] TO SC AB
          cccc6ccccd        0.002             20       [10] TO BT BT BT BT SC BT BT BT BT
            4ccccc5d        0.002             19       [8 ] TO RE BT BT BT BT BT AB
                           10000         1.00 
       B:seqhis_ana     -1:laser 
            8ccccccd        0.750           7500       [8 ] TO BT BT BT BT BT BT SA
                  4d        0.074            741       [2 ] TO AB
          cc9ccccccd        0.043            433       [10] TO BT BT BT BT BT BT DR BT BT
          cb9ccccccd        0.027            271       [10] TO BT BT BT BT BT BT DR BR BT
                4ccd        0.018            175       [4 ] TO BT BT AB
           8cccccc6d        0.014            138       [9 ] TO SC BT BT BT BT BT BT SA
              4ccccd        0.009             88       [6 ] TO BT BT BT BT AB
          4c9ccccccd        0.008             78       [10] TO BT BT BT BT BT BT DR BT AB
            4ccccccd        0.007             70       [8 ] TO BT BT BT BT BT BT AB
          cacccccc6d        0.004             35       [10] TO SC BT BT BT BT BT BT SR BT
           8cc6ccccd        0.003             25       [9 ] TO BT BT BT BT SC BT BT SA
          cccc6ccccd        0.002             22       [10] TO BT BT BT BT SC BT BT BT BT
           89ccccccd        0.002             22       [9 ] TO BT BT BT BT BT BT DR SA
          ccbccccc6d        0.002             22       [10] TO SC BT BT BT BT BT BR BT BT
               4cccd        0.002             21       [5 ] TO BT BT BT AB
           8cccc6ccd        0.002             21       [9 ] TO BT BT SC BT BT BT BT SA
           cac0ccc6d        0.002             21       [9 ] TO SC BT BT BT ?0? BT SR BT
                 46d        0.002             18       [3 ] TO SC AB
          cccccc6ccd        0.002             17       [10] TO BT BT SC BT BT BT BT BT BT
          bc9ccccccd        0.002             16       [10] TO BT BT BT BT BT BT DR BT BR
                           10000         1.00 








initial ana 
-------------

::

    ipython -i $(which tokg4.py) -- --det laser

    /Users/blyth/opticks/ana/tokg4.py --det laser
    writing opticks environment to /tmp/blyth/opticks/opticks_env.bash 
    [2016-10-02 11:10:22,331] p22488 {/Users/blyth/opticks/ana/tokg4.py:25} INFO - tag 1 src torch det laser c2max 2.0  
    [2016-10-02 11:10:22,397] p22488 {/Users/blyth/opticks/ana/tokg4.py:36} INFO -  a : laser/torch/  1 :  20161002-1106 /tmp/blyth/opticks/evt/laser/torch/1/fdom.npy 
    [2016-10-02 11:10:22,397] p22488 {/Users/blyth/opticks/ana/tokg4.py:37} INFO -  b : laser/torch/ -1 :  20161002-1106 /tmp/blyth/opticks/evt/laser/torch/-1/fdom.npy 
    A Evt(  1,"torch","laser","laser/torch/  1 : ", seqs="[]") 20161002-1106 /tmp/blyth/opticks/evt/laser/torch/1
    B Evt( -1,"torch","laser","laser/torch/ -1 : ", seqs="[]") 20161002-1106 /tmp/blyth/opticks/evt/laser/torch/-1
           A:seqhis_ana      1:laser 
                  8ccccd        0.767           7673       [6 ] TO BT BT BT BT SA
                      4d        0.055            553       [2 ] TO AB
              cccc9ccccd        0.024            242       [10] TO BT BT BT BT DR BT BT BT BT
                 8cccc6d        0.019            188       [7 ] TO SC BT BT BT BT SA
                    4ccd        0.012            122       [4 ] TO BT BT AB
                 8cccc5d        0.012            121       [7 ] TO RE BT BT BT BT SA
                     45d        0.006             65       [3 ] TO RE AB
                  4ccccd        0.006             63       [6 ] TO BT BT BT BT AB
                8cccc55d        0.005             52       [8 ] TO RE RE BT BT BT BT SA
                 8cc6ccd        0.004             39       [7 ] TO BT BT SC BT BT SA
                    455d        0.003             34       [4 ] TO RE RE AB
              cccccc6ccd        0.003             34       [10] TO BT BT SC BT BT BT BT BT BT
                 8cc5ccd        0.003             27       [7 ] TO BT BT RE BT BT SA
                 86ccccd        0.003             27       [7 ] TO BT BT BT BT SC SA
               8cccc555d        0.003             26       [9 ] TO RE RE RE BT BT BT BT SA
                   4cccd        0.003             25       [5 ] TO BT BT BT AB
              cacccccc5d        0.002             22       [10] TO RE BT BT BT BT BT BT SR BT
                     46d        0.002             21       [3 ] TO SC AB
              cccc6ccccd        0.002             20       [10] TO BT BT BT BT SC BT BT BT BT
                4ccccc5d        0.002             19       [8 ] TO RE BT BT BT BT BT AB
                               10000         1.00 
           B:seqhis_ana     -1:laser 
                8c0cc0cd        0.703           7030       [8 ] TO BT ?0? BT BT ?0? BT SA
                      4d        0.090            899       [2 ] TO AB
              4c9c0cc0cd        0.030            301       [10] TO BT ?0? BT BT ?0? BT DR BT AB
              cb9c0cc0cd        0.029            285       [10] TO BT ?0? BT BT ?0? BT DR BR BT
                  4cc0cd        0.022            217       [6 ] TO BT ?0? BT BT AB
                    40cd        0.020            201       [4 ] TO BT ?0? AB
               8cccccc6d        0.015            152       [9 ] TO SC BT BT BT BT BT BT SA
                4c0cc0cd        0.015            145       [8 ] TO BT ?0? BT BT ?0? BT AB
              bb9c0cc0cd        0.011            105       [10] TO BT ?0? BT BT ?0? BT DR BR BR
               cac0ccc6d        0.005             52       [9 ] TO SC BT BT BT ?0? BT SR BT
                     46d        0.005             49       [3 ] TO SC AB
              cc0b0ccc6d        0.004             44       [10] TO SC BT BT BT ?0? BR ?0? BT BT
              cc9c0cc0cd        0.004             43       [10] TO BT ?0? BT BT ?0? BT DR BT BT
              cacccccc6d        0.004             40       [10] TO SC BT BT BT BT BT BT SR BT
              4c6c0cc0cd        0.004             39       [10] TO BT ?0? BT BT ?0? BT SC BT AB
              cccc6cc0cd        0.002             21       [10] TO BT ?0? BT BT SC BT BT BT BT
                     4cd        0.002             20       [3 ] TO BT AB
              bc6c0cc0cd        0.002             17       [10] TO BT ?0? BT BT ?0? BT SC BT BR
              c9cccccc6d        0.002             17       [10] TO SC BT BT BT BT BT BT DR BT
              cccccccc6d        0.002             17       [10] TO SC BT BT BT BT BT BT BT BT
                               10000         1.00 

           A:seqmat_ana      1:laser 
                  443231        0.774           7736       [6 ] Gd Ac LS Ac MO MO
                      11        0.055            553       [2 ] Gd Gd
                 4432311        0.031            314       [7 ] Gd Gd Ac LS Ac MO MO
              3323443231        0.026            265       [10] Gd Ac LS Ac MO MO Ac LS Ac Ac
                    2231        0.012            122       [4 ] Gd Ac LS LS
                     111        0.009             86       [3 ] Gd Gd Gd
                44323111        0.007             72       [8 ] Gd Gd Gd Ac LS Ac MO MO
                 4432231        0.007             71       [7 ] Gd Ac LS LS Ac MO MO
                 4443231        0.005             46       [7 ] Gd Ac LS Ac MO MO MO
              fff3432311        0.004             39       [10] Gd Gd Ac LS Ac MO Ac Ai Ai Ai
              3323132231        0.004             39       [10] Gd Ac LS LS Ac Gd Ac LS Ac Ac
                    1111        0.004             35       [4 ] Gd Gd Gd Gd
              4433432311        0.003             33       [10] Gd Gd Ac LS Ac MO Ac Ac MO MO
               443231111        0.003             31       [9 ] Gd Gd Gd Gd Ac LS Ac MO MO
                aa332311        0.003             26       [8 ] Gd Gd Ac LS Ac Ac ES ES
                   33231        0.003             25       [5 ] Gd Ac LS Ac Ac
                   11111        0.002             20       [5 ] Gd Gd Gd Gd Gd
                dd432311        0.002             20       [8 ] Gd Gd Ac LS Ac MO Vm Vm
                44322231        0.002             17       [8 ] Gd Ac LS LS LS Ac MO MO
                     331        0.001             14       [3 ] Gd Ac Ac
                               10000         1.00 
           B:seqmat_ana     -1:laser 
                44332331        0.718           7175       [8 ] Gd Ac Ac LS Ac Ac MO MO
                      11        0.090            899       [2 ] Gd Gd
              ff44332331        0.034            340       [10] Gd Ac Ac LS Ac Ac MO MO Ai Ai
              3444332331        0.026            264       [10] Gd Ac Ac LS Ac Ac MO MO MO Ac
                  332331        0.022            217       [6 ] Gd Ac Ac LS Ac Ac
                    3331        0.020            201       [4 ] Gd Ac Ac Ac
               443432311        0.015            154       [9 ] Gd Gd Ac LS Ac MO Ac MO MO
              4444332331        0.013            134       [10] Gd Ac Ac LS Ac Ac MO MO MO MO
              33ff332311        0.005             52       [10] Gd Gd Ac LS Ac Ac Ai Ai Ac Ac
              f344332331        0.005             51       [10] Gd Ac Ac LS Ac Ac MO MO Ac Ai
                     111        0.005             49       [3 ] Gd Gd Gd
              3233332311        0.004             43       [10] Gd Gd Ac LS Ac Ac Ac Ac LS Ac
              3ff3432311        0.004             40       [10] Gd Gd Ac LS Ac MO Ac Ai Ai Ac
              3344332331        0.003             29       [10] Gd Ac Ac LS Ac Ac MO MO Ac Ac
              f444332331        0.003             29       [10] Gd Ac Ac LS Ac Ac MO MO MO Ai
                     331        0.002             20       [3 ] Gd Ac Ac
               444332331        0.002             19       [9 ] Gd Ac Ac LS Ac Ac MO MO MO
              3443432311        0.002             17       [10] Gd Gd Ac LS Ac MO Ac MO MO Ac
              3232332331        0.002             16       [10] Gd Ac Ac LS Ac Ac LS Ac LS Ac
              3433432311        0.002             15       [10] Gd Gd Ac LS Ac MO Ac Ac MO Ac
                               10000         1.00 


    





