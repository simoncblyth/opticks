CRecorder_duplicated_point_BTBT 
===================================

Summary
----------

* add dx.npy deluxe double step g4only array : to look into the microStep
* fix inadventent float narrowing on way into dx buffer
* add bn.npy boundary buffer 
* thinking about how to removing the 1e-3 onion, without conclusion


tds3gun after suppress microStep : allows comparison to start
-----------------------------------------------------------------

::

    In [3]: ab.b.seqhis_ana.table[:15]                                                                                                                                                                                                      
    Out[3]: 
    all_seqhis_ana
    .                     cfo:-  -1:g4live:tds3gun 
    .                              11278         1.00 
    0000               42        0.148        1665        [2 ] SI AB
    0001            7ccc2        0.109        1230        [5 ] SI BT BT BT SD
    0002            8ccc2        0.060         674        [5 ] SI BT BT BT SA
    0003           7ccc62        0.049         552        [6 ] SI SC BT BT BT SD
    0004              452        0.047         534        [3 ] SI RE AB
    0005             8cc2        0.041         464        [4 ] SI BT BT SA
    0006           7ccc52        0.035         397        [6 ] SI RE BT BT BT SD
    0007              462        0.033         367        [3 ] SI SC AB
    0008           8ccc62        0.024         267        [6 ] SI SC BT BT BT SA
    0009          7ccc662        0.019         213        [7 ] SI SC SC BT BT BT SD
    0010           8ccc52        0.017         188        [6 ] SI RE BT BT BT SA
    0011            8cc62        0.016         186        [5 ] SI SC BT BT SA
    0012          7ccc652        0.014         159        [7 ] SI RE SC BT BT BT SD
    0013           7cccc2        0.014         154        [6 ] SI BT BT BT BT SD
    0014               41        0.013         144        [2 ] CK AB
    .                              11278         1.00 

    In [4]: ab.a.seqhis_ana.table[:15]                                                                                                                                                                                                      
    Out[4]: 
    all_seqhis_ana
    .                     cfo:-  1:g4live:tds3gun 
    .                              11278         1.00 
    0000               42        0.147        1653        [2 ] SI AB
    0001            7ccc2        0.115        1292        [5 ] SI BT BT BT SD
    0002            8ccc2        0.052         590        [5 ] SI BT BT BT SA
    0003           7ccc62        0.052         581        [6 ] SI SC BT BT BT SD
    0004              452        0.037         422        [3 ] SI RE AB
    0005              462        0.035         392        [3 ] SI SC AB
    0006           7ccc52        0.034         380        [6 ] SI RE BT BT BT SD
    0007             4cc2        0.025         278        [4 ] SI BT BT AB
    0008           8ccc62        0.022         251        [6 ] SI SC BT BT BT SA
    0009          7ccc662        0.019         219        [7 ] SI SC SC BT BT BT SD
    0010            4cc62        0.017         197        [5 ] SI SC BT BT AB
    0011          7ccc652        0.014         157        [7 ] SI RE SC BT BT BT SD
    0012           8ccc52        0.014         154        [6 ] SI RE BT BT BT SA
    0013               41        0.013         142        [2 ] CK AB
    0014             4662        0.012         137        [4 ] SI SC SC AB
    .                              11278         1.00 


issue
-------

* Geant4/CRecorder giving an extra BT cf Opticks from a very close set of boundaries
* such a different history needs to be fixed for the Opticks matching machinery to work 

::

    als[:10]
    TO BT BT AB
    TO BT BT BT SD
    TO BT BT BT SA
    TO BT BT AB
    TO BT BT BT SD
    TO BT BT BT SD
    TO BT BT BT SA
    TO AB

    bls[:10]
    TO BT BT BT *BT* SA
    TO SC BT BT *BT* SA
    TO BT BT BT *BT* SD
    TO AB
    TO SC SC BT BT BT *BT* SD
    TO BT BT BT *BT* SA
    TO BT BT AB
    TO SC BT BT BR SA


tds3ip using --dbgseqhis


::

    tds3ip () 
    { 
        local name="CubeCorners";
        local path="$HOME/.opticks/InputPhotons/${name}.npy";
        export OPTICKS_EVENT_PFX=tds3ip;
        export INPUT_PHOTON_PATH=$path;
        tds3 --dbgseqhis 0x7ccccd
    }



    2021-06-11 22:06:36.140 INFO  [446353] [CDebug::dump@188] CDebug::postTrack
    2021-06-11 22:06:36.140 INFO  [446353] [CRec::dump@194] CDebug::dump record_id 5  origin[ 0.577-0.5770.577]   Ori[ 0.577-0.5770.577] 
    2021-06-11 22:06:36.140 INFO  [446353] [CRec::dump@200]  nstp 5
    (0 )  TO/BT     FrT                       PRE_SAVE POST_SAVE STEP_START 
    [   0](Stp ;opticalphoton stepNum    5(tk ;opticalphoton tid 6 pid 0 nm    440 mm  ori[    0.577  -0.577   0.577]  pos[ 11143.476-11143.09611143.087]  )
      pre                   pTarget              LS          noProc           Undefined pos[      0.000     0.000     0.000]  dir[    0.577  -0.577   0.577]  pol[   -0.707   0.000   0.707]  ns  0.600 nm 440.000 mm/ns 195.234
     post                  pAcrylic         Acrylic  Transportation        GeomBoundary pos[  10218.522-10218.522 10218.522]  dir[    0.577  -0.577   0.577]  pol[   -0.707   0.000   0.707]  ns 91.255 nm 440.000 mm/ns 193.809
     )
    (1 )  BT/BT     FrT                                           POST_SAVE 
    [   1](Stp ;opticalphoton stepNum    5(tk ;opticalphoton tid 6 pid 0 nm    440 mm  ori[    0.577  -0.577   0.577]  pos[ 11143.476-11143.09611143.087]  )
      pre                  pAcrylic         Acrylic  Transportation        GeomBoundary pos[  10218.522-10218.522 10218.522]  dir[    0.577  -0.577   0.577]  pol[   -0.707   0.000   0.707]  ns 91.255 nm 440.000 mm/ns 193.809
     post               pInnerWater           Water  Transportation        GeomBoundary pos[  10287.804-10287.804 10287.804]  dir[    0.577  -0.577   0.577]  pol[   -0.707   0.000   0.707]  ns 91.875 nm 440.000 mm/ns 216.910
     )
    (2 )  BT/BT     FrT                                           POST_SAVE 
    [   2](Stp ;opticalphoton stepNum    5(tk ;opticalphoton tid 6 pid 0 nm    440 mm  ori[    0.577  -0.577   0.577]  pos[ 11143.476-11143.09611143.087]  )
      pre               pInnerWater           Water  Transportation        GeomBoundary pos[  10287.804-10287.804 10287.804]  dir[    0.577  -0.577   0.577]  pol[   -0.707   0.000   0.707]  ns 91.875 nm 440.000 mm/ns 216.910
     post         pLPMT_NNVT_MCPPMT           Pyrex  Transportation        GeomBoundary pos[  11139.935-11139.935 11139.935]  dir[    0.621  -0.555   0.553]  pol[    0.695   0.717  -0.061]  ns 98.679 nm 440.000 mm/ns 196.979
     )
    (3 )  BT/BT     SAM                                           POST_SAVE 
    [   3](Stp ;opticalphoton stepNum    5(tk ;opticalphoton tid 6 pid 0 nm    440 mm  ori[    0.577  -0.577   0.577]  pos[ 11143.476-11143.09611143.087]  )
      pre         pLPMT_NNVT_MCPPMT           Pyrex  Transportation        GeomBoundary pos[  11139.935-11139.935 11139.935]  dir[    0.621  -0.555   0.553]  pol[    0.695   0.717  -0.061]  ns 98.679 nm 440.000 mm/ns 196.979
     post      NNVTMCPPMT_body_phys           Pyrex  Transportation        GeomBoundary pos[  11139.936-11139.936 11139.936]  dir[    0.621  -0.555   0.553]  pol[    0.695   0.717  -0.061]  ns 98.679 nm 440.000 mm/ns 196.979
     )
    (4 )  BT/SD     Det              POST_SAVE POST_DONE LAST_POST SURF_ABS 
    [   4](Stp ;opticalphoton stepNum    5(tk ;opticalphoton tid 6 pid 0 nm    440 mm  ori[    0.577  -0.577   0.577]  pos[ 11143.476-11143.09611143.087]  )
      pre      NNVTMCPPMT_body_phys           Pyrex  Transportation        GeomBoundary pos[  11139.936-11139.936 11139.936]  dir[    0.621  -0.555   0.553]  pol[    0.695   0.717  -0.061]  ns 98.679 nm 440.000 mm/ns 196.979
     post    NNVTMCPPMT_inner1_phys          Vacuum  Transportation        GeomBoundary pos[  11143.476-11143.096 11143.087]  dir[    0.621  -0.555   0.553]  pol[    0.695   0.717  -0.061]  ns 98.708 nm 440.000 mm/ns 196.979
     )
    2021-06-11 22:06:36.140 INFO  [446353] [CRec::dump@204]  npoi 0
    2021-06-11 22:06:36.140 INFO  [446353] [CDebug::dump_brief@204] CRecorder::dump_brief m_ctx._record_id        5 m_photon._badflag     0 --dbgseqhis  sas: POST_SAVE POST_DONE LAST_POST SURF_ABS 
    2021-06-11 22:06:36.140 INFO  [446353] [CDebug::dump_brief@213]  seqhis           7ccccd    TO BT BT BT BT SD                               
    2021-06-11 22:06:36.140 INFO  [446353] [CDebug::dump_brief@218]  mskhis             1840    SD|BT|TO
    2021-06-11 22:06:36.140 INFO  [446353] [CDebug::dump_brief@223]  seqmat           deefb1    LS Acrylic Water Pyrex Pyrex Vacuum - - - - - - - - - - 
    2021-06-11 22:06:36.140 INFO  [446353] [CDebug::dump_sequence@231] CDebug::dump_sequence
    2021-06-11 22:06:36.140 INFO  [446353] [CDebug::dump_points@257] CDeug::dump_points



Hmm why not exactly matching the above ? Must be going thru a float ? YEP move to `setQuad_`::

    In [3]: b.dx[5]                                                                                                                                                                                          
    Out[3]: 
    A([[[     0.5774,     -0.5774,      0.5774,      0.6   ],
        [    -0.7071,      0.    ,      0.7071,    440.    ]],

       [[ 10219.0996, -10219.0996,  10219.0996,     91.2555],
        [    -0.7071,      0.    ,      0.7071,    440.    ]],

       [[ 10288.3818, -10288.3818,  10288.3818,     91.8747],
        [    -0.7071,      0.    ,      0.7071,    440.    ]],

       [[ 11140.5127, -11140.5127,  11140.5127,     98.679 ],
        [     0.6946,      0.7168,     -0.0615,    440.    ]],

       [[ 11140.5127, -11140.5127,  11140.5127,     98.679 ],
        [     0.6946,      0.7168,     -0.0615,    440.    ]],

       [[ 11144.0537, -11143.6738,  11143.6641,     98.708 ],
        [     0.6946,      0.7168,     -0.0615,    440.    ]],

       [[     0.    ,      0.    ,      0.    ,      0.    ],
        [     0.    ,      0.    ,      0.    ,      0.    ]],




Notice for m_stp index 3 

* pre and post points are at same position, BUT are in different volumes : pLPMT_NNVT_MCPPMT, NNVTMCPPMT_body_phys
* same-ness was because of accidently getting narrowed to float precision on the way into deluxe buffer
* G4OpBoundaryProcessStatus SAM is abbrev for SameMaterial


tds3ip.sh 1::

   In [2]: 10218.522*math.sqrt(3)                                                                                                                                                                           
   Out[2]: 17698.999282260338

    In [3]: b.dx.shape                                                                                                                                                                                       
    Out[3]: (8, 10, 2, 4)

    In [4]: b.dx[5]                                                                                                                                                                                          
    Out[4]: 
    A([[[     0.5774,     -0.5774,      0.5774,      0.6   ],
        [    -0.7071,      0.    ,      0.7071,    440.    ]],

       [[ 10219.0996, -10219.0996,  10219.0996,     91.2555],
        [    -0.7071,      0.    ,      0.7071,    440.    ]],

       [[ 10288.3818, -10288.3818,  10288.3818,     91.8747],
        [    -0.7071,      0.    ,      0.7071,    440.    ]],

       [[ 11140.5127, -11140.5127,  11140.5127,     98.679 ],
        [     0.6946,      0.7168,     -0.0615,    440.    ]],

       [[ 11140.5127, -11140.5127,  11140.5127,     98.679 ],
        [     0.6946,      0.7168,     -0.0615,    440.    ]],

       [[ 11144.0537, -11143.6738,  11143.6641,     98.708 ],
        [     0.6946,      0.7168,     -0.0615,    440.    ]],

       [[     0.    ,      0.    ,      0.    ,      0.    ],
        [     0.    ,      0.    ,      0.    ,      0.    ]],

       [[     0.    ,      0.    ,      0.    ,      0.    ],
        [     0.    ,      0.    ,      0.    ,      0.    ]],

       [[     0.    ,      0.    ,      0.    ,      0.    ],
        [     0.    ,      0.    ,      0.    ,      0.    ]],

       [[     0.    ,      0.    ,      0.    ,      0.    ],
        [     0.    ,      0.    ,      0.    ,      0.    ]]])




After avoiding the float narrowing::


    In [4]: a[5,3]                                                                                                                                                                                       
    Out[4]: 
    array([[ 11140.512, -11140.512,  11140.512,     98.679],
           [     0.695,      0.717,     -0.061,    440.   ]])

    In [5]: a[5,4]                                                                                                                                                                                       
    Out[5]: 
    array([[ 11140.513, -11140.513,  11140.513,     98.679],
           [     0.695,      0.717,     -0.061,    440.   ]])

    In [6]: a[5,4] - a[5,3]                                                                                                                                                                              
    Out[6]: 
    array([[ 0.001, -0.001,  0.001,  0.   ],
           [ 0.   ,  0.   ,  0.   ,  0.   ]])








gpmt.py 
------------


Need to take a look at GDML plots again, as geometry has changed::

   scp P:/home/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/fe48b4d359786b95505117280fb5aac1/1/origin_CGDMLKludge.gdml /usr/local/opticks/
 
Also updates lvnames as no more virtual masks.

::


    cd ~/opticks/ana
    ipython
    > run gpmt.py 

    In [4]: lvs[0]                                                                                                                                                                                           
    Out[4]: 
    [101] Volume NNVTMCPPMT_log0x3a92ff0
    solid
    0 [457] Union NNVTMCPPMT_pmt_solid0x3a8f640    : right_xyz:0.0/0.0/-299.750
    l:0 [455] Union NNVTMCPPMT_pmt_solid_1_20x3a8f490   : right_xyz:0.0/0.0/-193.789
    l:0 [453] Ellipsoid NNVTMCPPMT_pmt_solid_1_Ellipsoid0x3a8ec40   : xyz 0.0,0.0,0.000   :  ax/by/cz 254.001/254.001/184.001  zcut1 -184.001 zcut2 184.001  
    r:0 [454] Polycone NNVTMCPPMT_pmt_solid_part20x3a8ed20   : xyz 0.0,0.0,0.000       :  zp_num  2 z:[19.7106720393278, -19.7106720393278] rmax:[50.001, 75.8277739512222] rmin:[0.0]  
    r:0 [456] Tube NNVTMCPPMT_pmt_solid_3_EndTube0x3a8f360   : xyz 0.0,0.0,172.501     :  rmin 0.0 rmax 50.001 hz 86.251 
    material
    [13] Material Pyrex0x3298a60 solid
    physvol 1
       Physvol NNVTMCPPMT_body_phys0x3a93320
     None None 

    In [5]: lvs[1]                                                                                                                                                                                           
    Out[5]: 
    [100] Volume NNVTMCPPMT_body_log0x3a92ee0
    solid
    0 [452] Union NNVTMCPPMT_body_solid0x3a905c0   : right_xyz:0.0/0.0/-299.750
    l:0 [450] Union NNVTMCPPMT_body_solid_1_20x3a903d0   : right_xyz:0.0/0.0/-193.789
    l:0 [448] Ellipsoid NNVTMCPPMT_body_solid_1_Ellipsoid0x3a8f8b0   : xyz 0.0,0.0,0.000   :  ax/by/cz 254.000/254.000/184.000  zcut1 -184.000 zcut2 184.000  
    r:0 [449] Polycone NNVTMCPPMT_body_solid_part20x3a8f990   : xyz 0.0,0.0,0.000      :  zp_num  2 z:[19.71113043771, -19.71113043771] rmax:[50.0, 75.8277739512222] rmin:[0.0]  
    r:0 [451] Tube NNVTMCPPMT_body_solid_3_EndTube0x3a902a0   : xyz 0.0,0.0,172.500    :  rmin 0.0 rmax 50.000 hz 86.250 
    material
    [13] Material Pyrex0x3298a60 solid
    physvol 2
       Physvol NNVTMCPPMT_inner1_phys0x3a933a0
     None None 
       Physvol NNVTMCPPMT_inner2_phys0x3a93450
     None None 



bn.npy
-------

::

    In [1]: a = np.load("bn.npy")

    In [2]: a
    Out[2]:
    array([[[  15208722,          0,          0,          0]],

           [[3890811154,          0,          0,          0]],

           [[3890811154,          0,          0,          0]],

           [[  15208722,          0,          0,          0]],

           [[3890811154,          0,          0,          0]],

           [[3890811154,          0,          0,          0]],

           [[3857256722,          0,          0,          0]],

           [[        18,          0,          0,          0]]], dtype=uint32)

    In [3]: a.shape
    Out[3]: (8, 1, 4)

    In [4]: a.view(np.int8)
    Out[4]:
    array([[[ 18,  17, -24,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]],
    TO BT BT AB

           [[ 18,  17, -23, -25,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]],
    TO BT BT BT SD

           [[ 18,  17, -23, -25,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]],
    TO BT BT BT SA

           [[ 18,  17, -24,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]],
    TO BT BT AB

           [[ 18,  17, -23, -25,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]],
    TO BT BT BT SD

           [[ 18,  17, -23, -25,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]],
    TO BT BT BT SD

           [[ 18,  17, -23, -27,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]],
    TO BT BT BT SA

           [[ 18,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]]], dtype=int8)
    TO AB


    als[:10]
    0: TO BT BT AB
    1: TO BT BT BT SD
    2: TO BT BT BT SA
    3: TO BT BT AB
    4: TO BT BT BT SD
    5: TO BT BT BT SD
    6: TO BT BT BT SA
    7: TO AB



* compare with final boundary from pflags

    In [8]: boundary    = (( a.ox[:,3,0].view(np.uint32) & 0xffff0000 ) >> 16 ).view(np.int16)[0::2]

    In [9]: boundary
    Out[9]: A([-24, -25, -25, -24, -25, -25, -27,  18], dtype=int16)

    In [10]: sensorIdx = (( a.ox[:,3,0].view(np.uint32) & 0x0000ffff ) >> 0 )

    In [11]: sensorIdx
    Out[11]: A([    0, 13954, 14041,     0,  3569,  3613,  3700,     0], dtype=uint32)




* need to up the stats : getting -25:Pyrex///Pyrex gives SD

* whats different about photon index 6, what that ones gets -27:Pyrex//NNVTMCPPMT_PMT_20inch_mirror_logsurf1/Vacuum



    In [5]:

           vv
     17 :  18 : Acrylic///LS 

     16 :  17 : Water///Acrylic 

     22 :  23 : Water///Water 

     23 :  24 : Water///Pyrex 

     24 :  25 : Pyrex///Pyrex 

     26 :  27 : Pyrex//NNVTMCPPMT_PMT_20inch_mirror_logsurf1/Vacuum 

::

    In [1]: run blib.py                                                                                                                                                                                  
     nbnd  36 nmat  17 nsur  20 
      0 :   1 : Galactic///Galactic 
      1 :   2 : Galactic///Rock 
      2 :   3 : Rock///Air 
      3 :   4 : Air///Air 
      4 :   5 : Air///LS 
      5 :   6 : Air///Steel 
      6 :   7 : Air///Tyvek 
      7 :   8 : Air///Aluminium 
      8 :   9 : Aluminium///Adhesive 
      9 :  10 : Adhesive///TiO2Coating 
     10 :  11 : TiO2Coating///Scintillator 
     11 :  12 : Rock///Tyvek 
     12 :  13 : Tyvek///vetoWater 
     13 :  14 : vetoWater///LatticedShellSteel 
     14 :  15 : vetoWater/CDTyvekSurface//Tyvek 
     15 :  16 : Tyvek///Water 
     16 :  17 : Water///Acrylic 
     17 :  18 : Acrylic///LS 
     18 :  19 : LS///Acrylic 
     19 :  20 : LS///PE_PA 
     20 :  21 : Water///Steel 
     21 :  22 : Water///PE_PA 
     22 :  23 : Water///Water 
     23 :  24 : Water///Pyrex 
     24 :  25 : Pyrex///Pyrex 
     25 :  26 : Pyrex/NNVTMCPPMT_PMT_20inch_photocathode_logsurf2/NNVTMCPPMT_PMT_20inch_photocathode_logsurf1/Vacuum 
     26 :  27 : Pyrex//NNVTMCPPMT_PMT_20inch_mirror_logsurf1/Vacuum 
     27 :  28 : Pyrex/HamamatsuR12860_PMT_20inch_photocathode_logsurf2/HamamatsuR12860_PMT_20inch_photocathode_logsurf1/Vacuum 
     28 :  29 : Pyrex//HamamatsuR12860_PMT_20inch_mirror_logsurf1/Vacuum 
     29 :  30 : Pyrex/PMT_3inch_photocathode_logsurf2/PMT_3inch_photocathode_logsurf1/Vacuum 
     30 :  31 : Pyrex//PMT_3inch_absorb_logsurf1/Vacuum 
     31 :  32 : Water///LS 
     32 :  33 : Water/Steel_surface/Steel_surface/Steel 
     33 :  34 : vetoWater///Water 
     34 :  35 : Pyrex/PMT_20inch_veto_photocathode_logsurf2/PMT_20inch_veto_photocathode_logsurf1/Vacuum 
     35 :  36 : Pyrex//PMT_20inch_veto_mirror_logsurf1/Vacuum 





removing the 1e-3 onion ?
-----------------------------

Notice use of the bizarre placement ctor : its really dumb using a PV to specify a LV.
in NNVTMCPPMTManager::helper_make_physical_volume

::

    // 1st ctor : makes most sense

    055     G4PVPlacement(G4RotationMatrix *pRot,
     56             const G4ThreeVector &tlate,
     57                   G4LogicalVolume *pCurrentLogical,
     58             const G4String& pName,
     59                   G4LogicalVolume *pMotherLogical,
     60                   G4bool pMany,
     61                   G4int  pCopyNo,
     62                   G4bool pSurfChk=false);


    // abhorrent ctor : using a PV to specify an LV : also bizarre, why move the pName ! 

    098     G4PVPlacement(G4RotationMatrix *pRot,
     99             const G4ThreeVector &tlate,
    100             const G4String &pName,
    101                   G4LogicalVolume *pLogical,
    102                   G4VPhysicalVolume *pMother,
    103                   G4bool pMany,
    104                   G4int pCopyNo,
    105                   G4bool pSurfChk=false);
    106       // A simple variation of the 1st constructor, only specifying the
    107       // mother volume as a pointer to its physical volume instead of its
    108       // logical volume. The effect is exactly the same.


    043 G4PVPlacement::G4PVPlacement( G4RotationMatrix *pRot,
     44                         const G4ThreeVector &tlate,
     45                         const G4String& pName,
     46                               G4LogicalVolume *pLogical,
     47                               G4VPhysicalVolume *pMother,
     48                               G4bool pMany,
     49                               G4int pCopyNo,
     50                               G4bool pSurfChk )
     51   : G4VPhysicalVolume(pRot,tlate,pName,pLogical,pMother),
     52     fmany(pMany), fallocatedRotM(false), fcopyNo(pCopyNo)
     53 {                       
     54   if (pMother)
     55   {
     56     G4LogicalVolume* motherLogical = pMother->GetLogicalVolume();
     57     if (pLogical == motherLogical)
     58     {
     59       G4Exception("G4PVPlacement::G4PVPlacement()", "GeomVol0002",
     60                   FatalException, "Cannot place a volume inside itself!");
     61     }
     62     SetMotherLogical(motherLogical);
     63     motherLogical->AddDaughter(this);
     64     if (pSurfChk) { CheckOverlaps(); }
     65   }
     66 }




hmm maybe the reason for the near degenerates is to "contain" the optical surface ?
--------------------------------------------------------------------------------------

* not convinced by that, why not just make the border between the pyrex outer pmt_solid and vacuum 
* could be related to the complicated CSG in used in standard offline

::

    656 void
    657 NNVTMCPPMTManager::helper_make_optical_surface()
    658 {
    659     new G4LogicalBorderSurface(GetName()+"_photocathode_logsurf1",
    660             inner1_phys, body_phys,
    661             Photocathode_opsurf);
    662     new G4LogicalBorderSurface(GetName()+"_photocathode_logsurf2",
    663             body_phys, inner1_phys,
    664             Photocathode_opsurf);
    665     new G4LogicalBorderSurface(GetName()+"_mirror_logsurf1",
    666             inner2_phys, body_phys,
    667             m_mirror_opsurf);
    668     new G4LogicalBorderSurface(GetName()+"_mirror_logsurf2",
    669             body_phys, inner2_phys,
    670             m_mirror_opsurf);
    671 }


