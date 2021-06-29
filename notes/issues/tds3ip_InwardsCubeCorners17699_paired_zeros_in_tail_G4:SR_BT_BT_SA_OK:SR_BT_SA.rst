tds3ip_InwardsCubeCorners17699_paired_zeros_in_tail_G4:SR_BT_BT_SA_OK:SR_BT_SA
======================================================================================


Related
--------

* prior: :doc:`tds3ip_InwardsCubeCorners17699_at_7_wavelengths`




G4 HAS EXTRA BT FROM A Pyrex/Water MICROSTEP GETTING BACK OUT THE PMT FOLLOWING A REFLECTION 
-----------------------------------------------------------------------------------------------


::

    (8 )  BT/BT     FrT                                           POST_SAVE 
    [   8](Stp ;opticalphoton stepNum   10(tk ;opticalphoton tid 122 pid 0 nm    480 mm  ori[ 10218.522-10218.522-10218.522 ; 17699.000]  pos[ -22085.45122504.35020717.901 ; 37728.561]  )
      pre pv    pLPMT_Hamamatsu_R12860 lv       HamamatsuR12860_log so matsuR12860_pmt_solid_1_9 mlv               lInnerWater mso               sInnerWater
      pre           Pyrex  Transportation        GeomBoundary pos[ -21820.249 22106.429 20379.713 ;  37150.372]  dir[   -0.345   0.599   0.722 ;    1.000]  pol[    0.349   0.796  -0.494 ;    1.000]  ns 190.743 nm 480.000 mm/ns 198.261
     post pv               pInnerWater lv               lInnerWater so               sInnerWater mlv            lReflectorInCD mso            sReflectorInCD
     post           Water  Transportation        GeomBoundary pos[ -21820.249 22106.430 20379.714 ;  37150.374]  dir[   -0.453   0.679   0.577 ;    1.000]  pol[    0.890   0.389   0.240 ;    1.000]  ns 190.743 nm 480.000 mm/ns 218.120
     Cfsp dpos[   -0.001   0.001   0.001 ;    0.002]  near_pos ddir[   -0.107   0.080  -0.145 ;    0.197]  dpol[    0.540  -0.408   0.734 ;    0.998]  dtim[    0.000]        epsilon 1e-06
     )



ATTEMPT FIX with m_suppress_all_microStep 
----------------------------------------------

* trying to match bookkeeping with despite slightly different geometry (the +0.001mm skin is not in GPU geometry)

::

    083 CRecorder::CRecorder(CCtx& ctx)
     84     :   
     85     m_ctx(ctx),
     86     m_ok(m_ctx.getOpticks()),
     87     m_microStep_mm(0.004),              //  see notes/issues/ok_lacks_SI-4BT-SD.rst
     88     m_suppress_same_material_microStep(true),
     89     m_suppress_all_microStep(true),
     90     m_mode(m_ok->getManagerMode()),   // --managermode
     91     m_recpoi(m_ok->isRecPoi()),   // --recpoi


    538         unsigned premat = m_material_bridge->getPreMaterial(step) ;
    539 
    540         unsigned postmat = m_material_bridge->getPostMaterial(step) ;
    541 
    542         bool suppress_microStep = false ;
    543         if(m_suppress_same_material_microStep ) suppress_microStep = premat == postmat && microStep ;
    544         if(m_suppress_all_microStep )           suppress_microStep = microStep ;
    545         // suppress_all_microStep trumps suppress_same_material_microStep
    546 



Most populated slots look agreeable, but chi2 very bad from paired zeros in tail : history migrations
--------------------------------------------------------------------------------------------------------

::

    epsilon:ana blyth$ tds3ip.sh 4

    ab.ahis
    .    all_seqhis_ana  cfo:sum  4:g4live:tds3ip   -4:g4live:tds3ip        c2        ab        ba 
    .                              80000     80000      1333.49/244 =  5.47   pvalue:P[c2>]:0.000  1-pvalue:P[c2<]:1.000  
       n             iseq         a         b    a-b       (a-b)^2/(a+b)         a/b                   b/a           [ns] label
    0000               4d     13237     13268    -31              0.04         0.998 +- 0.009        1.002 +- 0.009  [2 ] TO AB
    0001           7ccc6d      9959      9940     19              0.02         1.002 +- 0.010        0.998 +- 0.010  [6 ] TO SC BT BT BT SD
    0002            7cccd      5345      5374    -29              0.08         0.995 +- 0.014        1.005 +- 0.014  [5 ] TO BT BT BT SD
    0003              46d      4974      4974      0              0.00         1.000 +- 0.014        1.000 +- 0.014  [3 ] TO SC AB
    0004          7ccc66d      3803      3815    -12              0.02         0.997 +- 0.016        1.003 +- 0.016  [7 ] TO SC SC BT BT BT SD
    0005           8ccc6d      2432      2571   -139              3.86         0.946 +- 0.019        1.057 +- 0.021  [6 ] TO SC BT BT BT SA
    0006            8cc6d      2448      2329    119              2.96         1.051 +- 0.021        0.951 +- 0.020  [5 ] TO SC BT BT SA
    0007           7ccc5d      1912      1950    -38              0.37         0.981 +- 0.022        1.020 +- 0.023  [6 ] TO RE BT BT BT SD
    0008             466d      1836      1899    -63              1.06         0.967 +- 0.023        1.034 +- 0.024  [4 ] TO SC SC AB
    0009              45d      1771      1817    -46              0.59         0.975 +- 0.023        1.026 +- 0.024  [3 ] TO RE AB
    0010         7ccc666d      1507      1497     10              0.03         1.007 +- 0.026        0.993 +- 0.026  [8 ] TO SC SC SC BT BT BT SD
    0011            8cccd      1386      1294     92              3.16         1.071 +- 0.029        0.934 +- 0.026  [5 ] TO BT BT BT SA
    0012          8ccc66d       976       991    -15              0.11         0.985 +- 0.032        1.015 +- 0.032  [7 ] TO SC SC BT BT BT SA
    0013            4cc6d       895       911    -16              0.14         0.982 +- 0.033        1.018 +- 0.034  [5 ] TO SC BT BT AB
    0014           8cc66d       901       890     11              0.07         1.012 +- 0.034        0.988 +- 0.033  [6 ] TO SC SC BT BT SA
    0015           8ccc5d       823       824     -1              0.00         0.999 +- 0.035        1.001 +- 0.035  [6 ] TO RE BT BT BT SA
    0016          7ccc56d       758       781    -23              0.34         0.971 +- 0.035        1.030 +- 0.037  [7 ] TO SC RE BT BT BT SD
    0017             4c6d       735       747    -12              0.10         0.984 +- 0.036        1.016 +- 0.037  [4 ] TO SC BT AB
    0018            4666d       747       691     56              2.18         1.081 +- 0.040        0.925 +- 0.035  [5 ] TO SC SC SC AB
    .                              80000     80000      1333.49/244 =  5.47   pvalue:P[c2>]:0.000  1-pvalue:P[c2<]:1.000  



Distintive paired "zeros" out in the tail, differing in the G4:"SR BT BT SA" OK:"SR BT SA"::

    0051        8ccaccc6d         0       320   -320            320.00         0.000 +- 0.000        0.000 +- 0.000  [9 ] TO SC BT BT BT SR BT BT SA
    0058         8caccc6d       232         0    232            232.00         0.000 +- 0.000        0.000 +- 0.000  [8 ] TO SC BT BT BT SR BT SA

    0087       8ccaccc66d         0       146   -146            146.00         0.000 +- 0.000        0.000 +- 0.000  [10]  TO SC SC BT BT BT SR BT BT SA
    0103        8caccc66d       111         1    110            108.04       111.000 +- 10.536        0.009 +- 0.009  [9 ] TO SC SC BT BT BT SR BT SA




::

    In [12]: a.sel = "TO SC BT BT BT SR BT SA"

    In [13]: b.sel = "TO SC BT BT BT SR BT BT SA"


    In [19]: apos = a.ox[:,0,:3]

    In [20]: apos.shape
    Out[20]: (232, 3)

    In [21]: bpos = b.ox[:,0,:3]

    In [22]: bpos.shape
    Out[22]: (320, 3)

    In [23]: np.unique(np.sqrt(np.sum(apos*apos, axis=1)))     ## all ending on Tyvek 
    Out[23]: A([20049.998, 20050.   , 20050.002], dtype=float32)

    In [24]: np.unique(np.sqrt(np.sum(bpos*bpos, axis=1)))
    Out[24]: A([20049.998, 20050.   , 20050.002], dtype=float32)



    In [25]: a.mat                                                                                                                                                                                    
    Out[25]: 
    seqmat_ana
    .                     cfo:-  4:g4live:tds3ip 
    .                                232         1.00 
       n             iseq         frac           a    a-b      [ns] label
    0000         3eddeb11        1.000         232        [8 ] LS LS Ac Py Va Va Py Ty
       n             iseq         frac           a    a-b      [ns] label
    .                                232         1.00 

    In [26]: b.mat                                                                                                                                                                                    
    Out[26]: 
    seqmat_ana
    .                     cfo:-  -4:g4live:tds3ip 
    .                                320         1.00 
       n             iseq         frac           a    a-b      [ns] label
    0000        3edddeb11        1.000         320        [9 ] LS LS Ac Py Va Va Va Py Ty
       n             iseq         frac           a    a-b      [ns] label
    .                                320         1.00 



    In [29]: np.set_printoptions(edgeitems=16)

    In [30]: a.bn.view(np.int8).reshape(-1,16)
    Out[30]:
    A([[ 18,  18,  17, -19, -22,  19,  16,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  18,  17, -19, -22,  19,  16,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  18,  17, -19, -22,  19,  16,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  18,  17, -19, -24,  19,  16,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  18,  17, -19, -22,  19,  16,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  18,  17, -19, -22,  19,  16,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  18,  17, -19, -24,  19,  16,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  18,  17, -19, -22,  19,  16,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  18,  17, -19, -22,  19,  16,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  18,  17, -19, -22,  19,  16,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  18,  17, -19, -22,  19,  16,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  18,  17, -19, -22,  19,  16,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  18,  17, -19, -22,  19,  16,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  18,  17, -19, -22,  19,  16,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  18,  17, -19, -24,  19,  16,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  18,  17, -19, -22,  19,  16,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       ...,
       [ 18,  18,  17, -19, -24,  19,  16,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  18,  17, -19, -22,  19,  16,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  18,  17, -19, -22,  19,  16,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  18,  17, -19, -24,  19,  16,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  18,  17, -19, -22,  19,  16,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  18,  17, -19, -22,  19,  16,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  18,  17, -19, -22,  19,  16,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  18,  17, -19, -22,  19,  16,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  18,  17, -19, -24,  19,  16,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  18,  17, -19, -24,  19,  16,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  18,  17, -19, -22,  19,  16,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  18,  17, -19, -22,  19,  16,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  18,  17, -19, -22,  19,  16,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  18,  17, -19, -22,  19,  16,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  18,  17, -19, -22,  19,  16,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  18,  17, -19, -22,  19,  16,   0,   0,   0,   0,   0,   0,   0,   0,   0]], dtype=int8)


    In [33]: print(a.blib.format(a.bn.view(np.int8).reshape(-1,16)[0]))                                                                                                                               
     18 : Acrylic///LS          SC still in LS
     18 : Acrylic///LS          BT thru to Ac
     17 : Water///Acrylic       BT thru to Wa 
    -19 : LS///Acrylic          BT   ???? inconsistent boundary : one of those should be Water ???   DOES THIS MEAN OVERLAPPED VOLUME OR COINCIDENT SURFACE
    -22 : Water///PE_PA         SR
     19 : LS///Acrylic          BT
     16 : Tyvek//Implicit_RINDEX_NoRINDEX_pInnerWater_pCentralDetector/Water  SA


    In [34]: print(a.blib.format(a.bn.view(np.int8).reshape(-1,16)[3]))
     18 : Acrylic///LS          SC 
     18 : Acrylic///LS          BT
     17 : Water///Acrylic       BT 
    -19 : LS///Acrylic          BT    ??? inconsistent boundary ??? 
    -24 : Pyrex///Pyrex         SR
     19 : LS///Acrylic          BT
     16 : Tyvek//Implicit_RINDEX_NoRINDEX_pInnerWater_pCentralDetector/Water    SA

    In [35]: a.sel
    Out[35]: 'TO SC BT BT BT SR BT SA'

    :.,$s/\s*$//g







Debug run to get the volume names
--------------------------------------



:: 

    export DBGSEQHIS=0x8ccaccc6d  will make tds3 add the commandline option

    --dbgseqhis 0x8ccaccc6d


    P[blyth@localhost ~]$ jre
    P[blyth@localhost ~]$ export DBGSEQHIS=0x8ccaccc6d
    P[blyth@localhost ~]$ tds3ip



    2021-06-29 20:41:10.921 INFO  [126347] [CDebug::dump@188] CDebug::postTrack
    2021-06-29 20:41:10.921 INFO  [126347] [CRec::dump@194] CDebug::dump record_id 121  origin[ 10218.522-10218.522-10218.522 ; 17699.000]   Ori[ 10218.522-10218.522-10218.522 ; 17699.000] 
    2021-06-29 20:41:10.921 INFO  [126347] [CRec::dump@200]  nstp 10
    (0 )  TO/SC     NAB                       PRE_SAVE POST_SAVE STEP_START 
    [   0](Stp ;opticalphoton stepNum   10(tk ;opticalphoton tid 122 pid 0 nm    480 mm  ori[ 10218.522-10218.522-10218.522 ; 17699.000]  pos[ -22085.45122504.35020717.901 ; 37728.561]  )
      pre pv                   pTarget lv                   lTarget so                   sTarget mlv                  lAcrylic mso                  sAcrylic
      pre              LS          noProc           Undefined pos[      0.000     0.000     0.000 ;      0.000]  dir[   -0.577   0.577   0.577 ;    1.000]  pol[   -0.707   0.000  -0.707 ;    1.000]  ns  0.200 nm 480.000 mm/ns 195.663
     post pv                   pTarget lv                   lTarget so                   sTarget mlv                  lAcrylic mso                  sAcrylic
     post              LS      OpRayleigh    PostStepDoItProc pos[ -20086.920 20086.920 20086.920 ;  34791.566]  dir[   -0.648   0.748   0.143 ;    1.000]  pol[   -0.509  -0.286  -0.812 ;    1.000]  ns 178.014 nm 480.000 mm/ns 195.663
     Cfsp dpos[ -20086.92020086.92020086.920 ; 34791.566]  ddir[   -0.071   0.170  -0.434 ;    0.472]  dpol[    0.198  -0.286  -0.105 ;    0.363]  dtim[  177.814]        epsilon 1e-06
     )
    (1 )  SC/BT     FrT                                           POST_SAVE 
    [   1](Stp ;opticalphoton stepNum   10(tk ;opticalphoton tid 122 pid 0 nm    480 mm  ori[ 10218.522-10218.522-10218.522 ; 17699.000]  pos[ -22085.45122504.35020717.901 ; 37728.561]  )
      pre pv                   pTarget lv                   lTarget so                   sTarget mlv                  lAcrylic mso                  sAcrylic
      pre              LS      OpRayleigh    PostStepDoItProc pos[ -20086.920 20086.920 20086.920 ;  34791.566]  dir[   -0.648   0.748   0.143 ;    1.000]  pol[   -0.509  -0.286  -0.812 ;    1.000]  ns 178.014 nm 480.000 mm/ns 195.663
     post pv                  pAcrylic lv                  lAcrylic so                  sAcrylic mlv               lInnerWater mso               sInnerWater
     post         Acrylic  Transportation        GeomBoundary pos[ -20528.061 20595.787 20184.253 ;  35397.625]  dir[   -0.648   0.747   0.145 ;    1.000]  pol[   -0.512  -0.569   0.643 ;    1.000]  ns 181.491 nm 480.000 mm/ns 195.632
     Cfsp dpos[ -441.141 508.867  97.333 ;  680.459]  ddir[   -0.000  -0.000   0.002 ;    0.002]  dpol[   -0.003  -0.283   1.455 ;    1.482]  dtim[    3.478]        epsilon 1e-06
     )
    (2 )  BT/BT     FrT                                           POST_SAVE 
    [   2](Stp ;opticalphoton stepNum   10(tk ;opticalphoton tid 122 pid 0 nm    480 mm  ori[ 10218.522-10218.522-10218.522 ; 17699.000]  pos[ -22085.45122504.35020717.901 ; 37728.561]  )
      pre pv                  pAcrylic lv                  lAcrylic so                  sAcrylic mlv               lInnerWater mso               sInnerWater
      pre         Acrylic  Transportation        GeomBoundary pos[ -20528.061 20595.787 20184.253 ;  35397.625]  dir[   -0.648   0.747   0.145 ;    1.000]  pol[   -0.512  -0.569   0.643 ;    1.000]  ns 181.491 nm 480.000 mm/ns 195.632
     post pv               pInnerWater lv               lInnerWater so               sInnerWater mlv            lReflectorInCD mso            sReflectorInCD
     post           Water  Transportation        GeomBoundary pos[ -20614.661 20695.623 20203.665 ;  35517.054]  dir[   -0.647   0.757   0.092 ;    1.000]  pol[   -0.538  -0.539   0.649 ;    1.000]  ns 182.174 nm 480.000 mm/ns 218.120
     Cfsp dpos[  -86.600  99.835  19.412 ;  133.580]  ddir[    0.001   0.009  -0.053 ;    0.054]  dpol[   -0.026   0.031   0.005 ;    0.040]  dtim[    0.683]        epsilon 1e-06
     )
    (3 )  BT/BT     FrT                                           POST_SAVE 
    [   3](Stp ;opticalphoton stepNum   10(tk ;opticalphoton tid 122 pid 0 nm    480 mm  ori[ 10218.522-10218.522-10218.522 ; 17699.000]  pos[ -22085.45122504.35020717.901 ; 37728.561]  )
      pre pv               pInnerWater lv               lInnerWater so               sInnerWater mlv            lReflectorInCD mso            sReflectorInCD
      pre           Water  Transportation        GeomBoundary pos[ -20614.661 20695.623 20203.665 ;  35517.054]  dir[   -0.647   0.757   0.092 ;    1.000]  pol[   -0.538  -0.539   0.649 ;    1.000]  ns 182.174 nm 480.000 mm/ns 218.120
     post pv    pLPMT_Hamamatsu_R12860 lv       HamamatsuR12860_log so matsuR12860_pmt_solid_1_9 mlv               lInnerWater mso               sInnerWater
     post           Pyrex  Transportation        GeomBoundary pos[ -21809.558 22092.502 20373.216 ;  37132.242]  dir[   -0.675   0.731  -0.101 ;    1.000]  pol[   -0.674  -0.667  -0.317 ;    1.000]  ns 190.638 nm 480.000 mm/ns 198.261
     Cfsp dpos[ -1194.8971396.879 169.551 ; 1846.022]  ddir[   -0.028  -0.026  -0.193 ;    0.197]  dpol[   -0.136  -0.128  -0.966 ;    0.984]  dtim[    8.463]        epsilon 1e-06
     )


    (4 )  BT/BT     SAM                                           POST_SKIP 
    [   4](Stp ;opticalphoton stepNum   10(tk ;opticalphoton tid 122 pid 0 nm    480 mm  ori[ 10218.522-10218.522-10218.522 ; 17699.000]  pos[ -22085.45122504.35020717.901 ; 37728.561]  )
      pre pv    pLPMT_Hamamatsu_R12860 lv       HamamatsuR12860_log so matsuR12860_pmt_solid_1_9 mlv               lInnerWater mso               sInnerWater
      pre           Pyrex  Transportation        GeomBoundary pos[ -21809.558 22092.502 20373.216 ;  37132.242]  dir[   -0.675   0.731  -0.101 ;    1.000]  pol[   -0.674  -0.667  -0.317 ;    1.000]  ns 190.638 nm 480.000 mm/ns 198.261
     post pv HamamatsuR12860_body_phys lv  HamamatsuR12860_body_log so atsuR12860_body_solid_1_9 mlv       HamamatsuR12860_log mso matsuR12860_pmt_solid_1_9
     post           Pyrex  Transportation        GeomBoundary pos[ -21809.560 22092.504 20373.216 ;  37132.244]  dir[   -0.675   0.731  -0.101 ;    1.000]  pol[   -0.674  -0.667  -0.317 ;    1.000]  ns 190.638 nm 480.000 mm/ns 198.261
     Cfsp dpos[   -0.001   0.001  -0.000 ;    0.002]  near_pos same_dir same_pol dtim[    0.000]        epsilon 1e-06
     )


    (5 )  BT/SR     SpR                                  POST_SAVE MAT_SWAP 
    [   5](Stp ;opticalphoton stepNum   10(tk ;opticalphoton tid 122 pid 0 nm    480 mm  ori[ 10218.522-10218.522-10218.522 ; 17699.000]  pos[ -22085.45122504.35020717.901 ; 37728.561]  )
      pre pv HamamatsuR12860_body_phys lv  HamamatsuR12860_body_log so atsuR12860_body_solid_1_9 mlv       HamamatsuR12860_log mso matsuR12860_pmt_solid_1_9
      pre           Pyrex  Transportation        GeomBoundary pos[ -21809.560 22092.504 20373.216 ;  37132.244]  dir[   -0.675   0.731  -0.101 ;    1.000]  pol[   -0.674  -0.667  -0.317 ;    1.000]  ns 190.638 nm 480.000 mm/ns 198.261
     post pv mamatsuR12860_inner2_phys lv amamatsuR12860_inner2_log so suR12860_inner2_solid_1_9 mlv  HamamatsuR12860_body_log mso atsuR12860_body_solid_1_9
     post          Vacuum  Transportation        GeomBoundary pos[ -21816.633 22100.156 20372.154 ;  37140.369]  dir[   -0.345   0.599   0.722 ;    1.000]  pol[    0.349   0.796  -0.494 ;    1.000]  ns 190.690 nm 480.000 mm/ns 198.261
     Cfsp dpos[   -7.074   7.652  -1.062 ;   10.475]  ddir[    0.330  -0.131   0.824 ;    0.897]  dpol[    1.023   1.463  -0.177 ;    1.794]  dtim[    0.053]        epsilon 1e-06
     )


    (6 )  SR/NA     STS                                           POST_SKIP 
    [   6](Stp ;opticalphoton stepNum   10(tk ;opticalphoton tid 122 pid 0 nm    480 mm  ori[ 10218.522-10218.522-10218.522 ; 17699.000]  pos[ -22085.45122504.35020717.901 ; 37728.561]  )
      pre pv mamatsuR12860_inner2_phys lv amamatsuR12860_inner2_log so suR12860_inner2_solid_1_9 mlv  HamamatsuR12860_body_log mso atsuR12860_body_solid_1_9
      pre          Vacuum  Transportation        GeomBoundary pos[ -21816.633 22100.156 20372.154 ;  37140.369]  dir[   -0.345   0.599   0.722 ;    1.000]  pol[    0.349   0.796  -0.494 ;    1.000]  ns 190.690 nm 480.000 mm/ns 198.261
     post pv HamamatsuR12860_body_phys lv  HamamatsuR12860_body_log so atsuR12860_body_solid_1_9 mlv       HamamatsuR12860_log mso matsuR12860_pmt_solid_1_9
     post           Pyrex  Transportation        GeomBoundary pos[ -21816.633 22100.156 20372.154 ;  37140.369]  dir[   -0.345   0.599   0.722 ;    1.000]  pol[    0.349   0.796  -0.494 ;    1.000]  ns 190.690 nm 480.000 mm/ns 198.261
     Cfsp same_pos same_dir same_pol same_time       epsilon 1e-06
     )


    (7 )  NA/BT     SAM                                           POST_SAVE 
    [   7](Stp ;opticalphoton stepNum   10(tk ;opticalphoton tid 122 pid 0 nm    480 mm  ori[ 10218.522-10218.522-10218.522 ; 17699.000]  pos[ -22085.45122504.35020717.901 ; 37728.561]  )
      pre pv HamamatsuR12860_body_phys lv  HamamatsuR12860_body_log so atsuR12860_body_solid_1_9 mlv       HamamatsuR12860_log mso matsuR12860_pmt_solid_1_9
      pre           Pyrex  Transportation        GeomBoundary pos[ -21816.633 22100.156 20372.154 ;  37140.369]  dir[   -0.345   0.599   0.722 ;    1.000]  pol[    0.349   0.796  -0.494 ;    1.000]  ns 190.690 nm 480.000 mm/ns 198.261
     post pv    pLPMT_Hamamatsu_R12860 lv       HamamatsuR12860_log so matsuR12860_pmt_solid_1_9 mlv               lInnerWater mso               sInnerWater
     post           Pyrex  Transportation        GeomBoundary pos[ -21820.249 22106.429 20379.713 ;  37150.372]  dir[   -0.345   0.599   0.722 ;    1.000]  pol[    0.349   0.796  -0.494 ;    1.000]  ns 190.743 nm 480.000 mm/ns 198.261
     Cfsp dpos[   -3.615   6.274   7.559 ;   10.467]  same_dir same_pol dtim[    0.053]        epsilon 1e-06
     )

    (8 )  BT/BT     FrT                                           POST_SAVE 
    [   8](Stp ;opticalphoton stepNum   10(tk ;opticalphoton tid 122 pid 0 nm    480 mm  ori[ 10218.522-10218.522-10218.522 ; 17699.000]  pos[ -22085.45122504.35020717.901 ; 37728.561]  )
      pre pv    pLPMT_Hamamatsu_R12860 lv       HamamatsuR12860_log so matsuR12860_pmt_solid_1_9 mlv               lInnerWater mso               sInnerWater
      pre           Pyrex  Transportation        GeomBoundary pos[ -21820.249 22106.429 20379.713 ;  37150.372]  dir[   -0.345   0.599   0.722 ;    1.000]  pol[    0.349   0.796  -0.494 ;    1.000]  ns 190.743 nm 480.000 mm/ns 198.261
     post pv               pInnerWater lv               lInnerWater so               sInnerWater mlv            lReflectorInCD mso            sReflectorInCD
     post           Water  Transportation        GeomBoundary pos[ -21820.249 22106.430 20379.714 ;  37150.374]  dir[   -0.453   0.679   0.577 ;    1.000]  pol[    0.890   0.389   0.240 ;    1.000]  ns 190.743 nm 480.000 mm/ns 218.120
     Cfsp dpos[   -0.001   0.001   0.001 ;    0.002]  near_pos ddir[   -0.107   0.080  -0.145 ;    0.197]  dpol[    0.540  -0.408   0.734 ;    0.998]  dtim[    0.000]        epsilon 1e-06
     )

    THIS IS THE EXTRA BT : GETTING BACK OUT THE PMT FOLLOWING A REFLECTION 


    (9 )  BT/SA     NRI              POST_SAVE POST_DONE LAST_POST SURF_ABS 
    [   9](Stp ;opticalphoton stepNum   10(tk ;opticalphoton tid 122 pid 0 nm    480 mm  ori[ 10218.522-10218.522-10218.522 ; 17699.000]  pos[ -22085.45122504.35020717.901 ; 37728.561]  )
      pre pv               pInnerWater lv               lInnerWater so               sInnerWater mlv            lReflectorInCD mso            sReflectorInCD
      pre           Water  Transportation        GeomBoundary pos[ -21820.249 22106.430 20379.714 ;  37150.374]  dir[   -0.453   0.679   0.577 ;    1.000]  pol[    0.890   0.389   0.240 ;    1.000]  ns 190.743 nm 480.000 mm/ns 218.120
     post pv          pCentralDetector lv            lReflectorInCD so            sReflectorInCD mlv           lOuterWaterPool mso           sOuterWaterPool
     post           Tyvek  Transportation        GeomBoundary pos[ -22085.451 22504.350 20717.901 ;  37728.561]  dir[   -0.453   0.679   0.577 ;    1.000]  pol[    0.890   0.389   0.240 ;    1.000]  ns 193.428 nm 480.000 mm/ns 218.120
     Cfsp dpos[ -265.201 397.920 338.187 ;  585.698]  same_dir same_pol dtim[    2.685]        epsilon 1e-06
     )


    2021-06-29 20:41:10.923 INFO  [126347] [CRec::dump@204]  npoi 0
    2021-06-29 20:41:10.923 INFO  [126347] [CDebug::dump_brief@204] CRecorder::dump_brief m_ctx._record_id      121 m_photon._badflag     0 --dbgseqhis  sas: POST_SAVE POST_DONE LAST_POST SURF_ABS 
    2021-06-29 20:41:10.923 INFO  [126347] [CDebug::dump_brief@213]  seqhis        8ccaccc6d    TO SC BT BT BT SR BT BT SA                      
    2021-06-29 20:41:10.923 INFO  [126347] [CDebug::dump_brief@218]  mskhis             1aa0    SC|SA|SR|BT|TO
    2021-06-29 20:41:10.923 INFO  [126347] [CDebug::dump_brief@223]  seqmat        3edddeb11    LS LS Acrylic Water Pyrex Pyrex Pyrex Water Tyvek - - - - - - - 
    2021-06-29 20:41:10.923 INFO  [126347] [CDebug::dump_sequence@231] CDebug::dump_sequence



::

   0:TO/SC NAB LS-LS    
   1:SC/BT FrT LS-Ac
   2:BT/BT FrT Ac-Wa
   3:BT/BT FrT Wa-Py
   4:BT/BT SAM Py-Py    ## this one gets suppressed ? because the solid is skipped from GPU geom
   5:BT/SR SpR Py-Va     
   6:SR/NA STS Va-Py    ## STS:step-too-small following SpR:reflection is something I recall getting
   7:NA/BT SAM Py-Py
   8:BT/BT FrT Py-Wa
   9:BT/SA NRI Wa-Ty

   b.sel = "TO SC BT BT BT SR BT BT SA"


Hmnm : could just be a bookkeeping emulation issue in CRecorder::postTrackWriteSteps following SR 
-------------------------------------------------------------------------------------------------------

::

    In [39]: a.sel = "*SR*"                                                                                                                                                                             

    In [40]: a.his                                                                                                                                                                                      
    Out[40]: 
    seqhis_ana
    .                     cfo:-  4:g4live:tds3ip 
    .                                902         1.00 
       n             iseq         frac           a    a-b      [ns] label
    0000         8caccc6d        0.257         232        [8 ] TO SC BT BT BT SR BT SA
    0001        8caccc66d        0.123         111        [9 ] TO SC SC BT BT BT SR BT SA
    0002         8caccc5d        0.072          65        [8 ] TO RE BT BT BT SR BT SA
    0003       8caccc666d        0.040          36        [10] TO SC SC SC BT BT BT SR BT SA
    0004        8cacbcc6d        0.027          24        [9 ] TO SC BT BT BR BT SR BT SA
    0005        8caccc65d        0.021          19        [9 ] TO RE SC BT BT BT SR BT SA
    0006        8caccc56d        0.020          18        [9 ] TO SC RE BT BT BT SR BT SA
    0007       8caccc566d        0.019          17        [10] TO SC SC RE BT BT BT SR BT SA
    0008       8cabaccc6d        0.019          17        [10] TO SC BT BT BT SR BR SR BT SA
    0009       caccc6666d        0.018          16        [10] TO SC SC SC SC BT BT BT SR BT
    0010        8caccc55d        0.016          14        [9 ] TO RE RE BT BT BT SR BT SA
    0011       8cacbcc66d        0.016          14        [10] TO SC SC BT BT BR BT SR BT SA
    0012       caccaccc6d        0.014          13        [10] TO SC BT BT BT SR BT BT SR BT
    0013       8caccccc6d        0.012          11        [10] TO SC BT BT BT BT BT SR BT SA
    0014       8caccc555d        0.011          10        [10] TO RE RE RE BT BT BT SR BT SA
    0015       8caccc556d        0.009           8        [10] TO SC RE RE BT BT BT SR BT SA
    0016       8caccc665d        0.009           8        [10] TO RE SC SC BT BT BT SR BT SA
    0017       8caccc656d        0.008           7        [10] TO SC RE SC BT BT BT SR BT SA
    0018       abaccc666d        0.008           7        [10] TO SC SC SC BT BT BT SR BR SR
    0019        8cacbcc5d        0.008           7        [9 ] TO RE BT BT BR BT SR BT SA
    0020       cabaccc66d        0.007           6        [10] TO SC SC BT BT BT SR BR SR BT
    0021       8caccc655d        0.007           6        [10] TO RE RE SC BT BT BT SR BT SA
    0022         4caccc6d        0.007           6        [8 ] TO SC BT BT BT SR BT AB


    In [41]: b.sel = "*SR*"                                                                                                                                                                             

    In [42]: b.his[:20]                                                                                                                                                                                 
    Out[42]: 
    seqhis_ana
    .                     cfo:-  -4:g4live:tds3ip 
    .                               1055         1.00 
       n             iseq         frac           a    a-b      [ns] label
    0000        8ccaccc6d        0.303         320        [9 ] TO SC BT BT BT SR BT BT SA
    0001       8ccaccc66d        0.138         146        [10] TO SC SC BT BT BT SR BT BT SA
    0002       ccaccc666d        0.067          71        [10] TO SC SC SC BT BT BT SR BT BT
    0003        8ccaccc5d        0.060          63        [9 ] TO RE BT BT BT SR BT BT SA
    0004       8ccaccc65d        0.031          33        [10] TO RE SC BT BT BT SR BT BT SA
    0005       cabcaccc6d        0.027          29        [10] TO SC BT BT BT SR BT BR SR BT
    0006       8ccaccc55d        0.024          25        [10] TO RE RE BT BT BT SR BT BT SA
    0007       8ccaccc56d        0.024          25        [10] TO SC RE BT BT BT SR BT BT SA
    0008       8ccacbcc6d        0.017          18        [10] TO SC BT BT BR BT SR BT BT SA
    0009       abcaccc66d        0.016          17        [10] TO SC SC BT BT BT SR BT BR SR
    0010       caccc6666d        0.015          16        [10] TO SC SC SC SC BT BT BT SR BT
    0011       ccaccc656d        0.011          12        [10] TO SC RE SC BT BT BT SR BT BT
    0012       caccc5666d        0.010          11        [10] TO SC SC SC RE BT BT BT SR BT
    0013       cabcaccc5d        0.009          10        [10] TO RE BT BT BT SR BT BR SR BT
    0014       acccaccc6d        0.009          10        [10] TO SC BT BT BT SR BT BT BT SR
    0015       8ccacbcc5d        0.009          10        [10] TO RE BT BT BR BT SR BT BT SA
    0016       caccc6566d        0.009           9        [10] TO SC SC RE SC BT BT BT SR BT
    0017       ccaccc556d        0.009           9        [10] TO SC RE RE BT BT BT SR BT BT
    0018       ccacbcc66d        0.008           8        [10] TO SC SC BT BT BR BT SR BT BT
    .                               1055         1.00 



