GScene compareMeshes : bbox comparisons
==========================================

Some very different bbox obtained from GMesh from the two branches.

* GMesh are an approximation of the real CSG geometry, but so 
  long as polygonization is not a placeholder from a poly fail 
  it is expected to match within ~0.2mm 

* BUT ana branch polygonization has many fails (and some partials)
  so now comparing with the analytic prim/composite CSG bbox to avoid
  this noise

* real geometry check is to compare intersection positions between the two 
  CSG implementations... 

  * however bbox pre-checking is convenient way to find big problems.

  * bugs with polygonizations are non critical... 

  * CSG bugs are critical


Classification of top 25 issues down to 1mm
-----------------------------------------------


This document classifies causes of issues down to 1mm : at which point parsurf precision 
noise makes no point in continuing.
  
Two main causes, know how to proceed with:

* need for tube deltaphi implementation : PRIORITY 1
* some cyco coincidences that uncoincide fails to fix : PRIORITY 2 


Two lesser ones, must wait on CSG intersect comparisons 

* trapezoid (lvidx66) 
* g4poly bug missing coincident sub-objects from unions 



parsurf vs g4poly bbox chart
----------------------------------

::

    op --gltf 44           # dump the compare meshes table

    // :set nowrap
    2017-07-07 11:49:14.118 INFO  [3791106] [GScene::compareMeshes_GMeshBB@435] GScene::compareMeshes_GMeshBB num_meshes 249 cut 0.1 bbty CSG_BBOX_PARSURF parsurf_level 2 parsurf_target 200
       3869.75               RadialShieldUnit0xc3d7da8 lvidx  56 nsp    507 amn (  -2262.150 -2262.150  -498.500) bmn (   1607.600     0.000  -498.500) dmn (  -3869.750 -2262.150     0.000) amx (   2262.150  2262.150   498.500) bmx (   2262.150  1589.370   498.500) dmx (      0.000   672.780     0.000)
       3407.72               SstBotCirRibBase0xc26e2d0 lvidx  65 nsp    324 amn (  -2000.000 -2000.000  -215.000) bmn (   1407.720    12.467  -215.000) dmn (  -3407.720 -2012.468     0.000) amx (   1847.759  2000.000   215.000) bmx (   1998.360  1404.240   215.000) dmx (   -150.601   595.760     0.000)
       2074.65               SstTopCirRibBase0xc264f78 lvidx  69 nsp    352 amn (  -1220.000 -1220.000  -115.945) bmn (    854.653    10.020  -115.945) dmn (  -2074.653 -1230.020     0.000) amx (   1220.000  1220.000   115.945) bmx (   1218.680   854.688   115.945) dmx (      1.320   365.312     0.000)
        345.51                SstTopRadiusRib0xc271720 lvidx  66 nsp    408 amn (   -345.000   -10.000 -1114.250) bmn (   -345.510   -10.000 -1114.250) dmn (      0.510     0.000     0.000) amx (      0.000    10.000  1114.250) bmx (    345.510    10.000  1114.250) dmx (   -345.510     0.000     0.000)
           320                      SstTopHub0xc2643d8 lvidx  68 nsp    317 amn (   -220.500  -220.500  -340.000) bmn (   -220.500  -220.500  -340.000) dmn (      0.000     0.000     0.000) amx (    220.500   220.500     0.000) bmx (    220.500   220.500  -320.000) dmx (      0.000     0.000   320.000)
       28.0747              OcrGdsTfbInLsoOfl0xc2b5ba0 lvidx  83 nsp    243 amn (    -32.500   -32.500  -247.488) bmn (    -32.500   -32.500  -219.413) dmn (      0.000     0.000   -28.075) amx (     32.500    32.500   247.488) bmx (     32.500    32.500   247.488) dmx (      0.000     0.000    -0.000)
       26.2183                   OcrGdsLsoPrt0xc104978 lvidx  81 nsp    342 amn (    -98.000   -98.000     0.000) bmn (    -98.000   -98.000    26.218) dmn (      0.000     0.000   -26.218) amx (     98.000    98.000   214.596) bmx (     98.000    98.000   214.596) dmx (      0.000     0.000     0.000)
            20               headon-pmt-mount0xc2a7670 lvidx  55 nsp    365 amn (    -51.500   -51.500  -120.000) bmn (    -36.850   -36.850  -100.000) dmn (    -14.650   -14.650   -20.000) amx (     51.500    51.500   100.000) bmx (     36.850    36.850   100.000) dmx (     14.650    14.650     0.000)
            12           near_side_long_hbeam0xbf3b5d0 lvidx  17 nsp    450 amn (  -2000.000  -100.000  -147.000) bmn (  -2000.000   -99.876  -135.000) dmn (      0.000    -0.124   -12.000) amx (   2000.000   100.000   147.000) bmx (   2000.070   100.124   146.908) dmx (     -0.070    -0.124     0.092)
        10.035                   weight-shell0xc307920 lvidx 103 nsp    567 amn (    -10.035   -10.035   -28.510) bmn (    -10.035   -10.035   -18.475) dmn (      0.000     0.000   -10.035) amx (     10.035    10.035    28.510) bmx (     10.035    10.035    18.475) dmx (      0.000     0.000    10.035)
        10.035        AmCCo60AcrylicContainer0xc0b23b8 lvidx 131 nsp    219 amn (    -10.035   -10.035   -24.900) bmn (    -10.035   -10.035   -14.865) dmn (      0.000     0.000   -10.035) amx (     10.035    10.035    24.900) bmx (     10.035    10.036    24.899) dmx (     -0.000    -0.001     0.000)
        10.035                   source-shell0xc2d62d0 lvidx 111 nsp    567 amn (    -10.035   -10.035   -24.900) bmn (    -10.035   -10.035   -14.865) dmn (      0.000     0.000   -10.035) amx (     10.035    10.035    24.900) bmx (     10.035    10.035    14.865) dmx (      0.000     0.000    10.035)
        10.035               led-source-shell0xc3068f0 lvidx 100 nsp    567 amn (    -10.035   -10.035   -24.900) bmn (    -10.035   -10.035   -14.865) dmn (      0.000     0.000   -10.035) amx (     10.035    10.035    24.900) bmx (     10.035    10.035    14.865) dmx (      0.000     0.000    10.035)
       8.09241                    OcrGdsInLso0xbfa2190 lvidx  31 nsp    287 amn (    485.123 -1278.737  -242.962) bmn (    485.131 -1278.720  -251.054) dmn (     -0.008    -0.017     8.092) amx (    548.123 -1215.737   194.127) bmx (    548.131 -1215.720   195.139) dmx (     -0.008    -0.017    -1.012)
       7.54053                   pmt-hemi-vac0xc21e248 lvidx  46 nsp    665 amn (    -91.464   -91.464  -164.500) bmn (    -98.995   -99.003  -164.504) dmn (      7.531     7.539     0.004) amx (     91.464    91.464   128.000) bmx (     99.005    98.997   128.000) dmx (     -7.541    -7.533     0.000)
       5.01849                    source-assy0xc2d5d78 lvidx 112 nsp    480 amn (    -10.035   -10.035   -97.285) bmn (    -10.036   -10.036   -97.286) dmn (      0.001     0.001     0.001) amx (     10.035    10.035   102.303) bmx (     10.036    10.036   107.321) dmx (     -0.001    -0.001    -5.018)
       5.01749            amcco60-source-assy0xc0b1df8 lvidx 132 nsp    480 amn (    -10.035   -10.035   -97.285) bmn (    -10.036   -10.036   -97.286) dmn (      0.001     0.001     0.001) amx (     10.035    10.035   102.303) bmx (     10.036    10.036   107.320) dmx (     -0.001    -0.001    -5.017)
       5.01749                led-source-assy0xc3061d0 lvidx 105 nsp    480 amn (    -10.035   -10.035   -97.285) bmn (    -10.036   -10.036   -97.286) dmn (      0.001     0.001     0.001) amx (     10.035    10.035   102.303) bmx (     10.036    10.036   107.320) dmx (     -0.001    -0.001    -5.017)
             5                      LsoOflTnk0xc17d928 lvidx 140 nsp    315 amn (   -920.000  -920.000   -10.000) bmn (   -920.042  -920.000    -5.000) dmn (      0.042     0.000    -5.000) amx (    920.000   920.000   170.000) bmx (    920.000   920.031   170.057) dmx (      0.000    -0.031    -0.057)
       4.87451                 OcrGdsTfbInLso0xbfa2370 lvidx  30 nsp    464 amn (    484.123 -1279.737  -150.798) bmn (    484.130 -1279.740  -150.798) dmn (     -0.007     0.003    -0.000) amx (    549.123 -1214.737    82.816) bmx (    549.130 -1214.740    87.691) dmx (     -0.007     0.003    -4.875)
         3.882                   OcrCalLsoPrt0xc1076b0 lvidx  85 nsp    351 amn (    -98.000   -98.000     0.000) bmn (    -98.000   -98.000     3.882) dmn (      0.000     0.000    -3.882) amx (     98.000    98.000   214.596) bmx (     98.000    98.000   214.596) dmx (      0.000     0.000     0.000)
         1.782                 OcrGdsTfbInOav0xbf8f6c0 lvidx  39 nsp    255 amn (    484.123 -1279.737   -25.830) bmn (    484.128 -1279.740   -27.612) dmn (     -0.005     0.003     1.782) amx (    549.123 -1214.737    -6.894) bmx (    549.128 -1214.740    -6.797) dmx (     -0.005     0.003    -0.097)
       1.41823                 OcrCalLsoInOav0xc541388 lvidx  41 nsp    375 amn (   -728.306  1587.576   -49.501) bmn (   -728.313  1587.580   -50.919) dmn (      0.007    -0.004     1.418) amx (   -628.306  1687.576   -28.197) bmx (   -628.313  1687.580   -28.213) dmx (      0.007    -0.004     0.016)
       1.17236                 OcrGdsLsoInOav0xc354118 lvidx  40 nsp    510 amn (    466.623 -1297.237   -27.408) bmn (    466.616 -1297.240   -28.580) dmn (      0.007     0.003     1.172) amx (    566.623 -1197.237    -5.316) bmx (    566.616 -1197.240    -5.879) dmx (      0.007     0.003     0.564)
       1.01001                SstTopTshapeRib0xc272c80 lvidx  67 nsp    421 amn (  -1097.840   -50.000   -10.000) bmn (  -1097.840   -50.000   -10.000) dmn (      0.000     0.000     0.000) amx (   1097.840    50.000    10.000) bmx (   1096.830    50.000    10.000) dmx (      1.010     0.000     0.000)
      0.961575                    OcrGdsInOav0xc355130 lvidx  38 nsp    310 amn (    485.123 -1278.737   -26.619) bmn (    485.126 -1278.730   -27.581) dmn (     -0.003    -0.007     0.962) amx (    548.123 -1215.737    -6.894) bmx (    548.126 -1215.730    -6.849) dmx (     -0.003    -0.007    -0.045)
      0.799805                      near_rock0xc04ba08 lvidx 247 nsp    382 amn ( -25000.000-25000.000-12995.000) bmn ( -25000.000-25000.000-12994.200) dmn (      0.000     0.000    -0.800) amx (  25000.000 25000.000 25000.000) bmx (  25000.000 25000.000 25000.000) dmx (      0.000     0.000     0.000)
      0.685471                    OcrGdsInIav0xc405b10 lvidx  23 nsp    294 amn (    485.123 -1278.737   -37.074) bmn (    485.117 -1278.740   -37.759) dmn (      0.006     0.003     0.685) amx (    548.123 -1215.737   -19.499) bmx (    548.117 -1215.740   -19.750) dmx (      0.006     0.003     0.251)
           0.5            near_hall_top_dwarf0xc0316c8 lvidx  21 nsp    300 amn ( -30500.000 -7500.000 -7500.000) bmn ( -30500.500 -7500.390 -7500.290) dmn (      0.500     0.390     0.290) amx (  13500.000  7500.000  7500.000) bmx (  13500.000  7500.000  7500.000) dmx (      0.000     0.000     0.000)
      0.358002                near_span_hbeam0xc2a27d8 lvidx   9 nsp    450 amn (   -100.000 -5871.000  -147.000) bmn (   -100.358 -5871.000  -147.196) dmn (      0.358     0.000     0.196) amx (    100.000  5871.000   147.000) bmx (    100.358  5871.000   147.196) dmx (     -0.358     0.000    -0.196)
      0.247902                       pmt-hemi0xc0fed90 lvidx  47 nsp    674 amn (   -100.040  -100.040  -169.000) bmn (   -100.288  -100.288  -168.995) dmn (      0.248     0.248    -0.005) amx (    100.040   100.040   131.000) bmx (    100.288   100.288   131.000) dmx (     -0.248    -0.248     0.000)
        0.1313                   pmt-hemi-bot0xc22a958 lvidx  44 nsp    381 amn (    -98.143   -98.143   -99.000) bmn (    -98.143   -98.143   -99.000) dmn (      0.000     0.000     0.000) amx (     98.143    98.143   -13.000) bmx (     98.143    98.143   -12.869) dmx (     -0.000    -0.000    -0.131)
      0.119995                            oav0xc2ed7c8 lvidx  42 nsp    294 amn (  -2040.000 -2040.000 -1968.500) bmn (  -2040.070 -2040.120 -1968.500) dmn (      0.070     0.120     0.000) amx (   2040.000  2040.000  2126.121) bmx (   2039.930  2039.880  2126.210) dmx (      0.070     0.120    -0.089)
    2017-07-07 11:49:14.178 INFO  [3791106] [GScene::compareMeshes_GMeshBB@526] GScene::compareMeshes_GMeshBB num_meshes 249 cut 0.1 bbty CSG_BBOX_PARSURF num_discrepant 33 frac 0.13253
    Assertion failed: (0 && "GScene::init early exit for gltf==4 or gltf==44"), function init, file /Users/blyth/opticks/ggeo/GScene.cc, line 156.


bbox differences : review the top of the chart
-----------------------------------------------------


::

    op --gltf 4           # dump the compare meshes table

    2017-07-06 17:42:40.469 INFO  [3688598] [GScene::compareMeshes_GMeshBB@435] GScene::compareMeshes_GMeshBB num_meshes 249 cut 0.1 bbty CSG_BBOX_PARSURF parsurf_level 2 parsurf_target 200
     1   3869.75               RadialShieldUnit0xc3d7da8 lvidx  56 nsp    507                             intersection cylinder   nds[ 64]  4393 4394 4395 4396 4397 4398 4399 4400 4401 4402 ... 
     2   3407.72               SstBotCirRibBase0xc26e2d0 lvidx  65 nsp    324                          difference cylinder box3   nds[ 16]  4440 4441 4442 4443 4444 4445 4446 4447 6100 6101 ... 
     3   2074.65               SstTopCirRibBase0xc264f78 lvidx  69 nsp    352                        intersection cylinder box3   nds[ 16]  4465 4466 4467 4468 4469 4470 4471 4472 6125 6126 ... 
       ## big3 from tube.deltaphi
       
     4    345.51                SstTopRadiusRib0xc271720 lvidx  66 nsp    408                  difference box3 convexpolyhedron   nds[ 16]  4448 4449 4450 4451 4452 4453 4454 4455 6108 6109 ... 


     5       320                      SstTopHub0xc2643d8 lvidx  68 nsp    317                                    union cylinder   nds[  2]  4464 6124 . 

     6   28.0747              OcrGdsTfbInLsoOfl0xc2b5ba0 lvidx  83 nsp    243                          difference cylinder cone   nds[  2]  4515 6175 . 
     7   26.2183                   OcrGdsLsoPrt0xc104978 lvidx  81 nsp    342                    union difference cylinder cone   nds[  2]  4511 6171 . 
     8        20               headon-pmt-mount0xc2a7670 lvidx  55 nsp    365                         union difference cylinder   nds[ 12]  4357 4364 4371 4378 4385 4392 6017 6024 6031 6038 ... 
     9        12           near_side_long_hbeam0xbf3b5d0 lvidx  17 nsp    450                                        union box3   nds[  8]  2436 2437 2615 2616 2794 2795 2973 2974 . 
    10    10.035                   weight-shell0xc307920 lvidx 103 nsp    567                            union zsphere cylinder   nds[ 36]  4543 4547 4558 4562 4591 4595 4631 4635 4646 4650 ... 
    11    10.035        AmCCo60AcrylicContainer0xc0b23b8 lvidx 131 nsp    219                             union sphere cylinder   nds[  6]  4567 4655 4737 6227 6315 6397 . 
    12    10.035                   source-shell0xc2d62d0 lvidx 111 nsp    567                            union zsphere cylinder   nds[  6]  4552 4640 4722 6212 6300 6382 . 
    13    10.035               led-source-shell0xc3068f0 lvidx 100 nsp    567                            union zsphere cylinder   nds[  6]  4541 4629 4711 6201 6289 6371 . 
    14   8.09241                    OcrGdsInLso0xbfa2190 lvidx  31 nsp    287             intersection difference cylinder cone   nds[  2]  3168 4828 . 
    15   7.54053                   pmt-hemi-vac0xc21e248 lvidx  46 nsp    665                union intersection sphere cylinder   nds[672]  3200 3206 3212 3218 3224 3230 3236 3242 3248 3254 ... 
    16   5.01849                    source-assy0xc2d5d78 lvidx 112 nsp    480                            union zsphere cylinder   nds[  6]  4551 4639 4721 6211 6299 6381 . 
    17   5.01749            amcco60-source-assy0xc0b1df8 lvidx 132 nsp    480                            union zsphere cylinder   nds[  6]  4566 4654 4736 6226 6314 6396 . 
    18   5.01749                led-source-assy0xc3061d0 lvidx 105 nsp    480                            union zsphere cylinder   nds[  6]  4540 4628 4710 6200 6288 6370 . 
    19         5                      LsoOflTnk0xc17d928 lvidx 140 nsp    315                       union intersection cylinder   nds[  2]  4606 6266 . 
    20   4.87451                 OcrGdsTfbInLso0xbfa2370 lvidx  30 nsp    464             intersection difference cylinder cone   nds[  2]  3167 4827 . 
    21     3.882                   OcrCalLsoPrt0xc1076b0 lvidx  85 nsp    351                    union difference cylinder cone   nds[  2]  4517 6177 . 
    22     1.782                 OcrGdsTfbInOav0xbf8f6c0 lvidx  39 nsp    255             intersection difference cylinder cone   nds[  2]  3196 4856 . 
    23   1.41823                 OcrCalLsoInOav0xc541388 lvidx  41 nsp    375             intersection difference cylinder cone   nds[  2]  3198 4858 . 
    24   1.17236                 OcrGdsLsoInOav0xc354118 lvidx  40 nsp    510             intersection difference cylinder cone   nds[  2]  3195 4855 . 
    25   1.01001                SstTopTshapeRib0xc272c80 lvidx  67 nsp    421                          difference cylinder box3   nds[ 16]  4456 4457 4458 4459 4460 4461 4462 4463 6116 6117 ... 
    26  0.961575                    OcrGdsInOav0xc355130 lvidx  38 nsp    310             intersection difference cylinder cone   nds[  2]  3197 4857 . 
    27  0.799805                      near_rock0xc04ba08 lvidx 247 nsp    382                                   difference box3   nds[  1]  1 . 
    28  0.685471                    OcrGdsInIav0xc405b10 lvidx  23 nsp    294             intersection difference cylinder cone   nds[  2]  3160 4820 . 
    29       0.5            near_hall_top_dwarf0xc0316c8 lvidx  21 nsp    300                                        union box3   nds[  1]  2 . 
    30  0.358002                near_span_hbeam0xc2a27d8 lvidx   9 nsp    450                                        union box3   nds[ 18]  2359 2360 2432 2433 2434 2435 2611 2612 2613 2614 ... 
    31  0.247902                       pmt-hemi0xc0fed90 lvidx  47 nsp    674                union intersection sphere cylinder   nds[672]  3199 3205 3211 3217 3223 3229 3235 3241 3247 3253 ... 
    32    0.1313                   pmt-hemi-bot0xc22a958 lvidx  44 nsp    381                                difference zsphere   nds[672]  3202 3208 3214 3220 3226 3232 3238 3244 3250 3256 ... 
    33  0.119995                            oav0xc2ed7c8 lvidx  42 nsp    294                               union cylinder cone   nds[  2]  3156 4816 . 
    2017-07-06 17:42:40.530 INFO  [3688598] [GScene::compareMeshes_GMeshBB@518] GScene::compareMeshes_GMeshBB num_meshes 249 cut 0.1 bbty CSG_BBOX_PARSURF num_discrepant 33 frac 0.13253
    Assertion failed: (0 && "GScene::init early exit for gltf==4"), function init, file /Users/blyth/opticks/ggeo/GScene.cc, line 156.



lvidx66_again (4)
-------------------------------

* :doc:`lvidx66_again`
* suspect this issue is related to the trapezoid(convexpolyhedron) and manual bbox that they force



lvidx_65_69_56_tube_deltaphi (1,2,3)  HAVE BEEN KNOCKED DONE TABLE
--------------------------------------------------------------------

* DONE : implement tube deltaphi via intersect with CSG_SEGMENT

* :doc:`lvid65`

* :doc:`lvidx_65_69_56_tube_deltaphi`




lvidx68 (5) + lvidx55 (8) + lvidx17 (9) + lvidx103 (10) + lvidx131 (11)
-------------------------------------------------------------------------

* g4poly bug ? missing top cylinder 

* :doc:`lvidx68` TODO : difference out the inner cy

* :doc:`lvidx68` TODO : see if cleaved meshes are getting lost ?, check the G4DAE

* :doc:`lvidx17` g4poly (coincident union) misses T-bottom  of the H girder

* :doc:`lvidx103` parsurf bbox bigger in z, g4poly bb misses both zsphere ends of capsule

  * TODO: make zsphere endcaps ON the default with flags to switch them off
  * TODO: investigate no-shows in gltf viz

* :doc:`lvidx131` parsurf bbox bigger in z, g4poly bb misses one sphere end of capsule

These 4 look to be the same issue::

    10    10.035                   weight-shell0xc307920 lvidx 103 nsp    567                            union zsphere cylinder   nds[ 36]  4543 4547 4558 4562 4591 4595 4631 4635 4646 4650 ... 
    11    10.035        AmCCo60AcrylicContainer0xc0b23b8 lvidx 131 nsp    219                             union sphere cylinder   nds[  6]  4567 4655 4737 6227 6315 6397 . 
    12    10.035                   source-shell0xc2d62d0 lvidx 111 nsp    567                            union zsphere cylinder   nds[  6]  4552 4640 4722 6212 6300 6382 . 
    13    10.035               led-source-shell0xc3068f0 lvidx 100 nsp    567                            union zsphere cylinder   nds[  6]  4541 4629 4711 6201 6289 6371 . 

Probably these too::

    16   5.01849                    source-assy0xc2d5d78 lvidx 112 nsp    480                            union zsphere cylinder   nds[  6]  4551 4639 4721 6211 6299 6381 . 
    17   5.01749            amcco60-source-assy0xc0b1df8 lvidx 132 nsp    480                            union zsphere cylinder   nds[  6]  4566 4654 4736 6226 6314 6396 . 
    18   5.01749                led-source-assy0xc3061d0 lvidx 105 nsp    480                            union zsphere cylinder   nds[  6]  4540 4628 4710 6200 6288 6370 . 

    19         5                      LsoOflTnk0xc17d928 lvidx 140 nsp    315                       union intersection cylinder   nds[  2]  4606 6266 . 

* :doc:`lvidx140` (19)  g4poly misses some middle cylinders and the very bottom one



lvidx83 (6) + lvidx81 (7) both coincidence artifacts between cylinders and cones
------------------------------------------------------------------------------------

* :doc:`lvidx83`

* TODO : extend uncoincidence to handle "dicyco" ? when co-cy-endcaps are coincident

* :doc:`lvidx81`

* TODO : extend uncoincidence to handle uncycodi with left-right coincidence



lvidx31 (14) lvidx30 (20):  difference of large thin-z cones intersecting with cylinder : worst case for parsurf bb precision
--------------------------------------------------------------------------------------------------------------------

* :doc:`lvidx31`


* :doc:`lvidx30`

  * parsurf.bb.max.z is -4.87mm lower : but poor precision with ythis geometry 

    op --dlv30 --gltf 3  ## looks fine raytrace matching g4poly 



lvidx46 (15) 
----------------

* :doc:`lvidx46`


Unusually an xy discrep, parsurf ~symmetrically slimmer in xy

* my analytic bbox matches closely the g4poly one 
* slimmer parsurf could be from my  OpenMesh join kludge
* dont think there is issue with underlying CSG 

* familar PMT shape, intersection of three z-spheres and cylinder, 
  xy dimension comes from the intersect position of the middle two z-spheres  (b and c)


::

    op --dlv46 --gltf 3
         g4poly mesh extends lower : is there a multi transform mis interpretation ?

         * ~/opticks_refs/lvidx46_gltf_3_pmthemivac_is_there_a_multitransform_mis_interpretation.png 




lvid 39 : FIXED : 1214.74 OcrGdsTfbInLso0xbfa2370 cone-z should be centered
---------------------------------------------------------------------------------

* :doc:`lvid30_cone_z_misinterpretation`

lvid 185 : FAILED POLY false alarm
---------------------------------------------

* :doc:`lvid185`

lvid_41_40_39_23_38 : five with no par surface points : ~FIXED by move to adaptive
---------------------------------------------------------------------------------------

* :doc:`lvid_41_40_39_23_38`


try comparing CSG_BBOX_PARSURF against CSG_BBOX_G4POLY
--------------------------------------------------------

::
    
    2017-07-06 13:24:34.087 INFO  [3590380] [GScene::compareMeshes_GMeshBB@435] GScene::compareMeshes_GMeshBB num_meshes 249 cut 0.1 bbty CSG_BBOX_PARSURF

       3869.75               RadialShieldUnit0xc3d7da8 lvidx  56 amn (  -2262.150 -2262.150  -498.500) bmn (   1607.600     0.000  -498.500) dmn (  -3869.750 -2262.150     0.000) amx (   2262.150  2262.150   498.500) bmx (   2262.150  1589.370   498.500) dmx (      0.000   672.780     0.000)
       3407.72               SstBotCirRibBase0xc26e2d0 lvidx  65 amn (  -2000.000 -2000.000  -215.000) bmn (   1407.720    12.467  -215.000) dmn (  -3407.720 -2012.468     0.000) amx (      0.000  2000.000   215.000) bmx (   1998.360  1404.240   215.000) dmx (  -1998.360   595.760     0.000)
       2074.65               SstTopCirRibBase0xc264f78 lvidx  69 amn (  -1220.000 -1220.000  -115.945) bmn (    854.653    10.020  -115.945) dmn (  -2074.653 -1230.020     0.000) amx (   1220.000  1220.000   115.945) bmx (   1218.680   854.688   115.945) dmx (      1.320   365.312     0.000)

       1687.58                 OcrCalLsoInOav0xc541388 lvidx  41 amn (      0.000     0.000     0.000) bmn (   -728.313  1587.580   -50.919) dmn (    728.313 -1587.580    50.919) amx (      0.000     0.000     0.000) bmx (   -628.313  1687.580   -28.213) dmx (    628.313 -1687.580    28.213)
       1297.24                 OcrGdsLsoInOav0xc354118 lvidx  40 amn (      0.000     0.000     0.000) bmn (    466.616 -1297.240   -28.580) dmn (   -466.616  1297.240    28.580) amx (      0.000     0.000     0.000) bmx (    566.616 -1197.240    -5.879) dmx (   -566.616  1197.240     5.879)
       1279.74                 OcrGdsTfbInOav0xbf8f6c0 lvidx  39 amn (      0.000     0.000     0.000) bmn (    484.128 -1279.740   -27.612) dmn (   -484.128  1279.740    27.612) amx (      0.000     0.000     0.000) bmx (    549.128 -1214.740    -6.797) dmx (   -549.128  1214.740     6.797)
       1278.74                    OcrGdsInIav0xc405b10 lvidx  23 amn (      0.000     0.000     0.000) bmn (    485.117 -1278.740   -37.759) dmn (   -485.117  1278.740    37.759) amx (      0.000     0.000     0.000) bmx (    548.117 -1215.740   -19.750) dmx (   -548.117  1215.740    19.750)
       1278.73                    OcrGdsInOav0xc355130 lvidx  38 amn (      0.000     0.000     0.000) bmn (    485.126 -1278.730   -27.581) dmn (   -485.126  1278.730    27.581) amx (      0.000     0.000     0.000) bmx (    548.126 -1215.730    -6.849) dmx (   -548.126  1215.730     6.849)

        345.51                SstTopRadiusRib0xc271720 lvidx  66 amn (   -340.000   -10.000 -1114.250) bmn (   -345.510   -10.000 -1114.250) dmn (      5.510     0.000     0.000) amx (      0.000    10.000  1114.250) bmx (    345.510    10.000  1114.250) dmx (   -345.510     0.000     0.000)
           320                      SstTopHub0xc2643d8 lvidx  68 amn (   -220.500  -220.500  -340.000) bmn (   -220.500  -220.500  -340.000) dmn (      0.000     0.000     0.000) amx (    220.500   220.500     0.000) bmx (    220.500   220.500  -320.000) dmx (      0.000     0.000   320.000)
       115.448                    OcrGdsInLso0xbfa2190 lvidx  31 amn (    485.123 -1278.737  -135.606) bmn (    485.131 -1278.720  -251.054) dmn (     -0.008    -0.017   115.448) amx (    548.123 -1215.737   109.777) bmx (    548.131 -1215.720   195.139) dmx (     -0.008    -0.017   -85.362)
        29.001                   pmt-hemi-vac0xc21e248 lvidx  46 amn (    -70.004   -70.004  -164.500) bmn (    -98.995   -99.003  -164.504) dmn (     28.992    29.000     0.004) amx (     70.004    70.004   128.000) bmx (     99.005    98.997   128.000) dmx (    -29.001   -28.993     0.000)
       28.1631                       pmt-hemi0xc0fed90 lvidx  47 amn (    -72.125   -72.125  -169.000) bmn (   -100.288  -100.288  -168.995) dmn (     28.163    28.163    -0.005) amx (     72.125    72.125   131.000) bmx (    100.288   100.288   131.000) dmx (    -28.163   -28.163     0.000)
       28.0747              OcrGdsTfbInLsoOfl0xc2b5ba0 lvidx  83 amn (    -32.500   -32.500  -247.488) bmn (    -32.500   -32.500  -219.413) dmn (      0.000     0.000   -28.075) amx (     32.500    32.500   247.488) bmx (     32.500    32.500   247.488) dmx (      0.000     0.000    -0.000)
       26.2183                   OcrGdsLsoPrt0xc104978 lvidx  81 amn (    -98.000   -98.000     0.000) bmn (    -98.000   -98.000    26.218) dmn (      0.000     0.000   -26.218) amx (     98.000    98.000   214.596) bmx (     98.000    98.000   214.596) dmx (      0.000     0.000     0.000)
       22.9608                 OcrGdsTfbInLso0xbfa2370 lvidx  30 amn (    484.123 -1279.737  -150.798) bmn (    484.130 -1279.740  -150.798) dmn (     -0.007     0.003    -0.000) amx (    549.123 -1214.737    64.730) bmx (    549.130 -1214.740    87.691) dmx (     -0.007     0.003   -22.961)
            20               headon-pmt-mount0xc2a7670 lvidx  55 amn (    -51.500   -51.500  -120.000) bmn (    -36.850   -36.850  -100.000) dmn (    -14.650   -14.650   -20.000) amx (     51.500    51.500   100.000) bmx (     36.850    36.850   100.000) dmx (     14.650    14.650     0.000)
            12           near_side_long_hbeam0xbf3b5d0 lvidx  17 amn (  -2000.000  -100.000  -147.000) bmn (  -2000.000   -99.876  -135.000) dmn (      0.000    -0.124   -12.000) amx (   2000.000   100.000   147.000) bmx (   2000.070   100.124   146.908) dmx (     -0.070    -0.124     0.092)
        10.035               led-source-shell0xc3068f0 lvidx 100 amn (    -10.035   -10.035   -24.900) bmn (    -10.035   -10.035   -14.865) dmn (      0.000     0.000   -10.035) amx (     10.035    10.035    24.900) bmx (     10.035    10.035    14.865) dmx (      0.000     0.000    10.035)
        10.035        AmCCo60AcrylicContainer0xc0b23b8 lvidx 131 amn (    -10.035   -10.035   -24.900) bmn (    -10.035   -10.035   -14.865) dmn (      0.000     0.000   -10.035) amx (     10.035    10.035    24.900) bmx (     10.035    10.036    24.899) dmx (     -0.000    -0.001     0.000)
        10.035                   source-shell0xc2d62d0 lvidx 111 amn (    -10.035   -10.035   -24.900) bmn (    -10.035   -10.035   -14.865) dmn (      0.000     0.000   -10.035) amx (     10.035    10.035    24.900) bmx (     10.035    10.035    14.865) dmx (      0.000     0.000    10.035)
        10.035                   weight-shell0xc307920 lvidx 103 amn (    -10.035   -10.035   -28.510) bmn (    -10.035   -10.035   -18.475) dmn (      0.000     0.000   -10.035) amx (     10.035    10.035    28.510) bmx (     10.035    10.035    18.475) dmx (      0.000     0.000    10.035)
       5.01849                    source-assy0xc2d5d78 lvidx 112 amn (    -10.035   -10.035   -97.285) bmn (    -10.036   -10.036   -97.286) dmn (      0.001     0.001     0.001) amx (     10.035    10.035   102.303) bmx (     10.036    10.036   107.321) dmx (     -0.001    -0.001    -5.018)
       5.01749                led-source-assy0xc3061d0 lvidx 105 amn (    -10.035   -10.035   -97.285) bmn (    -10.036   -10.036   -97.286) dmn (      0.001     0.001     0.001) amx (     10.035    10.035   102.303) bmx (     10.036    10.036   107.320) dmx (     -0.001    -0.001    -5.017)
       5.01749            amcco60-source-assy0xc0b1df8 lvidx 132 amn (    -10.035   -10.035   -97.285) bmn (    -10.036   -10.036   -97.286) dmn (      0.001     0.001     0.001) amx (     10.035    10.035   102.303) bmx (     10.036    10.036   107.320) dmx (     -0.001    -0.001    -5.017)
             5                      LsoOflTnk0xc17d928 lvidx 140 amn (   -920.000  -920.000   -10.000) bmn (   -920.042  -920.000    -5.000) dmn (      0.042     0.000    -5.000) amx (    920.000   920.000   170.000) bmx (    920.000   920.031   170.057) dmx (      0.000    -0.031    -0.057)
         3.882                   OcrCalLsoPrt0xc1076b0 lvidx  85 amn (    -98.000   -98.000     0.000) bmn (    -98.000   -98.000     3.882) dmn (      0.000     0.000    -3.882) amx (     98.000    98.000   214.596) bmx (     98.000    98.000   214.596) dmx (      0.000     0.000     0.000)
       1.01001                SstTopTshapeRib0xc272c80 lvidx  67 amn (  -1097.840   -50.000   -10.000) bmn (  -1097.840   -50.000   -10.000) dmn (      0.000     0.000     0.000) amx (   1097.840    50.000    10.000) bmx (   1096.830    50.000    10.000) dmx (      1.010     0.000     0.000)
      0.799805                      near_rock0xc04ba08 lvidx 247 amn ( -25000.000-25000.000-12995.000) bmn ( -25000.000-25000.000-12994.200) dmn (      0.000     0.000    -0.800) amx (  25000.000 25000.000 25000.000) bmx (  25000.000 25000.000 25000.000) dmx (      0.000     0.000     0.000)
           0.5            near_hall_top_dwarf0xc0316c8 lvidx  21 amn ( -30500.000 -7500.000 -7500.000) bmn ( -30500.500 -7500.390 -7500.290) dmn (      0.500     0.390     0.290) amx (  13500.000  7500.000  7500.000) bmx (  13500.000  7500.000  7500.000) dmx (      0.000     0.000     0.000)
      0.358002                near_span_hbeam0xc2a27d8 lvidx   9 amn (   -100.000 -5871.000  -147.000) bmn (   -100.358 -5871.000  -147.196) dmn (      0.358     0.000     0.196) amx (    100.000  5871.000   147.000) bmx (    100.358  5871.000   147.196) dmx (     -0.358     0.000    -0.196)
        0.1313                   pmt-hemi-bot0xc22a958 lvidx  44 amn (    -98.143   -98.143   -99.000) bmn (    -98.143   -98.143   -99.000) dmn (      0.000     0.000     0.000) amx (     98.143    98.143   -13.000) bmx (     98.143    98.143   -12.869) dmx (     -0.000    -0.000    -0.131)
      0.119995                            oav0xc2ed7c8 lvidx  42 amn (  -2040.000 -2040.000 -1968.500) bmn (  -2040.070 -2040.120 -1968.500) dmn (      0.070     0.120     0.000) amx (   2040.000  2040.000  2126.121) bmx (   2039.930  2039.880  2126.210) dmx (      0.070     0.120    -0.089)
    2017-07-06 13:24:34.126 INFO  [3590380] [GScene::compareMeshes_GMeshBB@498] GScene::compareMeshes_GMeshBB num_meshes 249 cut 0.1 bbty CSG_BBOX_PARSURF num_discrepant 33 frac 0.13253
    Assertion failed: (0 && "GScene::init early exit for gltf==4"), function init, file /Users/blyth/opticks/ggeo/GScene.cc, line 156.
    /Users/blyth/opticks/bin/op.sh: line 633: 17204 Abort trap: 6           /usr/local/opticks/lib/OKTest --gltf 4





fixing the placed bbox difference bug moves lvid 30 down the chart
-----------------------------------------------------------------------

::

    // vim :set nowrap
    op --gltf 4

    2017-07-05 11:42:48.689 INFO  [3301919] [GScene::compareMeshes_GMeshBB@396] GScene::compareMeshes_GMeshBB num_meshes 249 cut 0.1 with_csg_bbox YES (csg bbox avoids ana branch polygonization issues) 
       12005.8                      near_rock0xc04ba08 lvidx 247 amn ( -25000.000-25000.000-25000.000) bmn ( -25000.000-25000.000-12994.200) dmn (      0.000     0.000-12005.800) amx (  25000.000 25000.000 25000.000) bmx (  25000.000 25000.000 25000.000) dmx (      0.000     0.000     0.000)
       3869.75               RadialShieldUnit0xc3d7da8 lvidx  56 amn (  -2262.150 -2262.150  -498.500) bmn (   1607.600     0.000  -498.500) dmn (  -3869.750 -2262.150     0.000) amx (   2262.150  2262.150   498.500) bmx (   2262.150  1589.370   498.500) dmx (      0.000   672.780     0.000)
       3407.72               SstBotCirRibBase0xc26e2d0 lvidx  65 amn (  -2000.000 -2000.000  -215.000) bmn (   1407.720    12.467  -215.000) dmn (  -3407.720 -2012.468     0.000) amx (   2000.000  2000.000   215.000) bmx (   1998.360  1404.240   215.000) dmx (      1.640   595.760     0.000)
       2074.65               SstTopCirRibBase0xc264f78 lvidx  69 amn (  -1220.000 -1220.000  -115.945) bmn (    854.653    10.020  -115.945) dmn (  -2074.653 -1230.020     0.000) amx (   1220.000  1220.000   115.945) bmx (   1218.680   854.688   115.945) dmx (      1.320   365.312     0.000)
           320                      SstTopHub0xc2643d8 lvidx  68 amn (   -220.500  -220.500  -340.000) bmn (   -220.500  -220.500  -340.000) dmn (      0.000     0.000     0.000) amx (    220.500   220.500     0.000) bmx (    220.500   220.500  -320.000) dmx (      0.000     0.000   320.000)
       84.5234                 OcrCalLsoInOav0xc541388 lvidx  41 amn (   -728.306  1587.576   -56.310) bmn (   -728.313  1587.580   -50.919) dmn (      0.007    -0.004    -5.391) amx (   -628.306  1687.576    56.310) bmx (   -628.313  1687.580   -28.213) dmx (      0.007    -0.004    84.523)
       64.4695                    OcrGdsInIav0xc405b10 lvidx  23 amn (    485.123 -1278.737   -44.720) bmn (    485.117 -1278.740   -37.759) dmn (      0.006     0.003    -6.960) amx (    548.123 -1215.737    44.720) bmx (    548.117 -1215.740   -19.750) dmx (      0.006     0.003    64.470)
        63.159                    OcrGdsInLso0xbfa2190 lvidx  31 amn (    485.123 -1278.737  -258.298) bmn (    485.131 -1278.720  -251.054) dmn (     -0.008    -0.017    -7.244) amx (    548.123 -1215.737   258.298) bmx (    548.131 -1215.720   195.139) dmx (     -0.008    -0.017    63.159)
       63.1589                    OcrGdsInOav0xc355130 lvidx  38 amn (    485.123 -1278.737   -56.310) bmn (    485.126 -1278.730   -27.581) dmn (     -0.003    -0.007   -28.730) amx (    548.123 -1215.737    56.310) bmx (    548.126 -1215.730    -6.849) dmx (     -0.003    -0.007    63.159)
       63.1071                 OcrGdsTfbInLso0xbfa2370 lvidx  30 amn (    484.123 -1279.737  -150.798) bmn (    484.130 -1279.740  -150.798) dmn (     -0.007     0.003    -0.000) amx (    549.123 -1214.737   150.798) bmx (    549.130 -1214.740    87.691) dmx (     -0.007     0.003    63.107)
       63.1071                 OcrGdsTfbInOav0xbf8f6c0 lvidx  39 amn (    484.123 -1279.737   -56.310) bmn (    484.128 -1279.740   -27.612) dmn (     -0.005     0.003   -28.698) amx (    549.123 -1214.737    56.310) bmx (    549.128 -1214.740    -6.797) dmx (     -0.005     0.003    63.107)
       62.1898                 OcrGdsLsoInOav0xc354118 lvidx  40 amn (    466.623 -1297.237   -56.310) bmn (    466.616 -1297.240   -28.580) dmn (      0.007     0.003   -27.730) amx (    566.623 -1197.237    56.310) bmx (    566.616 -1197.240    -5.879) dmx (      0.007     0.003    62.190)
       29.8624               pmt-hemi-cathode0xc2f1ce8 lvidx  43 amn (   -128.000  -128.000    56.000) bmn (    -98.138   -98.147    55.996) dmn (    -29.862   -29.853     0.004) amx (    128.000   128.000   128.000) bmx (     98.148    98.139   128.000) dmx (     29.852    29.861     0.000)
       28.0747              OcrGdsTfbInLsoOfl0xc2b5ba0 lvidx  83 amn (    -32.500   -32.500  -247.488) bmn (    -32.500   -32.500  -219.413) dmn (      0.000     0.000   -28.075) amx (     32.500    32.500   247.488) bmx (     32.500    32.500   247.488) dmx (      0.000     0.000    -0.000)
       26.2183                   OcrGdsLsoPrt0xc104978 lvidx  81 amn (    -98.000   -98.000     0.000) bmn (    -98.000   -98.000    26.218) dmn (      0.000     0.000   -26.218) amx (     98.000    98.000   214.596) bmx (     98.000    98.000   214.596) dmx (      0.000     0.000     0.000)
            20               headon-pmt-mount0xc2a7670 lvidx  55 amn (    -51.500   -51.500  -120.000) bmn (    -36.850   -36.850  -100.000) dmn (    -14.650   -14.650   -20.000) amx (     51.500    51.500   100.000) bmx (     36.850    36.850   100.000) dmx (     14.650    14.650     0.000)
            12           near_side_long_hbeam0xbf3b5d0 lvidx  17 amn (  -2000.000  -100.000  -147.000) bmn (  -2000.000   -99.876  -135.000) dmn (      0.000    -0.124   -12.000) amx (   2000.000   100.000   147.000) bmx (   2000.070   100.124   146.908) dmx (     -0.070    -0.124     0.092)
        10.035        AmCCo60AcrylicContainer0xc0b23b8 lvidx 131 amn (    -10.035   -10.035   -24.900) bmn (    -10.035   -10.035   -14.865) dmn (      0.000     0.000   -10.035) amx (     10.035    10.035    24.900) bmx (     10.035    10.036    24.899) dmx (     -0.000    -0.001     0.000)
        10.035                   weight-shell0xc307920 lvidx 103 amn (    -10.035   -10.035   -28.510) bmn (    -10.035   -10.035   -18.475) dmn (      0.000     0.000   -10.035) amx (     10.035    10.035    28.510) bmx (     10.035    10.035    18.475) dmx (      0.000     0.000    10.035)
        10.035                   source-shell0xc2d62d0 lvidx 111 amn (    -10.035   -10.035   -24.900) bmn (    -10.035   -10.035   -14.865) dmn (      0.000     0.000   -10.035) amx (     10.035    10.035    24.900) bmx (     10.035    10.035    14.865) dmx (      0.000     0.000    10.035)
        10.035               led-source-shell0xc3068f0 lvidx 100 amn (    -10.035   -10.035   -24.900) bmn (    -10.035   -10.035   -14.865) dmn (      0.000     0.000   -10.035) amx (     10.035    10.035    24.900) bmx (     10.035    10.035    14.865) dmx (      0.000     0.000    10.035)
             5                      LsoOflTnk0xc17d928 lvidx 140 amn (   -920.000  -920.000   -10.000) bmn (   -920.042  -920.000    -5.000) dmn (      0.042     0.000    -5.000) amx (    920.000   920.000   170.000) bmx (    920.000   920.031   170.057) dmx (      0.000    -0.031    -0.057)
         3.882                   OcrCalLsoPrt0xc1076b0 lvidx  85 amn (    -98.000   -98.000     0.000) bmn (    -98.000   -98.000     3.882) dmn (      0.000     0.000    -3.882) amx (     98.000    98.000   214.596) bmx (     98.000    98.000   214.596) dmx (      0.000     0.000     0.000)
         1.712                       pmt-hemi0xc0fed90 lvidx  47 amn (   -102.000  -102.000  -169.000) bmn (   -100.288  -100.288  -168.995) dmn (     -1.712    -1.712    -0.005) amx (    102.000   102.000   131.000) bmx (    100.288   100.288   131.000) dmx (      1.712     1.712     0.000)
       1.01001                SstTopTshapeRib0xc272c80 lvidx  67 amn (  -1097.840   -50.000   -10.000) bmn (  -1097.840   -50.000   -10.000) dmn (      0.000     0.000     0.000) amx (   1097.840    50.000    10.000) bmx (   1096.830    50.000    10.000) dmx (      1.010     0.000     0.000)
      0.857201                   pmt-hemi-bot0xc22a958 lvidx  44 amn (    -99.000   -99.000   -99.000) bmn (    -98.143   -98.143   -99.000) dmn (     -0.857    -0.857     0.000) amx (     99.000    99.000   -13.000) bmx (     98.143    98.143   -12.869) dmx (      0.857     0.857    -0.131)
           0.5            near_hall_top_dwarf0xc0316c8 lvidx  21 amn ( -30500.000 -7500.000 -7500.000) bmn ( -30500.500 -7500.390 -7500.290) dmn (      0.500     0.390     0.290) amx (  13500.000  7500.000  7500.000) bmx (  13500.000  7500.000  7500.000) dmx (      0.000     0.000     0.000)
      0.358002                near_span_hbeam0xc2a27d8 lvidx   9 amn (   -100.000 -5871.000  -147.000) bmn (   -100.358 -5871.000  -147.196) dmn (      0.358     0.000     0.196) amx (    100.000  5871.000   147.000) bmx (    100.358  5871.000   147.196) dmx (     -0.358     0.000    -0.196)
      0.119995                            oav0xc2ed7c8 lvidx  42 amn (  -2040.000 -2040.000 -1968.500) bmn (  -2040.070 -2040.120 -1968.500) dmn (      0.070     0.120     0.000) amx (   2040.000  2040.000  2126.121) bmx (   2039.930  2039.880  2126.210) dmx (      0.070     0.120    -0.089)
    2017-07-05 11:42:48.708 INFO  [3301919] [GScene::compareMeshes_GMeshBB@483] GScene::compareMeshes_GMeshBB num_meshes 249 cut 0.1 with_csg_bbox YES num_discrepant 29 frac 0.116466
    Assertion failed: (0 && "GScene::init early exit for gltf==4"), function init, file /Users/blyth/opticks/ggeo/GScene.cc, line 157.


tri.GMesh.bbox vs ana.CSG.bbox diff table
------------------------------------------------

* avoiding ana branch poly fails reducues discrepant meshes to ~12 percent

::

   // vim :set nowrap
   op --gltf 4

::

    2017-07-04 16:10:29.991 INFO  [3150221] [GScene::compareMeshes_GMeshBB@396] GScene::compareMeshes_GMeshBB num_meshes 249 cut 0.1 with_csg_bbox YES (csg bbox avoids ana branch polygonization issues) 
       12005.8                      near_rock0xc04ba08 lvidx 247 amn ( -25000.000-25000.000-25000.000) bmn ( -25000.000-25000.000-12994.200) dmn (      0.000     0.000-12005.800) amx (  25000.000 25000.000 25000.000) bmx (  25000.000 25000.000 25000.000) dmx (      0.000     0.000     0.000)
       3869.75               RadialShieldUnit0xc3d7da8 lvidx  56 amn (  -2262.150 -2262.150  -498.500) bmn (   1607.600     0.000  -498.500) dmn (  -3869.750 -2262.150     0.000) amx (   2262.150  2262.150   498.500) bmx (   2262.150  1589.370   498.500) dmx (      0.000   672.780     0.000)
       3407.72               SstBotCirRibBase0xc26e2d0 lvidx  65 amn (  -2000.000 -2000.000  -215.000) bmn (   1407.720    12.467  -215.000) dmn (  -3407.720 -2012.468     0.000) amx (   2000.000  2000.000   215.000) bmx (   1998.360  1404.240   215.000) dmx (      1.640   595.760     0.000)
       2074.65               SstTopCirRibBase0xc264f78 lvidx  69 amn (  -1220.000 -1220.000  -115.945) bmn (    854.653    10.020  -115.945) dmn (  -2074.653 -1230.020     0.000) amx (   1220.000  1220.000   115.945) bmx (   1218.680   854.688   115.945) dmx (      1.320   365.312     0.000)
       # lv:65,lv:69 known cause : missing tube deltaphi handling 

       1214.74                 OcrGdsTfbInLso0xbfa2370 lvidx  30 amn (      0.000 -1279.737  -150.798) bmn (    484.130 -1279.740  -150.798) dmn (   -484.130     0.003    -0.000) amx (    549.123     0.000   150.798) bmx (    549.130 -1214.740    87.691) dmx (     -0.007  1214.740    63.107)
           320                      SstTopHub0xc2643d8 lvidx  68 amn (   -220.500  -220.500  -340.000) bmn (   -220.500  -220.500  -340.000) dmn (      0.000     0.000     0.000) amx (    220.500   220.500     0.000) bmx (    220.500   220.500  -320.000) dmx (      0.000     0.000   320.000)
       84.5234                 OcrCalLsoInOav0xc541388 lvidx  41 amn (   -728.306  1587.576   -56.310) bmn (   -728.313  1587.580   -50.919) dmn (      0.007    -0.004    -5.391) amx (   -628.306  1687.576    56.310) bmx (   -628.313  1687.580   -28.213) dmx (      0.007    -0.004    84.523)
       64.4695                    OcrGdsInIav0xc405b10 lvidx  23 amn (    485.123 -1278.737   -44.720) bmn (    485.117 -1278.740   -37.759) dmn (      0.006     0.003    -6.960) amx (    548.123 -1215.737    44.720) bmx (    548.117 -1215.740   -19.750) dmx (      0.006     0.003    64.470)
        63.159                    OcrGdsInLso0xbfa2190 lvidx  31 amn (    485.123 -1278.737  -258.298) bmn (    485.131 -1278.720  -251.054) dmn (     -0.008    -0.017    -7.244) amx (    548.123 -1215.737   258.298) bmx (    548.131 -1215.720   195.139) dmx (     -0.008    -0.017    63.159)
       63.1589                    OcrGdsInOav0xc355130 lvidx  38 amn (    485.123 -1278.737   -56.310) bmn (    485.126 -1278.730   -27.581) dmn (     -0.003    -0.007   -28.730) amx (    548.123 -1215.737    56.310) bmx (    548.126 -1215.730    -6.849) dmx (     -0.003    -0.007    63.159)
       63.1071                 OcrGdsTfbInOav0xbf8f6c0 lvidx  39 amn (    484.123 -1279.737   -56.310) bmn (    484.128 -1279.740   -27.612) dmn (     -0.005     0.003   -28.698) amx (    549.123 -1214.737    56.310) bmx (    549.128 -1214.740    -6.797) dmx (     -0.005     0.003    63.107)
       62.1898                 OcrGdsLsoInOav0xc354118 lvidx  40 amn (    466.623 -1297.237   -56.310) bmn (    466.616 -1297.240   -28.580) dmn (      0.007     0.003   -27.730) amx (    566.623 -1197.237    56.310) bmx (    566.616 -1197.240    -5.879) dmx (      0.007     0.003    62.190)
       55.9963               pmt-hemi-cathode0xc2f1ce8 lvidx  43 amn (   -128.000  -128.000     0.000) bmn (    -98.138   -98.147    55.996) dmn (    -29.862   -29.853   -55.996) amx (    128.000   128.000   128.000) bmx (     98.148    98.139   128.000) dmx (     29.852    29.861     0.000)
       28.0747              OcrGdsTfbInLsoOfl0xc2b5ba0 lvidx  83 amn (    -32.500   -32.500  -247.488) bmn (    -32.500   -32.500  -219.413) dmn (      0.000     0.000   -28.075) amx (     32.500    32.500   247.488) bmx (     32.500    32.500   247.488) dmx (      0.000     0.000    -0.000)
       26.2183                   OcrGdsLsoPrt0xc104978 lvidx  81 amn (    -98.000   -98.000     0.000) bmn (    -98.000   -98.000    26.218) dmn (      0.000     0.000   -26.218) amx (     98.000    98.000   214.596) bmx (     98.000    98.000   214.596) dmx (      0.000     0.000     0.000)
            20               headon-pmt-mount0xc2a7670 lvidx  55 amn (    -51.500   -51.500  -120.000) bmn (    -36.850   -36.850  -100.000) dmn (    -14.650   -14.650   -20.000) amx (     51.500    51.500   100.000) bmx (     36.850    36.850   100.000) dmx (     14.650    14.650     0.000)
       12.8687                   pmt-hemi-bot0xc22a958 lvidx  44 amn (    -99.000   -99.000   -99.000) bmn (    -98.143   -98.143   -99.000) dmn (     -0.857    -0.857     0.000) amx (     99.000    99.000     0.000) bmx (     98.143    98.143   -12.869) dmx (      0.857     0.857    12.869)
            12           near_side_long_hbeam0xbf3b5d0 lvidx  17 amn (  -2000.000  -100.000  -147.000) bmn (  -2000.000   -99.876  -135.000) dmn (      0.000    -0.124   -12.000) amx (   2000.000   100.000   147.000) bmx (   2000.070   100.124   146.908) dmx (     -0.070    -0.124     0.092)
        10.035                   weight-shell0xc307920 lvidx 103 amn (    -10.035   -10.035   -28.510) bmn (    -10.035   -10.035   -18.475) dmn (      0.000     0.000   -10.035) amx (     10.035    10.035    28.510) bmx (     10.035    10.035    18.475) dmx (      0.000     0.000    10.035)
        10.035        AmCCo60AcrylicContainer0xc0b23b8 lvidx 131 amn (    -10.035   -10.035   -24.900) bmn (    -10.035   -10.035   -14.865) dmn (      0.000     0.000   -10.035) amx (     10.035    10.035    24.900) bmx (     10.035    10.036    24.899) dmx (     -0.000    -0.001     0.000)
        10.035                   source-shell0xc2d62d0 lvidx 111 amn (    -10.035   -10.035   -24.900) bmn (    -10.035   -10.035   -14.865) dmn (      0.000     0.000   -10.035) amx (     10.035    10.035    24.900) bmx (     10.035    10.035    14.865) dmx (      0.000     0.000    10.035)
        10.035               led-source-shell0xc3068f0 lvidx 100 amn (    -10.035   -10.035   -24.900) bmn (    -10.035   -10.035   -14.865) dmn (      0.000     0.000   -10.035) amx (     10.035    10.035    24.900) bmx (     10.035    10.035    14.865) dmx (      0.000     0.000    10.035)
             5                      LsoOflTnk0xc17d928 lvidx 140 amn (   -920.000  -920.000   -10.000) bmn (   -920.042  -920.000    -5.000) dmn (      0.042     0.000    -5.000) amx (    920.000   920.000   170.000) bmx (    920.000   920.031   170.057) dmx (      0.000    -0.031    -0.057)
         3.882                   OcrCalLsoPrt0xc1076b0 lvidx  85 amn (    -98.000   -98.000     0.000) bmn (    -98.000   -98.000     3.882) dmn (      0.000     0.000    -3.882) amx (     98.000    98.000   214.596) bmx (     98.000    98.000   214.596) dmx (      0.000     0.000     0.000)
         1.712                       pmt-hemi0xc0fed90 lvidx  47 amn (   -102.000  -102.000  -169.000) bmn (   -100.288  -100.288  -168.995) dmn (     -1.712    -1.712    -0.005) amx (    102.000   102.000   131.000) bmx (    100.288   100.288   131.000) dmx (      1.712     1.712     0.000)
       1.01001                SstTopTshapeRib0xc272c80 lvidx  67 amn (  -1097.840   -50.000   -10.000) bmn (  -1097.840   -50.000   -10.000) dmn (      0.000     0.000     0.000) amx (   1097.840    50.000    10.000) bmx (   1096.830    50.000    10.000) dmx (      1.010     0.000     0.000)
           0.5            near_hall_top_dwarf0xc0316c8 lvidx  21 amn ( -30500.000 -7500.000 -7500.000) bmn ( -30500.500 -7500.390 -7500.290) dmn (      0.500     0.390     0.290) amx (  13500.000  7500.000  7500.000) bmx (  13500.000  7500.000  7500.000) dmx (      0.000     0.000     0.000)
      0.358002                near_span_hbeam0xc2a27d8 lvidx   9 amn (   -100.000 -5871.000  -147.000) bmn (   -100.358 -5871.000  -147.196) dmn (      0.358     0.000     0.196) amx (    100.000  5871.000   147.000) bmx (    100.358  5871.000   147.196) dmx (     -0.358     0.000    -0.196)
      0.119995                            oav0xc2ed7c8 lvidx  42 amn (  -2040.000 -2040.000 -1968.500) bmn (  -2040.070 -2040.120 -1968.500) dmn (      0.070     0.120     0.000) amx (   2040.000  2040.000  2126.121) bmx (   2039.930  2039.880  2126.210) dmx (      0.070     0.120    -0.089)
    2017-07-04 16:10:30.011 INFO  [3150221] [GScene::compareMeshes_GMeshBB@483] GScene::compareMeshes_GMeshBB num_meshes 249 cut 0.1 with_csg_bbox YES num_discrepant 29 frac 0.116466
    Assertion failed: (0 && "GScene::init early exit for gltf==4"), function init, file /Users/blyth/opticks/ggeo/GScene.cc, line 157.




GMesh bbox diff table
-----------------------

* 38 percent of meshes have bbox discrep : by comparison with above, most of these are from ana branch poly fails


::

   // vim :set nowrap
   op --gltf 4

    2017-07-03 20:53:28.697 INFO  [2994395] [GScene::importMeshes@304] GScene::importMeshes DONE num_meshes 249
       3407.72               SstBotCirRibBase0xc26e2d0 lvidx  65 amn (  -2000.000 -2000.000  -215.000) bmn (   1407.720    12.467  -215.000) dmn (  -3407.720 -2012.468     0.000) amx (   2000.000  2000.000   215.000) bmx (   1998.360  1404.240   215.000) dmx (      1.640   595.760     0.000)
       2074.65               SstTopCirRibBase0xc264f78 lvidx  69 amn (  -1220.000 -1220.000  -115.945) bmn (    854.653    10.020  -115.945) dmn (  -2074.653 -1230.020     0.000) amx (   1220.000  1220.000   115.945) bmx (   1218.680   854.688   115.945) dmx (      1.320   365.312     0.000)
       ## top 2 : are due to need to add tube deltaphi 

       1214.74                 OcrGdsTfbInLso0xbfa2370 lvidx  30 amn (      0.000 -1279.737     0.000) bmn (    484.130 -1279.740  -150.798) dmn (   -484.130     0.003   150.798) amx (    549.123     0.000   150.798) bmx (    549.130 -1214.740    87.691) dmx (     -0.007  1214.740    63.107)
       ## FIXED : twas cone-z should be centered

        1155.6                       MOFTTube0xc046b40 lvidx 185 amn (    574.598   -29.010  -113.129) bmn (   -581.000  -581.000  -127.500) dmn (   1155.598   551.990    14.371) amx (    580.602    29.010   113.129) bmx (    581.000   581.000   127.500) dmx (     -0.398  -551.990   -14.371)
       ## LEAVE ASIS : just failed poly  

       503.343                        GDBTube0xc213f68 lvidx 171 amn (    248.968   -18.171   -97.799) bmn (   -254.375  -254.375  -100.190) dmn (    503.343   236.204     2.391) amx (    254.172    18.171    97.799) bmx (    254.375   254.375   100.190) dmx (     -0.203  -236.204    -2.391)
       494.793                      GdsOflTnk0xc3d5160 lvidx 142 amn (   -165.248  -165.248   -30.000) bmn (   -660.041  -660.030   -30.002) dmn (    494.793   494.782     0.002) amx (    659.559   165.248   225.000) bmx (    660.041   660.030   225.010) dmx (     -0.482  -494.782    -0.010)
       337.053                      IavTopRib0xbf8e168 lvidx  36 amn (   -373.143   -25.000   -54.500) bmn (   -710.196   -25.000   -54.500) dmn (    337.053     0.000     0.000) amx (    710.196    25.000    54.500) bmx (    710.196    25.000    54.500) dmx (      0.000     0.000     0.000)
           320                      SstTopHub0xc2643d8 lvidx  68 amn (   -220.500  -220.500  -340.000) bmn (   -220.500  -220.500  -340.000) dmn (      0.000     0.000     0.000) amx (    220.500   220.500     0.000) bmx (    220.500   220.500  -320.000) dmx (      0.000     0.000   320.000)
       251.054                    OcrGdsInLso0xbfa2190 lvidx  31 amn (    485.123 -1278.737     0.000) bmn (    485.131 -1278.720  -251.054) dmn (     -0.008    -0.017   251.054) amx (    548.123 -1215.737   258.298) bmx (    548.131 -1215.720   195.139) dmx (     -0.008    -0.017    63.159)
       210.937                    AdPmtCollar0xc2c5260 lvidx  48 amn (    104.937    -9.907    -6.350) bmn (   -106.000  -106.000    -6.350) dmn (    210.937    96.093     0.000) amx (    105.938     9.907     6.350) bmx (    106.000   106.000     6.350) dmx (     -0.062   -96.093     0.000)
       210.937                   pmt-top-ring0xc2f0608 lvidx 193 amn (    104.937    -5.634    -5.078) bmn (   -106.000  -106.000    -7.000) dmn (    210.937   100.366     1.922) amx (    105.937     5.634     5.078) bmx (    106.000   106.000     7.000) dmx (     -0.063  -100.366    -1.922)
       165.703               RadialShieldUnit0xc3d7da8 lvidx  56 amn (   1754.556   -92.444  -424.938) bmn (   1607.600     0.000  -498.500) dmn (    146.956   -92.444    73.562) amx (   2260.600  1423.667   424.938) bmx (   2262.150  1589.370   498.500) dmx (     -1.550  -165.703   -73.562)
       134.523                 OcrCalLsoInOav0xc541388 lvidx  41 amn (   -728.306  1587.576     0.000) bmn (   -728.313  1587.580   -50.919) dmn (      0.007    -0.004    50.919) amx (   -628.306  1687.576   106.310) bmx (   -628.313  1687.580   -28.213) dmx (      0.007    -0.004   134.523)
       113.159                    OcrGdsInOav0xc355130 lvidx  38 amn (    485.123 -1278.737     0.000) bmn (    485.126 -1278.730   -27.581) dmn (     -0.003    -0.007    27.581) amx (    548.123 -1215.737   106.310) bmx (    548.126 -1215.730    -6.849) dmx (     -0.003    -0.007   113.159)
       113.107                 OcrGdsTfbInOav0xbf8f6c0 lvidx  39 amn (    484.123 -1279.737     0.000) bmn (    484.128 -1279.740   -27.612) dmn (     -0.005     0.003    27.612) amx (    549.123 -1214.737   106.310) bmx (    549.128 -1214.740    -6.797) dmx (     -0.005     0.003   113.107)
        112.19                 OcrGdsLsoInOav0xc354118 lvidx  40 amn (    466.623 -1297.237     0.000) bmn (    466.616 -1297.240   -28.580) dmn (      0.007     0.003    28.580) amx (    566.623 -1197.237   106.310) bmx (    566.616 -1197.240    -5.879) dmx (      0.007     0.003   112.190)
       109.189                    OcrGdsInIav0xc405b10 lvidx  23 amn (    485.123 -1278.737     0.000) bmn (    485.117 -1278.740   -37.759) dmn (      0.006     0.003    37.759) amx (    548.123 -1215.737    89.440) bmx (    548.117 -1215.740   -19.750) dmx (      0.006     0.003   109.189)
       76.1304                 OcrGdsInLsoOfl0xc26f450 lvidx  82 amn (    -31.072   -31.072  -171.358) bmn (    -31.500   -31.500  -247.488) dmn (      0.428     0.428    76.130) amx (     31.072    31.072   247.488) bmx (     31.500    31.500   247.488) dmx (     -0.428    -0.428    -0.000)
       75.3835                 OcrGdsLsoInOil0xc540738 lvidx  84 amn (    -49.663   -49.663  -172.104) bmn (    -50.000   -50.000  -247.488) dmn (      0.337     0.337    75.384) amx (     49.663    49.663   247.488) bmx (     50.000    50.000   247.488) dmx (     -0.337    -0.337    -0.000)
       55.9963               pmt-hemi-cathode0xc2f1ce8 lvidx  43 amn (   -128.000  -128.000     0.000) bmn (    -98.138   -98.147    55.996) dmn (    -29.862   -29.853   -55.996) amx (    128.000   128.000   128.000) bmx (     98.148    98.139   128.000) dmx (     29.852    29.861     0.000)
       48.0453              OcrGdsTfbInLsoOfl0xc2b5ba0 lvidx  83 amn (    -32.080   -32.080  -171.368) bmn (    -32.500   -32.500  -219.413) dmn (      0.420     0.420    48.045) amx (     32.080    32.080   247.488) bmx (     32.500    32.500   247.488) dmx (     -0.420    -0.420    -0.000)
       26.2183                   OcrGdsLsoPrt0xc104978 lvidx  81 amn (    -98.000   -98.000     0.000) bmn (    -98.000   -98.000    26.218) dmn (      0.000     0.000   -26.218) amx (     98.000    98.000   214.596) bmx (     98.000    98.000   214.596) dmx (      0.000     0.000     0.000)
       21.1528                         GdsOfl0xbf73918 lvidx 143 amn (   -649.616  -649.616   -37.349) bmn (   -650.000  -650.000   -58.502) dmn (      0.384     0.384    21.153) amx (    649.616   649.616    23.500) bmx (    650.000   650.000    23.500) dmx (     -0.384    -0.384     0.000)
            20               headon-pmt-mount0xc2a7670 lvidx  55 amn (    -51.377   -51.377  -120.000) bmn (    -36.850   -36.850  -100.000) dmn (    -14.528   -14.528   -20.000) amx (     51.377    51.377   100.000) bmx (     36.850    36.850   100.000) dmx (     14.528    14.528     0.000)
       18.8069                            lso0xc028a38 lvidx  37 amn (  -1979.474 -1979.474 -1982.000) bmn (  -1982.000 -1982.000 -1982.000) dmn (      2.526     2.526     0.000) amx (   1979.474  1979.474  2075.723) bmx (   1982.000  1982.000  2094.530) dmx (     -2.526    -2.526   -18.807)
          17.5           inn_short_cable_tray0xc3a4bc8 lvidx 208 amn (    -30.000  -750.000    -2.500) bmn (    -30.000  -750.000   -20.000) dmn (      0.000     0.000    17.500) amx (     30.000   750.000    20.000) bmx (     30.000   750.000    20.000) dmx (      0.000     0.000     0.000)
       15.0031                         LsoOfl0xc348ac0 lvidx 141 amn (   -909.475  -909.475    -5.000) bmn (   -910.031  -910.056   -20.003) dmn (      0.556     0.581    15.003) amx (    909.475   909.475    47.600) bmx (    910.000   910.000    47.642) dmx (     -0.525    -0.525    -0.042)
       14.5597                            gds0xc28d3f0 lvidx  22 amn (  -1548.036 -1548.036 -1535.000) bmn (  -1550.000 -1550.000 -1535.000) dmn (      1.964     1.964     0.000) amx (   1548.036  1548.036  1609.830) bmx (   1550.000  1550.000  1624.390) dmx (     -1.964    -1.964   -14.560)
       13.4421                OflTnkContainer0xc17cf50 lvidx 145 amn (  -1001.618 -1001.618   -92.500) bmn (  -1015.060 -1015.040   -92.500) dmn (     13.442    13.422     0.000) amx (   1001.618  1001.618   207.500) bmx (   1014.940  1014.960   207.482) dmx (    -13.322   -13.342     0.018)
       12.8687                   pmt-hemi-bot0xc22a958 lvidx  44 amn (    -99.000   -99.000   -99.000) bmn (    -98.143   -98.143   -99.000) dmn (     -0.857    -0.857     0.000) amx (     99.000    99.000     0.000) bmx (     98.143    98.143   -12.869) dmx (      0.857     0.857    12.869)
            12           near_side_long_hbeam0xbf3b5d0 lvidx  17 amn (  -2000.000  -100.000  -147.000) bmn (  -2000.000   -99.876  -135.000) dmn (      0.000    -0.124   -12.000) amx (   2000.000   100.000   147.000) bmx (   2000.070   100.124   146.908) dmx (     -0.070    -0.124     0.092)
       9.97886        AmCCo60AcrylicContainer0xc0b23b8 lvidx 131 amn (    -10.007   -10.007   -24.844) bmn (    -10.035   -10.035   -14.865) dmn (      0.028     0.028    -9.979) amx (     10.007    10.007    24.844) bmx (     10.035    10.036    24.899) dmx (     -0.028    -0.029    -0.056)
       9.97886                   source-shell0xc2d62d0 lvidx 111 amn (    -10.007   -10.007   -24.844) bmn (    -10.035   -10.035   -14.865) dmn (      0.028     0.028    -9.979) amx (     10.007    10.007    24.844) bmx (     10.035    10.035    14.865) dmx (     -0.028    -0.028     9.979)
       9.97886               led-source-shell0xc3068f0 lvidx 100 amn (    -10.007   -10.007   -24.844) bmn (    -10.035   -10.035   -14.865) dmn (      0.028     0.028    -9.979) amx (     10.007    10.007    24.844) bmx (     10.035    10.035    14.865) dmx (     -0.028    -0.028     9.979)
       9.96946                   weight-shell0xc307920 lvidx 103 amn (    -10.002   -10.002   -28.444) bmn (    -10.035   -10.035   -18.475) dmn (      0.033     0.033    -9.969) amx (     10.002    10.002    28.444) bmx (     10.035    10.035    18.475) dmx (     -0.033    -0.033     9.969)
             8              near_pool_iws_box0xc288ce8 lvidx 211 amn (  -6904.000 -3904.000 -4454.000) bmn (  -6912.000 -3912.000 -4454.000) dmn (      8.000     8.000     0.000) amx (   6904.000  3904.000  4454.000) bmx (   6912.000  3912.000  4454.000) dmx (     -8.000    -8.000     0.000)
             8              near_pool_ows_box0xbf8c8a8 lvidx 232 amn (  -7908.000 -4908.000 -4956.000) bmn (  -7916.000 -4916.000 -4956.000) dmn (      8.000     8.000     0.000) amx (   7908.000  4908.000  4956.000) bmx (   7916.000  4916.000  4956.000) dmx (     -8.000    -8.000     0.000)
        7.0824                            iav0xc346f90 lvidx  24 amn (  -1558.018 -1558.018 -1542.500) bmn (  -1564.900 -1565.070 -1542.500) dmn (      6.882     7.052     0.000) amx (   1558.018  1558.018  1631.346) bmx (   1565.100  1564.930  1631.990) dmx (     -7.082    -6.912    -0.644)
             5                      LsoOflTnk0xc17d928 lvidx 140 amn (   -920.000  -920.000   -10.000) bmn (   -920.042  -920.000    -5.000) dmn (      0.042     0.000    -5.000) amx (    920.000   920.000   170.000) bmx (    920.000   920.031   170.057) dmx (      0.000    -0.031    -0.057)
         3.882                   OcrCalLsoPrt0xc1076b0 lvidx  85 amn (    -98.000   -98.000     0.000) bmn (    -98.000   -98.000     3.882) dmn (      0.000     0.000    -3.882) amx (     98.000    98.000   214.596) bmx (     98.000    98.000   214.596) dmx (      0.000     0.000     0.000)
       3.65576                            ade0xc2a7438 lvidx 192 amn (  -2746.344 -2746.344 -3005.000) bmn (  -2750.000 -2750.000 -3005.000) dmn (      3.656     3.656     0.000) amx (   2746.344  2746.344  3005.000) bmx (   2750.000  2750.000  3005.000) dmx (     -3.656    -3.656     0.000)
       3.12695                            sst0xbf4b060 lvidx  94 amn (  -2496.873 -2496.873 -2500.000) bmn (  -2500.000 -2500.000 -2500.000) dmn (      3.127     3.127     0.000) amx (   2496.873  2496.873  2500.000) bmx (   2500.000  2500.000  2500.000) dmx (     -3.127    -3.127     0.000)
       3.10327                            oil0xbf5ed48 lvidx  90 amn (  -2484.897 -2484.897 -2477.500) bmn (  -2488.000 -2488.000 -2477.500) dmn (      3.103     3.103     0.000) amx (   2484.897  2484.897  2477.500) bmx (   2488.000  2488.000  2477.500) dmx (     -3.103    -3.103     0.000)
       2.72401                    MCBTopCover0xbfa5080 lvidx 182 amn (   -216.276  -216.276    -7.500) bmn (   -219.000  -219.000    -7.500) dmn (      2.724     2.724     0.000) amx (    216.276   216.276     7.500) bmx (    219.000   219.000     7.500) dmx (     -2.724    -2.724     0.000)
       2.67761                            oav0xc2ed7c8 lvidx  42 amn (  -2037.442 -2037.442 -1968.500) bmn (  -2040.070 -2040.120 -1968.500) dmn (      2.628     2.678     0.000) amx (   2037.442  2037.442  2125.092) bmx (   2039.930  2039.880  2126.210) dmx (     -2.488    -2.438    -1.118)
       1.76476                 Slope_rib1_tub0xc0d8aa8 lvidx 203 amn (    -48.235   -48.235  -690.000) bmn (    -50.000   -50.000  -690.000) dmn (      1.765     1.765     0.000) amx (     48.235    48.235   690.000) bmx (     50.000    50.000   690.000) dmx (     -1.765    -1.765     0.000)
       1.30371                         RPCMod0xc13bfd8 lvidx   7 amn (  -1085.000 -1098.696   -39.000) bmn (  -1085.000 -1100.000   -39.000) dmn (      0.000     1.304     0.000) amx (   1085.000  1098.696    39.000) bmx (   1085.000  1100.000    39.000) dmx (      0.000    -1.304     0.000)
       1.10778                 Slope_rib5_tub0xc0d8d08 lvidx 204 amn (    -48.892   -48.892  -528.000) bmn (    -50.000   -50.000  -528.000) dmn (      1.108     1.108     0.000) amx (     48.892    48.892   528.000) bmx (     50.000    50.000   528.000) dmx (     -1.108    -1.108     0.000)
       1.07508                   pmt-hemi-vac0xc21e248 lvidx  46 amn (    -97.930   -97.930  -164.500) bmn (    -98.995   -99.003  -164.504) dmn (      1.066     1.074     0.004) amx (     97.930    97.930   127.743) bmx (     99.005    98.997   128.000) dmx (     -1.075    -1.067    -0.257)
       1.01001                SstTopTshapeRib0xc272c80 lvidx  67 amn (  -1097.840   -50.000   -10.000) bmn (  -1097.840   -50.000   -10.000) dmn (      0.000     0.000     0.000) amx (   1097.840    50.000    10.000) bmx (   1096.830    50.000    10.000) dmx (      1.010     0.000     0.000)
        1.0038                         IWSLeg0xc2d1338 lvidx 199 amn (   -121.496  -121.496  -694.000) bmn (   -122.500  -122.500  -694.000) dmn (      1.004     1.004     0.000) amx (    121.496   121.496   694.000) bmx (    122.500   122.500   694.000) dmx (     -1.004    -1.004     0.000)
      0.799805                      near_rock0xc04ba08 lvidx 247 amn ( -25000.000-25000.000-12995.000) bmn ( -25000.000-25000.000-12994.200) dmn (      0.000     0.000    -0.800) amx (  25000.000 25000.000 25000.000) bmx (  25000.000 25000.000 25000.000) dmx (      0.000     0.000     0.000)
      0.630913                         OWSLeg0xcced7f0 lvidx 214 amn (   -121.869  -121.869  -500.000) bmn (   -122.500  -122.500  -500.000) dmn (      0.631     0.631     0.000) amx (    121.869   121.869   500.000) bmx (    122.500   122.500   500.000) dmx (     -0.631    -0.631     0.000)
       0.60022                 OflTnkCnrSpace0xc3d3d30 lvidx 144 amn (   -923.400  -923.400  -100.000) bmn (   -924.000  -924.000  -100.000) dmn (      0.600     0.600     0.000) amx (    923.400   923.400   200.000) bmx (    924.000   924.000   199.988) dmx (     -0.600    -0.600     0.012)
      0.582367              OffCenterCalibELS0xc17a8a0 lvidx 167 amn (   -399.718  -399.718  -494.490) bmn (   -400.300  -400.300  -494.490) dmn (      0.582     0.582     0.000) amx (    399.718   399.718   494.490) bmx (    400.300   400.300   494.490) dmx (     -0.582    -0.582     0.000)
      0.582367            OffCenterCalibEGdLS0xc3d56a8 lvidx 157 amn (   -399.718  -399.718  -494.490) bmn (   -400.300  -400.300  -494.490) dmn (      0.582     0.582     0.000) amx (    399.718   399.718   494.490) bmx (    400.300   400.300   494.490) dmx (     -0.582    -0.582     0.000)
           0.5            near_hall_top_dwarf0xc0316c8 lvidx  21 amn ( -30500.000 -7500.000 -7500.000) bmn ( -30500.500 -7500.390 -7500.290) dmn (      0.500     0.390     0.290) amx (  13500.000  7500.000  7500.000) bmx (  13500.000  7500.000  7500.000) dmx (      0.000     0.000     0.000)
      0.489929          near_side_short_hbeam0xc2b1ea8 lvidx  10 amn (   -999.570  -100.000  -147.000) bmn (  -1000.060  -100.046  -147.034) dmn (      0.490     0.046     0.034) amx (    999.570   100.000   147.000) bmx (   1000.060   100.046   147.034) dmx (     -0.490    -0.046    -0.034)
      0.474579                       pmt-hemi0xc0fed90 lvidx  47 amn (   -100.763  -100.763  -169.000) bmn (   -100.288  -100.288  -168.995) dmn (     -0.475    -0.475    -0.005) amx (    100.763   100.763   130.734) bmx (    100.288   100.288   131.000) dmx (      0.475     0.475    -0.266)
        0.4552                   CenterCalibE0xc3a4250 lvidx 139 amn (   -399.845  -399.845  -344.490) bmn (   -400.300  -400.300  -344.490) dmn (      0.455     0.455     0.000) amx (    399.845   399.845   344.490) bmx (    400.300   400.300   344.490) dmx (     -0.455    -0.455     0.000)
      0.453979                MOOverflowTankE0xbfa5678 lvidx 191 amn (   -659.546  -659.546  -148.500) bmn (   -660.000  -660.000  -148.500) dmn (      0.454     0.454     0.000) amx (    659.546   659.546   148.500) bmx (    660.000   660.000   148.500) dmx (     -0.454    -0.454     0.000)
      0.424454                    source-assy0xc2d5d78 lvidx 112 amn (     -9.826    -9.826   -97.285) bmn (    -10.036   -10.036   -97.286) dmn (      0.210     0.210     0.001) amx (      9.826     9.826   106.897) bmx (     10.036    10.036   107.321) dmx (     -0.210    -0.211    -0.424)
      0.423454                led-source-assy0xc3061d0 lvidx 105 amn (     -9.826    -9.826   -97.285) bmn (    -10.036   -10.036   -97.286) dmn (      0.211     0.210     0.001) amx (      9.826     9.826   106.897) bmx (     10.036    10.036   107.320) dmx (     -0.210    -0.210    -0.423)
      0.423454            amcco60-source-assy0xc0b1df8 lvidx 132 amn (     -9.826    -9.826   -97.285) bmn (    -10.036   -10.036   -97.286) dmn (      0.211     0.210     0.001) amx (      9.826     9.826   106.897) bmx (     10.036    10.036   107.320) dmx (     -0.210    -0.210    -0.423)
      0.408173                CalibrationDome0xc349280 lvidx 138 amn (   -304.392  -304.392  -336.550) bmn (   -304.800  -304.800  -336.550) dmn (      0.408     0.408     0.000) amx (    304.392   304.392   336.550) bmx (    304.800   304.800   336.550) dmx (     -0.408    -0.408     0.000)
      0.404297                   DomeInterior0xc0ace30 lvidx 137 amn (   -299.636  -299.636  -334.170) bmn (   -300.040  -300.040  -334.170) dmn (      0.404     0.404     0.000) amx (    299.636   299.636   334.170) bmx (    300.040   300.040   334.170) dmx (     -0.404    -0.404     0.000)
      0.381062                 CtrGdsOflInLso0xbfa1178 lvidx  28 amn (    -31.119   -31.119  -230.091) bmn (    -31.500   -31.500  -230.091) dmn (      0.381     0.381     0.000) amx (     31.119    31.119   230.091) bmx (     31.500    31.500   230.091) dmx (     -0.381    -0.381    -0.000)
      0.358002                near_span_hbeam0xc2a27d8 lvidx   9 amn (   -100.000 -5871.000  -147.000) bmn (   -100.358 -5871.000  -147.196) dmn (      0.358     0.000     0.196) amx (    100.000  5871.000   147.000) bmx (    100.358  5871.000   147.196) dmx (     -0.358     0.000    -0.196)
      0.343018                       MOInMOFT0xc047100 lvidx 186 amn (   -574.657  -574.657   -41.500) bmn (   -575.000  -575.000   -41.500) dmn (      0.343     0.343     0.000) amx (    574.657   574.657    41.500) bmx (    575.000   575.000    41.500) dmx (     -0.343    -0.343     0.000)
       0.33654                      OcrCalLso0xc103c18 lvidx  86 amn (    -49.663   -49.663  -247.488) bmn (    -50.000   -50.000  -247.488) dmn (      0.337     0.337     0.000) amx (     49.663    49.663   247.488) bmx (     50.000    50.000   247.488) dmx (     -0.337    -0.337    -0.000)
      0.306915              CtrGdsOflInLsoOfl0xc103b70 lvidx  78 amn (    -31.193   -31.193  -200.190) bmn (    -31.500   -31.500  -200.190) dmn (      0.307     0.307     0.000) amx (     31.193    31.193   200.190) bmx (     31.500    31.500   200.190) dmx (     -0.307    -0.307    -0.000)
      0.303955               SupportSpoolGdLS0xc33f3f0 lvidx 154 amn (   -389.696  -389.696  -143.650) bmn (   -390.000  -390.000  -143.650) dmn (      0.304     0.304     0.000) amx (    389.696   389.696   143.650) bmx (    390.000   390.000   143.650) dmx (     -0.304    -0.304     0.000)
      0.303955                 SupportSpoolLS0xc17ac20 lvidx 166 amn (   -389.696  -389.696  -143.650) bmn (   -390.000  -390.000  -143.650) dmn (      0.304     0.304     0.000) amx (    389.696   389.696   143.650) bmx (    390.000   390.000   143.650) dmx (     -0.304    -0.304     0.000)
      0.301941           CtrGdsOflTfbInLsoOfl0xc183610 lvidx  79 amn (    -32.198   -32.198  -200.190) bmn (    -32.500   -32.500  -200.190) dmn (      0.302     0.302     0.000) amx (     32.198    32.198   200.190) bmx (     32.500    32.500   200.190) dmx (     -0.302    -0.302    -0.000)
      0.298584       SupportSpoolInteriorGdLS0xc33f780 lvidx 153 amn (   -379.701  -379.701  -143.650) bmn (   -380.000  -380.000  -143.650) dmn (      0.299     0.299     0.000) amx (    379.701   379.701   143.650) bmx (    380.000   380.000   143.650) dmx (     -0.299    -0.299     0.000)
      0.298584         SupportSpoolInteriorLS0xc17ae90 lvidx 165 amn (   -379.701  -379.701  -143.650) bmn (   -380.000  -380.000  -143.650) dmn (      0.299     0.299     0.000) amx (    379.701   379.701   143.650) bmx (    380.000   380.000   143.650) dmx (     -0.299    -0.299     0.000)
      0.250946                 CtrLsoOflInOil0xc1831a0 lvidx  80 amn (    -49.749   -49.749  -200.190) bmn (    -50.000   -50.000  -200.190) dmn (      0.251     0.251     0.000) amx (     49.749    49.749   200.190) bmx (     50.000    50.000   200.190) dmx (     -0.251    -0.251    -0.000)
      0.249634            GasDistributionBoxE0xc2d0b50 lvidx 176 amn (   -304.805  -304.805  -130.190) bmn (   -305.055  -305.055  -130.190) dmn (      0.250     0.250     0.000) amx (    304.805   304.805   130.190) bmx (    305.055   305.055   130.190) dmx (     -0.250    -0.250     0.000)
      0.199768                GDBTubeInterior0xc20d098 lvidx 172 amn (   -248.975  -248.975  -100.190) bmn (   -249.175  -249.175  -100.190) dmn (      0.200     0.200     0.000) amx (    248.975   248.975   100.190) bmx (    249.175   249.175   100.190) dmx (     -0.200    -0.200     0.000)
      0.188812                  MOClarityBoxE0xc20e8e0 lvidx 183 amn (   -218.811  -218.811  -107.500) bmn (   -219.000  -219.000  -107.500) dmn (      0.189     0.189     0.000) amx (    218.811   218.811   107.500) bmx (    219.000   219.000   107.500) dmx (     -0.189    -0.189     0.000)
      0.188141                      SsTBotHub0xc26d1d0 lvidx  64 amn (   -129.812  -129.812  -159.500) bmn (   -130.000  -130.000  -159.500) dmn (      0.188     0.188     0.000) amx (    129.812   129.812   159.500) bmx (    130.000   130.000   159.500) dmx (     -0.188    -0.188     0.000)
      0.186386            GdLSCalibTubAbvLidE0xc340400 lvidx 152 amn (   -169.814  -169.814  -137.500) bmn (   -170.000  -170.000  -137.500) dmn (      0.186     0.186     0.000) amx (    169.814   169.814   137.500) bmx (    170.000   170.000   137.500) dmx (     -0.186    -0.186     0.000)
      0.186386              LSCalibTubAbvLidE0xc17bb30 lvidx 164 amn (   -169.814  -169.814  -137.500) bmn (   -170.000  -170.000  -137.500) dmn (      0.186     0.186     0.000) amx (    169.814   169.814   137.500) bmx (    170.000   170.000   137.500) dmx (     -0.186    -0.186     0.000)
      0.151627                MCBTubeInterior0xc213790 lvidx 179 amn (   -174.848  -174.848   -87.000) bmn (   -175.000  -175.000   -87.000) dmn (      0.152     0.152     0.000) amx (    174.848   174.848    87.000) bmx (    175.000   175.000    87.000) dmx (     -0.152    -0.152     0.000)
      0.144089           GDBTopFlangeInterior0xc20d970 lvidx 174 amn (   -249.031  -249.031   -10.000) bmn (   -249.175  -249.175   -10.000) dmn (      0.144     0.144     0.000) amx (    249.031   249.031    10.000) bmx (    249.175   249.175    10.000) dmx (     -0.144    -0.144     0.000)
      0.127319                   MCBTopFlange0xc213a48 lvidx 180 amn (   -218.873  -218.873   -10.000) bmn (   -219.000  -219.000   -10.000) dmn (      0.127     0.127     0.000) amx (    218.873   218.873    10.000) bmx (    219.000   219.000    10.000) dmx (     -0.127    -0.127     0.000)
      0.125473                AcrylicCylinder0xc3d3830 lvidx 136 amn (   -199.875  -199.875   -25.000) bmn (   -200.000  -200.000   -25.000) dmn (      0.125     0.125     0.000) amx (    199.875   199.875    25.000) bmx (    200.000   200.000    25.000) dmx (     -0.125    -0.125     0.000)
      0.125076                      IavBotHub0xbf8cfd0 lvidx  35 amn (    -99.875   -99.875  -100.000) bmn (   -100.000  -100.000  -100.000) dmn (      0.125     0.125     0.000) amx (     99.875    99.875   100.000) bmx (    100.000   100.000   100.000) dmx (     -0.125    -0.125     0.000)
      0.123833                      OavBotHub0xc355030 lvidx  33 amn (    -99.876   -99.876   -98.500) bmn (   -100.000  -100.000   -98.500) dmn (      0.124     0.124     0.000) amx (     99.876    99.876    98.500) bmx (    100.000   100.000    98.500) dmx (     -0.124    -0.124     0.000)
      0.115061                  wall-led-assy0xc3a99a0 lvidx  89 amn (     -9.459    -9.459    -9.410) bmn (     -9.525    -9.525    -9.525) dmn (      0.066     0.066     0.115) amx (      9.459     9.459    77.620) bmx (      9.525     9.525    77.620) dmx (     -0.066    -0.066     0.000)
      0.105911         GdLSCalibTubAbvLidTub40xc340e28 lvidx 149 amn (   -169.894  -169.894   -20.000) bmn (   -170.000  -170.000   -20.000) dmn (      0.106     0.106     0.000) amx (    169.894   169.894    20.000) bmx (    170.000   170.000    20.000) dmx (     -0.106    -0.106     0.000)
      0.105911           LSCalibTubAbvLidTub40xc17c470 lvidx 161 amn (   -169.894  -169.894   -20.000) bmn (   -170.000  -170.000   -20.000) dmn (      0.106     0.106     0.000) amx (    169.894   169.894    20.000) bmx (    170.000   170.000    20.000) dmx (     -0.106    -0.106     0.000)
      0.104446         GdLSCalibTubAbvLidTub30xc340bd0 lvidx 148 amn (   -169.896  -169.896   -17.500) bmn (   -170.000  -170.000   -17.500) dmn (      0.104     0.104     0.000) amx (    169.896   169.896    17.500) bmx (    170.000   170.000    17.500) dmx (     -0.104    -0.104     0.000)
      0.104446           LSCalibTubAbvLidTub30xc17c220 lvidx 160 amn (   -169.896  -169.896   -17.500) bmn (   -170.000  -170.000   -17.500) dmn (      0.104     0.104     0.000) amx (    169.896   169.896    17.500) bmx (    170.000   170.000    17.500) dmx (     -0.104    -0.104     0.000)
       0.10289           MCBTopFlangeInterior0xc213c68 lvidx 181 amn (   -174.897  -174.897   -10.000) bmn (   -175.000  -175.000   -10.000) dmn (      0.103     0.103     0.000) amx (    174.897   174.897    10.000) bmx (    175.000   175.000    10.000) dmx (     -0.103    -0.103     0.000)
    2017-07-03 20:53:28.708 INFO  [2994395] [GScene::compareMeshes_GMeshBB@469] GScene::compareMeshes_GMeshBB num_meshes 249 cut 0.1 num_discrepant 95 frac 0.381526
    Assertion failed: (0 && "GScene::init early exit for gltf==4"), function init, file /Users/blyth/opticks/ggeo/GScene.cc, line 153.




