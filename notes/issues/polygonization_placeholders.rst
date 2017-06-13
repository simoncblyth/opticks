Polygonization Fails Producing Large BBox Placeholder
======================================================

Poly fails are replaced with placeholder bbox, but these bbox are crazy large

* Need to apply CSG to the bbox

::

    tgltf-gdml

    2017-06-13 12:28:22.546 INFO  [6939707] [NScene::load_mesh_extras@163] NScene::load_mesh_extras START m_verbosity 1 num_meshes 171
     mId    0 npr    1 nam                                    /dd/Geometry/AD/lvADE0xc2a78c0 iug 1 poly  IM smry  ht  0 nn    1 tri   4716 tmsg  iug 1 nd 1,4,4 tr 1,3,4,4 gtr 1,3,4,4 pln NULL
     mId    1 npr    1 nam                                    /dd/Geometry/AD/lvSST0xc234cd0 iug 1 poly  IM smry  ht  0 nn    1 tri   4460 tmsg  iug 1 nd 1,4,4 tr 1,3,4,4 gtr 1,3,4,4 pln NULL
     mId    2 npr    1 nam                                    /dd/Geometry/AD/lvOIL0xbf5e0b8 iug 1 poly  IM smry  ht  0 nn    1 tri   4460 tmsg  iug 1 nd 1,4,4 tr 1,3,4,4 gtr 1,3,4,4 pln NULL
     mId    3 npr    1 nam                                    /dd/Geometry/AD/lvOAV0xbf1c760 iug 1 poly  IM smry  ht  2 nn    7 tri   4460 tmsg  iug 1 nd 7,4,4 tr 5,3,4,4 gtr 2,3,4,4 pln NULL
     mId    4 npr    1 nam                                    /dd/Geometry/AD/lvLSO0xc403e40 iug 1 poly  IM smry  ht  2 nn    7 tri   4428 tmsg  iug 1 nd 7,4,4 tr 5,3,4,4 gtr 2,3,4,4 pln NULL
     mId    5 npr    1 nam                                    /dd/Geometry/AD/lvIAV0xc404ee8 iug 1 poly  IM smry  ht  2 nn    7 tri   4428 tmsg  iug 1 nd 7,4,4 tr 5,3,4,4 gtr 2,3,4,4 pln NULL
     mId    6 npr    1 nam                                    /dd/Geometry/AD/lvGDS0xbf6cbb8 iug 1 poly  IM smry  ht  2 nn    7 tri   4460 tmsg  iug 1 nd 7,4,4 tr 5,3,4,4 gtr 2,3,4,4 pln NULL
     mId    7 npr    1 nam                     /dd/Geometry/AdDetails/lvOcrGdsInIav0xbf6dd58 iug 1 poly  IM smry  ht  2 nn    7 tri     12 tmsg PLACEHOLDER iug 1 nd 7,4,4 tr 5,3,4,4 gtr 3,3,4,4 pln NULL
     mId    8 npr    1 nam                       /dd/Geometry/AdDetails/lvIavTopHub0xc129d88 iug 1 poly  IM smry  ht  1 nn    3 tri     12 tmsg PLACEHOLDER iug 1 nd 3,4,4 tr 3,3,4,4 gtr 1,3,4,4 pln NULL
     mId    9 npr    1 nam                 /dd/Geometry/AdDetails/lvCtrGdsOflBotClp0xc407eb0 iug 1 poly  IM smry  ht  1 nn    3 tri     12 tmsg PLACEHOLDER iug 1 nd 3,4,4 tr 3,3,4,4 gtr 1,3,4,4 pln NULL
     mId   10 npr    1 nam               /dd/Geometry/AdDetails/lvCtrGdsOflTfbInLso0xbfa0728 iug 1 poly  IM smry  ht  1 nn    3 tri     12 tmsg PLACEHOLDER iug 1 nd 3,4,4 tr 3,3,4,4 gtr 1,3,4,4 pln NULL
     mId   11 npr    1 nam                  /dd/Geometry/AdDetails/lvCtrGdsOflInLso0xc28cc88 iug 1 poly  IM smry  ht  0 nn    1 tri   2428 tmsg  iug 1 nd 1,4,4 tr 1,3,4,4 gtr 1,3,4,4 pln NULL
     mId   12 npr    1 nam                       /dd/Geometry/AdDetails/lvOcrGdsPrt0xc352630 iug 1 poly  IM smry  ht  2 nn    7 tri     12 tmsg PLACEHOLDER iug 1 nd 7,4,4 tr 5,3,4,4 gtr 2,3,4,4 pln NULL
     mId   13 npr    1 nam                  /dd/Geometry/AdDetails/lvOcrGdsTfbInLso0xc3529c0 iug 1 poly  IM smry  ht  2 nn    7 tri     12 tmsg PLACEHOLDER iug 1 nd 7,4,4 tr 5,3,4,4 gtr 2,3,4,4 pln NULL
     mId   14 npr    1 nam                     /dd/Geometry/AdDetails/lvOcrGdsInLso0xc353990 iug 1 poly  IM smry  ht  2 nn    7 tri     12 tmsg PLACEHOLDER iug 1 nd 7,4,4 tr 5,3,4,4 gtr 3,3,4,4 pln NULL
     mId   15 npr    1 nam                       /dd/Geometry/AdDetails/lvOavBotRib0xc353d30 iug 1 poly  IM smry  ht  0 nn    1 tri   1772 tmsg  iug 1 nd 1,4,4 tr 1,3,4,4 gtr 1,3,4,4 pln NULL
     mId   16 npr    1 nam                       /dd/Geometry/AdDetails/lvOavBotHub0xc3550d8 iug 1 poly  IM smry  ht  0 nn    1 tri   4460 tmsg  iug 1 nd 1,4,4 tr 1,3,4,4 gtr 1,3,4,4 pln NULL
     mId   17 npr    1 nam                       /dd/Geometry/AdDetails/lvIavBotRib0xc355990 iug 1 poly  IM smry  ht  0 nn    1 tri   1708 tmsg  iug 1 nd 1,4,4 tr 1,3,4,4 gtr 1,3,4,4 pln NULL

     ...
    mId  170 npr    1 nam                    /dd/Geometry/AdDetails/lvMOFTTopCover0xbfa5160 iug 1 poly  IM smry  ht  0 nn    1 tri     12 tmsg PLACEHOLDER iug 1 nd 1,4,4 tr 1,3,4,4 gtr 1,3,4,4 pln NULL
    2017-06-13 12:28:24.482 INFO  [6939707] [NScene::load_mesh_extras@219] NScene::load_mesh_extras DONE m_verbosity 1 num_meshes 171 m_num_global 171 m_num_csgskip 14 m_num_placeholder 59
    2017-06-13 12:28:24.482 INFO  [6939707] [NScene::NScene@75] NScene::NScene DONE



Added collection of lvnames of failed polygonizations in NScene and used the list 
in analytic/sc.py to dump the problematic CSG node trees at python level.

* intermediate goal is just to come up with a sensible bbox for these, currently 
  are getting crazy large ones

* notice lots of CSG difference, but also some simple primitives : how can IM fail 
  with those ... perhaps some epsilon tolerance regarding tris sticking out ?



::

    simon:analytic blyth$ ./sc.py --lvnlist /tmp/blyth/opticks/tgltf/PLACEHOLDER_FAILED_POLY.txt 

    [2017-06-13 13:51:02,002] p11969 {./sc.py:273} INFO - dump_all lvns 59 
     /dd/Geometry/AdDetails/lvOcrGdsInIav0xbf6dd58                : 1 : in(di(co,co),cy) height:2 totnodes:7  
     /dd/Geometry/AdDetails/lvIavTopHub0xc129d88                  : 1 : un(cy,cy) height:1 totnodes:3  
     /dd/Geometry/AdDetails/lvCtrGdsOflBotClp0xc407eb0            : 1 : un(cy,cy) height:1 totnodes:3  
     /dd/Geometry/AdDetails/lvCtrGdsOflTfbInLso0xbfa0728          : 1 : di(cy,cy) height:1 totnodes:3  
     /dd/Geometry/AdDetails/lvOcrGdsPrt0xc352630                  : 1 : di(un(cy,cy) height:1 totnodes:3 ,co) height:2 totnodes:7  
     /dd/Geometry/AdDetails/lvOcrGdsTfbInLso0xc3529c0             : 1 : in(co,di(cy,cy)) height:2 totnodes:7  
     /dd/Geometry/AdDetails/lvOcrGdsInLso0xc353990                : 1 : in(di(co,co),cy) height:2 totnodes:7  
     /dd/Geometry/AdDetails/lvIavTopRib0xbf8e280                  : 1 : di(di(bo,bo),co) height:2 totnodes:7  
     /dd/Geometry/AdDetails/lvOcrGdsLsoInOav0xbf8fd98             : 1 : in(di(co,co),cy) height:2 totnodes:7  
     /dd/Geometry/AdDetails/lvOcrGdsTfbInOav0xbfa35f8             : 1 : in(di(co,co),cy) height:2 totnodes:7  
     /dd/Geometry/AdDetails/lvOcrGdsInOav0xbfa3ab8                : 1 : in(di(co,co),cy) height:2 totnodes:7  
     /dd/Geometry/AdDetails/lvOcrCalLsoInOav0xbfa4d90             : 1 : in(di(co,co),cy) height:2 totnodes:7  
     /dd/Geometry/PMT/lvPmtHemiCathode0xc2cdca0                   : 1 : un(di(zs,zs),di(zs,zs)) height:2 totnodes:7  
     /dd/Geometry/PMT/lvPmtHemiBottom0xc12ad60                    : 1 : di(zs,zs) height:1 totnodes:3  
     /dd/Geometry/PMT/lvHeadonPmtCathode0xc2c8d98                 : 1 : cy height:0 totnodes:1  
     /dd/Geometry/AdDetails/lvTopReflector0xbf9be68               : 1 : di(di(di(di(di(cy,cy),cy),cy),cy),cy) height:5 totnodes:63  
     /dd/Geometry/AdDetails/lvTopRefGap0xbf9c648                  : 1 : di(di(di(di(di(cy,cy),cy),cy),cy),cy) height:5 totnodes:63  
     /dd/Geometry/AdDetails/lvTopESR0xc21fb88                     : 1 : di(di(di(di(di(di(di(di(di(cy,cy),cy),cy),cy),cy),cy),cy),cy),cy) height:9 totnodes:1023  
     /dd/Geometry/AdDetails/lvBotReflector0xc3cd4c0               : 1 : di(di(di(di(di(cy,cy),bo),bo),bo),bo) height:5 totnodes:63  
     /dd/Geometry/AdDetails/lvBotRefGap0xc34bc68                  : 1 : di(di(di(di(di(cy,cy),bo),bo),bo),bo) height:5 totnodes:63  
     /dd/Geometry/AdDetails/lvBotESR0xbfa74c0                     : 1 : di(di(di(di(di(di(di(di(cy,cy),bo),bo),bo),bo),cy),cy),cy) height:8 totnodes:511  
     /dd/Geometry/AdDetails/lvSstBotRib0xc26c650                  : 1 : un(di(bo,cy),di(bo,cy)) height:2 totnodes:7  
     /dd/Geometry/AdDetails/lvSstBotCirRibBase0xc26e220           : 1 : di(di(di(cy,cy),bo),bo) height:3 totnodes:15  
     /dd/Geometry/AdDetails/lvSstTopRadiusRib0xc2716c0            : 1 : di(di(tr,bo),bo) height:2 totnodes:7  
     /dd/Geometry/AdDetails/lvSstTopTshapeRib0xc2629f0            : 1 : di(di(bo,cy),di(cy,cy)) height:2 totnodes:7  
     /dd/Geometry/AdDetails/lvSstTopHub0xc2644f0                  : 1 : un(cy height:0 totnodes:1 ,cy height:0 totnodes:1 ) height:1 totnodes:3  
     /dd/Geometry/AdDetails/lvSstTopCirRibBase0xc2649f0           : 1 : di(di(di(di(di(cy,cy),bo),bo),bo),bo) height:5 totnodes:63  
     /dd/Geometry/AdDetails/lvSstInnVerRibBase0xbf31748           : 1 : di(bo,tr) height:1 totnodes:3  
     /dd/Geometry/AdDetails/lvBotRefRadialLongRib0xbf32988        : 1 : bo height:0 totnodes:1  
     /dd/Geometry/AdDetails/lvBotRefRadialShortRib0xbf339c8       : 1 : bo height:0 totnodes:1  
     /dd/Geometry/AdDetails/lvOavTopRib0xbf7bce8                  : 1 : di(di(bo,bo),co) height:2 totnodes:7  
     /dd/Geometry/AdDetails/lvOavTopHub0xbf366d0                  : 1 : un(un(cy,cy),cy) height:2 totnodes:7  
     /dd/Geometry/AdDetails/lvCtrLsoOflTfb0xc3a2ab0               : 1 : di(cy,cy) height:1 totnodes:3  
     /dd/Geometry/AdDetails/lvCtrLsoOflTopClp0xc26f5a0            : 1 : un(un(cy,cy),cy) height:2 totnodes:7  
     /dd/Geometry/AdDetails/lvOcrGdsLsoPrt0xc104a90               : 1 : di(un(cy,cy) height:1 totnodes:3 ,co) height:2 totnodes:7  
     /dd/Geometry/AdDetails/lvOcrGdsLsoOfl0xc1052d0               : 1 : di(cy,co) height:1 totnodes:3  
     /dd/Geometry/AdDetails/lvOcrGdsTfbInLsoOfl0xc105560          : 1 : di(cy,co) height:1 totnodes:3  
     /dd/Geometry/AdDetails/lvOcrGdsInLsoOfl0xc106018             : 1 : di(cy,co) height:1 totnodes:3  
     /dd/Geometry/AdDetails/lvOcrCalLsoPrt0xc1077c8               : 1 : di(un(cy,cy) height:1 totnodes:3 ,co) height:2 totnodes:7  
     /dd/Geometry/AdDetails/lvOcrCalLso0xc17e288                  : 1 : di(cy,co) height:1 totnodes:3  
     /dd/Geometry/CalibrationBox/lvBottomPlate0xc3a4168           : 1 : cy height:0 totnodes:1  
     /dd/Geometry/CalibrationBox/lvTurntable0xbf78630             : 1 : di(di(di(cy,cy),cy),cy) height:3 totnodes:15  
     /dd/Geometry/CalibrationSources/lvWeightCable0xc308988       : 1 : cy height:0 totnodes:1  
     /dd/Geometry/OverflowTanks/lvLsoOflTnk0xc0ad990              : 1 : un(un(un(di(cy,cy),di(cy,cy)),di(cy,cy)),di(cy,cy)) height:4 totnodes:31  
     /dd/Geometry/CalibrationBox/lvSpoolFlange0xc340a78           : 1 : cy height:0 totnodes:1  
     /dd/Geometry/CalibrationBox/lvSpoolFlangeInterior0xc340cd0   : 1 : cy height:0 totnodes:1  
     /dd/Geometry/AdDetails/lvAdVertiCableTray0xc3a27f0           : 1 : di(bo,bo) height:1 totnodes:3  
     /dd/Geometry/AdDetails/lvAdVertiCable0xc2d1f60               : 1 : bo height:0 totnodes:1  
     /dd/Geometry/AdDetails/lvGDBBottomPlate0xbfa8728             : 1 : cy height:0 totnodes:1  
     /dd/Geometry/AdDetails/lvGDBTopFlange0xc3d5420               : 1 : di(cy,cy) height:1 totnodes:3  
     /dd/Geometry/AdDetails/lvGDBTopCover0xc2d0ce0                : 1 : cy height:0 totnodes:1  
     /dd/Geometry/AdDetails/lvMCBBottomPlate0xc20de58             : 1 : cy height:0 totnodes:1  
     /dd/Geometry/AdDetails/lvMCBTube0xc20e168                    : 1 : di(cy,cy) height:1 totnodes:3  
     /dd/Geometry/AdDetails/lvMOFTBottomPlate0xc213678            : 1 : cy height:0 totnodes:1  
     /dd/Geometry/AdDetails/lvMOFTTube0xbfa58b0                   : 1 : di(cy,cy) height:1 totnodes:3  
     /dd/Geometry/AdDetails/lvMOFTTubeInterior0xc046d88           : 1 : cy height:0 totnodes:1  
     /dd/Geometry/AdDetails/lvMOFTTopFlange0xbfa5720              : 1 : di(cy,cy) height:1 totnodes:3  
     /dd/Geometry/AdDetails/lvMOFTTopFlangeInterior0xc047610      : 1 : cy height:0 totnodes:1  
     /dd/Geometry/AdDetails/lvMOFTTopCover0xbfa5160               : 1 : cy height:0 totnodes:1  



