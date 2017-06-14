CSG Deep Trees (solid level issue)
========================================

ISSUE : DYB has some deep CSG trees
--------------------------------------

* observed with scene.py, csg.py see some surprising totnodes arising from deep trees
* deep trees are very inefficiently handled as complete binary trees


Lots of subtraction of rotated boxes : results in deep trees
--------------------------------------------------------------

Looking at subsolid ranges with scene.py, csg.py see that 
all DYB solids have contiguous subsolid idx ranges, ie there is 
no cross referencing between different solid assemblies.

DYB Near has 707 solids. But those are not distinct solids, eg the sub-chains 
for solid.idx 664 are in contiguous range 640-664 occupying 25 solid idx, 
whereas logically theres really only one complex solid.idx 664.


How to handle deep unbalanced trees ?
-----------------------------------------

Moved literature search to env-;csg-




Most of these are differences...

* complement was implemented to allow tree rearrangement, use that 
  to create +ve form tree (with signs all in the leaves as complements) 
  so then have a fully commutative CSG expression tree 
  that can easily be balanced


    simon:analytic blyth$ ./sc.py --lvnlist /tmp/blyth/opticks/tgltf/tgltf-gdml--/CSGSKIP_DEEP_TREES.txt
    args: ./sc.py --lvnlist /tmp/blyth/opticks/tgltf/tgltf-gdml--/CSGSKIP_DEEP_TREES.txt
    [2017-06-13 18:01:12,699] p29002 {./sc.py:306} INFO -  gsel:3153 gidx:0 gmaxnode:0 gmaxdepth:0 
    [2017-06-13 18:01:12,699] p29002 {/Users/blyth/opticks/analytic/gdml.py:959} INFO - parsing gdmlpath /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.gdml 
    [2017-06-13 18:01:13,875] p29002 {./sc.py:194} WARNING - tlv( 38): csg.skip as height  7 > 3 lvn /dd/Geometry/AdDetails/lvRadialShieldUnit0xc3d7ec0 lvidx 56 
    [2017-06-13 18:01:13,883] p29002 {./sc.py:194} WARNING - tlv( 39): csg.skip as height  5 > 3 lvn /dd/Geometry/AdDetails/lvTopReflector0xbf9be68 lvidx 59 
    [2017-06-13 18:01:13,885] p29002 {./sc.py:194} WARNING - tlv( 40): csg.skip as height  5 > 3 lvn /dd/Geometry/AdDetails/lvTopRefGap0xbf9c648 lvidx 58 
    [2017-06-13 18:01:13,887] p29002 {./sc.py:194} WARNING - tlv( 41): csg.skip as height  9 > 3 lvn /dd/Geometry/AdDetails/lvTopESR0xc21fb88 lvidx 57 
    [2017-06-13 18:01:13,888] p29002 {./sc.py:194} WARNING - tlv( 42): csg.skip as height  5 > 3 lvn /dd/Geometry/AdDetails/lvBotReflector0xc3cd4c0 lvidx 62 
    [2017-06-13 18:01:13,889] p29002 {./sc.py:194} WARNING - tlv( 43): csg.skip as height  5 > 3 lvn /dd/Geometry/AdDetails/lvBotRefGap0xc34bc68 lvidx 61 
    [2017-06-13 18:01:13,891] p29002 {./sc.py:194} WARNING - tlv( 44): csg.skip as height  8 > 3 lvn /dd/Geometry/AdDetails/lvBotESR0xbfa74c0 lvidx 60 
    [2017-06-13 18:01:13,905] p29002 {./sc.py:194} WARNING - tlv( 51): csg.skip as height  5 > 3 lvn /dd/Geometry/AdDetails/lvSstTopCirRibBase0xc2649f0 lvidx 69 
    [2017-06-13 18:01:13,936] p29002 {./sc.py:194} WARNING - tlv( 82): csg.skip as height 10 > 3 lvn /dd/Geometry/CalibrationSources/lvLedSourceAssy0xc306328 lvidx 105 
    [2017-06-13 18:01:13,943] p29002 {./sc.py:194} WARNING - tlv( 89): csg.skip as height 10 > 3 lvn /dd/Geometry/CalibrationSources/lvGe68SourceAssy0xc2d4ad0 lvidx 112 
    [2017-06-13 18:01:13,949] p29002 {./sc.py:194} WARNING - tlv( 96): csg.skip as height 10 > 3 lvn /dd/Geometry/CalibrationSources/lvAmCCo60SourceAssy0xc0b1da0 lvidx 132 
    [2017-06-13 18:01:13,961] p29002 {./sc.py:194} WARNING - tlv(120): csg.skip as height  5 > 3 lvn /dd/Geometry/OverflowTanks/lvOflTnkContainer0xc17cee8 lvidx 145 
    [2017-06-13 18:01:13,963] p29002 {./sc.py:194} WARNING - tlv(122): csg.skip as height  4 > 3 lvn /dd/Geometry/OverflowTanks/lvLsoOflTnk0xc0ad990 lvidx 140 
    [2017-06-13 18:01:13,966] p29002 {./sc.py:194} WARNING - tlv(124): csg.skip as height  7 > 3 lvn /dd/Geometry/OverflowTanks/lvGdsOflTnk0xc3d52a0 lvidx 142 
    [2017-06-13 18:01:14,009] p29002 {./sc.py:237} INFO - add_tree_gdml DONE maxdepth:0 maxcsgheight:3 nodesCount: 1660 tlvCount:171  tgNd:Nd ndIdx:  0 soIdx:0 nch:11 par:-1 matrix:[-1.0, 1.2246468525851679e-16, 0.0, 0.0, -1.2246468525851679e-16, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2871.0, 0.0, -41.0, 1.0]  
    [2017-06-13 18:01:14,009] p29002 {./sc.py:273} INFO - dump_all lvns 14 


     ## these ones should work as csg.intersectiontree with all leaves complemented apart from the leftmost   

     /dd/Geometry/AdDetails/lvRadialShieldUnit0xc3d7ec0           : 1 : di(di(di(di(di(di(di(cy,cy),cy),cy),cy),cy),cy),cy) height:7 totnodes:255  
     /dd/Geometry/AdDetails/lvTopReflector0xbf9be68               : 1 : di(di(di(di(di(cy,cy),cy),cy),cy),cy) height:5 totnodes:63  
     /dd/Geometry/AdDetails/lvTopRefGap0xbf9c648                  : 1 : di(di(di(di(di(cy,cy),cy),cy),cy),cy) height:5 totnodes:63  
     /dd/Geometry/AdDetails/lvTopESR0xc21fb88                     : 1 : di(di(di(di(di(di(di(di(di(cy,cy),cy),cy),cy),cy),cy),cy),cy),cy) height:9 totnodes:1023  
     /dd/Geometry/AdDetails/lvBotReflector0xc3cd4c0               : 1 : di(di(di(di(di(cy,cy),bo),bo),bo),bo) height:5 totnodes:63  
     /dd/Geometry/AdDetails/lvBotRefGap0xc34bc68                  : 1 : di(di(di(di(di(cy,cy),bo),bo),bo),bo) height:5 totnodes:63  
     /dd/Geometry/AdDetails/lvBotESR0xbfa74c0                     : 1 : di(di(di(di(di(di(di(di(cy,cy),bo),bo),bo),bo),cy),cy),cy) height:8 totnodes:511  
     /dd/Geometry/AdDetails/lvSstTopCirRibBase0xc2649f0           : 1 : di(di(di(di(di(cy,cy),bo),bo),bo),bo) height:5 totnodes:63  

     ## directly uniontree 
  
     /dd/Geometry/CalibrationSources/lvLedSourceAssy0xc306328     : 1 : un(un(un(un(un(un(un(un(un(un(cy,zs),zs),cy),zs),zs),cy),cy),zs),zs),cy) height:10 totnodes:2047  
     /dd/Geometry/CalibrationSources/lvGe68SourceAssy0xc2d4ad0    : 1 : un(un(un(un(un(un(un(un(un(un(cy,zs),zs),cy),zs),zs),cy),cy),zs),zs),cy) height:10 totnodes:2047  
     /dd/Geometry/CalibrationSources/lvAmCCo60SourceAssy0xc0b1da0 : 1 : un(un(un(un(un(un(un(un(un(un(cy,zs),zs),cy),zs),zs),cy),cy),zs),zs),cy) height:10 totnodes:2047  
     /dd/Geometry/OverflowTanks/lvOflTnkContainer0xc17cee8        : 1 : un(un(un(un(un(cy,cy),cy),cy),cy),cy) height:5 totnodes:63  

     ### hmm not so easy ...

     /dd/Geometry/OverflowTanks/lvLsoOflTnk0xc0ad990              : 1 : un(un(un(di(cy,cy),di(cy,cy)),di(cy,cy)),di(cy,cy)) height:4 totnodes:31  
     /dd/Geometry/OverflowTanks/lvGdsOflTnk0xc3d52a0              : 1 : un(un(un(un(un(un(di(cy,cy),di(cy,cy)),di(cy,cy)),di(cy,cy)),di(cy,cy)),di(cy,cy)),di(cy,cy)) height:7 totnodes:255  

     ## need detection of tree patterns that are easily simplified ...
     ##  is_mono_operator_tree ?



Non Mono Balancing ?
------------------------


This one is fairly balanced already, how to detect that ?

::

    [2017-06-14 20:07:59,671] p94010 {/Users/blyth/opticks/analytic/csg.py:648} INFO - /dd/Geometry/OverflowTanks/lvLsoOflTnk0xc0ad990 name:LsoOflTnk0xc17d928
    un(un(un(di(cy,cy),di(cy,cy)),di(cy,cy)),di(cy,cy)) height:4 totnodes:31 

                                                 un            
                                 un                      di    
                 un                      di          cy      cy
         di              di          cy      cy                
     cy      cy      cy      cy                                

    [2017-06-14 20:07:59,672] p94010 {/Users/blyth/opticks/analytic/csg.py:648} INFO - /dd/Geometry/OverflowTanks/lvLsoOflTnk0xc0ad990 (converted to positive form) name:LsoOflTnk0xc17d928
    un(un(un(in(cy,!cy),in(cy,!cy)),in(cy,!cy)),in(cy,!cy)) height:4 totnodes:31 

                                                 un            
                                 un                      in    
                 un                      in          cy     !cy
         in              in          cy     !cy                
     cy     !cy      cy     !cy                                





    [2017-06-14 20:07:59,672] p94010 {/Users/blyth/opticks/analytic/treebuilder.py:21} WARNING - balancing of non-mono operator trees not implemented
    [2017-06-14 20:07:59,672] p94010 {./sc.py:300} WARNING - cannot balance




Hmm this definitely needs balancing::

    [2017-06-14 20:07:59,672] p94010 {/Users/blyth/opticks/analytic/csg.py:648} INFO - /dd/Geometry/OverflowTanks/lvGdsOflTnk0xc3d52a0 name:GdsOflTnk0xc3d5160
    un(un(un(un(un(un(di(cy,cy),di(cy,cy)),di(cy,cy)),di(cy,cy)),di(cy,cy)),di(cy,cy)),di(cy,cy)) height:7 totnodes:255 

                                                                                                 un            
                                                                                 un                      di    
                                                                 un                      di          cy      cy
                                                 un                      di          cy      cy                
                                 un                      di          cy      cy                                
                 un                      di          cy      cy                                                
         di              di          cy      cy                                                                
     cy      cy      cy      cy                                                                                
    [2017-06-14 20:07:59,673] p94010 {/Users/blyth/opticks/analytic/csg.py:648} INFO - /dd/Geometry/OverflowTanks/lvGdsOflTnk0xc3d52a0 (converted to positive form) name:GdsOflTnk0xc3d5160
    un(un(un(un(un(un(in(cy,!cy),in(cy,!cy)),in(cy,!cy)),in(cy,!cy)),in(cy,!cy)),in(cy,!cy)),in(cy,!cy)) height:7 totnodes:255 

                                                                                                 un            
                                                                                 un                      in    
                                                                 un                      in          cy     !cy
                                                 un                      in          cy     !cy                
                                 un                      in          cy     !cy                                
                 un                      in          cy     !cy                                                
         in              in          cy     !cy                                                                
     cy     !cy      cy     !cy      

                                                                          
    [2017-06-14 20:07:59,674] p94010 {/Users/blyth/opticks/analytic/treebuilder.py:21} WARNING - balancing of non-mono operator trees not implemented
    [2017-06-14 20:07:59,674] p94010 {./sc.py:300} WARNING - cannot balance

    
6 unions of 7 intersections, arranging like below can get to height 5 ... "a unionmajor tree"


                         un 
                 un                un
             un      un        un       un
           un  un  un  un       
         




Checking Deep Volumes with tboolean-deep
-------------------------------------------

::

    Node 4428 : dig a06f pig e31d depth 11 nchild 1  
    pv:PhysVol /dd/Geometry/AD/lvOIL#pvBotReflector0xc34c068
     Position mm 0.0 0.0 -2027.5  None 
    lv:[62] Volume /dd/Geometry/AdDetails/lvBotReflector0xc3cd4c0 /dd/Materials/Acrylic0xc02ab98 BotRefHols0xc3cd380
       [224] Subtraction BotRefHols0xc3cd380  
         l:[222] Subtraction BotReflector-ChildForBotRefHols0xc3cd1b8  
         l:[220] Subtraction BotReflector-ChildForBotRefHols0xc3ccff0  
         l:[218] Subtraction BotReflector-ChildForBotRefHols0xc0d5f30  
         l:[216] Tube BotReflector0xc0d4ac8 mm rmin 19.25 rmax 2250.0  x 0.0 y 0.0 z 20.0  
         r:[217] Box BoxHolInBotRef10xc2ce6d0 mm rmin 0.0 rmax 0.0  x 90.0 y 384.0 z 21.0  
         r:[219] Box BoxHolInBotRef20xc3ccfb0 mm rmin 0.0 rmax 0.0  x 90.0 y 384.0 z 21.0  
         r:[221] Box BoxHolInBotRef30xc3cd130 mm rmin 0.0 rmax 0.0  x 384.0 y 90.0 z 21.0  
         r:[223] Box BoxHolInBotRef40xc3cd2f8 mm rmin 0.0 rmax 0.0  x 384.0 y 90.0 z 21.0  
       [8] Material /dd/Materials/Acrylic0xc02ab98 solid
       PhysVol /dd/Geometry/AdDetails/lvBotReflector#pvBotRefGap0xbfa6458
     None None  : Position mm 0.0 0.0 -2027.5   
    [2017-05-04 15:09:54,667] p66920 {/Users/blyth/opticks/ana/pmt/treebase.py:154} INFO - rprogeny numProgeny:3 (maxnode:0 maxdepth:0 skip:{'count': 0, 'depth': 0, 'total': 0} ) 
    [2017-05-04 15:09:54,667] p66920 {/Users/blyth/opticks/dev/csg/translate_gdml.py:73} INFO -  subtree 3 nodes 
    [2017-05-04 15:09:54,667] p66920 {/Users/blyth/opticks/dev/csg/translate_gdml.py:81} INFO - [ 0] converting solid 'BotRefHols0xc3cd380' 


    BotRefHols0xc3cd380
    di(di(di(di(di(cy ,cy ) ,bo ) ,bo ) ,bo ) ,bo )height:5 totnodes:63  
                                         di    
                                 di          bo
                         di          bo        
                 di          bo                
         di          bo                        
     cy      cy                                

    BotRefGapCutHols0xc34bb28
    di(di(di(di(di(cy ,cy ) ,bo ) ,bo ) ,bo ) ,bo )height:5 totnodes:63  
                                         di    
                                 di          bo
                         di          bo        
                 di          bo                
         di          bo                        
     cy      cy                                

    BotESRCutHols0xbfa7368
    di(di(di(di(di(di(di(di(cy ,cy ) ,bo ) ,bo ) ,bo ) ,bo ) ,cy ) ,cy ) ,cy )height:8 totnodes:511  
                                                                 di    
                                                         di          cy
                                                 di          cy        
                                         di          cy                
                                 di          bo                        
                         di          bo                                
                 di          bo                                        
         di          bo                                                
     cy      cy                                                        [2017-05-04 15:09:54,671] p66920 {/Users/blyth/opticks/dev/csg/csg.py:243} INFO - CSG.Serialize : writing 4 trees to directory /tmp/blyth/opticks/tboolean-deep-8 
    analytic=1_csgpath=/tmp/blyth/opticks/tboolean-deep-8_name=tboolean-deep-8_mode=PyCsgInBox
    simon:csg blyth$ 




sc.py KLUDGE SKIPPING deep CSG until work out how to balance
----------------------------------------------------------------------

::

    simon:issues blyth$ tgltf-;tgltf-gdml-
    args: 
    [2017-05-24 11:01:03,663] p77724 {/Users/blyth/opticks/analytic/gdml.py:959} INFO - parsing gdmlpath /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.gdml 
    [2017-05-24 11:01:03,699] p77724 {/Users/blyth/opticks/analytic/gdml.py:973} INFO - wrapping gdml element  
    [2017-05-24 11:01:04,550] p77724 {/Users/blyth/opticks/analytic/sc.py:230} INFO - add_tree_gdml START maxdepth:0 maxcsgheight:3 nodesCount:    0 targetNode: Node  1 : dig 450a pig 369b depth 1 nchild 2  
    pv:PhysVol /dd/Structure/Sites/db-rock0xc15d358
     Position mm -16520.0 -802110.0 -2110.0  Rotation deg 0.0 0.0 -122.9  
    lv:[247] Volume /dd/Geometry/Sites/lvNearSiteRock0xc030350 /dd/Materials/Rock0xc0300c8 near_rock0xc04ba08
       [705] Subtraction near_rock0xc04ba08  
         l:[703] Box near_rock_main0xc21d4f0 mm rmin 0.0 rmax 0.0  x 50000.0 y 50000.0 z 50000.0  
         r:[704] Box near_rock_void0xc21d6c8 mm rmin 0.0 rmax 0.0  x 50010.0 y 50010.0 z 12010.0  
       [35] Material /dd/Materials/Rock0xc0300c8 solid
       PhysVol /dd/Geometry/Sites/lvNearSiteRock#pvNearHallTop0xbf89820
     Position mm 2500.0 -500.0 7500.0  None 
       PhysVol /dd/Geometry/Sites/lvNearSiteRock#pvNearHallBot0xcd2fa58
     Position mm 0.0 0.0 -5150.0  None  : Position mm -16520.0 -802110.0 -2110.0   
    [2017-05-24 11:01:04,553] p77724 {/Users/blyth/opticks/analytic/sc.py:206} WARNING - tlv(  2) csg.skip as height  4 > 3 lvn /dd/Geometry/PoolDetails/lvNearTopCover0xc137060 lvidx 0 
    [2017-05-24 11:01:05,114] p77724 {/Users/blyth/opticks/analytic/sc.py:206} WARNING - tlv( 24) csg.skip as height  4 > 3 lvn /dd/Geometry/Pool/lvNearPoolDead0xc2dc490 lvidx 236 
    [2017-05-24 11:01:05,116] p77724 {/Users/blyth/opticks/analytic/sc.py:206} WARNING - tlv( 25) csg.skip as height  4 > 3 lvn /dd/Geometry/Pool/lvNearPoolLiner0xc21e9d0 lvidx 234 
    [2017-05-24 11:01:05,120] p77724 {/Users/blyth/opticks/analytic/sc.py:206} WARNING - tlv( 26) csg.skip as height 12 > 3 lvn /dd/Geometry/Pool/lvNearPoolOWS0xbf93840 lvidx 232 
    [2017-05-24 11:01:05,121] p77724 {/Users/blyth/opticks/analytic/sc.py:206} WARNING - tlv( 27) csg.skip as height  4 > 3 lvn /dd/Geometry/Pool/lvNearPoolCurtain0xc2ceef0 lvidx 213 
    [2017-05-24 11:01:05,125] p77724 {/Users/blyth/opticks/analytic/sc.py:206} WARNING - tlv( 28) csg.skip as height 12 > 3 lvn /dd/Geometry/Pool/lvNearPoolIWS0xc28bc60 lvidx 211 
    [2017-05-24 11:01:05,424] p77724 {/Users/blyth/opticks/analytic/sc.py:206} WARNING - tlv( 66) csg.skip as height  7 > 3 lvn /dd/Geometry/AdDetails/lvRadialShieldUnit0xc3d7ec0 lvidx 56 
    [2017-05-24 11:01:05,433] p77724 {/Users/blyth/opticks/analytic/sc.py:206} WARNING - tlv( 67) csg.skip as height  5 > 3 lvn /dd/Geometry/AdDetails/lvTopReflector0xbf9be68 lvidx 59 
    [2017-05-24 11:01:05,434] p77724 {/Users/blyth/opticks/analytic/sc.py:206} WARNING - tlv( 68) csg.skip as height  5 > 3 lvn /dd/Geometry/AdDetails/lvTopRefGap0xbf9c648 lvidx 58 
    [2017-05-24 11:01:05,437] p77724 {/Users/blyth/opticks/analytic/sc.py:206} WARNING - tlv( 69) csg.skip as height  9 > 3 lvn /dd/Geometry/AdDetails/lvTopESR0xc21fb88 lvidx 57 
    [2017-05-24 11:01:05,438] p77724 {/Users/blyth/opticks/analytic/sc.py:206} WARNING - tlv( 70) csg.skip as height  5 > 3 lvn /dd/Geometry/AdDetails/lvBotReflector0xc3cd4c0 lvidx 62 
    [2017-05-24 11:01:05,439] p77724 {/Users/blyth/opticks/analytic/sc.py:206} WARNING - tlv( 71) csg.skip as height  5 > 3 lvn /dd/Geometry/AdDetails/lvBotRefGap0xc34bc68 lvidx 61 
    [2017-05-24 11:01:05,441] p77724 {/Users/blyth/opticks/analytic/sc.py:206} WARNING - tlv( 72) csg.skip as height  8 > 3 lvn /dd/Geometry/AdDetails/lvBotESR0xbfa74c0 lvidx 60 
    [2017-05-24 11:01:05,455] p77724 {/Users/blyth/opticks/analytic/sc.py:206} WARNING - tlv( 79) csg.skip as height  5 > 3 lvn /dd/Geometry/AdDetails/lvSstTopCirRibBase0xc2649f0 lvidx 69 
    [2017-05-24 11:01:05,486] p77724 {/Users/blyth/opticks/analytic/sc.py:206} WARNING - tlv(110) csg.skip as height 10 > 3 lvn /dd/Geometry/CalibrationSources/lvLedSourceAssy0xc306328 lvidx 105 
    [2017-05-24 11:01:05,492] p77724 {/Users/blyth/opticks/analytic/sc.py:206} WARNING - tlv(117) csg.skip as height 10 > 3 lvn /dd/Geometry/CalibrationSources/lvGe68SourceAssy0xc2d4ad0 lvidx 112 
    [2017-05-24 11:01:05,498] p77724 {/Users/blyth/opticks/analytic/sc.py:206} WARNING - tlv(124) csg.skip as height 10 > 3 lvn /dd/Geometry/CalibrationSources/lvAmCCo60SourceAssy0xc0b1da0 lvidx 132 
    [2017-05-24 11:01:05,510] p77724 {/Users/blyth/opticks/analytic/sc.py:206} WARNING - tlv(148) csg.skip as height  5 > 3 lvn /dd/Geometry/OverflowTanks/lvOflTnkContainer0xc17cee8 lvidx 145 
    [2017-05-24 11:01:05,512] p77724 {/Users/blyth/opticks/analytic/sc.py:206} WARNING - tlv(150) csg.skip as height  4 > 3 lvn /dd/Geometry/OverflowTanks/lvLsoOflTnk0xc0ad990 lvidx 140 
    [2017-05-24 11:01:05,514] p77724 {/Users/blyth/opticks/analytic/sc.py:206} WARNING - tlv(152) csg.skip as height  7 > 3 lvn /dd/Geometry/OverflowTanks/lvGdsOflTnk0xc3d52a0 lvidx 142 
    [2017-05-24 11:01:06,487] p77724 {/Users/blyth/opticks/analytic/sc.py:206} WARNING - tlv(207) csg.skip as height  5 > 3 lvn /dd/Geometry/PoolDetails/lvTablePanel0xc0101d8 lvidx 200 
    [2017-05-24 11:01:07,685] p77724 {/Users/blyth/opticks/analytic/sc.py:206} WARNING - tlv(247) csg.skip as height  4 > 3 lvn /dd/Geometry/RadSlabs/lvNearRadSlab90xc15c208 lvidx 245 
    [2017-05-24 11:01:07,686] p77724 {/Users/blyth/opticks/analytic/sc.py:232} INFO - add_tree_gdml DONE maxdepth:0 maxcsgheight:3 nodesCount:12229 tlvCount:248  tgNd:Nd ndIdx:  0 soIdx:0 nch:2 par:-1 matrix:[-0.5431744456291199, 0.8396198749542236, 0.0, 0.0, -0.8396198749542236, -0.5431744456291199, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -16520.0, -802110.0, -2110.0, 1.0]  
    [2017-05-24 11:01:07,686] p77724 {/Users/blyth/opticks/analytic/sc.py:254} INFO - saving to /tmp/blyth/opticks/tgltf/tgltf-gdml--.gltf 
    [2017-05-24 11:01:07,929] p77724 {/Users/blyth/opticks/analytic/sc.py:250} INFO - save_extras /tmp/blyth/opticks/tgltf/extras  : saved 248 
    /tmp/blyth/opticks/tgltf/tgltf-gdml--.gltf
    simon:issues blyth$ 



Deep Volumes, 22 out of 249 are have tree height > 3 
-------------------------------------------------------

Of the 22:

* 16 are difference only trees, maximally unbalanced, progressive subtraction of boxes
* 4 are union only trees, maximally unbalanced
* 2 are mixed unions of difference of cylinders : these are not so unbalanced

::

    [2017-05-04 15:40:50,454] p67638 {/Users/blyth/opticks/dev/csg/scene.py:139} INFO - analyse_solids nflatsolids:707 ntops:249 ndeep:22 

     1 : /dd/Geometry/PoolDetails/lvNearTopCover0xc137060             : di(di(di(di(bo,bo),bo),bo),bo)height:4 totnodes:31  
     2 : /dd/Geometry/AdDetails/lvRadialShieldUnit0xc3d7ec0           : di(di(di(di(di(di(di(cy,cy),cy),cy),cy),cy),cy),cy)height:7 totnodes:255  
     3 : /dd/Geometry/AdDetails/lvTopESR0xc21fb88                     : di(di(di(di(di(di(di(di(di(cy,cy),cy),cy),cy),cy),cy),cy),cy),cy)height:9 totnodes:1023  
     4 : /dd/Geometry/AdDetails/lvTopRefGap0xbf9c648                  : di(di(di(di(di(cy,cy),cy),cy),cy),cy)height:5 totnodes:63  
     5 : /dd/Geometry/AdDetails/lvTopReflector0xbf9be68               : di(di(di(di(di(cy,cy),cy),cy),cy),cy)height:5 totnodes:63  
     6 : /dd/Geometry/AdDetails/lvBotESR0xbfa74c0                     : di(di(di(di(di(di(di(di(cy,cy),bo),bo),bo),bo),cy),cy),cy)height:8 totnodes:511  
     7 : /dd/Geometry/AdDetails/lvBotRefGap0xc34bc68                  : di(di(di(di(di(cy,cy),bo),bo),bo),bo)height:5 totnodes:63  
     8 : /dd/Geometry/AdDetails/lvBotReflector0xc3cd4c0               : di(di(di(di(di(cy,cy),bo),bo),bo),bo)height:5 totnodes:63  
     9 : /dd/Geometry/AdDetails/lvSstTopCirRibBase0xc2649f0           : di(di(di(di(di(cy,cy),bo),bo),bo),bo)height:5 totnodes:63  
    16 : /dd/Geometry/PoolDetails/lvTablePanel0xc0101d8               : di(di(di(di(di(bo,bo),bo),bo),bo),bo)height:5 totnodes:63  
    17 : /dd/Geometry/Pool/lvNearPoolIWS0xc28bc60                     : di(di(di(di(di(di(di(di(di(di(di(di(bo,bo),bo),bo),bo),bo),bo),bo),bo),bo),bo),bo),bo)height:12 totnodes:8191  
    18 : /dd/Geometry/Pool/lvNearPoolCurtain0xc2ceef0                 : di(di(di(di(bo,bo),bo),bo),bo)height:4 totnodes:31  
    19 : /dd/Geometry/Pool/lvNearPoolOWS0xbf93840                     : di(di(di(di(di(di(di(di(di(di(di(di(bo,bo),bo),bo),bo),bo),bo),bo),bo),bo),bo),bo),bo)height:12 totnodes:8191  
    20 : /dd/Geometry/Pool/lvNearPoolLiner0xc21e9d0                   : di(di(di(di(bo,bo),bo),bo),bo)height:4 totnodes:31  
    21 : /dd/Geometry/Pool/lvNearPoolDead0xc2dc490                    : di(di(di(di(bo,bo),bo),bo),bo)height:4 totnodes:31  
    22 : /dd/Geometry/RadSlabs/lvNearRadSlab90xc15c208                : di(di(di(di(bo,bo),bo),bo),bo)height:4 totnodes:31  

    10 : /dd/Geometry/CalibrationSources/lvLedSourceAssy0xc306328     : un(un(un(un(un(un(un(un(un(un(cy,zs),zs),cy),zs),zs),cy),cy),zs),zs),cy)height:10 totnodes:2047  
    11 : /dd/Geometry/CalibrationSources/lvGe68SourceAssy0xc2d4ad0    : un(un(un(un(un(un(un(un(un(un(cy,zs),zs),cy),zs),zs),cy),cy),zs),zs),cy)height:10 totnodes:2047  
    12 : /dd/Geometry/CalibrationSources/lvAmCCo60SourceAssy0xc0b1da0 : un(un(un(un(un(un(un(un(un(un(cy,zs),zs),cy),zs),zs),cy),cy),zs),zs),cy)height:10 totnodes:2047  
    15 : /dd/Geometry/OverflowTanks/lvOflTnkContainer0xc17cee8        : un(un(un(un(un(cy,cy),cy),cy),cy),cy)height:5 totnodes:63  

    13 : /dd/Geometry/OverflowTanks/lvLsoOflTnk0xc0ad990              : un(un(un(di(cy,cy),di(cy,cy)),di(cy,cy)),di(cy,cy))height:4 totnodes:31  
    14 : /dd/Geometry/OverflowTanks/lvGdsOflTnk0xc3d52a0              : un(un(un(un(un(un(di(cy,cy),di(cy,cy)),di(cy,cy)),di(cy,cy)),di(cy,cy)),di(cy,cy)),di(cy,cy))height:7 totnodes:255  





::

    [2017-05-04 13:28:13,914] p63916 {/Users/blyth/opticks/ana/pmt/gdml.py:911} INFO - parsing gdmlpath /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.gdml 


flat lozenge::

    solid.idx:8  cn.totnodes:31 solid.name:near_top_cover_box0xc23f970 ideep:1 lvidx:0 lvn:/dd/Geometry/PoolDetails/lvNearTopCover0xc137060 
                                 di    
                         di          bo
                 di          bo        
         di          bo                
     bo      bo


tambourine with 6 holes::
                        
    solid.idx:156  cn.totnodes:255 solid.name:RadialShieldUnit0xc3d7da8 ideep:2 lvidx:56 lvn:/dd/Geometry/AdDetails/lvRadialShieldUnit0xc3d7ec0 
                                                         di    
                                                 di          cy
                                         di          cy        
                                 di          cy                
                         di          cy                        
                 di          cy                                
         di          cy                                        
     cy      cy                                                



3 solids each for top and bot reflectors::

    solid.idx:173  cn.totnodes:1023 solid.name:TopESRCutHols0xbf9de10 ideep:3 lvidx:57 lvn:/dd/Geometry/AdDetails/lvTopESR0xc21fb88 
                                                                         di    
                                                                 di          cy
                                                         di          cy        
                                                 di          cy                
                                         di          cy                        
                                 di          cy                                
                         di          cy                                        
                 di          cy                                                
         di          cy                                                        
     cy      cy                                                                
    solid.idx:182  cn.totnodes:63 solid.name:TopRefGapCutHols0xbf9cef8 ideep:4 lvidx:58 lvn:/dd/Geometry/AdDetails/lvTopRefGap0xbf9c648 
                                         di    
                                 di          cy
                         di          cy        
                 di          cy                
         di          cy                        
     cy      cy                                
    solid.idx:191  cn.totnodes:63 solid.name:TopRefCutHols0xbf9bd50 ideep:5 lvidx:59 lvn:/dd/Geometry/AdDetails/lvTopReflector0xbf9be68 
                                         di    
                                 di          cy
                         di          cy        
                 di          cy                
         di          cy                        
     cy      cy                                



    solid.idx:206  cn.totnodes:511 solid.name:BotESRCutHols0xbfa7368 ideep:6 lvidx:60 lvn:/dd/Geometry/AdDetails/lvBotESR0xbfa74c0 
                                                                 di    
                                                         di          cy
                                                 di          cy        
                                         di          cy                
                                 di          bo                        
                         di          bo                                
                 di          bo                                        
         di          bo                                                
     cy      cy                                                        
    solid.idx:215  cn.totnodes:63 solid.name:BotRefGapCutHols0xc34bb28 ideep:7 lvidx:61 lvn:/dd/Geometry/AdDetails/lvBotRefGap0xc34bc68 
                                         di    
                                 di          bo
                         di          bo        
                 di          bo                
         di          bo                        
     cy      cy                                
    solid.idx:224  cn.totnodes:63 solid.name:BotRefHols0xc3cd380 ideep:8 lvidx:62 lvn:/dd/Geometry/AdDetails/lvBotReflector0xc3cd4c0 
                                         di    
                                 di          bo
                         di          bo        
                 di          bo                
         di          bo                        
     cy      cy                                





    solid.idx:259  cn.totnodes:63 solid.name:SstTopCirRibBase0xc264f78 ideep:9 lvidx:69 lvn:/dd/Geometry/AdDetails/lvSstTopCirRibBase0xc2649f0 
                                         di    
                                 di          bo
                         di          bo        
                 di          bo                
         di          bo                        
     cy      cy                                

    solid.idx:349  cn.totnodes:2047 solid.name:led-source-assy0xc3061d0 ideep:10 lvidx:105 lvn:/dd/Geometry/CalibrationSources/lvLedSourceAssy0xc306328 
                                                                                 un    
                                                                         un          cy
                                                                 un          zs        
                                                         un          zs                
                                                 un          cy                        
                                         un          cy                                
                                 un          zs                                        
                         un          zs                                                
                 un          cy                                                        
         un          zs                                                                
     cy      zs                                                                        

    solid.idx:380  cn.totnodes:2047 solid.name:source-assy0xc2d5d78 ideep:11 lvidx:112 lvn:/dd/Geometry/CalibrationSources/lvGe68SourceAssy0xc2d4ad0 
                                                                                 un    
                                                                         un          cy
                                                                 un          zs        
                                                         un          zs                
                                                 un          cy                        
                                         un          cy                                
                                 un          zs                                        
                         un          zs                                                
                 un          cy                                                        
         un          zs                                                                
     cy      zs                                                                     

    solid.idx:428  cn.totnodes:2047 solid.name:amcco60-source-assy0xc0b1df8 ideep:12 lvidx:132 lvn:/dd/Geometry/CalibrationSources/lvAmCCo60SourceAssy0xc0b1da0 
                                                                                 un    
                                                                         un          cy
                                                                 un          zs        
                                                         un          zs                
                                                 un          cy                        
                                         un          cy                                
                                 un          zs                                        
                         un          zs                                                
                 un          cy                                                        
         un          zs                                                                
     cy      zs                                                         
               
    solid.idx:442  cn.totnodes:31 solid.name:LsoOflTnk0xc17d928 ideep:13 lvidx:140 lvn:/dd/Geometry/OverflowTanks/lvLsoOflTnk0xc0ad990 
                                                 un            
                                 un                      di    
                 un                      di          cy      cy
         di              di          cy      cy                
     cy      cy      cy      cy                                

    solid.idx:460  cn.totnodes:255 solid.name:GdsOflTnk0xc3d5160 ideep:14 lvidx:142 lvn:/dd/Geometry/OverflowTanks/lvGdsOflTnk0xc3d52a0 
                                                                                                 un            
                                                                                 un                      di    
                                                                 un                      di          cy      cy
                                                 un                      di          cy      cy                
                                 un                      di          cy      cy                                
                 un                      di          cy      cy                                                
         di              di          cy      cy                                                                
     cy      cy      cy      cy                                                                                

    solid.idx:479  cn.totnodes:63 solid.name:OflTnkContainer0xc17cf50 ideep:15 lvidx:145 lvn:/dd/Geometry/OverflowTanks/lvOflTnkContainer0xc17cee8 
                                         un    
                                 un          cy
                         un          cy        
                 un          cy                
         un          cy                        
     cy      cy                                

    solid.idx:548  cn.totnodes:63 solid.name:table_panel_box0xc00f558 ideep:16 lvidx:200 lvn:/dd/Geometry/PoolDetails/lvTablePanel0xc0101d8 
                                         di    
                                 di          bo
                         di          bo        
                 di          bo                
         di          bo                        
     bo      bo                                

    solid.idx:587  cn.totnodes:8191 solid.name:near_pool_iws_box0xc288ce8 ideep:17 lvidx:211 lvn:/dd/Geometry/Pool/lvNearPoolIWS0xc28bc60 
                                                                                                 di    
                                                                                         di          bo
                                                                                 di          bo        
                                                                         di          bo                
                                                                 di          bo                        
                                                         di          bo                                
                                                 di          bo                                        
                                         di          bo                                                
                                 di          bo                                                        
                         di          bo                                                                
                 di          bo                                                                        
         di          bo                                                                                
     bo      bo                                                                                        

    solid.idx:597  cn.totnodes:31 solid.name:near_pool_curtain_box0xc2cef48 ideep:18 lvidx:213 lvn:/dd/Geometry/Pool/lvNearPoolCurtain0xc2ceef0 
                                 di    
                         di          bo
                 di          bo        
         di          bo                
     bo      bo                        

    solid.idx:664  cn.totnodes:8191 solid.name:near_pool_ows_box0xbf8c8a8 ideep:19 lvidx:232 lvn:/dd/Geometry/Pool/lvNearPoolOWS0xbf93840 
                                                                                                 di    
                                                                                         di          bo
                                                                                 di          bo        
                                                                         di          bo                
                                                                 di          bo                        
                                                         di          bo                                
                                                 di          bo                                        
                                         di          bo                                                
                                 di          bo                                                        
                         di          bo                                                                
                 di          bo                                                                        
         di          bo                                                                                
     bo      bo                                                                                        

    solid.idx:674  cn.totnodes:31 solid.name:near_pool_liner_box0xc2dcc28 ideep:20 lvidx:234 lvn:/dd/Geometry/Pool/lvNearPoolLiner0xc21e9d0 
                                 di    
                         di          bo
                 di          bo        
         di          bo                
     bo      bo                        

    solid.idx:684  cn.totnodes:31 solid.name:near_pool_dead_box0xbf8a280 ideep:21 lvidx:236 lvn:/dd/Geometry/Pool/lvNearPoolDead0xc2dc490 
                                 di    
                         di          bo
                 di          bo        
         di          bo                
     bo      bo                        

    solid.idx:701  cn.totnodes:31 solid.name:near-radslab-box-90xcd31ea0 ideep:22 lvidx:245 lvn:/dd/Geometry/RadSlabs/lvNearRadSlab90xc15c208 
                                 di    
                         di          bo
                 di          bo        
         di          bo                
     bo      bo                        
    [2017-05-04 13:28:14,179] p63916 {/Users/blyth/opticks/dev/csg/scene.py:206} INFO - analyse_solids nflatsolids:707 ntops:249 ndeep:22 
    [2017-05-04 13:28:14,470] p63916 {/Users/blyth/opticks/dev/csg/scene.py:221} INFO - save_lvsolids nlvs:249 



Enumerating Distinct Top Solids
-----------------------------------


Enumeration of all the top solids with scene.py SNode.tops

* total:249 matches the number of LV
* regarding the serialization, perhaps just dont start with solids, instead start with the 249 lv and their solids


::

    In [60]: topidx = [top.idx for top in SNode.tops()]

    In [61]: lvsolids = [lv.solid.idx for lv in gdml.volumes.values()]

    In [62]: topidx == lvsolids
    Out[62]: True


::

    [2017-05-04 12:30:24,226] p63604 {/Users/blyth/opticks/dev/csg/scene.py:199} INFO - save_solids nsolids:707 ndeep:229 ntops:249

    In [9]: len(gdml.volumes)
    Out[9]: 249


Counts with increasing number of subsolids, extends to 24 subsolids::

    In [49]: [(_,len(SNode.tops(ssmin=_))) for _ in range(26)]
    Out[49]: 
    [(0, 249),
     (1, 88),
     (2, 88),
     (3, 47),
     (4, 47),
     (5, 26),
     (6, 26),
     (7, 21),
     (8, 21),
     (9, 11),
     (10, 11),
     (11, 9),
     (12, 9),
     (13, 7),
     (14, 7),
     (15, 6),
     (16, 6),
     (17, 5),
     (18, 5),
     (19, 5),
     (20, 5),
     (21, 2),
     (22, 2),
     (23, 2),
     (24, 2),
     (25, 0)]





::


    In [13]: gdml.solids(664).as_ncsg()
    Out[13]: di(di(di(di(di(di(di(di(di(di(di(di(bo ,bo ) ,bo ) ,bo ) ,bo ) ,bo ) ,bo ) ,bo ) ,bo ) ,bo ) ,bo ) ,bo ) ,bo )


    solid.idx:664  25:640-664    cn.totnodes:8191 solid.name:near_pool_ows_box0xbf8c8a8
                                                                                                 664  
                                                                                                  | 
                                                                                                 di    
                                                                                         di          bo
                                                                                 di          bo        \
                                                                         di          bo                663
                                                                 di          bo                        
                                                         di          bo                                
                                                 di          bo                                        
                                         di          bo                                                
                                 di          bo                                                        
        642              di          bo                                                                
         |       di          bo                                                                        
         di          bo                                                                                
     bo      bo                                        
     |       | 
     640     641



Big node trees 
------------------


::

    [2017-05-03 20:04:44,940] p60750 {/Users/blyth/opticks/dev/csg/csg.py:348} INFO - save /tmp/blyth/opticks/dev/csg/scene/solids/647 1 
    [2017-05-03 20:04:44,942] p60750 {/Users/blyth/opticks/dev/csg/csg.py:348} INFO - save /tmp/blyth/opticks/dev/csg/scene/solids/648 31 
    [2017-05-03 20:04:44,944] p60750 {/Users/blyth/opticks/dev/csg/csg.py:348} INFO - save /tmp/blyth/opticks/dev/csg/scene/solids/649 1 
    [2017-05-03 20:04:44,946] p60750 {/Users/blyth/opticks/dev/csg/csg.py:348} INFO - save /tmp/blyth/opticks/dev/csg/scene/solids/650 63 
    [2017-05-03 20:04:44,948] p60750 {/Users/blyth/opticks/dev/csg/csg.py:348} INFO - save /tmp/blyth/opticks/dev/csg/scene/solids/651 1 
    [2017-05-03 20:04:44,950] p60750 {/Users/blyth/opticks/dev/csg/csg.py:348} INFO - save /tmp/blyth/opticks/dev/csg/scene/solids/652 127 
    [2017-05-03 20:04:44,952] p60750 {/Users/blyth/opticks/dev/csg/csg.py:348} INFO - save /tmp/blyth/opticks/dev/csg/scene/solids/653 1 
    [2017-05-03 20:04:44,955] p60750 {/Users/blyth/opticks/dev/csg/csg.py:348} INFO - save /tmp/blyth/opticks/dev/csg/scene/solids/654 255 
    [2017-05-03 20:04:44,957] p60750 {/Users/blyth/opticks/dev/csg/csg.py:348} INFO - save /tmp/blyth/opticks/dev/csg/scene/solids/655 1 
    [2017-05-03 20:04:44,960] p60750 {/Users/blyth/opticks/dev/csg/csg.py:348} INFO - save /tmp/blyth/opticks/dev/csg/scene/solids/656 511 
    [2017-05-03 20:04:44,963] p60750 {/Users/blyth/opticks/dev/csg/csg.py:348} INFO - save /tmp/blyth/opticks/dev/csg/scene/solids/657 1 
    [2017-05-03 20:04:44,966] p60750 {/Users/blyth/opticks/dev/csg/csg.py:348} INFO - save /tmp/blyth/opticks/dev/csg/scene/solids/658 1023 
    [2017-05-03 20:04:44,968] p60750 {/Users/blyth/opticks/dev/csg/csg.py:348} INFO - save /tmp/blyth/opticks/dev/csg/scene/solids/659 1 
    [2017-05-03 20:04:44,971] p60750 {/Users/blyth/opticks/dev/csg/csg.py:348} INFO - save /tmp/blyth/opticks/dev/csg/scene/solids/660 2047 
    [2017-05-03 20:04:44,974] p60750 {/Users/blyth/opticks/dev/csg/csg.py:348} INFO - save /tmp/blyth/opticks/dev/csg/scene/solids/661 1 
    [2017-05-03 20:04:44,978] p60750 {/Users/blyth/opticks/dev/csg/csg.py:348} INFO - save /tmp/blyth/opticks/dev/csg/scene/solids/662 4095 
    [2017-05-03 20:04:44,981] p60750 {/Users/blyth/opticks/dev/csg/csg.py:348} INFO - save /tmp/blyth/opticks/dev/csg/scene/solids/663 1 
    [2017-05-03 20:04:44,985] p60750 {/Users/blyth/opticks/dev/csg/csg.py:348} INFO - save /tmp/blyth/opticks/dev/csg/scene/solids/664 8191 
    [2017-05-03 20:04:44,988] p60750 {/Users/blyth/opticks/dev/csg/csg.py:348} INFO - save /tmp/blyth/opticks/dev/csg/scene/solids/665 1 
    [2017-05-03 20:04:44,989] p60750 {/Users/blyth/opticks/dev/csg/csg.py:348} INFO - save /tmp/blyth/opticks/dev/csg/scene/solids/666 1 
    [2017-05-03 20:04:44,990] p60750 {/Users/blyth/opticks/dev/csg/csg.py:348} INFO - save /tmp/blyth/opticks/dev/csg/scene/solids/667 1 

::

    In [107]: t.filternodes_so("near_pool_ows")[0].name
    Out[107]: 'Node 3150 : dig 9ff6 pig 29c2 depth 5 nchild 2938 '

::

    In [108]: g.solids(658)
    Out[108]: 
    [658] Subtraction near_pool_ows-ChildFornear_pool_ows_box0xc356df8  
         l:[656] Subtraction near_pool_ows-ChildFornear_pool_ows_box0xc2c4a40  
         l:[654] Subtraction near_pool_ows-ChildFornear_pool_ows_box0xc21d530  
         l:[652] Subtraction near_pool_ows-ChildFornear_pool_ows_box0xc12e148  
         l:[650] Subtraction near_pool_ows-ChildFornear_pool_ows_box0xbf97a68  
         l:[648] Subtraction near_pool_ows-ChildFornear_pool_ows_box0xc12de98  
         l:[646] Subtraction near_pool_ows-ChildFornear_pool_ows_box0xc357900  
         l:[644] Subtraction near_pool_ows-ChildFornear_pool_ows_box0xc12f640  
         l:[642] Subtraction near_pool_ows-ChildFornear_pool_ows_box0xbf8c148  
         l:[640] Box near_pool_ows0xc2bc1d8 mm rmin 0.0 rmax 0.0  x 15832.0 y 9832.0 z 9912.0  
         r:[641] Box near_pool_ows_sub00xc55ebf8 mm rmin 0.0 rmax 0.0  x 4179.41484434 y 4179.41484434 z 9922.0  
         r:[643] Box near_pool_ows_sub10xc21e940 mm rmin 0.0 rmax 0.0  x 4179.41484434 y 4179.41484434 z 9922.0  
         r:[645] Box near_pool_ows_sub20xc2344b0 mm rmin 0.0 rmax 0.0  x 4179.41484434 y 4179.41484434 z 9922.0  
         r:[647] Box near_pool_ows_sub30xbf5f5b8 mm rmin 0.0 rmax 0.0  x 4179.41484434 y 4179.41484434 z 9922.0  
         r:[649] Box near_pool_ows_sub40xbf979e0 mm rmin 0.0 rmax 0.0  x 4176.10113585 y 4176.10113585 z 9912.0  
         r:[651] Box near_pool_ows_sub50xc12e0c0 mm rmin 0.0 rmax 0.0  x 4176.10113585 y 4176.10113585 z 9912.0  
         r:[653] Box near_pool_ows_sub60xc2a23c8 mm rmin 0.0 rmax 0.0  x 4176.10113585 y 4176.10113585 z 9912.0  
         r:[655] Box near_pool_ows_sub70xc21d660 mm rmin 0.0 rmax 0.0  x 4176.10113585 y 4176.10113585 z 9912.0  
         r:[657] Box near_pool_ows_sub80xc2c4b70 mm rmin 0.0 rmax 0.0  x 15824.0 y 10.0 z 9912.0  


    In [150]: s = g.solids(658)

    In [151]: s.subsolids
    Out[151]: [658, 656, 654, 652, 650, 648, 646, 644, 642, 640, 641, 643, 645, 647, 649, 651, 653, 655, 657] 

    In [153]: len(g.solids(658).subsolids)
    Out[153]: 19



    In [109]: cn = g.solids(658).as_ncsg()

    In [110]: cn
    Out[110]: di(di(di(di(di(di(di(di(di(bo ,bo ) ,bo ) ,bo ) ,bo ) ,bo ) ,bo ) ,bo ) ,bo ) ,bo ) 

    In [111]: cn.analyse()

    In [112]: cn
    Out[112]: di(di(di(di(di(di(di(di(di(bo ,bo ) ,bo ) ,bo ) ,bo ) ,bo ) ,bo ) ,bo ) ,bo ) ,bo )height:9 totnodes:1023  


    In [114]: print cn.txt
                                                                         di    
                                                                 di          bo
                                                         di          bo        
                                                 di          bo                
                                         di          bo                        
                                 di          bo                                
                         di          bo                                        
                 di          bo                                                
         di          bo                                                        
     bo      bo                                    












Hmm need a better way to get from a solid to a list of the lvs that use it...


/tmp/g4_00.gdml::

     1800     <box lunit="mm" name="near_pool_iws_sub30xc2cac98" x="3347.67401109936" y="3347.67401109936" z="8918"/>
     1801     <subtraction name="near_pool_iws-ChildFornear_pool_iws_box0xc287ea8">
     1802       <first ref="near_pool_iws-ChildFornear_pool_iws_box0xc287d20"/>
     1803       <second ref="near_pool_iws_sub30xc2cac98"/>
     1804       <position name="near_pool_iws-ChildFornear_pool_iws_box0xc287ea8_pos" unit="mm" x="-6912" y="-3912" z="0"/>
     1805       <rotation name="near_pool_iws-ChildFornear_pool_iws_box0xc287ea8_rot" unit="deg" x="0" y="0" z="45"/>
     1806     </subtraction>
     1807     <box lunit="mm" name="near_pool_iws_sub40xc287fe8" x="3344.36030260037" y="3344.36030260037" z="8908"/>
     1808     <subtraction name="near_pool_iws-ChildFornear_pool_iws_box0xc288070">
     1809       <first ref="near_pool_iws-ChildFornear_pool_iws_box0xc287ea8"/>
     1810       <second ref="near_pool_iws_sub40xc287fe8"/>
     1811       <position name="near_pool_iws-ChildFornear_pool_iws_box0xc288070_pos" unit="mm" x="6908" y="3908" z="-100"/>
     1812       <rotation name="near_pool_iws-ChildFornear_pool_iws_box0xc288070_rot" unit="deg" x="0" y="0" z="45"/>
     1813     </subtraction>
     1814     <box lunit="mm" name="near_pool_iws_sub50xc2881b0" x="3344.36030260037" y="3344.36030260037" z="8908"/>
     1815     <subtraction name="near_pool_iws-ChildFornear_pool_iws_box0xc288238">
     1816       <first ref="near_pool_iws-ChildFornear_pool_iws_box0xc288070"/>
     1817       <second ref="near_pool_iws_sub50xc2881b0"/>
     1818       <position name="near_pool_iws-ChildFornear_pool_iws_box0xc288238_pos" unit="mm" x="6908" y="-3908" z="-100"/>
     1819       <rotation name="near_pool_iws-ChildFornear_pool_iws_box0xc288238_rot" unit="deg" x="0" y="0" z="45"/>
     1820     </subtraction>



Checking detdesc, repeated bevel subtraction of rotated boxes::

     33 <!-- Far Pool top cover -->
     34 <logvol name="lvFarTopCover" material="PPE">
     35   <subtraction name="far_top_cover_box">
     36     <box name="far_top_cover" sizeX="FarPoolDeadSizeX" sizeY="FarPoolDeadSizeY" sizeZ="TopCoverSizeZ" />
     37     <box name="far_top_cover_sub0" sizeX="PoolDeadBevelSize" sizeY="PoolDeadBevelSize" sizeZ="1*cm+TopCoverSizeZ" />
     38     <posXYZ x="0.5*FarPoolDeadSizeX" y="0.5*FarPoolDeadSizeY" z="0*m" />
     39     <rotXYZ rotZ="45*degree" />
     40     <box name="far_top_cover_sub1" sizeX="PoolDeadBevelSize" sizeY="PoolDeadBevelSize" sizeZ="1*cm+TopCoverSizeZ" />
     41     <posXYZ x="0.5*FarPoolDeadSizeX" y="-0.5*FarPoolDeadSizeY" z="0*m" />
     42     <rotXYZ rotZ="45*degree" />
     43     <box name="far_top_cover_sub2" sizeX="PoolDeadBevelSize" sizeY="PoolDeadBevelSize" sizeZ="1*cm+TopCoverSizeZ" />
     44     <posXYZ x="-0.5*FarPoolDeadSizeX" y="0.5*FarPoolDeadSizeY" z="0*m" />
     45     <rotXYZ rotZ="45*degree" />
     46     <box name="far_top_cover_sub3" sizeX="PoolDeadBevelSize" sizeY="PoolDeadBevelSize" sizeZ="1*cm+TopCoverSizeZ" />
     47     <posXYZ x="-0.5*FarPoolDeadSizeX" y="-0.5*FarPoolDeadSizeY" z="0*m" />
     48     <rotXYZ rotZ="45*degree" />
     49   </subtraction>
     50 </logvol>



