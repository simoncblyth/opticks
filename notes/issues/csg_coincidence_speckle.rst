CSG Coincidence Speckle
==========================


FIXED : Very Thin Cylinder Speckles in CSG difference holes
---------------------------------------------------------------

* ~/opticks_refs/speckle_lvTopESR.png

* FIXED using new primitive CSG_DISC


Complex box cuts from hollowed cylinder speckles
----------------------------------------------------

* ~/opticks_refs/edge_speckle.png
* ~/opticks_refs/speckle_lvSstTopCirRibBase

* how to generalize nudges to avoid subtractions that lead to ghost speckle coincidence ???

* have implemeted bileaf uncoincidence detection and nudge fixing for box3 - box3, 
  but thats not directly applicable as the box cuts from an already 
  composite: cy - cy (for cylinder with inner radius)

  * could reorder this inner subtraction ?? ie cut out the boxes from the cylinder endcap

* suspect this situation is rather rare, so fixing it via metadata
  labelled nudge hints seems appropriate

* could add inner radius param handling to cylinder primitive, 
  but thats significant work, will probably eventually do this but not a priority


::

   tboolean-;tboolean-rip sc


    In [3]: target.lv.solid
    Out[3]: 
    [259] Subtraction SstTopCirRibBase0xc264f78  
         l:[257] Subtraction SstTopCirRibPri-ChildForSstTopCirRibBase0xc264e78  
         l:[255] Subtraction SstTopCirRibPri-ChildForSstTopCirRibBase0xbf755c0  
         l:[253] Subtraction SstTopCirRibPri-ChildForSstTopCirRibBase0xc354ef8  
         l:[251] Tube SstTopCirRibPri0xc2648b8 mm rmin 1200.0 rmax 1220.0  x 0.0 y 0.0 z 231.89  
         r:[252] Box Cutbox10xc264960 mm rmin 0.0 rmax 0.0  x 2460.0 y 20.0 z 231.89  
         r:[254] Box Cutbox20xc265a38 mm rmin 0.0 rmax 0.0  x 2460.0 y 100.0 z 20.0  
         r:[256] Box Cutbox30xc265b98 mm rmin 0.0 rmax 0.0  x 2460.0 y 20.0 z 231.89  
         r:[258] Box Cutbox40xc265cf8 mm rmin 0.0 rmax 0.0  x 2460.0 y 100.0 z 20.0  

    In [6]: obj.dump(detailed=True)
    [2017-06-16 20:34:57,767] p44626 {/Users/blyth/opticks/analytic/csg.py:712} INFO - CSG.dump name:SstTopCirRibBase0xc264f78_balanced
    in(in(in(cy,!cy),in(!bo,!bo)),in(!bo,!bo)) height:3 totnodes:15 
     intersection;SstTopCirRibBase0xc264f78_balanced   : None None  
        intersection;treebuilder_midop                 : None None  
           intersection;treebuilder_bileaf             : None None  
              cylinder;SstTopCirRibPri0xc2648b8_outer  : array([    0.,     0.,     0.,  1220.], dtype=float32) array([-115.945,  115.945,    0.   ,    0.   ], dtype=float32)  
              cylinder;SstTopCirRibPri0xc2648b8_inner  : array([    0.,     0.,     0.,  1200.], dtype=float32) array([-117.1044,  117.1044,    0.    ,    0.    ], dtype=float32)  
           intersection;treebuilder_bileaf             : None None  
              box3;Cutbox10xc264960                    : array([ 2460.  ,    20.  ,   231.89,     0.  ], dtype=float32) array([ 0.,  0.,  0.,  0.], dtype=float32)  
              box3;Cutbox20xc265a38                    : array([ 2460.,   100.,    20.,     0.], dtype=float32) array([ 0.,  0.,  0.,  0.], dtype=float32)  
        intersection;treebuilder_bileaf                : None None  
           box3;Cutbox30xc265b98                       : array([ 2460.  ,    20.  ,   231.89,     0.  ], dtype=float32) array([ 0.,  0.,  0.,  0.], dtype=float32)  
           box3;Cutbox40xc265cf8                       : array([ 2460.,   100.,    20.,     0.], dtype=float32) array([ 0.,  0.,  0.,  0.], dtype=float32)  

                                 in            
                 in                      in    
         in              in         !bo     !bo
     cy     !cy     !bo     !bo                


::

    In [7]: 231.89/2.
    Out[7]: 115.945

    In [9]: 1.01*231.89/2.
    Out[9]: 117.10445         # z size Tube subtraction inner is nudged up by 1%

    In [10]: 1220*2.        # 2460 is more than the diameter, each subtracted does cuts on both sides 
    Out[10]: 2440.0



::

    [2017-06-16 20:58:34,978] p45431 {/Users/blyth/opticks/analytic/csg.py:712} INFO - BALANCED name:SstTopCirRibBase0xc264f78_balanced
    in(in(in(cy,!cy),in(!bo,!bo)),in(!bo,!bo)) height:3 totnodes:15 
     intersection;SstTopCirRibBase0xc264f78_balanced   : None None 
    None 
        intersection;treebuilder_midop                 : None None 
    None 
           intersection;treebuilder_bileaf             : None None 
    None 
              cylinder;SstTopCirRibPri0xc2648b8_outer  : array([    0.,     0.,     0.,  1220.], dtype=float32) array([-115.945,  115.945,    0.   ,    0.   ], dtype=float32) 
    None 
              cylinder;SstTopCirRibPri0xc2648b8_inner  : array([    0.,     0.,     0.,  1200.], dtype=float32) array([-117.1044,  117.1044,    0.    ,    0.    ], dtype=float32) 
    None 
           intersection;treebuilder_bileaf             : None None 
    None 
              box3;Cutbox10xc264960                    : array([ 2460.  ,    20.  ,   231.89,     0.  ], dtype=float32) array([ 0.,  0.,  0.,  0.], dtype=float32) 
    array([[ 1.,  0.,  0.,  0.],
           [ 0.,  1.,  0.,  0.],
           [ 0.,  0.,  1.,  0.],
           [ 0.,  0.,  0.,  1.]], dtype=float32) 
              box3;Cutbox20xc265a38                    : array([ 2460.,   100.,    20.,     0.], dtype=float32) array([ 0.,  0.,  0.,  0.], dtype=float32) 
    array([[   1.   ,    0.   ,    0.   ,    0.   ],
           [   0.   ,    1.   ,    0.   ,    0.   ],
           [   0.   ,    0.   ,    1.   ,    0.   ],
           [   0.   ,    0.   , -105.945,    1.   ]], dtype=float32) 
        intersection;treebuilder_bileaf                : None None 
    None 
           box3;Cutbox30xc265b98                       : array([ 2460.  ,    20.  ,   231.89,     0.  ], dtype=float32) array([ 0.,  0.,  0.,  0.], dtype=float32) 
    array([[ 0.7071, -0.7071,  0.    ,  0.    ],
           [ 0.7071,  0.7071,  0.    ,  0.    ],
           [ 0.    ,  0.    ,  1.    ,  0.    ],
           [ 0.    ,  0.    ,  0.    ,  1.    ]], dtype=float32) 
           box3;Cutbox40xc265cf8                       : array([ 2460.,   100.,    20.,     0.], dtype=float32) array([ 0.,  0.,  0.,  0.], dtype=float32) 
    array([[   0.7071,   -0.7071,    0.    ,    0.    ],
           [   0.7071,    0.7071,    0.    ,    0.    ],
           [   0.    ,    0.    ,    1.    ,    0.    ],
           [   0.    ,    0.    , -105.945 ,    1.    ]], dtype=float32) 

                                 in            
                 in                      in    
         in              in         !bo     !bo
     cy     !cy     !bo     !bo                
    [2017-06-16 20:58:34,983] p45431 {/Users/blyth/opticks/analytic/csg.py:321} INFO - CSG.Serialize : writing 2 trees to directory /tmp/blyth/opticks/tboolean-sc-- 
    analytic=1_csgpath=/tmp/blyth/opticks/tboolean-sc--_name=tboolean-sc--_mode=PyCsgInBox





