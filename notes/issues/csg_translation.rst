CSG Translation Issues (solid level)
========================================


Big node trees 
------------------

From scene.py see some surprising totnodes in the CSG trees::

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



Checking detdesc::

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



