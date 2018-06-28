okg4.OKX4Test : checking direct from G4 conversion, starting from a GDML loaded geometry
===========================================================================================

Investigate some missing geometry in the visualization
----------------------------------------------------------

Run without args::

    OKX4Test


Observations from visualization
----------------------------------

* missing some geometry at edge of pool, maybe girders are the X4Mesh skipped lvIdx ? 
* Coloring very different, maybe not picking up dyb prefs for "g4live" detector 
* geometry normals visible
* triangulated ray trace is notably faster that usual (missing some expensive geometry ?)    

How to debug ?
---------------

* investigate the skips (soIdx 27, soIdx 29) 

  * big box with 12 rotated boxes subtracted one by one
  * there is only one each of those meshes (?), so a placeholder for them doesnt explain what is seen
  * dumping the nnode for the polygonization skips shows very big trees  
  * huge boxes with 45 degree rotated boxes subtracted  : they are the near_pool_ows and near_pool_iws
    so they do not explain the missing girders

* check volume counts, mesh counts and usage totals 


GDML near_pool_ows0xc2bc1d8
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   epsilon:DayaBay_VGDX_20140414-1300 blyth$ cp g4_00.gdml /tmp/


Its a box with 12 rotated boxes subtracted one by one::

     1981     <box lunit="mm" name="near_pool_ows0xc2bc1d8" x="15832" y="9832" z="9912"/>
     1982     <box lunit="mm" name="near_pool_ows_sub00xc55ebf8" x="4179.41484434453" y="4179.41484434453" z="9922"/>
     1983     <subtraction name="near_pool_ows-ChildFornear_pool_ows_box0xbf8c148">
     1984       <first ref="near_pool_ows0xc2bc1d8"/>
     1985       <second ref="near_pool_ows_sub00xc55ebf8"/>
     1986       <position name="near_pool_ows-ChildFornear_pool_ows_box0xbf8c148_pos" unit="mm" x="7916" y="4916" z="0"/>
     1987       <rotation name="near_pool_ows-ChildFornear_pool_ows_box0xbf8c148_rot" unit="deg" x="0" y="0" z="45"/>
     1988     </subtraction>
     1989     <box lunit="mm" name="near_pool_ows_sub10xc21e940" x="4179.41484434453" y="4179.41484434453" z="9922"/>
     1990     <subtraction name="near_pool_ows-ChildFornear_pool_ows_box0xc12f640">
     1991       <first ref="near_pool_ows-ChildFornear_pool_ows_box0xbf8c148"/>
     1992       <second ref="near_pool_ows_sub10xc21e940"/>
     1993       <position name="near_pool_ows-ChildFornear_pool_ows_box0xc12f640_pos" unit="mm" x="7916" y="-4916" z="0"/>
     1994       <rotation name="near_pool_ows-ChildFornear_pool_ows_box0xc12f640_rot" unit="deg" x="0" y="0" z="45"/>
     1995     </subtraction>
     .....
     2050     <box lunit="mm" name="near_pool_ows_sub100xbf8c640" x="15824" y="10" z="9912"/>
     2051     <subtraction name="near_pool_ows-ChildFornear_pool_ows_box0xbf8c6c8">
     2052       <first ref="near_pool_ows-ChildFornear_pool_ows_box0xbf8c500"/>
     2053       <second ref="near_pool_ows_sub100xbf8c640"/>
     2054       <position name="near_pool_ows-ChildFornear_pool_ows_box0xbf8c6c8_pos" unit="mm" x="7913" y="0" z="-100"/>
     2055       <rotation name="near_pool_ows-ChildFornear_pool_ows_box0xbf8c6c8_rot" unit="deg" x="0" y="0" z="90"/>
     2056     </subtraction>
     2057     <box lunit="mm" name="near_pool_ows_sub110xbf8c820" x="15824" y="10" z="9912"/>
     2058     <subtraction name="near_pool_ows_box0xbf8c8a8">
     2059       <first ref="near_pool_ows-ChildFornear_pool_ows_box0xbf8c6c8"/>
     2060       <second ref="near_pool_ows_sub110xbf8c820"/>
     2061       <position name="near_pool_ows_box0xbf8c8a8_pos" unit="mm" x="-7913" y="0" z="-100"/>
     2062       <rotation name="near_pool_ows_box0xbf8c8a8_rot" unit="deg" x="0" y="0" z="90"/>
     2063     </subtraction>



soIdx 27 : near_pool_ows0xc2bc1d8_box3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::


    2018-06-28 13:57:10.865 ERROR [385204] [*X4PhysicalVolume::convertNode@503]  csgnode::dump START for skipped solid soIdx 27
     du [ 0:di di] C OPER  v:0  bb  mi (  -7916.000 -4916.000 -4956.000) mx (   7916.000  4916.000  4956.000) si (  15832.000  9832.000  9912.000)

     du [ 0:di di] C OPER  v:0  bb  mi (  -7916.000 -4916.000 -4956.000) mx (   7916.000  4916.000  4956.000) si (  15832.000  9832.000  9912.000)

     du [ 0:di di] C OPER  v:0  bb  mi (  -7916.000 -4916.000 -4956.000) mx (   7916.000  4916.000  4956.000) si (  15832.000  9832.000  9912.000)

     du [ 0:di di] C OPER  v:0  bb  mi (  -7916.000 -4916.000 -4956.000) mx (   7916.000  4916.000  4956.000) si (  15832.000  9832.000  9912.000)

     du [ 0:di di] C OPER  v:0  bb  mi (  -7916.000 -4916.000 -4956.000) mx (   7916.000  4916.000  4956.000) si (  15832.000  9832.000  9912.000)

     du [ 0:di di] C OPER  v:0  bb  mi (  -7916.000 -4916.000 -4956.000) mx (   7916.000  4916.000  4956.000) si (  15832.000  9832.000  9912.000)

     du [ 0:di di] C OPER  v:0  bb  mi (  -7916.000 -4916.000 -4956.000) mx (   7916.000  4916.000  4956.000) si (  15832.000  9832.000  9912.000)

     du [ 0:di di] C OPER  v:0  bb  mi (  -7916.000 -4916.000 -4956.000) mx (   7916.000  4916.000  4956.000) si (  15832.000  9832.000  9912.000)

     du [ 0:di di] C OPER  v:0  bb  mi (  -7916.000 -4916.000 -4956.000) mx (   7916.000  4916.000  4956.000) si (  15832.000  9832.000  9912.000)

     du [ 0:di di] C OPER  v:0  bb  mi (  -7916.000 -4916.000 -4956.000) mx (   7916.000  4916.000  4956.000) si (  15832.000  9832.000  9912.000)

     du [ 0:di di] C OPER  v:0  bb  mi (  -7916.000 -4916.000 -4956.000) mx (   7916.000  4916.000  4956.000) si (  15832.000  9832.000  9912.000)

     du [ 0:di di] C OPER  v:0  bb  mi (  -7916.000 -4916.000 -4956.000) mx (   7916.000  4916.000  4956.000) si (  15832.000  9832.000  9912.000)

     du [ 0:bo near_pool_ows0xc2bc1d8_box3] P PRIM  v:0  bb  mi (  -7916.000 -4916.000 -4956.000) mx (   7916.000  4916.000  4956.000) si (  15832.000  9832.000  9912.000)
     gt [ 0:bo near_pool_ows0xc2bc1d8_box3] P NO gtransform 
     du [ 0:bo near_pool_ows_sub00xc55ebf8_box3] P PRIM  v:0  bb  mi (   4960.707  1960.707 -4961.000) mx (  10871.293  7871.293  4961.000) si (   5910.586  5910.586  9922.000)
     gt [ 0:bo near_pool_ows_sub00xc55ebf8_box3] P      gt.t
                0.707   0.707   0.000   0.000 


soIdx 29 : near_pool_iws0xc2cab98_box3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    2018-06-28 13:57:10.914 ERROR [385204] [*X4PhysicalVolume::convertNode@503]  csgnode::dump START for skipped solid soIdx 29
     du [ 0:di di] C OPER  v:0  bb  mi (  -6912.000 -3912.000 -4454.000) mx (   6912.000  3912.000  4454.000) si (  13824.000  7824.000  8908.000)

     du [ 0:di di] C OPER  v:0  bb  mi (  -6912.000 -3912.000 -4454.000) mx (   6912.000  3912.000  4454.000) si (  13824.000  7824.000  8908.000)

     du [ 0:di di] C OPER  v:0  bb  mi (  -6912.000 -3912.000 -4454.000) mx (   6912.000  3912.000  4454.000) si (  13824.000  7824.000  8908.000)

     du [ 0:di di] C OPER  v:0  bb  mi (  -6912.000 -3912.000 -4454.000) mx (   6912.000  3912.000  4454.000) si (  13824.000  7824.000  8908.000)

     du [ 0:di di] C OPER  v:0  bb  mi (  -6912.000 -3912.000 -4454.000) mx (   6912.000  3912.000  4454.000) si (  13824.000  7824.000  8908.000)

     du [ 0:di di] C OPER  v:0  bb  mi (  -6912.000 -3912.000 -4454.000) mx (   6912.000  3912.000  4454.000) si (  13824.000  7824.000  8908.000)

     du [ 0:di di] C OPER  v:0  bb  mi (  -6912.000 -3912.000 -4454.000) mx (   6912.000  3912.000  4454.000) si (  13824.000  7824.000  8908.000)

     du [ 0:di di] C OPER  v:0  bb  mi (  -6912.000 -3912.000 -4454.000) mx (   6912.000  3912.000  4454.000) si (  13824.000  7824.000  8908.000)

     du [ 0:di di] C OPER  v:0  bb  mi (  -6912.000 -3912.000 -4454.000) mx (   6912.000  3912.000  4454.000) si (  13824.000  7824.000  8908.000)

     du [ 0:di di] C OPER  v:0  bb  mi (  -6912.000 -3912.000 -4454.000) mx (   6912.000  3912.000  4454.000) si (  13824.000  7824.000  8908.000)

     du [ 0:di di] C OPER  v:0  bb  mi (  -6912.000 -3912.000 -4454.000) mx (   6912.000  3912.000  4454.000) si (  13824.000  7824.000  8908.000)

     du [ 0:di di] C OPER  v:0  bb  mi (  -6912.000 -3912.000 -4454.000) mx (   6912.000  3912.000  4454.000) si (  13824.000  7824.000  8908.000)

     du [ 0:bo near_pool_iws0xc2cab98_box3] P PRIM  v:0  bb  mi (  -6912.000 -3912.000 -4454.000) mx (   6912.000  3912.000  4454.000) si (  13824.000  7824.000  8908.000)
     gt [ 0:bo near_pool_iws0xc2cab98_box3] P NO gtransform 


