OKX4Test_j1808_x4gen_to_investigate_PMT_csg
==============================================

boot with GDML, direct convert to GGeo, persist to geocache with codegen 
--------------------------------------------------------------------------

::

    opticksdata- ; OKX4Test --gdmlpath $(opticksdata-j) --g4codegen


generate CMakeLists.txt and scripts to build the generated code into executables for every solid
--------------------------------------------------------------------------------------------------

::

    x4gen--


x4gen- shakedown
-------------------

::

    [blyth@localhost ~]$ x4gen-deep
     so:016 lv:016 rmx:11 bmx:02 soName: sFasteners0x4c01080                  ## looks too good to be true : must be loosing geometry ??
    ----------------------------- height greater than 7 skipped in kernel -------
     so:018 lv:018 rmx:04 bmx:04 soName: PMT_20inch_inner1_solid0x4cb3610
     so:019 lv:019 rmx:04 bmx:04 soName: PMT_20inch_inner2_solid0x4cb3870
     so:020 lv:020 rmx:03 bmx:03 soName: PMT_20inch_body_solid0x4c90e50
     so:021 lv:021 rmx:03 bmx:03 soName: PMT_20inch_pmt_solid0x4c81b40


survey the 40 solids of j1808
-------------------------------------------------------------

::

    [blyth@localhost 1]$ cat solids.txt 
    written by GGeoGLTF::solidRecTable 
    num_solid 40
     so:000 lv:000 rmx:00 bmx:00 soName: Upper_LS_tube0x5b2e9f0
     so:001 lv:001 rmx:01 bmx:01 soName: Upper_Steel_tube0x5b2eb10
     so:002 lv:002 rmx:01 bmx:01 soName: Upper_Tyvek_tube0x5b2ec30
     so:003 lv:003 rmx:00 bmx:00 soName: Upper_Chimney0x5b2e8e0
     so:004 lv:004 rmx:00 bmx:00 soName: sBar0x5b34ab0
     so:005 lv:005 rmx:00 bmx:00 soName: sBar0x5b34920
     so:006 lv:006 rmx:00 bmx:00 soName: sModuleTape0x5b34790
     so:007 lv:007 rmx:00 bmx:00 soName: sModule0x5b34600
     so:008 lv:008 rmx:00 bmx:00 soName: sPlane0x5b34470
     so:009 lv:009 rmx:00 bmx:00 soName: sWall0x5b342e0
     so:010 lv:010 rmx:01 bmx:01 soName: sAirTT0x5b34000

           LV=10 x4gen-csg  
           
           looks like a box subtract a cylinder 
           with serious case of coincidence subtraction speckles

     so:011 lv:011 rmx:00 bmx:00 soName: sExpHall0x4bcd390
     so:012 lv:012 rmx:00 bmx:00 soName: sTopRock0x4bccfc0
     so:013 lv:013 rmx:01 bmx:01 soName: sTarget0x4bd4340
     so:014 lv:014 rmx:01 bmx:01 soName: sAcrylic0x4bd3cd0
     so:015 lv:015 rmx:01 bmx:01 soName: sStrut0x4bd4b80

     so:016 lv:016 rmx:11 bmx:02 soName: sFasteners0x4c01080

           LV=16 x4gen-csg  

            bizarre shape, raytrace get:
               evaluative_csg : perfect tree height 11 exceeds current limit 
            .. not using the balanced or balancing failed ?
            had to force quit the raytrace

     so:017 lv:017 rmx:02 bmx:02 soName: sMask0x4ca38d0

           LV=17 x4gen-csg  

           observatory dome shape, polygonization failed, raytrace looks OK 

     so:018 lv:018 rmx:04 bmx:04 soName: PMT_20inch_inner1_solid0x4cb3610

           LV=18 x4gen-csg  

           wow : profligate use of a depth 4 tree (31 nodes)
           when a single node would do: ellipsoid with z range

     so:019 lv:019 rmx:04 bmx:04 soName: PMT_20inch_inner2_solid0x4cb3870

           LV=19 x4gen-csg  

           speckle neck 
           also : profligate use of CSG intersection to chop the cathode off 

     so:020 lv:020 rmx:03 bmx:03 soName: PMT_20inch_body_solid0x4c90e50

           LV=20 x4gen-csg  

           speckle neck from the torus subtraction, 
           but this time the speckle disappears when closeup 
           and from some angles 

     so:021 lv:021 rmx:03 bmx:03 soName: PMT_20inch_pmt_solid0x4c81b40

           LV=21 x4gen-csg  

           ditto : speckle neck 


     so:022 lv:022 rmx:00 bmx:00 soName: sMask_virtual0x4c36e10


     so:023 lv:023 rmx:00 bmx:00 soName: PMT_3inch_inner1_solid_ell_helper0x510ae30
     so:024 lv:024 rmx:00 bmx:00 soName: PMT_3inch_inner2_solid_ell_helper0x510af10
     so:025 lv:025 rmx:00 bmx:00 soName: PMT_3inch_body_solid_ell_ell_helper0x510ada0
     so:026 lv:026 rmx:00 bmx:00 soName: PMT_3inch_cntr_solid0x510afa0
     so:027 lv:027 rmx:01 bmx:01 soName: PMT_3inch_pmt_solid0x510aae0
     so:028 lv:028 rmx:01 bmx:01 soName: sChimneyAcrylic0x5b310c0
     so:029 lv:029 rmx:00 bmx:00 soName: sChimneyLS0x5b312e0
     so:030 lv:030 rmx:01 bmx:01 soName: sChimneySteel0x5b314f0
     so:031 lv:031 rmx:00 bmx:00 soName: sWaterTube0x5b30eb0
     so:032 lv:032 rmx:00 bmx:00 soName: svacSurftube0x5b3bf50
     so:033 lv:033 rmx:00 bmx:00 soName: sSurftube0x5b3ab80
     so:034 lv:034 rmx:01 bmx:01 soName: sInnerWater0x4bd3660
     so:035 lv:035 rmx:01 bmx:01 soName: sReflectorInCD0x4bd3040
     so:036 lv:036 rmx:00 bmx:00 soName: sOuterWaterPool0x4bd2960
     so:037 lv:037 rmx:00 bmx:00 soName: sPoolLining0x4bd1eb0
     so:038 lv:038 rmx:00 bmx:00 soName: sBottomRock0x4bcd770
     so:039 lv:039 rmx:00 bmx:00 soName: sWorld0x4bc2350



