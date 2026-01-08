geometry_positions_dumping
=============================


Dump all prim::

     TEST=descPrimRange ~/o/CSG/tests/CSGFoundryLoadTest.sh



Dump prim with ce.z in provided range::

    (ok) A[blyth@localhost CSG]$ TEST=descPrimRange CSGPrim__DescRange_CE_ZMIN_ZMAX=-24000,-17900 ~/o/CSG/tests/CSGFoundryLoadTest.sh
    ...

    ]CSGFoundry::descPrimRange solidIdx 10
    [CSGFoundry::descPrimRange solidIdx 11 so.primOffset  2735 so.numPrim   336
    [CSGPrim::Desc numPrim 336 mg_subs 0 EXTENT_DIFF :    200.000 [CSGPrim__DescRange_EXTENT_DIFF]
     [CSGPrim__DescRange_CE_ZMIN_ZMAX] CE_ZMIN : -24000.000 CE_ZMAX : -17900.000
       5 :  ; ce = np.array([     0.000,     0.000,-21312.000,  1225.791])  ; bb = np.array([ -1225.791, -1225.791,-21547.000,  1225.791,  1225.791,-21077.000])  # CSGPrim::descRangeNumPy  lvid  35 ridx    0 pidx     5 so[Waterdistributor_1]
      11 :  ; ce = np.array([     0.000,     0.000,-18838.000,  1225.791])  ; bb = np.array([ -1225.791, -1225.791,-19073.000,  1225.791,  1225.791,-18603.000])  # CSGPrim::descRangeNumPy  lvid 128 ridx    0 pidx    11 so[Waterdistributor_3]
       6 :  ; ce = np.array([     0.000,     0.000,-21312.000,  1219.998])  ; bb = np.array([ -1219.998, -1219.998,-21547.000,  1219.998,  1219.998,-21077.000])  # CSGPrim::descRangeNumPy  lvid  34 ridx    0 pidx     6 so[Waterdistributor_0]
      12 :  ; ce = np.array([     0.000,     0.000,-18838.000,  1219.998])  ; bb = np.array([ -1219.998, -1219.998,-19073.000,  1219.998,  1219.998,-18603.000])  # CSGPrim::descRangeNumPy  lvid 127 ridx    0 pidx    12 so[Waterdistributor_2]
       8 :  ; ce = np.array([  -124.232,     0.000,-20044.180,   438.970])  ; bb = np.array([  -306.500,   -63.924,-20483.150,    58.035,    63.924,-19605.211])  # CSGPrim::descRangeNumPy  lvid  81 ridx    0 pidx     8 so[CutTube2_Seg2_Upper]
       7 :  ; ce = np.array([  -262.750,     0.000,-20768.062,   399.250])  ; bb = np.array([  -662.000,   -66.500,-21073.500,   136.500,    66.500,-20462.623])  # CSGPrim::descRangeNumPy  lvid  39 ridx    0 pidx     7 so[WaterDistributorPartIIIUnion_Seg1]
      13 :  ; ce = np.array([   -88.520,     0.000,-19396.762,   280.561])  ; bb = np.array([  -302.041,  -125.000,-19677.322,   125.000,   125.000,-19116.201])  # CSGPrim::descRangeNumPy  lvid 135 ridx    0 pidx    13 so[WaterDistributorPartIIIUnion_Seg3]
     numPrim 336 mg_subs 0 EXTENT_DIFF :    200.000 [CSGPrim__DescRange_EXTENT_DIFF]
     [CSGPrim__DescRange_CE_ZMIN_ZMAX] CE_ZMIN : -24000.000 CE_ZMAX : -17900.000
    ]CSGPrim::Desc
    ]CSGFoundry::descPrimRange solidIdx 11
    [CSGFoundry::descPrimRange num_solid 12
    [CSGFoundry.descBase 
     CFBase       /home/blyth/junosw/InstallArea/yupd_bottompipe_adjust/.opticks/GEOM/J25_7_2_opticks_Debug
     OriginCFBase /home/blyth/junosw/InstallArea/yupd_bottompipe_adjust/.opticks/GEOM/J25_7_2_opticks_Debug
    ]CSGFoundry.descBase 
    ]CSGFoundryLoadTest::descPrimRange



Dump by grepping name
-----------------------

::

    (ok) A[blyth@localhost CSG]$ TEST=descPrimRange ~/o/CSG/tests/CSGFoundryLoadTest.sh | grep PartIII
     214 :  ; ce = np.array([   426.250,  -457.100, 21110.000,   640.000])  ; bb = np.array([   346.750,  -536.600, 20470.000,   505.750,  -377.600, 21750.000])  # CSGPrim::descRangeNumPy  lvid  28 ridx    0 pidx   214 so[WaterDistributorPipePartIII_0]
     215 :  ; ce = np.array([  -140.590,   608.980, 21110.000,   640.000])  ; bb = np.array([  -220.090,   529.480, 20470.000,   -61.090,   688.480, 21750.000])  # CSGPrim::descRangeNumPy  lvid  29 ridx    0 pidx   215 so[WaterDistributorPipePartIII_1]
     216 :  ; ce = np.array([  -511.970,  -358.490, 21110.000,   640.000])  ; bb = np.array([  -591.470,  -437.990, 20470.000,  -432.470,  -278.990, 21750.000])  # CSGPrim::descRangeNumPy  lvid  30 ridx    0 pidx   216 so[WaterDistributorPipePartIII_2]
       7 :  ; ce = np.array([  -262.750,     0.000,-20768.062,   399.250])  ; bb = np.array([  -662.000,   -66.500,-21073.500,   136.500,    66.500,-20462.623])  # CSGPrim::descRangeNumPy  lvid  39 ridx    0 pidx     7 so[WaterDistributorPartIIIUnion_Seg1]
      13 :  ; ce = np.array([   -88.520,     0.000,-19396.762,   280.561])  ; bb = np.array([  -302.041,  -125.000,-19677.322,   125.000,   125.000,-19116.201])  # CSGPrim::descRangeNumPy  lvid 135 ridx    0 pidx    13 so[WaterDistributorPartIIIUnion_Seg3]
    (ok) A[blyth@localhost CSG]$ 


cxt_min.sh simtrace targetting
---------------------------------

The normal approach to doing simtrace targetting is to set MOI eg::

    moi=WaterDistributorPartIIIUnion:0:-2    # this is the lower curly pipe : initially with enormous extent, now fixed 

But with shapes that are offset within the frame may need more precise targetting. Then try::

    moi=-262.750,0.000,-20768.062,399.250   # target CE of WaterDistributorPartIIIUnion obtained from : TEST=descPrimRange ~/o/CSG/tests/CSGFoundryLoadTest.sh

    cxt_min.sh

    GSGRID=1 cxt_min.sh pdb    # potentially increase extent to see more of the context



