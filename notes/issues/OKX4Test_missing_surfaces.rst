OKX4Test_missing_surfaces
============================

This may just be a problem of the test enviroment. Need to reconstruct the 
surfaces into G4 context (cfg4 has this capability ?) in addition to the 
MPT which are already reconstructed in order to get matching.


mm2/3/4 all have missing surfaces::

    epsilon:5 blyth$ AB_TAIL=4 ab-prim
    /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103/GPartsAnalytic/4
    prim (1, 4) part (1, 4, 4) tran (1, 3, 4, 4) 

    primIdx 0 prim array([0, 1, 0, 0], dtype=int32) partOffset 0 numParts 1 tranOffset 0 planOffset 0  
        Part 17  1             box3    83 IwsWater/PmtMtRib1Surface//UnstStainlessSteel   tz:     0.000      
    /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts/4
    prim (1, 4) part (1, 4, 4) tran (1, 3, 4, 4) 

    primIdx 0 prim array([0, 1, 0, 0], dtype=int32) partOffset 0 numParts 1 tranOffset 0 planOffset 0  
        Part 17  1             box3    72 IwsWater///UnstStainlessSteel   tz:     0.000      


    epsilon:5 blyth$ AB_TAIL=3 ab-prim
    /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103/GPartsAnalytic/3
    prim (1, 4) part (1, 4, 4) tran (1, 3, 4, 4) 

    primIdx 0 prim array([0, 1, 0, 0], dtype=int32) partOffset 0 numParts 1 tranOffset 0 planOffset 0  
        Part 17  1             box3    86 IwsWater/PmtMtRib3Surface//UnstStainlessSteel   tz:     0.000      
    /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts/3
    prim (1, 4) part (1, 4, 4) tran (1, 3, 4, 4) 

    primIdx 0 prim array([0, 1, 0, 0], dtype=int32) partOffset 0 numParts 1 tranOffset 0 planOffset 0  
        Part 17  1             box3    72 IwsWater///UnstStainlessSteel   tz:     0.000      


    epsilon:5 blyth$ AB_TAIL=2 ab-prim
    /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103/GPartsAnalytic/2
    prim (1, 4) part (1, 4, 4) tran (1, 3, 4, 4) 

    primIdx 0 prim array([0, 1, 0, 0], dtype=int32) partOffset 0 numParts 1 tranOffset 0 planOffset 0  
        Part 17  1             box3    85 IwsWater/PmtMtRib2Surface//UnstStainlessSteel   tz:     0.000      
    /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts/2
    prim (1, 4) part (1, 4, 4) tran (1, 3, 4, 4) 

    primIdx 0 prim array([0, 1, 0, 0], dtype=int32) partOffset 0 numParts 1 tranOffset 0 planOffset 0  
        Part 17  1             box3    72 IwsWater///UnstStainlessSteel   tz:     0.000      


