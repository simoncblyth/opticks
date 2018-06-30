OKX4Test_mm0_has_85k_parts_vs_12k
===================================

Large differnce in number of parts from the lack of 
tree balancing implementation in the direct approach.


::

    In [9]: pb[:20]
    Out[9]: 
    array([[ 0,  1,  0,  0],
           [ 1,  1,  1,  0],
           [ 2,  1,  2,  0],
           [ 3,  7,  3,  0],
           [10,  7,  5,  0],
           [17,  7,  7,  0],
           [24,  7,  9,  0],
           [31,  7, 11,  0],
           [38,  7, 14,  0],
           [45,  7, 15,  0],
           [52,  3, 16,  0],
           [55,  1, 17,  0],
           [56, 15, 18,  0],
           [71,  7, 20,  0],
           [78,  7, 21,  0],
           [85,  7, 23,  0],
           [92,  1, 26,  0],
           [93,  1, 27,  0],
           [94,  1, 28,  0],
           [95,  1, 29,  0]], dtype=int32)

    In [10]: pa[:20]
    Out[10]: 
    array([[ 0,  1,  0,  0],
           [ 1,  1,  1,  0],
           [ 2,  1,  2,  0],
           [ 3,  7,  3,  0],
           [10,  7,  5,  0],
           [17,  7,  7,  0],
           [24,  7,  9,  0],
           [31,  7, 11,  0],
           [38,  3, 14,  0],
           [41,  3, 15,  0],
           [44,  3, 16,  0],
           [47,  1, 17,  0],
           [48,  7, 18,  0],
           [55,  3, 20,  0],
           [58,  7, 21,  0],
           [65,  7, 23,  0],
           [72,  1, 26,  0],
           [73,  1, 27,  0],
           [74,  1, 28,  0],
           [75,  1, 29,  0]], dtype=int32)


::

    epsilon:GParts blyth$ AB_TAIL="0" ab-diff
    Files /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103/GPartsAnalytic/0/GParts.txt and /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts/0/GParts.txt differ
    Files /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103/GPartsAnalytic/0/partBuffer.npy and /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts/0/partBuffer.npy differ
    Files /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103/GPartsAnalytic/0/planBuffer.npy and /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts/0/planBuffer.npy differ
    Files /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103/GPartsAnalytic/0/primBuffer.npy and /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts/0/primBuffer.npy differ
    Files /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103/GPartsAnalytic/0/tranBuffer.npy and /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts/0/tranBuffer.npy differ
    /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103/GPartsAnalytic/0
            ./GParts.txt : 11984 
        ./planBuffer.npy : (672, 4) 
        ./partBuffer.npy : (11984, 4, 4) 
        ./tranBuffer.npy : (5344, 3, 4, 4) 
        ./primBuffer.npy : (3116, 4) 
    MD5 (GParts.txt) = 5eeee07e08a9a50278a2339dd0b47ac4
    MD5 (partBuffer.npy) = 8d837fba380dfc643968bd23f99d656f
    MD5 (planBuffer.npy) = 94e18d5e55d190c9ed73e04b45ebb404
    MD5 (primBuffer.npy) = e21f1c240c4d5e9450aff3ddc0fb78d6
    MD5 (tranBuffer.npy) = 77359e6d3d628e93cb7cf0a4a3824ab3
    /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts/0
            ./GParts.txt : 85264 
        ./planBuffer.npy : (672, 4) 
        ./partBuffer.npy : (85264, 4, 4) 
        ./tranBuffer.npy : (5344, 3, 4, 4) 
        ./primBuffer.npy : (3116, 4) 
    MD5 (GParts.txt) = 6f533aade1075bb4419f716f575ee114
    MD5 (partBuffer.npy) = 95d75b7805b1aca5754de4db4514c3a3
    MD5 (planBuffer.npy) = 43f2892dbf4b8e91231e5d830dee9e03
    MD5 (primBuffer.npy) = bb75be942f2a3efbf60bfc793ff58cbe
    MD5 (tranBuffer.npy) = 74a6d92ff0d830990e81e10434865714
    epsilon:0 blyth$ 
    epsilon:0 blyth$ 

