tboolean-proxy-scan
======================

Context
-----------

Have lots of deviation comparison machinery, need to 
get back into context with it.

* :doc:`deviation-comparison-review`


Issue : many ana fails from deviations
-----------------------------------------

* this is non-aligned comparison : 
  so it relies on accidental history aligment 
  making it suffer from poor stats  



::

    [blyth@localhost okg4]$ scan--
    scan-- : env PROXYLV=0 tboolean.sh --compute ========== RC 88
    scan-- : env PROXYLV=1 tboolean.sh --compute ========== RC 88
    scan-- : env PROXYLV=2 tboolean.sh --compute ========== RC 88
    scan-- : env PROXYLV=3 tboolean.sh --compute ========== RC 88
    scan-- : env PROXYLV=4 tboolean.sh --compute ========== RC 88
    scan-- : env PROXYLV=5 tboolean.sh --compute ========== RC 88
    scan-- : env PROXYLV=6 tboolean.sh --compute ========== RC 88
    scan-- : env PROXYLV=7 tboolean.sh --compute ========== RC 88
    scan-- : env PROXYLV=8 tboolean.sh --compute ========== RC 88
    scan-- : env PROXYLV=9 tboolean.sh --compute ========== RC 88
    scan-- : env PROXYLV=10 tboolean.sh --compute ========== RC 88
    scan-- : env PROXYLV=11 tboolean.sh --compute ========== RC 88
    scan-- : env PROXYLV=12 tboolean.sh --compute ========== RC 0
    scan-- : env PROXYLV=13 tboolean.sh --compute ========== RC 99
    scan-- : env PROXYLV=14 tboolean.sh --compute ========== RC 99
    scan-- : env PROXYLV=15 tboolean.sh --compute ========== RC 0
    scan-- : env PROXYLV=16 tboolean.sh --compute ========== RC 0
    scan-- : env PROXYLV=17 tboolean.sh --compute ========== RC 0
    scan-- : env PROXYLV=18 tboolean.sh --compute ========== RC 99
    scan-- : env PROXYLV=19 tboolean.sh --compute ========== RC 99
    scan-- : env PROXYLV=20 tboolean.sh --compute ========== RC 99
    scan-- : env PROXYLV=21 tboolean.sh --compute ========== RC 99
    scan-- : env PROXYLV=22 tboolean.sh --compute ========== RC 0
    scan-- : env PROXYLV=23 tboolean.sh --compute ========== RC 0
    scan-- : env PROXYLV=24 tboolean.sh --compute ========== RC 0
    scan-- : env PROXYLV=25 tboolean.sh --compute ========== RC 0
    scan-- : env PROXYLV=26 tboolean.sh --compute ========== RC 0
    scan-- : env PROXYLV=27 tboolean.sh --compute ========== RC 99
    scan-- : env PROXYLV=28 tboolean.sh --compute ========== RC 77
    scan-- : env PROXYLV=29 tboolean.sh --compute ========== RC 0
    scan-- : env PROXYLV=30 tboolean.sh --compute ========== RC 88
    scan-- : env PROXYLV=31 tboolean.sh --compute ========== RC 0
    scan-- : env PROXYLV=32 tboolean.sh --compute ========== RC 0
    scan-- : env PROXYLV=33 tboolean.sh --compute ========== RC 0
    scan-- : env PROXYLV=34 tboolean.sh --compute ========== RC 99
    scan-- : env PROXYLV=35 tboolean.sh --compute ========== RC 99
    scan-- : env PROXYLV=36 tboolean.sh --compute ========== RC 88
    scan-- : env PROXYLV=37 tboolean.sh --compute ========== RC 88
    scan-- : env PROXYLV=38 tboolean.sh --compute ========== RC 0
    scan-- : env PROXYLV=39 tboolean.sh --compute ========== RC 0


Duplicate that with the analysis RC as expected using all the events in /tmp/tboolean-proxy-0 etc..

::

    [blyth@localhost okg4]$ scan--
    scan-- : tboolean.py --pfx tboolean-proxy-0 ========== RC 88
    scan-- : tboolean.py --pfx tboolean-proxy-1 ========== RC 88
    scan-- : tboolean.py --pfx tboolean-proxy-2 ========== RC 88
    scan-- : tboolean.py --pfx tboolean-proxy-3 ========== RC 88
    scan-- : tboolean.py --pfx tboolean-proxy-4 ========== RC 88
    scan-- : tboolean.py --pfx tboolean-proxy-5 ========== RC 88
    scan-- : tboolean.py --pfx tboolean-proxy-6 ========== RC 88
    scan-- : tboolean.py --pfx tboolean-proxy-7 ========== RC 88
    scan-- : tboolean.py --pfx tboolean-proxy-8 ========== RC 88
    scan-- : tboolean.py --pfx tboolean-proxy-9 ========== RC 88
    scan-- : tboolean.py --pfx tboolean-proxy-10 ========== RC 88
    scan-- : tboolean.py --pfx tboolean-proxy-11 ========== RC 88
    scan-- : tboolean.py --pfx tboolean-proxy-12 ========== RC 0
    scan-- : tboolean.py --pfx tboolean-proxy-13 ========== RC 99
    scan-- : tboolean.py --pfx tboolean-proxy-14 ========== RC 99
    scan-- : tboolean.py --pfx tboolean-proxy-15 ========== RC 0
    scan-- : tboolean.py --pfx tboolean-proxy-16 ========== RC 0
    scan-- : tboolean.py --pfx tboolean-proxy-17 ========== RC 0
    scan-- : tboolean.py --pfx tboolean-proxy-18 ========== RC 99
    scan-- : tboolean.py --pfx tboolean-proxy-19 ========== RC 99
    scan-- : tboolean.py --pfx tboolean-proxy-20 ========== RC 99
    scan-- : tboolean.py --pfx tboolean-proxy-21 ========== RC 99
    scan-- : tboolean.py --pfx tboolean-proxy-22 ========== RC 0
    scan-- : tboolean.py --pfx tboolean-proxy-23 ========== RC 0
    scan-- : tboolean.py --pfx tboolean-proxy-24 ========== RC 0
    scan-- : tboolean.py --pfx tboolean-proxy-25 ========== RC 0
    scan-- : tboolean.py --pfx tboolean-proxy-26 ========== RC 0
    scan-- : tboolean.py --pfx tboolean-proxy-27 ========== RC 99
    scan-- : tboolean.py --pfx tboolean-proxy-28 ========== RC 77
    scan-- : tboolean.py --pfx tboolean-proxy-29 ========== RC 0
    scan-- : tboolean.py --pfx tboolean-proxy-30 ========== RC 88
    scan-- : tboolean.py --pfx tboolean-proxy-31 ========== RC 0
    scan-- : tboolean.py --pfx tboolean-proxy-32 ========== RC 0
    scan-- : tboolean.py --pfx tboolean-proxy-33 ========== RC 0
    scan-- : tboolean.py --pfx tboolean-proxy-34 ========== RC 99
    scan-- : tboolean.py --pfx tboolean-proxy-35 ========== RC 99
    scan-- : tboolean.py --pfx tboolean-proxy-36 ========== RC 88
    scan-- : tboolean.py --pfx tboolean-proxy-37 ========== RC 88
    scan-- : tboolean.py --pfx tboolean-proxy-38 ========== RC 0
    scan-- : tboolean.py --pfx tboolean-proxy-39 ========== RC 0


::

    [blyth@localhost ggeo]$ GMeshLibTest 
    ,,,
    2019-06-18 09:23:30.638 INFO  [416436] [test_dump1@103]  num_mesh 41
     0                       Upper_LS_tube0x5b2e9f0 ce0           0.0000,0.0000,0.0000,1750.0000 ce1           0.0000,0.0000,0.0000,1750.0000  0
     1                    Upper_Steel_tube0x5b2eb10 ce0           0.0000,0.0000,0.0000,1750.0000 ce1           0.0000,0.0000,0.0000,1750.0000  1
     2                    Upper_Tyvek_tube0x5b2ec30 ce0           0.0000,0.0000,0.0000,1750.0000 ce1           0.0000,0.0000,0.0000,1750.0000  2
     3                       Upper_Chimney0x5b2e8e0 ce0           0.0000,0.0000,0.0000,1750.0000 ce1           0.0000,0.0000,0.0000,1750.0000  3
     4                                sBar0x5b34ab0 ce0           0.0000,0.0000,0.0000,3430.0000 ce1           0.0000,0.0000,0.0000,3430.0000  4
     5                                sBar0x5b34920 ce0           0.0000,0.0000,0.0000,3430.0000 ce1           0.0000,0.0000,0.0000,3430.0000  5
     6                         sModuleTape0x5b34790 ce0           0.0000,0.0000,0.0000,3430.0000 ce1           0.0000,0.0000,0.0000,3430.0000  6
     7                             sModule0x5b34600 ce0           0.0000,0.0000,0.0000,3430.6001 ce1           0.0000,0.0000,0.0000,3430.6001  7
     8                              sPlane0x5b34470 ce0           0.0000,0.0000,0.0000,3430.6001 ce1           0.0000,0.0000,0.0000,3430.6001  8
     9                               sWall0x5b342e0 ce0           0.0000,0.0000,0.0000,3430.6001 ce1           0.0000,0.0000,0.0000,3430.6001  9
    10                              sAirTT0x5b34000 ce0          0.0000,0.0000,0.0000,24000.0000 ce1          0.0000,0.0000,0.0000,24000.0000 10
    11                            sExpHall0x4bcd390 ce0          0.0000,0.0000,0.0000,24000.0000 ce1          0.0000,0.0000,0.0000,24000.0000 11
    12                            sTopRock0x4bccfc0 ce0          0.0000,0.0000,0.0000,27000.0000+ce1          0.0000,0.0000,0.0000,27000.0000 12
    13                             sTarget0x4bd4340 ce0         0.0000,0.0000,60.0000,17760.0000 ce1          0.0000,0.0000,0.0000,17760.0000 13
    14                            sAcrylic0x4bd3cd0 ce0          0.0000,0.0000,0.0000,17820.0000 ce1          0.0000,0.0000,0.0000,17820.0000 14
    15                              sStrut0x4bd4b80 ce0            0.0000,0.0000,0.0000,600.0000+ce1            0.0000,0.0000,0.0000,600.0000 15
    16                          sFasteners0x4c01080 ce0          0.0000,0.0000,-92.5000,150.0000+ce1            0.0000,0.0000,0.0000,150.0000 16
    17                               sMask0x4ca38d0 ce0          0.0000,0.0000,-78.9500,274.9500+ce1            0.0000,0.0000,0.0000,274.9500 17
    18             PMT_20inch_inner1_solid0x4cb3610 ce0           0.0000,0.0000,89.5000,249.0000-ce1            0.0000,0.0000,0.0000,249.0000 18
    19             PMT_20inch_inner2_solid0x4cb3870 ce0         0.0000,0.0000,-167.0050,249.0000-ce1            0.0000,0.0000,0.0000,249.0000 19
    20               PMT_20inch_body_solid0x4c90e50 ce0          0.0000,0.0000,-77.5050,261.5050-ce1            0.0000,0.0000,0.0000,261.5050 20
    21                PMT_20inch_pmt_solid0x4c81b40 ce0          0.0000,0.0000,-77.5050,261.5060-ce1           0.0000,0.0000,-0.0000,261.5060 21
    22                       sMask_virtual0x4c36e10 ce0          0.0000,0.0000,-79.0000,275.0500+ce1            0.0000,0.0000,0.0000,275.0500 22
    23   PMT_3inch_inner1_solid_ell_helper0x510ae30 ce0            0.0000,0.0000,14.5216,38.0000+ce1             0.0000,0.0000,0.0000,38.0000 23
    24   PMT_3inch_inner2_solid_ell_helper0x510af10 ce0            0.0000,0.0000,-4.4157,38.0000+ce1             0.0000,0.0000,0.0000,38.0000 24
    25 PMT_3inch_body_solid_ell_ell_helper0x510ada0 ce0             0.0000,0.0000,4.0627,40.0000+ce1             0.0000,0.0000,0.0000,40.0000 25
    26                PMT_3inch_cntr_solid0x510afa0 ce0           0.0000,0.0000,-45.8740,29.9995+ce1             0.0000,0.0000,0.0000,29.9995 26
    27                 PMT_3inch_pmt_solid0x510aae0 ce0           0.0000,0.0000,-17.9373,57.9383-ce1             0.0000,0.0000,0.0000,57.9383 27
    28                     sChimneyAcrylic0x5b310c0 ce0            0.0000,0.0000,0.0000,520.0000-ce1            0.0000,0.0000,0.0000,520.0000 28
    29                          sChimneyLS0x5b312e0 ce0           0.0000,0.0000,0.0000,1965.0000+ce1           0.0000,0.0000,0.0000,1965.0000 29
    30                       sChimneySteel0x5b314f0 ce0           0.0000,0.0000,0.0000,1665.0000-ce1           0.0000,0.0000,0.0000,1665.0000 30
    31                          sWaterTube0x5b30eb0 ce0           0.0000,0.0000,0.0000,1965.0000+ce1           0.0000,0.0000,0.0000,1965.0000 31
    32                        svacSurftube0x5b3bf50 ce0              0.0000,0.0000,0.0000,4.0000+ce1              0.0000,0.0000,0.0000,4.0000 32
    33                           sSurftube0x5b3ab80 ce0              0.0000,0.0000,0.0000,5.0000+ce1              0.0000,0.0000,0.0000,5.0000 33
    34                         sInnerWater0x4bd3660 ce0        0.0000,0.0000,850.0000,20900.0000-ce1          0.0000,0.0000,0.0000,20900.0000 34
    35                      sReflectorInCD0x4bd3040 ce0        0.0000,0.0000,849.0000,20901.0000-ce1          0.0000,0.0000,0.0000,20901.0000 35
    36                     sOuterWaterPool0x4bd2960 ce0          0.0000,0.0000,0.0000,21750.0000-ce1          0.0000,0.0000,0.0000,21750.0000 36
    37                         sPoolLining0x4bd1eb0 ce0         0.0000,0.0000,-1.5000,21753.0000-ce1          0.0000,0.0000,0.0000,21753.0000 37
    38                         sBottomRock0x4bcd770 ce0      0.0000,0.0000,-1500.0000,24750.0000+ce1          0.0000,0.0000,0.0000,24750.0000 38
    39                              sWorld0x4bc2350 ce0          0.0000,0.0000,0.0000,60000.0000+ce1          0.0000,0.0000,0.0000,60000.0000 39

    40                          sFasteners0x4c01080 ce0          0.0000,0.0000,-92.5000,150.0000 ce1            0.0000,0.0000,0.0000,150.0000 40





Non-aligned deviation checking reminder
---------------------------------------------


Take a look at tboolean-proxy-18::

    tp(){  tboolean-;PROXYLV=18 tboolean-proxy-ip $* ; }


Vague recollection, non-aligned deviation checking relies
on "accidental" history alignment. The problem with this is 
that you rapidly get very low statistics to compare.  

* :doc:



::

    In [22]: ab
    Out[22]: 
    AB(1,torch,tboolean-proxy-18)  None 0 
    A tboolean-proxy-18/tboolean-proxy-18/torch/  1 :  20190617-2331 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/tboolean-proxy-18/evt/tboolean-proxy-18/torch/1/fdom.npy () 
    B tboolean-proxy-18/tboolean-proxy-18/torch/ -1 :  20190617-2331 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/tboolean-proxy-18/evt/tboolean-proxy-18/torch/-1/fdom.npy (recstp) 
    tboolean-proxy-18



    In [16]: av,bv = ab.ox_dv.dvs[2].av, ab.ox_dv.dvs[2].bv

    In [17]: av.shape
    Out[17]: (27, 3, 4)

    In [18]: bv.shape
    Out[18]: (27, 3, 4)

    In [19]: av[0]
    Out[19]: 
    A()sliced
    A([[-451.8267, -183.0164, -747.0001,    7.0898],
       [  -0.5488,   -0.2223,   -0.8058,    1.    ],
       [   0.6132,   -0.7622,   -0.2074,  380.    ]], dtype=float32)

    In [20]: bv[0]
    Out[20]: 
    A()sliced
    A([[-451.8271, -183.0166, -747.    ,    7.0898],
       [  -0.5488,   -0.2223,   -0.8058,    1.    ],
       [   0.6132,   -0.7622,   -0.2074,  380.    ]], dtype=float32)

    In [21]: ab.ox_dv
    Out[21]: 
    ab.ox_dv maxdvmax: 0.00238 maxdv:0.0001221        0  0.00238  skip:SC AB RE
      idx        msg :                            sel :    lcu1     lcu2  :       nitem     nelem/    ndisc: fdisc  mx/mn/av        mx/       mn/      avg  eps:eps    
     0000            :                    TO BT BT SA :    8794     8794  :        7710     92520/        0: 0.000  mx/mn/av 0.0001221/        0/3.652e-06  eps:0.0002    
     0001            :                       TO BR SA :     580      617  :          33       396/        0: 0.000  mx/mn/av         0/        0/        0  eps:0.0002    
     0002            :                 TO BT BR BT SA :     561      527  :          27       324/       14: 0.043  mx/mn/av   0.00238/        0/3.823e-05  eps:0.0002    





