tboolean-proxy-scan
======================

Context
-----------

Have lots of deviation comparison machinery, need to 
get back into context with it.

* :doc:`deviation-comparison-review`

TODO
----------

* combine RC of up to 8 sections into the process RC  
* try non-aligned with --relectcheat 


DONE
--------

* use 8-bits of RC more meaningfully
* split c2max rdvmax pdvmax cuts into three floats for : warning, error, fatal   
  where exceeding error level yields non-zero RC for that comparison
* report these levels in the output
* ansi colors are good interactively but need to show that in pure text way too
* have an RC for every line,  doing it as WARN/ERROR/FATAL for each line, that gets combined into max RC for each section 


Command shortcuts
---------------------

::

    lv(){ echo 21 ; }
    # default geometry LV index to test 

    ts(){  PROXYLV=${1:-$(lv)} tboolean.sh --align --dbgskipclearzero --dbgnojumpzero --dbgkludgeflatzero $* ; } 
    # **simulate** : aligned bi-simulation creating OK+G4 events 

    tv(){  PROXYLV=${1:-$(lv)} tboolean.sh --load $* ; } 
    # **visualize** : load events and visualize the propagation

    tv4(){ tv --vizg4 $* ; }
    # **visualize** the geant4 propagation 

    ta(){  tboolean-;PROXYLV=${1:-$(lv)} tboolean-proxy-ip ; } 
    # **analyse** : load events and analyse the propagation



DONE : visualization needs auto time domain : to be of any use for large geometry
---------------------------------------------------------------------------------------

* :doc:`large-extent-geometry-sparse-photon-visualization`


TODO : python scanning "ali.py" 
------------------------------------

Hmm need more detailed summary information than just RC ... 
eg names of proxies, extents 


LV 0-9 inclusive, chisq 0, no serious deviations : extents all less than 3.5m
-------------------------------------------------------------------------------------

* aligned scan with the new RC,  RC offset rpost_dv:0, rpol_dv:1, ox_dv:2 
* chisq 0 means the simulation histories stayed perfectly aligned 

::

    [blyth@localhost ana]$ scan--
               scan-- :       tboolean.py --pfx tboolean-proxy-0 ======= RC   4  RC 0x04       chisq:0    ab.ox_dv  maxdvmax:0.0354

               scan-- :       tboolean.py --pfx tboolean-proxy-1 ======= RC   0  RC 0x00       chisq:0

               scan-- :       tboolean.py --pfx tboolean-proxy-2 ======= RC   4  RC 0x04       chisq:0     ab.ox_dv  maxdvmax:0.0206
               scan-- :       tboolean.py --pfx tboolean-proxy-3 ======= RC   4  RC 0x04       chisq:0     ab.ox_dv  maxdvmax:0.0225

               scan-- :       tboolean.py --pfx tboolean-proxy-4 ======= RC   0  RC 0x00       chisq:0     ab.ox_dv  maxdvmax:0.0068 
               scan-- :       tboolean.py --pfx tboolean-proxy-5 ======= RC   0  RC 0x00                             maxdvmax:0.0068  
               scan-- :       tboolean.py --pfx tboolean-proxy-6 ======= RC   0  RC 0x00 
               scan-- :       tboolean.py --pfx tboolean-proxy-7 ======= RC   0  RC 0x00 
               scan-- :       tboolean.py --pfx tboolean-proxy-8 ======= RC   0  RC 0x00 
               scan-- :       tboolean.py --pfx tboolean-proxy-9 ======= RC   0  RC 0x00 

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



LV 10-14 inclusive, very big extents 17-24m : 10:dropout-zero, 11+12+13+14:truncation difference 
-----------------------------------------------------------------------------------------------------

Two different issues


1. LV 10 : speckle in the hole coincidence problem
2. LV 11,12,13,14 : truncated big bouncers loosing alignment  


::

    .
               scan-- :      tboolean.py --pfx tboolean-proxy-10 ======= RC   5  RC 0x05          
               scan-- :      tboolean.py --pfx tboolean-proxy-11 ======= RC   4  RC 0x04 
               scan-- :      tboolean.py --pfx tboolean-proxy-12 ======= RC   4  RC 0x04 
               scan-- :      tboolean.py --pfx tboolean-proxy-13 ======= RC   6  RC 0x06 
               scan-- :      tboolean.py --pfx tboolean-proxy-14 ======= RC   4  RC 0x04 

    10                              sAirTT0x5b34000 ce0          0.0000,0.0000,0.0000,24000.0000 ce1          0.0000,0.0000,0.0000,24000.0000 10
    11                            sExpHall0x4bcd390 ce0          0.0000,0.0000,0.0000,24000.0000 ce1          0.0000,0.0000,0.0000,24000.0000 11
    12                            sTopRock0x4bccfc0 ce0          0.0000,0.0000,0.0000,27000.0000 ce1          0.0000,0.0000,0.0000,27000.0000 12
    13                             sTarget0x4bd4340 ce0         0.0000,0.0000,60.0000,17760.0000 ce1          0.0000,0.0000,0.0000,17760.0000 13
    14                            sAcrylic0x4bd3cd0 ce0          0.0000,0.0000,0.0000,17820.0000 ce1          0.0000,0.0000,0.0000,17820.0000 14





LV:10 fixing -ve rpost times from too small time domain
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ta 10::

    In [28]: a.rpost().shape
    Out[28]: (23, 5, 4)

    In [26]: ab.a.rpost()       # negative time at the top 
    Out[26]: 
    A()sliced
    A([[[    39.5525,   -188.9732, -71998.8026,      0.    ],
        [    39.5525,   -188.9732,  -2500.5993,    231.8218],
        [    39.5525,   -188.9732,   1500.7991,    245.1671],
        [    39.5525,   -188.9732,   2500.5993,    251.2319],
        [    39.5525,   -188.9732,  72001.    ,   -480.0213]],

       [[  -239.5126,    -92.2893, -71998.8026,      0.    ],
        [  -239.5126,    -92.2893,  -2500.5993,    231.8218],
        [  -239.5126,    -92.2893,   1500.7991,    245.1671],
        [  -239.5126,    -92.2893,   2500.5993,    251.2319],
        [  -239.5126,    -92.2893,  72001.    ,   -480.0213]],


    In [29]: ab.a.fdom
    Out[29]: 
    A(torch,1,tboolean-proxy-10)(metadata) 3*float4 domains of position, time, wavelength (used for compression)
    A([[[    0.    ,     0.    ,     0.    , 72001.    ]],

       [[    0.    ,   480.0067,   480.0067,     0.    ]],

       [[   60.    ,   820.    ,    20.    ,   760.    ]]], dtype=float32)


Not fitting in short spits out SHRT_MIN -32767 as the compressed time
becoming -480.0 the negated timemax : which stands out like a sore thumb.  

::

   TMAX=500 ts 10    ## it was close 



    In [1]: ab.sel = "TO BT BT BT SA"
    [2019-06-21 23:30:18,744] p83910 {evt.py    :876} WARNING  - _init_selection EMPTY nsel 0 len(psel) 10000 

    In [2]: a.rpost().shape
    Out[2]: (23, 5, 4)

    In [3]: ab.a.rpost()
    Out[3]: 
    A()sliced
    A([[[    39.5525,   -188.9732, -71998.8026,      0.    ],
        [    39.5525,   -188.9732,  -2500.5993,    231.8339],
        [    39.5525,   -188.9732,   1500.7991,    245.1704],
        [    39.5525,   -188.9732,   2500.5993,    251.2284],
        [    39.5525,   -188.9732,  72001.    ,    483.0622]],

       [[  -239.5126,    -92.2893, -71998.8026,      0.    ],
        [  -239.5126,    -92.2893,  -2500.5993,    231.8339],
        [  -239.5126,    -92.2893,   1500.7991,    245.1704],
        [  -239.5126,    -92.2893,   2500.5993,    251.2284],
        [  -239.5126,    -92.2893,  72001.    ,    483.0622]],

::

     
     82 /**
     83 shortnorm
     84 ------------
     85 
     86 range of short is -32768 to 32767
     87 Expect no positions out of range, as constrained by the geometry are bouncing on,
     88 but getting times beyond the range eg 0.:100 ns is expected
     89 
     90 **/
     91 
     92 __device__ short shortnorm( float v, float center, float extent )
     93 {
     94     int inorm = __float2int_rn(32767.0f * (v - center)/extent ) ;    // linear scaling into -1.f:1.f * float(SHRT_MAX)
     95     return fitsInShort(inorm) ? short(inorm) : SHRT_MIN  ;
     96 }   



After re-simulating to fix the time domain and using automated rule of thumb to 
set the timedomain based on geometry extent the -ve times are gone and 
the visualized propagation looks more reasonable::

    TMAX=-1 ts 10 
    TMAX=-1 tv 10 


LV:10 sAirTT COINCIDENCE/SPECKLE + HISTORY ALIGNMENT LOSSES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* large deviations from a few photons failing to stay in history alignment

* :doc:`tboolean-proxy-scan-LV10-coincidence-speckle`  FIXED
* :doc:`tboolean-proxy-scan-LV10-history-misaligned-big-bouncer`


LV:11 sExpHall0x4bcd390 : maxdvmax:0.1265  THIS ONE IS A BIG EXTENT SOLIDS THATS CLOSE TO BEING OK
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ta 11, chisq aligned : apart from one that looks like a truncation difference::

      0009       bbbbbbb6cd        43        39             0.20        1.103 +- 0.168        0.907 +- 0.145  [10] TO BT SC BR BR BR BR BR BR BR


LV:12 sTopRock0x4bccfc0  maxdvmax:0.1836  LOOKS LIKE SAME TRUNCATION ISSUE TO LV:11 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
     
ta 12, maxdvmax:0.1836, same as LV:11::

      0009       bbbbbbb6cd        47        42             0.28        1.119 +- 0.163        0.894 +- 0.138  [10] TO BT SC BR BR BR BR BR BR BR


LV:13 sTarget0x4bd4340
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Difference in big bouncers::

    tboolean-proxy-13/tboolean-proxy-13/torch/ -1 :  20190620-1638 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/tboolean-proxy-13/evt/tboolean-proxy-13/torch/-1/fdom.npy (recstp) 
    tboolean-proxy-13
    .
    ab.his
    .                seqhis_ana  1:tboolean-proxy-13:tboolean-proxy-13   -1:tboolean-proxy-13:tboolean-proxy-13        c2        ab        ba 
    .                              10000     10000         0.18/13 =  0.01  (pval:1.000 prob:0.000)  
    0000             8ccd      7631      7632             0.00        1.000 +- 0.011        1.000 +- 0.011  [4 ] TO BT BT SA                        <<<<<<<<
    0001              8bd       537       537             0.00        1.000 +- 0.043        1.000 +- 0.043  [3 ] TO BR SA
    0002            8cbcd       470       470             0.00        1.000 +- 0.046        1.000 +- 0.046  [5 ] TO BT BR BT SA
    0003              4cd       288       288             0.00        1.000 +- 0.059        1.000 +- 0.059  [3 ] TO BT AB
    0004              86d       273       273             0.00        1.000 +- 0.061        1.000 +- 0.061  [3 ] TO SC SA
    0005            86ccd       245       245             0.00        1.000 +- 0.064        1.000 +- 0.064  [5 ] TO BT BT SC SA
    0006            8c6cd       194       194             0.00        1.000 +- 0.072        1.000 +- 0.072  [5 ] TO BT SC BT SA
    0007       bbbbbbb6cd        47        43             0.18        1.093 +- 0.159        0.915 +- 0.140  [10] TO BT SC BR BR BR BR BR BR BR     <<<<<<<<<<<
    0008          8cc6ccd        36        36             0.00        1.000 +- 0.167        1.000 +- 0.167  [7 ] TO BT BT SC BT BT SA
    0009            8bccd        31        31             0.00        1.000 +- 0.180        1.000 +- 0.180  [5 ] TO BT BT BR SA
    0010           8cbbcd        27        27             0.00        1.000 +- 0.192        1.000 +- 0.192  [6 ] TO BT BR BR BT SA
    0011            8cc6d        24        24             0.00        1.000 +- 0.204        1.000 +- 0.204  [5 ] TO SC BT BT SA
    0012             4bcd        21        21             0.00        1.000 +- 0.218        1.000 +- 0.218  [4 ] TO BT BR AB
    0013             86bd        18        18             0.00        1.000 +- 0.236        1.000 +- 0.236  [4 ] TO BR SC SA
    0014           8c6bcd        14        14             0.00        1.000 +- 0.267        1.000 +- 0.267  [6 ] TO BT BR SC BT SA
    0015           866ccd        14        14             0.00        1.000 +- 0.267        1.000 +- 0.267  [6 ] TO BT BT SC SC SA
    0016           86cbcd        13        13             0.00        1.000 +- 0.277        1.000 +- 0.277  [6 ] TO BT BR BT SC SA
    0017       bbbbbbcccd        12        11             0.00        1.091 +- 0.315        0.917 +- 0.276  [10] TO BT BT BT BR BR BR BR BR BR     <<<<<<<<
    0018             866d        11        11             0.00        1.000 +- 0.302        1.000 +- 0.302  [4 ] TO SC SC SA
    0019             8b6d         9         9             0.00        1.000 +- 0.333        1.000 +- 0.333  [4 ] TO SC BR SA
    .                              10000     10000         0.18/13 =  0.01  (pval:1.000 prob:0.000)  

LV:14 sAcrylic0x4bd3cd0  
~~~~~~~~~~~~~~~~~~~~~~~~~~

ta 14, maxdvmax:0.5522, again big bouncer truncation looses alignment ::

    0000             8ccd      7669      7668             0.00        1.000 +- 0.011        1.000 +- 0.011  [4 ] TO BT BT SA
    0007       bbbbbbb6cd        49        45             0.17        1.089 +- 0.156        0.918 +- 0.137  [10] TO BT SC BR BR BR BR BR BR BR



LV 15,16 
-------------

::

    .          scan-- :      tboolean.py --pfx tboolean-proxy-15 ======= RC   0  RC 0x00 
               scan-- :      tboolean.py --pfx tboolean-proxy-16 ======= RC   0  RC 0x00 

    15                              sStrut0x4bd4b80 ce0            0.0000,0.0000,0.0000,600.0000 ce1            0.0000,0.0000,0.0000,600.0000 15
    16                          sFasteners0x4c01080 ce0          0.0000,0.0000,-92.5000,150.0000 ce1            0.0000,0.0000,0.0000,150.0000 16







LV 17-22 
-----------------

* LV:18 TODO: POSSIBLE POLZ BUG TO CHASE
* LV:19/20/21 TODO: UNDERSTAND WHY ALIGNMENT LOST FOR HANDFUL OF PHOTONS OUT OF 10k 


::

    .          scan-- :      tboolean.py --pfx tboolean-proxy-17 ======= RC   4  RC 0x04     observatory dome,  chisq 0, maxdvmax:0.0241 just beneath cut
               scan-- :      tboolean.py --pfx tboolean-proxy-18 ======= RC   6  RC 0x06     cathode cap,       chisq 0, maxdvmax:1.0000   POLZ WRONG ?
               scan-- :      tboolean.py --pfx tboolean-proxy-19 ======= RC   5  RC 0x05     cap chopped PMT,   chisq != 0, maxdvmax:0.1598   
               scan-- :      tboolean.py --pfx tboolean-proxy-20 ======= RC   5  RC 0x05     full PMT,          chisq != 0, maxdvmax:0.0479 
               scan-- :      tboolean.py --pfx tboolean-proxy-21 ======= RC   5  RC 0x05     full PMT           chisq != 0, maxdvmax:0.0670
               scan-- :      tboolean.py --pfx tboolean-proxy-22 ======= RC   0  RC 0x00     cylinder           chisq 0, no warnings even


    17                               sMask0x4ca38d0 ce0          0.0000,0.0000,-78.9500,274.9500 ce1            0.0000,0.0000,0.0000,274.9500 17
    18             PMT_20inch_inner1_solid0x4cb3610 ce0           0.0000,0.0000,89.5000,249.0000 ce1            0.0000,0.0000,0.0000,249.0000 18
    19             PMT_20inch_inner2_solid0x4cb3870 ce0         0.0000,0.0000,-167.0050,249.0000 ce1            0.0000,0.0000,0.0000,249.0000 19
    20               PMT_20inch_body_solid0x4c90e50 ce0          0.0000,0.0000,-77.5050,261.5050 ce1            0.0000,0.0000,0.0000,261.5050 20
    21                PMT_20inch_pmt_solid0x4c81b40 ce0          0.0000,0.0000,-77.5050,261.5060 ce1           0.0000,0.0000,-0.0000,261.5060 21
    22                       sMask_virtual0x4c36e10 ce0          0.0000,0.0000,-79.0000,275.0500 ce1            0.0000,0.0000,0.0000,275.0500 22



LV 18 : polarization wrong ? for "TO BT BR BR BR BT SA"  0x8cbbbcd
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    0005          8cbbbcd         7         7             0.00        1.000 +- 0.378        1.000 +- 0.378  [7 ] TO BT BR BR BR BT SA


    ab.rpol_dv
    maxdvmax:1.0000  level:FATAL  RC:1       skip:
                     :                                :                   :                       :                   : 0.0078 0.0118 0.0157 :                                    
      idx        msg :                            sel :    lcu1     lcu2  :       nitem     nelem :  nwar  nerr  nfat :   fwar   ferr   ffat :        mx        mn       avg      
     0000            :                    TO BT BT SA :    8794     8794  :        8794    105528 :     0     0     0 : 0.0000 0.0000 0.0000 :    0.0000    0.0000    0.0000   :        INFO :  
     0001            :                       TO BR SA :     580      580  :         580      5220 :     0     0     0 : 0.0000 0.0000 0.0000 :    0.0000    0.0000    0.0000   :        INFO :  
     0002            :                 TO BT BR BT SA :     561      561  :         561      8415 :     0     0     0 : 0.0000 0.0000 0.0000 :    0.0000    0.0000    0.0000   :        INFO :  
     0003            :              TO BT BR BR BT SA :      37       37  :          37       666 :     0     0     0 : 0.0000 0.0000 0.0000 :    0.0000    0.0000    0.0000   :        INFO :  
     0004            :                       TO SC SA :       8        8  :           8        72 :     0     0     0 : 0.0000 0.0000 0.0000 :    0.0000    0.0000    0.0000   :        INFO :  
     0005            :           TO BT BR BR BR BT SA :       7        7  :           7       147 :     4     4     4 : 0.0272 0.0272 0.0272 :    1.0000    0.0000    0.0269   :  FATAL :   > dvmax[2] 0.0157  
     0006            :                 TO BT BT SC SA :       7        7  :           7       105 :     0     0     0 : 0.0000 0.0000 0.0000 :    0.0000    0.0000    0.0000   :        INFO :  
     0007            :                       TO BT AB :       2        2  :           2        18 :     0     0     0 : 0.0000 0.0000 0.0000 :    0.0000    0.0000    0.0000   :        INFO :  
     0008            :           TO BT BT SC BT BT SA :       1        1  :           1        21 :     0     0     0 : 0.0000 0.0000 0.0000 :    0.0000    0.0000    0.0000   :        INFO :  
     0009            :        TO BT SC BR BR BR BT SA :       1        1  :           1        24 :     0     0     0 : 0.0000 0.0000 0.0000 :    0.0000    0.0000    0.0000   :        INFO :  
     0010            :              TO BR SC BT BT SA :       1        1  :           1        18 :     0     0     0 : 0.0000 0.0000 0.0000 :    0.0000    0.0000    0.0000   :        INFO :  
     0011            :                 TO BT SC BT SA :       1        1  :           1        15 :     0     0     0 : 0.0000 0.0000 0.0000 :    0.0000    0.0000    0.0000   :        INFO :  
    .
    ab.ox_dv
    maxdvmax:0.9989  level:FATAL  RC:1       skip:
                     :                                :                   :                       :                   : 0.0010 0.0200 0.1000 :                                    
      idx        msg :                            sel :    lcu1     lcu2  :       nitem     nelem :  nwar  nerr  nfat :   fwar   ferr   ffat :        mx        mn       avg      
     0000            :                    TO BT BT SA :    8794     8794  :        8794    105528 :     0     0     0 : 0.0000 0.0000 0.0000 :    0.0001    0.0000    0.0000   :        INFO :  
     0001            :                       TO BR SA :     580      580  :         580      6960 :     0     0     0 : 0.0000 0.0000 0.0000 :    0.0000    0.0000    0.0000   :        INFO :  
     0002            :                 TO BT BR BT SA :     561      561  :         561      6732 :    23     0     0 : 0.0034 0.0000 0.0000 :    0.0030    0.0000    0.0000   :     WARNING :   > dvmax[0] 0.0010  
     0003            :              TO BT BR BR BT SA :      37       37  :          37       444 :     0     0     0 : 0.0000 0.0000 0.0000 :    0.0003    0.0000    0.0000   :        INFO :  
     0004            :                       TO SC SA :       8        8  :           8        96 :     0     0     0 : 0.0000 0.0000 0.0000 :    0.0002    0.0000    0.0000   :        INFO :  
     0005            :           TO BT BR BR BR BT SA :       7        7  :           7        84 :     3     2     2 : 0.0357 0.0238 0.0238 :    0.9989    0.0000    0.0235   :  FATAL :   > dvmax[2] 0.1000  
     0006            :                 TO BT BT SC SA :       7        7  :           7        84 :     0     0     0 : 0.0000 0.0000 0.0000 :    0.0004    0.0000    0.0000   :        INFO :  
     0007            :                       TO BT AB :       2        2  :           2        24 :     0     0     0 : 0.0000 0.0000 0.0000 :    0.0000    0.0000    0.0000   :        INFO :  
     0008            :           TO BT BT SC BT BT SA :       1        1  :           1        12 :     0     0     0 : 0.0000 0.0000 0.0000 :    0.0000    0.0000    0.0000   :        INFO :  
     0009            :        TO BT SC BR BR BR BT SA :       1        1  :           1        12 :     1     0     0 : 0.0833 0.0000 0.0000 :    0.0048    0.0000    0.0004   :     WARNING :   > dvmax[0] 0.0010  
     0010            :              TO BR SC BT BT SA :       1        1  :           1        12 :     0     0     0 : 0.0000 0.0000 0.0000 :    0.0001    0.0000    0.0000   :        INFO :  
     0011            :                 TO BT SC BT SA :       1        1  :           1        12 :     0     0     0 : 0.0000 0.0000 0.0000 :    0.0001    0.0000    0.0000   :        INFO :  
    .
    RC 0x06


LV 19 : chisq non-zero : lost alignment for a few photons
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    ab
    AB(1,torch,tboolean-proxy-19)  None 0 
    A tboolean-proxy-19/tboolean-proxy-19/torch/  1 :  20190620-1639 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/tboolean-proxy-19/evt/tboolean-proxy-19/torch/1/fdom.npy () 
    B tboolean-proxy-19/tboolean-proxy-19/torch/ -1 :  20190620-1639 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/tboolean-proxy-19/evt/tboolean-proxy-19/torch/-1/fdom.npy (recstp) 
    tboolean-proxy-19
    .
    ab.his
    .                seqhis_ana  1:tboolean-proxy-19:tboolean-proxy-19   -1:tboolean-proxy-19:tboolean-proxy-19        c2        ab        ba 
    .                              10000     10000         0.01/3 =  0.00  (pval:1.000 prob:0.000)  
    0000             8ccd      8668      8668             0.00        1.000 +- 0.011        1.000 +- 0.011  [4 ] TO BT BT SA
    0001              8bd       710       710             0.00        1.000 +- 0.038        1.000 +- 0.038  [3 ] TO BR SA
    0002            8cbcd       503       503             0.00        1.000 +- 0.045        1.000 +- 0.045  [5 ] TO BT BR BT SA
    0003           8cbbcd        74        73    ####     0.01        1.014 +- 0.118        0.986 +- 0.115  [6 ] TO BT BR BR BT SA
    0004          8cbbbcd        10        11    ####     0.00        0.909 +- 0.287        1.100 +- 0.332  [7 ] TO BT BR BR BR BT SA
    0005         8cbbbbcd        10        10             0.00        1.000 +- 0.316        1.000 +- 0.316  [8 ] TO BT BR BR BR BR BT SA
    0006              86d         8         8             0.00        1.000 +- 0.354        1.000 +- 0.354  [3 ] TO SC SA
    0007            86ccd         7         7             0.00        1.000 +- 0.378        1.000 +- 0.378  [5 ] TO BT BT SC SA
    0008            8cccd         2         0    ####     0.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] TO BT BT BT SA
    0009              4cd         2         2             0.00        1.000 +- 0.707        1.000 +- 0.707  [3 ] TO BT AB
    0010            8c6cd         2         2             0.00        1.000 +- 0.707        1.000 +- 0.707  [5 ] TO BT SC BT SA
    0011       bbbbbbbbcd         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [10] TO BT BR BR BR BR BR BR BR BR
    0012       8cbbbbbbcd         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [10] TO BT BR BR BR BR BR BR BT SA
    0013         8cbc6ccd         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [8 ] TO BT BT SC BT BR BT SA
    0014          8cbc6bd         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [7 ] TO BR SC BT BR BT SA
    0015           8ccccd         0         2    ####     0.00        0.000 +- 0.000        0.000 +- 0.000  [6 ] TO BT BT BT BT SA
    .                              10000     10000         0.01/3 =  0.00  (pval:1.000 prob:0.000)  



LV 20 : chisq non-zero/alignment lost,  maxdvmax:0.0479
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    ab.his
    .                seqhis_ana  1:tboolean-proxy-20:tboolean-proxy-20   -1:tboolean-proxy-20:tboolean-proxy-20        c2        ab        ba 
    .                              10000     10000         0.00/4 =  0.00  (pval:1.000 prob:0.000)  
    0000             8ccd      8681      8681             0.00        1.000 +- 0.011        1.000 +- 0.011  [4 ] TO BT BT SA
    0001              8bd       696       696             0.00        1.000 +- 0.038        1.000 +- 0.038  [3 ] TO BR SA
    0002            8cbcd       345       346             0.00        0.997 +- 0.054        1.003 +- 0.054  [5 ] TO BT BR BT SA
    0003           8cbbcd       174       174             0.00        1.000 +- 0.076        1.000 +- 0.076  [6 ] TO BT BR BR BT SA
    0004          8cbbbcd        54        54             0.00        1.000 +- 0.136        1.000 +- 0.136  [7 ] TO BT BR BR BR BT SA
    0005          8cccbcd        10        10             0.00        1.000 +- 0.316        1.000 +- 0.316  [7 ] TO BT BR BT BT BT SA
    0006         8cccbbcd         8         8             0.00        1.000 +- 0.354        1.000 +- 0.354  [8 ] TO BT BR BR BT BT BT SA
    0007              86d         7         7             0.00        1.000 +- 0.378        1.000 +- 0.378  [3 ] TO SC SA
    0008            86ccd         5         5             0.00        1.000 +- 0.447        1.000 +- 0.447  [5 ] TO BT BT SC SA
    0009          8bcbbcd         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [7 ] TO BT BR BR BT BR SA
    0010              4cd         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [3 ] TO BT AB
    0011         8cbbbbcd         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [8 ] TO BT BR BR BR BR BT SA
    0012            8c6cd         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [5 ] TO BT SC BT SA
    0013          8cc6ccd         2         2             0.00        1.000 +- 0.707        1.000 +- 0.707  [7 ] TO BT BT SC BT BT SA
    0014            8cccd         2         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] TO BT BT BT SA
    0015           8cc6bd         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [6 ] TO BR SC BT BT SA
    0016        8cbbbbbcd         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [9 ] TO BT BR BR BR BR BR BT SA
    0017          8cbcbcd         1         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [7 ] TO BT BR BT BR BT SA
    0018       bbbbbbb6cd         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [10] TO BT SC BR BR BR BR BR BR BR
    0019           8ccccd         0         2             0.00        0.000 +- 0.000        0.000 +- 0.000  [6 ] TO BT BT BT BT SA
    .                              10000     10000         0.00/4 =  0.00  (pval:1.000 prob:0.000)  



LV 21,  BT difference ?  maxdvmax:0.0719
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    ab.his
    .                seqhis_ana  1:tboolean-proxy-21:tboolean-proxy-21   -1:tboolean-proxy-21:tboolean-proxy-21        c2        ab        ba 
    .                              10000     10000         0.00/4 =  0.00  (pval:1.000 prob:0.000)  
    0000             8ccd      8681      8681             0.00        1.000 +- 0.011        1.000 +- 0.011  [4 ] TO BT BT SA
    0001              8bd       696       696             0.00        1.000 +- 0.038        1.000 +- 0.038  [3 ] TO BR SA
    0002            8cbcd       346       346             0.00        1.000 +- 0.054        1.000 +- 0.054  [5 ] TO BT BR BT SA
    0003           8cbbcd       174       174             0.00        1.000 +- 0.076        1.000 +- 0.076  [6 ] TO BT BR BR BT SA
    0004          8cbbbcd        54        54             0.00        1.000 +- 0.136        1.000 +- 0.136  [7 ] TO BT BR BR BR BT SA
    0005          8cccbcd        10        10             0.00        1.000 +- 0.316        1.000 +- 0.316  [7 ] TO BT BR BT BT BT SA
    0006         8cccbbcd         8         8             0.00        1.000 +- 0.354        1.000 +- 0.354  [8 ] TO BT BR BR BT BT BT SA
    0007              86d         7         7             0.00        1.000 +- 0.378        1.000 +- 0.378  [3 ] TO SC SA
    0008            86ccd         5         5             0.00        1.000 +- 0.447        1.000 +- 0.447  [5 ] TO BT BT SC SA
    0009          8bcbbcd         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [7 ] TO BT BR BR BT BR SA
    0010              4cd         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [3 ] TO BT AB
    0011         8cbbbbcd         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [8 ] TO BT BR BR BR BR BT SA
    0012            8c6cd         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [5 ] TO BT SC BT SA
    0013          8cc6ccd         2         2             0.00        1.000 +- 0.707        1.000 +- 0.707  [7 ] TO BT BT SC BT BT SA
    0014            8cccd         2         0   ###       0.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] TO BT BT BT SA
    0015           8cc6bd         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [6 ] TO BR SC BT BT SA
    0016        8cbbbbbcd         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [9 ] TO BT BR BR BR BR BR BT SA
    0017       bbbbbbb6cd         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [10] TO BT SC BR BR BR BR BR BR BR
    0018           8ccccd         0         2   ###       0.00        0.000 +- 0.000        0.000 +- 0.000  [6 ] TO BT BT BT BT SA
    .                              10000     10000         0.00/4 =  0.00  (pval:1.000 prob:0.000)  


LV 23-27 : small extent, chisq 0
------------------------------------------

::

    23   PMT_3inch_inner1_solid_ell_helper0x510ae30 ce0            0.0000,0.0000,14.5216,38.0000 ce1             0.0000,0.0000,0.0000,38.0000 23
    24   PMT_3inch_inner2_solid_ell_helper0x510af10 ce0            0.0000,0.0000,-4.4157,38.0000 ce1             0.0000,0.0000,0.0000,38.0000 24
    25 PMT_3inch_body_solid_ell_ell_helper0x510ada0 ce0             0.0000,0.0000,4.0627,40.0000 ce1             0.0000,0.0000,0.0000,40.0000 25
    26                PMT_3inch_cntr_solid0x510afa0 ce0           0.0000,0.0000,-45.8740,29.9995 ce1             0.0000,0.0000,0.0000,29.9995 26
    27                 PMT_3inch_pmt_solid0x510aae0 ce0           0.0000,0.0000,-17.9373,57.9383 ce1             0.0000,0.0000,0.0000,57.9383 27



               scan-- :      tboolean.py --pfx tboolean-proxy-23 ======= RC   0  RC 0x00 
               scan-- :      tboolean.py --pfx tboolean-proxy-24 ======= RC   0  RC 0x00 
               scan-- :      tboolean.py --pfx tboolean-proxy-25 ======= RC   0  RC 0x00 
               scan-- :      tboolean.py --pfx tboolean-proxy-26 ======= RC   0  RC 0x00 
               scan-- :      tboolean.py --pfx tboolean-proxy-27 ======= RC   0  RC 0x00 



LV 28-31 
------------------------------------

* LV:28 not a fair test, the photons almost entirely missed 
* LV:29 perfect chisq zero, maxdvmax:0.0550 : ERROR CUT 0.0200  TOO TIGHT ? 


::

    28                     sChimneyAcrylic0x5b310c0 ce0            0.0000,0.0000,0.0000,520.0000 ce1            0.0000,0.0000,0.0000,520.0000 28
    29                          sChimneyLS0x5b312e0 ce0           0.0000,0.0000,0.0000,1965.0000 ce1           0.0000,0.0000,0.0000,1965.0000 29
    30                       sChimneySteel0x5b314f0 ce0           0.0000,0.0000,0.0000,1665.0000 ce1           0.0000,0.0000,0.0000,1665.0000 30
    31                          sWaterTube0x5b30eb0 ce0           0.0000,0.0000,0.0000,1965.0000 ce1           0.0000,0.0000,0.0000,1965.0000 31



               scan-- :      tboolean.py --pfx tboolean-proxy-28 ======= RC   0  RC 0x00 
               scan-- :      tboolean.py --pfx tboolean-proxy-29 ======= RC   4  RC 0x04      perfect chisq zero, maxdvmax:0.0550  
               scan-- :      tboolean.py --pfx tboolean-proxy-30 ======= RC   0  RC 0x00 
               scan-- :      tboolean.py --pfx tboolean-proxy-31 ======= RC   0  RC 0x00 



LV 32-33, torus placeholder small boxes
------------------------------------------

::

    32                        svacSurftube0x5b3bf50 ce0              0.0000,0.0000,0.0000,4.0000 ce1              0.0000,0.0000,0.0000,4.0000 32
    33                           sSurftube0x5b3ab80 ce0              0.0000,0.0000,0.0000,5.0000 ce1              0.0000,0.0000,0.0000,5.0000 33

               scan-- :      tboolean.py --pfx tboolean-proxy-32 ======= RC   0  RC 0x00 
               scan-- :      tboolean.py --pfx tboolean-proxy-33 ======= RC   0  RC 0x00 



LV 34-39, big extent 
--------------------------

* looks like similar issues to LV 10-14


::

    34                         sInnerWater0x4bd3660 ce0        0.0000,0.0000,850.0000,20900.0000 ce1          0.0000,0.0000,0.0000,20900.0000 34
    35                      sReflectorInCD0x4bd3040 ce0        0.0000,0.0000,849.0000,20901.0000 ce1          0.0000,0.0000,0.0000,20901.0000 35
    36                     sOuterWaterPool0x4bd2960 ce0          0.0000,0.0000,0.0000,21750.0000 ce1          0.0000,0.0000,0.0000,21750.0000 36
    37                         sPoolLining0x4bd1eb0 ce0         0.0000,0.0000,-1.5000,21753.0000 ce1          0.0000,0.0000,0.0000,21753.0000 37
    38                         sBottomRock0x4bcd770 ce0      0.0000,0.0000,-1500.0000,24750.0000 ce1          0.0000,0.0000,0.0000,24750.0000 38
    39                              sWorld0x4bc2350 ce0          0.0000,0.0000,0.0000,60000.0000 ce1          0.0000,0.0000,0.0000,60000.0000 39


               scan-- :      tboolean.py --pfx tboolean-proxy-34 ======= RC   5  RC 0x05    sphere with protrusion, non zero chisq, big bouncers again
               scan-- :      tboolean.py --pfx tboolean-proxy-35 ======= RC   5  RC 0x05 
               scan-- :      tboolean.py --pfx tboolean-proxy-36 ======= RC   4  RC 0x04 
               scan-- :      tboolean.py --pfx tboolean-proxy-37 ======= RC   4  RC 0x04 
               scan-- :      tboolean.py --pfx tboolean-proxy-38 ======= RC   4  RC 0x04 
               scan-- :      tboolean.py --pfx tboolean-proxy-39 ======= RC   5  RC 0x05     

                              tp 39 : handful of photons are way out, failed to stay aligned ?



















tp 0/1/2/3
----------------------

With large extent geometries suspect some errors just from rpost domain compression bin edges.

* NOW CONFIRMED 

Note same deviation number appearing 0.1603 for the first four which have same extent 
which gets scaled to make the domain.

::

    GMeshLibTest

    2019-06-20 17:12:06.694 INFO  [374159] [test_dump1@103]  num_mesh 41
     0                       Upper_LS_tube0x5b2e9f0 ce0           0.0000,0.0000,0.0000,1750.0000 ce1           0.0000,0.0000,0.0000,1750.0000  0
     1                    Upper_Steel_tube0x5b2eb10 ce0           0.0000,0.0000,0.0000,1750.0000 ce1           0.0000,0.0000,0.0000,1750.0000  1
     2                    Upper_Tyvek_tube0x5b2ec30 ce0           0.0000,0.0000,0.0000,1750.0000 ce1           0.0000,0.0000,0.0000,1750.0000  2
     3                       Upper_Chimney0x5b2e8e0 ce0           0.0000,0.0000,0.0000,1750.0000 ce1           0.0000,0.0000,0.0000,1750.0000  3



* TODO: get domain extent into the report 
* DONE: automate the rdvmax cuts based on the compression bin sizes for the fdomain in use


::

    ab.b.metadata:/tmp/tboolean-proxy-0/evt/tboolean-proxy-0/torch/-1          ox:7a76a0edf3bfc0ae98538fd2bff8e027 rx:04b5725eed5ebda2b1b7a828df5aa5ed np:  10000 pr:    2.2949 COMPUTE_MODE compute_requested 
    WITH_SEED_BUFFER WITH_RECORD WITH_SOURCE WITH_ALIGN_DEV WITH_LOGDOUBLE 
    []
    .
    ab.rpost_dv
    maxdvmax:0.1603 maxdv:0.1603 0.1603 0.0000 0.0000 0.1603 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.1603 0.0000  skip:
      idx        msg :                            sel :    lcu1     lcu2  :       nitem     nelem      nerr:  ferr          mx        mn       avg      
     0000            :                          TO SA :    5411     5411  :        5411     43288         2: 0.000      0.1603    0.0000    0.0000   :       ERROR :   > dvmax[1] 0.1000  
     0001            :                    TO BT BT SA :    3950     3950  :        3950     63200         4: 0.000      0.1603    0.0000    0.0000   :       ERROR :   > dvmax[1] 0.1000  
     0002            :                 TO BT BR BT SA :     260      260  :         260      5200         0: 0.000      0.0000    0.0000    0.0000   :             :  
     0003            :                       TO BR SA :     253      253  :         253      3036         0: 0.000      0.0000    0.0000    0.0000   :             :  
     0004            :                       TO SC SA :      55       55  :          55       660         1: 0.002      0.1603    0.0000    0.0002   :       ERROR :   > dvmax[1] 0.1000  
     0005            :                 TO BT BT SC SA :      22       22  :          22       440         0: 0.000      0.0000    0.0000    0.0000   :             :  
     0006            :                       TO BT AB :      14       14  :          14       168         0: 0.000      0.0000    0.0000    0.0000   :             :  
     0007            :              TO BT BR BR BT SA :      13       13  :          13       312         0: 0.000      0.0000    0.0000    0.0000   :             :  
     0008            :                 TO SC BT BT SA :       6        6  :           6       120         0: 0.000      0.0000    0.0000    0.0000   :             :  
     0009            :  TO BT SC BR BR BR BR BR BR BR :       4        4  :           4       160         0: 0.000      0.0000    0.0000    0.0000   :             :  

::

    ab.b.metadata:/tmp/tboolean-proxy-1/evt/tboolean-proxy-1/torch/-1          ox:fbb6fe2129d4bc18c43684ea75e2e7de rx:447d0281e37832dc4901f81393b5e2da np:  10000 pr:    2.0586 COMPUTE_MODE compute_requested 
    WITH_SEED_BUFFER WITH_RECORD WITH_SOURCE WITH_ALIGN_DEV WITH_LOGDOUBLE 
    []
    .
    ab.rpost_dv
    maxdvmax:0.1603 maxdv:0.1603 0.0000 0.1603 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000  skip:
      idx        msg :                            sel :    lcu1     lcu2  :       nitem     nelem      nerr:  ferr          mx        mn       avg      
     0000            :                          TO SA :    9776     9776  :        9776     78208         4: 0.000      0.1603    0.0000    0.0000   :       ERROR :   > dvmax[1] 0.1000  
     0001            :                    TO BT BT SA :     115      115  :         115      1840         0: 0.000      0.0000    0.0000    0.0000   :             :  
     0002            :                       TO SC SA :      68       68  :          68       816         1: 0.001      0.1603    0.0000    0.0002   :       ERROR :   > dvmax[1] 0.1000  
     0003            :                       TO BR SA :      10       10  :          10       120         0: 0.000      0.0000    0.0000    0.0000   :             :  
     0004            :                 TO SC BT BT SA :      10       10  :          10       200         0: 0.000      0.0000    0.0000    0.0000   :             :  
     0005            :           TO SC BT BT BT BT SA :       6        6  :           6       168         0: 0.000      0.0000    0.0000    0.0000   :             :  


    ab.b.metadata:/tmp/tboolean-proxy-2/evt/tboolean-proxy-2/torch/-1          ox:e60cbb075eed2952304dbf187fd3aabf rx:c911af1562f91d2ca6ad17990b99e6ad np:  10000 pr:    2.0293 COMPUTE_MODE compute_requested 
    WITH_SEED_BUFFER WITH_RECORD WITH_SOURCE WITH_ALIGN_DEV WITH_LOGDOUBLE 
    []
    .
    ab.rpost_dv
    maxdvmax:0.1603 maxdv:0.1603 0.1603 0.0000 0.1603 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000  skip:
      idx        msg :                            sel :    lcu1     lcu2  :       nitem     nelem      nerr:  ferr          mx        mn       avg      
     0000            :                          TO SA :    9861     9861  :        9861     78888         4: 0.000      0.1603    0.0000    0.0000   :       ERROR :   > dvmax[1] 0.1000  
     0001            :                       TO SC SA :      70       70  :          70       840         1: 0.001      0.1603    0.0000    0.0002   :       ERROR :   > dvmax[1] 0.1000  
     0002            :                    TO BT BT SA :      41       41  :          41       656         0: 0.000      0.0000    0.0000    0.0000   :             :  
     0003            :                 TO SC BT BT SA :      10       10  :          10       200         1: 0.005      0.1603    0.0000    0.0008   :       ERROR :   > dvmax[1] 0.1000  
     0004            :           TO SC BT BT BT BT SA :       6        6  :           6       168         0: 0.000      0.0000    0.0000    0.0000   :             :  
     0005            :                 TO BT BR BT SA :       4        4  :           4        80         0: 0.000      0.0000    0.0000    0.0000   :             :  
     0006            :                    TO SC BR SA :       4        4  :           4        64         0: 0.000      0.0000    0.0000    0.0000   :             :  


* also tp 3, 

::

    In [1]: ab.a.fdom
    Out[1]: 
    A(torch,1,tboolean-proxy-1)(metadata) 3*float4 domains of position, time, wavelength (used for compression)
    A([[[   0.,    0.,    0., 5251.]],



    In [1]: 1750*3    ## factor 3 from container scale
    Out[1]: 5250

    In [2]: 1750*3*2     ## 2 as extent is [-extent, extent]
    Out[2]: 10500

    In [3]: 1750*3*2./0.1603
    Out[3]: 65502.18340611354

    In [5]: 0x1 << 16
    Out[5]: 65536



Hmm this gives a hint of what is the appropriate deviation cut for rpost checking 
based on the domain of the geometry.

::

    In [4]: ab.a.fdom[0,0,3]
    Out[4]: 5251.0

    In [5]: ab.a.fdom[0,0,3]*2.0/(0x1 << 16)
    Out[5]: 0.160247802734375


Was using a fixed triplet::

    In [7]: ab.ok.rdvmax
    Out[7]: [0.01, 0.1, 1.0]


Can instead can now use a more motivated cut.  DONE using [eps, 1.5*eps, 2.0*eps] as warn/error/fatal levels where eps is compression bin size 

::

    In [10]: np.float(ab.rpost_dv.dvs[0].dv.max())
    Out[10]: 0.16025269325837144

    In [16]: 2.*5251./(65536.-1.)
    Out[16]: 0.16025024795910583




::

     84 __device__ short shortnorm( float v, float center, float extent )
     85 {
     86     // range of short is -32768 to 32767
     87     // Expect no positions out of range, as constrained by the geometry are bouncing on,
     88     // but getting times beyond the range eg 0.:100 ns is expected
     89     //
     90     int inorm = __float2int_rn(32767.0f * (v - center)/extent ) ;    // linear scaling into -1.f:1.f * float(SHRT_MAX)
     91     return fitsInShort(inorm) ? short(inorm) : SHRT_MIN  ;
     92 }
     93 




tp 4-9
---------

::

     4                                sBar0x5b34ab0 ce0           0.0000,0.0000,0.0000,3430.0000 ce1           0.0000,0.0000,0.0000,3430.0000  4
     5                                sBar0x5b34920 ce0           0.0000,0.0000,0.0000,3430.0000 ce1           0.0000,0.0000,0.0000,3430.0000  5
     6                         sModuleTape0x5b34790 ce0           0.0000,0.0000,0.0000,3430.0000 ce1           0.0000,0.0000,0.0000,3430.0000  6
     7                             sModule0x5b34600 ce0           0.0000,0.0000,0.0000,3430.6001 ce1           0.0000,0.0000,0.0000,3430.6001  7
     8                              sPlane0x5b34470 ce0           0.0000,0.0000,0.0000,3430.6001 ce1           0.0000,0.0000,0.0000,3430.6001  8
     9                               sWall0x5b342e0 ce0           0.0000,0.0000,0.0000,3430.6001 ce1           0.0000,0.0000,0.0000,3430.6001  9








aligned scan
---------------

* hmm generally the number of photons with discreps is small, could return that number as the RC ?



::

    [blyth@localhost issues]$ scan--
    scan-- : tboolean.py --pfx tboolean-proxy-0 ========== RC 99
    scan-- : tboolean.py --pfx tboolean-proxy-1 ========== RC 88
    scan-- : tboolean.py --pfx tboolean-proxy-2 ========== RC 99
    scan-- : tboolean.py --pfx tboolean-proxy-3 ========== RC 99
    scan-- : tboolean.py --pfx tboolean-proxy-4 ========== RC 88
    scan-- : tboolean.py --pfx tboolean-proxy-5 ========== RC 88
    scan-- : tboolean.py --pfx tboolean-proxy-6 ========== RC 88
    scan-- : tboolean.py --pfx tboolean-proxy-7 ========== RC 88
    scan-- : tboolean.py --pfx tboolean-proxy-8 ========== RC 88
    scan-- : tboolean.py --pfx tboolean-proxy-9 ========== RC 88
    scan-- : tboolean.py --pfx tboolean-proxy-10 ========== RC 99
    scan-- : tboolean.py --pfx tboolean-proxy-11 ========== RC 99
    scan-- : tboolean.py --pfx tboolean-proxy-12 ========== RC 99
    scan-- : tboolean.py --pfx tboolean-proxy-13 ========== RC 99
    scan-- : tboolean.py --pfx tboolean-proxy-14 ========== RC 99
    scan-- : tboolean.py --pfx tboolean-proxy-15 ========== RC 0
    scan-- : tboolean.py --pfx tboolean-proxy-16 ========== RC 0
    scan-- : tboolean.py --pfx tboolean-proxy-17 ========== RC 99
    scan-- : tboolean.py --pfx tboolean-proxy-18 ========== RC 99
    scan-- : tboolean.py --pfx tboolean-proxy-19 ========== RC 99
    scan-- : tboolean.py --pfx tboolean-proxy-20 ========== RC 99
    scan-- : tboolean.py --pfx tboolean-proxy-21 ========== RC 99
    scan-- : tboolean.py --pfx tboolean-proxy-22 ========== RC 0
    scan-- : tboolean.py --pfx tboolean-proxy-23 ========== RC 0
    scan-- : tboolean.py --pfx tboolean-proxy-24 ========== RC 0
    scan-- : tboolean.py --pfx tboolean-proxy-25 ========== RC 0
    scan-- : tboolean.py --pfx tboolean-proxy-26 ========== RC 0
    scan-- : tboolean.py --pfx tboolean-proxy-27 ========== RC 0
    scan-- : tboolean.py --pfx tboolean-proxy-28 ========== RC 0
    scan-- : tboolean.py --pfx tboolean-proxy-29 ========== RC 99
    scan-- : tboolean.py --pfx tboolean-proxy-30 ========== RC 88
    scan-- : tboolean.py --pfx tboolean-proxy-31 ========== RC 88
    scan-- : tboolean.py --pfx tboolean-proxy-32 ========== RC 0
    scan-- : tboolean.py --pfx tboolean-proxy-33 ========== RC 0
    scan-- : tboolean.py --pfx tboolean-proxy-34 ========== RC 99
    scan-- : tboolean.py --pfx tboolean-proxy-35 ========== RC 99
    scan-- : tboolean.py --pfx tboolean-proxy-36 ========== RC 99
    scan-- : tboolean.py --pfx tboolean-proxy-37 ========== RC 99
    scan-- : tboolean.py --pfx tboolean-proxy-38 ========== RC 99
    scan-- : tboolean.py --pfx tboolean-proxy-39 ========== RC 99
    [blyth@localhost issues]$ 



analysis check
-----------------

::





PROXYLV 21 : PMT shape : aligned running
-------------------------------------------

* the photons are incident from the back side and bounce off the neck, making pretty patterns 
  so its challenging 


tp::

    .
    ab.ox_dv
    maxdvmax:0.0670 maxdv:0.0013 0.0048 0.0045 0.0149 0.0078 0.0087 0.0068 0.0002 0.0002 0.0670 0.0001 0.0099 0.0005 0.0020 0.0005 0.0004 0.0020  skip:
      idx        msg :                            sel :    lcu1     lcu2  :       nitem     nelem      nerr:  ferr          mx        mn       avg      
     0000            :                    TO BT BT SA :    8681     8681  :        8681    104172         0: 0.000      0.0013    0.0000    0.0000   :     WARNING :   > dvmax[0] 0.0010  
     0001            :                       TO BR SA :     696      696  :         696      8352         0: 0.000      0.0048    0.0000    0.0001   :     WARNING :   > dvmax[0] 0.0010  
     0002            :                 TO BT BR BT SA :     346      346  :         346      4152         0: 0.000      0.0045    0.0000    0.0001   :     WARNING :   > dvmax[0] 0.0010  
     0003            :              TO BT BR BR BT SA :     174      174  :         174      2088         0: 0.000      0.0149    0.0000    0.0002   :     WARNING :   > dvmax[0] 0.0010  
     0004            :           TO BT BR BR BR BT SA :      54       54  :          54       648         0: 0.000      0.0078    0.0000    0.0002   :     WARNING :   > dvmax[0] 0.0010  
     0005            :           TO BT BR BT BT BT SA :      10       10  :          10       120         0: 0.000      0.0087    0.0000    0.0004   :     WARNING :   > dvmax[0] 0.0010  
     0006            :        TO BT BR BR BT BT BT SA :       8        8  :           8        96         0: 0.000      0.0068    0.0000    0.0004   :     WARNING :   > dvmax[0] 0.0010  
     0007            :                       TO SC SA :       7        7  :           7        84         0: 0.000      0.0002    0.0000    0.0000   :             :  
     0008            :                 TO BT BT SC SA :       5        5  :           5        60         0: 0.000      0.0002    0.0000    0.0000   :             :  
     0009            :           TO BT BR BR BT BR SA :       3        3  :           3        36         3: 0.083      0.0670    0.0000    0.0037   :       ERROR :   > dvmax[1] 0.0200  
     0010            :                       TO BT AB :       3        3  :           3        36         0: 0.000      0.0001    0.0000    0.0000   :             :  
     0011            :        TO BT BR BR BR BR BT SA :       3        3  :           3        36         0: 0.000      0.0099    0.0000    0.0008   :     WARNING :   > dvmax[0] 0.0010  
     0012            :                 TO BT SC BT SA :       3        3  :           3        36         0: 0.000      0.0005    0.0000    0.0000   :             :  
     0013            :           TO BT BT SC BT BT SA :       2        2  :           2        24         0: 0.000      0.0020    0.0000    0.0001   :     WARNING :   > dvmax[0] 0.0010  
     0015            :              TO BR SC BT BT SA :       1        1  :           1        12         0: 0.000      0.0005    0.0000    0.0000   :             :  
     0016            :     TO BT BR BR BR BR BR BT SA :       1        1  :           1        12         0: 0.000      0.0004    0.0000    0.0000   :             :  
     0017            :  TO BT SC BR BR BR BR BR BR BR :       1        1  :           1        12         0: 0.000      0.0020    0.0000    0.0002   :     WARNING :   > dvmax[0] 0.0010  
    .
    ab.rc     .rc  99      [0, 0, 99] 
    ab.rc.c2p .rc   0  .mx   0.001 .cut   1.500/  2.000/  2.500   seqmat_ana :  0.00143  pflags_ana :        0  seqhis_ana :        0   
    ab.rc.rdv .rc   0  .mx   0.072 .cut   0.010/  0.100/  1.000      rpol_dv :  0.00787    rpost_dv :   0.0719   
    ab.rc.pdv .rc  99  .mx   0.067 .cut   0.001/  0.020/  0.100        ox_dv :    0.067   
    .
    [2019-06-20 16:14:24,490] p266034 {tboolean.py:71} CRITICAL -  RC 99 


This multi reflectors "TO BT BR BR BT BR SA" ending up with z-position off::

    In [4]: a.oxa
    Out[4]: 
    A()sliced
    A([[[ 617.6166,  784.5179, -320.8895,   11.3237],
        [   0.5998,    0.7619,   -0.2443,    1.    ],
        [   0.7802,   -0.6247,   -0.0325,  380.    ]],

       [[-784.5179,  160.4182, -181.011 ,   10.5866],
        [  -0.9716,    0.1987,   -0.1286,    1.    ],
        [  -0.1947,   -0.9799,   -0.0428,  380.    ]],

       [[-784.5179, -222.7686,  -54.3965,   10.6167],
        [  -0.961 ,   -0.2729,    0.0441,    1.    ],
        [   0.2656,   -0.9558,   -0.1262,  380.    ]]], dtype=float32)

    In [5]: b.oxa
    Out[5]: 
    A()sliced
    A([[[ 617.6168,  784.5179, -320.8486,   11.3237],
        [   0.5998,    0.7619,   -0.2442,    1.    ],
        [   0.7802,   -0.6247,   -0.0326,  380.    ]],

       [[-784.5179,  160.4182, -180.9848,   10.5866],
        [  -0.9716,    0.1987,   -0.1286,    1.    ],
        [  -0.1947,   -0.9799,   -0.0428,  380.    ]],

       [[-784.5179, -222.7686,  -54.3296,   10.6167],
        [  -0.961 ,   -0.2729,    0.0442,    1.    ],
        [   0.2656,   -0.9558,   -0.1263,  380.    ]]], dtype=float32)

    In [6]: a.oxa - b.oxa
    Out[6]: 
    A()sliced
    A([[[-0.0001, -0.0001, -0.0408,  0.    ],
        [-0.    , -0.    , -0.    ,  0.    ],
        [ 0.    ,  0.    ,  0.0001,  0.    ]],

       [[ 0.    ,  0.    , -0.0262,  0.    ],
        [ 0.    , -0.    , -0.    ,  0.    ],
        [ 0.    , -0.    ,  0.    ,  0.    ]],

       [[ 0.    ,  0.    , -0.067 , -0.    ],               ###### 
        [-0.    , -0.    , -0.0001,  0.    ],
        [ 0.    , -0.    ,  0.    ,  0.    ]]], dtype=float32)


Looking in the visualization *tv* see that those three photons manage to 
bounce around inside the polycone neck. 

::

    In [1]: ab.aselhis = "TO BT BT SA"          # set history selection, in aligned mode

    In [2]: a.oxa.shape
    Out[2]: (8681, 3, 4)

    In [3]: b.oxa.shape
    Out[3]: (8681, 3, 4)

    In [4]: 8681*12
    Out[4]: 104172          ## nelem 


    In [8]: d = a.oxa - b.oxa ; d
    Out[8]: 
    A()sliced
    A([[[-0.0001,  0.0004,  0.    ,  0.    ],
        [-0.    ,  0.    , -0.    ,  0.    ],
        [-0.    ,  0.    ,  0.    ,  0.    ]],

       [[ 0.    ,  0.    ,  0.0001,  0.    ],
        [ 0.    ,  0.    ,  0.    ,  0.    ],
        [ 0.    ,  0.    , -0.    ,  0.    ]],

       [[-0.0002,  0.0002,  0.    ,  0.    ],
        [-0.    ,  0.    , -0.    ,  0.    ],
        [ 0.    ,  0.    ,  0.    ,  0.    ]],

The larger numbers of position and time give the deviations rather than 
direction and polarization with values in 0. to 1.:: 


    In [10]: a.oxa[:2]
    Out[10]: 
    A()sliced
    A([[[-120.5809,  369.9754,  784.5179,    6.6946],
        [  -0.1466,    0.4497,    0.8811,    1.    ],
        [  -0.0092,   -0.8913,    0.4534,  380.    ]],

       [[   0.8022,    0.0085,  784.518 ,    6.6567],
        [   0.0896,    0.0009,    0.996 ,    1.    ],
        [   0.    ,   -1.    ,    0.001 ,  380.    ]]], dtype=float32)

    In [11]: b.oxa[:2]
    Out[11]: 
    A()sliced
    A([[[-120.5807,  369.975 ,  784.5179,    6.6946],
        [  -0.1466,    0.4497,    0.8811,    1.    ],
        [  -0.0092,   -0.8913,    0.4534,  380.    ]],

       [[   0.8021,    0.0085,  784.5179,    6.6567],
        [   0.0896,    0.0009,    0.996 ,    1.    ],
        [  -0.    ,   -1.    ,    0.001 ,  380.    ]]], dtype=float32)



Largest deviations in direction and polarization less than 2e-6 level:: 

    In [15]: (d[:,1:].max(), d[:,1:].min())
    Out[15]: 
    (A()sliced
     A(0., dtype=float32), A()sliced
     A(-0., dtype=float32))

    In [16]: np.set_printoptions(suppress=False)

    In [17]: (d[:,1:].max(), d[:,1:].min())
    Out[17]: 
    (A()sliced
     A(1.3439e-06, dtype=float32), A()sliced
     A(-1.5851e-06, dtype=float32))



Hmm need different deviation cuts for the position and time than 
for direction and polarization OR do relative cuts ? Relative to domain
extent perhaps. Going relative to each value is a recipe for problems.

Its simpler to explain and understand fixed absolute cuts ? 
Hmm need to try with some big geometry. 







Issue : many ana fails from deviations
-----------------------------------------

* this is non-aligned comparison : 
  so it relies on accidental history aligment 
  making it suffer from poor stats  

  * also was not using "--reflectcheat" so it really has little hope 


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



Jump into ipython to check the those deviations and peruse the metadata with::

    tp(){  tboolean-;PROXYLV=18 tboolean-proxy-ip $* ; }



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





