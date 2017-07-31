Torus Quartic Cubic Numerical Stability
=========================================

Issue
------

Ray trace of torus, is afflicted by fake intersects assumed 
to be due to numerical problems in cubic/quartic root finding.

::

   tboolean-;tboolean-torus


Workaround Ideas
------------------


* intersect with bounding planes


* pullout a uniform scaling normalizing torus major radius to 1, 
  to try to avoid very large coefficients  

* pre-checking virtual cone, swept sphere intersect to 
  skip the root finding ?


* review code for non robust subtractions and find equivalent alt 



Small neumark[0] not the only issue
-------------------------------------

See big resiuals an qs with non-small neumark[0]

::

    torus residual   193.3350  qsd     3.0383  qn(      -14.5,       85.7,       -244,        278) efg(       7.04,      -3.21,       2.38 ) neumark(       14.1,         40,      -10.3 )
    torus residual   194.9941  qsd     3.0430  qn(      -14.2,       83.3,       -239,        278) efg(       8.06,       -4.1,          5 ) neumark(       16.1,         45,      -16.8 )
    torus residual   192.0515  qsd     3.0351  qn(      -14.7,       87.6,       -248,        278) efg(       6.21,      -2.51,      0.831 ) neumark(       12.4,       35.3,      -6.29 )
    torus residual   193.8790  qsd     3.0398  qn(      -14.4,       84.9,       -242,        278) efg(       7.38,      -3.51,       3.16 ) neumark(       14.8,       41.8,      -12.3 )
    torus residual   192.3991  qsd     3.0359  qn(      -14.7,       87.1,       -247,        278) efg(       6.44,       -2.7,       1.21 ) neumark(       12.9,       36.7,      -7.28 )
    torus residual   193.3884  qsd     3.0384  qn(      -14.5,       85.6,       -244,        278) efg(       7.07,      -3.24,       2.45 ) neumark(       14.1,       40.2,      -10.5 )
    torus residual   192.3433  qsd     3.0358  qn(      -14.7,       87.1,       -247,        278) efg(        6.4,      -2.67,       1.14 ) neumark(       12.8,       36.4,      -7.12 )
    torus residual   193.1987  qsd     3.0379  qn(      -14.5,       85.9,       -244,        278) efg(       6.95,      -3.14,       2.19 ) neumark(       13.9,       39.6,      -9.84 )
    torus residual   194.6670  qsd     3.0420  qn(      -14.2,       83.8,       -240,        278) efg(       7.86,      -3.93,       4.43 ) neumark(       15.7,       44.1,      -15.4 )




Problems seem to correspond at small neumark[0]
-------------------------------------------------


::


    torus qsd    32.6273  qn(      -15.4,       92.6,       -255,        270) efg(       3.38,      0.066,     -0.984 )
    torus qsd    18.5793  qn(      -15.4,       92.8,       -256,        270) efg(       3.29,     0.0677,     -0.972 )
    torus qsd    12.8417  qn(      -15.5,         93,       -256,        270) efg(       3.15,     0.0385,      -0.95 )
    torus qsd    10.8251  qn(      -15.3,       91.3,       -253,        270) efg(       4.13,     0.0204,     -0.954 )
    torus qsd    10.2456  qn(      -15.5,       92.9,       -256,        270) efg(       3.37,      0.477,     -0.953 )
    torus qsd    13.5304  qn(      -15.6,         94,       -258,        270) efg(       2.53,     0.0116,      -0.75 )
    torus qsd    16.7430  qn(      -15.3,       92.1,       -254,        270) efg(       3.71,     0.0529,     -0.999 )
    torus qsd    10.6018  qn(      -15.4,       92.6,       -255,        270) efg(        3.5,      0.249,     -0.987 )
    torus qsd    18.1685  qn(      -15.4,       92.6,       -255,        270) efg(       3.44,      0.129,     -0.988 )
    torus qsd    11.2096  qn(      -15.6,       93.6,       -257,        270) efg(       2.78,     0.0126,     -0.849 )
    torus qsd    11.8618  qn(      -15.1,       90.2,       -250,        268) efg(       4.61,     0.0172,     -0.756 )
    torus qsd    11.4991  qn(      -15.1,       90.4,       -251,        268) efg(       4.47,    0.00248,     -0.813 )
    torus qsd    18.0072  qn(      -15.4,       92.3,       -254,        268) efg(       3.47,      0.336,     -0.989 )
    torus qsd    17.5617  qn(      -15.1,         90,       -250,        268) efg(       4.73,     0.0143,     -0.702 )
    torus qsd    16.4299  qn(      -15.4,       92.1,       -254,        268) efg(       3.52,      0.128,     -0.999 )
    torus qsd    10.8914  qn(      -15.4,       92.3,       -254,        268) efg(       3.41,      0.151,     -0.995 )
    torus qsd    16.5060  qn(      -15.4,       92.2,       -254,        268) efg(       3.41,     0.0625,     -0.998 )
    torus qsd    11.9469  qn(      -15.4,       92.5,       -254,        268) efg(       3.25,     0.0379,     -0.986 )
    torus qsd    12.3096  qn(      -14.9,       88.8,       -246,        265) efg(       5.12,     0.0155,     -0.341 )
    torus qsd    13.9225  qn(      -15.2,         91,       -250,        265) efg(       3.81,     0.0167,     -0.949 )
    torus qsd    13.2572  qn(      -15.4,       91.8,       -252,        265) efg(       3.46,      0.396,     -0.988 )
    torus qsd    11.6840  qn(      -15.3,       91.4,       -251,        265) efg(       3.55,     0.0893,     -0.989 )
    torus qsd    25.1951  qn(      -15.4,       91.8,       -252,        265) efg(        3.4,      0.174,     -0.997 )
    torus qsd    26.0162  qn(      -15.3,       91.7,       -252,        265) efg(       3.41,     0.0907,     -0.998 )
    torus qsd    10.3480  qn(      -15.4,         92,       -252,        265) efg(       3.18,     0.0271,     -0.996 )
    torus qsd    12.7627  qn(      -15.5,       92.7,       -254,        265) efg(       2.79,    0.00792,     -0.945 )
    torus qsd    18.0496  qn(      -15.3,       91.5,       -251,        265) efg(       3.54,      0.111,      -0.99 )


    Looks like one problem from small neumark[0] which is f**2

    torus qsd    14.1680  qn(        -15,       88.6,       -244,        261) efg(       4.68,    0.00684,     -0.415 ) neumark(       9.37,       23.6,  -4.67e-05 )
    torus qsd    16.0347  qn(      -15.1,       89.6,       -246,        261) efg(       4.11,    0.00941,     -0.743 ) neumark(       8.23,       19.9,  -8.85e-05 )
    torus qsd    26.5659  qn(      -15.3,         91,       -249,        261) efg(       3.38,      0.189,     -0.974 ) neumark(       6.75,       15.3,    -0.0359 )
    torus qsd    24.6562  qn(      -15.3,       90.9,       -249,        261) efg(       3.35,     0.0487,     -0.975 ) neumark(       6.69,       15.1,   -0.00237 )
    torus qsd    10.3825  qn(      -15.3,       91.4,       -249,        261) efg(       3.05,     0.0135,     -0.999 ) neumark(        6.1,       13.3,  -0.000183 )
    torus qsd    31.3347  qn(      -15.4,       92.1,       -251,        261) efg(       2.63,    0.00727,     -0.973 ) neumark(       5.26,       10.8,  -5.28e-05 )
    torus qsd    47.3215  qn(      -15.2,       90.6,       -249,        264) efg(       3.93,     0.0101,     -0.905 ) neumark(       7.85,         19,  -0.000103 )
    torus qsd    10.0382  qn(      -15.3,       91.6,       -251,        264) efg(       3.38,     0.0872,     -0.997 ) neumark(       6.76,       15.4,    -0.0076 )
    torus qsd    12.2173  qn(      -15.4,         92,       -252,        264) efg(        3.1,     0.0231,     -0.996 ) neumark(       6.19,       13.6,  -0.000536 )
    torus qsd    10.3301  qn(      -15.6,       93.3,       -254,        264) efg(       2.35,    0.00374,     -0.838 ) neumark(       4.69,       8.85,   -1.4e-05 )
    torus qsd    23.8388  qn(      -14.9,       88.6,       -247,        267) efg(       5.42,     0.0146,     -0.215 ) neumark(       10.8,       30.3,  -0.000212 )
    torus qsd    20.5006  qn(      -15.2,         91,       -251,        267) efg(       4.01,     0.0194,     -0.935 ) neumark(       8.01,       19.8,  -0.000378 )
    torus qsd    12.4469  qn(      -15.4,       92.2,       -254,        267) efg(       3.44,      0.474,     -0.982 ) neumark(       6.88,       15.8,     -0.225 )
    torus qsd    10.0255  qn(      -15.4,       92.2,       -254,        267) efg(       3.44,      0.368,     -0.989 ) neumark(       6.87,       15.8,     -0.135 )





Check
------

::

    In [171]: run cubic.py
    z**3 - 7.0*z**2 + 41.0*z - 87.0
    a:-7.00000000000000 b:41.0000000000000 c:-87.0000000000000  
    y**3 + 24.6666666666667*y - 16.7407407407407
    p:24.6666666666667 q:-16.7407407407407 (p/3)^3:555.862825788752  (q/2)^2: 70.0631001371742  
    delta:67600.0000000000 disc:625.925925925926 sdisc:25.0185116648838 
    complex coeff, descending 
    3 : 1.00000000000000     0 
    2 : -7.00000000000000     0 
    1 : 41.0000000000000     0 
    0 : -87.0000000000000     0 
    iroot: (3, (2+5j), (2-5j))  (from input) 
    oroot: [3.00000000000000, 2.0 - 5.0*I, 2.0 + 5.0*I]  (from solving the expression) 


    delta:cu blyth$ clang Vecgeom_Solve.cc -lc++ && ./a.out && rm a.out
    test_one_real_root  r0 : (3,0) r1 : (2,5) r2 : (2,-5)
     nr 1 zr0      (3,0)      (2,5)     (2,-5)  r1          3  abc (      -7      41     -87)  pq ( 24.6667 -16.7407)  delta 67600 disc 625.926 sdisc 25.0185 VECGEOM 
     nr 1 zr0      (3,0)      (2,5)     (2,-5)  r1          3  abc (      -7      41     -87)  pq ( 24.6667 -16.7407)  delta 67600 disc 625.926 sdisc 25.0185 UNOBFUSCATED 
     nr 1 zr0      (3,0)      (2,5)     (2,-5)  r1          3  abc (      -7      41     -87)  pq ( 24.6667 -16.7407)  delta 67600 disc 625.926 sdisc 25.0185 UNOBFUSCATED ROBUSTQUAD 





CubicTest rootfinding tests
------------------------------

Currently unclear what disposition of cubic roots/coeffs is susceptible
to the numerical error.


::

    delta:cu blyth$ clang Vecgeom_Solve.cc -lc++ && ./a.out && rm a.out


    sc[0]: 1 sc[1]: 1000 sc[2]: 100
     nr 3 zr0      (1,0)      (2,0)      (3,0)  r1          1          2          3  co         -6         11         -6  VECGEOM 
     nr 3 zr0      (1,0)      (2,0)      (3,0)  r1          1          2          3  co         -6         11         -6  UNOBFUSCATED 
     nr 3 zr0      (1,0)      (2,0)      (3,0)  r1          1          2          3  co         -6         11         -6  UNOBFUSCATED ROBUSTQUAD 

     nr 3 zr0    (101,0)      (2,0)      (3,0)  r1    2.00101    2.99898        101  co       -606        511       -106  VECGEOM 
     nr 3 zr0    (101,0)      (2,0)      (3,0)  r1    2.00099      2.999        101  co       -606        511       -106  UNOBFUSCATED 
     nr 3 zr0    (101,0)      (2,0)      (3,0)  r1    2.00099      2.999        101  co       -606        511       -106  UNOBFUSCATED ROBUSTQUAD 

     nr 3 zr0    (201,0)      (2,0)      (3,0)  r1    2.00398    2.99599        201  co      -1206       1011       -206  VECGEOM 
     nr 3 zr0    (201,0)      (2,0)      (3,0)  r1    2.00395    2.99603        201  co      -1206       1011       -206  UNOBFUSCATED 
     nr 3 zr0    (201,0)      (2,0)      (3,0)  r1    2.00395    2.99603        201  co      -1206       1011       -206  UNOBFUSCATED ROBUSTQUAD 

     nr 3 zr0    (301,0)      (2,0)      (3,0)  r1    2.00794    2.99203        301  co      -1806       1511       -306  VECGEOM 
     nr 3 zr0    (301,0)      (2,0)      (3,0)  r1          2          3        301  co      -1806       1511       -306  UNOBFUSCATED 
     nr 3 zr0    (301,0)      (2,0)      (3,0)  r1          2          3        301  co      -1806       1511       -306  UNOBFUSCATED ROBUSTQUAD 

     nr 3 zr0    (401,0)      (2,0)      (3,0)  r1      2.016    2.98393        401  co      -2406       2011       -406  VECGEOM 
     nr 3 zr0    (401,0)      (2,0)      (3,0)  r1    2.01594    2.98403        401  co      -2406       2011       -406  UNOBFUSCATED 
     nr 3 zr0    (401,0)      (2,0)      (3,0)  r1    2.01594    2.98403        401  co      -2406       2011       -406  UNOBFUSCATED ROBUSTQUAD 

     nr 3 zr0    (501,0)      (2,0)      (3,0)  r1    2.03243    2.96751        501  co      -3006       2511       -506  VECGEOM 
     nr 3 zr0    (501,0)      (2,0)      (3,0)  r1          2          3        501  co      -3006       2511       -506  UNOBFUSCATED 
     nr 3 zr0    (501,0)      (2,0)      (3,0)  r1          2          3        501  co      -3006       2511       -506  UNOBFUSCATED ROBUSTQUAD 

     nr 3 zr0    (601,0)      (2,0)      (3,0)  r1    2.10504    2.89484        601  co      -3606       3011       -606  VECGEOM 
     nr 3 zr0    (601,0)      (2,0)      (3,0)  r1          2          3        601  co      -3606       3011       -606  UNOBFUSCATED 
     nr 3 zr0    (601,0)      (2,0)      (3,0)  r1          2          3        601  co      -3606       3011       -606  UNOBFUSCATED ROBUSTQUAD 

     nr 3 zr0    (701,0)      (2,0)      (3,0)  r1    2.06728     2.9326        701  co      -4206       3511       -706  VECGEOM 
     nr 3 zr0    (701,0)      (2,0)      (3,0)  r1          2          3        701  co      -4206       3511       -706  UNOBFUSCATED 
     nr 3 zr0    (701,0)      (2,0)      (3,0)  r1          2          3        701  co      -4206       3511       -706  UNOBFUSCATED ROBUSTQUAD 

     nr 3 zr0    (801,0)      (2,0)      (3,0)  r1    2.06728     2.9326        801  co      -4806       4011       -806  VECGEOM 
     nr 3 zr0    (801,0)      (2,0)      (3,0)  r1    2.06713    2.93281        801  co      -4806       4011       -806  UNOBFUSCATED 
     nr 3 zr0    (801,0)      (2,0)      (3,0)  r1    2.06713    2.93281        801  co      -4806       4011       -806  UNOBFUSCATED ROBUSTQUAD 

     nr 3 zr0    (901,0)      (2,0)      (3,0)  r1    2.14682    2.85306        901  co      -5406       4511       -906  VECGEOM 
     nr 3 zr0    (901,0)      (2,0)      (3,0)  r1          2          3        901  co      -5406       4511       -906  UNOBFUSCATED 
     nr 3 zr0    (901,0)      (2,0)      (3,0)  r1          2          3        901  co      -5406       4511       -906  UNOBFUSCATED ROBUSTQUAD 


