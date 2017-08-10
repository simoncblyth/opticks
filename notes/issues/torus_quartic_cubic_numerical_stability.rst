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








intersect_analytic_test with s=1
-----------------------------------


::


    simon:cu blyth$ intersect_analytic_test
    2017-08-08 20:19:40.961 INFO  [2552272] [OptiXTest::init@39] OptiXTest::init cu intersect_analytic_test.cu ptxpath /usr/local/opticks/build/optixrap/OptiXRap_generated_intersect_analytic_test.cu.ptx raygen intersect_analytic_test exception exception
    2017-08-08 20:19:40.994 INFO  [2552272] [OptiXTest::Summary@72] intersect_analytic_test cu intersect_analytic_test.cu ptxpath /usr/local/opticks/build/optixrap/OptiXRap_generated_intersect_analytic_test.cu.ptx raygen intersect_analytic_test exception exception
    2017-08-08 20:19:40.994 INFO  [2552272] [OGeo::CreateInputUserBuffer@836] OGeo::CreateInputUserBuffer name planBuffer src shape 6,4 numBytes 96 elementSize 16 size 6
## intersect_analytic_test 0
    // pid 0 
    // csg_intersect_torus_test  r R rmax (10 100 110) ray_origin (-220 0 0) ray_direction (1 0 0) 
    // csg_intersect_torus_test t_min          0 x_expect p.x (      -110       -110)  isect:(    -1.000      0.000      0.000    110.000) p:(  -110.000      0.000      0.000) 
    // csg_intersect_torus_test t_min        110 x_expect p.x (       -90   -90.0002)  isect:(     1.000     -0.000      0.000    130.000) p:(   -90.000      0.000      0.000) 
    // csg_intersect_torus_test t_min        130 x_expect p.x (        90    90.0002)  isect:(    -1.000     -0.000      0.000    310.000) p:(    90.000      0.000      0.000) 
    // csg_intersect_torus_test t_min        310 x_expect p.x (       110        110)  isect:(     1.000      0.000      0.000    330.000) p:(   110.000      0.000      0.000) 
    save result npy to $TMP/oxrap/intersect_analytic_test.npy




suspect cbrt segv may be resource issue
------------------------------------------

* sorta behaving like every call takes more stack ?? possibly nothing special about cbrt(double) other
  that it corresponds to a large chunk of code 

* why SolveCubic.h which is very similar to SolveCubicNumericalRecipe.h is less taxing ?

  * it uses poly longdiv, resulting in less double precision trig 


cbrtTest.cu::

     76     unsigned nr = 0 ;  
     77     Solve_t xx[3] ; 
     78     nr = SolveCubic(p,q,r,xx, 0u ); 
     79     nr = SolveCubic(p,q,r,xx, 0u ); 
     80     //nr = SolveCubic(p,q,r,xx, 0u ); 
     81     //nr = SolveCubic(p,q,r,xx, 0u ); 
     82 
     83     // HMM : doing twice works with default stacksize of 1024, more than twice segv in createProgramFromPTX
     84     //       three times works with 2* stacksize 
     85 
       


optixtest CubicRoot discrep
-------------------------------

::

    simon:cu blyth$ t optixtest
    optixtest () 
    { 
        local nam=${1:-cbrtTest};
        local exe=/tmp/$nam;
        local ptx=/tmp/$nam.ptx;
        local cc=../tests/$nam.cc;
        local cu=$nam.cu;
        local ver=OptiX_380;
        local inc=/Developer/$ver/include;
        local lib=/Developer/$ver/lib64;
        clang -std=c++11 -I/usr/local/cuda/include -I$inc -L$lib -loptix -lc++ -Wl,-rpath,$lib $cc -o $exe;
        nvcc -arch=sm_30 -m64 -std=c++11 -use_fast_math -ptx $cu -I$inc -o $ptx;
        echo $exe $ptx $nam;
        $exe $ptx $nam
    }



Discrepancy for the tough root, between SolveCubicNumericalRecipe.h and SolveCubic.h::

    simon:cu blyth$ optixtest
    ./SolveCubicNumericalRecipe.h(20): warning: variable "otwo" was declared but never referenced

    /tmp/cbrtTest /tmp/cbrtTest.ptx cbrtTest
     ptxpath /tmp/cbrtTest.ptx raygen cbrtTest
     ptxpath /tmp/cbrtTest.ptx raygen cbrtTest exception exception
    cbrtTest crf:3.000000 crd:3  
    SolveCubicTest pqr (        49526.8     4.08573e+08    -1.48348e+06)  x^3 + p x^2 + q x + r = 0   -r/q 0.00363087   
    nr 3  
    xx[0] =        -39069.1  residual     -0.00236215  x3210 (   -5.96349e+13     7.55974e+13    -1.59626e+13    -1.48348e+06) x3_x2     1.59626e+13 x1_x0    -1.59626e+13 x3_x2_x1_x0     -0.00195312    
    xx[1] =      0.00363087  residual      -0.0029249  x3210 (    4.78666e-08        0.652923     1.48348e+06    -1.48348e+06) x3_x2        0.652923 x1_x0       -0.655848 x3_x2_x1_x0      -0.0029249    
    xx[2] =        -10457.7  residual    -0.000507358  x3210 (   -1.14369e+12     5.41643e+12    -4.27274e+12    -1.48348e+06) x3_x2     4.27274e+12 x1_x0    -4.27274e+12 x3_x2_x1_x0     -0.00146484    
    simon:cu blyth$ 
    simon:cu blyth$ 
    simon:cu blyth$ optixtest

    /tmp/cbrtTest /tmp/cbrtTest.ptx cbrtTest
     ptxpath /tmp/cbrtTest.ptx raygen cbrtTest
     ptxpath /tmp/cbrtTest.ptx raygen cbrtTest exception exception
    cbrtTest crf:3.000000 crd:3  
    SolveCubicTest pqr (        49526.8     4.08573e+08    -1.48348e+06)  x^3 + p x^2 + q x + r = 0   -r/q 0.00363087   
    nr 3  
    xx[0] =    4.94066e-324  residual    -1.48348e+06  x3210 (              0               0    2.01862e-315    -1.48348e+06) x3_x2               0 x1_x0    -1.48348e+06 x3_x2_x1_x0    -1.48348e+06    
    xx[1] =        -39069.1  residual    -1.48348e+06  x3210 (   -5.96349e+13     7.55974e+13    -1.59626e+13    -1.48348e+06) x3_x2     1.59626e+13 x1_x0    -1.59626e+13 x3_x2_x1_x0    -1.48348e+06    
    xx[2] =        -10457.7  residual    -1.48348e+06  x3210 (   -1.14369e+12     5.41642e+12    -4.27273e+12    -1.48348e+06) x3_x2     4.27273e+12 x1_x0    -4.27274e+12 x3_x2_x1_x0    -1.48348e+06    
    simon:cu blyth$ 


Without use_fast_math get same, this is expected as should be done pure double::

    SolveCubicTest pqr (        49526.8     4.08573e+08    -1.48348e+06)  x^3 + p x^2 + q x + r = 0   -r/q 0.00363087   
    nr 3  
    xx[0] =    4.94066e-324  residual    -1.48348e+06  x3210 (              0               0    2.01862e-315    -1.48348e+06) x3_x2               0 x1_x0    -1.48348e+06 x3_x2_x1_x0    -1.48348e+06    
    xx[1] =        -39069.1  residual    -1.48348e+06  x3210 (   -5.96349e+13     7.55974e+13    -1.59626e+13    -1.48348e+06) x3_x2     1.59626e+13 x1_x0    -1.59626e+13 x3_x2_x1_x0    -1.48348e+06    
    xx[2] =        -10457.7  residual    -1.48348e+06  x3210 (   -1.14369e+12     5.41642e+12    -4.27273e+12    -1.48348e+06) x3_x2     4.27273e+12 x1_x0    -4.27274e+12 x3_x2_x1_x0    -1.48348e+06    
    simon:cu blyth$ 





GPU segv with pure double SolveCubicNumericalRecipe.h
--------------------------------------------------------

::

    simon:cu blyth$ intersect_analytic_test
    2017-08-10 10:19:42.377 INFO  [2804492] [OptiXTest::init@39] OptiXTest::init cu intersect_analytic_test.cu ptxpath /usr/local/opticks/build/optixrap/OptiXRap_generated_intersect_analytic_test.cu.ptx raygen intersect_analytic_test exception exception
    Segmentation fault: 11
    simon:cu blyth$ 



* segv happens early, ptx level ?
* somehow OptiX is implicated as pure CUDA in SolveCubicNumericalRecipeTest.cu does not have the issue.
* binary code search points finger at **cbrt**


Observe that difficult to determine root corresponds to "x = -r/q" 
where "qx + r" x1_x0 is close to zero, from a subtraction of two large values. 

Where "x = -r/q"    -> x^3 + p x^2 = 0   

::

    simon:cu blyth$ nvcc -arch=sm_30 SolveCubicNumericalRecipeTest.cu -run ; rm a.out
    SolveCubicTest pqr (        49526.8     4.08573e+08    -1.48348e+06)  x^3 + p x^2 + q x + r = 0   -r/q 0.00363087   
    nr 3  
    xx[0] =        -39069.1  residual     -0.00236215  x3210 (   -5.96349e+13     7.55974e+13    -1.59626e+13    -1.48348e+06) x3_x2     1.59626e+13 x1_x0    -1.59626e+13 x3_x2_x1_x0     -0.00195312    
    xx[1] =      0.00363087  residual      -0.0029249  x3210 (    4.78666e-08        0.652923     1.48348e+06    -1.48348e+06) x3_x2        0.652923 x1_x0       -0.655848 x3_x2_x1_x0      -0.0029249    
    xx[2] =        -10457.7  residual    -0.000507358  x3210 (   -1.14369e+12     5.41643e+12    -4.27274e+12    -1.48348e+06) x3_x2     4.27274e+12 x1_x0    -4.27274e+12 x3_x2_x1_x0     -0.00146484    
    simon:cu blyth$ 


    // use_fast_math  should have no effect with doubles, just checkin

    simon:cu blyth$ nvcc -arch=sm_30 -use_fast_math SolveCubicNumericalRecipeTest.cu -run ; rm a.out
    SolveCubicTest pqr (        49526.8     4.08573e+08    -1.48348e+06)  x^3 + p x^2 + q x + r = 0   -r/q 0.00363087   
    nr 3  
    xx[0] =        -39069.1  residual     -0.00236215  x3210 (   -5.96349e+13     7.55974e+13    -1.59626e+13    -1.48348e+06) x3_x2     1.59626e+13 x1_x0    -1.59626e+13 x3_x2_x1_x0     -0.00195312    
    xx[1] =      0.00363087  residual      -0.0029249  x3210 (    4.78666e-08        0.652923     1.48348e+06    -1.48348e+06) x3_x2        0.652923 x1_x0       -0.655848 x3_x2_x1_x0      -0.0029249    
    xx[2] =        -10457.7  residual    -0.000507358  x3210 (   -1.14369e+12     5.41643e+12    -4.27274e+12    -1.48348e+06) x3_x2     4.27274e+12 x1_x0    -4.27274e+12 x3_x2_x1_x0     -0.00146484    
    simon:cu blyth$ 


Pure CUDA nvcc giving same as clang::

    // purely doubles
    simon:cu blyth$ clang SolveCubicTest.cc -lc++ && ./a.out && rm a.out
     nr 3 zr0      (0,0)      (0,0)      (0,0)  r1   -39069.1 0.00363087   -10457.7  abc ( 49526.8 4.08573e+08 -1.48348e+06)  pq ( -4.09062e+08 2.25375e+12)  delta -1.36653e+26 disc -1.2653e+24 sdisc nan UNOBFUSCATED ROBUST_VIETA ROBUSTQUAD_1 ROBUSTCUBIC_0 ROBUSTCUBIC_1 ROBUSTCUBIC_2 
     i 0 rt/err/del/frac ( -39069.1 1.80948e-12 -0.00202268 ; -4.63149e-17)
     i 1 rt/err/del/frac (0.00363087 5.70431e-12 -0.00233063 ; 1.57106e-09)
     i 2 rt/err/del/frac ( -10457.7 2.6574e-12 -0.00079512 ; -2.54109e-16)


Spruce-ing up the old SolveCubicRoot to be pure double::

    // csg_intersect_torus_test  r R rmax (10 100 110) ray_origin (-0.646 0.005311 3.947) ray_direction (0.00059 0.0007738 -0.009953) 
    // csg_intersect_torus R r unit (99.9955 9.99955 0.0100005)  oxyz (-64.5971 0.531076 394.682) sxyz (0.0589973 0.0773765 -0.995255 ) t_min (0)   
    // csg_intersect_torus HGIJKL (-301570 378.678 1.66907e+08 1 -793.158 169846)  ABCDE (1 -1586.32 968414 -2.69128e+08 2.86808e+10 ) 
    // csg_intersect_torus qn (-1586.32 968414 -2.69128e+08 2.86808e+10) reverse 0 
    SolveQuartic abcd (-1586.32 968414 -2.69128e+08 2.86808e+10)  pqr (49526.8 4.08573e+08 -1.48348e+06) 
    // SOLVE_QUARTIC_DEBUG.cubic_sqroot   pqr (         49526.79994          408572956.1         -1483476.478)  ireal 3  xx (0.00363087 -39069.1 -10457.7)
    // SOLVE_QUARTIC_DEBUG.cubic_sqroot   ireal 3 i 0 xx 0.00363087 residual 0.00265358  
    // SOLVE_QUARTIC_DEBUG.cubic_sqroot   ireal 3 i 1 xx -39069.1 residual 0.00728161  
    // SOLVE_QUARTIC_DEBUG.cubic_sqroot   ireal 3 i 2 xx -10457.7 residual 0.00251214  
    // SOLVE_QUARTIC_DEBUG cubic_sqroot h 0.0602567 
     t_cand 0  p0 (-64.5971 0.531076 394.682) 
    ERROR no isect 
    save result npy to $TMP/oxrap/intersect_analytic_test.npy
    simon:cu blyth$ 





in-the-hole artifact rings
-----------------------------

Examining artifact intersect in the hole obtained by dumping ray ori, dir.

::

    // csg_intersect_torus_scale_test uscale 100 
    // T(transform)
     100.000    0.000    0.000    0.000
       0.000  100.000    0.000    0.000
       0.000    0.000  100.000    0.000
       0.000    0.000    0.000    1.000
    // V(inverse)
       0.010    0.000    0.000    0.000
       0.000    0.010    0.000    0.000
       0.000    0.000    0.010    0.000
       0.000    0.000    0.000    1.000
    // Q(inverse-transposed)
       0.010    0.000    0.000    0.000
       0.000    0.010    0.000    0.000
       0.000    0.000    0.010    0.000
       0.000    0.000    0.000    1.000
    // pid 0 
    // csg_intersect_torus_test  r R rmax (10 100 110) ray_origin (-0.646 0.005311 3.947) ray_direction (0.00059 0.0007738 -0.009953) 
    // csg_intersect_torus R r unit (99.9955 9.99955 0.0100005)  oxyz (-64.5971 0.531076 394.682) sxyz (0.0589973 0.0773765 -0.995255 ) t_min (0)   
    // csg_intersect_torus HGIJKL (-301570 378.678 1.66907e+08 1 -793.158 169846)  ABCDE (1 -1586.32 968414 -2.69128e+08 2.86808e+10 ) 
    // csg_intersect_torus qn (-1586.32 968414 -2.69128e+08 2.86808e+10) reverse 0 
    SolveQuartic abcd (-1586.32 968414 -2.69128e+08 2.86808e+10)  pqr (49526.8 4.08573e+08 -1.48348e+06) 
    // SOLVE_QUARTIC_DEBUG.cubic_sqroot   pqr (49526.8 4.08573e+08 -1.48348e+06)  ireal 3  xx (0.00211941 -39069.1 -10457.7)
    // SOLVE_QUARTIC_DEBUG.cubic_sqroot   ireal 3 i 0 xx 0.00211941 residual -617545  
    // SOLVE_QUARTIC_DEBUG.cubic_sqroot   ireal 3 i 1 xx -39069.1 residual -617545  
    // SOLVE_QUARTIC_DEBUG.cubic_sqroot   ireal 3 i 2 xx -10457.7 residual -617545  
    // SOLVE_QUARTIC_DEBUG cubic_sqroot h 0.046037 
    // SOLVE_QUARTIC_DEBUG solve-exit  ireal 4 i 0 root 367.46 residual 7.28441e+07  dis12 ( 3386.31 241742 ) h 0.046037  pqr (49526.8 4.08573e+08 -1.48348e+06 )  j g/j (-846.578 -60435.4 )  
    // SOLVE_QUARTIC_DEBUG solve-exit  ireal 4 i 1 root 425.652 residual 7.28441e+07  dis12 ( 3386.31 241742 ) h 0.046037  pqr (49526.8 4.08573e+08 -1.48348e+06 )  j g/j (-846.578 -60435.4 )  
    // SOLVE_QUARTIC_DEBUG solve-exit  ireal 4 i 2 root 642.438 residual 5.20213e+09  dis12 ( 3386.31 241742 ) h 0.046037  pqr (49526.8 4.08573e+08 -1.48348e+06 )  j g/j (-846.578 -60435.4 )  
    // SOLVE_QUARTIC_DEBUG solve-exit  ireal 4 i 3 root 150.766 residual 5.19824e+09  dis12 ( 3386.31 241742 ) h 0.046037  pqr (49526.8 4.08573e+08 -1.48348e+06 )  j g/j (-846.578 -60435.4 )  
     t_cand 150.766  p0 (-55.7023 12.1968 244.631) 
     pr 57.022 float3 ori = make_float3(     -64.6f,    0.5311f,     394.7f); float3 dir = make_float3(     0.059f,   0.07738f,   -0.9953f); p (-55.7023 12.1968 244.631) 
     // csg_intersect_torus_test t_min          0    tt:(     0.002     -0.000      0.010    150.766) p:(   -55.705     12.197    244.642) 
    save result npy to $TMP/oxrap/intersect_analytic_test.npy
    simon:issues blyth$ 



Copying over pqr into SolveCubicTest gets close, see the small +ve cubic root has 60% error::

    simon:cu blyth$ clang SolveCubicTest.cc -lc++ && ./a.out && rm a.out
     nr 3 zr0      (0,0)      (0,0)      (0,0)  r1   -39069.1   -10457.7 0.00225949  abc ( 49526.8 4.08573e+08 -1.48348e+06)  pq ( -4.09062e+08 2.25375e+12)  delta -1.36653e+26 disc -1.2653e+24 sdisc nan UNOBFUSCATED ROBUST_VIETA ROBUSTQUAD_1 ROBUSTCUBIC_0 ROBUSTCUBIC_1 ROBUSTCUBIC_2 
     i 0 rt/err/del/frac ( -39069.1 0.000501256   -560315 ; -1.283e-08)
     i 1 rt/err/del/frac ( -10457.7 0.00187265   -560315 ; -1.79069e-07)
     i 2 rt/err/del/frac (0.00225949 0.00137139   -560315 ; 0.60695)
    simon:cu blyth$ 

    simon:cu blyth$ clang SolveCubicTest.cc -lc++ && ./a.out && rm a.out
     nr 3 zr0      (0,0)      (0,0)      (0,0)  r1   -39069.1 0.00429867   -10457.7  abc ( 49526.8 4.08573e+08 -1.48348e+06)  pq ( -4.09062e+08 2.25375e+12)  delta -1.36653e+26 disc -1.2653e+24 sdisc nan UNOBFUSCATED ROBUST_VIETA ROBUSTQUAD_1 ROBUSTCUBIC_0 ROBUSTCUBIC_1 ROBUSTCUBIC_2 
     i 0 rt/err/del/frac ( -39069.1 0.00044495   -497375 ; -1.13888e-08)
     i 1 rt/err/del/frac (0.00429867 0.000667799    272845 ; 0.15535)
     i 2 rt/err/del/frac ( -10457.7 0.000740136   -221456 ; -7.07742e-08)
    simon:cu blyth$ 



Wow getting the precise result requires to use purely doubles, even doubles converted from constant floats mess up precision::

    // constants converted from floats
    simon:cu blyth$ clang SolveCubicTest.cc -lc++ && ./a.out && rm a.out
     nr 3 zr0      (0,0)      (0,0)      (0,0)  r1   -39069.1 0.00429867   -10457.7  abc ( 49526.8 4.08573e+08 -1.48348e+06)  pq ( -4.09062e+08 2.25375e+12)  delta -1.36653e+26 disc -1.2653e+24 sdisc nan UNOBFUSCATED ROBUST_VIETA ROBUSTQUAD_1 ROBUSTCUBIC_0 ROBUSTCUBIC_1 ROBUSTCUBIC_2 
     i 0 rt/err/del/frac ( -39069.1 0.00044495   -497375 ; -1.13888e-08)
     i 1 rt/err/del/frac (0.00429867 0.000667799    272845 ; 0.15535)
     i 2 rt/err/del/frac ( -10457.7 0.000740136   -221456 ; -7.07742e-08)
    simon:cu blyth$ 
    simon:cu blyth$ 

    // purely doubles
    simon:cu blyth$ clang SolveCubicTest.cc -lc++ && ./a.out && rm a.out
     nr 3 zr0      (0,0)      (0,0)      (0,0)  r1   -39069.1 0.00363087   -10457.7  abc ( 49526.8 4.08573e+08 -1.48348e+06)  pq ( -4.09062e+08 2.25375e+12)  delta -1.36653e+26 disc -1.2653e+24 sdisc nan UNOBFUSCATED ROBUST_VIETA ROBUSTQUAD_1 ROBUSTCUBIC_0 ROBUSTCUBIC_1 ROBUSTCUBIC_2 
     i 0 rt/err/del/frac ( -39069.1 1.80948e-12 -0.00202268 ; -4.63149e-17)
     i 1 rt/err/del/frac (0.00363087 5.70431e-12 -0.00233063 ; 1.57106e-09)
     i 2 rt/err/del/frac ( -10457.7 2.6574e-12 -0.00079512 ; -2.54109e-16)


     137 static unsigned SolveCubicNumericalRecipe(Solve_t a, Solve_t b, Solve_t c, Solve_t* xx, unsigned )
     138 {
     139     //  p185 NUMERICAL RECIPES IN C 
     140     //  x**3 + a x**2 + b x + x = 0 
     141 
     142     const Solve_t zero(0) ;  
     143     const Solve_t one(1) ;  
     144     const Solve_t three(3) ;  
     145     const Solve_t othree = one/three ;
     146     const Solve_t nine(9) ;  
     147     const Solve_t two(2) ;  
     148     const Solve_t twentyseven(27) ;
     149     const Solve_t fiftyfour(54) ;
     150     const Solve_t twpi = M_PI*two  ;
     151 
     152     const Solve_t a3 = a*othree ;
     153     const Solve_t aa = a*a ;
     154     const Solve_t Q = (aa - three*b)/nine ;
     155     const Solve_t R = ((two*aa - nine*b)*a + twentyseven*c)/fiftyfour ;  // a,b,c real so Q,R real
     156     const Solve_t R2 = R*R ;
     157     const Solve_t Q3 = Q*Q*Q ;
     158     const Solve_t R2_Q3 = R2 - Q3 ;
     159 
     160     unsigned nr =  R2_Q3 < zero ? 3u : 1u ;
     161 
     162     if( nr == 3 ) // three real roots
     163     {
     164          const Solve_t theta = acos( R/sqrt(Q3) );
     165          const Solve_t qs = sqrt(Q);
     166 
     167          xx[0] = -two*qs*cos(theta*othree) - a3 ;
     168          xx[1] = -two*qs*cos((theta+twpi)*othree) - a3 ;
     169          xx[2] = -two*qs*cos((theta-twpi)*othree) - a3 ;
     170     }
     171     else
     172     {
     173          const Solve_t A = -copysign(one, R)*cbrt( fabs(R) +  sqrt(R2_Q3) ) ;
     174          const Solve_t B = A != zero ? Q/A : zero ;
     175 
     176          xx[0] = (A + B) - a3  ; 
     177     } 
     178 
     179 #ifdef SOLVE_QUARTIC_DEBUG
     180     rtPrintf("// SOLVE_QUARTIC_DEBUG.SolveCubicNumericalRecipe  "
     181              " abc (%20.10g %20.10g %20.10g) " 
     182              " nr %u "
     183              " xx (%g %g %g)"
     184              "\n"
     185              ,
     186              a,b,c
     187              ,
     188              nr
     189              ,
     190              xx[0],xx[1],xx[2]
     191             );
     192 #endif
     193     return nr ;
     194 }   



    simon:cu blyth$ clang SolveCubicTest.cc -lc++ && ./a.out && rm a.out
     nr 3 zr0      (0,0)      (0,0)      (0,0)  r1   -39069.1 0.00363087   -10457.7  abc ( 49526.8 4.08573e+08 -1.48348e+06)  pq ( -4.09062e+08 2.25375e+12)  delta -1.36653e+26 disc -1.2653e+24 sdisc nan UNOBFUSCATED ROBUST_VIETA ROBUSTQUAD_1 ROBUSTCUBIC_0 ROBUSTCUBIC_1 ROBUSTCUBIC_2 
     i 0 rt/err/del/frac ( -39069.1 1.80948e-12 -0.00202268 ; -4.63149e-17)
     i 1 rt/err/del/frac (0.00363087 5.70431e-12 -0.00233063 ; 1.57106e-09)
     i 2 rt/err/del/frac ( -10457.7 2.6574e-12 -0.00079512 ; -2.54109e-16)






::

    In [40]: d,e = -2.69128e+08,2.86808e+10

    In [43]: t = 150.766

    In [44]: t*d + e
    Out[44]: -11894552048.0

    In [45]: t*d
    Out[45]: -40575352048.0

    In [46]: e
    Out[46]: 28680800000.0

    In [47]: (t*d)/e
    Out[47]: -1.4147217667568548

    n [50]: math.sqrt(2)
    Out[50]: 1.4142135623730951




Proper normalization suffers familiar artifacts
--------------------------------------------------

::

    1583 static __device__
    1584 bool csg_intersect_torus(const quad& q0, const float& t_min, float4& isect, const float3& ray_origin, const float3& ray_direction )
    1585 {
    1586     const Torus_t R_ = q0.f.w ;
    1587     const Torus_t r_ = q0.f.z ;  // R_ > r_ by assertion, so torus has a hole   
    1588 
    1589     const Torus_t ss = dot( ray_direction, ray_direction );
    1590     const Torus_t unit = sqrt(ss);
    1591 
    1592     const Torus_t sx = ray_direction.x/unit ;
    1593     const Torus_t sy = ray_direction.y/unit ;
    1594     const Torus_t sz = ray_direction.z/unit ;
    1595 
    1596     const Torus_t ox = ray_origin.x/unit ;
    1597     const Torus_t oy = ray_origin.y/unit ;
    1598     const Torus_t oz = ray_origin.z/unit ;
    1599 
    1600     const Torus_t R = R_/unit ;
    1601     const Torus_t r = r_/unit ;
    1602 
    1603     // scaled ray dir, ori too close to origin for numerical comfort
    1604     // due to scale factors to enable use of small R_ r_ 
    1605     // so divide by unit to bring into viscinity of unity 
    1606     // but must treat all lengths same ... so the radii get blown up ???
    1607     // and upshot is the coeffs come out the same ???
    1608     //
    1609     // Need to check quartic coeff disparity to see what approach is best
    1610 


Arghh after implementing proper normalization using transform scaling etc 
and a common length unit, end up with same coeffs whether use scaling 
or not, and the same artifacts are manifest.

The prior artifact remission occurred when trying to both normalize ray direction
and length scale simultaneously with t scaling ???  So it probably corresponded
to a very small torus or smth like that ?

Actually its true it somehow scaling t-values to be smaller, would be beneficial::

    In [30]: 100**4
    Out[30]: 100000000

::

    In [34]: a,b,c,d,e = symbols("a,b,c,d,e")

    In [35]: et = a*t**4 + b*t**3 + c*t**2 + d*t + e

    In [36]: et
    Out[36]: a*t**4 + b*t**3 + c*t**2 + d*t + e

    In [37]: et.subs(t,t*100)
    Out[37]: 100000000*a*t**4 + 1000000*b*t**3 + 10000*c*t**2 + 100*d*t + e

    In [39]: et.subs(t,t/100)
    Out[39]: a*t**4/100000000 + b*t**3/1000000 + c*t**2/10000 + d*t/100 + e




Switching off scaling, making ray_direction normalized to 1. much reduces artifacts
--------------------------------------------------------------------------------------

But small issues remain, possibly from coeff cuts (added for artifact reduction pre-normalization) 

* ~/opticks_refs/torus_unscaled_crease_artifact.png 
* ~/opticks_refs/torus_normalized_ray_direction_cut_artifact

Normalizing seems effective way to reduce coeff disparity.


Select fakes artifact intersects in the hole
----------------------------------------

Ring artifacts appear from specific directions (close to axial but not axial) 
and move around like ripples as change close to axial viewpoint 

Need more systematic way to study : so capture ray param for some 
instances can examine with intersect_analytic_test 





::

      pr 0.3005 ray_origin (-0.387017 -0.122478 1.44327) ray_direction (0.000778693 0.00168768 -0.00982575 ) p (-0.278972 0.111691 0.0799291) 
      pr 0.348184 ray_origin (-0.387017 -0.122478 1.44327) ray_direction (0.000745689 0.00167414 -0.00983062 ) p (-0.346711 -0.0319875 0.911909) 
      pr 0.39452 ray_origin (-0.387017 -0.122478 1.44327) ray_direction (0.000632683 -0.000786791 -0.0099489 ) p (-0.364898 -0.149985 1.09546) 
      pr 0.380523 ray_origin (-0.387017 -0.122478 1.44327) ray_direction (0.000598338 -0.000765124 -0.00995272 ) p (-0.307345 -0.224359 0.11801) 
      pr 0.382953 ray_origin (-0.387017 -0.122478 1.44327) ray_direction (0.000564007 -0.000743459 -0.00995636 ) p (-0.317526 -0.21408 0.216552) 
      pr 0.393114 ray_origin (-0.387017 -0.122478 1.44327) ray_direction (0.000472168 -0.000678253 -0.00996579 ) p (-0.355577 -0.16764 0.779696) 
      pr 0.388127 ray_origin (-0.387017 -0.122478 1.44327) ray_direction (0.000221803 0.00126497 -0.00991719 ) p (-0.379671 -0.0805804 1.1148) 
      pr 0.399708 ray_origin (-0.387017 -0.122478 1.44327) ray_direction (0.000286536 -0.000503222 -0.00998322 ) p (-0.368239 -0.155458 0.789012) 
      pr 0.368297 ray_origin (-0.387017 -0.122478 1.44327) ray_direction (0.000170521 0.00119365 -0.00992704 ) p (-0.368177 0.00940458 0.34646) 
      pr 0.400145 ray_origin (-0.387017 -0.122478 1.44327) ray_direction (0.000215842 -0.000415247 -0.00998904 ) p (-0.361468 -0.171631 0.260867) 
      pr 0.398703 ray_origin (-0.387017 -0.122478 1.44327) ray_direction (0.000140053 0.00114637 -0.00993309 ) p (-0.384803 -0.104358 1.28626) 
      pr 0.374726 ray_origin (-0.387017 -0.122478 1.44327) ray_direction (9.99094e-05 0.00107586 -0.00994146 ) p (-0.374536 0.0119243 0.201325) 
      pr 0.400964 ray_origin (-0.387017 -0.122478 1.44327) ray_direction (0.000192086 -0.000382195 -0.00999085 ) p (-0.363256 -0.169756 0.207408) 
      pr 0.40332 ray_origin (-0.387017 -0.122478 1.44327) ray_direction (0.000168336 -0.000349147 -0.00999248 ) p (-0.377102 -0.143044 0.854693) 
      pr 0.379205 ray_origin (-0.387017 -0.122478 1.44327) ray_direction (7.01609e-05 0.00101736 -0.00994787 ) p (-0.379121 -0.00798276 0.323727) 
      pr 0.405114 ray_origin (-0.387017 -0.122478 1.44327) ray_direction (5.87379e-05 -0.000149992 -0.0099987 ) p (-0.380129 -0.140069 0.27069) 
      pr 0.384358 ray_origin (-0.387017 -0.122478 1.44327) ray_direction (2.17256e-05 0.000901163 -0.00995929 ) p (-0.384266 -0.00837857 0.182288) 
      pr 0.405135 ray_origin (-0.387017 -0.122478 1.44327) ray_direction (4.5938e-05 -0.000116631 -0.00999921 ) p (-0.380952 -0.137877 0.123053) 
      pr 0.405809 ray_origin (-0.387017 -0.122478 1.44327) ray_direction (7.54046e-06 -1.65516e-05 -0.00999998 ) p (-0.386581 -0.123435 0.865469) 
      pr 0.394294 ray_origin (-0.387017 -0.122478 1.44327) ray_direction (-1.56711e-05 0.000785739 -0.00996907 ) p (-0.388068 -0.0697943 0.774843) 
      pr 0.395898 ray_origin (-0.387017 -0.122478 1.44327) ray_direction (-2.46897e-05 0.000751259 -0.00997171 ) p (-0.388546 -0.0759418 0.825579) 
      pr 0.405618 ray_origin (-0.387017 -0.122478 1.44327) ray_direction (-5.89059e-06 2.80596e-05 -0.00999996 ) p (-0.387701 -0.119222 0.28257) 
      pr 0.404043 ray_origin (-0.387017 -0.122478 1.44327) ray_direction (-5.06318e-05 0.000240747 -0.00999697 ) p (-0.393405 -0.092104 0.18198) 
      pr 0.405733 ray_origin (-0.387017 -0.122478 1.44327) ray_direction (-3.40233e-05 0.000139803 -0.00999896 ) p (-0.387764 -0.119408 1.22367) 
      pr 0.404764 ray_origin (-0.387017 -0.122478 1.44327) ray_direction (-4.99958e-05 0.000229477 -0.00999724 ) p (-0.390483 -0.106569 0.750161) 
      pr 0.405615 ray_origin (-0.387017 -0.122478 1.44327) ray_direction (-1.93219e-05 7.26696e-05 -0.00999972 ) p (-0.389331 -0.113776 0.245779) 
      pr 0.398044 ray_origin (-0.387017 -0.122478 1.44327) ray_direction (-5.75378e-05 0.000557059 -0.00998431 ) p (-0.393418 -0.0605073 0.33255) 
      pr 0.400453 ray_origin (-0.387017 -0.122478 1.44327) ray_direction (-6.27132e-05 0.000454983 -0.00998945 ) p (-0.394127 -0.0708931 0.310686) 
      pr 0.395965 ray_origin (-0.387017 -0.122478 1.44327) ray_direction (-4.98115e-05 0.000614046 -0.009981 ) p (-0.393086 -0.0476598 0.227135) 
      pr 0.341025 ray_origin (-0.369421 -0.120755 1.44802) ray_direction (0.000950677 -0.000878185 -0.0099159 ) p (-0.254925 -0.22652 0.253789) 
      pr 0.297176 ray_origin (-0.369421 -0.120755 1.44802) ray_direction (0.000741857 0.00162548 -0.00983908 ) p (-0.293737 0.0450754 0.44425) 
      pr 0.302086 ray_origin (-0.369421 -0.120755 1.44802) ray_direction (0.00065372 0.0015857 -0.00985181 ) p (-0.297097 0.0546797 0.358068) 
      pr 0.313391 ray_origin (-0.369421 -0.120755 1.44802) ray_direction (0.00050109 0.00149622 -0.00987473 ) p (-0.305333 0.0706063 0.185084) 
      pr 0.36803 ray_origin (-0.369421 -0.120755 1.44802) ray_direction (0.000516596 -0.000685543 -0.00996309 ) p (-0.302971 -0.208936 0.166472) 
      pr 0.371133 ray_origin (-0.369421 -0.120755 1.44802) ray_direction (0.000470472 -0.000652877 -0.00996757 ) p (-0.314889 -0.196429 0.292688) 
      pr 0.374728 ray_origin (-0.369421 -0.120755 1.44802) ray_direction (0.000401058 -0.000598279 -0.00997403 ) p (-0.322887 -0.190172 0.290764) 
      pr 0.335652 ray_origin (-0.369421 -0.120755 1.44802) ray_direction (0.000297844 0.00132391 -0.0099075 ) p (-0.333264 0.0399604 0.245312) 
      pr 0.385617 ray_origin (-0.369421 -0.120755 1.44802) ray_direction (0.000389401 -0.00058731 -0.00997514 ) p (-0.362749 -0.130817 1.27712) 
      pr 0.375353 ray_origin (-0.369421 -0.120755 1.44802) ray_direction (0.000377746 -0.000576341 -0.00997623 ) p (-0.320281 -0.195729 0.150241) 
      pr 0.382703 ray_origin (-0.369421 -0.120755 1.44802) ray_direction (0.000319488 -0.00052151 -0.00998128 ) p (-0.35335 -0.146988 0.945948) 
      pr 0.377792 ray_origin (-0.369421 -0.120755 1.44802) ray_direction (0.000331137 -0.000532475 -0.00998032 ) p (-0.32869 -0.186251 0.220418) 
      pr 0.360908 ray_origin (-0.369421 -0.120755 1.44802) ray_direction (0.000213968 0.00122764 -0.00992205 ) p (-0.357276 -0.0510741 0.884847) 
      pr 0.383643 ray_origin (-0.369421 -0.120755 1.44802) ray_direction (0.000249049 -0.000444502 -0.00998701 ) p (-0.353315 -0.1495 0.802179) 
      pr 0.384494 ray_origin (-0.369421 -0.120755 1.44802) ray_direction (0.000213554 -0.000400388 -0.0099897 ) p (-0.354796 -0.148174 0.763922) 
      pr 0.350341 ray_origin (-0.369421 -0.120755 1.44802) ray_direction (0.000162291 0.0011562 -0.00993161 ) p (-0.349838 0.0187556 0.249647) 
      pr 0.354039 ray_origin (-0.369421 -0.120755 1.44802) ray_direction (0.00013158 0.00110884 -0.00993746 ) p (-0.353896 0.010079 0.275484) 
      pr 0.383872 ray_origin (-0.369421 -0.120755 1.44802) ray_direction (0.000189698 -0.000367233 -0.00999145 ) p (-0.347005 -0.16415 0.267358) 
      pr 0.385967 ray_origin (-0.369421 -0.120755 1.44802) ray_direction (0.000165848 -0.000334082 -0.00999304 ) p (-0.358985 -0.141776 0.819228) 




    2017-08-08 13:48:18.679 INFO  [2427687] [Frame::key_pressed@695] Frame::key_pressed escape
    torus num_roots 4  t_cand        136  pr      0.411  ABCDE (      1e-08  -5.77e-06    0.00143     -0.173        8.3 )   neumark(   3.67e+04,   2.69e+08,  -1.09e+06 ) qsd     0.0937 
    torus num_roots 4  t_cand        123  pr      0.414  ABCDE (      1e-08  -5.76e-06    0.00143     -0.173        8.3 )   neumark(   3.68e+04,   2.74e+08,  -8.74e+05 ) qsd     0.1221 
    torus num_roots 4  t_cand        116  pr       0.41  ABCDE (      1e-08  -5.77e-06    0.00143     -0.173        8.3 )   neumark(   3.66e+04,   2.66e+08,  -6.96e+05 ) qsd     0.1536 
    torus num_roots 4  t_cand       45.3  pr      0.411  ABCDE (      1e-08  -5.77e-06    0.00143     -0.173        8.3 )   neumark(   3.65e+04,   2.63e+08,  -2.76e+05 ) qsd     0.6501 
    torus num_roots 4  t_cand        122  pr      0.414  ABCDE (      1e-08  -5.76e-06    0.00143     -0.173        8.3 )   neumark(   3.68e+04,   2.74e+08,  -8.36e+05 ) qsd     0.1281 
    torus num_roots 4  t_cand       76.5  pr      0.414  ABCDE (      1e-08  -5.76e-06    0.00143     -0.173        8.3 )   neumark(   3.68e+04,   2.74e+08,  -3.62e+05 ) qsd     0.3946 
    torus num_roots 4  t_cand       41.2  pr      0.413  ABCDE (      1e-08  -5.77e-06    0.00143     -0.173        8.3 )   neumark(   3.67e+04,   2.69e+08,  -5.48e+05 ) qsd     0.6851 
    torus num_roots 4  t_cand       73.5  pr      0.414  ABCDE (      1e-08  -5.76e-06    0.00143     -0.173        8.3 )   neumark(   3.69e+04,   2.74e+08,  -3.52e+05 ) qsd     0.4175 
    torus num_roots 4  t_cand        123  pr      0.407  ABCDE (      1e-08  -5.77e-06    0.00143     -0.173        8.3 )   neumark(   3.64e+04,   2.61e+08,  -7.94e+05 ) qsd     0.1304 
    torus num_roots 4  t_cand       29.1  pr      0.411  ABCDE (      1e-08  -5.77e-06    0.00143     -0.173        8.3 )   neumark(   3.63e+04,   2.59e+08,  -5.64e+05 ) qsd     0.7924 
    torus num_roots 4  t_cand       23.1  pr      0.414  ABCDE (      1e-08  -5.77e-06    0.00143     -0.173        8.3 )   neumark(   3.68e+04,   2.71e+08,  -6.34e+05 ) qsd     0.8441 
    torus num_roots 4  t_cand       53.8  pr      0.413  ABCDE (      1e-08  -5.77e-06    0.00143     -0.173        8.3 )   neumark(   3.68e+04,   2.72e+08,  -5.02e+05 ) qsd     0.5766 
    torus num_roots 4  t_cand       26.4  pr      0.414  ABCDE (      1e-08  -5.76e-06    0.00143     -0.173        8.3 )   neumark(   3.68e+04,   2.74e+08,  -2.65e+05 ) qsd     0.8145 
    torus num_roots 4  t_cand       49.7  pr      0.412  ABCDE (      1e-08  -5.77e-06    0.00143     -0.173        8.3 )   neumark(   3.67e+04,   2.69e+08,  -5.07e+05 ) qsd     0.6123 
    torus num_roots 4  t_cand       75.9  pr      0.413  ABCDE (      1e-08  -5.77e-06    0.00143     -0.173        8.3 )   neumark(   3.67e+04,    2.7e+08,  -3.56e+05 ) qsd     0.4006 
    torus num_roots 4  t_cand       46.4  pr      0.407  ABCDE (      1e-08  -5.78e-06    0.00143     -0.173        8.3 )   neumark(    3.6e+04,    2.5e+08,  -4.68e+05 ) qsd     0.6435 

    torus num_roots 4  t_cand        140  pr      0.243  ABCDE (      1e-08  -5.92e-06    0.00146     -0.174       8.53 )   neumark(   3.03e+04,   6.19e+07,  -1.67e+05 ) qsd     0.2612 
    torus num_roots 4  t_cand       35.6  pr      0.338  ABCDE (      1e-08  -5.84e-06    0.00145     -0.175       8.53 )   neumark(   3.44e+04,    1.8e+08,   -3.3e+05 ) qsd     0.7880 
    torus num_roots 4  t_cand        120  pr      0.335  ABCDE (      1e-08  -5.85e-06    0.00145     -0.175       8.53 )   neumark(   3.44e+04,   1.78e+08,  -4.33e+05 ) qsd     0.2132 
    torus num_roots 4  t_cand        127  pr      0.328  ABCDE (      1e-08  -5.85e-06    0.00145     -0.175       8.53 )   neumark(    3.4e+04,   1.67e+08,  -4.32e+05 ) qsd     0.1983 
    torus num_roots 4  t_cand        110  pr      0.326  ABCDE (      1e-08  -5.85e-06    0.00145     -0.175       8.53 )   neumark(   3.39e+04,   1.63e+08,  -3.45e+05 ) qsd     0.2635 
    torus num_roots 4  t_cand        111  pr      0.322  ABCDE (      1e-08  -5.86e-06    0.00146     -0.175       8.53 )   neumark(   3.37e+04,   1.57e+08,  -3.37e+05 ) qsd     0.2634 
    torus num_roots 4  t_cand        129  pr       0.31  ABCDE (      1e-08  -5.87e-06    0.00146     -0.175       8.53 )   neumark(   3.31e+04,   1.41e+08,  -3.47e+05 ) qsd     0.2131 
    torus num_roots 4  t_cand       28.8  pr      0.329  ABCDE (      1e-08  -5.87e-06    0.00146     -0.175       8.53 )   neumark(   3.32e+04,   1.42e+08,   -2.7e+05 ) qsd     0.8518 
    torus num_roots 4  t_cand        125  pr      0.308  ABCDE (      1e-08  -5.87e-06    0.00146     -0.175       8.53 )   neumark(    3.3e+04,   1.38e+08,  -3.27e+05 ) qsd     0.2255 
    torus num_roots 4  t_cand       41.9  pr      0.324  ABCDE (      1e-08  -5.87e-06    0.00146     -0.175       8.53 )   neumark(    3.3e+04,   1.37e+08,  -2.46e+05 ) qsd     0.7436 
    torus num_roots 4  t_cand       59.7  pr       0.31  ABCDE (      1e-08  -5.88e-06    0.00146     -0.175       8.53 )   neumark(   3.23e+04,   1.19e+08,  -1.93e+05 ) qsd     0.6088 
    torus num_roots 4  t_cand       63.8  pr        0.3  ABCDE (      1e-08  -5.89e-06    0.00146     -0.175       8.53 )   neumark(   3.18e+04,   1.03e+08,  -1.75e+05 ) qsd     0.5842 
    torus num_roots 4  t_cand       45.2  pr      0.307  ABCDE (      1e-08   -5.9e-06    0.00146     -0.175       8.53 )   neumark(   3.15e+04,   9.65e+07,  -1.83e+05 ) qsd     0.7270 
    torus num_roots 4  t_cand        113  pr      0.267  ABCDE (      1e-08   -5.9e-06    0.00146     -0.175       8.53 )   neumark(   3.11e+04,   8.31e+07,  -1.87e+05 ) qsd     0.3099 
    torus num_roots 4  t_cand        124  pr      0.322  ABCDE (      1e-08  -5.86e-06    0.00146     -0.175       8.53 )   neumark(   3.38e+04,    1.6e+08,  -3.92e+05 ) qsd     0.2137 
    torus num_roots 4  t_cand        122  pr      0.313  ABCDE (      1e-08  -5.87e-06    0.00146     -0.175       8.53 )   neumark(   3.33e+04,   1.46e+08,  -3.42e+05 ) qsd     0.2296 
    torus num_roots 4  t_cand       40.4  pr      0.335  ABCDE (      1e-08  -5.85e-06    0.00145     -0.175       8.53 )   neumark(   3.43e+04,   1.75e+08,  -3.15e+05 ) qsd     0.7492 




High residuals all with small cubic root h and cubic x^1 term f
-----------------------------------------------------------------

::

     ireal 4 root 12.1166 residual 6755.68  dis12 ( 0.59588 307.346 ) h 0.00213352  efg (10.8946 0.0238796 11.4462 )  
     ireal 4 root 7.80383 residual 621.823  dis12 ( 2.3157 79.3397 ) h 0.00205761  efg (10.9071 0.024825 11.4829 )  
     ireal 4 root -2.92037 residual 2008.1  dis12 ( 0.922162 159.775 ) h 0.000954811  efg (10.1145 0.0100976 9.20862 )  
     ireal 4 root -2.05183 residual 1192.57  dis12 ( 1.24188 118.883 ) h 0.00206139  efg (10.1248 0.0221511 9.22736 )  
     ireal 4 root 0.994075 residual 102.693  dis12 ( 8.28043 22.1929 ) h 0.000858502  efg (10.9043 0.0129158 11.4854 )  
     ireal 4 root -5.41463 residual 6749.1  dis12 ( 0.59588 307.346 ) h 0.00213352  efg (10.8946 0.0238796 11.4462 )  
     ireal 4 root -1.10345 residual 620.675  dis12 ( 2.3157 79.3397 ) h 0.00205761  efg (10.9071 0.024825 11.4829 )  
     ireal 4 root 0.541423 residual 159.452  dis12 ( 31.629 5.69974 ) h 0.000648497  efg (10.8331 0.0172809 11.2673 )  
     ireal 4 root 6.16539 residual 159.452  dis12 ( 31.629 5.69974 ) h 0.000648497  efg (10.8331 0.0172809 11.2673 )  
     ireal 4 root 10.6992 residual 3375.76  dis12 ( 0.676716 212.831 ) h 0.00158196  efg (10.0407 0.0164192 9.00159 )  
     ireal 4 root 6.34329 residual 169.717  dis12 ( 4.13438 34.4957 ) h 0.00110981  efg (10.0075 0.0134006 8.91364 )  
     ireal 4 root 6.7037 residual 259.875  dis12 ( 4.05056 44.9273 ) h 0.00167452  efg (10.8698 0.0215931 11.3738 )  
     ireal 4 root -3.88953 residual 3372.84  dis12 ( 0.676716 212.831 ) h 0.00158196  efg (10.0407 0.0164192 9.00159 )  
     ireal 4 root 0.469989 residual 169.461  dis12 ( 4.13438 34.4957 ) h 0.00110981  efg (10.0075 0.0134006 8.91364 )  
     ireal 4 root 0.00091958 residual 259.356  dis12 ( 4.05056 44.9273 ) h 0.00167452  efg (10.8698 0.0215931 11.3738 )  
     ireal 4 root 0.773045 residual 127.707  dis12 ( 26.6484 6.74415 ) h 0.000648682  efg (10.821 0.0156626 11.2325 )  
     ireal 4 root 5.93525 residual 127.707  dis12 ( 26.6484 6.74415 ) h 0.000648682  efg (10.821 0.0156626 11.2325 )  
     ireal 4 root 7.68331 residual 523.377  dis12 ( 1.91011 72.9907 ) h 0.00163207  efg (9.93475 0.0177729 8.71375 )  
     ireal 4 root 13.7465 residual 12828.2  dis12 ( 0.412294 431.697 ) h 0.00213402  efg (10.7869 0.0234595 11.124 )  
     ireal 4 root -0.860146 residual 522.578  dis12 ( 1.91011 72.9907 ) h 0.00163207  efg (9.93475 0.0177729 8.71375 )  
     ireal 4 root -7.0308 residual 12817.7  dis12 ( 0.412294 431.697 ) h 0.00213402  efg (10.7869 0.0234595 11.124 )  
     ireal 4 root 0.636714 residual 145.607  dis12 ( 29.6045 5.9973 ) h 0.000322252  efg (10.7732 0.00824175 11.0967 )  
     ireal 4 root 6.07771 residual 145.607  dis12 ( 29.6045 5.9973 ) h 0.000322252  efg (10.7732 0.00824175 11.0967 )  
     ireal 4 root 7.39851 residual 419.002  dis12 ( 2.18302 63.5538 ) h 0.00142414  efg (9.91843 0.0156797 8.6712 )  
     ireal 4 root 6.9682 residual 320.453  dis12 ( 3.37713 52.0685 ) h 0.00123896  efg (10.7395 0.0153979 10.9901 )  
     ireal 4 root -0.573558 residual 418.404  dis12 ( 2.18302 63.5538 ) h 0.00142414  efg (9.91843 0.0156797 8.6712 )  
     ireal 4 root -0.247646 residual 320.013  dis12 ( 3.37713 52.0685 ) h 0.00123896  efg (10.7395 0.0153979 10.9901 )  
     ireal 4 root 9.45734 residual 1789.2  dis12 ( 1.17221 148.568 ) h 0.00197987  efg (10.7058 0.0223566 10.8845 )  
     ireal 4 root 5.74827 residual 103.805  dis12 ( 7.59489 22.7292 ) h 0.00102934  efg (10.6712 0.0148932 10.7891 )  
     ireal 4 root 11.4815 residual 5060.8  dis12 ( 0.658718 263.604 ) h 0.00181227  efg (10.6942 0.0199778 10.8525 )  
     ireal 4 root -2.73149 residual 1786.88  dis12 ( 1.17221 148.568 ) h 0.00197987  efg (10.7058 0.0223566 10.8845 )  
     ireal 4 root 0.980752 residual 103.625  dis12 ( 7.59489 22.7292 ) h 0.00102934  efg (10.6712 0.0148932 10.7891 )  
     ireal 4 root -4.7544 residual 5056.28  dis12 ( 0.658718 263.604 ) h 0.00181227  efg (10.6942 0.0199778 10.8525 )  
     ireal 4 root 9.07339 residual 1347.82  dis12 ( 1.05991 127.968 ) h 0.00177263  efg (9.84652 0.0183937 8.47711 )  
     ireal 4 root 6.95694 residual 288.63  dis12 ( 2.6945 50.0953 ) h 0.00142475  efg (9.83047 0.0159255 8.43634 )  
     ireal 4 root -2.2389 residual 1346.13  dis12 ( 1.05991 127.968 ) h 0.00177263  efg (9.84652 0.0183937 8.47711 )  
     ireal 4 root -0.120862 residual 288.165  dis12 ( 2.6945 50.0953 ) h 0.00142475  efg (9.83047 0.0159255 8.43634 )  


Resolvent cubic constant term close to zero::

     ireal 4 root 4.03937 residual 133.735  dis12 ( 10.2215 119.382 ) h 0.000935691  pqr (39.8675 92.2881 -0.000549148 )  j g/j (-2.55538 -29.8455 )  
     ireal 4 root 4.71118 residual 205.839  dis12 ( 59.234 20.622 ) h 0.000489403  pqr (39.8896 92.4146 -0.000588339 )  j g/j (-14.8085 -5.1555 )  
     ireal 4 root 7.90485 residual 1562.48  dis12 ( 10.2215 119.382 ) h 0.000935691  pqr (39.8675 92.2881 -0.000549148 )  j g/j (-2.55538 -29.8455 )  
     ireal 4 root 0.170039 residual 205.661  dis12 ( 59.234 20.622 ) h 0.000489403  pqr (39.8896 92.4146 -0.000588339 )  j g/j (-14.8085 -5.1555 )  
     ireal 4 root -3.02135 residual 1561.41  dis12 ( 10.2215 119.382 ) h 0.000935691  pqr (39.8675 92.2881 -0.000549148 )  j g/j (-2.55538 -29.8455 )  
     ireal 4 root -0.685963 residual 369.603  dis12 ( 40.629 28.0141 ) h 0.000503756  pqr (38.4546 85.1429 -0.000396784 )  j g/j (-10.1573 -7.00354 )  
     ireal 4 root 5.68813 residual 369.603  dis12 ( 40.629 28.0141 ) h 0.000503756  pqr (38.4546 85.1429 -0.000396784 )  j g/j (-10.1573 -7.00354 )  
     ireal 4 root 5.14801 residual 254.942  dis12 ( 40.629 28.0141 ) h 0.000503756  pqr (38.4546 85.1429 -0.000396784 )  j g/j (-10.1573 -7.00354 )  
     ireal 4 root -0.144834 residual 254.748  dis12 ( 40.629 28.0141 ) h 0.000503756  pqr (38.4546 85.1429 -0.000396784 )  j g/j (-10.1573 -7.00354 )  
     ireal 4 root 16.8877 residual 47777.9  dis12 ( 1.46054 834.626 ) h 0.00109253  pqr (39.8453 92.1615 -0.000509134 )  j g/j (-0.365134 -208.657 )  
     ireal 4 root -12.0022 residual 47763.5  dis12 ( 1.46054 834.626 ) h 0.00109253  pqr (39.8453 92.1615 -0.000509134 )  j g/j (-0.365134 -208.657 )  
     ireal 4 root -0.455355 residual 315.678  dis12 ( 35.0303 32.3747 ) h 0.000756509  pqr (38.3899 84.9215 -0.000771257 )  j g/j (-8.75759 -8.09367 )  
     ireal 4 root 1.03084 residual 117.151  dis12 ( 8.70158 130.025 ) h 0.00109935  pqr (38.3419 84.6699 -0.000668665 )  j g/j (-2.17539 -32.5062 )  
     ireal 4 root 5.46329 residual 315.678  dis12 ( 35.0303 32.3747 ) h 0.000756509  pqr (38.3899 84.9215 -0.000771257 )  j g/j (-8.75759 -8.09367 )  
     ireal 4 root 3.98069 residual 117.151  dis12 ( 8.70158 130.025 ) h 0.00109935  pqr (38.3419 84.6699 -0.000668665 )  j g/j (-2.17539 -32.5062 )  
     ireal 4 root 5.34966 residual 291.901  dis12 ( 35.0303 32.3747 ) h 0.000756509  pqr (38.3899 84.9215 -0.000771257 )  j g/j (-8.75759 -8.09367 )  




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


