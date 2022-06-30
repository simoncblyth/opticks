maybe_replacing_G4Log_G4UniformRand_in_Absorption_and_Scattering_with_float_version_will_avoid_deviations
============================================================================================================

* from :doc:`higher_stats_U4RecorderTest_cxs_rainbow_random_aligned_comparison`

u4/tests/U4LogTest.cc::

     09 int main(int argc, char** argv)
     10 {
     11     unsigned ni = SSys::getenvunsigned("U4LogTest_ni", 1001) ;
     12     unsigned nj = 5 ;
     13 
     14     NP* a = NP::Make<double>(ni, nj );
     15     double* aa = a->values<double>();
     16 
     17     for(unsigned i=0 ; i < ni ; i++)
     18     { 
     19         double d =  double(i)/double(ni-1) ;
     20         float  f = float(d) ;
     21 
     22         double d0 = -1.*std::log( d );
     23         float  f0 = -1.f*std::log( f );
     24    
     25         double d4 = -1.*G4Log( d ) ;
     26         float  f4 = -1.f*G4Logf( f ) ;
     27         
     28         aa[nj*i+0] = d ; 
     29         aa[nj*i+1] = d0 ; 
     30         aa[nj*i+2] = f0 ; 
     31         aa[nj*i+3] = d4 ;
     32         aa[nj*i+4] = f4 ; 
     33     }
     34 
     35     a->save(FOLD, "a.npy") ; 
     36     return 0 ;  
     37 }


::

    In [19]: a[:10]
    Out[19]: 
    array([[  0.   ,     inf,     inf, 709.09 ,  88.03 ],
           [  0.001,   6.908,   6.908,   6.908,   6.908],
           [  0.002,   6.215,   6.215,   6.215,   6.215],
           [  0.003,   5.809,   5.809,   5.809,   5.809],
           [  0.004,   5.521,   5.521,   5.521,   5.521],
           [  0.005,   5.298,   5.298,   5.298,   5.298],
           [  0.006,   5.116,   5.116,   5.116,   5.116],
           [  0.007,   4.962,   4.962,   4.962,   4.962],
           [  0.008,   4.828,   4.828,   4.828,   4.828],
           [  0.009,   4.711,   4.711,   4.711,   4.711]])

::

    In [20]: U,D0,F0,D4,F4 = range(5)


    In [23]: np.abs(a[1:,D0]-a[1:,D4]).max()   ## std::log(double) vs G4Log
    Out[23]: 8.881784197001252e-16

    In [24]: np.abs(a[1:,F0]-a[1:,F4]).max()   ## std::log(float) vs G4Logf
    Out[24]: 2.384185791015625e-07

    In [26]: np.abs(a[1:,D0]-a[1:,F0]).max()   ## std::log(double) vs std::log(float)
    Out[26]: 2.1071941791461768e-07

    In [27]: np.abs(a[1:,D4]-a[1:,F4]).max()   ## G4Log vs G4Logf
    Out[27]: 2.1071941791461768e-07


* float/double differences at 1e-7 level 
* BUT absorption and scattering lengths are long so positions can be deviated > 0.1 mm ? 


sysrap/tests/logTest.cu::

     01 // ./logTest.sh
      2 
      3 #include "NP.hh"
      4 
      5 __global__ void test_log_(double* dd)
      6 {
      7     unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;
      8     unsigned nx = blockDim.x ;
      9 
     10     double d = double(ix)/double(nx-1) ;
     11     float  f = float(d) ;
     12 
     13     double d0 = -1.*log( d );
     14     float  f0 = -1.f*log( f );
     15 
     16     dd[ix*4+0] = d ;
     17     dd[ix*4+1] = d0 ;
     18     dd[ix*4+2] = f0 ;
     19     dd[ix*4+3] = 0. ;
     20 
     21     //printf("//test_log  (ix,iy,nx) (%2d, %2d, %2d) \n", ix, iy, nx );
     22 }
     23 
     24 void test_log()
     25 {
     26     unsigned ni = 1001 ;
     27     unsigned nj = 4 ;
     28 
     29     dim3 block(ni,1);
     30     dim3 grid(1,1);
     31 
     32     NP* h = NP::Make<double>( ni, nj ) ;
     33     unsigned arr_bytes = h->arr_bytes() ;
     34     double* hh = h->values<double>();
     35 
     36     double* dd = nullptr ;
     37     cudaMalloc(reinterpret_cast<void**>( &dd ), arr_bytes );
     38 
     39     test_log_<<<grid,block>>>(dd);
     40 
     41     cudaMemcpy( hh, dd, arr_bytes, cudaMemcpyDeviceToHost ) ;
     42     cudaDeviceSynchronize();
     43 
     44     h->save("/tmp/logTest.npy");
     45 }
     46 
     47 int main()
     48 {
     49     test_log();
     50     return 0 ;
     51 }


Comparing log values in float and double
---------------------------------------------

::

    epsilon:tests blyth$ ./U4LogTest.sh ana
    a (1001, 4) a_path /tmp/logTest.npy 
    b (1001, 5) b_path /tmp/blyth/opticks/U4LogTest/a.npy 
    [[ 0.       inf    inf  0.   ]
     [ 0.001  6.908  6.908  0.   ]
     [ 0.002  6.215  6.215  0.   ]
     ...
     [ 0.998  0.002  0.002  0.   ]
     [ 0.999  0.001  0.001  0.   ]
     [ 1.    -0.    -0.     0.   ]]
    [[  0.        inf     inf 709.09   88.03 ]
     [  0.001   6.908   6.908   6.908   6.908]
     [  0.002   6.215   6.215   6.215   6.215]
     ...
     [  0.998   0.002   0.002   0.002   0.002]
     [  0.999   0.001   0.001   0.001   0.001]
     [  1.     -0.     -0.     -0.     -0.   ]]

    In [1]:                                                               

::

    In [7]: np.abs(a[1:,D0]-b[1:,D0]).max()   ## compares CUDA and CPU log(double)  
    Out[7]: 2.220446049250313e-16

    In [8]: np.abs(a[1:,F0]-b[1:,F0]).max()   ## compares CUDA and CPU log(float)
    Out[8]: 2.384185791015625e-07



Change the AB and SC shims to allow reducing the precision of the log(u)
----------------------------------------------------------------------------

::

    u4/ShimG4OpAbsorption.h
    u4/ShimG4OpRayleigh.h


HMM original has drop out zeros::

    u4t
    ./U4RecorderTest_ab.sh 


    In [1]: XFold.BaseSymbol(a)
    Out[1]: 'B'

    In [2]: XFold.BaseSymbol(b)
    Out[2]: 'B'

    In [3]: at = stag.Unpack(a.tag)

    In [4]: bt = stag.Unpack(b.tag)


    In [7]: at[0]
    Out[7]: array([3, 4, 5, 6, 7, 8, 3, 4, 5, 6, 7, 8, 3, 4, 5, 6, 7, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=uint8)

    In [8]: bt[0]
    Out[8]: array([3, 4, 0, 0, 7, 8, 3, 4, 0, 0, 7, 8, 3, 4, 0, 0, 7, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=uint8)

    In [9]: b.base
    Out[9]: '/tmp/blyth/opticks/U4RecorderTest/ShimG4OpAbsorption_ORIGINAL_ShimG4OpRayleigh_ORIGINAL'

    In [10]: a.base
    Out[10]: '/tmp/blyth/opticks/U4RecorderTest/ShimG4OpAbsorption_FLOAT_ShimG4OpRayleigh_FLOAT'

Fixed by rerun::

    In [5]: np.all( at == bt )
    Out[5]: True




    In [9]: w = np.unique(np.where( np.abs(a.photon - b.photon) > 1e-6 )[0]) ; w
    Out[9]: array([ 75, 230, 387, 549])

    In [10]: seqhis_(a.seq[w,0])
    Out[10]: ['TO BT AB', 'TO BT AB', 'TO SC SA', 'TO BT AB']



10k check::


    ./U4RecorderTest_ab.sh ## u4t 
    w = np.unique(np.where( np.abs(a.photon - b.photon) > 0.1 )[0]) : [5156 5208 7203 8393 9964]
    s = a.seq[w,0]                                     : [ 35693  19661   2157  19661 575181]
    o = cuss(s,w)                                      : 
    [['w0' '                   TO BT BT AB' '           19661' '               2']
     ['w1' '                TO BT SC BT SA' '          575181' '               1']
     ['w2' '                   TO SC BR SA' '           35693' '               1']
     ['w3' '                      TO SC SA' '            2157' '               1']]
    w1                                                 : [9964]
    abw0 = a.photon[w0,:4] - b.photon[w0,:4]           : 
    [[[ 0.156 -0.051 -0.417 -0.001]
      [-0.     0.    -0.     0.   ]
      [ 0.    -0.     0.     0.   ]
      [ 0.     0.    -0.     0.   ]]

     [[-0.181  0.099 -0.425 -0.002]
      [-0.     0.     0.     0.   ]
      [-0.     0.     0.     0.   ]
      [ 0.     0.    -0.     0.   ]]]
    a.base                                             : /tmp/blyth/opticks/GeoChain/BoxedSphere/CXRaindropTest
    b.base                                             : /tmp/blyth/opticks/U4RecorderTest/ShimG4OpAbsorption_FLOAT_ShimG4OpRayleigh_FLOAT



    ./U4RecorderTest_ab.sh ## u4t 
    w = np.unique(np.where( np.abs(a.photon - b.photon) > 0.1 )[0]) : [5156 5208 7203 8393 9964]
    s = a.seq[w,0]                                     : [ 35693  19661   2157  19661 575181]
    o = cuss(s,w)                                      : 
    [['w0' '                   TO BT BT AB' '           19661' '               2']
     ['w1' '                TO BT SC BT SA' '          575181' '               1']
     ['w2' '                   TO SC BR SA' '           35693' '               1']
     ['w3' '                      TO SC SA' '            2157' '               1']]
    w1                                                 : [9964]
    abw0 = a.photon[w0,:4] - b.photon[w0,:4]           : 
    [[[ 0.156 -0.051 -0.417 -0.001]
      [-0.     0.    -0.     0.   ]
      [ 0.    -0.     0.     0.   ]
      [ 0.     0.    -0.     0.   ]]

     [[-0.181  0.099 -0.425 -0.002]
      [-0.     0.     0.     0.   ]
      [-0.     0.     0.     0.   ]
      [ 0.     0.    -0.     0.   ]]]
    a.base                                             : /tmp/blyth/opticks/GeoChain/BoxedSphere/CXRaindropTest
    b.base                                             : /tmp/blyth/opticks/U4RecorderTest/ShimG4OpAbsorption_ORIGINAL_ShimG4OpRayleigh_ORIGINAL



HMM looks like no difference from the log(float) ? is it being applied ?


Need to access the distance from both contexts::

     453 inline QSIM_METHOD int qsim::propagate_to_boundary(unsigned& flag, curandStateXORWOW& rng, sctx& ctx)
     454 {
     455     sphoton& p = ctx.p ;
     456     const sstate& s = ctx.s ;
     457 
     458     const float& absorption_length = s.material1.y ;
     459     const float& scattering_length = s.material1.z ;
     ...
     469     float u_scattering = curand_uniform(&rng) ;
     470     float u_absorption = curand_uniform(&rng) ;
     471 
     480     float scattering_distance = -scattering_length*logf(u_scattering);
     481     float absorption_distance = -absorption_length*logf(u_absorption);


::

    071 G4double G4VDiscreteProcess::PostStepGetPhysicalInteractionLength(
     72                              const G4Track& track,
     73                  G4double   previousStepSize,
     74                  G4ForceCondition* condition
     75                 )
     76 {
     77   if ( (previousStepSize < 0.0) || (theNumberOfInteractionLengthLeft<=0.0)) {
     78     // beggining of tracking (or just after DoIt of this process)
     79     ResetNumberOfInteractionLengthLeft();
     80   } else if ( previousStepSize > 0.0) {
     81     // subtract NumberOfInteractionLengthLeft 
     82     SubtractNumberOfInteractionLengthLeft(previousStepSize);
     83   } else {
     84     // zero step
     85     //  DO NOTHING
     86   }
     87 
     88   // condition is set to "Not Forced"
     89   *condition = NotForced;
     90 
     91   // get mean free path
     92   currentInteractionLength = GetMeanFreePath(track, previousStepSize, condition);
     93 
     94   G4double value;
     95   if (currentInteractionLength <DBL_MAX) {
     96     value = theNumberOfInteractionLengthLeft * currentInteractionLength;
     97   } else {
     98     value = DBL_MAX;
     99   }
    100 #ifdef G4VERBOSE
    101   if (verboseLevel>1){
    102     G4cout << "G4VDiscreteProcess::PostStepGetPhysicalInteractionLength ";
    103     G4cout << "[ " << GetProcessName() << "]" <<G4endl;
    104     track.GetDynamicParticle()->DumpInfo();
    105     G4cout << " in Material  " <<  track.GetMaterial()->GetName() <<G4endl;
    106     G4cout << "InteractionLength= " << value/cm <<"[cm] " <<G4endl;
    107   }
    108 #endif
    109   return value;
    110 }




Check deviation distances::

    In [4]: ar = a.record[w0[0],:4]      

    In [18]: br = b.record[w0[0],:4]

    In [23]: ar23 = ar[3,0,:3] - ar[2,0,:3]  ; ar23
    Out[23]: array([-318.174,  105.244,  850.909], dtype=float32)

    In [24]: br23 = br[3,0,:3] - br[2,0,:3] ; br23 
    Out[24]: array([-318.329,  105.295,  851.326], dtype=float32)

    In [25]: np.sqrt( np.sum(ar23*ar23))
    Out[25]: 914.525

    In [26]: np.sqrt( np.sum(br23*br23))
    Out[26]: 914.9732

    In [29]: seqhis_(a.seq[w0[0],0])   # AB in air 
    Out[29]: 'TO BT BT AB'


::

    In [30]: A(w0[0])
    Out[30]: 
    A(5208) : TO BT BT AB
           A.t : (10000, 48) 
           A.n : (10000,) 
          A.ts : (10000, 10, 29) 
          A.fs : (10000, 10, 29) 
         A.ts2 : (10000, 10, 29) 
     0 :     0.3262 :  1 :     to_sci : qsim::propagate_to_boundary u_to_sci burn 
     1 :     0.2852 :  2 :     to_bnd : qsim::propagate_to_boundary u_to_bnd burn 
     2 :     0.3563 :  3 :     to_sca : qsim::propagate_to_boundary u_scattering 
     3 :     0.2718 :  4 :     to_abs : qsim::propagate_to_boundary u_absorption 
     4 :     0.6653 :  5 : at_burn_sf_sd : at_boundary_burn at_surface ab/sd  
     5 :     0.1049 :  6 :     at_ref : u_reflect > TransCoeff 

     6 :     0.3963 :  1 :     to_sci : qsim::propagate_to_boundary u_to_sci burn 
     7 :     0.0073 :  2 :     to_bnd : qsim::propagate_to_boundary u_to_bnd burn 
     8 :     0.7812 :  3 :     to_sca : qsim::propagate_to_boundary u_scattering 
     9 :     0.0899 :  4 :     to_abs : qsim::propagate_to_boundary u_absorption 
    10 :     0.4851 :  5 : at_burn_sf_sd : at_boundary_burn at_surface ab/sd  
    11 :     0.2859 :  6 :     at_ref : u_reflect > TransCoeff 

    12 :     0.0153 :  1 :     to_sci : qsim::propagate_to_boundary u_to_sci burn 
    13 :     0.7635 :  2 :     to_bnd : qsim::propagate_to_boundary u_to_bnd burn 
    14 :     0.5736 :  3 :     to_sca : qsim::propagate_to_boundary u_scattering 
    15 :     0.9999 :  4 :     to_abs : qsim::propagate_to_boundary u_absorption 
    16 :     0.0000 :  0 :      undef : undef 
    17 :     0.0000 :  0 :      undef : undef 


* AHHA u_scattering close to 1.  

::

    In [32]: a.flat[w0[0],15]
    Out[32]: 0.9999085

    In [33]: b.flat[w0[0],15]
    Out[33]: 0.9999085

    In [35]: -1e7*np.log(a.flat[w0[0],15])
    Out[35]: 914.9731340585276

::

    epsilon:tests blyth$ U=0.9999085 U4LogTest 
     u   0.9999085
             d0  0.0000915      d0*sc 915.0418638 f(d0)*f(sc) 915.0418701
             f0  0.0000915      f0*sc 914.9731341 f(f0)*f(sc) 914.9731445
             d4  0.0000915      d4*sc 915.0418638 f(d4)*f(sc) 915.0418701
             f4  0.0000915      f4*sc 914.9731341 f(f4)*f(sc) 914.9731445
    epsilon:tests blyth$ 


When U is close to 1 its easy to get float/double differences, 
as are using float to hold a very small number "-log(u)" 
which are multiplying by a very big number (eg air absorption length 1e7 mm)
So float imprecision gets scaled up::

    epsilon:tests blyth$ U=0.999908506 U4LogTest 
     u   0.9999085
             d0  0.0000915      d0*sc 914.9818583 f(d0)*f(sc) 914.9818726
             f0  0.0000915      f0*sc 914.9731341 f(f0)*f(sc) 914.9731445
             d4  0.0000915      d4*sc 914.9818583 f(d4)*f(sc) 914.9818726
             f4  0.0000915      f4*sc 914.9731341 f(f4)*f(sc) 914.9731445
    epsilon:tests blyth$ U=0.9999085 U4LogTest 
     u   0.9999085
             d0  0.0000915      d0*sc 915.0418638 f(d0)*f(sc) 915.0418701
             f0  0.0000915      f0*sc 914.9731341 f(f0)*f(sc) 914.9731445
             d4  0.0000915      d4*sc 915.0418638 f(d4)*f(sc) 915.0418701
             f4  0.0000915      f4*sc 914.9731341 f(f4)*f(sc) 914.9731445

    In [45]:  u = np.float64(0.9999085)
    In [46]: -1e7*np.log(u)
    Out[46]: 915.0418638039326

    In [48]: -np.float32(1e7)*np.log(np.float32(u))
    Out[48]: 914.97314

    In [49]: -np.float32(1e7)*np.float32(np.log(np.float32(u)))
    Out[49]: 914.97314


    epsilon:tests blyth$ U=0.99999 U4LogTest 
     u   0.9999900
             d0  0.0000100      d0*sc 100.0005000 f(d0)*f(sc) 100.0004959
             f0  0.0000100      f0*sc 100.1363034 f(f0)*f(sc) 100.1363068
             d4  0.0000100      d4*sc 100.0005000 f(d4)*f(sc) 100.0004959
             f4  0.0000100      f4*sc 100.1363034 f(f4)*f(sc) 100.1363068
    epsilon:tests blyth$ U=0.9999 U4LogTest 
     u   0.9999000
             d0  0.0001000      d0*sc 1000.0500033 f(d0)*f(sc) 1000.0500488
             f0  0.0001000      f0*sc 1000.2159252 f(f0)*f(sc) 1000.2159424
             d4  0.0001000      d4*sc 1000.0500033 f(d4)*f(sc) 1000.0500488
             f4  0.0001000      f4*sc 1000.2159252 f(f4)*f(sc) 1000.2159424
    epsilon:tests blyth$ U=0.99999 U4LogTest 
     u   0.9999900
             d0  0.0000100      d0*sc 100.0005000 f(d0)*f(sc) 100.0004959
             f0  0.0000100      f0*sc 100.1363034 f(f0)*f(sc) 100.1363068
             d4  0.0000100      d4*sc 100.0005000 f(d4)*f(sc) 100.0004959
             f4  0.0000100      f4*sc 100.1363034 f(f4)*f(sc) 100.1363068
    epsilon:tests blyth$ U=0.999999 U4LogTest 
     u   0.9999990
             d0  0.0000010      d0*sc 10.0000050 f(d0)*f(sc) 10.0000048
             f0  0.0000010      f0*sc 10.1327953 f(f0)*f(sc) 10.1327953
             d4  0.0000010      d4*sc 10.0000050 f(d4)*f(sc) 10.0000048
             f4  0.0000010      f4*sc 10.1327953 f(f4)*f(sc) 10.1327953
    epsilon:tests blyth$ 



::

    In [39]: "%11.20f " % a.flat[w0[0],15]
    Out[39]: '0.99990850687026977539 '

    epsilon:tests blyth$ U=0.99990850687026977539 U4LogTest 
     u   0.9999085
             d0  0.0000915      d0*sc 914.9731548 f(d0)*f(sc) 914.9731445
             f0  0.0000915      f0*sc 914.9731341 f(f0)*f(sc) 914.9731445
             d4  0.0000915      d4*sc 914.9731548 f(d4)*f(sc) 914.9731445
             f4  0.0000915      f4*sc 914.9731341 f(f4)*f(sc) 914.9731445
    epsilon:tests blyth$ 



-ln(1-x) is very close to x for small x::

    In [9]: -np.log(1-1e-3)
    Out[9]: 0.0010005003335835344

    In [10]: -np.log(1-1e-4)
    Out[10]: 0.00010000500033334732

    In [11]: -np.log(1-1e-5)
    Out[11]: 1.0000050000287824e-05

    In [12]: -np.log(1-1e-6)
    Out[12]: 1.000000500029089e-06

    In [13]: -np.log(1-1e-7)
    Out[13]: 1.0000000494736474e-07

    In [14]: -np.log(1-1e-8)
    Out[14]: 1.0000000100247594e-08

    In [15]: -np.log(1-1e-9)
    Out[15]: 9.999999722180686e-10


::


    In [81]: uu = 1-np.logspace(-10, 0,11) ; uu
    Out[81]: array([1.   , 1.   , 1.   , 1.   , 1.   , 1.   , 1.   , 0.999, 0.99 , 0.9  , 0.   ])

    In [84]: np.logspace(-10, -1,10)
    Out[84]: array([1.e-10, 1.e-09, 1.e-08, 1.e-07, 1.e-06, 1.e-05, 1.e-04, 1.e-03, 1.e-02, 1.e-01])

    In [89]: np.set_printoptions(suppress=False, precision=12 )

    In [90]: uu = 1.-np.logspace(-10, -1,10) ; uu
    Out[90]: array([0.9999999999, 0.999999999 , 0.99999999  , 0.9999999   , 0.999999    , 0.99999     , 0.9999      , 0.999       , 0.99        , 0.9         ])

In [92]: -np.log(uu)
Out[92]: 
array([1.000000082790e-10, 9.999999722181e-10, 1.000000010025e-08, 1.000000049474e-07, 1.000000500029e-06, 1.000005000029e-05, 1.000050003333e-04, 1.000500333584e-03, 1.005033585350e-02,
       1.053605156578e-01])



Select scatterers::

    In [5]: sc = np.where( a.photon[:,3,3].view(np.int32) & ( 0x1 << 5 )  ) [0] ; sc 
    Out[5]: array([ 387, 1091, 1292, 1338, 1701, 1859, 2537, 3276, 3846, 4203, 4573, 5156, 5687, 6555, 7203, 7604, 7737, 7791, 8235, 8587, 9654, 9964])

    In [7]: seqhis_(a.seq[sc,0])
    Out[7]: 
    ['TO SC SA',
     'TO BT SC BT SA',
     'TO BT BT SC SA',
     'TO SC SA',
     'TO SC SA',
     'TO SC SA',
     'TO SC SA',
     'TO BR SC SA',
     'TO BT BT SC SA',
     'TO BT BR BT SC SA',
     'TO BT BR BT SC SA',
     'TO SC BR SA',
     'TO BT BT SC SA',
     'TO BT BT SC SA',
     'TO SC SA',
     'TO BT BT SC SA',
     'TO SC SA',
     'TO SC SA',
     'TO SC SA',
     'TO BT BT SC SA',
     'TO BT BT SC SA',
     'TO BT SC BT SA']

    In [9]: ab = np.where( a.photon[:,3,3].view(np.int32) & ( 0x1 << 3 )  ) [0]   ; ab
    Out[9]: 
    array([  75,  201,  230,  549, 1156, 1475, 1483, 1616, 2263, 2413, 2515, 2671, 2761, 3029, 3631, 3771, 3778, 4057, 4097, 4442, 4468, 4524, 4777, 5208, 5729, 6797, 6907, 6925, 7258, 7554, 7690, 7769,
           8393])


Select absorb::

    In [9]: ab = np.where( a.photon[:,3,3].view(np.int32) & ( 0x1 << 3 )  ) [0]   ; a
    Out[9]: 
    array([  75,  201,  230,  549, 1156, 1475, 1483, 1616, 2263, 2413, 2515, 2671, 2761, 3029, 3631, 3771, 3778, 4057, 4097, 4442, 4468, 4524, 4777, 5208, 5729, 6797, 6907, 6925, 7258, 7554, 7690, 7769,
           8393])

    In [10]: seqhis_(a.seq[ab,0])
    Out[10]: 
    ['TO BT AB',
     'TO BT AB',
     'TO BT AB',
     'TO BT AB',
     'TO BT AB',
     'TO BT AB',
     'TO BT AB',
     'TO BT AB',
     'TO BT AB',
     'TO BT AB',
     'TO BT AB',
     'TO BT AB',
     'TO BT AB',
     'TO BT AB',
     'TO BT AB',
     'TO BT BR BR BR BR AB',
     'TO BT AB',
     'TO BT AB',
     'TO BT AB',
     'TO BT AB',
     'TO BT AB',
     'TO BT AB',
     'TO BT AB',
     'TO BT BT AB',
     'TO BT AB',
     'TO BT AB',
     'TO BT BR AB',
     'TO BT AB',
     'TO BT AB',
     'TO BT AB',
     'TO BT AB',
     'TO BT BR AB',
     'TO BT BT AB']

    In [14]: np.set_printoptions(formatter={'int':hex})

    In [15]: a.seq[ab,0]
    Out[15]: 
    array([0x4cd, 0x4cd, 0x4cd, 0x4cd, 0x4cd, 0x4cd, 0x4cd, 0x4cd, 0x4cd, 0x4cd, 0x4cd, 0x4cd, 0x4cd, 0x4cd, 0x4cd, 0x4bbbbcd, 0x4cd, 0x4cd, 0x4cd, 0x4cd, 0x4cd, 0x4cd, 0x4cd, 0x4ccd, 0x4cd, 0x4cd,
           0x4bcd, 0x4cd, 0x4cd, 0x4cd, 0x4cd, 0x4bcd, 0x4ccd], dtype=uint64)


::

    In [23]: ab3 = np.where( (a.seq[:,0] >> 8 ) == 0x4)[0] ; ab3
    Out[23]: array([  75,  201,  230,  549, 1156, 1475, 1483, 1616, 2263, 2413, 2515, 2671, 2761, 3029, 3631, 3778, 4057, 4097, 4442, 4468, 4524, 4777, 5729, 6797, 6925, 7258, 7554, 7690])

    In [24]: seqhis_(a.seq[ab3,0])
    Out[24]: 
    ['TO BT AB',
     'TO BT AB',
     'TO BT AB',
     'TO BT AB',
     'TO BT AB',
     'TO BT AB',
     'TO BT AB',
     'TO BT AB',
     'TO BT AB',
     'TO BT AB',
     'TO BT AB',
     'TO BT AB',
     'TO BT AB',
     'TO BT AB',
     'TO BT AB',
     'TO BT AB',
     'TO BT AB',
     'TO BT AB',
     'TO BT AB',
     'TO BT AB',
     'TO BT AB',
     'TO BT AB',
     'TO BT AB',
     'TO BT AB',
     'TO BT AB',
     'TO BT AB',
     'TO BT AB',
     'TO BT AB']

Finding the flat index of u_absorption::

    In [11]: A(ab[0])
    Out[11]: 
    A(75) : TO BT AB
           A.t : (10000, 48) 
           A.n : (10000,) 
          A.ts : (10000, 10, 29) 
          A.fs : (10000, 10, 29) 
         A.ts2 : (10000, 10, 29) 
     0 :     0.3727 :  1 :     to_sci : qsim::propagate_to_boundary u_to_sci burn 
     1 :     0.8539 :  2 :     to_bnd : qsim::propagate_to_boundary u_to_bnd burn 
     2 :     0.0380 :  3 :     to_sca : qsim::propagate_to_boundary u_scattering 
     3 :     0.2685 :  4 :     to_abs : qsim::propagate_to_boundary u_absorption 
     4 :     0.9740 :  5 : at_burn_sf_sd : at_boundary_burn at_surface ab/sd  
     5 :     0.5896 :  6 :     at_ref : u_reflect > TransCoeff 

     6 :     0.2975 :  1 :     to_sci : qsim::propagate_to_boundary u_to_sci burn 
     7 :     0.2261 :  2 :     to_bnd : qsim::propagate_to_boundary u_to_bnd burn 
     8 :     0.9222 :  3 :     to_sca : qsim::propagate_to_boundary u_scattering 
     9 :     0.9992 :  4 :     to_abs : qsim::propagate_to_boundary u_absorption 
    10 :     0.0000 :  0 :      undef : undef 
    11 :     0.0000 :  0 :      undef : undef 


The randoms that lead to absorption in the water sphere::

    In [26]: np.set_printoptions(precision=10)
    In [27]: a.flat[ab3,9]
    Out[27]: 
    array([0.9992083 , 0.9979085 , 0.9988845 , 0.99773157, 0.9987508 , 0.9985411 , 0.99959373, 0.9990112 , 0.9981463 , 0.9999338 , 0.999499  , 0.9993434 , 0.99755085, 0.9990241 , 0.99945176, 0.9992132 ,
           0.9984617 , 0.99773735, 0.99861896, 0.99947524, 0.9983674 , 0.9996877 , 0.99987745, 0.9984901 , 0.99854493, 0.9999999 , 0.9993231 , 0.9995929 ], dtype=float32)

    In [28]:                           


    In [28]: ab4 = np.where( (a.seq[:,0] >> 12) == 0x4 )[0] ; ab4
    Out[28]: array([5208, 6907, 7769, 8393])

    In [29]: seqhis_(a.seq[ab4,0])
    Out[29]: ['TO BT BT AB', 'TO BT BR AB', 'TO BT BR AB', 'TO BT BT AB']

Point 4 randoms, these two in water sphere::

    In [35]: a.flat[ab4[1],19]
    Out[35]: 0.99828494

    In [36]: a.flat[ab4[2],19]
    Out[36]: 0.99772733

These two in air on other side::

    In [37]: a.flat[ab4[0],15]
    Out[37]: 0.9999085

    In [38]: a.flat[ab4[3],15]
    Out[38]: 0.99999297


Find the scatter randoms::

    In [41]: np.set_printoptions(formatter={'int':hex})

    In [42]: a.seq[sc,0]
    Out[42]: 
    array([0x86d, 0x8c6cd, 0x86ccd, 0x86d, 0x86d, 0x86d, 0x86d, 0x86bd, 0x86ccd, 0x86cbcd, 0x86cbcd, 0x8b6d, 0x86ccd, 0x86ccd, 0x86d, 0x86ccd, 0x86d, 0x86d, 0x86d, 0x86ccd, 0x86ccd, 0x8c6cd],
          dtype=uint64)

    In [43]: sc1 = np.where( a.seq[:,0] == 0x86d )[0] ; sc1
    Out[43]: array([0x183, 0x53a, 0x6a5, 0x743, 0x9e9, 0x1c23, 0x1e39, 0x1e6f, 0x202b])

    In [44]: np.set_printoptions(formatter={'int':None})  

    In [45]: sc1 = np.where( a.seq[:,0] == 0x86d )[0] ; sc1
    Out[45]: array([ 387, 1338, 1701, 1859, 2537, 7203, 7737, 7791, 8235])

    In [46]: seqhis_(a.seq[sc1,0])
    Out[46]: 
    ['TO SC SA',
     'TO SC SA',
     'TO SC SA',
     'TO SC SA',
     'TO SC SA',
     'TO SC SA',
     'TO SC SA',
     'TO SC SA',
     'TO SC SA']

    In [47]:                         


    In [48]: A(sc1[1])
    Out[48]: 
    A(1338) : TO SC SA
           A.t : (10000, 48) 
           A.n : (10000,) 
          A.ts : (10000, 10, 29) 
          A.fs : (10000, 10, 29) 
         A.ts2 : (10000, 10, 29) 
     0 :     0.6689 :  1 :     to_sci : qsim::propagate_to_boundary u_to_sci burn 
     1 :     0.8334 :  2 :     to_bnd : qsim::propagate_to_boundary u_to_bnd burn 
     2 :     0.9997 :  3 :     to_sca : qsim::propagate_to_boundary u_scattering 
     3 :     0.0396 :  4 :     to_abs : qsim::propagate_to_boundary u_absorption 
     4 :     0.6777 :  8 :         sc : qsim::rayleigh_scatter 
     5 :     0.4505 :  8 :         sc : qsim::rayleigh_scatter 
     6 :     0.4775 :  8 :         sc : qsim::rayleigh_scatter 
     7 :     0.7707 :  8 :         sc : qsim::rayleigh_scatter 
     8 :     0.1893 :  8 :         sc : qsim::rayleigh_scatter 

     9 :     0.8990 :  1 :     to_sci : qsim::propagate_to_boundary u_to_sci burn 
    10 :     0.0684 :  2 :     to_bnd : qsim::propagate_to_boundary u_to_bnd burn 
    11 :     0.5886 :  3 :     to_sca : qsim::propagate_to_boundary u_scattering 
    12 :     0.2529 :  4 :     to_abs : qsim::propagate_to_boundary u_absorption 
    13 :     0.4337 :  5 : at_burn_sf_sd : at_boundary_burn at_surface ab/sd  
    14 :     0.9271 :  7 :    sf_burn : qsim::propagate_at_surface burn 
    15 :     0.0000 :  0 :      undef : undef 
    16 :     0.0000 :  0 :      undef : undef 

    In [49]: a.flat[sc1,2]
    Out[49]: array([0.99929315, 0.9997302 , 0.99909383, 0.99983627, 0.99907964, 0.999954  , 0.99951273, 0.99929965, 0.99926066], dtype=float32)



