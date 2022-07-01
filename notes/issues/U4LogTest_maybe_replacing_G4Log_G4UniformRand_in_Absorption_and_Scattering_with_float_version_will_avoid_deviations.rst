U4LogTest_maybe_replacing_G4Log_G4UniformRand_in_Absorption_and_Scattering_with_float_version_will_avoid_deviations
=====================================================================================================================

* from :doc:`higher_stats_U4RecorderTest_cxs_rainbow_random_aligned_comparison`


Overview
---------

The leading cause of deviations in random aligned comparisons of:

A: single precision Opticks
B: double precision Geant4 

is the position of "AB" BULK_ABSORB and "SC" BULK_SCATTER positions. 

When SC and AB processes win the random number u is typically very close to 1, eg 0.9999 = 1 - 1e-4
which means that -log(u) is small.  
When there are deviations the scattering and absorption lengths are usually large, eg 1e7 for 
absorption in Air so the distance to the scatter or absoption takes the form of the product 
of large and small numbers::

    -length*log(1-x)      # eg x = 1e-4             # -log(1-x) ~ x    for x small 
    ~   x*length        # 1e-4*1e7 = 1e3 


Given that the scattering/absorption distance is the result 
of a random throw it really does not matter if its 0.1mm OR 1 mm OR 2 mm different. 
BUT the deviation between A and B is inconvenient as it blunts the usefulness of the 
random aligned comparison. So perhaps can degrade the Geant4 calulation to make it match 
the Opticks float calc ?


Partial float degradation doesnt get close to the float result : need to do whole thing in float
--------------------------------------------------------------------------------------------------

Thinking about how to degrade the Geant4 side to make it 
match the purely float calculation : find that have to do whole calc in float : suggesting 
the loss of precision is not "concentrated" in one aspect of the calc. 

::

    In [16]: u = np.float64(0.9999085)

    In [17]: -1e7*np.log(u)
    Out[17]: 915.0418638039326

    In [18]: -np.float32(1e7)*np.log(np.float32(u))
    Out[18]: 914.97314

    In [19]: -1e7*np.float32(np.log(u))
    Out[19]: 915.0418918579817

    In [20]: np.log(u)
    Out[20]: -9.150418638039327e-05

    In [21]: -np.float32(1e7)*np.log(u)
    Out[21]: 915.0418638039326

    In [22]: -np.float32(1e7)*np.float32(np.log(u))
    Out[22]: 915.0419

    In [23]: -np.float32(1e7)*np.float32(np.log(np.float32(u)))
    Out[23]: 914.97314



10k : Try FLOAT shimming G4VDiscreteProcess::PostStepGetPhysicalInteractionLength
-----------------------------------------------------------------------------------

* difference in AB and SC not as big as expected 

::

    095      if( FLOAT )
     96      {
     97           float fvalue = float(theNumberOfInteractionLengthLeft) * float(currentInteractionLength) ;
     98           value = fvalue ;
     99      }
    100      else
    101      {
    102           value = theNumberOfInteractionLengthLeft * currentInteractionLength ;
    103      }
    104 



::


    a.base                                             : /tmp/blyth/opticks/U4RecorderTest/ShimG4OpAbsorption_FLOAT_ShimG4OpRayleigh_FLOAT
    b.base                                             : /tmp/blyth/opticks/U4RecorderTest/ShimG4OpAbsorption_ORIGINAL_ShimG4OpRayleigh_ORIGINAL

    In [1]: np.where( np.abs(a.photon-b.photon) > 0.1 )
    Out[1]: (array([], dtype=int64), array([], dtype=int64), array([], dtype=int64))

    In [2]: np.where( np.abs(a.photon-b.photon) > 0.01 )
    Out[2]: (array([], dtype=int64), array([], dtype=int64), array([], dtype=int64))

    In [7]: w = np.unique(np.where( np.abs(a.photon-b.photon) > 1e-4  )[0]) ; w
    Out[7]: array([5156, 9964])

    In [8]: seqhis_(a.seq[w,0])
    Out[8]: ['TO SC BR SA', 'TO BT SC BT SA']

    In [9]: w = np.unique(np.where( np.abs(a.photon-b.photon) > 1e-6 )[0]) ; w
    Out[9]: array([ 201,  230,  387,  549, 1292, 1338, 1475, 2263, 2515, 3276, 3771, 3846, 4097, 4468, 4524, 4573, 5156, 6555, 6797, 6907, 6925, 7203, 7554, 7604, 7791, 8235, 8393, 9654, 9964])

    In [10]: seqhis_(a.seq[w,0])
    Out[10]: 
    ['TO BT AB',
     'TO BT AB',
     'TO SC SA',
     'TO BT AB',
     'TO BT BT SC SA',
     'TO SC SA',
     'TO BT AB',
     'TO BT AB',
     'TO BT AB',
     'TO BR SC SA',
     'TO BT BR BR BR BR AB',
     'TO BT BT SC SA',
     'TO BT AB',
     'TO BT AB',
     'TO BT AB',
     'TO BT BR BT SC SA',
     'TO SC BR SA',
     'TO BT BT SC SA',
     'TO BT AB',
     'TO BT BR AB',
     'TO BT AB',
     'TO SC SA',
     'TO BT AB',
     'TO BT BT SC SA',
     'TO SC SA',
     'TO SC SA',
     'TO BT BT AB',
     'TO BT BT SC SA',
     'TO BT SC BT SA']



__logf
---------

::

    __device__ float __logf ( float  x )
        Calculate the fast approximate base e logarithm of the input argument.
    Returns

    Returns an approximation to loge(x)

    Note:
    For accuracy information see the CUDA C++ Programming Guide, Mathematical Functions Appendix, Intrinsic Functions section.



* https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#math-libraries

The -use_fast_math compiler option of nvcc coerces every functionName() call to the equivalent __functionName() 


Suspect SC AB deviants may be impact of CUDA fast math ?
--------------------------------------------------------------

* float degrading the Geant4 calc does not change much at all : this makes me suspect CUDA fast math  


* https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__SINGLE.html
* https://forums.developer.nvidia.com/t/fastmath-functions-speed-or-accuracy/8587/2

MisterAnderson42
Apr 16 '09

The biggest thing to be aware of with the fastmath operations is not how
good/bad your input data is, but rather its range! See the list of all fastmath
functions (i.e. __cosf()) in the programming guide. They only produce valid
results for a given range of input values. There have been a number of
questions on the forum in the past few months about invalid values from math
functions that turned out to be the result of passing input values outside the
range and using the fastmath compiler option.

I always compile without the fastmath option so there are no surprises and then
directly call the fastmath intrinsic functions in the code where and when I am
positive the input values will not be outside the defined range.


Difference between __logf and logf is jumpy close to zero rather constant close to one::

    epsilon:tests blyth$ ./logTest.sh 
    === ./logTest.sh : opt
    /tmp/logTest.npy
    [[ 0.       inf    inf    inf]
     [ 0.    13.816 13.816 13.816]
     [ 0.    13.122 13.122 13.122]
     ...
     [ 1.     0.     0.     0.   ]
     [ 1.     0.     0.     0.   ]
     [ 1.    -0.    -0.    -0.   ]]

    In [1]: (a[1:,2] - a[1:,3]).max()                                                                                                                                                             
    Out[1]: 9.5367431640625e-07



    In [3]: (a[1:100:,2] - a[1:100:,3])*1e7
    Out[3]: 
    array([0.   , 0.   , 0.   , 0.   , 0.   , 9.537, 9.537, 0.   , 9.537, 0.   , 9.537, 0.   , 0.   , 9.537, 0.   , 0.   , 0.   , 0.   , 9.537, 0.   , 9.537, 0.   , 9.537, 9.537, 0.   , 0.   , 0.   ,
           0.   , 0.   , 9.537, 0.   , 0.   , 9.537, 9.537, 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 9.537, 0.   , 0.   , 0.   , 0.   , 0.   , 9.537, 0.   , 0.   , 9.537, 0.   , 0.   ,
           0.   , 9.537, 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 9.537, 0.   , 9.537, 0.   , 0.   , 9.537, 0.   , 0.   , 9.537, 0.   , 0.   , 9.537, 9.537, 9.537, 0.   , 0.   , 9.537,
           9.537, 0.   , 9.537, 0.   , 0.   , 9.537, 0.   , 0.   , 0.   , 0.   , 9.537, 9.537, 9.537, 9.537, 9.537, 0.   , 0.   , 0.   ])




    In [2]:  (a[-100:,2] - a[-100:,3])*1e7 
    Out[2]: 
    array([0.455, 0.459, 0.463, 0.471, 0.465, 0.478, 0.463, 0.472, 0.457, 0.47 , 0.474, 0.477, 0.467, 0.48 , 0.484, 0.497, 0.478, 0.453, 0.466, 0.461, 0.474, 0.473, 0.477, 0.462, 0.475, 0.475, 0.488,
           0.464, 0.477, 0.481, 0.49 , 0.475, 0.479, 0.483, 0.492, 0.486, 0.499, 0.475, 0.442, 0.451, 0.455, 0.468, 0.453, 0.462, 0.466, 0.48 , 0.465, 0.46 , 0.459, 0.473, 0.467, 0.481, 0.481, 0.485,
           0.461, 0.474, 0.478, 0.487, 0.454, 0.458, 0.472, 0.453, 0.466, 0.461, 0.475, 0.47 , 0.479, 0.483, 0.478, 0.454, 0.463, 0.468, 0.481, 0.467, 0.481, 0.481, 0.494, 0.48 , 0.456, 0.465, 0.461,
           0.474, 0.479, 0.483, 0.455, 0.469, 0.473, 0.487, 0.478, 0.473, 0.477, 0.491, 0.477, 0.486, 0.482, 0.496, 0.491, 0.5  , 0.505, 0.   ])



When using "-use_fast_math" there is no difference between __logf and logf because logf uses __logf::

    In [1]: (a[-100:,2] - a[-100:,3])*1e7
    Out[1]: 
    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0.])






2 ipython sessions with differnt B_FOLD::

    In [13]: seqhis_(a.seq[w0,0])
    Out[13]: ['TO BT BT AB', 'TO BT BT AB']

    In [12]: rdist_(a,2)[w0]   ## dist from point 2->3 
    Out[12]: array([914.525,  69.861], dtype=float32)

    In [14]: rdist_(b,2)[w0]    
    Out[14]: array([914.973,  70.334], dtype=float32)

    In [16]: (a.base,b.base)
    Out[16]: 
    ('/tmp/blyth/opticks/GeoChain/BoxedSphere/CXRaindropTest',
     '/tmp/blyth/opticks/U4RecorderTest/ShimG4OpAbsorption_FLOAT_ShimG4OpRayleigh_FLOAT')



    In [12]: A(w0[0])
    Out[12]: 
    A(5208) : TO BT BT AB
    ...
    12 :     0.0153 :  1 :     to_sci : qsim::propagate_to_boundary u_to_sci burn 
    13 :     0.7635 :  2 :     to_bnd : qsim::propagate_to_boundary u_to_bnd burn 
    14 :     0.5736 :  3 :     to_sca : qsim::propagate_to_boundary u_scattering 
    15 :     0.9999 :  4 :     to_abs : qsim::propagate_to_boundary u_absorption 


    In [13]: a.flat[w0[0],15]
    Out[13]: 0.9999085

    In [14]: -1e7*np.log(a.flat[w0[0],15])
    Out[14]: 914.9731340585276




    N[blyth@localhost CSGOptiX]$ PIDX=5208 ./cxs_raindrop.sh 
    ...
    //qsim.propagate_at_boundary idx 5208 mom_1 (   -0.3479     0.1151     0.9304) 
    //qsim.propagate_at_boundary idx 5208 pol_1 (   -0.3140    -0.9494     0.0000) 
    //qsim.propagate idx 5208 bnc 2 cosTheta     0.9304 dir (   -0.3479     0.1151     0.9304) nrm (    0.0000     0.0000     1.0000) 
    //qsim.propagate_to_boundary[ idx 5208 u_absorption 0.99990851 logf(u_absorption) -0.00009145 absorption_length 10000000.0000 absorption_distance 914.525269 
    //qsim.propagate idx 5208 bounce 2 command 1 flag 8 s.optical.x 99 


    2022-07-01 14:19:04.197 INFO  [34923287] [U4Recorder::PreUserTrackingAction_Optical@113]  label.id 6000
    ShimG4OpRayleigh::PostStepGetPhysicalInteractionLength PIDX 5208 currentInteractionLength 1000000.0000000 theNumberOfInteractionLengthLeft  1.0320371 value 1032037.1250000
    ShimG4OpAbsorption::PostStepGetPhysicalInteractionLength PIDX 5208 currentInteractionLength 10000000.0000000 theNumberOfInteractionLengthLeft  1.3027622 value 13027622.0000000
    ShimG4OpRayleigh::PostStepGetPhysicalInteractionLength PIDX 5208 currentInteractionLength 374117.6507688 theNumberOfInteractionLengthLeft  0.2469110 value 92373.7812500
    ShimG4OpAbsorption::PostStepGetPhysicalInteractionLength PIDX 5208 currentInteractionLength 38563.0354202 theNumberOfInteractionLengthLeft  2.4086330 value 92884.1953125
    ShimG4OpRayleigh::PostStepGetPhysicalInteractionLength PIDX 5208 currentInteractionLength 1000000.0000000 theNumberOfInteractionLengthLeft  0.5557740 value 555774.0625000
    ShimG4OpAbsorption::PostStepGetPhysicalInteractionLength PIDX 5208 currentInteractionLength 10000000.0000000 theNumberOfInteractionLengthLeft  0.0000915 value 914.9731445
    2022-07-01 14:19:10.289 INFO  [34923287] [U4Recorder::PreUserTrackingAction_Optical@113]  label.id 5000



Use sysrap/tests/logTest.cu array to lookup the __logf::

    In [19]: tr = np.where( np.abs( a[:,0] - 0.9999085 ) < 1e-5 )[0] ; tr
    Out[19]: array([999899, 999900, 999901, 999902, 999903, 999904, 999905, 999906, 999907, 999908, 999909, 999910, 999911, 999912, 999913, 999914, 999915, 999916, 999917, 999918])

    In [20]: np.set_printoptions(precision=10)

    In [21]: a[tr]*1e7
    Out[21]: 
    array([[9998990.          ,    1009.753686958 ,    1009.753686958 ,    1009.2915908899],
           [9999000.          ,    1000.2159251599,    1000.2159251599,     999.7455345001],
           [9999010.          ,     990.0821896736,     990.0821896736,     989.6268602461],
           [9999020.          ,     979.9483814277,     979.9483814277,     979.4894140214],
           [9999030.          ,     969.8145731818,     969.8145731818,     969.3518950371],
           [9999040.          ,     960.2769569028,     960.2769569028,     959.8058386473],
           [9999050.          ,     950.1431486569,     950.1431486569,     949.6778511675],
           [9999060.          ,     940.0094131706,     940.0094131706,     939.5310189575],
           [9999070.          ,     929.8756776843,     929.8756776843,     929.4123447035],
           [9999080.          ,     920.3380614053,     920.3380614053,     919.8662883136],
           [9999090.          ,     910.204325919 ,     910.204325919 ,     909.7475413   ],
           [9999100.          ,     900.0706631923,     900.0706631923,     899.6008546092],
           [9999110.          ,     889.936927706 ,     889.936927706 ,     889.4634083845],
           [9999120.          ,     879.8032649793,     879.8032649793,     879.3259621598],
           [9999130.          ,     870.2656487003,     870.2656487003,     869.798604981 ],
           [9999140.          ,     860.1319859736,     860.1319859736,     859.651772771 ],
           [9999150.          ,     849.9983232468,     849.9983232468,     849.5143993059],
           [9999160.          ,     839.8647332797,     839.8647332797,     839.3675670959],
           [9999170.          ,     830.3271897603,     830.3271897603,     829.8496686621],
           [9999180.          ,     820.1935270336,     820.1935270336,     819.7403076338]])

    In [22]: tr = np.where( np.abs( a[:,0] - 0.9999085 ) < 1e-6 )[0] ; tr
    Out[22]: array([999908, 999909])

    In [23]: a[tr]*1e7
    Out[23]: 
    array([[9999080.          ,     920.3380614053,     920.3380614053,     919.8662883136],
           [9999090.          ,     910.204325919 ,     910.204325919 ,     909.7475413   ]])

    In [25]: np.interp( 0.9999085, a[:,0], a[:,3] )*1e7    ## interpolating the __logf result gets close 
    Out[25]: 914.8069148073934

    In [26]: np.interp( 0.9999085, a[:,0], a[:,2] )*1e7    ## interpolating the logf result 
    Out[26]: 915.2711936627552

::

     06 __global__ void test_log_(double* dd, unsigned ni)
      7 {
      8     unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;
      9 
     10     double d = double(ix)/double(ni-1) ;
     11     float  f = float(d) ;
     12 
     13     double d0 = -1.*logf( d );
     14     float  f0 = -1.f*logf( f );
     15     float  f1 = -1.f*__logf( f );
     16 
     17     dd[ix*4+0] = d ;
     18     dd[ix*4+1] = d0 ;
     19     dd[ix*4+2] = f0 ;
     20     dd[ix*4+3] = f1 ;
     21 
     22     //printf("//test_log  (ix,iy,ni) (%2d, %2d, %2d) \n", ix, iy, ni );
     23 }


Looks pretty clear that the smaller Opticks distance arises due to __logf imprecision. 

::

    In [8]: rdist_(a,2)[w0]
    Out[8]: array([914.525,  69.861], dtype=float32)

    In [9]: rdist_(b,2)[w0]
    Out[9]: array([914.973,  70.334], dtype=float32)

    In [10]: (a.base,b.base)
    Out[10]: 
    ('/tmp/blyth/opticks/GeoChain/BoxedSphere/CXRaindropTest',
     '/tmp/blyth/opticks/U4RecorderTest/ShimG4OpAbsorption_ORIGINAL_ShimG4OpRayleigh_ORIGINAL')


Is the __logf difference constant enough for a kludge fix
-------------------------------------------------------------

::

    #define KLUDGE_FASTMATH_LOGF(u) (u < 0.998f ? __logf(u) : __logf(u) - 0.46735790f*1e-7f )


sysrap/tests/logTest.ch::

    In [32]: d23 = (a[:,2] - a[:,3])*1e7   ## -logf - -__logf value 
    /Users/blyth/miniconda3/bin/ipython:1: RuntimeWarning: invalid value encountered in subtract
      #!/Users/blyth/miniconda3/bin/python

    In [33]: np.c_[a[-2000:],d23[-2000:]]                                                                                                                                                          
    Out[33]: 
    array([[ 0.998001    ,  0.0020010213,  0.0020010213,  0.0020009747,  0.4656612873],
           [ 0.998002    ,  0.0020000059,  0.0020000059,  0.0019999619,  0.4400499165],
           [ 0.998003    ,  0.0019989908,  0.0019989908,  0.0019989482,  0.4260800779],
           ...,
           [ 0.999998    ,  0.0000020266,  0.0000020266,  0.0000019766,  0.5000629244],
           [ 0.999999    ,  0.0000010133,  0.0000010133,  0.0000009628,  0.5047138529],
           [ 1.          , -0.          , -0.          , -0.          ,  0.          ]])


    In [39]: d23[-2000:-1].min()
    Out[39]: 0.417931005358696

    In [40]: d23[-2000:-1].max() 
    Out[40]: 0.5122274160385132

    In [41]: d23[-2000:-1].sum()
    Out[41]: 934.7158072614548

    In [42]: d23[-2000:-1].sum()/2000.
    Out[42]: 0.4673579036307274






Search for a better way to do : -length*log(u) in float precision : yields nothing
---------------------------------------------------------------------------------------

* :google:`improving float precision logarithmic functions`

* https://cme.h-its.org/exelixis/pubs/Exelixis-RRDR-2009-4.pdf

* https://blog.demofox.org/2017/11/21/floating-point-precision/



Check G4Log vs std::log
-------------------------

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


Take look close to 1, close to 0 and in the middle::

    In [7]: 1e7*(b[-100:, D0]-b[-100:, F0])
    Out[7]: 
    array([-0.0331814394,  0.0996417103,  0.2324748608, -0.2308739536, -0.0980207993,  0.0347695986,  0.1675699972, -0.2957388097, -0.1629184074, -0.0301607618,  0.1026796441,  0.2354572922,
           -0.2278015057, -0.0950038529,  0.0378038004,  0.1705486959, -0.2927428537, -0.1599051953, -0.0271302948,  0.1056546072,  0.238449512 , -0.2247920305, -0.0920498833,  0.040775025 ,
            0.173609936 , -0.2897371196, -0.1568822067, -0.024090052 ,  0.1087121053,  0.2414515033, -0.2217727855, -0.0890133847,  0.0438287781,  0.1766081816, -0.2867216209, -0.1539222148,
           -0.0211128063,  0.1117066024,  0.2444996325, -0.2187801638, -0.0860035099,  0.0468195238,  0.1796161789, -0.2836599925, -0.1508797138, -0.0180894352,  0.1147108446,  0.2475211265,
           -0.2157777991, -0.082983896 ,  0.0498200081,  0.1826339142, -0.2806613871, -0.14786386  , -0.0150563319,  0.1177248184,  0.2505159684, -0.2127657082, -0.0799545564,  0.0528302175,
            0.1856249911, -0.2776530621, -0.1448382869, -0.0120498898,  0.1207485071,  0.2535205248, -0.2097439046, -0.0769518844,  0.0558501353,  0.1886439657, -0.2746350308, -0.141839388 ,
           -0.0090337457,  0.1237455173,  0.2565529718, -0.2067305917, -0.073939517 ,  0.0588615581,  0.1916544448, -0.2716073068, -0.1388126102, -0.0060261032,  0.1267704053,  0.2595587232,
           -0.2037075963, -0.0709174677,  0.0618735673,  0.1946655065, -0.2685971917, -0.1358034419, -0.0030178807,  0.1297776796,  0.2625695976, -0.2006931219, -0.0679039398,  0.0648861462,
            0.1976771374, -0.2655796878, -0.1327902959,  0.          ])

    In [9]: 1e7*(b[1:100, D0]-b[1:100, F0])                                                                                                                                                       
    Out[9]: 
    array([-1.918526209 ,  2.8688919151,  3.0481862545, -1.8804331248, -1.617818377 , -1.7011387854,  1.1288845236,  2.9069849994, -1.521844446 ,  3.1695997471,  3.4833728968,  3.0862793388,
            4.4000469046, -3.6204405163,  3.3488940865, -1.8423400405,  2.7024350757,  3.2655736781, -2.3669288929, -1.5797252928, -3.4411461769, -1.2659521431, -1.756040362 , -1.6630457012,
           -1.3171105628, -0.3492781353,  3.4448679997,  1.1669776079,  1.970329766 , -1.4004309534,  2.211549095 ,  2.9450781014, -1.0866578037, -2.0468899642,  1.4295923556, -1.4837513618,
           -0.5945289949,  2.4204892313, -0.1699837959,  3.2076928314,  2.4357930251,  1.3462719473, -3.5637430784,  3.521465981 , -1.2211366318,  3.0313777621,  4.0742609109,  3.124372423 ,
            4.1762952563,  3.4703075613, -1.8675956426,  4.4381399888, -0.1403511263, -1.3044570402,  3.7840807288, -3.5823474143,  2.5997835706, -2.7789952739, -0.9913724597,  3.3869871707,
            1.1999945926, -2.5377759449,  1.5255662866, -1.8042469385,  4.7007547366,  3.7007603204,  0.2933258614,  2.74052816  ,  3.2106721015, -3.3197326843, -0.1518299797,  3.3036667624,
           -0.001274767 ,  4.1928891292,  3.6496019007, -2.3288358086, -3.0059595346, -4.9193088358, -3.361897587 , -1.5416322086, -1.1251627008, -2.3135320149, -3.9132402385, -3.4030530927,
            3.0031428899,  1.2236750457, -2.5997009523, -1.2278590411,  4.0893525899,  3.5662814923, -2.0892855268, -1.7179472778, -2.3584816233, -0.675064129 , -2.0662210609, -1.6249526169,
           -3.2440873809, -0.5730297659,  3.8800546598])

    In [10]: 1e7*(b[500000:500100, D0]-b[500000:500100, F0])                                                                                                                                      
    Out[10]: 
    array([-0.0190465432,  0.2465526805,  0.5121919056, -0.414221768 , -0.1485025458,  0.1172566755,  0.3830558981, -0.5431977779, -0.2773185592, -0.0113993415,  0.2545598765,  0.5205590903,
           -0.4054945923, -0.1394153815,  0.1267038285,  0.3928630354, -0.5330306563, -0.2667914556, -0.0005122547,  0.2658069409,  0.5321661334, -0.3935275716, -0.1270883843,  0.1393907989,
            0.4059099779, -0.5196237429, -0.2530245702,  0.0136145972,  0.2802937593,  0.5470129205, -0.3783208224, -0.111521673 ,  0.15531747  ,  0.422196611 , -0.502977151 , -0.2360180229,
            0.0309810988,  0.2980202174,  0.5650993284, -0.3598744625, -0.0927153643,  0.1744837297,  0.441722815 , -0.4830910028, -0.2157719303,  0.0515871368,  0.3189861941,  0.5864252439,
           -0.3381886093, -0.0706695724,  0.1968894547,  0.4644884743, -0.4599654102, -0.1922864057,  0.075432588 ,  0.343191574 ,  0.6109905526, -0.3132633752, -0.0453844162,  0.222534533 ,
            0.4904934747, -0.43360049  , -0.16556157  ,  0.1025173413,  0.3706362439,  0.6387951335, -0.2850988823, -0.0168600123,  0.251418848 ,  0.5197376962, -0.403996363 , -0.1355975376,
            0.1328412791,  0.4013200816,  0.6698388721, -0.2536952437,  0.0149035251,  0.2835422808,  0.5522210234, -0.3711531427, -0.1023944229,  0.1664042826,  0.4352429739,  0.7041216521,
           -0.219052575 ,  0.049906077 ,  0.3189047149,  0.5879433418, -0.3350709432, -0.0659523469,  0.2032062363,  0.4724048064,  0.7416433589, -0.1811709971,  0.0881475259,  0.357506037 ,
            0.6269045305, -0.2957498868, -0.0262714239,  0.2432470259])



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

    -log(1-x) = x + x*x/2 


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



