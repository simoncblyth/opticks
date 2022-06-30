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






