curand-impl-review-and-compare-to-curand-done-right.rst
===========================================================

TODO
-----

* large scale generation, comparison + performance test
* investigate how to use curandStatePhilox4_32_10_OpticksLite within Opticks
* currently have mix of curandState  curandState_t and curandStateXORWOW
* https://stackoverflow.com/questions/10747810/what-is-the-difference-between-typedef-and-using   

::

    using opticks_curandState_t = curandStatePhilox4_32_10_OpticksLite ;


~/o/sysrap/tests/curand_uniform_test.sh
-------------------------------------------

+---------------------------------------+----------------+--------------------------------------------------+
|                                       |  sizeof bytes  |   notes                                          |
+=======================================+================+==================================================+
| curandStateXORWOW                     |    48          |  curand default, expensive init => complications |
+---------------------------------------+----------------+--------------------------------------------------+
| curandStatePhilox4_32_10              |    64          |  cheap init (TODO: check in practice)            |
+---------------------------------------+----------------+--------------------------------------------------+
| curandStatePhilox4_32_10_OpticksLite  |    32          |  slim state to uint4 + uint2, gets padded to 32  |
+---------------------------------------+----------------+--------------------------------------------------+


The 2nd two are giving the same random streams.   Slimming accomplished by removing functionality 
not needed by Opticks. 


::

    P[blyth@localhost notes]$ MODE=0 ~/o/sysrap/tests/curand_uniform_test.sh
             BASH_SOURCE : /home/blyth/o/sysrap/tests/curand_uniform_test.sh 
                    SDIR :  
                    name : curand_uniform_test 
                 altname :  
                     src : curand_uniform_test.cu 
                  script : curand_uniform_test.py 
                     bin : /tmp/curand_uniform_test/curand_uniform_test 
                    FOLD : /tmp/curand_uniform_test 
    opt
    test_curand_uniform<curandStateXORWOW>()//test_curand_uniform  sizeof(T) 48 
    //_test_curand_uniform sizeof(T) 48 
    a.shape
     (1000, 16)
    a[:10]
     [[0.74022 0.43845 0.51701 0.15699 0.07137 0.46251 0.22764 0.32936 0.14407 0.1878  0.91538 0.54012 0.97466 0.54747 0.65316 0.23024]
     [0.92099 0.46036 0.33346 0.37252 0.4896  0.56727 0.07991 0.23337 0.50938 0.08898 0.00671 0.95423 0.54671 0.82455 0.52706 0.93013]
     [0.03902 0.25021 0.18448 0.96242 0.52055 0.93996 0.83058 0.40973 0.08162 0.80677 0.69529 0.61771 0.25633 0.21368 0.34242 0.22408]
     [0.96896 0.49474 0.67338 0.56277 0.12019 0.97649 0.13583 0.58897 0.49062 0.32844 0.91143 0.19068 0.9637  0.89755 0.62429 0.71015]
     [0.92514 0.05301 0.1631  0.88969 0.56664 0.24142 0.49369 0.32123 0.07861 0.14788 0.59866 0.42647 0.24347 0.48918 0.40953 0.66764]
     [0.44635 0.3377  0.20723 0.98454 0.40279 0.1781  0.45992 0.16001 0.36089 0.62038 0.45004 0.30574 0.50284 0.45595 0.5516  0.84838]
     [0.66732 0.39676 0.15829 0.5423  0.7056  0.12585 0.15365 0.65258 0.37992 0.85478 0.20781 0.0901  0.70118 0.43362 0.10571 0.08183]
     [0.10993 0.87442 0.98075 0.96693 0.16233 0.42767 0.93141 0.01003 0.84566 0.37989 0.81176 0.15237 0.27327 0.41338 0.78616 0.08703]
     [0.47022 0.48217 0.42791 0.44174 0.78041 0.85861 0.61435 0.80234 0.65919 0.59214 0.18296 0.71884 0.92713 0.42197 0.01055 0.82696]
     [0.51319 0.04284 0.95184 0.92588 0.25979 0.91341 0.39325 0.83318 0.27532 0.75222 0.66639 0.03765 0.87857 0.96512 0.03355 0.81466]]
    a[-10:]
     [[0.41888 0.56394 0.26219 0.00544 0.34131 0.24802 0.02585 0.42882 0.45842 0.68441 0.1162  0.07948 0.70902 0.93657 0.54654 0.41797]
     [0.6406  0.80706 0.12232 0.20049 0.90991 0.13225 0.18421 0.27288 0.83271 0.89976 0.48249 0.51084 0.22823 0.63753 0.43524 0.96682]
     [0.29197 0.19001 0.98212 0.68296 0.65355 0.74176 0.84946 0.58338 0.30676 0.91659 0.78078 0.0342  0.73427 0.05188 0.61055 0.85   ]
     [0.84055 0.33497 0.81023 0.68106 0.82873 0.87127 0.75434 0.55597 0.85694 0.36502 0.91378 0.68908 0.53978 0.20404 0.01672 0.14249]
     [0.6201  0.62216 0.83531 0.72095 0.70984 0.75301 0.60597 0.11183 0.2665  0.62516 0.12829 0.27882 0.71579 0.59997 0.41287 0.72082]
     [0.42809 0.7106  0.64159 0.94931 0.23182 0.09769 0.12973 0.39439 0.7484  0.05785 0.79519 0.12628 0.15853 0.12913 0.14954 0.98629]
     [0.93038 0.01259 0.53405 0.20617 0.06964 0.78301 0.62946 0.97189 0.22707 0.7842  0.72258 0.9895  0.12467 0.85368 0.76313 0.08281]
     [0.15602 0.99039 0.6817  0.11667 0.13779 0.3867  0.73269 0.66636 0.00007 0.97589 0.64677 0.22477 0.44537 0.20699 0.73511 0.35352]
     [0.16356 0.4678  0.83821 0.44082 0.21579 0.71205 0.03324 0.69551 0.22208 0.92826 0.24047 0.18735 0.79577 0.88763 0.34437 0.94503]
     [0.21777 0.24313 0.72559 0.24963 0.08471 0.51074 0.23489 0.12473 0.75238 0.91716 0.68549 0.11767 0.76911 0.00663 0.21612 0.32016]]


    P[blyth@localhost notes]$ MODE=1 ~/o/sysrap/tests/curand_uniform_test.sh run_ana
    test_curand_uniform<curandStatePhilox4_32_10>()//test_curand_uniform  sizeof(T) 64 
    //_test_curand_uniform sizeof(T) 64 
    a.shape
     (1000, 16)
    a[:10]
     [[0.39905 0.88052 0.73571 0.60548 0.97224 0.36209 0.69393 0.03709 0.01945 0.31945 0.14095 0.27175 0.78737 0.41511 0.60463 0.42678]
     [0.51668 0.93966 0.05899 0.51551 0.79097 0.8507  0.49376 0.42733 0.77335 0.74334 0.20882 0.66294 0.13441 0.62506 0.31619 0.2022 ]
     [0.02493 0.71293 0.14895 0.46576 0.46904 0.13516 0.84546 0.42525 0.86505 0.8156  0.91845 0.57245 0.21896 0.9205  0.09499 0.67676]
     [0.94008 0.06776 0.74559 0.72045 0.33    0.91784 0.96384 0.77223 0.80969 0.03124 0.77272 0.06174 0.81929 0.41844 0.41939 0.97252]
     [0.94585 0.55899 0.02431 0.07736 0.33448 0.30884 0.17946 0.89425 0.66656 0.48205 0.88184 0.89836 0.6969  0.79549 0.25767 0.3373 ]
     [0.79673 0.66475 0.48393 0.10628 0.37827 0.10689 0.0897  0.23682 0.36525 0.95412 0.30005 0.00428 0.73043 0.31533 0.6468  0.33497]
     [0.41501 0.57781 0.79775 0.9886  0.76398 0.47173 0.44653 0.56775 0.36369 0.39063 0.06833 0.21797 0.23275 0.95811 0.83907 0.5221 ]
     [0.82026 0.58734 0.16943 0.19428 0.64049 0.76335 0.77852 0.48482 0.56807 0.13779 0.41269 0.0573  0.90283 0.31792 0.37071 0.46046]
     [0.22904 0.89237 0.06683 0.9178  0.11026 0.4369  0.70088 0.83708 0.26365 0.56926 0.61337 0.23371 0.39    0.98473 0.46428 0.68643]
     [0.90959 0.65702 0.16538 0.91668 0.35935 0.10573 0.14646 0.07193 0.72813 0.47244 0.20138 0.04001 0.6177  0.54076 0.19357 0.5996 ]]
    a[-10:]
     [[0.27815 0.79028 0.07982 0.45943 0.75886 0.37247 0.49381 0.11436 0.18936 0.98539 0.20821 0.8152  0.40681 0.32394 0.44613 0.73638]
     [0.67692 0.46743 0.51213 0.77866 0.4888  0.8207  0.22989 0.45666 0.45505 0.93782 0.60281 0.34289 0.24519 0.80186 0.63956 0.06417]
     [0.38146 0.43257 0.49413 0.06068 0.1002  0.85488 0.53535 0.66639 0.55232 0.54146 0.6255  0.11547 0.24554 0.04963 0.55223 0.69744]
     [0.62771 0.09779 0.79002 0.8909  0.63099 0.22769 0.68006 0.30199 0.77518 0.26527 0.54442 0.53448 0.42896 0.36567 0.65241 0.72206]
     [0.95036 0.29309 0.35829 0.56482 0.76213 0.17375 0.60631 0.21577 0.91425 0.19238 0.83841 0.71495 0.00334 0.37765 0.54948 0.46644]
     [0.7642  0.47318 0.93713 0.36356 0.24742 0.08482 0.24282 0.34807 0.2046  0.12787 0.04553 0.30839 0.04495 0.69015 0.63913 0.25015]
     [0.92507 0.42076 0.37747 0.66725 0.54503 0.19478 0.64196 0.35383 0.86129 0.63485 0.364   0.71312 0.15553 0.95252 0.20993 0.69912]
     [0.96775 0.35898 0.02535 0.2017  0.50908 0.26318 0.32378 0.26241 0.06272 0.05882 0.83958 0.97788 0.32591 0.57277 0.02442 0.4984 ]
     [0.37228 0.20774 0.51485 0.6704  0.53394 0.51579 0.72155 0.86344 0.33862 0.31515 0.94727 0.1328  0.57794 0.76964 0.37578 0.17121]
     [0.07151 0.22553 0.89116 0.43006 0.25292 0.02127 0.86665 0.26139 0.4433  0.15422 0.62573 0.41278 0.11604 0.45563 0.70743 0.23984]]
    P[blyth@localhost notes]$ 
    P[blyth@localhost notes]$ 
    P[blyth@localhost notes]$ MODE=2 ~/o/sysrap/tests/curand_uniform_test.sh run_ana
    test_curand_uniform<curandStatePhilox4_32_10_OpticksLite>()//test_curand_uniform  sizeof(T) 32 
    //_test_curand_uniform sizeof(T) 32 
    a.shape
     (1000, 16)
    a[:10]
     [[0.39905 0.88052 0.73571 0.60548 0.97224 0.36209 0.69393 0.03709 0.01945 0.31945 0.14095 0.27175 0.78737 0.41511 0.60463 0.42678]
     [0.51668 0.93966 0.05899 0.51551 0.79097 0.8507  0.49376 0.42733 0.77335 0.74334 0.20882 0.66294 0.13441 0.62506 0.31619 0.2022 ]
     [0.02493 0.71293 0.14895 0.46576 0.46904 0.13516 0.84546 0.42525 0.86505 0.8156  0.91845 0.57245 0.21896 0.9205  0.09499 0.67676]
     [0.94008 0.06776 0.74559 0.72045 0.33    0.91784 0.96384 0.77223 0.80969 0.03124 0.77272 0.06174 0.81929 0.41844 0.41939 0.97252]
     [0.94585 0.55899 0.02431 0.07736 0.33448 0.30884 0.17946 0.89425 0.66656 0.48205 0.88184 0.89836 0.6969  0.79549 0.25767 0.3373 ]
     [0.79673 0.66475 0.48393 0.10628 0.37827 0.10689 0.0897  0.23682 0.36525 0.95412 0.30005 0.00428 0.73043 0.31533 0.6468  0.33497]
     [0.41501 0.57781 0.79775 0.9886  0.76398 0.47173 0.44653 0.56775 0.36369 0.39063 0.06833 0.21797 0.23275 0.95811 0.83907 0.5221 ]
     [0.82026 0.58734 0.16943 0.19428 0.64049 0.76335 0.77852 0.48482 0.56807 0.13779 0.41269 0.0573  0.90283 0.31792 0.37071 0.46046]
     [0.22904 0.89237 0.06683 0.9178  0.11026 0.4369  0.70088 0.83708 0.26365 0.56926 0.61337 0.23371 0.39    0.98473 0.46428 0.68643]
     [0.90959 0.65702 0.16538 0.91668 0.35935 0.10573 0.14646 0.07193 0.72813 0.47244 0.20138 0.04001 0.6177  0.54076 0.19357 0.5996 ]]
    a[-10:]
     [[0.27815 0.79028 0.07982 0.45943 0.75886 0.37247 0.49381 0.11436 0.18936 0.98539 0.20821 0.8152  0.40681 0.32394 0.44613 0.73638]
     [0.67692 0.46743 0.51213 0.77866 0.4888  0.8207  0.22989 0.45666 0.45505 0.93782 0.60281 0.34289 0.24519 0.80186 0.63956 0.06417]
     [0.38146 0.43257 0.49413 0.06068 0.1002  0.85488 0.53535 0.66639 0.55232 0.54146 0.6255  0.11547 0.24554 0.04963 0.55223 0.69744]
     [0.62771 0.09779 0.79002 0.8909  0.63099 0.22769 0.68006 0.30199 0.77518 0.26527 0.54442 0.53448 0.42896 0.36567 0.65241 0.72206]
     [0.95036 0.29309 0.35829 0.56482 0.76213 0.17375 0.60631 0.21577 0.91425 0.19238 0.83841 0.71495 0.00334 0.37765 0.54948 0.46644]
     [0.7642  0.47318 0.93713 0.36356 0.24742 0.08482 0.24282 0.34807 0.2046  0.12787 0.04553 0.30839 0.04495 0.69015 0.63913 0.25015]
     [0.92507 0.42076 0.37747 0.66725 0.54503 0.19478 0.64196 0.35383 0.86129 0.63485 0.364   0.71312 0.15553 0.95252 0.20993 0.69912]
     [0.96775 0.35898 0.02535 0.2017  0.50908 0.26318 0.32378 0.26241 0.06272 0.05882 0.83958 0.97788 0.32591 0.57277 0.02442 0.4984 ]
     [0.37228 0.20774 0.51485 0.6704  0.53394 0.51579 0.72155 0.86344 0.33862 0.31515 0.94727 0.1328  0.57794 0.76964 0.37578 0.17121]
     [0.07151 0.22553 0.89116 0.43006 0.25292 0.02127 0.86665 0.26139 0.4433  0.15422 0.62573 0.41278 0.11604 0.45563 0.70743 0.23984]]
    P[blyth@localhost notes]$ 



    P[blyth@localhost opticks]$ ~/o/sysrap/tests/curanddr_uniform_test.sh
             BASH_SOURCE : /home/blyth/o/sysrap/tests/curanddr_uniform_test.sh 
                    SDIR :  
                    name : curanddr_uniform_test 
                 altname :  
                     src : curanddr_uniform_test.cu 
                  script : curanddr_uniform_test.py 
                     bin : /tmp/curanddr_uniform_test/curanddr_uniform_test 
                    FOLD : /tmp/curanddr_uniform_test 
    opt
    //test_curanddr_uniform   
    a.shape
     (1000, 16)
    a[:10]
     [[0.39905 0.88052 0.73571 0.60548 0.97224 0.36209 0.69393 0.03709 0.01945 0.31945 0.14095 0.27175 0.78737 0.41511 0.60463 0.42678]
     [0.51668 0.93966 0.05899 0.51551 0.79097 0.8507  0.49376 0.42733 0.77335 0.74334 0.20882 0.66294 0.13441 0.62506 0.31619 0.2022 ]
     [0.02493 0.71293 0.14895 0.46576 0.46904 0.13516 0.84546 0.42525 0.86505 0.8156  0.91845 0.57245 0.21896 0.9205  0.09499 0.67676]
     [0.94008 0.06776 0.74559 0.72045 0.33    0.91784 0.96384 0.77223 0.80969 0.03124 0.77272 0.06174 0.81929 0.41844 0.41939 0.97252]
     [0.94585 0.55899 0.02431 0.07736 0.33448 0.30884 0.17946 0.89425 0.66656 0.48205 0.88184 0.89836 0.6969  0.79549 0.25767 0.3373 ]
     [0.79673 0.66475 0.48393 0.10628 0.37827 0.10689 0.0897  0.23682 0.36525 0.95412 0.30005 0.00428 0.73043 0.31533 0.6468  0.33497]
     [0.41501 0.57781 0.79775 0.9886  0.76398 0.47173 0.44653 0.56775 0.36369 0.39063 0.06833 0.21797 0.23275 0.95811 0.83907 0.5221 ]
     [0.82026 0.58734 0.16943 0.19428 0.64049 0.76335 0.77852 0.48482 0.56807 0.13779 0.41269 0.0573  0.90283 0.31792 0.37071 0.46046]
     [0.22904 0.89237 0.06683 0.9178  0.11026 0.4369  0.70088 0.83708 0.26365 0.56926 0.61337 0.23371 0.39    0.98473 0.46428 0.68643]
     [0.90959 0.65702 0.16538 0.91668 0.35935 0.10573 0.14646 0.07193 0.72813 0.47244 0.20138 0.04001 0.6177  0.54076 0.19357 0.5996 ]]
    a[-10:]
     [[0.27815 0.79028 0.07982 0.45943 0.75886 0.37247 0.49381 0.11436 0.18936 0.98539 0.20821 0.8152  0.40681 0.32394 0.44613 0.73638]
     [0.67692 0.46743 0.51213 0.77866 0.4888  0.8207  0.22989 0.45666 0.45505 0.93782 0.60281 0.34289 0.24519 0.80186 0.63956 0.06417]
     [0.38146 0.43257 0.49413 0.06068 0.1002  0.85488 0.53535 0.66639 0.55232 0.54146 0.6255  0.11547 0.24554 0.04963 0.55223 0.69744]
     [0.62771 0.09779 0.79002 0.8909  0.63099 0.22769 0.68006 0.30199 0.77518 0.26527 0.54442 0.53448 0.42896 0.36567 0.65241 0.72206]
     [0.95036 0.29309 0.35829 0.56482 0.76213 0.17375 0.60631 0.21577 0.91425 0.19238 0.83841 0.71495 0.00334 0.37765 0.54948 0.46644]
     [0.7642  0.47318 0.93713 0.36356 0.24742 0.08482 0.24282 0.34807 0.2046  0.12787 0.04553 0.30839 0.04495 0.69015 0.63913 0.25015]
     [0.92507 0.42076 0.37747 0.66725 0.54503 0.19478 0.64196 0.35383 0.86129 0.63485 0.364   0.71312 0.15553 0.95252 0.20993 0.69912]
     [0.96775 0.35898 0.02535 0.2017  0.50908 0.26318 0.32378 0.26241 0.06272 0.05882 0.83958 0.97788 0.32591 0.57277 0.02442 0.4984 ]
     [0.37228 0.20774 0.51485 0.6704  0.53394 0.51579 0.72155 0.86344 0.33862 0.31515 0.94727 0.1328  0.57794 0.76964 0.37578 0.17121]
     [0.07151 0.22553 0.89116 0.43006 0.25292 0.02127 0.86665 0.26139 0.4433  0.15422 0.62573 0.41278 0.11604 0.45563 0.70743 0.23984]]
    P[blyth@localhost opticks]$ 



To get the match between the three different Philox impl, you have to be careful regards which slot has which counter::

     18 __global__ void _test_curanddr_uniform(float* ff, int ni, int nj)
     19 {
     20     uint ix = blockIdx.x * blockDim.x + threadIdx.x;
     21     uint nk = nj/4 ;
     22     for(uint k=0 ; k < nk ; k++)
     23     {
     24         float* ffk = ff + 4*(ix*nk + k) ;
     25         curanddr::uniforms_into_buffer<4>( ffk, uint4{k,0,ix,0}, 0 ); 
     26     }
     27 }
     


     17 template<typename T>
     18 __global__ void _test_curand_uniform(float* ff, int ni, int nj)
     19 {
     20     unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;
     21 
     22     unsigned long long seed = 0ull ;
     23     unsigned long long subsequence = ix ;    // follow approach of ~/o/qudarap/QCurandState.cu 
     24     unsigned long long offset = 0ull ;
     25 
     26     T rng ;
     27 
     28     curand_init( seed, subsequence, offset, &rng );
     29 
     30     if(ix == 0) printf("//_test_curand_uniform sizeof(T) %lu \n", sizeof(T));
     31 
     32     int nk = nj/4 ;
     33 
     34     for(int k=0 ; k < nk ; k++)
     35     {
     36         float4 ans = curand_uniform4(&rng);
     37         ff[4*(ix*nk+k)+0] = ans.x ;
     38         ff[4*(ix*nk+k)+1] = ans.y ;
     39         ff[4*(ix*nk+k)+2] = ans.z ;
     40         ff[4*(ix*nk+k)+3] = ans.w ;
     41     }
     42 }
        









Philox C++ standard applic
---------------------------

* https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2023/p2075r3.pdf


curand-done-right : interesting lower level curand with no state : profiting from Philox counter based PRNG
-----------------------------------------------------------------------------------------------------------------

Investigations over in env::

::

   cd
   git clone https://github.com/kshitijl/curand-done-right.git

   cp ~/curand-done-right/src/curand-done-right/curanddr.hxx ~/env/cuda/curand-done-right/
   cp ~/curand-done-right/examples/basic-pi.cu ~/env/cuda/curand-done-right/
   cp ~/curand-done-right/Makefile ~/env/cuda/curand-done-right/


Getting 2 or 4 doesnt change the first 2::

    P[blyth@localhost curand-done-right]$ ./basic-pi.sh 
    //estimate_pi  index 0  xx  0.178931 yy  0.075331 
    //estimate_pi  index 1  xx  0.072204 yy  0.117255 
    //estimate_pi  index 2  xx  0.312774 yy  0.602896 
    //estimate_pi  index 3  xx  0.081673 yy  0.547574 
    //estimate_pi  index 4  xx  0.944169 yy  0.364360 
    //estimate_pi  index 5  xx  0.278512 yy  0.287804 
    //estimate_pi  index 6  xx  0.111264 yy  0.254863 
    //estimate_pi  index 7  xx  0.838473 yy  0.444990 
    //estimate_pi  index 8  xx  0.947367 yy  0.443467 
    //estimate_pi  index 9  xx  0.853467 yy  0.653512 
    785338 448
    3.141352


    P[blyth@localhost curand-done-right]$ ./basic-pi.sh 
    //estimate_pi  index 0  xx  0.178931 yy  0.075331 zz  0.988173 ww  0.634883 
    //estimate_pi  index 1  xx  0.072204 yy  0.117255 zz  0.283267 ww  0.039935 
    //estimate_pi  index 2  xx  0.312774 yy  0.602896 zz  0.603033 ww  0.364543 
    //estimate_pi  index 3  xx  0.081673 yy  0.547574 zz  0.456981 ww  0.642444 
    //estimate_pi  index 4  xx  0.944169 yy  0.364360 zz  0.667021 ww  0.933453 
    //estimate_pi  index 5  xx  0.278512 yy  0.287804 zz  0.692024 ww  0.071551 
    //estimate_pi  index 6  xx  0.111264 yy  0.254863 zz  0.414897 ww  0.567098 
    //estimate_pi  index 7  xx  0.838473 yy  0.444990 zz  0.229636 ww  0.167966 
    //estimate_pi  index 8  xx  0.947367 yy  0.443467 zz  0.803773 ww  0.519327 
    //estimate_pi  index 9  xx  0.853467 yy  0.653512 zz  0.157435 ww  0.020901 
    785338 448
    3.141352


curanddr.hxx API needs no init and just takes counter arguments::

     36     //curanddr::vector_t<2,float> randoms = curanddr::uniforms<2>(uint4{0,0,0,seed}, index); 
     37     curanddr::vector_t<4,float> randoms = curanddr::uniforms<4>(uint4{0,0,0,seed}, index);


Second call with same index gives same values::

    37     curanddr::vector_t<4,float> randoms = curanddr::uniforms<4>(uint4{0,0,0,seed}, index);
    38     curanddr::vector_t<4,float> randoms1 = curanddr::uniforms<4>(uint4{0,0,0,seed}, index);

    P[blyth@localhost curand-done-right]$ ./basic-pi.sh 
    //estimate_pi  index 0  xx  0.178931 yy  0.075331 zz  0.988173 ww  0.634883 
    //estimate_pi  index 1  xx  0.072204 yy  0.117255 zz  0.283267 ww  0.039935 
    //estimate_pi  index 2  xx  0.312774 yy  0.602896 zz  0.603033 ww  0.364543 
    //estimate_pi  index 3  xx  0.081673 yy  0.547574 zz  0.456981 ww  0.642444 
    //estimate_pi  index 4  xx  0.944169 yy  0.364360 zz  0.667021 ww  0.933453 
    //estimate_pi  index 5  xx  0.278512 yy  0.287804 zz  0.692024 ww  0.071551 
    //estimate_pi  index 6  xx  0.111264 yy  0.254863 zz  0.414897 ww  0.567098 
    //estimate_pi  index 7  xx  0.838473 yy  0.444990 zz  0.229636 ww  0.167966 
    //estimate_pi  index 8  xx  0.947367 yy  0.443467 zz  0.803773 ww  0.519327 
    //estimate_pi  index 9  xx  0.853467 yy  0.653512 zz  0.157435 ww  0.020901 
    //estimate_pi  index 0  xx1  0.178931 yy1  0.075331 zz1  0.988173 ww1  0.634883 
    //estimate_pi  index 1  xx1  0.072204 yy1  0.117255 zz1  0.283267 ww1  0.039935 
    //estimate_pi  index 2  xx1  0.312774 yy1  0.602896 zz1  0.603033 ww1  0.364543 
    //estimate_pi  index 3  xx1  0.081673 yy1  0.547574 zz1  0.456981 ww1  0.642444 
    //estimate_pi  index 4  xx1  0.944169 yy1  0.364360 zz1  0.667021 ww1  0.933453 
    //estimate_pi  index 5  xx1  0.278512 yy1  0.287804 zz1  0.692024 ww1  0.071551 
    //estimate_pi  index 6  xx1  0.111264 yy1  0.254863 zz1  0.414897 ww1  0.567098 
    //estimate_pi  index 7  xx1  0.838473 yy1  0.444990 zz1  0.229636 ww1  0.167966 
    //estimate_pi  index 8  xx1  0.947367 yy1  0.443467 zz1  0.803773 ww1  0.519327 
    //estimate_pi  index 9  xx1  0.853467 yy1  0.653512 zz1  0.157435 ww1  0.020901 
    785338 448
    3.141352
    P[blyth@localhost curand-done-right]$


Second call with index+1 repeats values from other thread::

    37     curanddr::vector_t<4,float> randoms = curanddr::uniforms<4>(uint4{0,0,0,seed}, index);
    38     curanddr::vector_t<4,float> randoms1 = curanddr::uniforms<4>(uint4{0,0,0,seed}, index+1);

    P[blyth@localhost curand-done-right]$ ./basic-pi.sh 
    //estimate_pi  index 0  xx  0.178931 yy  0.075331 zz  0.988173 ww  0.634883 
    //estimate_pi  index 1  xx  0.072204 yy  0.117255 zz  0.283267 ww  0.039935 
    //estimate_pi  index 2  xx  0.312774 yy  0.602896 zz  0.603033 ww  0.364543 
    //estimate_pi  index 3  xx  0.081673 yy  0.547574 zz  0.456981 ww  0.642444 
    //estimate_pi  index 4  xx  0.944169 yy  0.364360 zz  0.667021 ww  0.933453 
    //estimate_pi  index 5  xx  0.278512 yy  0.287804 zz  0.692024 ww  0.071551 
    //estimate_pi  index 6  xx  0.111264 yy  0.254863 zz  0.414897 ww  0.567098 
    //estimate_pi  index 7  xx  0.838473 yy  0.444990 zz  0.229636 ww  0.167966 
    //estimate_pi  index 8  xx  0.947367 yy  0.443467 zz  0.803773 ww  0.519327 
    //estimate_pi  index 9  xx  0.853467 yy  0.653512 zz  0.157435 ww  0.020901 
    //estimate_pi  index 0  xx1  0.072204 yy1  0.117255 zz1  0.283267 ww1  0.039935 
    //estimate_pi  index 1  xx1  0.312774 yy1  0.602896 zz1  0.603033 ww1  0.364543 
    //estimate_pi  index 2  xx1  0.081673 yy1  0.547574 zz1  0.456981 ww1  0.642444 
    //estimate_pi  index 3  xx1  0.944169 yy1  0.364360 zz1  0.667021 ww1  0.933453 
    //estimate_pi  index 4  xx1  0.278512 yy1  0.287804 zz1  0.692024 ww1  0.071551 
    //estimate_pi  index 5  xx1  0.111264 yy1  0.254863 zz1  0.414897 ww1  0.567098 
    //estimate_pi  index 6  xx1  0.838473 yy1  0.444990 zz1  0.229636 ww1  0.167966 
    //estimate_pi  index 7  xx1  0.947367 yy1  0.443467 zz1  0.803773 ww1  0.519327 
    //estimate_pi  index 8  xx1  0.853467 yy1  0.653512 zz1  0.157435 ww1  0.020901 
    //estimate_pi  index 9  xx1  0.030141 yy1  0.643481 zz1  0.333829 ww1  0.758343 
    785338 448
    3.141352
    P[blyth@localhost curand-done-right]$ 



check what curanddr.hxx does
-----------------------------

Usage::

    37     curanddr::vector_t<4,float> randoms = curanddr::uniforms<4>(uint4{0,0,0,seed}, index);


/home/blyth/env/cuda/curand-done-right/curanddr.hxx::

    076   template<int Arity>
     77   __device__ vector_t<Arity> uniforms(uint4 counter, uint key) {
     78     enum { n_blocks = (Arity + 4 - 1)/4 };
     79 
     80     float scratch[n_blocks * 4];
     81 
     82     iterate<n_blocks>([&](uint index) {
     83         uint2 local_key{key, index};
     84         uint4 result = curand_Philox4x32_10(counter, local_key);
     85 
     86         uint ii = index*4;
     87         scratch[ii]   = _curand_uniform(result.x);
     88         scratch[ii+1] = _curand_uniform(result.y);
     89         scratch[ii+2] = _curand_uniform(result.z);
     90         scratch[ii+3] = _curand_uniform(result.w);
     91       });
     92 
     93     vector_t<Arity> answer;
     94 
     95     iterate<Arity>([&](uint index) {
     96         answer.values[index] = scratch[index];
     97       });
     98 
     99     return answer;
    100   }


The iterate is template meta programming, compile time unrolling ? 

::

    In [10]: (np.arange(1,20) + 4 - 1)//4
    Out[10]: array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5])



sizeof(curandStatePhilox4_32_10) 64 : BUT SEEMS NOT REALLY NEEDED AS NO INIT
-------------------------------------------------------------------------------

/usr/local/cuda/include/curand_philox4x32_x.h::

    092 struct curandStatePhilox4_32_10 {

     93    uint4 ctr;                      // 16 
     94    uint4 output;                   // 16   < also used to fake 1-by-1 when actually 4-by-4   
     95    uint2 key;                      //  8
     96    unsigned int STATE;             //  4  < 0,1,2,3,(4): used to fake 1-by-1 when actually 4-by-4 

     97    int boxmuller_flag;              // 4  < used by curand_normal faking 1-by-1 when 2-by-2
     98    int boxmuller_flag_double;       // 4  
     99    float boxmuller_extra;           // 4 
    100    double boxmuller_extra_double;   // 8 
    101 };                                // ------
    102                                       64    total bytes

                                              24    (16+8 uint4 ctr, uint2 key)  counters  

                                              20    used to fake 1-by-1 when actually 4-by-4 for curand_uniform
                                              20    used to fake 1-by-1 when actually 2-by-2 for curand_normal



curand_kernel.h::

     140 struct curandStateXORWOW {
     141     unsigned int d, v[5];
     142     int boxmuller_flag;
     143     int boxmuller_flag_double;
     144     float boxmuller_extra;
     145     double boxmuller_extra_double;
     146 };


boxuller fields only used for gaussian/normal 
-------------------------------------------------

::

    342 QUALIFIERS float curand_normal(curandStatePhilox4_32_10_t *state)
    343 {
    344     if(state->boxmuller_flag != EXTRA_FLAG_NORMAL) {
    345         unsigned int x, y;
    346         x = curand(state);
    347         y = curand(state);
    348         float2 v = _curand_box_muller(x, y);
    349         state->boxmuller_extra = v.y;
    350         state->boxmuller_flag = EXTRA_FLAG_NORMAL;
    351         return v.x;
    352     }
    353     state->boxmuller_flag = 0;
    354     return state->boxmuller_extra;
    355 }

    /// AHHA : the boxmuller_extra and boxmuller_flag is again
    ///        making something that naturally gives 2 normally 
    ///        distrib values look like it gives 1 without 
    ///        ... and costs 20 bytes for this "fib"

    402 QUALIFIERS float2 curand_normal2(curandStateXORWOW_t *state)
    403 {
    404     return curand_box_muller(state);
    405 }

    151 template <typename R>
    152 QUALIFIERS float2 curand_box_muller(R *state)
    153 {
    154     float2 result;
    155     unsigned int x = curand(state);
    156     unsigned int y = curand(state);
    157     result = _curand_box_muller(x, y);
    158     return result;
    159 }

     69 QUALIFIERS float2 _curand_box_muller(unsigned int x, unsigned int y)
     70 {
     71     float2 result;
     72     float u = x * CURAND_2POW32_INV + (CURAND_2POW32_INV/2);
     73     float v = y * CURAND_2POW32_INV_2PI + (CURAND_2POW32_INV_2PI/2);
     74 #if __CUDA_ARCH__ > 0
     75     float s = sqrtf(-2.0f * logf(u));
     76     __sincosf(v, &result.x, &result.y);
     77 #else
     78     float s = sqrtf(-2.0f * logf(u));
     79     result.x = sinf(v);
     80     result.y = cosf(v);
     81 #endif
     82     result.x *= s;
     83     result.y *= s;
     84     return result;
     85 }







::

    1012 QUALIFIERS void curand_init(unsigned long long seed,
    1013                                  unsigned long long subsequence,
    1014                                  unsigned long long offset,
    1015                                  curandStatePhilox4_32_10_t *state)
    1016 {
    1017     state->ctr = make_uint4(0, 0, 0, 0);
    1018     state->key.x = (unsigned int)seed;
    1019     state->key.y = (unsigned int)(seed>>32);
    1020     state->STATE = 0;
    1021     state->boxmuller_flag = 0;
    1022     state->boxmuller_flag_double = 0;
    1023     state->boxmuller_extra = 0.f;
    1024     state->boxmuller_extra_double = 0.;
    1025     skipahead_sequence(subsequence, state);
    1026     skipahead(offset, state);
    1027 }

    0961 QUALIFIERS void skipahead(unsigned long long n, curandStatePhilox4_32_10_t *state)
     962 {
     963     state->STATE += (n & 3);
     964     n /= 4;
     965     if( state->STATE > 3 ){
     966         n += 1;
     967         state->STATE -= 4;
     968     }
     969     Philox_State_Incr(state, n);
     970     state->output = curand_Philox4x32_10(state->ctr,state->key);
     971 }
    /// looks to be fiddling to enable generator that returns sets of four random uint 
    /// to look like it can be skipped ahead not in steps of four by item by item 
    /// [n & 3 is 0,1,2,3 only, whats range if STATE? 0,1,2,3 only ? ]

    106 QUALIFIERS void Philox_State_Incr(curandStatePhilox4_32_10_t* s, unsigned long long n)
    107 {
    108    unsigned int nlo = (unsigned int)(n);
    109    unsigned int nhi = (unsigned int)(n>>32);
    110 
    111    s->ctr.x += nlo;
    112    if( s->ctr.x < nlo )
    113       nhi++;
    114 
    115    s->ctr.y += nhi;
    116    if(nhi <= s->ctr.y)
    117       return;
    118    if(++s->ctr.z) return;
    119    ++s->ctr.w;
    120 }


    170 QUALIFIERS uint4 curand_Philox4x32_10( uint4 c, uint2 k)
    171 {
    172    c = _philox4x32round(c, k);                           // 1 
    173    k.x += PHILOX_W32_0; k.y += PHILOX_W32_1;
    174    c = _philox4x32round(c, k);                           // 2
    175    k.x += PHILOX_W32_0; k.y += PHILOX_W32_1;
    176    c = _philox4x32round(c, k);                           // 3 
    177    k.x += PHILOX_W32_0; k.y += PHILOX_W32_1;
    178    c = _philox4x32round(c, k);                           // 4 
    179    k.x += PHILOX_W32_0; k.y += PHILOX_W32_1;
    180    c = _philox4x32round(c, k);                           // 5 
    181    k.x += PHILOX_W32_0; k.y += PHILOX_W32_1;
    182    c = _philox4x32round(c, k);                           // 6 
    183    k.x += PHILOX_W32_0; k.y += PHILOX_W32_1;
    184    c = _philox4x32round(c, k);                           // 7 
    185    k.x += PHILOX_W32_0; k.y += PHILOX_W32_1;
    186    c = _philox4x32round(c, k);                           // 8 
    187    k.x += PHILOX_W32_0; k.y += PHILOX_W32_1;
    188    c = _philox4x32round(c, k);                           // 9 
    189    k.x += PHILOX_W32_0; k.y += PHILOX_W32_1;
    190    return _philox4x32round(c, k);                        // 10
    191 }

* Notice arg structs used as workspace




/usr/local/cuda/include/curand_kernel.h::

    255 QUALIFIERS float curand_uniform(curandStatePhilox4_32_10_t *state)
    256 {
    257    return _curand_uniform(curand(state));
    258 }





/usr/local/cuda/include/curand_uniform.h::

     69 QUALIFIERS float _curand_uniform(unsigned int x)
     70 {
     71     return x * CURAND_2POW32_INV + (CURAND_2POW32_INV/2.0f);
     72 }





     878 QUALIFIERS unsigned int curand(curandStatePhilox4_32_10_t *state)
     879 {
     880     // Maintain the invariant: output[STATE] is always "good" and
     881     //  is the next value to be returned by curand.
     882     unsigned int ret;
     883     switch(state->STATE++){
     884     default:
     885         ret = state->output.x;
     886         break;
     887     case 1:
     888         ret = state->output.y;
     889         break;
     890     case 2:
     891         ret = state->output.z;
     892         break;
     893     case 3:
     894         ret = state->output.w;
     895         break;
     896     }
     897     if(state->STATE == 4){
     898         Philox_State_Incr(state);
     899         state->output = curand_Philox4x32_10(state->ctr,state->key);
     900         state->STATE = 0;
     901     }
     902     return ret;
     903 }


/home/blyth/env/cuda/curand-done-right/curanddr.hxx::

    076   template<int Arity>
     77   __device__ vector_t<Arity> uniforms(uint4 counter, uint key) {
     78     enum { n_blocks = (Arity + 4 - 1)/4 };
     79 
     80     float scratch[n_blocks * 4];
     81 
     82     iterate<n_blocks>([&](uint index) {
     83         uint2 local_key{key, index};
     84         uint4 result = curand_Philox4x32_10(counter, local_key);
     85 
     86         uint ii = index*4;
     87         scratch[ii]   = _curand_uniform(result.x);
     88         scratch[ii+1] = _curand_uniform(result.y);
     89         scratch[ii+2] = _curand_uniform(result.z);
     90         scratch[ii+3] = _curand_uniform(result.w);
     91       });
     92 
     93     vector_t<Arity> answer;
     94 
     95     iterate<Arity>([&](uint index) {
     96         answer.values[index] = scratch[index];
     97       });
     98 
     99     return answer;
    100   }



::

     905 /**
     906  * \brief Return tuple of 4 32-bit pseudorandoms from a Philox4_32_10 generator.
     907  *
     908  * Return 128 bits of pseudorandomness from the Philox4_32_10 generator in \p state,
     909  * increment position of generator by four.
     910  *
     911  * \param state - Pointer to state to update
     912  *
     913  * \return 128-bits of pseudorandomness as a uint4, all bits valid to use.
     914  */
     915 
     916 QUALIFIERS uint4 curand4(curandStatePhilox4_32_10_t *state)
     917 {
     918     uint4 r;
     919 
     920     uint4 tmp = state->output;
     921     Philox_State_Incr(state);
     922     state->output= curand_Philox4x32_10(state->ctr,state->key);
     923     switch(state->STATE){
     924     case 0:
     925         return tmp;
     926     case 1:
     927         r.x = tmp.y;
     928         r.y = tmp.z;
     929         r.z = tmp.w;
     930         r.w = state->output.x;
     931         break;
     932     case 2:
     933         r.x = tmp.z;
     934         r.y = tmp.w;
     935         r.z = state->output.x;
     936         r.w = state->output.y;
     937         break;
     938     case 3:
     939         r.x = tmp.w;
     940         r.y = state->output.x;
     941         r.z = state->output.y;
     942         r.w = state->output.z;
     943         break;
     944     default:
     945         // NOT possible but needed to avoid compiler warnings
     946         return tmp;
     947     }
     948     return r;
     949 }


::

    P[blyth@localhost include]$ pwd
    /usr/local/cuda/include
    P[blyth@localhost include]$ grep curand_Philox4x32_10 *.h
    curand_kernel.h:        state->output = curand_Philox4x32_10(state->ctr,state->key);
    curand_kernel.h:    state->output= curand_Philox4x32_10(state->ctr,state->key);
    curand_kernel.h:    state->output = curand_Philox4x32_10(state->ctr,state->key);
    curand_kernel.h:    state->output = curand_Philox4x32_10(state->ctr,state->key);
    curand_philox4x32_x.h:QUALIFIERS uint4 curand_Philox4x32_10( uint4 c, uint2 k)
    P[blyth@localhost include]$ 



How to make normal curand API with curandState use this ? 
------------------------------------------------------------

Could construct::

   curandStatePhilox4_32_10_OpticksLite
   {
        uint4 ctr ; 
        uint2 key ; 
   };

   float4 curand_uniform4( curandStatePhilox4_32_10_OpticksLite* state )
   {
       uint4 result = curand_Philox4x32_10(state->ctr, state->key);  

       // increment counter leaving the two uint of the key for user, eg eventID and photonID
       if(++state->ctr.x==0) 
       if(++state->ctr.y==0) 
       if(++state->ctr.z==0) 
       ++state->ctr.w ; 

       return _curand_uniform4(answer) ; 
   }



    255 QUALIFIERS float curand_uniform(curandStatePhilox4_32_10_t *state)
    256 {
    257    return _curand_uniform(curand(state));
    258 }

::

    287 static __forceinline__ __device__ void simulate( const uint3& launch_idx, const uint3& dim, quad2* prd )
    288 {
    289     sevent* evt = params.evt ;
    290     if (launch_idx.x >= evt->num_photon) return;
    291 
    292     unsigned idx = launch_idx.x ;  // aka photon_idx
    293     unsigned genstep_idx = evt->seed[idx] ;
    294     const quad6& gs = evt->genstep[genstep_idx] ;
    295 
    296     qsim* sim = params.sim ;
    297 
    298 //#define OLD_WITHOUT_SKIPAHEAD 1
    299 #ifdef OLD_WITHOUT_SKIPAHEAD
    300     curandState rng = sim->rngstate[idx] ;
    301 #else
    302     curandState rng ;
    303     sim->rng->get_rngstate_with_skipahead( rng, sim->evt->index, idx );
    304 #endif
    305 
    306 

::

     53 inline QRNG_METHOD void qrng::get_rngstate_with_skipahead(curandStateXORWOW& rng, unsigned event_idx, unsigned photon_idx )
     54 {
     55     unsigned long long skipahead_ = skipahead_event_offset*event_idx ;
     56     rng = *(rng_states + photon_idx) ;
     57     skipahead( skipahead_, &rng );
     58 }



