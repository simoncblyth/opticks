cerenkov_debugging_and_matching 
==========================================

summary 
---------

* chi2 comparison between cks:G4Cerenkov_modifiedTest.cc and qu:QCtxTest.cc:K "cerenkov_photon" is poor
  although the distributions look reasonable::

  ARG=8 ipython -i wavelength_cfplot.py

* examining the plot in energy with RINDEX bins shows most of 
  the largest chi2 contributions come from close to bin edges 

* the very largest chi2 contib is just in the region where the 
  distribution vs energy is the steepest at close to 7.3 eV
  (this is also where most of the 109/1M deviants are from) 

  * so chief suspect is the "energy" interpolation in 1nm wavelength bins procedure 
  * clearly the energies corresponding to 1nm wavelength bin edges do not 
    correspond to the RINDEX energy bin edges : to the interpolation 
    can clearly suffer  

  * DONE: CUDA port of G4PhysicsVector::Value, in QUDARap/QProp/qprop

  * DONE: create RINDEX QProp/qprop from the original energy RINDEX 
    and use that in QSimTest rather than using the standard bndlib to 
    facilitate moving to HD rindex 


* random aligned comparison using the same random sequence (1M,16,16)  
  cks:G4Cerenkov_modifiedTest.cc and qu:QCtxTest.cc:K "cerenkov_photon" 
  give chi2 zero with 109/1M wavelength deviants ( > 1e-4 nm) 

::

    In [3]: 1. - 109./1e6, 109./1e6                                                                                                                                                           
    Out[3]: (0.999891, 0.000109)

      
  * examining dumps of the first ~15 deviants suggests the cause is cut-edges and float/double difference
  * so the GPU implementation effectively reproduces the G4Cerenkov_modified one **WHEN USING SAME RANDOMS**
  * more deviation tends to happen with energy sample close to 7.3eV : that in on the RINDEX peak

* comparing 1M samples of cks:G4Cerenkov_modifiedTest.cc against itself with different seeds gives chi2/ndf 1.03

* comparing 1M samples of cks:G4Cerenkov_modifiedTest.cc against itself (seeds 1 and 2) with FLOAT_TEST restriction s
  still shows OK chi2/ndf 1.03::

  ARG=10 ipython -i wavelength_cfplot.py

* trying G4Cerenkov_modifiedTest with FLOAT_TEST restricting the rejection sampling to float precision 
  still gives poor chi2 comparing to cks : so the explanation is not all float/double 

* statistically comparing QSimTest.cc:K against itself by flipping randoms u -> 1-u shows poor chi2
  with similar type of disagreement "infinity wiggle pattern" to that between cks and qu with multiple 2nm bins 
  below and then above in range 250-350 nm:: 

  ARG=9 ipython -i wavelength_cfplot.py
  
* HMM how to explain lumps on the wavelength distrib ? Need to favor one range above another ?
  Does the RINDEX mountain cause a "lobe" effect where the random sampling will flip one side or the 
  other yieldins broadly similar RINDEX on either side ?


* ana/ck.py 2d scatter plots of:

  1. (u0,u1) 
  2. (en,ct) with overlay of BetaInverse/rindex bins with drawstyle="steps-post"
     this makes the relationship of (en,ct) with the rindex bins very plain 

* (July 25, 2021) random aligned comparison using QSim::cerenkov_photon_enprop using the new QProp/qprop 
  interpolation functionality gives perfect match chi2 zero with zero (nm 1e-4) deviants out of 1M, max deviation is ~7e-5::

  In [6]: np.abs(wa-wb).max()
  Out[6]: 7.145989718537749e-05

  ARG=12 ipython -i wavelength.py
  ARG=12 ipython -i wavelength_cfplot.py

    In [10]: h_wab =  np.histogram(wab)

    In [11]: h_wab[0]
    Out[11]: array([379545, 326842, 196659,  71002,  18147,   5159,   1876,    592,    155,     23])

    In [12]: h_wab[1]
    Out[12]: array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    In [13]: h_wab[1]*1e6
    Out[13]: array([ 0.   ,  7.146, 14.292, 21.438, 28.584, 35.73 , 42.876, 50.022, 57.168, 64.314, 71.46 ])



* (July 25, 2021) wow : statistical comparison still poor chi2/ndf:15 even with enprop 

   ARG=13 ipython -i wavelength_cfplot.py 

* using cks FLOAT_TEST does not change thr poor chi2/ndf:15

   ARG=14 ipython -i wavelength_cfplot.py 


* compare 2d : energy vs ri : no apparent offsets : red and blue scatter all on top of each other

  ARG=14 ipython -i wavelength.py   

* what is left to try ? 

  * check cks FLOAT_TEST still using lots of double, eg OpticksDebug collection  
  * doing it in double precision on GPU !

* **resorting to double precision gets cerenkov chi2 to match**::

  ARG=15 ipython -i wavelength.py   
  ARG=15 ipython -i wavelength_cfplot.py   

* (circa July 27, 2021) ana/rindex.py suggests inverse transform approach can be used with Cerenkov generation

* (Aug 1, 2021) ana/ckn.py GetAverageNumberOfPhotons_s2 avoids small -ve numPhotons close to rindex peak and is a simpler algorithm  
  implemented in C++ in opticks/examples/Geant4/CerenkovStandalone/G4Cerenkov_modified.cc G4Cerenkov_modified::GetAverageNumberOfPhotons_s2


* (Sep 15, 2021) QUDARap/QCerenkov,QCK provides both energy sampling and icdf lookup *energy_lookup* *energy_sample*

  * the number of BetaInverse and energy edges is configurable, currently using (2000,2000) but suspect that 
    it is not necessary to use so many the c2 was similar with much less, just increased to try to 
    see the impact on c2 

  * tests/QCKTest.py compares histograms of energy distribs : the distribs broadly match at various BetaInverse
    but the chi2/ndf is not good (~6) 

  * all c2 poppies are just below rindex edges : this suggests a problem with the cumulative integral 
    (the old issue of the partial integrals slightly exceeding full bin integrals was never fixed, so this
    looks) 

  * rindex is piecewise linear and transforming that into s2 is rather simple suggesting that 
    might be able to use sympy to arrive at a piecewise analytic expression for the cumulative integral and hence the CDF

    * **did this in ana/piecewise.py : but it fails to integrate (could try doing it bin-by-bin and adding up)**
    * that would have the advantage of being an analytic function rather than estimates at edges
    * https://docs.sympy.org/latest/modules/functions/elementary.html#piecewise
    * can use the analytic answer to check the numerical approximations and find where the problem is perhaps
    * more ambitiously could find a way to export the sympy analytic integral function into generated CUDA/C code 

* Sep 16, 2021 QUDARap/QCerenkov switching to `getS2Integral_WithCut_` which fixes the problem of cutting rhs triangle into trapezoid
  greatly improves chi2/ndf for BetaInverse 1->1.4 but still chi2/ndf is not good for BetaInverse=1.5,1.6 
  with the same single bin causing the troubles 
 
  * BetaInverse 1.7 : c2poppy at upper edge of allowed : hmm need to change energy range of comparison
    as the permitted energy range shrinks otherwise the bins are effectively getting huge  
 
  * DONE: plotting s2 together with chi2 
  * DONE: plotting s2cn (the CDF) together with chi2 
  * DONE: check with divide_edges to avoid bin migration effects


Comparing with s2 things look pretty good with sampling and lookup taking the same form 
until get down to fractions of a photon, when things fall to pieces. 
But the chi2/ndf, aint that good : mostly from extremes. 
 
+----------+----------+----------+----------+----------+----------+----------+
|        bi|     c2sum|       ndf|       c2p|       emn|       emx|     avgph|
+==========+==========+==========+==========+==========+==========+==========+
|    1.0000|   80.2314|   99.0000|    0.8104|    1.5500|   15.5000|  293.2454|
+----------+----------+----------+----------+----------+----------+----------+
|    1.0500|   93.2445|   99.0000|    0.9419|    1.5500|   15.5000|  270.4324|
+----------+----------+----------+----------+----------+----------+----------+
|    1.1000|   95.2011|   99.0000|    0.9616|    1.5500|   15.5000|  246.5068|
+----------+----------+----------+----------+----------+----------+----------+
|    1.1500|   81.5968|   99.0000|    0.8242|    1.5500|   15.5000|  221.4688|
+----------+----------+----------+----------+----------+----------+----------+
|    1.2000|   98.5169|   99.0000|    0.9951|    1.5500|   15.5000|  195.3183|
+----------+----------+----------+----------+----------+----------+----------+
|    1.2500|  121.6160|   99.0000|    1.2284|    1.5500|   15.5000|  168.0553|
+----------+----------+----------+----------+----------+----------+----------+
|    1.3000|  120.5107|   99.0000|    1.2173|    1.5500|   15.5000|  139.6798|
+----------+----------+----------+----------+----------+----------+----------+
|    1.3500|   95.7665|   99.0000|    0.9673|    1.5500|   15.5000|  110.1918|
+----------+----------+----------+----------+----------+----------+----------+
|    1.4000|  111.3815|   99.0000|    1.1251|    1.5500|   15.5000|   79.5914|
+----------+----------+----------+----------+----------+----------+----------+
|    1.4500|  300.7510|   99.0000|    3.0379|    1.5500|   15.5000|   47.8784|
+----------+----------+----------+----------+----------+----------+----------+
|    1.5000|  294.1136|   99.0000|    2.9708|    3.1039|    9.9670|   28.8255|
+----------+----------+----------+----------+----------+----------+----------+
|    1.5500|  646.9821|   91.0000|    7.1097|    4.6586|    9.5747|   16.5780|
+----------+----------+----------+----------+----------+----------+----------+
|    1.6000|  464.0584|   77.0000|    6.0267|    5.7805|    9.2567|    9.1903|
+----------+----------+----------+----------+----------+----------+----------+
|    1.6500|  739.3216|   99.0000|    7.4679|    7.4770|    8.9440|    5.3722|
+----------+----------+----------+----------+----------+----------+----------+
|    1.7000| 1272.2710|   99.0000|   12.8512|    7.5725|    8.6777|    2.8450|
+----------+----------+----------+----------+----------+----------+----------+
|    1.7500|  134.5231|   99.0000|    1.3588|    7.6681|    8.4287|    0.9755|
+----------+----------+----------+----------+----------+----------+----------+
|    1.7920|50563.7852|   99.0000|  510.7453|    7.7477|    7.8092|    0.0015|
+----------+----------+----------+----------+----------+----------+----------+



BetaInverse 1.45 slightly non-monotonic::

    (lldb) p item
    (int) $1 = 1134
    (lldb) f 4
    frame #4: 0x00000001001745f7 libQUDARap.dylib`double NP::pdomain<double>(this=0x00000001014026a0, value=0.99214520962982877, item=1134, dump=false) const at NP.hh:1075
       1072	            const T y1 = vv[nj*(i+1)+jval] ;
       1073	
       1074	            const T dy = y1 - y0 ;  
    -> 1075	            assert( dy >= zero );   // must be monotonic for this to make sense
       1076	
       1077	            if( y0 <= yv && yv < y1 )
       1078	            { 
    (lldb) p dy
    (const double) $2 = -0.0000030085381202971107
    (lldb) p y1
    (const double) $3 = 0.97945642413092681
    (lldb) p y0
    (const double) $4 = 0.97945943266904711
    (lldb) p jval
    (unsigned int) $5 = 2

    NP::pdomain ERROR : non-monotonic dy less than zero   i  1257 x0  10.321961 x1  10.328939 y0   0.979459 y1   0.979456 yv   0.986214 dy  -0.000003
    NP::pdomain ERROR : non-monotonic dy less than zero   i  1257 x0  10.321961 x1  10.328939 y0   0.979459 y1   0.979456 yv   0.988575 dy  -0.000003
    NP::pdomain ERROR : non-monotonic dy less than zero   i  1257 x0  10.321961 x1  10.328939 y0   0.979459 y1   0.979456 yv   0.999524 dy  -0.000003
    NP::pdomain ERROR : non-monotonic dy less than zero   i  1257 x0  10.321961 x1  10.328939 y0   0.979459 y1   0.979456 yv   0.998180 dy  -0.000003
    NP::pdomain ERROR : non-monotonic dy less than zero   i  1257 x0  10.321961 x1  10.328939 y0   0.979459 y1   0.979456 yv   0.991240 dy  -0.000003
    NP::pdomain ERROR : non-monotonic dy less than zero   i  1257 x0  10.321961 x1  10.328939 y0   0.979459 y1   0.979456 yv   0.994275 dy  -0.000003
    NP::pdomain ERROR : non-monotonic dy less than zero   i  1257 x0  10.321961 x1  10.328939 y0   0.979459 y1   0.979456 yv   0.995296 dy  -0.000003
    NP::pdomain ERROR : non-monotonic dy less than zero   i  1257 x0  10.321961 x1  10.328939 y0   0.979459 y1   0.979456 yv   0.980150 dy  -0.000003
    NP::pdomain ERROR : non-monotonic dy less than zero   i  1257 x0  10.321961 x1  10.328939 y0   0.979459 y1   0.979456 yv   0.982399 dy  -0.000003
    NP::pdomain ERROR : non-monotonic dy less than zero   i  1257 x0  10.321961 x1  10.328939 y0   0.979459 y1   0.979456 yv   0.988238 dy  -0.000003
    NP::pdomain ERROR : non-monotonic dy less than zero   i  1257 x0  10.321961 x1  10.328939 y0   0.979459 y1   0.979456 yv   0.992084 dy  -0.000003
    NP::pdomain ERROR : non-monotonic dy less than zero   i  1257 x0  10.321961 x1  10.328939 y0   0.979459 y1   0.979456 yv   0.983232 dy  -0.000003
    NP::pdomain ERROR : non-monotonic dy less than zero   i  1257 x0  10.321961 x1  10.328939 y0   0.979459 y1   0.979456 yv   0.991422 dy  -0.000003
    NP::pdomain ERROR : non-monotonic dy less than zero   i  1257 x0  10.321961 x1  10.328939 y0   0.979459 y1   0.979456 yv   0.995707 dy  -0.000003



What are the relative merits of: 

* inverse transform sampling 
* rejection sampling 

* https://math.stackexchange.com/questions/1311282/why-use-rejection-sampling-in-monte-carlo-simulations


See Also
-----------

ana/ck.py 

ana/ckn.py
    comparing G4Cerenkov GetAverageNumberOfPhotons implementations

ana/rindex.py 
    attempt to form a Cerenkov 2d ICDF, using the s2sliver integrals

ana/rindex.sh 
    run rindex.py with ipython




mul 1 getS2Integral_Cumulative comparing with sympy ana/piecewise.py discrepancy at <0.1 photon level
-------------------------------------------------------------------------------------------------------

::

    epsilon:qudarap blyth$ ipython -i tests/QCerenkovTest.py 
    -rw-r--r--  1 blyth  wheel  10496 Sep 21 11:28 /tmp/blyth/opticks/QCerenkovTest/test_getS2Integral_Cumulative/s2c.npy
    -rw-r--r--  1 blyth  wheel  10496 Sep 21 11:28 /tmp/blyth/opticks/QCerenkovTest/test_getS2Integral_Cumulative/s2cn.npy
    -rw-r--r--  1 blyth  wheel  200 Sep 21 11:28 /tmp/blyth/opticks/QCerenkovTest/test_getS2Integral_Cumulative/bis.npy
    -rw-r--r--  1 blyth  wheel  2720 Sep 21 11:29 /tmp/ana/piecewise/scan.npy
    -rw-r--r--  1 blyth  wheel  200 Sep 21 11:29 /tmp/ana/piecewise/bis.npy
    INFO:__main__: sa:p_s2c a:(9, 18, 2) sb:b_s2c b:(9, 18, 8) 
    BetaInverse :     1.0000  dfmax  5.684e-14  df [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.] 
    BetaInverse :     1.1000  dfmax  5.684e-14  df [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.] 
    BetaInverse :     1.2000  dfmax  1.705e-13  df [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.] 
    BetaInverse :     1.3000  dfmax    2.7e-13  df [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.] 
    BetaInverse :     1.4000  dfmax  1.563e-13  df [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.] 
    BetaInverse :     1.5000  dfmax    0.02608  df [0.    0.    0.    0.    0.    0.    0.    0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.026 0.026] 
    BetaInverse :     1.6000  dfmax     0.0978  df [0.    0.    0.    0.    0.    0.    0.    0.    0.012 0.012 0.014 0.014 0.074 0.074 0.074 0.098 0.098 0.098] 
    BetaInverse :     1.7000  dfmax    0.06422  df [0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.042 0.042 0.064 0.064 0.064 0.064] 
    BetaInverse :     1.7920  dfmax  1.571e-05  df [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.] 


* dont understand why getting differences up to almost 0.1 photon 

  * a bug or numerical imprecision ?

* as s2 is piecewise linear would expect the numerical integral to be at something like 1e-4 level to the analytic one 
  (hmm lots of flops both numerically and analytic) 
 


::

    epsilon:qudarap blyth$ ipython -i tests/QCerenkovTest.py 
    -rw-r--r--  1 blyth  wheel  20288 Sep 21 11:58 /tmp/blyth/opticks/QCerenkovTest/test_getS2Integral_Cumulative/s2c.npy
    -rw-r--r--  1 blyth  wheel  20288 Sep 21 11:58 /tmp/blyth/opticks/QCerenkovTest/test_getS2Integral_Cumulative/s2cn.npy
    -rw-r--r--  1 blyth  wheel  200 Sep 21 11:58 /tmp/blyth/opticks/QCerenkovTest/test_getS2Integral_Cumulative/bis.npy
    -rw-r--r--  1 blyth  wheel  5168 Sep 21 11:56 /tmp/ana/piecewise/scan.npy
    -rw-r--r--  1 blyth  wheel  200 Sep 21 11:56 /tmp/ana/piecewise/bis.npy
    INFO:__main__: sa:p_s2c a:(9, 35, 2) sb:b_s2c b:(9, 35, 8) 
    BetaInverse :     1.0000  dfmax     0.1478  df [0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.001 0.002 0.022 0.042 0.042 0.042 0.045 0.049 0.049 0.05  0.073 0.096 0.096 0.096 0.102 0.109 0.117 0.126 0.137 0.148
     0.148 0.148] 
    BetaInverse :     1.1000  dfmax     0.1788  df [0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.001 0.003 0.027 0.051 0.051 0.051 0.055 0.059 0.06  0.06  0.088 0.116 0.116 0.116 0.124 0.132 0.142 0.152 0.166 0.179
     0.179 0.179] 
    BetaInverse :     1.2000  dfmax     0.2128  df [0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.002 0.003 0.032 0.061 0.061 0.061 0.065 0.07  0.071 0.071 0.105 0.138 0.138 0.138 0.147 0.157 0.169 0.181 0.197 0.213
     0.213 0.213] 
    BetaInverse :     1.3000  dfmax     0.2497  df [0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.002 0.004 0.037 0.071 0.071 0.071 0.077 0.083 0.083 0.084 0.123 0.162 0.162 0.162 0.173 0.184 0.198 0.213 0.231 0.25
     0.25  0.25 ] 
    BetaInverse :     1.4000  dfmax     0.2896  df [0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.002 0.004 0.043 0.082 0.082 0.082 0.089 0.096 0.096 0.097 0.142 0.188 0.188 0.188 0.201 0.213 0.23  0.247 0.268 0.29
     0.29  0.29 ] 
    BetaInverse :     1.5000  dfmax     0.3106  df [0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.003 0.005 0.05  0.095 0.095 0.095 0.102 0.11  0.111 0.112 0.164 0.216 0.216 0.216 0.23  0.245 0.264 0.283 0.308 0.311
     0.311 0.311] 
    BetaInverse :     1.6000  dfmax     0.1925  df [0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.012 0.012 0.012 0.014 0.014 0.014 0.014 0.072 0.131 0.131 0.131 0.148 0.164 0.186 0.192 0.192 0.192
     0.192 0.192] 
    BetaInverse :     1.7000  dfmax    0.07452  df [0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.042 0.042 0.043 0.061 0.075 0.075 0.075 0.075 0.075
     0.075 0.075] 
    BetaInverse :     1.7920  dfmax  1.571e-05  df [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.] 


     


Checking GetAverageNumberOfPhotons_s2
-----------------------------------------

* opticks/ana/ckn.py results

  * ckn.scan[:,1]  GetAverageNumberOfPhotons_asis      : reproduces the ASIS C++ in python 
  * ckn.scan[:,2]  GetAverageNumberOfPhotons_s2
  * ckn.scan[:,3]  GetAverageNumberOfPhotons_s2messy


* opticks/examples/Geant4/CerenkovStandalone/G4Cerenkov_modified.cc results 

  * /tmp/G4Cerenkov_modifiedTest/scan_GetAverageNumberOfPhotons.npy
  * ckn.scan2[:,1] : ASIS GetAverageNumberOfPhotons 
  * ckn.scan2[:,2] : GetAverageNumberOfPhotons_s2 


::

    In [3]: np.abs( ckn.scan[:,1] - ckn.scan2[:,1] ).max()    # ckn.py:GetAverageNumberOfPhotons_asis reproduces ASIS C++
    Out[3]: 1.5631940186722204e-13

    In [5]: np.abs( ckn.scan[:,2] - ckn.scan[:,3] ).max()     # _s2 and _s2messy are effectively the same 
    Out[5]: 8.526512829121202e-14

    In [8]: np.abs( ckn.scan[:,2] - ckn.scan2[:,2] ).max()    # python and C++ implementations give same results
    Out[8]: 1.5631940186722204e-13

    In [9]: np.abs( ckn.scan[:,3] - ckn.scan2[:,2] ).max()
    Out[9]: 1.5631940186722204e-13


::

    In [1]: np.c_[ckn.scan, ckn.scan2]                                                                                                                                                              
    Out[1]:  #ckn.py  _asis     _s2       _s2messy C++ "cks"  ASIS      _s2          
    array([[  1.    , 293.2454, 293.2454, 293.2454,   1.    , 293.2454, 293.2454],
           [  1.01  , 288.7704, 288.7704, 288.7704,   1.01  , 288.7704, 288.7704],
           [  1.02  , 284.2508, 284.2508, 284.2508,   1.02  , 284.2508, 284.2508],
           [  1.03  , 279.6867, 279.6867, 279.6867,   1.03  , 279.6867, 279.6867],
           [  1.04  , 275.0781, 275.0781, 275.0781,   1.04  , 275.0781, 275.0781],
           [  1.05  , 270.4249, 270.4249, 270.4249,   1.05  , 270.4249, 270.4249],
           [  1.06  , 265.7272, 265.7272, 265.7272,   1.06  , 265.7272, 265.7272],
           [  1.07  , 260.985 , 260.985 , 260.985 ,   1.07  , 260.985 , 260.985 ],
           [  1.08  , 256.1982, 256.1982, 256.1982,   1.08  , 256.1982, 256.1982],
           [  1.09  , 251.367 , 251.367 , 251.367 ,   1.09  , 251.367 , 251.367 ],
           [  1.1   , 246.4912, 246.4912, 246.4912,   1.1   , 246.4912, 246.4912],
           [  1.11  , 241.5708, 241.5708, 241.5708,   1.11  , 241.5708, 241.5708],
           [  1.12  , 236.606 , 236.606 , 236.606 ,   1.12  , 236.606 , 236.606 ],
           [  1.13  , 231.5966, 231.5966, 231.5966,   1.13  , 231.5966, 231.5966],
           [  1.14  , 226.5427, 226.5427, 226.5427,   1.14  , 226.5427, 226.5427],
           [  1.15  , 221.4442, 221.4442, 221.4442,   1.15  , 221.4442, 221.4442],
           [  1.16  , 216.3012, 216.3012, 216.3012,   1.16  , 216.3012, 216.3012],
           [  1.17  , 211.1137, 211.1137, 211.1137,   1.17  , 211.1137, 211.1137],
           [  1.18  , 205.8817, 205.8817, 205.8817,   1.18  , 205.8817, 205.8817],
           [  1.19  , 200.6051, 200.6051, 200.6051,   1.19  , 200.6051, 200.6051],
           [  1.2   , 195.2841, 195.2841, 195.2841,   1.2   , 195.2841, 195.2841],
           [  1.21  , 189.9185, 189.9185, 189.9185,   1.21  , 189.9185, 189.9185],
           [  1.22  , 184.5083, 184.5083, 184.5083,   1.22  , 184.5083, 184.5083],
           [  1.23  , 179.0536, 179.0536, 179.0536,   1.23  , 179.0536, 179.0536],
           [  1.24  , 173.5544, 173.5544, 173.5544,   1.24  , 173.5544, 173.5544],
           [  1.25  , 168.0107, 168.0107, 168.0107,   1.25  , 168.0107, 168.0107],
           [  1.26  , 162.4225, 162.4225, 162.4225,   1.26  , 162.4225, 162.4225],
           [  1.27  , 156.7897, 156.7897, 156.7897,   1.27  , 156.7897, 156.7897],
           [  1.28  , 151.1124, 151.1124, 151.1124,   1.28  , 151.1124, 151.1124],
           [  1.29  , 145.3906, 145.3906, 145.3906,   1.29  , 145.3906, 145.3906],
           [  1.3   , 139.6242, 139.6242, 139.6242,   1.3   , 139.6242, 139.6242],
           [  1.31  , 133.8133, 133.8133, 133.8133,   1.31  , 133.8133, 133.8133],
           [  1.32  , 127.9579, 127.9579, 127.9579,   1.32  , 127.9579, 127.9579],
           [  1.33  , 122.0579, 122.0579, 122.0579,   1.33  , 122.0579, 122.0579],
           [  1.34  , 116.1135, 116.1135, 116.1135,   1.34  , 116.1135, 116.1135],
           [  1.35  , 110.1245, 110.1245, 110.1245,   1.35  , 110.1245, 110.1245],
           [  1.36  , 104.0909, 104.0909, 104.0909,   1.36  , 104.0909, 104.0909],
           [  1.37  ,  98.0129,  98.0129,  98.0129,   1.37  ,  98.0129,  98.0129],
           [  1.38  ,  91.8903,  91.8903,  91.8903,   1.38  ,  91.8903,  91.8903],
           [  1.39  ,  85.7232,  85.7232,  85.7232,   1.39  ,  85.7232,  85.7232],
           [  1.4   ,  79.5115,  79.5115,  79.5115,   1.4   ,  79.5115,  79.5115],
           [  1.41  ,  73.2554,  73.2554,  73.2554,   1.41  ,  73.2554,  73.2554],
           [  1.42  ,  66.9547,  66.9547,  66.9547,   1.42  ,  66.9547,  66.9547],
           [  1.43  ,  60.6094,  60.6094,  60.6094,   1.43  ,  60.6094,  60.6094],
           [  1.44  ,  54.2197,  54.2197,  54.2197,   1.44  ,  54.2197,  54.2197],
           [  1.45  ,  47.7854,  47.7854,  47.7854,   1.45  ,  47.7854,  47.7854],
           [  1.46  ,  42.8926,  43.0029,  43.0029,   1.46  ,  42.8926,  43.0029],
           [  1.47  ,  38.924 ,  39.1805,  39.1805,   1.47  ,  38.924 ,  39.1805],
           [  1.48  ,  35.0169,  35.386 ,  35.386 ,   1.48  ,  35.0169,  35.386 ],
           [  1.49  ,  31.4464,  31.9007,  31.9007,   1.49  ,  31.4464,  31.9007],
           [  1.5   ,  28.2344,  28.7516,  28.7516,   1.5   ,  28.2344,  28.7516],
           [  1.51  ,  25.2077,  25.8716,  25.8716,   1.51  ,  25.2077,  25.8716],
           [  1.52  ,  22.6185,  23.2019,  23.2019,   1.52  ,  22.6185,  23.2019],
           [  1.53  ,  20.154 ,  20.738 ,  20.738 ,   1.53  ,  20.154 ,  20.738 ],
           [  1.54  ,  17.5415,  18.4959,  18.4959,   1.54  ,  17.5415,  18.4959],
           [  1.55  ,  15.38  ,  16.4809,  16.4809,   1.55  ,  15.38  ,  16.4809],
           [  1.56  ,  13.3531,  14.6795,  14.6795,   1.56  ,  13.3531,  14.6795],
           [  1.57  ,  11.4056,  13.0446,  13.0446,   1.57  ,  11.4056,  13.0446],
           [  1.58  ,   9.7689,  11.567 ,  11.567 ,   1.58  ,   9.7689,  11.567 ],
           [  1.59  ,   8.4461,  10.2436,  10.2436,   1.59  ,   8.4461,  10.2436],
           [  1.6   ,   7.4401,   9.0712,   9.0712,   1.6   ,   7.4401,   9.0712],
           [  1.61  ,   6.7542,   8.0471,   8.0471,   1.61  ,   6.7542,   8.0471],
           [  1.62  ,   6.3178,   7.1958,   7.1958,   1.62  ,   6.3178,   7.1958],
           [  1.63  ,   5.6532,   6.5362,   6.5362,   1.63  ,   5.6532,   6.5362],
           [  1.64  ,   5.0593,   5.9089,   5.9089,   1.64  ,   5.0593,   5.9089],
           [  1.65  ,   4.5366,   5.3135,   5.3135,   1.65  ,   4.5366,   5.3135],
           [  1.66  ,   4.0859,   4.7491,   4.7491,   1.66  ,   4.0859,   4.7491],
           [  1.67  ,   3.5389,   4.2152,   4.2152,   1.67  ,   3.5389,   4.2152],
           [  1.68  ,   2.9289,   3.7108,   3.7108,   1.68  ,   2.9289,   3.7108],
           [  1.69  ,   2.3748,   3.2354,   3.2354,   1.69  ,   2.3748,   3.2354],
           [  1.7   ,   1.8773,   2.7883,   2.7883,   1.7   ,   1.8773,   2.7883],
           [  1.71  ,   1.4369,   2.3692,   2.3692,   1.71  ,   1.4369,   2.3692],
           [  1.72  ,   1.054 ,   1.9773,   1.9773,   1.72  ,   1.054 ,   1.9773],
           [  1.73  ,   0.7293,   1.6122,   1.6122,   1.73  ,   0.7293,   1.6122],
           [  1.74  ,   0.4632,   1.2735,   1.2735,   1.74  ,   0.4632,   1.2735],
           [  1.75  ,   0.2563,   0.9605,   0.9605,   1.75  ,   0.2563,   0.9605],
           [  1.76  ,   0.109 ,   0.673 ,   0.673 ,   1.76  ,   0.109 ,   0.673 ],
           [  1.77  ,   0.022 ,   0.4103,   0.4103,   1.77  ,   0.022 ,   0.4103],
           [  1.78  ,  -0.0042,   0.172 ,   0.172 ,   1.78  ,  -0.0042,   0.172 ],
           [  1.79  ,  -0.0479,   0.0094,   0.0094,   1.79  ,  -0.0479,   0.0094],
           [  1.8   ,   0.    ,   0.    ,   0.    ,   1.8   ,   0.    ,   0.    ],
           [  1.81  ,   0.    ,   0.    ,   0.    ,   1.81  ,   0.    ,   0.    ],
           [  1.82  ,   0.    ,   0.    ,   0.    ,   1.82  ,   0.    ,   0.    ],



    In [3]: np.allclose(ckn.scan[:,1], ckn.scan2[:,1])                                                                                                                                              
    Out[3]: True

    In [4]: np.allclose(ckn.scan[:,2], ckn.scan2[:,2])                                                                                                                                              
    Out[4]: True

    In [5]: np.allclose(ckn.scan[:,3], ckn.scan2[:,2])                                                                                                                                              
    Out[5]: True






G4PhysicsVector::Value
------------------------

* g4-cls G4PhysicsVector

::

    498 G4double G4PhysicsVector::Value(G4double theEnergy, size_t& lastIdx) const
    499 {
    500   G4double y;
    501   if(theEnergy <= edgeMin) {
    502     lastIdx = 0;
    503     y = dataVector[0];
    504   } else if(theEnergy >= edgeMax) {
    505     lastIdx = numberOfNodes-1;
    506     y = dataVector[lastIdx];
    507   } else {
    508     lastIdx = FindBin(theEnergy, lastIdx);
    509     y = Interpolation(lastIdx, theEnergy);
    510   }
    511   return y;
    512 }

    215 inline size_t G4PhysicsVector::FindBin(G4double e, size_t idx) const
    216 { 
    217   size_t id = idx;
    218   if(e < binVector[1]) { 
    219     id = 0;
    220   } else if(e >= binVector[numberOfNodes-2]) {
    221     id = numberOfNodes - 2;
    222   } else if(idx >= numberOfNodes || e < binVector[idx]
    223             || e > binVector[idx+1]) {
    224     id = FindBinLocation(e);
    225   }
    226   return id;
    227 }

    inline
     G4double G4PhysicsVector::LinearInterpolation(size_t idx, G4double e) const
    { 
      // Linear interpolation is used to get the value. Before this method
      // is called it is ensured that the energy is inside the bin
      // 0 < idx < numberOfNodes-1
      
      return dataVector[idx] +
             ( dataVector[idx + 1]-dataVector[idx] ) * (e - binVector[idx])
             /( binVector[idx + 1]-binVector[idx] );
    }



* https://bitbucket.org/simoncblyth/chroma/src/master/chroma/cuda/interpolate.h


::

    __device__ float
    interp(float x, int n, float *xp, float *fp)
    {
        int lower = 0;
        int upper = n-1;

        if (x <= xp[lower])
        return fp[lower];

        if (x >= xp[upper])
        return fp[upper];

        while (lower < upper-1)
        {
        int half = (lower+upper)/2;

        if (x < xp[half])
            upper = half;
        else
            lower = half;
        }

        float df = fp[upper] - fp[lower];
        float dx = xp[upper] - xp[lower];

        return fp[lower] + df*(x-xp[lower])/dx;
    }


Developed this into::

   NP::Interp
   NPY::Interp 
   qudarap/QProp.cc
   qudarap/qprop.h  qprop::interpolate 



G4Cerenkov
--------------

::

    251   G4double Pmin = Rindex->GetMinLowEdgeEnergy();
    252   G4double Pmax = Rindex->GetMaxLowEdgeEnergy();
    253   G4double dp = Pmax - Pmin;
    254 
    255   G4double nMax = Rindex->GetMaxValue();
    256 
    257   G4double BetaInverse = 1./beta;
    258 
    259   G4double maxCos = BetaInverse / nMax;
    260   G4double maxSin2 = (1.0 - maxCos) * (1.0 + maxCos);
    261 


    280       do {
    281          rand = G4UniformRand();
    282          sampledEnergy = Pmin + rand * dp;      

    ///   start with flat energy sampling 

    283          sampledRI = Rindex->Value(sampledEnergy);
    284          cosTheta = BetaInverse / sampledRI;
    286          sin2Theta = (1.0 - cosTheta)*(1.0 + cosTheta);

    ///  across entire bins of rindex only one value for : ct, s2 

    285 
    287          rand = G4UniformRand();

    ///  another sampling dimension 

    288 
    290       } while (rand*maxSin2 > sin2Theta);
    291 



TO CHECK
----------

* the deviation wiggles are happening all within a huge RINDEX bins, 
  so reconsider the algorithm in the light of constant RINDEX

* compare RINDEX interpolation results across energy bin edges to see 
  in detail how the "step" between bins differs with Geant4 and texture access ?  

* create energy binned texture and use it from an qctx::en_cerenkov_photon 



qudarap/tests/QCtxTest  QCtxTest::rng_sequence
-------------------------------------------------

qudarap/QCtx.cu::

     15 __global__ void _QCtx_rng_sequence(qctx* ctx, float* rs, unsigned num_items )
     16 {
     17     unsigned id = blockIdx.x*blockDim.x + threadIdx.x;
     18     if (id >= num_items) return;
     19     curandState rng = *(ctx->r + id) ; 
     20     float u = curand_uniform(&rng) ;
     21     if(id % 100000 == 0) printf("//_QCtx_rng_sequence id %d u %10.4f    \n", id, u  );
     22     rs[id] = u ; 
     23 }

* currently just collects the first random float from each photon slot  


thrustrap/tests/TRngBufTest
-----------------------------

* collects 16*16 double randoms for each photon slot

::

    In [1]: a = np.load("/tmp/blyth/opticks/TRngBufTest_0.npy")                                                                                                                                    

    In [2]: a                                                                                                                                                                                      
    Out[2]: 
    array([[[0.74 , 0.438, 0.517, ..., 0.547, 0.653, 0.23 ],
            [0.339, 0.761, 0.546, ..., 0.855, 0.489, 0.189],
            [0.507, 0.021, 0.958, ..., 0.748, 0.488, 0.318],
            ...,
            [0.153, 0.327, 0.894, ..., 0.94 , 0.946, 0.197],
            [0.856, 0.657, 0.063, ..., 0.624, 0.968, 0.532],
            [0.902, 0.429, 0.674, ..., 0.598, 0.82 , 0.145]],

           [[0.921, 0.46 , 0.333, ..., 0.825, 0.527, 0.93 ],
            [0.163, 0.785, 0.942, ..., 0.492, 0.543, 0.934],
            [0.479, 0.449, 0.126, ..., 0.042, 0.379, 0.715],

    In [5]: a.shape                                                                                                                                                                                
    Out[5]: (10000, 16, 16)




compare those
------------------

::

    In [21]: a[:100,0,0]                                                                                                                                                                           
    Out[21]: 
    array([0.74 , 0.921, 0.039, 0.969, 0.925, 0.446, 0.667, 0.11 , 0.47 , 0.513, 0.776, 0.295, 0.714, 0.359, 0.681, 0.292, 0.319, 0.811, 0.154, 0.445, 0.208, 0.611, 0.307, 0.416, 0.234, 0.879, 0.646,
           0.926, 0.579, 0.554, 0.356, 0.723, 0.278, 0.619, 0.588, 0.375, 0.24 , 0.415, 0.094, 0.633, 0.285, 0.779, 0.213, 0.413, 0.033, 0.536, 0.721, 0.355, 0.253, 0.985, 0.92 , 0.187, 0.182, 0.598,
           0.708, 0.042, 0.731, 0.94 , 0.843, 0.612, 0.267, 0.021, 0.833, 0.722, 0.609, 0.63 , 0.53 , 0.813, 0.059, 0.48 , 0.991, 0.879, 1.   , 0.207, 0.437, 0.373, 0.447, 0.238, 0.034, 0.731, 0.494,
           0.303, 0.809, 0.129, 0.783, 0.073, 0.124, 0.223, 0.742, 0.627, 0.153, 0.012, 0.173, 0.478, 0.805, 0.687, 0.302, 0.808, 0.407, 0.751])

    In [22]: r[:100]                                                                                                                                                                               
    Out[22]: 
    array([0.74 , 0.921, 0.039, 0.969, 0.925, 0.446, 0.667, 0.11 , 0.47 , 0.513, 0.776, 0.295, 0.714, 0.359, 0.681, 0.292, 0.319, 0.811, 0.154, 0.445, 0.208, 0.611, 0.307, 0.416, 0.234, 0.879, 0.646,
           0.926, 0.579, 0.554, 0.356, 0.723, 0.278, 0.619, 0.588, 0.375, 0.24 , 0.415, 0.094, 0.633, 0.285, 0.779, 0.213, 0.413, 0.033, 0.536, 0.721, 0.355, 0.253, 0.985, 0.92 , 0.187, 0.182, 0.598,
           0.708, 0.042, 0.731, 0.94 , 0.843, 0.612, 0.267, 0.021, 0.833, 0.722, 0.609, 0.63 , 0.53 , 0.813, 0.059, 0.48 , 0.991, 0.879, 1.   , 0.207, 0.437, 0.373, 0.447, 0.238, 0.034, 0.731, 0.494,
           0.303, 0.809, 0.129, 0.783, 0.073, 0.124, 0.223, 0.742, 0.627, 0.153, 0.012, 0.173, 0.478, 0.805, 0.687, 0.302, 0.808, 0.407, 0.751], dtype=float32)

    In [23]:              



cerenkov generation check using random alignment
---------------------------------------------------

* getting geant4 to use the same randoms in cks opticks/examples/Geant4/CerenkovStandalone/G4Cerenkov_modifiedTest.cc
  would be real helpful for debugging why the cerenkov wavelength histogram sample matching is poor

* potential cause : float vs double, if so need to drill down as to exactly where
 


Getting G4Cerenkov_modified to use precooked randoms using OpticksRandom
-----------------------------------------------------------------------------

::

    G4Cerenkov_modifiedTest::PSDI [BetaInverse_1.500_step_length_100.000_SKIP_CONTINUE]
     i 0 rand0    0.74022 Pmin/eV    1.55000 Pmax/eV   15.50000 dp    0.00001 sampledEnergy/eV   11.87606 sampledRI    1.45360 cosTheta    1.03192 sin2Theta   -0.06486 rand1    0.43845
     i 0 rand0    0.51701 Pmin/eV    1.55000 Pmax/eV   15.50000 dp    0.00001 sampledEnergy/eV    8.76233 sampledRI    1.68320 cosTheta    0.89116 sin2Theta    0.20583 rand1    0.15699



Use same precooked randoms from python
----------------------------------------

::

    epsilon:CerenkovStandalone blyth$ ipython -i cks.py 
    idx     0 u0    0.74022 sampledEnergy   11.87606 sampledRI    1.45360 cosTheta    1.03192 sin2Theta   -0.06486 u1    0.43845
    idx     0 u0    0.51701 sampledEnergy    8.76233 sampledRI    1.68320 cosTheta    0.89116 sin2Theta    0.20583 u1    0.15699

    idx     1 u0    0.92099 sampledEnergy   14.39786 sampledRI    1.45360 cosTheta    1.03192 sin2Theta   -0.06486 u1    0.46036
    idx     1 u0    0.33346 sampledEnergy    6.20182 sampledRI    1.61849 cosTheta    0.92679 sin2Theta    0.14107 u1    0.37252

    idx     2 u0    0.03902 sampledEnergy    2.09434 sampledRI    1.48406 cosTheta    1.01074 sin2Theta   -0.02160 u1    0.25021
    idx     2 u0    0.18448 sampledEnergy    4.12356 sampledRI    1.52616 cosTheta    0.98286 sin2Theta    0.03399 u1    0.96242
    idx     2 u0    0.52055 sampledEnergy    8.81174 sampledRI    1.67328 cosTheta    0.89644 sin2Theta    0.19639 u1    0.93996
    idx     2 u0    0.83058 sampledEnergy   13.13657 sampledRI    1.45360 cosTheta    1.03192 sin2Theta   -0.06486 u1    0.40973
    idx     2 u0    0.08162 sampledEnergy    2.68863 sampledRI    1.49337 cosTheta    1.00444 sin2Theta   -0.00890 u1    0.80677
    idx     2 u0    0.69529 sampledEnergy   11.24924 sampledRI    1.45360 cosTheta    1.03192 sin2Theta   -0.06486 u1    0.61771
    idx     2 u0    0.25633 sampledEnergy    5.12587 sampledRI    1.57064 cosTheta    0.95502 sin2Theta    0.08793 u1    0.21368

    idx     3 u0    0.96896 sampledEnergy   15.06703 sampledRI    1.45360 cosTheta    1.03192 sin2Theta   -0.06486 u1    0.49474
    idx     3 u0    0.67338 sampledEnergy   10.94366 sampledRI    1.45360 cosTheta    1.03192 sin2Theta   -0.06486 u1    0.56277
    idx     3 u0    0.12019 sampledEnergy    3.22671 sampledRI    1.50301 cosTheta    0.99800 sin2Theta    0.00400 u1    0.97649
    idx     3 u0    0.13583 sampledEnergy    3.44485 sampledRI    1.50864 cosTheta    0.99427 sin2Theta    0.01142 u1    0.58897
    idx     3 u0    0.49062 sampledEnergy    8.39412 sampledRI    1.75709 cosTheta    0.85368 sin2Theta    0.27122 u1    0.32844

    idx     4 u0    0.92514 sampledEnergy   14.45571 sampledRI    1.45360 cosTheta    1.03192 sin2Theta   -0.06486 u1    0.05301
    idx     4 u0    0.16310 sampledEnergy    3.82528 sampledRI    1.51846 cosTheta    0.98784 sin2Theta    0.02417 u1    0.88969



    2021-07-15 20:09:06.370 INFO  [795925] [QCtx::generate_cerenkov_photon@277] [ num_photon 100
    //QCtx_generate_cerenkov_photon num_photon 100 
    //qctx::cerenkov_photon id 0 u0     0.7402 sampledRI     1.4536 cosTheta     1.0319 sin2Theta    -0.0649 u1     0.4385 
    //qctx::cerenkov_photon id 0 u0     0.5170 sampledRI     1.6834 cosTheta     0.8910 sin2Theta     0.2060 u1     0.1570 
    //_QCtx_generate_cerenkov_photon id 0 





    In [26]: np.set_printoptions(precision=6)

    In [27]: a[0,0,:4]
    Out[27]: array([0.740219, 0.438451, 0.517013, 0.156989])


    In [15]: Pmin = 1.55                                                                                                                                                                           

    In [16]: Pmax = 15.5                                                                                                                                                                           

    In [17]: e = Pmin + u0*(Pmax - Pmin)                                                                                                                                                           

    In [18]: e                                                                                                                                                                                     
    Out[18]: 11.876059997081757

    rindex = np.load("/tmp/G4Cerenkov_modifiedTest/BetaInverse_1.500_step_length_100.000_SKIP_CONTINUE/RINDEX.npy") 

    rindex_ = lambda ev:np.interp( ev, rindex[:,0], rindex[:,1] )  


    In [28]: rindex_(11.87606)                                                                                                                                                                     
    Out[28]: 1.4536

    In [29]: 1.5/rindex_(11.87606)                                                                                                                                                                 
    Out[29]: 1.0319207484865163


    In [30]: cosTheta_ = lambda ev:1.5/rindex_(ev)                                                                                                                                                 

    In [31]: cosTheta_(e)                                                                                                                                                                          
    Out[31]: 1.0319207484865163

    In [32]: e                                                                                                                                                                                     
    Out[32]: 11.876059997081757

    In [33]: rindex_(e)                                                                                                                                                                            
    Out[33]: 1.4536

    In [34]: r = rindex_(e) ; r                                                                                                                                                                    
    Out[34]: 1.4536

    In [35]: sin2Theta_ = lambda e:(1.0 - cosTheta_(e))*(1.0 + cosTheta_(e))                                                                                                                       

    In [36]: sin2Theta_(e)                                                                                                                                                                         
    Out[36]: -0.06486043115697197





::

    0258   G4double Pmin = Rindex->GetMinLowEdgeEnergy();
     259   G4double Pmax = Rindex->GetMaxLowEdgeEnergy();


     g4-cls G4MaterialPropertyVector
     g4-cls G4PhysicsOrderedFreeVector


    124 inline
    125 G4double G4PhysicsOrderedFreeVector::GetMaxLowEdgeEnergy()
    126 {
    127   return binVector.back();
    128 }
    129 
    130 inline
    131 G4double G4PhysicsOrderedFreeVector::GetMinLowEdgeEnergy()
    132 {
    133   return binVector.front();
    134 }



    079 void G4PhysicsOrderedFreeVector::InsertValues(G4double energy, G4double value)
     80 {
     81         std::vector<G4double>::iterator binLoc =
     82                  std::lower_bound(binVector.begin(), binVector.end(), energy);
     83 
     84         size_t binIdx = binLoc - binVector.begin(); // Iterator difference!
     85 
     86         std::vector<G4double>::iterator dataLoc = dataVector.begin() + binIdx;
     87 
     88         binVector.insert(binLoc, energy);
     89         dataVector.insert(dataLoc, value);
     90 
     91         ++numberOfNodes;
     92         edgeMin = binVector.front();
     93         edgeMax = binVector.back();
     94 }

     96 G4double G4PhysicsOrderedFreeVector::GetEnergy(G4double aValue)
     97 {
     98         G4double e;
     99         if (aValue <= GetMinValue()) {
    100           e = edgeMin;
    101         } else if (aValue >= GetMaxValue()) {
    102           e = edgeMax;
    103         } else {
    104           size_t closestBin = FindValueBinLocation(aValue);
    105           e = LinearInterpolationOfEnergy(aValue, closestBin);
    106     }
    107         return e;
    108 }


    231 inline
    232  G4double G4PhysicsVector::Value(G4double theEnergy) const
    233 {
    234   size_t idx=0;
    235   return Value(theEnergy, idx);
    236 }
    237 

    498 G4double G4PhysicsVector::Value(G4double theEnergy, size_t& lastIdx) const
    499 {
    500   G4double y;
    501   if(theEnergy <= edgeMin) {
    502     lastIdx = 0;
    503     y = dataVector[0];
    504   } else if(theEnergy >= edgeMax) {
    505     lastIdx = numberOfNodes-1;
    506     y = dataVector[lastIdx];
    507   } else {
    508     lastIdx = FindBin(theEnergy, lastIdx);
    509     y = Interpolation(lastIdx, theEnergy);
    510   }
    511   return y;
    512 }




One more bin edge than value ? Not in the below ? Artificial repetition of last line probably ?
--------------------------------------------------------------------------------------------------

::

    O[blyth@localhost junotop]$ cat data/Simulation/DetSim/Material/LS/RINDEX
    1.55                *eV   1.4781              
    1.79505             *eV   1.48                
    2.10499             *eV   1.4842              
    2.27077             *eV   1.4861              
    2.55111             *eV   1.4915              
    2.84498             *eV   1.4955              
    3.06361             *eV   1.4988              
    4.13281             *eV   1.5264              
    6.2                 *eV   1.6185              
    6.526               *eV   1.6176              
    6.889               *eV   1.527               
    7.294               *eV   1.5545              
    7.75                *eV   1.793               
    8.267               *eV   1.7826              
    8.857               *eV   1.6642              
    9.538               *eV   1.5545              
    10.33               *eV   1.4536              
    15.5                *eV   1.4536              
    O[blyth@localhost junotop]$ 


* ~/j/issues/material_properties_one_more_edge_than_value.rst 

Most but not all material RINDEX properties end with a duplicated value, looks like artificial duplication
to provide some value for the last edge.





cks : Three way comparison ckcf.py 
-------------------------------------


::

In [3]: a[0]                                                                                                                                                                                         
Out[3]: 
array([[  8.7623, 141.5149,   1.6834,   0.891 ],
       [  0.206 ,   0.    ,   0.    ,   1.5   ],
       [452.2491, 141.5149,   0.517 ,   0.157 ],
       [  0.    ,   0.    ,   0.    ,   0.    ]], dtype=float32)



    In [8]: b[0]                                                                                                                                                                                         
    Out[8]: 
    array([[  8.7623, 141.4969,   1.6832,   0.8912],
           [  0.2058,   0.    ,   0.    ,   1.5   ]])

    In [9]: c[0]                                                                                                                                                                                         
    Out[9]: 
    array([[  8.7623, 141.5149,   1.6832,   0.8912],
           [  0.2058,   0.    ,   0.    ,   0.    ],
           [  0.    ,   0.    ,   0.    ,   0.    ],
           [  0.    ,   0.    ,   0.    ,   0.    ]])

    In [10]: b.shape                                                                                                                                                                                     
    Out[10]: (10000, 2, 4)

    In [11]: c.shape                                                                                                                                                                                     
    Out[11]: (10000, 4, 4)

    In [12]: b[10]                                                                                                                                                                                       
    Out[12]: 
    array([[  6.6084, 187.6163,   1.597 ,   0.9392],
           [  0.1178,   0.    ,   0.    ,   1.5   ]])

    In [13]: c[10]                                                                                                                                                                                       
    Out[13]: 
    array([[  6.6084, 187.6402,   1.597 ,   0.9392],
           [  0.1178,   0.    ,   0.    ,   0.    ],
           [  0.    ,   0.    ,   0.    ,   0.    ],
           [  0.    ,   0.    ,   0.    ,   0.    ]])

    In [14]: b[100]                                                                                                                                                                                      
    Out[14]: 
    array([[  8.8041, 140.825 ,   1.6748,   0.8956],
           [  0.1979,   0.    ,   0.    ,   1.5   ]])

    In [15]: c[100]                                                                                                                                                                                      
    Out[15]: 
    array([[  8.8041, 140.8429,   1.6748,   0.8956],
           [  0.1979,   0.    ,   0.    ,   0.    ],
           [  0.    ,   0.    ,   0.    ,   0.    ],
           [  0.    ,   0.    ,   0.    ,   0.    ]])

    In [16]: b[1000]                                                                                                                                                                                     
    Out[16]: 
    array([[  7.9196, 156.5544,   1.7896,   0.8382],
           [  0.2975,   0.    ,   0.    ,   1.5   ]])

    In [17]: c[1000]                                                                                                                                                                                     
    Out[17]: 
    array([[  7.9196, 156.5743,   1.7896,   0.8382],
           [  0.2975,   0.    ,   0.    ,   0.    ],
           [  0.    ,   0.    ,   0.    ,   0.    ],
           [  0.    ,   0.    ,   0.    ,   0.    ]])


    In [4]: a[-1]                                                                                                                                                                                        
    Out[4]: 
    array([[  7.4727, 165.9384,   1.6475,   0.9105],
           [  0.171 ,   0.    ,   0.    ,   1.5   ],
           [385.6852, 165.9384,   0.4246,   0.4489],
           [  0.    ,   0.    ,   0.    ,   0.    ]], dtype=float32)

    In [18]: b[-1]                                                                                                                                                                                       
    Out[18]: 
    array([[  7.4727, 165.9173,   1.6479,   0.9102],
           [  0.1715,   0.    ,   0.    ,   1.5   ]])

    In [19]: c[-1]                                                                                                                                                                                       
    Out[19]: 
    array([[  7.4727, 165.9384,   1.6479,   0.9102],
           [  0.1715,   0.    ,   0.    ,   0.    ],
           [  0.    ,   0.    ,   0.    ,   0.    ],
           [  0.    ,   0.    ,   0.    ,   0.    ]])

    In [20]:                                                   





python energy very closely matches the G4Cerenkov_modified
------------------------------------------------------------

::


    In [15]: c[:,0,0]                                                                                                                                                                                    
    Out[15]: array([8.7623, 6.2018, 5.1259, ..., 4.111 , 7.8475, 7.4727])

    In [16]: b[:,0,0]                                                                                                                                                                                    
    Out[16]: array([8.7623, 6.2018, 5.1259, ..., 4.111 , 7.8475, 7.4727])

    In [17]: bc = b[:,0,0] - c[:,0,0]                                                                                                                                                                    

    In [18]: bc.min()                                                                                                                                                                                    
    Out[18]: -1.7763568394002505e-15

    In [19]: bc.max()                                                                                                                                                                                    
    Out[19]: 1.7763568394002505e-15






8/10k are way off::


    In [20]: np.histogram( a[:,0,0] - b[:,0,0], 100 )                                                                                                                                                    
    Out[20]: 
    (array([   1,    0,    0,    0,    0,    0,    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 9992,    0,
               0,    0,    0,    1,    0,    0,    0,    0,    0,    1,    0,    0,    0,    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    1,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    1]),
     array([-2.8501, -2.8042, -2.7584, -2.7125, -2.6667, -2.6208, -2.575 , -2.5291, -2.4833, -2.4374, -2.3916, -2.3457, -2.2999, -2.254 , -2.2082, -2.1623, -2.1165, -2.0707, -2.0248, -1.979 , -1.9331,
            -1.8873, -1.8414, -1.7956, -1.7497, -1.7039, -1.658 , -1.6122, -1.5663, -1.5205, -1.4746, -1.4288, -1.3829, -1.3371, -1.2912, -1.2454, -1.1995, -1.1537, -1.1078, -1.062 , -1.0161, -0.9703,
            -0.9244, -0.8786, -0.8327, -0.7869, -0.741 , -0.6952, -0.6493, -0.6035, -0.5576, -0.5118, -0.466 , -0.4201, -0.3743, -0.3284, -0.2826, -0.2367, -0.1909, -0.145 , -0.0992, -0.0533, -0.0075,
             0.0384,  0.0842,  0.1301,  0.1759,  0.2218,  0.2676,  0.3135,  0.3593,  0.4052,  0.451 ,  0.4969,  0.5427,  0.5886,  0.6344,  0.6803,  0.7261,  0.772 ,  0.8178,  0.8637,  0.9095,  0.9554,
             1.0012,  1.047 ,  1.0929,  1.1387,  1.1846,  1.2304,  1.2763,  1.3221,  1.368 ,  1.4138,  1.4597,  1.5055,  1.5514,  1.5972,  1.6431,  1.6889,  1.7348]))


    In [21]: deviants = np.abs( a[:,0,0] - b[:,0,0] ) > 0.001                                                                                                                                            

    In [22]: a[deviants]                                                                                                                                                                                 
    Out[22]: 
    array([[[  4.3884, 282.5663,   1.5378,   0.9754],
            [  0.0485,   0.    ,   0.    ,   1.5   ],
            [226.4955, 282.5663,   0.2035,   0.0268],
            [  0.    ,   0.    ,   0.    ,   0.    ]],

           [[  6.0812, 203.9058,   1.6132,   0.9298],
            [  0.1354,   0.    ,   0.    ,   1.5   ],
            [313.8704, 203.9058,   0.3248,   0.1889],
            [  0.    ,   0.    ,   0.    ,   0.    ]],

           [[  8.3739, 148.0784,   1.7613,   0.8516],
            [  0.2747,   0.    ,   0.    ,   1.5   ],
            [432.2036, 148.0784,   0.4892,   0.3752],
            [  0.    ,   0.    ,   0.    ,   0.    ]],

           [[  7.89  , 157.1604,   1.7902,   0.8379],
            [  0.2979,   0.    ,   0.    ,   1.5   ],
            [407.2274, 157.1604,   0.4545,   0.2969],
            [  0.    ,   0.    ,   0.    ,   0.    ]],

           [[  6.7663, 183.2618,   1.5578,   0.9629],
            [  0.0729,   0.    ,   0.    ,   1.5   ],
            [349.2272, 183.2618,   0.3739,   0.2424],
            [  0.    ,   0.    ,   0.    ,   0.    ]],

           [[  9.0393, 137.1786,   1.635 ,   0.9174],
            [  0.1583,   0.    ,   0.    ,   1.5   ],
            [466.545 , 137.1786,   0.5369,   0.5272],
            [  0.    ,   0.    ,   0.    ,   0.    ]],

           [[  5.117 , 242.3309,   1.5702,   0.9553],
            [  0.0874,   0.    ,   0.    ,   1.5   ],
            [264.1017, 242.3309,   0.2557,   0.0638],
            [  0.    ,   0.    ,   0.    ,   0.    ]],

           [[  8.4108, 147.4297,   1.7539,   0.8552],
            [  0.2686,   0.    ,   0.    ,   1.5   ],
            [434.1053, 147.4297,   0.4918,   0.8946],
            [  0.    ,   0.    ,   0.    ,   0.    ]]], dtype=float32)

    In [23]: b[deviants]                                                                                                                                                                                 
    Out[23]: 
    array([[[  7.2384, 171.2861,   1.5507,   0.9673],
            [  0.0644,   0.    ,   0.    ,   1.5   ]],

           [[  4.3465, 285.253 ,   1.5359,   0.9766],
            [  0.0462,   0.    ,   0.    ,   1.5   ]],

           [[  7.1317, 173.8487,   1.5435,   0.9718],
            [  0.0555,   0.    ,   0.    ,   1.5   ]],

           [[  7.3915, 167.7392,   1.6055,   0.9343],
            [  0.1271,   0.    ,   0.    ,   1.5   ]],

           [[  6.0449, 205.1064,   1.6116,   0.9308],
            [  0.1337,   0.    ,   0.    ,   1.5   ]],

           [[  8.7732, 141.3213,   1.681 ,   0.8923],
            [  0.2038,   0.    ,   0.    ,   1.5   ]],

           [[  7.647 , 162.1349,   1.7391,   0.8625],
            [  0.2561,   0.    ,   0.    ,   1.5   ]],

           [[  9.4831, 130.7423,   1.5633,   0.9595],
            [  0.0794,   0.    ,   0.    ,   1.5   ]]])

    In [24]:                          



Excluding the 8 deviants gives a very close energy match::


    In [27]: aa = a[np.logical_not(deviants)]                                                                                                                                                            

    In [28]: bb = b[np.logical_not(deviants)]                                                                                                                                                            

    In [29]: aa[:,0,0] - bb[:,0,0]                                                                                                                                                                       
    Out[29]: array([ 0., -0., -0., ...,  0.,  0.,  0.])

    In [30]: ab = aa[:,0,0] - bb[:,0,0]                                                                                                                                                                  

    In [31]: ab.min()                                                                                                                                                                                    
    Out[31]: -1.8984079375172769e-06

    In [32]: ab.max()                                                                                                                                                                                    
    Out[32]: 2.041459083557129e-06


    In [33]: deviants                                                                                                                                                                                    
    Out[33]: array([False, False, False, ..., False, False, False])

    In [34]: np.where(deviants)                                                                                                                                                                          
    Out[34]: (array([ 213,  817, 1351, 1902, 2236, 3114, 4812, 6139]),)




Looking at wavelength_cfplot curious that discrepancy peaks at 330nm and just prior::

    ARG=5 ipython -i wavelength_cfplot.py

What is special about there ? The rindex is close to BetaIndex in that range. So are near the threshold ?
Near threshold presumably means more samples are rejected to find a permissable energy so higher probability 
for difference from close to cuts ?


Look at rejection looping:: 

    In [60]: np.where( a_loop != b_loop )                                                                                                                                                                
    Out[60]: (array([ 213,  817, 1351, 1902, 2236, 3114, 4812, 6139]),)


    In [63]: b_loop.min(), b_loop.max()                                                                                                                                                                  
    Out[63]: (1, 42)

    In [64]: c_loop.min(), c_loop.max()                                                                                                                                                                  
    Out[64]: (1, 42)

    In [65]: a_loop.min(), a_loop.max()                                                                                                                                                                  
    Out[65]: (1, 42)



::

    In [2]: np.unique( a_loop , return_counts=True )                                                                                                                                                     
    Out[2]: 
    (array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 40, 42], dtype=int32),
     array([1912, 1521, 1204,  997,  822,  694,  540,  459,  348,  309,  232,  181,  146,  110,   87,   84,   57,   52,   51,   39,   33,   22,   18,   17,   15,   11,    8,    8,    3,    1,    2,    4,
               1,    5,    2,    1,    1,    2,    1]))

    In [3]: np.unique( b_loop , return_counts=True )                                                                                                                                                     
    Out[3]: 
    (array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 40, 42], dtype=int32),
     array([1913, 1521, 1203,  999,  821,  694,  538,  459,  348,  309,  233,  181,  145,  110,   87,   84,   57,   52,   53,   40,   32,   22,   17,   17,   15,   11,    8,    8,    3,    1,    2,    4,
               1,    5,    2,    1,    1,    2,    1]))

    In [4]: np.unique( c_loop , return_counts=True )                                                                                                                                                     
    Out[4]: 
    (array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 40, 42]),
     array([1913, 1521, 1203,  999,  821,  694,  538,  459,  348,  309,  233,  181,  145,  110,   87,   84,   57,   52,   53,   40,   32,   22,   17,   17,   15,   11,    8,    8,    3,    1,    2,    4,
               1,    5,    2,    1,    1,    2,    1]))



Using hc_eVnm/15.5 hc_eVnm/1.55 rather than the close 80. 800. gets the wavelengths very close::


    In [1]: a[:,0,1]                                                                                                                                                                                     
    Out[1]: array([141.4969, 199.9157, 241.8792, ..., 301.5883, 157.9929, 165.9173], dtype=float32)

    In [2]: b[:,0,1]                                                                                                                                                                                     
    Out[2]: array([141.4969, 199.9157, 241.8792, ..., 301.5883, 157.9929, 165.9173])

    In [3]: c[:,0,1]                                                                                                                                                                                     
    Out[3]: array([141.4969, 199.9157, 241.8792, ..., 301.5883, 157.9929, 165.9173])


Huh ... wow. Getting precisely the same range gets aligned 10k running to match precisely.
Looks like perfect match in aligned 10k running, "chi2" zero.

BUT : comparing non-aligned 1M samples in wavelength_cfplot.py still get bad chi2:: 

    ARG=6 ipython -i wavelength_cfplot.py

* so the problem is finding the cause of extreme fragility in some regions 

DONE: cooked (1M, 16,16) randoms in qctx and check using that 



::

    epsilon:ana blyth$ ARG=6 ipython -i wavelength.py 

    In [1]: wa                                                                                                                                                                         
    Out[1]: array([141.497, 199.916, 241.879, ..., 159.323, 234.615, 163.714])

    In [2]: wb                                                                                                                                                                         
    Out[2]: array([141.497, 199.916, 241.879, ..., 159.323, 234.615, 163.714], dtype=float32)

    In [3]: wab = np.abs(wa - wb)                   


    In [7]: np.where( wab > 1e-4 )                                                                                                                                                     
    Out[7]: 
    (array([ 11264,  35904,  37969,  57112,  69670,  69761,  70952,  77035,  77501,  81643, 105525, 106447, 120673, 132599, 148922, 172494, 176234, 195173, 198025, 203785, 205613, 211717, 212547, 255649,
            256969, 258248, 262498, 264640, 276232, 286639, 322426, 359703, 370264, 371478, 387119, 387515, 394157, 394967, 400192, 400739, 401858, 404798, 406048, 414252, 420658, 439106, 441025, 452366,
            456014, 486019, 502414, 502648, 506567, 507707, 512139, 517370, 530743, 538441, 541806, 545645, 561119, 561918, 567720, 569773, 571278, 572149, 585078, 599422, 602754, 607974, 611131, 641770,
            647384, 671380, 674386, 675539, 678265, 678858, 691496, 701648, 705779, 712878, 740118, 741500, 768456, 773621, 776522, 787463, 795257, 799561, 807476, 814903, 823558, 842623, 847850, 884195,
            888557, 896674, 928872, 931275, 932706, 937849, 939082, 950967, 953438, 972014, 977639, 987635, 991957]),)

    In [8]:                                                                                                                                                                            

    In [8]:                                                                                                                                                                            

    In [8]: np.where( wab > 1e-4 )[0].shape                                                                                                                                            
    Out[8]: (109,)


109/1M deviants, mostly way off::

    In [10]: dev = np.where( wab > 1e-4 )                                                                                                                                              
    In [12]: np.c_[wa[dev],wb[dev]]                                                                                                                                                    
    Out[12]: 
    array([[160.056, 169.41 ],
           [206.383, 169.873],
           [144.358, 169.576],
           [208.995, 169.812],
           [180.896, 140.112],
           [158.538, 165.391],
           [131.332, 138.672],
           [138.121, 168.259],
           [149.294, 160.428],
           [131.209, 244.854],
           [159.717, 235.169],
           [159.907, 162.419],
           [128.353, 150.621],
           [125.189, 160.932],
           [143.04 , 227.904],


Run just the 109 in 1M deviants : where rand1*maxSin2 - sin2Theta gets very close to zero float/double difference can fall either way::

    G4Cerenkov_modifiedTest::PSDI rnd seq or seqmask constrains the number of photon indices to 109
    G4Cerenkov_modifiedTest::PSDI [BetaInverse_1.500_override_fNumPhotons_109_SKIP_CONTINUE]
     i      0 seqidx   11264 Pmin/eV    1.55000 Pmax/eV   15.50000 dp/eV   13.95000 maxSin2    0.30012
     tc      1 u0    0.16277 eV    3.82060 ri    1.51834 ct    0.98792 s2    0.02401 rand1*maxSin2    0.21889 rand1*maxSin2 - sin2Theta    0.19488 loop Y
     tc      2 u0    0.41352 eV    7.31858 ri    1.56736 ct    0.95702 s2    0.08410 rand1*maxSin2    0.08429 rand1*maxSin2 - sin2Theta    0.00019 loop Y
     tc      3 u0    0.72366 eV   11.64508 ri    1.45360 ct    1.03192 s2   -0.06486 rand1*maxSin2    0.10068 rand1*maxSin2 - sin2Theta    0.16554 loop Y
     tc      4 u0    0.60160 eV    9.94235 ri    1.50299 ct    0.99801 s2    0.00397 rand1*maxSin2    0.29380 rand1*maxSin2 - sin2Theta    0.28983 loop Y
     tc      5 u0    0.27100 eV    5.33041 ri    1.57976 ct    0.94951 s2    0.09842 rand1*maxSin2    0.13884 rand1*maxSin2 - sin2Theta    0.04042 loop Y
     tc      6 u0    0.78736 eV   12.53364 ri    1.45360 ct    1.03192 s2   -0.06486 rand1*maxSin2    0.02526 rand1*maxSin2 - sin2Theta    0.09012 loop Y
     tc      7 u0    0.44418 eV    7.74628 ri    1.79106 ct    0.83749 s2    0.29860 rand1*maxSin2    0.06050 rand1*maxSin2 - sin2Theta   -0.23811 loop N

    epsilon:qudarap blyth$ PINDEX=11264 QCtxTest 
    //_QCtx_generate_cerenkov_photon id 0 
    //qctx::cerenkov_photon id 11264 loop   1 u0    0.16277 ri    1.51834 ct    0.98792 s2    0.02401 u_mxs2_s2    0.19488 
    //qctx::cerenkov_photon id 11264 loop   2 u0    0.41352 ri    1.56754 ct    0.95691 s2    0.08432 u_mxs2_s2   -0.00002 


     i      1 seqidx   35904 Pmin/eV    1.55000 Pmax/eV   15.50000 dp/eV   13.95000 maxSin2    0.30012
     tc      1 u0    0.65123 eV   10.63469 ri    1.45360 ct    1.03192 s2   -0.06486 rand1*maxSin2    0.03328 rand1*maxSin2 - sin2Theta    0.09814 loop Y
     tc      2 u0    0.76881 eV   12.27492 ri    1.45360 ct    1.03192 s2   -0.06486 rand1*maxSin2    0.15708 rand1*maxSin2 - sin2Theta    0.22194 loop Y
     tc      3 u0    0.17597 eV    4.00473 ri    1.52309 ct    0.98484 s2    0.03010 rand1*maxSin2    0.21724 rand1*maxSin2 - sin2Theta    0.18714 loop Y
     tc      4 u0    0.41209 eV    7.29864 ri   *1.55693*ct    0.96344 s2    0.07179 rand1*maxSin2    0.07216 rand1*maxSin2 - sin2Theta    0.00037 loop Y
     tc      5 u0    0.10045 eV    2.95123 ri    1.49710 ct    1.00193 s2   -0.00387 rand1*maxSin2    0.05687 rand1*maxSin2 - sin2Theta    0.06074 loop Y
     tc      6 u0    0.83771 eV   13.23605 ri    1.45360 ct    1.03192 s2   -0.06486 rand1*maxSin2    0.14369 rand1*maxSin2 - sin2Theta    0.20855 loop Y
     tc      7 u0    0.71750 eV   11.55914 ri    1.45360 ct    1.03192 s2   -0.06486 rand1*maxSin2    0.11813 rand1*maxSin2 - sin2Theta    0.18299 loop Y
     tc      8 u0    0.85208 eV   13.43656 ri    1.45360 ct    1.03192 s2   -0.06486 rand1*maxSin2    0.00927 rand1*maxSin2 - sin2Theta    0.07413 loop Y
     tc      9 u0    0.12719 eV    3.32431 ri    1.50553 ct    0.99633 s2    0.00733 rand1*maxSin2    0.14743 rand1*maxSin2 - sin2Theta    0.14010 loop Y
     tc     10 u0    0.29738 eV    5.69840 ri    1.59615 ct    0.93976 s2    0.11685 rand1*maxSin2    0.24078 rand1*maxSin2 - sin2Theta    0.12393 loop Y
     tc     11 u0    0.90887 eV   14.22870 ri    1.45360 ct    1.03192 s2   -0.06486 rand1*maxSin2    0.11009 rand1*maxSin2 - sin2Theta    0.17495 loop Y
     tc     12 u0    0.99516 eV   15.43251 ri    1.45360 ct    1.03192 s2   -0.06486 rand1*maxSin2    0.22275 rand1*maxSin2 - sin2Theta    0.28761 loop Y
     tc     13 u0    0.31953 eV    6.00748 ri    1.60992 ct    0.93172 s2    0.13189 rand1*maxSin2    0.04203 rand1*maxSin2 - sin2Theta   -0.08987 loop N

    epsilon:qudarap blyth$ PINDEX=35904 QCtxTest 
    //qctx::cerenkov_photon id 35904 loop   1 u0    0.65123 ri    1.45360 ct    1.03192 s2   -0.06486 u_mxs2_s2    0.09814 
    //qctx::cerenkov_photon id 35904 loop   2 u0    0.76881 ri    1.45360 ct    1.03192 s2   -0.06486 u_mxs2_s2    0.22194 
    //qctx::cerenkov_photon id 35904 loop   3 u0    0.17597 ri    1.52309 ct    0.98484 s2    0.03010 u_mxs2_s2    0.18714 
    //qctx::cerenkov_photon id 35904 loop   4 u0    0.41209 ri   *1.55731*ct    0.96320 s2    0.07224 u_mxs2_s2   -0.00008 

                WHY THE DIFFERENT RI ?  RI BIN EDGE ? 

        [  1.55  ,   1.4781, 800.    ,   1.4781],
        [  1.7951,   1.48  , 690.7886,   1.48  ],
        [  2.105 ,   1.4842, 589.0764,   1.4842],
        [  2.2708,   1.4861, 546.0703,   1.4861],
        [  2.5511,   1.4915, 486.0629,   1.4915],
        [  2.845 ,   1.4955, 435.8554,   1.4955],
        [  3.0636,   1.4988, 404.7513,   1.4988],
        [  4.1328,   1.5264, 300.038 ,   1.5264],
        [  6.2   ,   1.6185, 200.    ,   1.6185],
        [  6.526 ,   1.6176, 190.0092,   1.6176],
        [  6.889 ,   1.527 , 179.9971,   1.527 ],
       *[  7.294 ,   1.5545, 170.0027,   1.5545],*
        [  7.75  ,   1.793 , 160.    ,   1.793 ],
        [  8.267 ,   1.7826, 149.994 ,   1.7826],
        [  8.857 ,   1.6642, 140.0023,   1.6642],
        [  9.538 ,   1.5545, 130.0063,   1.5545],
        [ 10.33  ,   1.4536, 120.0387,   1.4536],
        [ 15.5   ,   1.4536,  80.    ,   1.4536]])




    //_QCtx_generate_cerenkov_photon id 0 
    //qctx::cerenkov_photon id 35904 u0     0.6512 sampledRI     1.4536 cosTheta     1.0319 sin2Theta    -0.0649 u1     0.1109 
    //qctx::cerenkov_photon id 35904 u0     0.7688 sampledRI     1.4536 cosTheta     1.0319 sin2Theta    -0.0649 u1     0.5234 
    //qctx::cerenkov_photon id 35904 u0     0.1760 sampledRI     1.5231 cosTheta     0.9848 sin2Theta     0.0301 u1     0.7238 
    //qctx::cerenkov_photon id 35904 u0     0.4121 sampledRI     1.5573 cosTheta     0.9632 sin2Theta     0.0722 u1     0.2404 
    //_QCtx_generate_cerenkov_photon id 100000 
    //_QCtx_generate_cerenkov_photon id 200000 



     i      2 seqidx   37969 Pmin/eV    1.55000 Pmax/eV   15.50000 dp/eV   13.95000 maxSin2    0.30012
     tc      1 u0    0.95147 eV   14.82301 ri    1.45360 ct    1.03192 s2   -0.06486 rand1*maxSin2    0.21398 rand1*maxSin2 - sin2Theta    0.27884 loop Y
     tc      2 u0    0.77318 eV   12.33582 ri    1.45360 ct    1.03192 s2   -0.06486 rand1*maxSin2    0.01378 rand1*maxSin2 - sin2Theta    0.07865 loop Y
     tc      3 u0    0.08577 eV    2.74643 ri    1.49416 ct    1.00391 s2   -0.00783 rand1*maxSin2    0.15107 rand1*maxSin2 - sin2Theta    0.15890 loop Y
     tc      4 u0    0.11242 eV    3.11830 ri    1.50021 ct    0.99986 s2    0.00028 rand1*maxSin2    0.09906 rand1*maxSin2 - sin2Theta    0.09878 loop Y
     tc      5 u0    0.01777 eV    1.79792 ri    1.48004 ct    1.01349 s2   -0.02716 rand1*maxSin2    0.10909 rand1*maxSin2 - sin2Theta    0.13624 loop Y
     tc      6 u0    0.98852 eV   15.33979 ri    1.45360 ct    1.03192 s2   -0.06486 rand1*maxSin2    0.24481 rand1*maxSin2 - sin2Theta    0.30967 loop Y
     tc      7 u0    0.78130 eV   12.44908 ri    1.45360 ct    1.03192 s2   -0.06486 rand1*maxSin2    0.16663 rand1*maxSin2 - sin2Theta    0.23149 loop Y
     tc      8 u0    0.23526 eV    4.83185 ri    1.55754 ct    0.96305 s2    0.07253 rand1*maxSin2    0.09766 rand1*maxSin2 - sin2Theta    0.02513 loop Y

     tc      9 u0    0.41300 eV    7.31142 ri    1.56361 ct    0.95932 s2    0.07971 rand1*maxSin2    0.07975 rand1*maxSin2 - sin2Theta    0.00004 loop Y

     tc     10 u0    0.27359 eV    5.36665 ri    1.58137 ct    0.94854 s2    0.10026 rand1*maxSin2    0.21361 rand1*maxSin2 - sin2Theta    0.11335 loop Y
     tc     11 u0    0.53549 eV    9.02009 ri    1.63793 ct    0.91579 s2    0.16133 rand1*maxSin2    0.29529 rand1*maxSin2 - sin2Theta    0.13396 loop Y
     tc     12 u0    0.14305 eV    3.54558 ri    1.51124 ct    0.99256 s2    0.01482 rand1*maxSin2    0.13950 rand1*maxSin2 - sin2Theta    0.12468 loop Y
     tc     13 u0    0.20872 eV    4.46168 ri    1.54105 ct    0.97336 s2    0.05257 rand1*maxSin2    0.19031 rand1*maxSin2 - sin2Theta    0.13774 loop Y
     tc     14 u0    0.33341 eV    6.20110 ri    1.61850 ct    0.92679 s2    0.14107 rand1*maxSin2    0.28035 rand1*maxSin2 - sin2Theta    0.13928 loop Y
     tc     15 u0    0.24960 eV    5.03188 ri    1.56646 ct    0.95758 s2    0.08305 rand1*maxSin2    0.13278 rand1*maxSin2 - sin2Theta    0.04973 loop Y
     tc     16 u0    0.81286 eV   12.88946 ri    1.45360 ct    1.03192 s2   -0.06486 rand1*maxSin2    0.21760 rand1*maxSin2 - sin2Theta    0.28246 loop Y
     tc     17 u0    0.64523 eV   10.55094 ri    1.45360 ct    1.03192 s2   -0.06486 rand1*maxSin2    0.29589 rand1*maxSin2 - sin2Theta    0.36075 loop Y
     tc     18 u0    0.31051 eV    5.88158 ri    1.60431 ct    0.93498 s2    0.12581 rand1*maxSin2    0.21354 rand1*maxSin2 - sin2Theta    0.08773 loop Y
     tc     19 u0    0.00786 eV    1.65959 ri    1.47895 ct    1.01423 s2   -0.02867 rand1*maxSin2    0.08108 rand1*maxSin2 - sin2Theta    0.10975 loop Y
     tc     20 u0    0.84745 eV   13.37191 ri    1.45360 ct    1.03192 s2   -0.06486 rand1*maxSin2    0.05782 rand1*maxSin2 - sin2Theta    0.12268 loop Y
     tc     21 u0    0.81947 eV   12.98167 ri    1.45360 ct    1.03192 s2   -0.06486 rand1*maxSin2    0.01439 rand1*maxSin2 - sin2Theta    0.07925 loop Y
     tc     22 u0    0.96168 eV   14.96544 ri    1.45360 ct    1.03192 s2   -0.06486 rand1*maxSin2    0.02420 rand1*maxSin2 - sin2Theta    0.08906 loop Y
     tc     23 u0    0.58786 eV    9.75064 ri    1.52741 ct    0.98205 s2    0.03557 rand1*maxSin2    0.15284 rand1*maxSin2 - sin2Theta    0.11728 loop Y
     tc     24 u0    0.50456 eV    8.58867 ri    1.71805 ct    0.87308 s2    0.23772 rand1*maxSin2    0.06501 rand1*maxSin2 - sin2Theta   -0.17272 loop N


    //_QCtx_generate_cerenkov_photon id 0 
    //qctx::cerenkov_photon id 37969 loop   1 u0    0.95147 ri    1.45360 ct    1.03192 s2   -0.06486 u_mxs2_s2    0.27884 
    //qctx::cerenkov_photon id 37969 loop   2 u0    0.77318 ri    1.45360 ct    1.03192 s2   -0.06486 u_mxs2_s2    0.07865 
    //qctx::cerenkov_photon id 37969 loop   3 u0    0.08577 ri    1.49416 ct    1.00391 s2   -0.00783 u_mxs2_s2    0.15890 
    //qctx::cerenkov_photon id 37969 loop   4 u0    0.11242 ri    1.50021 ct    0.99986 s2    0.00028 u_mxs2_s2    0.09878 
    //qctx::cerenkov_photon id 37969 loop   5 u0    0.01777 ri    1.48004 ct    1.01349 s2   -0.02716 u_mxs2_s2    0.13624 
    //qctx::cerenkov_photon id 37969 loop   6 u0    0.98852 ri    1.45360 ct    1.03192 s2   -0.06486 u_mxs2_s2    0.30967 
    //qctx::cerenkov_photon id 37969 loop   7 u0    0.78130 ri    1.45360 ct    1.03192 s2   -0.06486 u_mxs2_s2    0.23149 
    //qctx::cerenkov_photon id 37969 loop   8 u0    0.23526 ri    1.55755 ct    0.96305 s2    0.07253 u_mxs2_s2    0.02513 
    //qctx::cerenkov_photon id 37969 loop   9 u0    0.41300 ri    1.56390 ct    0.95914 s2    0.08005 u_mxs2_s2   -0.00030 
    //_QCtx_generate_cerenkov_photon id 200000 



     i      3 seqidx   57112 Pmin/eV    1.55000 Pmax/eV   15.50000 dp/eV   13.95000 maxSin2    0.30012
     tc      1 u0    0.96256 eV   14.97773 ri    1.45360 ct    1.03192 s2   -0.06486 rand1*maxSin2    0.09842 rand1*maxSin2 - sin2Theta    0.16328 loop Y
     tc      2 u0    0.04524 eV    2.18113 ri    1.48507 ct    1.01005 s2   -0.02020 rand1*maxSin2    0.00984 rand1*maxSin2 - sin2Theta    0.03004 loop Y
     tc      3 u0    0.77217 eV   12.32171 ri    1.45360 ct    1.03192 s2   -0.06486 rand1*maxSin2    0.12101 rand1*maxSin2 - sin2Theta    0.18587 loop Y
     tc      4 u0    0.77843 eV   12.40907 ri    1.45360 ct    1.03192 s2   -0.06486 rand1*maxSin2    0.11029 rand1*maxSin2 - sin2Theta    0.17515 loop Y
     tc      5 u0    0.41228 eV    7.30128 ri    1.55831 ct    0.96258 s2    0.07344 rand1*maxSin2    0.07374 rand1*maxSin2 - sin2Theta    0.00030 loop Y

     tc      6 u0    0.64143 eV   10.49792 ri    1.45360 ct    1.03192 s2   -0.06486 rand1*maxSin2    0.09615 rand1*maxSin2 - sin2Theta    0.16101 loop Y
     tc      7 u0    0.25953 eV    5.17051 ri    1.57263 ct    0.95381 s2    0.09024 rand1*maxSin2    0.11015 rand1*maxSin2 - sin2Theta    0.01991 loop Y
     tc      8 u0    0.31415 eV    5.93239 ri    1.60658 ct    0.93366 s2    0.12828 rand1*maxSin2    0.00485 rand1*maxSin2 - sin2Theta   -0.12342 loop N


    //_QCtx_generate_cerenkov_photon id 0 
    //qctx::cerenkov_photon id 57112 loop   1 u0    0.96256 ri    1.45360 ct    1.03192 s2   -0.06486 u_mxs2_s2    0.16328 
    //qctx::cerenkov_photon id 57112 loop   2 u0    0.04524 ri    1.48507 ct    1.01005 s2   -0.02020 u_mxs2_s2    0.03004 
    //qctx::cerenkov_photon id 57112 loop   3 u0    0.77217 ri    1.45360 ct    1.03192 s2   -0.06486 u_mxs2_s2    0.18587 
    //qctx::cerenkov_photon id 57112 loop   4 u0    0.77843 ri    1.45360 ct    1.03192 s2   -0.06486 u_mxs2_s2    0.17515 
    //qctx::cerenkov_photon id 57112 loop   5 u0    0.41228 ri    1.55861 ct    0.96240 s2    0.07379 u_mxs2_s2   -0.00006 

          ANOTHER ONE DEVIATING AT CLOSE TO 7.3 eV 



     i      4 seqidx   69670 Pmin/eV    1.55000 Pmax/eV   15.50000 dp/eV   13.95000 maxSin2    0.30012
     tc      1 u0    0.10435 eV    3.00563 ri    1.49792 ct    1.00139 s2   -0.00277 rand1*maxSin2    0.21707 rand1*maxSin2 - sin2Theta    0.21984 loop Y
     tc      2 u0    0.88072 eV   13.83602 ri    1.45360 ct    1.03192 s2   -0.06486 rand1*maxSin2    0.02204 rand1*maxSin2 - sin2Theta    0.08690 loop Y
     tc      3 u0    0.18484 eV    4.12852 ri    1.52629 ct    0.98278 s2    0.03415 rand1*maxSin2    0.12525 rand1*maxSin2 - sin2Theta    0.09110 loop Y
     tc      4 u0    0.38021 eV    6.85388 ri    1.53577 ct    0.97671 s2    0.04603 rand1*maxSin2    0.04601 rand1*maxSin2 - sin2Theta   -0.00003 loop N

    //qctx::cerenkov_photon id 69670 loop   1 u0    0.10435 ri    1.49793 ct    1.00139 s2   -0.00277 u_mxs2_s2    0.21984 
    //qctx::cerenkov_photon id 69670 loop   2 u0    0.88072 ri    1.45360 ct    1.03192 s2   -0.06486 u_mxs2_s2    0.08690 
    //qctx::cerenkov_photon id 69670 loop   3 u0    0.18484 ri    1.52629 ct    0.98278 s2    0.03415 u_mxs2_s2    0.09110 
    //qctx::cerenkov_photon id 69670 loop   4 u0    0.38021 ri    1.53574 ct    0.97673 s2    0.04601 u_mxs2_s2    0.00000 
    //qctx::cerenkov_photon id 69670 loop   5 u0    0.52322 ri    1.66583 ct    0.90045 s2    0.18918 u_mxs2_s2   -0.17099 




     i      5 seqidx   69761 Pmin/eV    1.55000 Pmax/eV   15.50000 dp/eV   13.95000 maxSin2    0.30012
     tc      1 u0    0.72333 eV   11.64046 ri    1.45360 ct    1.03192 s2   -0.06486 rand1*maxSin2    0.09165 rand1*maxSin2 - sin2Theta    0.15651 loop Y
     tc      2 u0    0.43039 eV    7.55389 ri    1.69043 ct    0.88735 s2    0.21261 rand1*maxSin2    0.28231 rand1*maxSin2 - sin2Theta    0.06970 loop Y
     tc      3 u0    0.55687 eV    9.31835 ri    1.58988 ct    0.94347 s2    0.10987 rand1*maxSin2    0.25522 rand1*maxSin2 - sin2Theta    0.14535 loop Y
     tc      4 u0    0.41501 eV    7.33942 ri    1.57825 ct    0.95042 s2    0.09671 rand1*maxSin2    0.12314 rand1*maxSin2 - sin2Theta    0.02643 loop Y
     tc      5 u0    0.92436 eV   14.44486 ri    1.45360 ct    1.03192 s2   -0.06486 rand1*maxSin2    0.15244 rand1*maxSin2 - sin2Theta    0.21730 loop Y
     tc      6 u0    0.83171 eV   13.15238 ri    1.45360 ct    1.03192 s2   -0.06486 rand1*maxSin2    0.17171 rand1*maxSin2 - sin2Theta    0.23658 loop Y
     tc      7 u0    0.42627 eV    7.49643 ri    1.66038 ct    0.90341 s2    0.18385 rand1*maxSin2    0.18386 rand1*maxSin2 - sin2Theta    0.00001 loop Y

                      7.49 not near bin edge 

     tc      8 u0    0.83247 eV   13.16290 ri    1.45360 ct    1.03192 s2   -0.06486 rand1*maxSin2    0.21731 rand1*maxSin2 - sin2Theta    0.28217 loop Y
     tc      9 u0    0.07905 eV    2.65279 ri    1.49288 ct    1.00477 s2   -0.00956 rand1*maxSin2    0.21548 rand1*maxSin2 - sin2Theta    0.22504 loop Y
     tc     10 u0    0.10878 eV    3.06752 ri    1.49890 ct    1.00073 s2   -0.00147 rand1*maxSin2    0.19377 rand1*maxSin2 - sin2Theta    0.19523 loop Y
     tc     11 u0    0.34725 eV    6.39408 ri    1.61796 ct    0.92709 s2    0.14050 rand1*maxSin2    0.20454 rand1*maxSin2 - sin2Theta    0.06404 loop Y
     tc     12 u0    0.44950 eV    7.82049 ri    1.79158 ct    0.83725 s2    0.29901 rand1*maxSin2    0.18097 rand1*maxSin2 - sin2Theta   -0.11805 loop N

    //qctx::cerenkov_photon id 69761 loop   1 u0    0.72333 ri    1.45360 ct    1.03192 s2   -0.06486 u_mxs2_s2    0.15651 
    //qctx::cerenkov_photon id 69761 loop   2 u0    0.43039 ri    1.69045 ct    0.88734 s2    0.21263 u_mxs2_s2    0.06968 
    //qctx::cerenkov_photon id 69761 loop   3 u0    0.55687 ri    1.58989 ct    0.94346 s2    0.10988 u_mxs2_s2    0.14534 
    //qctx::cerenkov_photon id 69761 loop   4 u0    0.41501 ri    1.57825 ct    0.95042 s2    0.09670 u_mxs2_s2    0.02644 
    //qctx::cerenkov_photon id 69761 loop   5 u0    0.92436 ri    1.45360 ct    1.03192 s2   -0.06486 u_mxs2_s2    0.21730 
    //qctx::cerenkov_photon id 69761 loop   6 u0    0.83171 ri    1.45360 ct    1.03192 s2   -0.06486 u_mxs2_s2    0.23657 
    //qctx::cerenkov_photon id 69761 loop   7 u0    0.42627 ri    1.66042 ct    0.90339 s2    0.18389 u_mxs2_s2   -0.00003 



     i      6 seqidx   70952 Pmin/eV    1.55000 Pmax/eV   15.50000 dp/eV   13.95000 maxSin2    0.30012
     tc      1 u0    0.91380 eV   14.29750 ri    1.45360 ct    1.03192 s2   -0.06486 rand1*maxSin2    0.29951 rand1*maxSin2 - sin2Theta    0.36437 loop Y
     tc      2 u0    0.17346 eV    3.96975 ri    1.52219 ct    0.98542 s2    0.02894 rand1*maxSin2    0.20108 rand1*maxSin2 - sin2Theta    0.17213 loop Y
     tc      3 u0    0.17906 eV    4.04787 ri    1.52421 ct    0.98412 s2    0.03151 rand1*maxSin2    0.23341 rand1*maxSin2 - sin2Theta    0.20190 loop Y
     tc      4 u0    0.65641 eV   10.70686 ri    1.45360 ct    1.03192 s2   -0.06486 rand1*maxSin2    0.25349 rand1*maxSin2 - sin2Theta    0.31835 loop Y
     tc      5 u0    0.76884 eV   12.27526 ri    1.45360 ct    1.03192 s2   -0.06486 rand1*maxSin2    0.05198 rand1*maxSin2 - sin2Theta    0.11684 loop Y
     tc      6 u0    0.01979 eV    1.82606 ri    1.48042 ct    1.01323 s2   -0.02663 rand1*maxSin2    0.01037 rand1*maxSin2 - sin2Theta    0.03699 loop Y
     tc      7 u0    0.70850 eV   11.43356 ri    1.45360 ct    1.03192 s2   -0.06486 rand1*maxSin2    0.03084 rand1*maxSin2 - sin2Theta    0.09570 loop Y
     tc      8 u0    0.26270 eV    5.21466 ri    1.57460 ct    0.95262 s2    0.09251 rand1*maxSin2    0.25519 rand1*maxSin2 - sin2Theta    0.16268 loop Y
     tc      9 u0    0.63738 eV   10.44140 ri    1.45360 ct    1.03192 s2   -0.06486 rand1*maxSin2    0.05090 rand1*maxSin2 - sin2Theta    0.11576 loop Y
     tc     10 u0    0.64227 eV   10.50964 ri    1.45360 ct    1.03192 s2   -0.06486 rand1*maxSin2    0.29375 rand1*maxSin2 - sin2Theta    0.35861 loop Y
     tc     11 u0    0.88112 eV   13.84156 ri    1.45360 ct    1.03192 s2   -0.06486 rand1*maxSin2    0.19409 rand1*maxSin2 - sin2Theta    0.25895 loop Y
     tc     12 u0    0.04071 eV    2.11796 ri    1.48435 ct    1.01054 s2   -0.02120 rand1*maxSin2    0.20665 rand1*maxSin2 - sin2Theta    0.22785 loop Y
     tc     13 u0    0.98530 eV   15.29490 ri    1.45360 ct    1.03192 s2   -0.06486 rand1*maxSin2    0.18633 rand1*maxSin2 - sin2Theta    0.25119 loop Y
     tc     14 u0    0.06283 eV    2.42652 ri    1.48910 ct    1.00732 s2   -0.01469 rand1*maxSin2    0.22140 rand1*maxSin2 - sin2Theta    0.23610 loop Y
     tc     15 u0    0.56563 eV    9.44052 ri    1.57020 ct    0.95529 s2    0.08742 rand1*maxSin2    0.08741 rand1*maxSin2 - sin2Theta   -0.00001 loop N


    //qctx::cerenkov_photon id 70952 loop   1 u0    0.91380 ri    1.45360 ct    1.03192 s2   -0.06486 u_mxs2_s2    0.36437 
    //qctx::cerenkov_photon id 70952 loop   2 u0    0.17346 ri    1.52219 ct    0.98542 s2    0.02894 u_mxs2_s2    0.17213 
    //qctx::cerenkov_photon id 70952 loop   3 u0    0.17906 ri    1.52421 ct    0.98412 s2    0.03151 u_mxs2_s2    0.20190 
    //qctx::cerenkov_photon id 70952 loop   4 u0    0.65641 ri    1.45360 ct    1.03192 s2   -0.06486 u_mxs2_s2    0.31835 
    //qctx::cerenkov_photon id 70952 loop   5 u0    0.76884 ri    1.45360 ct    1.03192 s2   -0.06486 u_mxs2_s2    0.11684 
    //qctx::cerenkov_photon id 70952 loop   6 u0    0.01979 ri    1.48042 ct    1.01323 s2   -0.02663 u_mxs2_s2    0.03699 
    //qctx::cerenkov_photon id 70952 loop   7 u0    0.70850 ri    1.45360 ct    1.03192 s2   -0.06486 u_mxs2_s2    0.09570 
    //qctx::cerenkov_photon id 70952 loop   8 u0    0.26270 ri    1.57460 ct    0.95262 s2    0.09251 u_mxs2_s2    0.16268 
    //qctx::cerenkov_photon id 70952 loop   9 u0    0.63738 ri    1.45360 ct    1.03192 s2   -0.06486 u_mxs2_s2    0.11576 
    //qctx::cerenkov_photon id 70952 loop  10 u0    0.64227 ri    1.45360 ct    1.03192 s2   -0.06486 u_mxs2_s2    0.35861 
    //qctx::cerenkov_photon id 70952 loop  11 u0    0.88112 ri    1.45360 ct    1.03192 s2   -0.06486 u_mxs2_s2    0.25895 
    //qctx::cerenkov_photon id 70952 loop  12 u0    0.04071 ri    1.48435 ct    1.01054 s2   -0.02120 u_mxs2_s2    0.22785 
    //qctx::cerenkov_photon id 70952 loop  13 u0    0.98530 ri    1.45360 ct    1.03192 s2   -0.06486 u_mxs2_s2    0.25119 
    //qctx::cerenkov_photon id 70952 loop  14 u0    0.06283 ri    1.48910 ct    1.00732 s2   -0.01469 u_mxs2_s2    0.23610 
    //qctx::cerenkov_photon id 70952 loop  15 u0    0.56563 ri    1.57018 ct    0.95530 s2    0.08740 u_mxs2_s2    0.00001 
    //qctx::cerenkov_photon id 70952 loop  16 u0    0.52981 ri    1.65068 ct    0.90872 s2    0.17423 u_mxs2_s2   -0.13066 



     i      7 seqidx   77035 Pmin/eV    1.55000 Pmax/eV   15.50000 dp/eV   13.95000 maxSin2    0.30012
     tc      1 u0    0.38414 eV    6.90879 ri    1.52834 ct    0.98145 s2    0.03675 rand1*maxSin2    0.15183 rand1*maxSin2 - sin2Theta    0.11509 loop Y
     tc      2 u0    0.32376 eV    6.06651 ri    1.61255 ct    0.93020 s2    0.13472 rand1*maxSin2    0.17722 rand1*maxSin2 - sin2Theta    0.04250 loop Y
     tc      3 u0    0.51447 eV    8.72684 ri    1.69032 ct    0.88741 s2    0.21251 rand1*maxSin2    0.23260 rand1*maxSin2 - sin2Theta    0.02009 loop Y
     tc      4 u0    0.06684 eV    2.48243 ri    1.49018 ct    1.00659 s2   -0.01323 rand1*maxSin2    0.01703 rand1*maxSin2 - sin2Theta    0.03026 loop Y
     tc      5 u0    0.43190 eV    7.57503 ri    1.70149 ct    0.88158 s2    0.22281 rand1*maxSin2    0.25045 rand1*maxSin2 - sin2Theta    0.02764 loop Y
     tc      6 u0    0.94901 eV   14.78865 ri    1.45360 ct    1.03192 s2   -0.06486 rand1*maxSin2    0.00431 rand1*maxSin2 - sin2Theta    0.06917 loop Y
     tc      7 u0    0.41711 eV    7.36864 ri    1.59354 ct    0.94130 s2    0.11395 rand1*maxSin2    0.11396 rand1*maxSin2 - sin2Theta    0.00001 loop Y

     tc      8 u0    0.99869 eV   15.48173 ri    1.45360 ct    1.03192 s2   -0.06486 rand1*maxSin2    0.20907 rand1*maxSin2 - sin2Theta    0.27393 loop Y
     tc      9 u0    0.94692 eV   14.75948 ri    1.45360 ct    1.03192 s2   -0.06486 rand1*maxSin2    0.01769 rand1*maxSin2 - sin2Theta    0.08255 loop Y
     tc     10 u0    0.73283 eV   11.77301 ri    1.45360 ct    1.03192 s2   -0.06486 rand1*maxSin2    0.26386 rand1*maxSin2 - sin2Theta    0.32872 loop Y
     tc     11 u0    0.17093 eV    3.93442 ri    1.52128 ct    0.98601 s2    0.02778 rand1*maxSin2    0.07192 rand1*maxSin2 - sin2Theta    0.04414 loop Y
     tc     12 u0    0.53237 eV    8.97650 ri    1.64495 ct    0.91188 s2    0.16847 rand1*maxSin2    0.14618 rand1*maxSin2 - sin2Theta   -0.02229 loop N

    //qctx::cerenkov_photon id 77035 loop   1 u0    0.38414 ri    1.52850 ct    0.98136 s2    0.03694 u_mxs2_s2    0.11489 
    //qctx::cerenkov_photon id 77035 loop   2 u0    0.32376 ri    1.61255 ct    0.93020 s2    0.13473 u_mxs2_s2    0.04250 
    //qctx::cerenkov_photon id 77035 loop   3 u0    0.51447 ri    1.69029 ct    0.88742 s2    0.21248 u_mxs2_s2    0.02011 
    //qctx::cerenkov_photon id 77035 loop   4 u0    0.06684 ri    1.49018 ct    1.00659 s2   -0.01323 u_mxs2_s2    0.03026 
    //qctx::cerenkov_photon id 77035 loop   5 u0    0.43190 ri    1.70150 ct    0.88158 s2    0.22282 u_mxs2_s2    0.02763 
    //qctx::cerenkov_photon id 77035 loop   6 u0    0.94901 ri    1.45360 ct    1.03192 s2   -0.06486 u_mxs2_s2    0.06917 
    //qctx::cerenkov_photon id 77035 loop   7 u0    0.41711 ri    1.59360 ct    0.94127 s2    0.11402 u_mxs2_s2   -0.00005 



     i      8 seqidx   77501 Pmin/eV    1.55000 Pmax/eV   15.50000 dp/eV   13.95000 maxSin2    0.30012
     tc      1 u0    0.60134 eV    9.93866 ri    1.50346 ct    0.99770 s2    0.00459 rand1*maxSin2    0.13484 rand1*maxSin2 - sin2Theta    0.13025 loop Y
     tc      2 u0    0.74981 eV   12.00981 ri    1.45360 ct    1.03192 s2   -0.06486 rand1*maxSin2    0.22144 rand1*maxSin2 - sin2Theta    0.28630 loop Y
     tc      3 u0    0.10063 eV    2.95383 ri    1.49714 ct    1.00191 s2   -0.00382 rand1*maxSin2    0.23485 rand1*maxSin2 - sin2Theta    0.23867 loop Y
     tc      4 u0    0.80985 eV   12.84747 ri    1.45360 ct    1.03192 s2   -0.06486 rand1*maxSin2    0.10910 rand1*maxSin2 - sin2Theta    0.17396 loop Y
     tc      5 u0    0.96392 eV   14.99671 ri    1.45360 ct    1.03192 s2   -0.06486 rand1*maxSin2    0.06149 rand1*maxSin2 - sin2Theta    0.12635 loop Y
     tc      6 u0    0.48421 eV    8.30468 ri    1.77504 ct    0.84505 s2    0.28589 rand1*maxSin2    0.28588 rand1*maxSin2 - sin2Theta   -0.00000 loop N

    //qctx::cerenkov_photon id 77501 loop   1 u0    0.60134 ri    1.50345 ct    0.99771 s2    0.00458 u_mxs2_s2    0.13026 
    //qctx::cerenkov_photon id 77501 loop   2 u0    0.74981 ri    1.45360 ct    1.03192 s2   -0.06486 u_mxs2_s2    0.28630 
    //qctx::cerenkov_photon id 77501 loop   3 u0    0.10063 ri    1.49714 ct    1.00191 s2   -0.00382 u_mxs2_s2    0.23867 
    //qctx::cerenkov_photon id 77501 loop   4 u0    0.80985 ri    1.45360 ct    1.03192 s2   -0.06486 u_mxs2_s2    0.17396 
    //qctx::cerenkov_photon id 77501 loop   5 u0    0.96392 ri    1.45360 ct    1.03192 s2   -0.06486 u_mxs2_s2    0.12635 
    //qctx::cerenkov_photon id 77501 loop   6 u0    0.48421 ri    1.77493 ct    0.84510 s2    0.28580 u_mxs2_s2    0.00008 

    //qctx::cerenkov_photon id 77501 loop   7 u0    0.19367 ri    1.53170 ct    0.97930 s2    0.04096 u_mxs2_s2    0.12786 
    //qctx::cerenkov_photon id 77501 loop   8 u0    0.38280 ri    1.52736 ct    0.98208 s2    0.03551 u_mxs2_s2    0.19009 
    //qctx::cerenkov_photon id 77501 loop   9 u0    0.98008 ri    1.45360 ct    1.03192 s2   -0.06486 u_mxs2_s2    0.18380 
    //qctx::cerenkov_photon id 77501 loop  10 u0    0.26729 ri    1.57745 ct    0.95090 s2    0.09579 u_mxs2_s2    0.18082 
    //qctx::cerenkov_photon id 77501 loop  11 u0    0.28075 ri    1.58582 ct    0.94588 s2    0.10531 u_mxs2_s2    0.19156 
    //qctx::cerenkov_photon id 77501 loop  12 u0    0.85420 ri    1.45360 ct    1.03192 s2   -0.06486 u_mxs2_s2    0.25783 
    //qctx::cerenkov_photon id 77501 loop  13 u0    0.16042 ri    1.51750 ct    0.98847 s2    0.02293 u_mxs2_s2    0.04417 
    //qctx::cerenkov_photon id 77501 loop  14 u0    0.14016 ri    1.51020 ct    0.99325 s2    0.01346 u_mxs2_s2    0.03445 
    //qctx::cerenkov_photon id 77501 loop  15 u0    0.71971 ri    1.45360 ct    1.03192 s2   -0.06486 u_mxs2_s2    0.12184 
    //qctx::cerenkov_photon id 77501 loop  16 u0    0.44289 ri    1.78176 ct    0.84186 s2    0.29127 u_mxs2_s2   -0.14472 




     i      9 seqidx   81643 Pmin/eV    1.55000 Pmax/eV   15.50000 dp/eV   13.95000 maxSin2    0.30012
     tc      1 u0    0.24282 eV    4.93735 ri    1.56224 ct    0.96016 s2    0.07810 rand1*maxSin2    0.09242 rand1*maxSin2 - sin2Theta    0.01432 loop Y
     tc      2 u0    0.98302 eV   15.26312 ri    1.45360 ct    1.03192 s2   -0.06486 rand1*maxSin2    0.17711 rand1*maxSin2 - sin2Theta    0.24197 loop Y
     tc      3 u0    0.06673 eV    2.48091 ri    1.49015 ct    1.00661 s2   -0.01327 rand1*maxSin2    0.27202 rand1*maxSin2 - sin2Theta    0.28529 loop Y
     tc      4 u0    0.26372 eV    5.22886 ri    1.57523 ct    0.95224 s2    0.09324 rand1*maxSin2    0.27110 rand1*maxSin2 - sin2Theta    0.17786 loop Y
     tc      5 u0    0.04646 eV    2.19809 ri    1.48527 ct    1.00992 s2   -0.01994 rand1*maxSin2    0.21834 rand1*maxSin2 - sin2Theta    0.23827 loop Y
     tc      6 u0    0.97055 eV   15.08917 ri    1.45360 ct    1.03192 s2   -0.06486 rand1*maxSin2    0.11778 rand1*maxSin2 - sin2Theta    0.18264 loop Y
     tc      7 u0    0.27440 eV    5.37794 ri    1.58187 ct    0.94824 s2    0.10084 rand1*maxSin2    0.27769 rand1*maxSin2 - sin2Theta    0.17686 loop Y
     tc      8 u0    0.32912 eV    6.14116 ri    1.61588 ct    0.92829 s2    0.13828 rand1*maxSin2    0.26658 rand1*maxSin2 - sin2Theta    0.12830 loop Y
     tc      9 u0    0.56627 eV    9.44940 ri    1.56877 ct    0.95616 s2    0.08575 rand1*maxSin2    0.08575 rand1*maxSin2 - sin2Theta   -0.00000 loop N

    //qctx::cerenkov_photon id 81643 loop   1 u0    0.24282 ri    1.56225 ct    0.96016 s2    0.07810 u_mxs2_s2    0.01432 
    //qctx::cerenkov_photon id 81643 loop   2 u0    0.98302 ri    1.45360 ct    1.03192 s2   -0.06486 u_mxs2_s2    0.24197 
    //qctx::cerenkov_photon id 81643 loop   3 u0    0.06673 ri    1.49015 ct    1.00661 s2   -0.01327 u_mxs2_s2    0.28529 
    //qctx::cerenkov_photon id 81643 loop   4 u0    0.26372 ri    1.57523 ct    0.95224 s2    0.09324 u_mxs2_s2    0.17786 
    //qctx::cerenkov_photon id 81643 loop   5 u0    0.04646 ri    1.48527 ct    1.00992 s2   -0.01994 u_mxs2_s2    0.23827 
    //qctx::cerenkov_photon id 81643 loop   6 u0    0.97055 ri    1.45360 ct    1.03192 s2   -0.06486 u_mxs2_s2    0.18264 
    //qctx::cerenkov_photon id 81643 loop   7 u0    0.27440 ri    1.58187 ct    0.94824 s2    0.10084 u_mxs2_s2    0.17686 
    //qctx::cerenkov_photon id 81643 loop   8 u0    0.32912 ri    1.61588 ct    0.92829 s2    0.13828 u_mxs2_s2    0.12830 
    //qctx::cerenkov_photon id 81643 loop   9 u0    0.56627 ri    1.56874 ct    0.95618 s2    0.08572 u_mxs2_s2    0.00004 

    //qctx::cerenkov_photon id 81643 loop  10 u0    0.09310 ri    1.49556 ct    1.00297 s2   -0.00595 u_mxs2_s2    0.13416 
    //qctx::cerenkov_photon id 81643 loop  11 u0    0.02792 ri    1.48196 ct    1.01217 s2   -0.02450 u_mxs2_s2    0.03925 
    //qctx::cerenkov_photon id 81643 loop  12 u0    0.03671 ri    1.48362 ct    1.01104 s2   -0.02221 u_mxs2_s2    0.15285 
    //qctx::cerenkov_photon id 81643 loop  13 u0    0.11091 ri    1.49967 ct    1.00022 s2   -0.00044 u_mxs2_s2    0.00418 
    //qctx::cerenkov_photon id 81643 loop  14 u0    0.92990 ri    1.45360 ct    1.03192 s2   -0.06486 u_mxs2_s2    0.36377 
    //qctx::cerenkov_photon id 81643 loop  15 u0    0.25187 ri    1.56787 ct    0.95671 s2    0.08470 u_mxs2_s2   -0.01139 



Try comparing G4Cerenkov_modified against itself with different seeds
------------------------------------------------------------------------

* chi2 ok 1.03  

::

    epsilon:CerenkovStandalone blyth$ SEED=1 ./G4Cerenkov_modifiedTest.sh
    epsilon:CerenkovStandalone blyth$ SEED=2 ./G4Cerenkov_modifiedTest.sh

    /tmp/G4Cerenkov_modifiedTest/BetaInverse_1.500_override_fNumPhotons_1000000_SKIP_CONTINUEseed_1_/
    /tmp/G4Cerenkov_modifiedTest/BetaInverse_1.500_override_fNumPhotons_1000000_SKIP_CONTINUEseed_2_/

    epsilon:ana blyth$ ARG=7 ipython -i wavelength_cfplot.py 



