QSimTest_propagate_at_boundary_vs_BoundaryStandalone_G4OpBoundaryProcessTest
===============================================================================


How to do random aligned propagate_at_boundary comparison between QSim and Geant4
------------------------------------------------------------------------------------

::

    qu tests
         ## OR: cd ~/opticks/qudarap/tests      

    om  
        ## update build

    TEST=rng_sequence NUM=1000000 ./QSimTest.sh 
        ##  generate pre-cooked randoms so Geant4 can consume the curand generated random stream 




    TEST=hemisphere_s_polarized NUM=1000000 ./QSimTest.sh 
        ## generate a sample of 1M S-polarized photons all incident at origin  
        ## this sample of photons is used by both QSimTest.sh below and G4OpBoundaryProcessTest.sh

    TEST=hemisphere_p_polarized NUM=1000000 ./QSimTest.sh 
        ## generate a sample of 1M P-polarized photons all incident at origin  
        ## this sample of photons is used by both QSimTest.sh below and G4OpBoundaryProcessTest.sh



    TEST=propagate_at_boundary_s_polarized NUM=1000000 ./QSimTest.sh 
        ## mutate the hemisphere_s_polarized photons doing boundary reflect or transmit   

    TEST=propagate_at_boundary_p_polarized NUM=1000000 ./QSimTest.sh 
        ## mutate the hemisphere_p_polarized photons doing boundary reflect or transmit   



    bst 
        ## OR : cd ~/opticks/examples/Geant4/BoundaryStandalone 

    vi G4OpBoundaryProcessTest.sh
        ## check the envvars which control where the input pre-cooked randoms and photons will come from 

    ./G4OpBoundaryProcessTest.sh
        ## compiles and runs loading the photons from eg hemisphere_s_polarized and bouncing them 


Random Aligned comparison::

    epsilon:BoundaryStandalone blyth$ ./G4OpBoundaryProcessTest_cf_QSimTest.sh
    a.shape (1000000, 4, 4) : /tmp/blyth/opticks/QSimTest/propagate_at_boundary_s_polarized/p.npy  
    b.shape (1000000, 4, 4) : /tmp/G4OpBoundaryProcessTest/p.npy  
     a_flag (array([1024, 2048], dtype=uint32), array([ 45024, 954976])) 
     b_flag (array([1024, 2048], dtype=uint32), array([ 45025, 954975])) 
    a_TransCoeff [0.784 0.799 0.588 ... 0.853 0.481 0.959] 
    b_TransCoeff [0.784 0.799 0.588 ... 0.853 0.481 0.959] 
     a_flat [0.438 0.46  0.25  ... 0.557 0.184 0.992] 
     b_flat [0.438 0.46  0.25  ... 0.557 0.184 0.992] 

    In [1]:                                           

After tidy up filing::

    epsilon:BoundaryStandalone blyth$ ./G4OpBoundaryProcessTest.sh cf
    === ./G4OpBoundaryProcessTest.sh : G4OpBoundaryProcessTest.cc
    a_key :  OPTICKS_QSIM_DSTDIR  A_FOLD : /tmp/blyth/opticks/QSimTest/propagate_at_boundary_s_polarized
    b_key :   OPTICKS_BST_DSTDIR  B_FOLD : /tmp/blyth/opticks/G4OpBoundaryProcessTest/propagate_at_boundary_s_polarized
    a.shape (1000000, 4, 4) : /tmp/blyth/opticks/QSimTest/propagate_at_boundary_s_polarized/p.npy  
    b.shape (1000000, 4, 4) : /tmp/blyth/opticks/G4OpBoundaryProcessTest/propagate_at_boundary_s_polarized/p.npy  
    a_flag (array([1024, 2048], dtype=uint32), array([ 45024, 954976])) 
    b_flag (array([1024, 2048], dtype=uint32), array([ 45025, 954975])) 
    np.where( a_flag != b_flag )  : (array([209411]),)
    a_TransCoeff [0.784 0.799 0.588 ... 0.853 0.481 0.959] 
    b_TransCoeff [0.784 0.799 0.588 ... 0.853 0.481 0.959] 
    np.where( np.abs( a_TransCoeff - b_TransCoeff) > 1e-6 )  : (array([], dtype=int64),)
    a_flat [0.438 0.46  0.25  ... 0.557 0.184 0.992] 
    b_flat [0.438 0.46  0.25  ... 0.557 0.184 0.992] 
    np.where( a_flat != b_flat )  : (array([], dtype=int64),)
    np.where( np.abs(a[:,0] - b[:,0]) > 1e-6 )  : (array([], dtype=int64), array([], dtype=int64))
    np.where( np.abs(a[:,1] - b[:,1]) > 1e-6 )  : (array([209411, 209411, 209411]), array([0, 1, 2]))
    np.where( np.abs(a[:,2] - b[:,2]) > 1e-6 )  : (array([209411, 209411]), array([0, 1]))
    np.where( np.abs(a[:,3] - b[:,3]) > 1e-6 )  : (array([], dtype=int64), array([], dtype=int64))

    In [1]:                                    




Once flag discrepancy to chase::

    In [2]: np.where( a_flag != b_flag )
    Out[2]: (array([209411]),)

    In [6]: np.where( np.abs(a_TransCoeff-b_TransCoeff) > 1e-6 )
    Out[6]: (array([], dtype=int64),)

    In [5]: np.all( a_flat == b_flat  )   ## the random numbers are input so perfect agreement is expected and is found
    Out[5]: True

    In [14]: np.where( np.abs(a[:,0] - b[:,0]) > 1e-6 )     
    Out[14]: (array([], dtype=int64), array([], dtype=int64))    

    In [15]: np.all( a[:,0] == b[:,0] )   ## this is input mom : so perfect agreement is expected and is found
    Out[15]: True

    In [17]: np.all( a[:,3,:3] == b[:,3,:3] )  ## input pol : so perfect agreement is expected and is found
    Out[17]: True 




    In [13]: np.where( np.abs(a[:,1] - b[:,1]) > 1e-6 )
    Out[13]: (array([209411, 209411, 209411]), array([0, 1, 2]))


    In [18]: np.where( np.abs(a[:,1] - b[:,1]) > 1e-6 )          ## one in a million with different mom  
    Out[18]: (array([209411, 209411, 209411]), array([0, 1, 2]))

    In [2]: np.where( a_flag != b_flag )     ## its the one with discrepant flag 
    Out[2]: (array([209411]),)


The flag jumper is at the TransCoeff cut::

    In [25]: a[209411]                                                                                                                                                                                        
    Out[25]: 
    array([[ -0.136,  -0.264,  -0.955,   0.955],
           [ -0.09 ,  -0.176,  -0.98 ,   0.955],
           [  0.89 ,  -0.456,  -0.   , 500.   ],
           [  0.89 ,  -0.456,   0.   ,   0.   ]], dtype=float32)

    In [26]: b[209411]                                                                                                                                                                                        
    Out[26]: 
    array([[ -0.136,  -0.264,  -0.955,   0.955],
           [ -0.136,  -0.264,   0.955,   0.955],
           [ -0.89 ,   0.456,  -0.   , 500.   ],
           [  0.89 ,  -0.456,   0.   ,   0.   ]], dtype=float32)

    In [14]: a[209411,0,3] > a[209411,1,3]                                                                                                                                                                    
    Out[14]: False

    In [15]: b[209411,0,3] > b[209411,1,3]                                                                                                                                                                    
    Out[15]: True




    In [19]: np.where( np.abs(a[:,2] - b[:,2]) > 1e-6 )   ## 3 with different polarization, 1 is the flag differ one 
    Out[19]: 
    (array([209411, 209411, 251959, 251959, 251959, 317933, 317933, 317933]),
     array([0, 1, 0, 1, 2, 0, 1, 2]))

    In [24]: np.where( np.abs(a[:,2] - b[:,2]) > 1e-1 )  ## difference in pol.x pol.y and it is not small 
    Out[24]: 
    (array([209411, 209411, 251959, 251959, 317933, 317933]),
     array([0, 1, 0, 1, 0, 1]))



The other discrepant two are very nearly at normal incidence and seems to have an x-y flip:: 

    In [16]: a[251959]
    Out[16]: 
    array([[ -0.   ,  -0.001,  -1.   ,   1.   ],
           [ -0.   ,  -0.001,   1.   ,   0.96 ],
           [  0.16 ,   0.987,   0.001, 500.   ],
           [  0.987,  -0.16 ,   0.   ,   0.   ]], dtype=float32)

    In [17]: b[251959]
    Out[17]: 
    array([[ -0.   ,  -0.001,  -1.   ,   1.   ],
           [ -0.   ,  -0.001,   1.   ,   0.96 ],
           [ -0.987,   0.16 ,  -0.   , 500.   ],
           [  0.987,  -0.16 ,   0.   ,   0.   ]], dtype=float32)


::

    2022-03-24 20:39:34.885 INFO  [570874] [QSimTest<float>::photon_launch_mutate@504]  loaded (1000000, 4, 4, ) from src_subfold hemisphere_s_polarized
    //QSim_photon_launch sim 0x703a40a00 photon 0x7042c0000 num_photon 1000000 dbg 0x703a40c00 type 22 name propagate_at_boundary_s_polarized 
    //qsim.propagate_at_boundary id 251959 
    //qsim.propagate_at_boundary surface_normal (    0.0000,     0.0000,     1.0000) 
    //qsim.propagate_at_boundary direction (   -0.0002,    -0.0011,    -1.0000) 
    //qsim.propagate_at_boundary polarization (    0.9871,    -0.1603,     0.0000) 
    //qsim.propagate_at_boundary c1     1.0000 normal_incidence 1 
    //qsim.propagate_at_boundary RR.x     0.0000 A_trans (    0.9871    -0.1603     0.0000 )  RR.y     1.0000  A_paral (    0.1603     0.9871     0.0011 ) 
    //qsim.propagate_at_boundary reflect 1  tir 0 polarization (    0.1603,     0.9871,     0.0011) 

At normal incidence the new polarization comes all from A_paral as RR.x is zero.



::


    G4OpBoundaryProcessTest::init  normal (     0.0000     0.0000     1.0000) n1     1.0000 n2     1.5000
    G4OpBoundaryProcessTest::set_prd_normal OPTICKS_INPUT_PRD  normal (     0.0000     0.0000     1.0000) n1     1.0000 n2     1.5000
    didi idx 251959 Rindex1 1.00000 Rindex2 1.50000
     TransCoeff     0.9600 E1_perp    -1.0000 E1_parl     0.0000 E2_perp    -0.8000 E2_parl     0.0000
     incident ray oblique  E2_parl 0.0000 E2_perp 0.2000 C_parl 0.0000 C_perp 1.0000  NewPolarization ( -0.9871 0.1603 -0.0000)

    G4OpBoundaryProcessTest::init  normal (     0.0000     0.0000     1.0000) n1     1.0000 n2     1.5000
    G4OpBoundaryProcessTest::set_prd_normal OPTICKS_INPUT_PRD  normal (     0.0000     0.0000     1.0000) n1     1.0000 n2     1.5000
    didi idx 251959 Rindex1 1.00000 Rindex2 1.50000
     TransCoeff     0.9600 E1_perp    -1.0000 E1_parl     0.0000 E2_perp    -0.8000 E2_parl     0.0000
     C_parl 0.0000 A_paral ( -0.1603 -0.9871 -0.0011) 
     C_perp 1.0000 A_trans ( -0.9871 0.1603 0.0000) 
     incident ray oblique  E2_parl 0.0000 E2_perp 0.2000  NewPolarization ( -0.9871 0.1603 -0.0000)
    p.shape (1000000, 4, 4) 


Notice sign flip for A_paral and A_trans between G4 and OK that is causing the deviation in polarization at normal incidence::


    1236                        E2_total  = E2_perp*E2_perp + E2_parl*E2_parl;
    1237                        A_paral   = NewMomentum.cross(A_trans);
    1238                        A_paral   = A_paral.unit();
    1239                        E2_abs    = std::sqrt(E2_total);


    0688     const float3 A_trans = normal_incidence ? *polarization : normalize(cross(*direction, surface_normal)) ; //   OLD POLARIZATION AT NORMAL 
    0727     const float3 A_paral = normalize(cross(*direction, A_trans));   ## thIS IS THE NEW DIRECTION 



::

    In [18]: a[317933]
    Out[18]: 
    array([[ -0.   ,  -0.   ,  -1.   ,   1.   ],
           [ -0.   ,  -0.   ,   1.   ,   0.96 ],
           [  0.479,   0.878,   0.   , 500.   ],
           [  0.878,  -0.479,   0.   ,   0.   ]], dtype=float32)

    In [19]: b[317933]
    Out[19]: 
    array([[ -0.   ,  -0.   ,  -1.   ,   1.   ],
           [ -0.   ,  -0.   ,   1.   ,   0.96 ],
           [ -0.878,   0.479,  -0.   , 500.   ],
           [  0.878,  -0.479,   0.   ,   0.   ]], dtype=float32)


* b (G4) at normal incidence the polarization is flipped
* a (OK) at normal incidence x and y get flipped 



That is strange the random number of the two discrepants is very close to 1::

    In [20]: a_flat[251959]
    Out[20]: 0.99999934

    In [21]: b_flat[251959]   ## exactly the same as a_flat as its an input 
    Out[21]: 0.99999934

    In [22]: b_flat[317933]
    Out[22]: 0.9999999

    In [23]: a_flat[317933]   ## again exact match 
    Out[23]: 0.9999999

Bizarre, surely that cannot be a coincidence ? The two near normal incidence discrepants consume a random very close to 1::

    In [25]: np.where( a_flat > 0.999999 )
    Out[25]: (array([251959, 317933]),)



Cross Product Sign Convention
--------------------------------

::

    255 inline double Hep3Vector::dot(const Hep3Vector & p) const {
    256   return dx*p.x() + dy*p.y() + dz*p.z();
    257 }
    258 

    259 inline Hep3Vector Hep3Vector::cross(const Hep3Vector & p) const {
    260   return Hep3Vector(dy*p.z()-p.y()*dz, dz*p.x()-p.z()*dx, dx*p.y()-p.x()*dy);
    261 }

        d.cross(p) 


    0539 /** cross product */
     540 SUTIL_INLINE SUTIL_HOSTDEVICE float3 cross(const float3& a, const float3& b)
     541 {
     542   return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
     543 }

        cross(d, p) 


       //                  a <-> d
       //                  b <-> p 

         So : OldMomentum.cross(theFacetNormal) 
         us  cross( 




    1152               if (sint1 > 0.0) {
    1153                  A_trans = OldMomentum.cross(theFacetNormal);
    1154                  A_trans = A_trans.unit();
    1155                  E1_perp = OldPolarization * A_trans;
    1156                  E1pp    = E1_perp * A_trans;
    1157                  E1pl    = OldPolarization - E1pp;
    1158                  E1_parl = E1pl.mag();
    1159               }
    1160               else {
    1161                  A_trans  = OldPolarization;
    1162                  // Here we Follow Jackson's conventions and we set the
    1163                  // parallel component = 1 in case of a ray perpendicular
    1164                  // to the surface
    1165                  E1_perp  = 0.0;
    1166                  E1_parl  = 1.0;
    1167               }




Aligning normal incidence
----------------------------

Change normal incidence cut to match Geant4 "sint1==0."::

    -    const bool normal_incidence = fabs(c1) > 0.999999f ; 
    +    //const bool normal_incidence = fabs(c1) > 0.999999f ; 
    +    const bool normal_incidence = fabs(c1) == 1.f ; 


    2022-03-25 09:51:12.182 INFO  [793717] [QSimTest<float>::photon_launch_mutate@504]  loaded (1000000, 4, 4, ) from src_subfold hemisphere_s_polarized
    //QSim_photon_launch sim 0x703a40a00 photon 0x7042c0000 num_photon 1000000 dbg 0x703a40c00 type 22 name propagate_at_boundary_s_polarized 
    //qsim.propagate_at_boundary id 251959 
    //qsim.propagate_at_boundary surface_normal (    0.0000,     0.0000,     1.0000) 
    //qsim.propagate_at_boundary direction (   -0.0002,    -0.0011,    -1.0000) 
    //qsim.propagate_at_boundary polarization (    0.9871,    -0.1603,     0.0000) 
    //qsim.propagate_at_boundary c1     1.0000 normal_incidence 0 
    //qsim.propagate_at_boundary RR.x     1.0000 A_trans (   -0.9871     0.1603     0.0000 )  RR.y     0.0000  A_paral (   -0.1603    -0.9871    -0.0011 ) 
    //qsim.propagate_at_boundary reflect 1  tir 0 polarization (   -0.9871,     0.1603,     0.0000) 
    NP::Write dtype <f4 ni        1 nj  4 nk  4 nl  -1 nm  -1 path /tmp/blyth/opticks/QSimTest/propagate_at_boundary_s_polarized/p0.npy
    NP::Write dtype <f4 ni        1 nj  4 nk  4 nl  -1 nm  -1 path /tmp/blyth/opticks/QSimTest/propagate_at_boundary_s_polarized/prd.npy
    === ./QSimTest.sh : invoking analysis script QSimTest_propagate_at_boundary_x_polarized.py



::

    In [1]: a[251959]                                                                                                                                                                               
    Out[1]: 
    array([[ -0.   ,  -0.001,  -1.   ,   1.   ],
           [ -0.   ,  -0.001,   1.   ,   0.96 ],
           [ -0.987,   0.16 ,   0.   , 500.   ],
           [  0.987,  -0.16 ,   0.   ,   0.   ]], dtype=float32)

    In [2]: b[251959]                                                                                                                                                                               
    Out[2]: 
    array([[ -0.   ,  -0.001,  -1.   ,   1.   ],
           [ -0.   ,  -0.001,   1.   ,   0.96 ],
           [ -0.987,   0.16 ,  -0.   , 500.   ],
           [  0.987,  -0.16 ,   0.   ,   0.   ]], dtype=float32)

    In [3]: np.where( np.abs(a[:,2] - b[:,2]) > 1e-6 )                                                                                                                                              
    Out[3]: (array([209411, 209411]), array([0, 1]))




Now left with the 1 in a million cut edger::

    In [4]: np.where( np.abs(a[:,0] - b[:,0]) > 1e-6 )
    Out[4]: (array([], dtype=int64), array([], dtype=int64))

    In [5]: np.where( np.abs(a[:,1] - b[:,1]) > 1e-6 )
    Out[5]: (array([209411, 209411, 209411]), array([0, 1, 2]))

    In [6]: np.where( np.abs(a[:,2] - b[:,2]) > 1e-6 )
    Out[6]: (array([209411, 209411]), array([0, 1]))

    In [7]: np.where( np.abs(a[:,3] - b[:,3]) > 1e-6 )
    Out[7]: (array([], dtype=int64), array([], dtype=int64))




P-polarized comparison : get 1-in-a-million TransCoeff cut edger just like S-polarized
-----------------------------------------------------------------------------------------

::

    epsilon:BoundaryStandalone blyth$ ./G4OpBoundaryProcessTest.sh cf
    === ./G4OpBoundaryProcessTest.sh : G4OpBoundaryProcessTest.cc
    a_key :  OPTICKS_QSIM_DSTDIR  A_FOLD : /tmp/blyth/opticks/QSimTest/propagate_at_boundary_p_polarized
    b_key :   OPTICKS_BST_DSTDIR  B_FOLD : /tmp/blyth/opticks/G4OpBoundaryProcessTest/propagate_at_boundary_p_polarized
    a.shape (1000000, 4, 4) : /tmp/blyth/opticks/QSimTest/propagate_at_boundary_p_polarized/p.npy  
    b.shape (1000000, 4, 4) : /tmp/blyth/opticks/G4OpBoundaryProcessTest/propagate_at_boundary_p_polarized/p.npy  
    a_flag (array([1024, 2048], dtype=uint32), array([ 36015, 963985])) 
    b_flag (array([1024, 2048], dtype=uint32), array([ 36016, 963984])) 
    np.where( a_flag != b_flag )  : (array([104859]),)
    a_TransCoeff [0.99  0.994 0.884 ... 1.    0.784 0.961] 
    b_TransCoeff [0.99  0.994 0.884 ... 1.    0.784 0.961] 
    np.where( np.abs( a_TransCoeff - b_TransCoeff) > 1e-6 )  : (array([], dtype=int64),)
    a_flat [0.438 0.46  0.25  ... 0.557 0.184 0.992] 
    b_flat [0.438 0.46  0.25  ... 0.557 0.184 0.992] 
    np.where( a_flat != b_flat )  : (array([], dtype=int64),)
    np.where( np.abs(a[:,0] - b[:,0]) > 1e-6 )  : (array([], dtype=int64), array([], dtype=int64))
    np.where( np.abs(a[:,1] - b[:,1]) > 1e-6 )  : (array([104859, 104859, 104859]), array([0, 1, 2]))
    np.where( np.abs(a[:,2] - b[:,2]) > 1e-6 )  : (array([104859, 104859, 104859]), array([0, 1, 2]))
    np.where( np.abs(a[:,3] - b[:,3]) > 1e-6 )  : (array([], dtype=int64), array([], dtype=int64))

    In [1]: a[104859]                                                                                                                                                                               
    Out[1]: 
    array([[  0.264,  -0.036,  -0.964,   0.964],
           [  0.176,  -0.024,  -0.984,   0.964],
           [ -0.975,   0.133,  -0.178, 500.   ],
           [  0.955,  -0.13 ,   0.266,   0.   ]], dtype=float32)

    In [2]: b[104859]                                                                                                                                                                               
    Out[2]: 
    array([[  0.264,  -0.036,  -0.964,   0.964],
           [  0.264,  -0.036,   0.964,   0.964],
           [  0.955,  -0.13 ,  -0.266, 500.   ],
           [  0.955,  -0.13 ,   0.266,   0.   ]], dtype=float32)

    In [3]:                                                                       




"X"-polarized : equal admixture of S and P : deviation less than 1 in a million
---------------------------------------------------------------------------------

::

    epsilon:BoundaryStandalone blyth$ ./G4OpBoundaryProcessTest.sh cf
    === ./G4OpBoundaryProcessTest.sh : G4OpBoundaryProcessTest.cc
    a_key :  OPTICKS_QSIM_DSTDIR  A_FOLD : /tmp/blyth/opticks/QSimTest/propagate_at_boundary_x_polarized
    b_key :   OPTICKS_BST_DSTDIR  B_FOLD : /tmp/blyth/opticks/G4OpBoundaryProcessTest/propagate_at_boundary_x_polarized
    a.shape (1000000, 4, 4) : /tmp/blyth/opticks/QSimTest/propagate_at_boundary_x_polarized/p.npy  
    b.shape (1000000, 4, 4) : /tmp/blyth/opticks/G4OpBoundaryProcessTest/propagate_at_boundary_x_polarized/p.npy  
    a_flag (array([1024, 2048], dtype=uint32), array([ 40034, 959966])) 
    b_flag (array([1024, 2048], dtype=uint32), array([ 40034, 959966])) 
    np.where( a_flag != b_flag )  : (array([], dtype=int64),)
    a_TransCoeff [0.887 0.896 0.736 ... 0.926 0.633 0.96 ] 
    b_TransCoeff [0.887 0.896 0.736 ... 0.926 0.633 0.96 ] 
    np.where( np.abs( a_TransCoeff - b_TransCoeff) > 1e-6 )  : (array([], dtype=int64),)
    a_flat [0.438 0.46  0.25  ... 0.557 0.184 0.992] 
    b_flat [0.438 0.46  0.25  ... 0.557 0.184 0.992] 
    np.where( a_flat != b_flat )  : (array([], dtype=int64),)
    np.where( np.abs(a[:,0] - b[:,0]) > 1e-6 )  : (array([], dtype=int64), array([], dtype=int64))
    np.where( np.abs(a[:,1] - b[:,1]) > 1e-6 )  : (array([], dtype=int64), array([], dtype=int64))
    np.where( np.abs(a[:,2] - b[:,2]) > 1e-6 )  : (array([], dtype=int64), array([], dtype=int64))
    np.where( np.abs(a[:,3] - b[:,3]) > 1e-6 )  : (array([], dtype=int64), array([], dtype=int64))





Trying to test "with the normal" directions
------------------------------------------------

Simply flipping the normal to [0,0,-1] does not test "with the normal" directions
because the directions are all getting oriented wrt the normal to make them against the 
normal. 

TODO: test at lower level to check with the normal or provide way to not auto-orient  

::

    epsilon:BoundaryStandalone blyth$ ./G4OpBoundaryProcessTest.sh cf
    === ./G4OpBoundaryProcessTest.sh : G4OpBoundaryProcessTest.cc
    a_key :  OPTICKS_QSIM_DSTDIR  A_FOLD : /tmp/blyth/opticks/QSimTest/propagate_at_boundary_s_polarized
    b_key :   OPTICKS_BST_DSTDIR  B_FOLD : /tmp/blyth/opticks/G4OpBoundaryProcessTest/propagate_at_boundary_s_polarized
    a.shape (1000000, 4, 4) : /tmp/blyth/opticks/QSimTest/propagate_at_boundary_s_polarized/p.npy  
    b.shape (1000000, 4, 4) : /tmp/blyth/opticks/G4OpBoundaryProcessTest/propagate_at_boundary_s_polarized/p.npy  
    aprd.shape  (1, 4, 4) : /tmp/blyth/opticks/QSimTest/propagate_at_boundary_s_polarized/prd.npy  
    bprd.shape  (1, 4, 4) : /tmp/blyth/opticks/G4OpBoundaryProcessTest/propagate_at_boundary_s_polarized/prd.npy  

    aprd : 
    [[[  0.   0.  -1. 100.]
      [  0.   0.   0.   0.]
      [  0.   0.   0.   0.]
      [  0.   0.   0.   0.]]]

    bprd : 
    [[[  0.   0.  -1. 100.]
      [  0.   0.   0.   0.]
      [  0.   0.   0.   0.]
      [  0.   0.   0.   0.]]]
    a_flag (array([1024, 2048], dtype=uint32), array([ 45024, 954976])) 
    b_flag (array([1024, 2048], dtype=uint32), array([ 45025, 954975])) 
    np.where( a_flag != b_flag )  : (array([209411]),)
    a_TransCoeff [0.784 0.799 0.588 ... 0.853 0.481 0.959] 
    b_TransCoeff [0.784 0.799 0.588 ... 0.853 0.481 0.959] 
    np.where( np.abs( a_TransCoeff - b_TransCoeff) > 1e-6 )  : (array([], dtype=int64),)
    a_flat [0.438 0.46  0.25  ... 0.557 0.184 0.992] 
    b_flat [0.438 0.46  0.25  ... 0.557 0.184 0.992] 
    np.where( a_flat != b_flat )  : (array([], dtype=int64),)
    np.where( np.abs(a[:,0] - b[:,0]) > 1e-6 )  : (array([], dtype=int64), array([], dtype=int64))
    np.where( np.abs(a[:,1] - b[:,1]) > 1e-6 )  : (array([209411, 209411, 209411]), array([0, 1, 2]))
    np.where( np.abs(a[:,2] - b[:,2]) > 1e-6 )  : (array([209411, 209411]), array([0, 1]))
    np.where( np.abs(a[:,3] - b[:,3]) > 1e-6 )  : (array([], dtype=int64), array([], dtype=int64))




Normal incidence polarization x-y flip
------------------------------------------

::

    === ./G4OpBoundaryProcessTest.sh : script_cf ../../../qudarap/tests/propagate_at_boundary_cf.py
    a_key :               A_FOLD  A_FOLD : /tmp/blyth/opticks/QSimTest/propagate_at_boundary
    b_key :               B_FOLD  B_FOLD : /tmp/blyth/opticks/G4OpBoundaryProcessTest/propagate_at_boundary
    a.shape (100000, 4, 4) : /tmp/blyth/opticks/QSimTest/propagate_at_boundary/p.npy  
    b.shape (100000, 4, 4) : /tmp/blyth/opticks/G4OpBoundaryProcessTest/propagate_at_boundary/p.npy  
    aprd.shape  (1, 4, 4) : /tmp/blyth/opticks/QSimTest/propagate_at_boundary/prd.npy  

    aprd                                              : 
    [[[  0.   0.   1. 100.]
      [  0.   0.   0.   0.]
      [  0.   0.   0.   0.]
      [  0.   0.   0.   0.]]]
    a_flag=a[:,3,3].view(np.uint32)                    : [2048 2048 2048 ... 2048 2048 2048]
    b_flag=b[:,3,3].view(np.uint32)                    : [2048 2048 2048 ... 2048 2048 2048]
    w_flag=np.where( a_flag != b_flag )                : (array([], dtype=int64),)
    ua_flag=np.unique(a_flag, return_counts=True)      : (array([1024, 2048], dtype=uint32), array([ 3980, 96020]))
    ub_flag=np.unique(b_flag, return_counts=True)      : (array([1024, 2048], dtype=uint32), array([ 3980, 96020]))
    a_TransCoeff=a[:,1,3]                              : [0.96 0.96 0.96 ... 0.96 0.96 0.96]
    b_TransCoeff=b[:,1,3]                              : [0.96 0.96 0.96 ... 0.96 0.96 0.96]
    w_TransCoeff=np.where( np.abs( a_TransCoeff - b_TransCoeff) > 1e-6 ) : (array([], dtype=int64),)
    a_flat=a[:,0,3]                                    : [0.438 0.46  0.25  ... 0.202 0.053 0.44 ]
    b_flat=b[:,0,3]                                    : [0.438 0.46  0.25  ... 0.202 0.053 0.44 ]
    w_flat=np.where(a_flat != b_flat)                  : (array([], dtype=int64),)
    w_ab0=np.where( np.abs(a[:,0] - b[:,0]) > 1e-6 )   : (array([], dtype=int64), array([], dtype=int64))
    w_ab1=np.where( np.abs(a[:,1] - b[:,1]) > 1e-6 )   : (array([], dtype=int64), array([], dtype=int64))
    w_ab2=np.where( np.abs(a[:,2] - b[:,2]) > 1e-6 )   : (array([    0,     0,     1, ..., 99998, 99999, 99999]), array([0, 1, 0, ..., 1, 0, 1]))
    w_ab3=np.where( np.abs(a[:,3] - b[:,3]) > 1e-6 )   : (array([], dtype=int64), array([], dtype=int64))

    In [1]: w_ab2                                                                                                                                                                                   
    Out[1]: 
    (array([    0,     0,     1, ..., 99998, 99999, 99999]),
     array([0, 1, 0, ..., 1, 0, 1]))



    In [4]: a[:,2]                                                                                                                                                                                  
    Out[4]: 
    array([[  1.,   0.,   0., 500.],
           [  1.,   0.,   0., 500.],
           [  1.,   0.,   0., 500.],
           ...,
           [  1.,   0.,   0., 500.],
           [  1.,   0.,   0., 500.],
           [  1.,   0.,   0., 500.]], dtype=float32)

    In [5]: b[:,2]                                                                                                                                                                                  
    Out[5]: 
    array([[  0.,   1.,   0., 500.],
           [  0.,   1.,   0., 500.],
           [  0.,   1.,   0., 500.],
           ...,
           [  0.,   1.,   0., 500.],
           [  0.,   1.,   0., 500.],
           [  0.,   1.,   0., 500.]], dtype=float32)


Geant4 does not change polarization (or direction of course) at for transmission at normal incidence::

    In [6]: b                                                                                                                                                                                       
    Out[6]: 
    array([[[  0.   ,   0.   ,  -1.   ,   0.438],
            [  0.   ,   0.   ,  -1.   ,   0.96 ],
            [  0.   ,   1.   ,   0.   , 500.   ],
            [  0.   ,   1.   ,   0.   ,   0.   ]],

           [[  0.   ,   0.   ,  -1.   ,   0.46 ],
            [  0.   ,   0.   ,  -1.   ,   0.96 ],
            [  0.   ,   1.   ,   0.   , 500.   ],
            [  0.   ,   1.   ,   0.   ,   0.   ]],



* aligned this



Also for reflection at normal incidence Geant4 has a special case handling::


    1275 
    1276                     else {               // incident ray perpendicular
    1277 
    1278 #ifdef MOCK_DUMP
    1279               if( photon_idx == photon_idx_debug )
    1280               {       
    1281                       std::cout << " incident ray perpendicular  " << std::endl ;
    1282               }
    1283 #endif
    1284 
    1285 
    1286 
    1287                        if (Rindex2 > Rindex1) {
    1288                           NewPolarization = - OldPolarization;
    1289                        }
    1290                        else {
    1291                           NewPolarization =   OldPolarization;
    1292                        }
    1293 
    1294                     }
    1295                  }
    1296               }





