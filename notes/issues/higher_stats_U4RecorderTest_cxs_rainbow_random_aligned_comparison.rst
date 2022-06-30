higher_stats_U4RecorderTest_cxs_rainbow_random_aligned_comparison
========================================================================

* from :doc:`U4RecorderTest_U4Stack_stag_enum_random_alignment`


Overview : what is the purpose of random aligned comparison
-----------------------------------------------------------------

* the purpose of the random aligned comparison is to find and fix any unexpected problems (ie bugs) 
  that will manifest as array differences which cannot be explained

* also the level of difference is something that needs to be 
  reported as a characteristic of the Opticks simulation 

* so finding the levels and reasons behind differences is sufficient : the point in doing 
  the comparison is to find and investigate any unexplained differences 

* of course if the levels of difference can be reduced without costing performance 
  then could consider making changes : but typically reducing differences requires
  using double rather than float which is to be avoided if at all possible


TODO : more complex geometry eg single Geant4 PMT grabbed from PMTSim and translated 
----------------------------------------------------------------------------------------

* :doc:`more_complex_test_geometry_using_bringing_PMTSim_to_U4`


HMM : how much would replacing the double G4Log(G4UniformRand()) for Scattering and Absorption align SC and AB positions ?
----------------------------------------------------------------------------------------------------------------------------

* :doc:`U4LogTest_maybe_replacing_G4Log_G4UniformRand_in_Absorption_and_Scattering_with_float_version_will_avoid_deviations`


DONE : pump up the volume to 1M with the simple geometry
-------------------------------------------------------------

* wow thats real heavy on Geant4 side, taking several hours on laptop 

  * as do not want to spend the time and energy to recreate this sample have added ~/opticks/bin/AB_FOLD_COPY.sh 
    and used it to copy the A_FOLD B_FOLD to more permanant KEEP location
    that ~/opticks/bin/AB_FOLD.sh returns when FOLD_MODE is set to KEEP rather 
    than the default of TMP

* tagging every random consumption via backtraces and storing all the randoms is rather intensive when push to 1M  
* will need to exclude the indices where not enough randoms

1/1M is not history aligned, BR<->BT from float/double random sitting either side of TransCoeff knife edge::

    In [11]: wq = np.where( a.seq[:,0] != b.seq[:,0] )[0] ; wq
    Out[11]: array([726637])

    In [12]: seqhis_(a.seq[wq,0])
    Out[12]: ['TO BT BR BR BT SA']

    In [13]: seqhis_(b.seq[wq,0])
    Out[13]: ['TO BT BT SA']

    In [15]: A(wq[0])
    Out[15]: 
    A(726637) : TO BT BR BR BT SA
           A.t : (1000000, 48) 
           A.n : (1000000,) 
          A.ts : (1000000, 10, 44) 
          A.fs : (1000000, 10, 44) 
         A.ts2 : (1000000, 10, 44) 
     0 :     0.7496 :  1 :     to_sci : qsim::propagate_to_boundary u_to_sci burn 
     1 :     0.9443 :  2 :     to_bnd : qsim::propagate_to_boundary u_to_bnd burn 
     2 :     0.7756 :  3 :     to_sca : qsim::propagate_to_boundary u_scattering 
     3 :     0.3336 :  4 :     to_abs : qsim::propagate_to_boundary u_absorption 
     4 :     0.4643 :  5 : at_burn_sf_sd : at_boundary_burn at_surface ab/sd  
     5 :     0.5304 :  6 :     at_ref : u_reflect > TransCoeff 

     6 :     0.7131 :  1 :     to_sci : qsim::propagate_to_boundary u_to_sci burn 
     7 :     0.1302 :  2 :     to_bnd : qsim::propagate_to_boundary u_to_bnd burn 
     8 :     0.1077 :  3 :     to_sca : qsim::propagate_to_boundary u_scattering 
     9 :     0.2754 :  4 :     to_abs : qsim::propagate_to_boundary u_absorption 
    10 :     0.6640 :  5 : at_burn_sf_sd : at_boundary_burn at_surface ab/sd  
    11 :     0.6240 :  6 :     at_ref : u_reflect > TransCoeff 

    12 :     0.5618 :  1 :     to_sci : qsim::propagate_to_boundary u_to_sci burn 
    13 :     0.6591 :  2 :     to_bnd : qsim::propagate_to_boundary u_to_bnd burn 
    14 :     0.6729 :  3 :     to_sca : qsim::propagate_to_boundary u_scattering 
    15 :     0.3685 :  4 :     to_abs : qsim::propagate_to_boundary u_absorption 

    16 :     0.9081 :  1 :     to_sci : qsim::propagate_to_boundary u_to_sci burn 
    17 :     0.1008 :  2 :     to_bnd : qsim::propagate_to_boundary u_to_bnd burn 
    18 :     0.8054 :  3 :     to_sca : qsim::propagate_to_boundary u_scattering 
    19 :     0.6100 :  4 :     to_abs : qsim::propagate_to_boundary u_absorption 
    20 :     0.4183 :  5 : at_burn_sf_sd : at_boundary_burn at_surface ab/sd  
    21 :     0.6362 :  6 :     at_ref : u_reflect > TransCoeff 

    22 :     0.6149 :  1 :     to_sci : qsim::propagate_to_boundary u_to_sci burn 
    23 :     0.9692 :  2 :     to_bnd : qsim::propagate_to_boundary u_to_bnd burn 
    24 :     0.8735 :  3 :     to_sca : qsim::propagate_to_boundary u_scattering 
    25 :     0.7992 :  4 :     to_abs : qsim::propagate_to_boundary u_absorption 

    26 :     0.2129 :  1 :     to_sci : qsim::propagate_to_boundary u_to_sci burn 
    27 :     0.2093 :  2 :     to_bnd : qsim::propagate_to_boundary u_to_bnd burn 
    28 :     0.8324 :  3 :     to_sca : qsim::propagate_to_boundary u_scattering 
    29 :     0.7697 :  4 :     to_abs : qsim::propagate_to_boundary u_absorption 
    30 :     0.7639 :  5 : at_burn_sf_sd : at_boundary_burn at_surface ab/sd  
    31 :     0.1712 :  6 :     at_ref : u_reflect > TransCoeff 

    32 :     0.2939 :  1 :     to_sci : qsim::propagate_to_boundary u_to_sci burn 
    33 :     0.5738 :  2 :     to_bnd : qsim::propagate_to_boundary u_to_bnd burn 
    34 :     0.9891 :  3 :     to_sca : qsim::propagate_to_boundary u_scattering 
    35 :     0.2023 :  4 :     to_abs : qsim::propagate_to_boundary u_absorption 
    36 :     0.7197 :  5 : at_burn_sf_sd : at_boundary_burn at_surface ab/sd  
    37 :     0.6063 :  7 :    sf_burn : qsim::propagate_at_surface burn 
    38 :     0.0000 :  0 :      undef : undef 
    39 :     0.0000 :  0 :      undef : undef 

    In [16]: B(wq[0])
    Out[16]: 
    B(726637) : TO BT BT SA
           B.t : (1000000, 48) 
           B.n : (1000000,) 
          B.ts : (1000000, 10, 44) 
          B.fs : (1000000, 10, 44) 
         B.ts2 : (1000000, 10, 44) 
     0 :     0.7496 :  3 : ScintDiscreteReset :  
     1 :     0.9443 :  4 : BoundaryDiscreteReset :  
     2 :     0.7756 :  5 : RayleighDiscreteReset :  
     3 :     0.3336 :  6 : AbsorptionDiscreteReset :  
     4 :     0.4643 :  7 : BoundaryBurn_SurfaceReflectTransmitAbsorb :  
     5 :     0.5304 :  8 : BoundaryDiDiTransCoeff :  

     6 :     0.7131 :  3 : ScintDiscreteReset :  
     7 :     0.1302 :  4 : BoundaryDiscreteReset :  
     8 :     0.1077 :  5 : RayleighDiscreteReset :  
     9 :     0.2754 :  6 : AbsorptionDiscreteReset :  
    10 :     0.6640 :  7 : BoundaryBurn_SurfaceReflectTransmitAbsorb :  
    11 :     0.6240 :  8 : BoundaryDiDiTransCoeff :           ######## THIS IS WHERE BR/BT HISTORY DIVERGES 

    12 :     0.5618 :  3 : ScintDiscreteReset :  
    13 :     0.6591 :  4 : BoundaryDiscreteReset :  
    14 :     0.6729 :  5 : RayleighDiscreteReset :  
    15 :     0.3685 :  6 : AbsorptionDiscreteReset :  
    16 :     0.9081 :  7 : BoundaryBurn_SurfaceReflectTransmitAbsorb :  
    17 :     0.1008 :  9 : AbsorptionEffDetect :  
    18 :     0.0000 :  0 : Unclassified :  
    19 :     0.0000 :  0 : Unclassified :  

::

    N[blyth@localhost CSGOptiX]$ PIDX=726637 ./cxs_raindrop.sh 
    ...
    //qsim.propagate idx 726637 bnc 0 cosTheta    -0.2235 dir (    0.0000     0.0000     1.0000) nrm (   -0.9217    -0.3169    -0.2235) 
    //qsim.propagate idx 726637 bounce 0 command 3 flag 0 s.optical.x 0 
    //qsim.propagate_at_boundary idx 726637 nrm   (   -0.9217    -0.3169    -0.2235) 
    //qsim.propagate_at_boundary idx 726637 mom_0 (    0.0000     0.0000     1.0000) 
    //qsim.propagate_at_boundary idx 726637 pol_0 (   -0.3252     0.9457     0.0000) 
    //qsim.propagate_at_boundary idx 726637 c1     0.2235 normal_incidence 0 
    //qsim.propagate_at_boundary idx 726637 normal_incidence 0 p.pol (   -0.3252,    0.9457,    0.0000) p.mom (    0.0000,    0.0000,    1.0000) o_normal (   -0.9217,   -0.3169,   -0.2235)
    //qsim.propagate_at_boundary idx 726637 TransCoeff     0.6240 n1c1     0.2236 n2c2     0.9325 E2_t (   -0.3868,    0.0000) A_trans (    0.3252,   -0.9457,    0.0000) 
    //qsim.propagate_at_boundary idx 726637 u_boundary_burn     0.4643 u_reflect     0.5304 TransCoeff     0.6240 reflect 0 
    //qsim.propagate_at_boundary idx 726637 reflect 0 tir 0 TransCoeff     0.6240 u_reflect     0.5304 
    //qsim.propagate_at_boundary idx 726637 mom_1 (    0.4843     0.1665     0.8589) 
    //qsim.propagate_at_boundary idx 726637 pol_1 (   -0.3252     0.9457    -0.0000) 
    //qsim.propagate idx 726637 bnc 1 cosTheta     0.6912 dir (    0.4843     0.1665     0.8589) nrm (   -0.2522    -0.0867     0.9638) 
    //qsim.propagate idx 726637 bounce 1 command 3 flag 0 s.optical.x 0 
    //qsim.propagate_at_boundary idx 726637 nrm   (    0.2522     0.0867    -0.9638) 
    //qsim.propagate_at_boundary idx 726637 mom_0 (    0.4843     0.1665     0.8589) 
    //qsim.propagate_at_boundary idx 726637 pol_0 (   -0.3252     0.9457    -0.0000) 
    //qsim.propagate_at_boundary idx 726637 c1     0.6912 normal_incidence 0 
    //qsim.propagate_at_boundary idx 726637 normal_incidence 0 p.pol (   -0.3252,    0.9457,   -0.0000) p.mom (    0.4843,    0.1665,    0.8589) o_normal (    0.2522,    0.0867,   -0.9638)
    //qsim.propagate_at_boundary idx 726637 TransCoeff     0.6240 n1c1     0.9325 n2c2     0.2236 E2_t (    1.6132,    0.0000) A_trans (   -0.3252,    0.9457,    0.0000) 
    //qsim.propagate_at_boundary idx 726637 u_boundary_burn     0.6640 u_reflect     0.6240 TransCoeff     0.6240 reflect 1 

    ######  u_reflect is on the TransCoeff cut edge 

    //qsim.propagate_at_boundary idx 726637 reflect 1 tir 0 TransCoeff     0.6240 u_reflect     0.6240 
    //qsim.propagate_at_boundary idx 726637 mom_1 (    0.8330     0.2864    -0.4734) 
    //qsim.propagate_at_boundary idx 726637 pol_1 (   -0.3252     0.9457     0.0000) 
    //qsim.propagate idx 726637 bnc 2 cosTheta     0.6912 dir (    0.8330     0.2864    -0.4734) nrm (    0.8993     0.3092     0.3093) 
    //qsim.propagate idx 726637 bounce 2 command 3 flag 0 s.optical.x 0 
    //qsim.propagate_at_boundary idx 726637 nrm   (   -0.8993    -0.3092    -0.3093) 
    //qsim.propagate_at_boundary idx 726637 mom_0 (    0.8330     0.2864    -0.4734) 
    //qsim.propagate_at_boundary idx 726637 pol_0 (   -0.3252     0.9457     0.0000) 
    //qsim.propagate_at_boundary idx 726637 c1     0.6912 normal_incidence 0 
    //qsim.propagate_at_boundary idx 726637 normal_incidence 0 p.pol (   -0.3252,    0.9457,    0.0000) p.mom (    0.8330,    0.2864,   -0.4734) o_normal (   -0.8993,   -0.3092,   -0.3093)
    //qsim.propagate_at_boundary idx 726637 TransCoeff     0.6240 n1c1     0.9325 n2c2     0.2236 E2_t (    1.6132,    0.0000) A_trans (   -0.3252,    0.9457,    0.0000) 
    //qsim.propagate_at_boundary idx 726637 u_boundary_burn     0.4183 u_reflect     0.6362 TransCoeff     0.6240 reflect 1 
    //qsim.propagate_at_boundary idx 726637 reflect 1 tir 0 TransCoeff     0.6240 u_reflect     0.6362 
    //qsim.propagate_at_boundary idx 726637 mom_1 (   -0.4102    -0.1411    -0.9010) 
    //qsim.propagate_at_boundary idx 726637 pol_1 (   -0.3252     0.9457    -0.0000) 
    //qsim.propagate idx 726637 bnc 3 cosTheta     0.6912 dir (   -0.4102    -0.1411    -0.9010) nrm (    0.3322     0.1142    -0.9363) 
    //qsim.propagate idx 726637 bounce 3 command 3 flag 0 s.optical.x 0 
    //qsim.propagate_at_boundary idx 726637 nrm   (   -0.3322    -0.1142     0.9363) 
    //qsim.propagate_at_boundary idx 726637 mom_0 (   -0.4102    -0.1411    -0.9010) 
    //qsim.propagate_at_boundary idx 726637 pol_0 (   -0.3252     0.9457    -0.0000) 
    //qsim.propagate_at_boundary idx 726637 c1     0.6912 normal_incidence 0 
    //qsim.propagate_at_boundary idx 726637 normal_incidence 0 p.pol (   -0.3252,    0.9457,   -0.0000) p.mom (   -0.4102,   -0.1411,   -0.9010) o_normal (   -0.3322,   -0.1142,    0.9363)
    //qsim.propagate_at_boundary idx 726637 TransCoeff     0.6240 n1c1     0.9325 n2c2     0.2236 E2_t (    1.6132,    0.0000) A_trans (   -0.3252,    0.9457,    0.0000) 
    //qsim.propagate_at_boundary idx 726637 u_boundary_burn     0.7639 u_reflect     0.1712 TransCoeff     0.6240 reflect 0 
    //qsim.propagate_at_boundary idx 726637 reflect 0 tir 0 TransCoeff     0.6240 u_reflect     0.1712 
    //qsim.propagate_at_boundary idx 726637 mom_1 (   -0.7887    -0.2712    -0.5517) 
    //qsim.propagate_at_boundary idx 726637 pol_1 (   -0.3252     0.9457    -0.0000) 
    //qsim.propagate idx 726637 bnc 4 cosTheta     0.7887 dir (   -0.7887    -0.2712    -0.5517) nrm (   -1.0000     0.0000     0.0000) 
    //qsim.propagate idx 726637 bounce 4 command 3 flag 0 s.optical.x 99 
    2022-06-30 02:26:47.383 INFO  [147639] [SEvt::save@1089] DefaultDir /tmp/blyth/opticks/GeoChain/BoxedSphere/CXRaindropTest


Deviants mostly have SC or AB or lots of BR or truncation::

    In [3]: w = np.unique(np.where( np.abs(a.photon - b.photon) > 0.1 )[0])
    In [5]: len(w)
    Out[5]: 503              ######### 503/1M with > 0.1 deviants 
    In [6]: s = a.seq[w,0]
    In [7]: o = cuss(s,w)                                                                                                                                                                                   
    In [8]: o
    Out[8]: 
    CUSS([['w0', '                TO BT SC BT SA', '          575181', '             141'],
          ['w1', '                   TO BT BT AB', '           19661', '              93'],
          ['w2', '                         TO AB', '              77', '              82'],
          ['w3', '                      TO SC SA', '            2157', '              37'],
          ['w4', '                TO BT BT SC SA', '          552141', '              37'],
          ['w5', '                TO SC BT BT SA', '          576621', '              21'],
          ['w6', ' TO BT SC BR BR BR BR BR BR BR', '    806308525773', '              19'],
          ['w7', '                      TO BR AB', '            1213', '              15'],
          ['w8', '          TO BT BT SC BT BT SA', '       147614925', '              13'],
          ['w9', '             TO BT SC BR BT SA', '         9221837', '               8'],
          ['w10', ' TO BT BR BR BR BR BR BR BR BT', '    875028003789', '               6'],
          ['w11', '             TO BT BR SC BT SA', '         9202637', '               6'],
          ['w12', '                TO BT BR BT AB', '          314317', '               4'],
          ['w13', ' TO BT BR SC BR BR BR BR BR BR', '    806308506573', '               3'],
          ['w14', '                   TO BR SC SA', '           34493', '               3'],
          ['w15', ' TO BT BR BR BR BR BR BR BR BR', '    806308527053', '               2'],
          ['w16', '       TO SC BT BR BR BR BT SA', '      2361113709', '               2'],
          ['w17', '             TO BT BR BR BT AB', '         5028813', '               1'],
          ['w18', '       TO BT BR SC BR BR BT SA', '      2361093069', '               1'],
          ['w19', '             TO BT BR BR BT SA', '         9223117', '               1'],
          ['w20', '    TO BT SC BR BR BR BR BT SA', '     37777815245', '               1'],
          ['w21', '             TO BT SC BT SC SA', '         8832717', '               1'],
          ['w22', '                   TO SC BR SA', '           35693', '               1'],
          ['w23', '             TO BT BT SC BR SA', '         9137357', '               1'],
          ['w24', '    TO BT BT SC BT BR BR BT SA', '     37777861837', '               1'],
          ['w25', ' TO BT BR SC BR BR BR BR BR BT', '    875027983309', '               1'],
          ['w26', '          TO BT SC BR BR BT SA', '       147568333', '               1'],
          ['w27', '             TO SC BT BR BT SA', '         9223277', '               1']], dtype=object)


Checking in full sample can see that the most frequent categories do not have 
SC or AB in them::

    In [20]: cuss(a.seq[:,0])
    Out[20]: 
    CUSS([['w0', '                   TO BT BT SA', '           36045', '          883284'],
          ['w1', '                      TO BR SA', '            2237', '           59840'],
          ['w2', '                TO BT BR BT SA', '          576461', '           46165'],
          ['w3', '             TO BT BR BR BT SA', '         9223117', '            4714'],
          ['w4', '                      TO BT AB', '            1229', '            2179'],
          ['w5', '          TO BT BR BR BR BT SA', '       147569613', '             947'],
          ['w6', '                      TO SC SA', '            2157', '             917'],
          ['w7', '                TO BT BT SC SA', '          552141', '             907'],
          ['w8', '       TO BT BR BR BR BR BT SA', '      2361113549', '             218'],
          ['w9', '                TO BT SC BT SA', '          575181', '             187'],
          ['w10', '                   TO BT BR AB', '           19405', '             106'],
          ['w11', '                   TO BT BT AB', '           19661', '              93'],
          ['w12', '                         TO AB', '              77', '              82'],
          ['w13', '    TO BT BR BR BR BR BR BT SA', '     37777816525', '              71'],
          ['w14', '                   TO BR SC SA', '           34493', '              66'],
          ['w15', '             TO BT BR BT SC SA', '         8833997', '              53'],
          ['w16', '                TO SC BT BT SA', '          576621', '              25'],
          ['w17', ' TO BT BR BR BR BR BR BR BT SA', '    604445064141', '              24'],
          ['w18', ' TO BT SC BR BR BR BR BR BR BR', '    806308525773', '              19'],
          ['w19', '          TO BT BT SC BT BT SA', '       147614925', '              15'],
          ['w20', '                      TO BR AB', '            1213', '              15'],
          ['w21', '             TO BT BR SC BT SA', '         9202637', '              12'],
          ['w22', '                TO BT BR BR AB', '          310221', '              11'],
          ['w23', '             TO BT SC BR BT SA', '         9221837', '               8'],
          ['w24', ' TO BT BR BR BR BR BR BR BR BT', '    875028003789', '               6'],
          ['w25', '          TO BT BR BR BT SC SA', '       141343693', '               5'],
          ['w26', '                   TO SC SC SA', '           34413', '               4'],
          ['w27', '                TO BT BR BT AB', '          314317', '               4'],
          ['w28', '             TO BT BR BR BR AB', '         4963277', '               3'],
          ['w29', ' TO BT BR SC BR BR BR BR BR BR', '    806308506573', '               3'],
          ['w30', ' TO BT BR BR BR BR BR BR BR BR', '    806308527053', '               2'],
          ['w31', '       TO SC BT BR BR BR BT SA', '      2361113709', '               2'],
          ['w32', '             TO BT SC BT SC SA', '         8832717', '               1'],
          ['w33', '    TO BT BT SC BT BR BR BT SA', '     37777861837', '               1'],
          ['w34', '    TO BT SC BR BR BR BR BT SA', '     37777815245', '               1'],
          ['w35', '    TO BT BR BR BR BR BR BR AB', '     20329511885', '               1'],
          ['w36', '                   TO SC BR SA', '           35693', '               1'],
          ['w37', '       TO BT BR SC BR BR BT SA', '      2361093069', '               1'],
          ['w38', '             TO BT BR BR BT AB', '         5028813', '               1'],
          ['w39', '          TO SC BT BR BR BT SA', '       147569773', '               1'],
          ['w40', '             TO BT BT SC BR SA', '         9137357', '               1'],
          ['w41', '          TO BT SC BR BR BT SA', '       147568333', '               1'],
          ['w42', '          TO BT BR BR BR BR AB', '        79412173', '               1'],
          ['w43', '             TO SC BT BR BT SA', '         9223277', '               1'],
          ['w44', ' TO BT BR SC BR BR BR BR BR BT', '    875027983309', '               1']], dtype=object)






DONE : change geometry/input photon shape to avoid encouraging edge skimmers
---------------------------------------------------------------------------------------------------------------------------

Reduce the radius of the disc beam from 50 to 49 to avoid encouraging edge skimming on the sphere of radius 50. 
Avoiding the skimmers greatly reduces deviation, with only 4/10k now > 0.1 (down from 17/10k)::

    u4t
    ./U4RecorderTest.sh ab 

    In [1]: w = np.unique(np.where( np.abs(a.photon - b.photon) > 0.1 )[0]) ; s = a.seq[w,0] ; cuss(s,w)
    Out[1]: 
    CUSS([['w0', '                   TO BT BT AB', '           19661', '               2'],
          ['w1', '                TO BT SC BT SA', '          575181', '               1'],
          ['w2', '                   TO SC BR SA', '           35693', '               1'],
          ['w3', '                      TO SC SA', '            2157', '               1']], dtype=object)

* all the deviations are now due to either absorption position 
  or scattering position that then grows


w0 : TO BT BT AB  : deviation at the absorption position 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    In [6]: a.record[w0,3] - b.record[w0,3]
    Out[6]: 
    array([[[ 0.156, -0.051, -0.417, -0.001],
            [-0.   ,  0.   , -0.   ,  0.   ],
            [ 0.   , -0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   , -0.   ,  0.   ]],

           [[-0.181,  0.099, -0.425, -0.002],
            [-0.   ,  0.   ,  0.   ,  0.   ],
            [-0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   , -0.   ,  0.   ]]], dtype=float32)


w1 : TO BT SC BT SA : deviation starts from scatter position and grows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    In [9]: a.record[w1,:5] - b.record[w1,:5]
    Out[9]: 
    array([[[[ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]],

            [[ 0.   ,  0.   , -0.   , -0.   ],
             [-0.   , -0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]],

            [[-0.   , -0.   , -0.018, -0.   ],
             [-0.   , -0.   , -0.   ,  0.   ],
             [ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]],

            [[-0.   , -0.   , -0.018, -0.   ],
             [ 0.   ,  0.   ,  0.   ,  0.   ],
             [-0.   , -0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]],

            [[ 0.606,  0.221,  0.   ,  0.001],
             [ 0.   ,  0.   ,  0.   ,  0.   ],
             [-0.   , -0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]]]], dtype=float32)


w2 : TO SC BR SA : again deviation starting from scatter position that grows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    In [12]: a.record[w2,:4] - b.record[w2,:4]
    Out[12]: 
    array([[[[ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]],

            [[ 0.   ,  0.   , -0.047, -0.   ],
             [ 0.   , -0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]],

            [[-0.018,  0.049,  0.049,  0.   ],
             [ 0.   , -0.001,  0.003,  0.   ],
             [-0.   ,  0.   , -0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]],

            [[-0.221,  0.   ,  3.544,  0.005],
             [ 0.   , -0.001,  0.003,  0.   ],
             [-0.   ,  0.   , -0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]]]], dtype=float32)


w3 : TO SC SA : yet again deviation in scatter position that grows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    In [14]: a.record[w3,:3] - b.record[w3,:3]
    Out[14]: 
    array([[[[ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]],

            [[ 0.   ,  0.   , -0.048, -0.   ],
             [-0.   , -0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]],

            [[-0.316, -0.15 ,  0.   , -0.001],
             [-0.   , -0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]]]], dtype=float32)



Overall level of deviation reduced too::

    A_FOLD : /tmp/blyth/opticks/GeoChain/BoxedSphere/CXRaindropTest 
    B_FOLD : /tmp/blyth/opticks/U4RecorderTest 
    ./dv.sh   # cd ~/opticks/sysrap

                     pdv :         1e-06 1e-05  0.0001 0.001  0.01   0.1    1      10     100    1000    

                     pos : array([[   30,   125,  1778,  4518,  2751,   793,     4,     1,     0,     0],
                    time :        [ 2892,  5445,  1576,    83,     4,     0,     0,     0,     0,     0],
                     mom :        [ 6569,  2945,   484,     1,     1,     0,     0,     0,     0,     0],
                     pol :        [ 9994,     3,     0,     3,     0,     0,     0,     0,     0,     0],
                      wl :        [10000,     0,     0,     0,     0,     0,     0,     0,     0,     0]], dtype=uint32)

                     rdv :         1e-06 1e-05  0.0001 0.001  0.01   0.1    1      10     100    1000    

                     pos : array([[    5,    22,  1202,  5222,  2751,   793,     4,     1,     0,     0],
                    time :        [ 2871,  5464,  1570,    91,     4,     0,     0,     0,     0,     0],
                     mom :        [ 6555,  2959,   484,     1,     1,     0,     0,     0,     0,     0],
                     pol :        [ 9994,     3,     0,     3,     0,     0,     0,     0,     0,     0],
                      wl :        [10000,     0,     0,     0,     0,     0,     0,     0,     0,     0]], dtype=uint32)




DONE : systematic presentation of deviation level : opticks.sysrap.dv using opticks.ana.array_repr_mixin and sysrap/dv.sh
----------------------------------------------------------------------------------------------------------------------------

::

    A_FOLD : /tmp/blyth/opticks/GeoChain/BoxedSphere/CXRaindropTest 
    B_FOLD : /tmp/blyth/opticks/U4RecorderTest 
    ./dv.sh   # cd ~/opticks/sysrap

                     pdv :         1e-06 1e-05  0.0001 0.001  0.01   0.1    1      10     100    1000    

                     pos : array([[   47,   117,  1732,  4412,  2710,   965,    16,     1,     0,     0],
                    time :        [ 2746,  5430,  1724,    96,     4,     0,     0,     0,     0,     0],
                     mom :        [ 6404,  2937,   647,    11,     1,     0,     0,     0,     0,     0],
                     pol :        [ 9995,     1,     1,     3,     0,     0,     0,     0,     0,     0],
                      wl :        [10000,     0,     0,     0,     0,     0,     0,     0,     0,     0]], dtype=uint32)

                     rdv :         1e-06 1e-05  0.0001 0.001  0.01   0.1    1      10     100    1000    

                     pos : array([[    4,    25,  1124,  5155,  2710,   965,    16,     1,     0,     0],
                    time :        [ 2732,  5441,  1719,   104,     4,     0,     0,     0,     0,     0],
                     mom :        [ 6388,  2953,   647,    11,     1,     0,     0,     0,     0,     0],
                     pol :        [ 9995,     1,     1,     3,     0,     0,     0,     0,     0,     0],
                      wl :        [10000,     0,     0,     0,     0,     0,     0,     0,     0,     0]], dtype=uint32)



* review what was done in old workflow ab.py and cherrypick 
* ana/ab.py not easy to cherry pick from : until have a specific need which can go hunt for, like amax::

    1286     def rpost_dv_where(self, cut):
    1287         """
    1288         :return photon indices with item deviations exceeding the cut: 
    1289         """
    1290         av = self.a.rpost()
    1291         bv = self.b.rpost()
    1292         dv = np.abs( av - bv )
    1293         return self.a.where[np.where(dv.max(axis=(1,2)) > cut) ]
    1294 

* in redoing : focus on generic handling, so can do more with less code more systematically 

A general requirement is to know the deviation profile of various quantities::

    wseq = np.where( a.seq[:,0] == b.seq[:,0] )     
    abp = np.abs( a.photon[wseq] - b.photon[wseq] )  ## for deviations to be meaningful needs to be same history  

    abp_pos  = np.amax( abp[:,0,:3], axis=1 )        ## amax of the 3 position deviations, so can operate at photon position level, not x,y,z level 
    abp_time = abp[:,0,3]
    abp_mom  = np.amax( abp[:,1,:3], axis=1 )
    abp_pol  = np.amax( abp[:,2,:3], axis=1 )

    assert abp_pos.shape == abp_time.shape == abp_mom.shape == abp_pol.shape

So it comes down to histogramming bin count frequencies of an array with lots of small values.::

   bins = np.array( [0.,1e-6,1e-5,1e-4,1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], dtype=np.float32 )  
   prof, bins2 = np.histogram( abp_pos, bins=bins )
   

DONE : Pumped up the volume to 10,000 with raindrop geometry using box factor 10. 
------------------------------------------------------------------------------------

Surprised to find the 10k are fully history aligned without any more work when including scatter from the higher stats::

    In [2]: np.where( a.seq[:,0] != b.seq[:,0] )
    Out[2]: (array([], dtype=int64),)

Substantial deviation::

    In [6]: np.abs( a.photon - b.photon ).max()
    Out[6]: 4.0538635

    In [7]: np.abs( a.record - b.record ).max()
    Out[7]: 4.0538635


    In [13]: np.where( np.abs(a.photon - b.photon) > 0.1 )
    Out[13]: 
    (array([ 675,  911, 1355, 1355, 1957, 2293, 2436, 2436, 2597, 4029, 5156, 5156, 5208, 5208, 7203, 7203, 7628, 7781, 8149, 8393, 8393, 8393, 9516, 9964, 9964]),
     array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
     array([1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 2, 0, 2, 0, 1, 2, 1, 1, 0, 1, 2, 0, 0, 1]))

    In [50]: w = np.where( np.abs(a.photon - b.photon) > 0.1 )[0] ; w
    Out[50]: array([ 675,  911, 1355, 1355, 1957, 2293, 2436, 2436, 2597, 4029, 5156, 5156, 5208, 5208, 7203, 7203, 7628, 7781, 8149, 8393, 8393, 8393, 9516, 9964, 9964])

    In [88]: w = np.unique(np.where( np.abs(a.photon - b.photon) > 0.1 )[0] ) ; w   ## need to unique it to avoid same photon index appearing multiple times
    Out[88]: array([ 675,  911, 1355, 1957, 2293, 2436, 2597, 4029, 5156, 5208, 7203, 7628, 7781, 8149, 8393, 9516, 9964])

    In [89]: seqhis_(a.seq[w,0])
    Out[89]: 
    ['TO BR SA',
     'TO BR SA',
     'TO BR SA',
     'TO BR SA',
     'TO BR SA',
     'TO BR SA',
     'TO BR SA',
     'TO BR SA',
     'TO SC BR SA',
     'TO BT BT AB',
     'TO SC SA',
     'TO BT BR BR BR BR BT SA',
     'TO BR SA',
     'TO BR SA',
     'TO BT BT AB',
     'TO BR SA',
     'TO BT SC BT SA']


more systematic look at 17/10k > 0.1 mm deviants (~1 in a thousand level) using ana/p.py:cuss 
---------------------------------------------------------------------------------------------------

::

    In [66]: w = np.where( np.abs(a.photon - b.photon) > 0.1 )[0] ; w
    Out[66]: array([ 675,  911, 1355, 1355, 1957, 2293, 2436, 2436, 2597, 4029, 5156, 5156, 5208, 5208, 7203, 7203, 7628, 7781, 8149, 8393, 8393, 8393, 9516, 9964, 9964])


    In [10]: cuss(s,w)
    Out[10]: 
    CUSS([['w0', '                      TO BR SA', '            2237', '              11'],
          ['w1', '                   TO BT BT AB', '           19661', '               2'],
          ['w2', '       TO BT BR BR BR BR BT SA', '      2361113549', '               1'],
          ['w3', '                TO BT SC BT SA', '          575181', '               1'],
          ['w4', '                   TO SC BR SA', '           35693', '               1'],
          ['w5', '                      TO SC SA', '            2157', '               1']], dtype=object)


::

     w = np.where( np.abs(a.photon - b.photon) > 0.1 )[0] ; s = a.seq[w,0] ; cuss(s,w)

In summary::

    In [28]: w = np.unique(np.where( np.abs(a.photon - b.photon) > 0.1 )[0]) ; s = a.seq[w,0] ; cuss(s,w)
    Out[28]: 
    CUSS([['w0', '                      TO BR SA', '            2237', '              11'],
          ['w1', '                   TO BT BT AB', '           19661', '               2'],
          ['w2', '       TO BT BR BR BR BR BT SA', '      2361113549', '               1'],
          ['w3', '                TO BT SC BT SA', '          575181', '               1'],
          ['w4', '                   TO SC BR SA', '           35693', '               1'],
          ['w5', '                      TO SC SA', '            2157', '               1']], dtype=object)



    In [1]: cuss(a.seq[:,0])
    Out[1]: 
    CUSS([['w0', '                   TO BT BT SA', '           36045', '            8653'],
          ['w1', '                      TO BR SA', '            2237', '             691'],
          ['w2', '                TO BT BR BT SA', '          576461', '             513'],
          ['w3', '             TO BT BR BR BT SA', '         9223117', '              60'],
          ['w4', '                      TO BT AB', '            1229', '              27'],
          ['w5', '          TO BT BR BR BR BT SA', '       147569613', '              23'],
          ['w6', '                      TO SC SA', '            2157', '               9'],
          ['w7', '                TO BT BT SC SA', '          552141', '               7'],
          ['w8', '       TO BT BR BR BR BR BT SA', '      2361113549', '               4'],
          ['w9', '                TO BT SC BT SA', '          575181', '               2'],
          ['w10', '                   TO BR SC SA', '           34493', '               2'],
          ['w11', '                   TO BT BT AB', '           19661', '               2'],
          ['w12', '                   TO BT BR AB', '           19405', '               2'],
          ['w13', '             TO BT BR BT SC SA', '         8833997', '               2'],
          ['w14', '    TO BT BR BR BR BR BR BT SA', '     37777816525', '               1'],
          ['w15', '                   TO SC BR SA', '           35693', '               1'],
          ['w16', ' TO BT BR BR BR BR BR BR BT SA', '    604445064141', '               1']], dtype=object)



Summary of > 0.1 mm deviants : skimmers and absorption/scatter distance diff : these are expected float/double differences
-----------------------------------------------------------------------------------------------------------------------------

::

    In [28]: w = np.unique(np.where( np.abs(a.photon - b.photon) > 0.1 )[0]) ; s = a.seq[w,0] ; cuss(s,w)
    Out[28]: 
    CUSS([['w0', '                      TO BR SA', '            2237', '              11'],          ## skimmers  
          ['w1', '                   TO BT BT AB', '           19661', '               2'],          ## absorption position
          ['w2', '       TO BT BR BR BR BR BT SA', '      2361113549', '               1'],          ## lots of bounces 
          ['w3', '                TO BT SC BT SA', '          575181', '               1'],          ## scatter position 
          ['w4', '                   TO SC BR SA', '           35693', '               1'],          ## scatter position 
          ['w5', '                      TO SC SA', '            2157', '               1']], dtype=object)  ## scatter position 



w0 : TO BR SA : > 0.1 mm deviants from 10k sample : they are all tangential grazing incidence edge skimmers
---------------------------------------------------------------------------------------------------------------

::

    In [19]: seqhis_(a.seq[w0,0])
    Out[19]: 
    ['TO BR SA',
     'TO BR SA',
     'TO BR SA',
     'TO BR SA',
     'TO BR SA',
     'TO BR SA',
     'TO BR SA',
     'TO BR SA',
     'TO BR SA',
     'TO BR SA',
     'TO BR SA']

These BR all end up at top ? Edge skimmer ?::

    In [12]: a.record[w0,:3,0]
    Out[12]: 
    array([[[   1.403,  -49.872, -990.   ,    0.   ],
            [   1.403,  -49.872,   -3.279,    3.292],
            [   5.126, -182.258, 1000.   ,    6.669]],

           [[  43.282,  -24.992, -990.   ,    0.   ],
            [  43.282,  -24.992,   -1.458,    3.298],
            [  93.917,  -54.23 , 1000.   ,    6.645]],

           [[ -38.393,   31.995, -990.   ,    0.   ],
            [ -38.393,   31.995,   -1.521,    3.298],
            [ -85.258,   71.05 , 1000.   ,    6.646]],

           [[ -22.29 ,   44.614, -990.   ,    0.   ],
            [ -22.29 ,   44.614,   -3.579,    3.291],
            [ -87.009,  174.153, 1000.   ,    6.674]],

           [[ -49.146,   -8.528, -990.   ,    0.   ],
            [ -49.146,   -8.528,   -3.455,    3.292],
            [-186.776,  -32.411, 1000.   ,    6.672]],

           [[  15.008,  -47.688, -990.   ,    0.   ],
            [  15.008,  -47.688,   -0.829,    3.3  ],
            [  24.977,  -79.366, 1000.   ,    6.642]],

           [[  -0.671,  -49.849, -990.   ,    0.   ],
            [  -0.671,  -49.849,   -3.824,    3.29 ],
            [  -2.756, -204.756, 1000.   ,    6.679]],

           [[ -47.523,  -15.129, -990.   ,    0.   ],
            [ -47.523,  -15.129,   -3.553,    3.291],
            [-184.473,  -58.728, 1000.   ,    6.674]],

           [[  -0.895,   49.92 , -990.   ,    0.   ],
            [  -0.895,   49.92 ,   -2.669,    3.294],
            [  -2.823,  157.42 , 1000.   ,    6.659]],

           [[  19.233,   46.065, -990.   ,    0.   ],
            [  19.233,   46.065,   -2.839,    3.294],
            [  63.329,  151.683, 1000.   ,    6.661]],

           [[  46.313,  -17.856, -990.   ,    0.   ],
            [  46.313,  -17.856,   -6.021,    3.283],
            [ 277.431, -106.965, 1000.   ,    6.74 ]]], dtype=float32)


    In [15]: a.record[w0[0],:3]  - b.record[w0[0],:3]
    Out[15]: 
    array([[[ 0.   ,  0.   ,  0.   ,  0.   ],
        [ 0.   ,  0.   ,  0.   ,  0.   ],
        [ 0.   ,  0.   ,  0.   ,  0.   ],
        [ 0.   ,  0.   , -0.   ,  0.   ]],

       [[ 0.   ,  0.   ,  0.004,  0.   ],
        [-0.   ,  0.   ,  0.   ,  0.   ],
        [-0.   , -0.   , -0.   ,  0.   ],
        [ 0.   ,  0.   , -0.   ,  0.   ]],

       [[-0.005,  0.165,  0.   , -0.   ],
        [-0.   ,  0.   ,  0.   ,  0.   ],
        [-0.   , -0.   , -0.   ,  0.   ],
        [ 0.   ,  0.   , -0.   ,  0.   ]]], dtype=float32)

    In [16]: a.record[w0[1],:3]  - b.record[w0[1],:3]
    Out[16]: 
    array([[[ 0.   ,  0.   ,  0.   ,  0.   ],
        [ 0.   ,  0.   ,  0.   ,  0.   ],
        [ 0.   ,  0.   ,  0.   ,  0.   ],
        [ 0.   ,  0.   , -0.   ,  0.   ]],

       [[ 0.   ,  0.   , -0.004, -0.   ],
        [ 0.   , -0.   , -0.   ,  0.   ],
        [-0.   , -0.   , -0.   ,  0.   ],
        [ 0.   ,  0.   , -0.   ,  0.   ]],

       [[ 0.134, -0.077, -0.   ,  0.   ],
        [ 0.   , -0.   , -0.   ,  0.   ],
        [-0.   , -0.   , -0.   ,  0.   ],
        [ 0.   ,  0.   , -0.   ,  0.   ]]], dtype=float32)

radius of 50 does not shows its a tangent edge skimmer, just shows sphere intersect, see below need to check xy::

    In [38]: np.sqrt(np.sum(xpos*xpos,axis=1))
    Out[38]: array([ 991.261,   50.   , 1003.455], dtype=float32)

    In [65]: seqhis_(a.seq[w0,0]) 
    Out[65]: 
    ['TO BR SA',
     'TO BR SA',
     'TO BR SA',
     'TO BR SA',
     'TO BR SA',
     'TO BR SA',
     'TO BR SA',
     'TO BR SA',
     'TO BR SA',
     'TO BR SA',
     'TO BR SA',
     'TO BR SA',
     'TO BR SA']

    In [20]: a.record[w0,1,0,:3]
    Out[20]: 
    array([[  1.403, -49.872,  -3.279],
           [ 43.282, -24.992,  -1.458],
           [-38.393,  31.995,  -1.521],
           [-22.29 ,  44.614,  -3.579],
           [-49.146,  -8.528,  -3.455],
           [ 15.008, -47.688,  -0.829],
           [ -0.671, -49.849,  -3.824],
           [-47.523, -15.129,  -3.553],
           [ -0.895,  49.92 ,  -2.669],
           [ 19.233,  46.065,  -2.839],
           [ 46.313, -17.856,  -6.021]], dtype=float32)

    In [22]: a.record[w0,1,0,:3] - b.record[w0,1,0,:3]  ## deviation in z of intersect 
    Out[22]: 
    array([[ 0.   ,  0.   ,  0.004],
           [ 0.   ,  0.   , -0.004],
           [ 0.   ,  0.   , -0.006],
           [ 0.   ,  0.   , -0.003],
           [ 0.   ,  0.   , -0.003],
           [ 0.   ,  0.   , -0.018],
           [ 0.   ,  0.   ,  0.003],
           [ 0.   ,  0.   ,  0.003],
           [ 0.   ,  0.   ,  0.006],
           [ 0.   ,  0.   ,  0.005],
           [ 0.   ,  0.   ,  0.002]], dtype=float32)


    In [70]: x = a.record[ww,1,0,:3]

    In [71]: np.sqrt(np.sum(x*x,axis=1))
    Out[71]: array([50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50.], dtype=float32)


Actually the 50. does not say its an edge skimmer, any hit on the sphere will give that, need to look at xy::

    In [23]: xy = a.record[w0,1,0,:2]
    In [24]: xy
    Out[24]: 
    array([[  1.403, -49.872],
           [ 43.282, -24.992],
           [-38.393,  31.995],
           [-22.29 ,  44.614],
           [-49.146,  -8.528],
           [ 15.008, -47.688],
           [ -0.671, -49.849],
           [-47.523, -15.129],
           [ -0.895,  49.92 ],
           [ 19.233,  46.065],
           [ 46.313, -17.856]], dtype=float32)

    In [25]: np.sqrt(np.sum(xy*xy,axis=1))
    Out[25]: array([49.892, 49.979, 49.977, 49.872, 49.881, 49.993, 49.853, 49.873, 49.928, 49.919, 49.636], dtype=float32)

    In [26]: 50.-np.sqrt(np.sum(xy*xy,axis=1))
    Out[26]: array([0.108, 0.021, 0.023, 0.128, 0.119, 0.007, 0.147, 0.127, 0.072, 0.081, 0.364], dtype=float32)


Looking at the xy radius shows that these are photons hitting the sphere within around 0.1mm of its projected edge. 



w1 : TO BT BT AB : deviation all in the absorption position : known log(u_float) vs log(u_double) issue 
-----------------------------------------------------------------------------------------------------------

::

    In [9]: a.record[w1,:4] - b.record[w1,:4]
    Out[9]: 
    array([[[[ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]],

            [[ 0.   ,  0.   ,  0.   ,  0.   ],
             [-0.   ,  0.   , -0.   ,  0.   ],
             [-0.   ,  0.   , -0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]],

            [[-0.   ,  0.   ,  0.   , -0.   ],
             [-0.   ,  0.   , -0.   ,  0.   ],
             [ 0.   , -0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]],

            [[ 0.159, -0.053, -0.417, -0.001],
             [-0.   ,  0.   , -0.   ,  0.   ],
             [ 0.   , -0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]]],


           [[[ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]],

            [[ 0.   ,  0.   , -0.   , -0.   ],
             [-0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]],

            [[-0.   ,  0.   , -0.   , -0.   ],
             [-0.   ,  0.   ,  0.   ,  0.   ],
             [-0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]],

            [[-0.187,  0.102, -0.422, -0.002],
             [-0.   ,  0.   ,  0.   ,  0.   ],
             [-0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]]]], dtype=float32)




w2 : TO BT BR BR BR BR BT SA
--------------------------------


::

    In [28]: w = np.unique(np.where( np.abs(a.photon - b.photon) > 0.1 )[0]) ; s = a.seq[w,0] ; cuss(s,w)
    Out[28]: 
    CUSS([['w0', '                      TO BR SA', '            2237', '              11'],
          ['w1', '                   TO BT BT AB', '           19661', '               2'],
          ['w2', '       TO BT BR BR BR BR BT SA', '      2361113549', '               1'],
          ['w3', '                TO BT SC BT SA', '          575181', '               1'],
          ['w4', '                   TO SC BR SA', '           35693', '               1'],
          ['w5', '                      TO SC SA', '            2157', '               1']], dtype=object)



    In [33]: a.record[w2,:9] - b.record[w2,:9]
    Out[33]: 
    array([[[[ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]],

            [[ 0.   ,  0.   ,  0.003,  0.   ],
             [-0.   ,  0.   , -0.   ,  0.   ],
             [-0.   ,  0.   , -0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]],

            [[-0.002,  0.002,  0.   ,  0.   ],
             [ 0.   , -0.   , -0.   ,  0.   ],
             [-0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]],

            [[-0.001,  0.   , -0.003,  0.   ],
             [ 0.   , -0.   ,  0.   ,  0.   ],
             [-0.   ,  0.   , -0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]]

            [[ 0.002, -0.001, -0.001,  0.   ],
             [-0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   , -0.   , -0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]],

            [[ 0.001, -0.001,  0.002,  0.   ],
             [-0.   ,  0.   , -0.   ,  0.   ],
             [ 0.   , -0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]],

            [[-0.001,  0.001,  0.001,  0.   ],
             [-0.   ,  0.   , -0.   ,  0.   ],
             [ 0.   , -0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]],

            [[ 0.   ,  0.   , -0.171, -0.   ],     ### combination of small after 6 bounces on the sphere  
             [-0.   ,  0.   , -0.   ,  0.   ],
             [ 0.   , -0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]],

            [[ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   ,  0.   ,  0.   ]]]], dtype=float32)




w3 : TO BT SC BT SA : deviation starts from where the scatter happens
------------------------------------------------------------------------

::

    In [2]: w = np.unique(np.where( np.abs(a.photon - b.photon) > 0.1 )[0]) ; s = a.seq[w,0] ; cuss(s,w)
    Out[2]: 
    CUSS([['w0', '                      TO BR SA', '            2237', '              11'],
          ['w1', '                   TO BT BT AB', '           19661', '               2'],
          ['w2', '       TO BT BR BR BR BR BT SA', '      2361113549', '               1'],
          ['w3', '                TO BT SC BT SA', '          575181', '               1'],
          ['w4', '                   TO SC BR SA', '           35693', '               1'],
          ['w5', '                      TO SC SA', '            2157', '               1']], dtype=object)


    In [6]: a.record[w3,:5] - b.record[w3,:5]
    Out[6]: 
    array([[[[ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]],

            [[ 0.   ,  0.   , -0.   , -0.   ],
             [-0.   , -0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]],

            [[-0.   , -0.   , -0.018, -0.   ],
             [ 0.   , -0.   , -0.   ,  0.   ],
             [-0.   , -0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]],

            [[-0.   , -0.   , -0.018, -0.   ],
             [ 0.   ,  0.   ,  0.   ,  0.   ],
             [-0.   , -0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]],

            [[ 0.602,  0.219,  0.   ,  0.001],
             [ 0.   ,  0.   ,  0.   ,  0.   ],
             [-0.   , -0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]]]], dtype=float32)



w4 : "TO SC BR SA" : a 1/10k > 0.1 mm deviant : small scatter position diff gets lever armed into big diff
-------------------------------------------------------------------------------------------------------------------------

::

    In [10]: w = np.unique(np.where( np.abs(a.photon - b.photon) > 0.1 )[0]) ; s = a.seq[w,0] ; cuss(s,w) 
    Out[10]: 
    CUSS([['w0', '                      TO BR SA', '            2237', '              11'],
          ['w1', '                   TO BT BT AB', '           19661', '               2'],
          ['w2', '       TO BT BR BR BR BR BT SA', '      2361113549', '               1'],
          ['w3', '                TO BT SC BT SA', '          575181', '               1'],
          ['w4', '                   TO SC BR SA', '           35693', '               1'],
          ['w5', '                      TO SC SA', '            2157', '               1']], dtype=object)



* HMM: this is float/double difference in handling the calculation of scattering length

* I could reduce the difference by doing the log of rand calc in double precision 
  (did that previously in old workflow) but I am inclined to now say that there is no point in doing that : 
  where the scatter point is the result of the an random throw so worrying over the exact position is pointless

::

    In [7]: seqhis_(a.seq[w4,0])
    Out[7]: ['TO SC BR SA']


Initial 0.047 mm difference in scatter position gets lever armed into a larger deviations::

    In [9]:  a.record[w4,:4] - b.record[w4,:4]
    Out[9]: 
    array([[[[ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]],

            [[ 0.   ,  0.   , -0.047, -0.   ],
             [ 0.   , -0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]],

            [[-0.019,  0.052,  0.055,  0.   ],
             [ 0.   , -0.001,  0.003,  0.   ],
             [-0.   ,  0.   , -0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]],

            [[-0.249,  0.   ,  4.054,  0.006],
             [ 0.   , -0.001,  0.003,  0.   ],
             [-0.   ,  0.   , -0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]]]], dtype=float32)



w5 : TO SC SA : same again, difference in scattering length is cause
--------------------------------------------------------------------------

::

    In [10]: w = np.unique(np.where( np.abs(a.photon - b.photon) > 0.1 )[0]) ; s = a.seq[w,0] ; cuss(s,w) 
    Out[10]: 
    CUSS([['w0', '                      TO BR SA', '            2237', '              11'],
          ['w1', '                   TO BT BT AB', '           19661', '               2'],
          ['w2', '       TO BT BR BR BR BR BT SA', '      2361113549', '               1'],
          ['w3', '                TO BT SC BT SA', '          575181', '               1'],
          ['w4', '                   TO SC BR SA', '           35693', '               1'],
          ['w5', '                      TO SC SA', '            2157', '               1']], dtype=object)


    In [14]: a.record[w5,:3] - b.record[w5,:3]
    Out[14]: 
    array([[[[ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]],

            [[ 0.   ,  0.   , -0.048, -0.   ],
             [-0.   , -0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]],

            [[-0.316, -0.15 ,  0.   , -0.001],
             [-0.   , -0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]]]], dtype=float32)



Biggest > 0.5 mm deviants : skimmer and two scatters
-------------------------------------------------------

::

    In [18]: w = np.unique(np.where( np.abs(a.photon - b.photon) > 0.5 )[0]) ; w
    Out[18]: array([2436, 5156, 9964]
    In [20]: seqhis_(a.seq[w,0]) 
    Out[20]: ['TO BR SA', 'TO SC BR SA', 'TO BT SC BT SA']


