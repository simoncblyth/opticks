tboolean-proxy-scan-polarization
=====================================

Context
----------

* :doc:`tboolean-proxy-scan`


Overview
-----------

* so far all photons with discrepant polz have undergone
  several very close (but not quite) normal incidence BT and BR   

* the geometry LV 13 and 18 featuring spheres or ellipsoids with 
  close to pole reflections


Command shortcuts
---------------------

::

    lv(){ echo 21 ; }
    # default geometry LV index to test 

    ts(){  LV=${1:-$(lv)} tboolean.sh --align --dbgskipclearzero --dbgnojumpzero --dbgkludgeflatzero $* ; } 
    # **simulate** : aligned bi-simulation creating OK+G4 events 

    tv(){  LV=${1:-$(lv)} tboolean.sh --load $* ; } 
    # **visualize** : load events and visualize the propagation

    tv4(){  LV=${1:-$(lv)} tboolean.sh --load --vizg4 $* ; } 
    # **visualize** the geant4 propagation 

    ta(){  tboolean-;LV=${1:-$(lv)} tboolean-proxy-ip ; } 
    # **analyse** : load events and analyse the propagation



ISSUE : small numbers of photons with discrepant polarization
------------------------------------------------------------------

* what is special about these photons ? on the edge of critical angle or smth ?


LV:13 sTarget0x4bd4340
~~~~~~~~~~~~~~~~~~~~~~~~~~~~


ta 13::

    0002            :                 TO BT BR BT SA :     470      470  :         470      5640 :     2     1     1 : 0.0004 0.0002 0.0002 :    1.9908    0.0000    0.0007   :                FATAL :   > dvmax[2] 0.5000 


    In [1]: ab.aselhis = "TO BT BR BT SA"

    In [3]: a.rpol().shape
    Out[3]: (470, 5, 3)

    In [6]: ab.rpol_dv_where(1)   ## absolute photon index
    Out[6]: array([8511])

    In [3]: ab.rpol_dv_where_(1)   ## within rpol index
    Out[3]: (array([398]),)

::

    In [4]: a.rpol()[398]
    Out[4]: 
    A()sliced
    A([[ 0.    , -1.    ,  0.    ],                    TO 
       [ 0.    , -1.    ,  0.    ],                    BT
       [ 0.1339,  0.9921,  0.    ],                    BR      <<< Opticks changes polz on BR  
       [ 0.1339,  0.9921,  0.    ],                    BT      <<< but stays same on BT ??? 
       [ 0.1339,  0.9921,  0.    ]], dtype=float32)    SA

    In [5]: b.rpol()[398]
    Out[5]: 
    A()sliced
    A([[ 0.    , -1.    ,  0.    ],                    TO
       [ 0.    , -1.    ,  0.    ],                    BT
       [ 0.1339,  0.9921,  0.    ],                    BR 
       [ 0.    , -1.    ,  0.    ],                    BT      <<< G4 changing polz on BT ???
       [ 0.    , -1.    ,  0.    ]], dtype=float32)    SA




    In [8]: a.ox[398]
    Out[8]: 
    A()sliced
    A([[    -8.4037,   -123.8595, -53280.    ,    667.4361],
       [    -0.0001,     -0.0017,     -1.    ,      1.    ],
       [     0.1351,      0.9908,     -0.0025,    380.    ],
       [     0.    ,      0.    ,      0.    ,      0.    ]], dtype=float32)

    In [9]: b.ox[398]
    Out[9]: 
    A()sliced
    A([[    -8.4037,   -123.8595, -53280.    ,    667.4361],
       [    -0.0001,     -0.0017,     -1.    ,      1.    ],
       [     0.    ,     -1.    ,      0.0017,    380.    ],
       [     0.    ,      0.    ,      0.    ,      0.    ]], dtype=float32)




* hmm : makes me want to see these numbers without the compression 



::

     <solids>
        <sphere aunit="deg" deltaphi="360" deltatheta="180" lunit="mm" name="sTarget_bottom_ball0x4bd40d0" rmax="17700" rmin="0" startphi="0" starttheta="0"/>
        <tube aunit="deg" deltaphi="360" lunit="mm" name="sTarget_top_tube0x4bd4260" rmax="400" rmin="0" startphi="0" z="124.520352"/>
        <union name="sTarget0x4bd4340">
          <first ref="sTarget_bottom_ball0x4bd40d0"/>
          <second ref="sTarget_top_tube0x4bd4260"/>
          <position name="sTarget0x4bd4340_pos" unit="mm" x="0" y="0" z="17757.739824"/>
        </union>
      </solids>



LV:13 Geometry is a sphere with a squat cylinder protrusion at +Z, phtoon::

   ts 13 --mask 8511 --pindex 0 --pindexlog 

* no visible speckle in raytrace

Photon 8511, all BT and BR are at very close (but not quite) normal incidence
to the bottom pole of the sphere and the cylinder cap. 


::

    In [15]: a.rpost()[398]
    Out[15]: 
    A()sliced
    A([[     8.1303,    108.9458, -53279.3739,      0.    ],   TO
       [     8.1303,    108.9458, -17759.7913,    118.4908],   BT
       [     1.6261,     22.7648,  17759.7913,    333.7156],   BR
       [    -4.8782,    -65.0423, -17759.7913,   -532.8263],   BT            -ve times, viz will be messed up
       [    -8.1303,   -123.5803, -53279.3739,   -532.8263]])  SA

    In [16]: b.rpost()[398]
    Out[16]: 
    A()sliced
    A([[     8.1303,    108.9458, -53279.3739,      0.    ],
       [     8.1303,    108.9458, -17759.7913,    118.4908],
       [     1.6261,     22.7648,  17759.7913,    333.7156],
       [    -4.8782,    -65.0423, -17759.7913,   -532.8263],
       [    -8.1303,   -123.5803, -53279.3739,   -532.8263]])





Huh pindexlog empty for 8511::


    blyth@localhost location]$ l ox_*
    -rw-rw-r--. 1 blyth blyth 3201 Jun 23 23:04 ox_6368.log
    -rw-rw-r--. 1 blyth blyth    0 Jun 23 22:52 ox_8511.log
    -rw-rw-r--. 1 blyth blyth    0 Jun 23 21:32 ox_2301.log
    -rw-rw-r--. 1 blyth blyth 2162 Jun 23 20:59 ox_8021.log
    -rw-rw-r--. 1 blyth blyth 4750 Jun 23 20:01 ox_2180.log
    -rw-rw-r--. 1 blyth blyth 3716 Jun 23 19:41 ox_360.log
    -rw-rw-r--. 1 blyth blyth 5356 Jun 22 22:20 ox_5207.log

Rerunning creates it::


    [blyth@localhost opticks]$ cat /tmp/blyth/location/ox_8511.log
    WITH_ALIGN_DEV_DEBUG photon_id:0 bounce:0 
    propagate_to_boundary  u_OpBoundary:0.47524184 speed:299.79245 
    propagate_to_boundary  u_OpRayleigh:0.59458822   scattering_length(s.material1.z):1000000 scattering_distance:519886.188 
    propagate_to_boundary  u_OpAbsorption:0.493517905   absorption_length(s.material1.y):1e+09 absorption_distance:706196160 
    propagate_at_boundary  u_OpBoundary_DiDiTransCoeff:0.907800138  reflect:0   TransCoeff:   0.93847  c2c2:    1.0000 tir:0  pos (    7.4297   109.5039 -17759.6602)   

    WITH_ALIGN_DEV_DEBUG photon_id:0 bounce:0 
    propagate_to_boundary  u_OpBoundary:0.0701162666 speed:165.028061 
    propagate_to_boundary  u_OpRayleigh:0.609997571   scattering_length(s.material1.z):1000000 scattering_distance:494300.312 
    propagate_to_boundary  u_OpAbsorption:0.166104496   absorption_length(s.material1.y):1000000 absorption_distance:1795138.25 
    propagate_at_boundary  u_OpBoundary_DiDiTransCoeff:0.965329826  reflect:1   TransCoeff:   0.93847  c2c2:    1.0000 tir:0  pos (    1.5030    22.1526 17760.0000)   

    WITH_ALIGN_DEV_DEBUG photon_id:0 bounce:0 
    propagate_to_boundary  u_OpBoundary:0.887843072 speed:165.028061 
    propagate_to_boundary  u_OpRayleigh:0.536982119   scattering_length(s.material1.z):1000000 scattering_distance:621790.5 
    propagate_to_boundary  u_OpAbsorption:0.17540665   absorption_length(s.material1.y):1000000 absorption_distance:1740648.25 
    propagate_at_boundary  u_OpBoundary_DiDiTransCoeff:0.542280197  reflect:0   TransCoeff:   0.93847  c2c2:    1.0000 tir:0  pos (   -4.4237   -65.1992 -17759.8789)   

    WITH_ALIGN_DEV_DEBUG photon_id:0 bounce:0 
    propagate_to_boundary  u_OpBoundary:0.808059037 speed:299.79245 
    propagate_to_boundary  u_OpRayleigh:0.310746968   scattering_length(s.material1.z):1000000 scattering_distance:1168776.25 
    propagate_to_boundary  u_OpAbsorption:0.886376798   absorption_length(s.material1.y):1e+09 absorption_distance:120613136 
    propagate_at_surface   u_OpBoundary_DiDiReflectOrTransmit:        0.952486753 
    propagate_at_surface   u_OpBoundary_DoAbsorption:   0.780495644 
     WITH_ALIGN_DEV_DEBUG psave (-8.40369511 -123.85952 -53280 667.436096) ( 1, 0, 67305987, 7296 ) 
    [blyth@localhost opticks]$ 




::

    268 __device__ void propagate_at_boundary_geant4_style( Photon& p, State& s, curandState &rng)
    269 {
    270     // see g4op-/G4OpBoundaryProcess.cc annotations to follow this
    271 
    272     const float n1 = s.material1.x ;
    273     const float n2 = s.material2.x ;
    274     const float eta = n1/n2 ;
    275 
    276     const float c1 = -dot(p.direction, s.surface_normal ); // c1 arranged to be +ve   
    277     const float eta_c1 = eta * c1 ;
    278 
    279     const float c2c2 = 1.f - eta*eta*(1.f - c1 * c1 ) ;   // Snells law 
    280     
    281     bool tir = c2c2 < 0.f ;
    282     const float EdotN = dot(p.polarization , s.surface_normal ) ;  // used for TIR polarization
    283 
    284     const float c2 = tir ? 0.f : sqrtf(c2c2) ;   // c2 chosen +ve, set to 0.f for TIR => reflection_coefficient = 1.0f : so will always reflect
    285 
    286     const float n1c1 = n1*c1 ;
    287     const float n2c2 = n2*c2 ;
    288     const float n2c1 = n2*c1 ;
    289     const float n1c2 = n1*c2 ;
    290 
    291     const float3 A_trans = fabs(c1) > 0.999999f ? p.polarization : normalize(cross(p.direction, s.surface_normal)) ;
    292    
    293     // decompose p.polarization onto incident orthogonal basis
    294 
    295     const float E1_perp = dot(p.polarization, A_trans);   // fraction of E vector perpendicular to plane of incidence, ie S polarization
    296     const float3 E1pp = E1_perp * A_trans ;               // S-pol transverse component   
    297     const float3 E1pl = p.polarization - E1pp ;           // P-pol parallel component 
    298     const float E1_parl = length(E1pl) ;
    299 
    300     // G4OpBoundaryProcess at normal incidence, mentions Jackson and uses 
    301     //      A_trans  = OldPolarization; E1_perp = 0. E1_parl = 1. 
    302     // but that seems inconsistent with the above dot product, above is swapped cf that
    303 
    304     const float E2_perp_t = 2.f*n1c1*E1_perp/(n1c1+n2c2);  // Fresnel S-pol transmittance
    305     const float E2_parl_t = 2.f*n1c1*E1_parl/(n2c1+n1c2);  // Fresnel P-pol transmittance
    306 
    307     const float E2_perp_r = E2_perp_t - E1_perp;           // Fresnel S-pol reflectance
    308     const float E2_parl_r = (n2*E2_parl_t/n1) - E1_parl ;  // Fresnel P-pol reflectance
    309 
    310     const float2 E2_t = make_float2( E2_perp_t, E2_parl_t ) ;
    311     const float2 E2_r = make_float2( E2_perp_r, E2_parl_r ) ;
    312 
    313     const float  E2_total_t = dot(E2_t,E2_t) ;
    314 
    315     const float2 T = normalize(E2_t) ;
    316     const float2 R = normalize(E2_r) ;
    317 
    318     const float TransCoeff =  tir ? 0.0f : n2c2*E2_total_t/n1c1 ;
    319     //  above 0.0f was until 2016/3/4 incorrectly a 1.0f 
    320     //  resulting in TIR yielding BT where BR is expected
    321 
    322     const float u_reflect = s.ureflectcheat >= 0.f ? s.ureflectcheat : curand_uniform(&rng) ;
    323     bool reflect = u_reflect > TransCoeff  ;
    324 
    325 #ifdef WITH_ALIGN_DEV_DEBUG
    326     rtPrintf("propagate_at_boundary  u_OpBoundary_DiDiTransCoeff:%.9g  reflect:%d   TransCoeff:%10.5f  c2c2:%10.4f tir:%d  pos (%10.4f %10.4f %10.4f)   \n",
    327          u_reflect, reflect, TransCoeff, c2c2, tir, p.position.x, p.position.y, p.position.z  );
    328 #endif




om-cls DsG4OpBoundaryProcess

g4-cls G4OpBoundaryProcess::

    1140               if (sint1 > 0.0) {
    1141                  A_trans = OldMomentum.cross(theFacetNormal);
    1142                  A_trans = A_trans.unit();
    1143                  E1_perp = OldPolarization * A_trans;
    1144                  E1pp    = E1_perp * A_trans;
    1145                  E1pl    = OldPolarization - E1pp;
    1146                  E1_parl = E1pl.mag();
    1147               }
    1148               else {
    1149                  A_trans  = OldPolarization;
    1150                  // Here we Follow Jackson's conventions and we set the
    1151                  // parallel component = 1 in case of a ray perpendicular
    1152                  // to the surface
    1153                  E1_perp  = 0.0;
    1154                  E1_parl  = 1.0;
    1155               }
    1156 



* see g4op-vi for my annotation of G4OpBoundaryProcess


* http://www.phys.unm.edu/msbahae/Optics%20Lab/Polarization.pdf



LV 18 : polarization wrong ? for "TO BT BR BR BR BT SA"  0x8cbbbcd
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:: 

    ts 18
    ta 18 
    tv 18


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




    nph:   10000 A:    0.0039 B:    2.6367 B/A:     675.0 COMPUTE_MODE compute_requested  ALIGN non-reflectcheat 
    ab.a.metadata:/tmp/tboolean-proxy-18/evt/tboolean-proxy-18/torch/1         ox:90156ab21fdc9e565a275dcaeb26cbd6 rx:ed8bfb373a8eb1280e204118c286efe6 np:  10000 pr:    0.0039 COMPUTE_MODE compute_requested 
    ab.b.metadata:/tmp/tboolean-proxy-18/evt/tboolean-proxy-18/torch/-1        ox:95a60469de257b1edcdd42ff8eeaecf0 rx:a1928894ddfabcaf9e83989c773f7608 np:  10000 pr:    2.6367 COMPUTE_MODE compute_requested 
    WITH_SEED_BUFFER WITH_RECORD WITH_SOURCE WITH_ALIGN_DEV WITH_ALIGN_DEV_DEBUG WITH_LOGDOUBLE 
    []
    .
    [2019-06-23 22:11:26,614] p39013 {tboolean.py:71} CRITICAL -  RC 0x06 0b110 

    In [1]: ab.aselhis = "TO BT BR BR BR BT SA"

    In [2]: ab.rpol_dv_max()
    Out[2]: 
    A()sliced
    A([0., 0., 0., 0., 1., 0., 0.], dtype=float32)

    In [3]: ab.rpol_dv_where_(0.5)
    Out[3]: (array([4]),)


    In [4]: a.rpol()[4]
    Out[4]: 
    A()sliced
    A([[ 0.    , -1.    ,  0.    ],      TO
       [ 0.    , -1.    ,  0.    ],      BT 
       [ 0.    , -1.    , -0.0157],      BR
       [ 0.    , -1.    ,  0.0157],      BR 
       [ 0.    , -1.    ,  0.    ],      BR
       [ 0.    , -1.    ,  0.    ],      BT
       [ 0.    , -1.    ,  0.    ]],     SA        dtype=float32)

    In [5]: b.rpol()[4]
    Out[5]: 
    A()sliced
    A([[ 0.    , -1.    ,  0.    ],      TO 
       [ 0.    , -1.    ,  0.    ],      BT
       [ 0.    , -1.    , -0.0157],      BR
       [ 0.    , -1.    ,  0.0157],      BR
       [ 0.    , -1.    ,  0.    ],      BR  
       [ 1.    , -0.0236,  0.    ],      BT
       [ 1.    , -0.0236,  0.    ]],     SA      dtype=float32)



    In [3]: ab.rpol_dv_where(0.5)
    Out[3]: array([6368])



Almost perfect M shape BT-BR-BR-BR-BT at pole of the cap::

   ts 18 --mask 6368 --pindex 0 --pindexlog 

That means again there are lots of very close but not quite normal 
incidences.





