intersect_leaf_normals_from_Ellipsoid_not_normalized  CONFIRMED FIX
========================================================================


Context
----------

* g4cx/tests/G4CXTest.sh with single PMT comparisons of photon histories
  between GPU and CPU simulations revealed significant discrepancies 

  * for details see ~/j/issues/G4CXTest_comparison.rst 

* this prompted detailed PIDX (single photon) debug revealing 
  non-normalized normals for ellipsoid intersects

::

   APID=552 BPID=552 MODE=2 ~/opticks/g4cx/tests/G4CXTest.sh tra
   PIDX=552          MODE=2 ~/opticks/g4cx/tests/G4CXTest.sh tra    


THE FIX
----------

CSGOptiX/CSGOptiX7.cu::

    265 #endif
    266     while( bounce < evt->max_bounce )
    267     {
    268         trace( params.handle, ctx.p.pos, ctx.p.mom, params.tmin, params.tmax, prd);  // geo query filling prd      
    269         if( prd->boundary() == 0xffffu ) break ; // SHOULD ONLY HAPPEN FOR PHOTONS STARTING OUTSIDE WORLD
    270 
    271         // HMM: do this here or within CSG ?
    272         float3* normal = prd->normal();
    273         *normal = normalize(*normal);
    274 
    275         // propagate can do nothing meaningful without a boundary 
    276 #ifndef PRODUCTION
    277         ctx.trace(bounce);
    278 #endif
    279         command = sim->propagate(bounce, rng, ctx);



Standalone bi-simulation workflow
------------------------------------

::

    c4 ; ./rsync_put.sh                 # laptop
    o  ; ./bin/rsync_put.sh             # laptop

    c4 ; ./build_into_junosw.sh        # workstation
    o ; oo                             # workstation  

    gxt ; ./G4CXTest.sh                # workstation
    gxt ; ./G4CXTest.sh grab           # laptop

    PICK=AB MODE=2 ./G4CXTest.sh ana   # laptop
    PIDX=552 MODE=2 ./G4CXTest.sh tra   # laptop

    GEOM get # laptop


Overview
-----------


DONE : PIDX bi-Debug Investigations : reveal smoking gun : non-normalized normals off PMT Ellipsoid
------------------------------------------------------------------------------------------------------

Dumped values correspond to the arrays as expected::

    #C4OpBoundaryProcess::PostStepDoIt.Y.DiDi.0 PIDX 552
    mom0 = np.array([-0.02711790,0.00000000,-0.99963224])
    pol0 = np.array([0.00000000,-1.00000000,0.00000000])
    #C4OpBoundaryProcess::DielectricDielectric Rindex1 1.48426314 Rindex2 1.00000100
    #C4OpBoundaryProcess::PostStepDoIt.Y.DiDi.1 PIDX 552
    mom1 = np.array([0.11403715,0.00000000,-0.99347649])
    pol1 = np.array([0.00000000,-1.00000000,-0.00000000])
    nrm = np.array([0.30020198,0.00000000,0.95387566])


::

    In [3]: np.set_printoptions(precision=8)

    In [4]: b.r
    Out[4]: 
    array([[[ 100.        ,    0.        ,  195.        ,    0.        ],
            [   0.        ,    0.        ,   -1.        ,    0.        ],
            [  -0.        ,   -1.        ,   -0.        ,  420.        ],
            [   0.        ,    0.        ,    0.        ,    0.        ]],

           [[ 100.        ,    0.        ,  169.14095   ,    0.11859877],
            [  -0.0271179 ,    0.        ,   -0.99963224,           nan],
            [   0.        ,   -1.        ,    0.        ,  420.        ],
            [   0.        ,    0.        ,    0.        ,    0.        ]],

           [[  99.85985   ,    0.        ,  163.97456   ,    0.14493875],
            [   0.11403715,    0.        ,   -0.9934765 ,    0.        ],
            [   0.        ,   -1.        ,   -0.        ,  420.        ],
            [   0.        ,    0.        ,    0.        ,    0.        ]],

           [[ 135.89851   ,    0.        , -149.98953   ,    1.199088  ],
            [   0.11403715,    0.        ,   -0.9934765 ,    0.        ],
            [   0.        ,   -1.        ,    0.        ,  420.        ],
            [   0.        ,    0.        ,    0.        ,    0.        ]]], dtype=float32)




    In [5]: a.r
    Out[5]: 
    array([[[ 100.        ,    0.        ,  195.        ,    0.        ],
            [   0.        ,    0.        ,   -1.        ,    0.        ],
            [  -0.        ,   -1.        ,   -0.        ,  420.        ],
            [   0.        ,    0.        ,    0.        ,    0.        ]],

           [[ 100.        ,    0.        ,  169.14096   ,    0.11859874],
            [  -0.02701617,    0.        ,   -0.9993043 ,    0.        ],
            [   0.        ,   -1.        ,    0.        ,  420.        ],
            [   0.        ,    0.        ,   -0.        ,    0.        ]],

           [[  99.86032   ,    0.        ,  163.97441   ,    0.14494817],
            [   0.11815427,    0.        ,   -0.98039055,    0.        ],
            [   0.        ,   -1.        ,   -0.        ,  420.        ],
            [   0.        ,    0.        ,   -0.        ,    0.        ]],

           [[ 137.60153   ,    0.        , -149.185     ,    1.2104301 ],
            [   0.11815427,    0.        ,   -0.98039055,    0.        ],
            [   0.        ,   -1.        ,   -0.        ,  420.        ],
            [   0.        ,    0.        ,    0.        ,    0.        ]]], dtype=float32)

    //qsim.propagate_at_boundary idx 552 TransCoeff     0.9207 n1c1     1.3703 n2c2     0.8214 E2_t (    1.2505,    0.0000) A_trans (    0.0000,   -1.0000,   -0.0000) 
    //qsim.propagate_at_boundary idx 552 u_reflect     0.1287 TransCoeff     0.9207 reflect 0 
    //qsim.propagate_at_boundary idx 552 : mom0 = np.array([-0.02701617,0.00000000,-0.99930429])  
    //qsim.propagate_at_boundary idx 552 : nrm = np.array([0.28830180,0.00000000,0.91605818])  
    //qsim.propagate_at_boundary idx 552 : eta = 1.48426175 ; eta_c1 = 1.37028480 ; c2 = 0.82136923 ; eta_c1__c2 = 0.54891557 
    //qsim.propagate_at_boundary idx 552 reflect 0 tir 0 TransCoeff     0.9207 u_reflect     0.1287 
    //qsim.propagate_at_boundary idx 552 : mom1 = np.array([0.11815427,0.00000000,-0.98039055]) 
    //qsim.propagate_at_boundary idx 552 : pol1 = np.array([0.00000000,-1.00000000,-0.00000000]) 



::

     857     p.mom = reflect
     858                     ?
     859                        p.mom + 2.0f*c1*oriented_normal
     860                     :
     861                        eta*(p.mom) + (eta*c1 - c2)*oriented_normal
     862                     ;
     863 


::


    In [6]: eta = 1.48426175 ; eta_c1 = 1.37028480 ; c2 = 0.82136923 ; eta_c1__c2 = 0.54891557

    In [7]: mom0 = np.array([-0.02701617,0.00000000,-0.99930429])

    In [8]: nrm = np.array([0.28830180,0.00000000,0.91605818])

    In [9]: eta*mom0 + eta_c1__c2*nrm
    Out[9]: array([ 0.11815428,  0.        , -0.98039054])

    In [10]: mom1 = np.array([0.11815427,0.00000000,-0.98039055])

    In [11]: check_mom1 = eta*mom0 + eta_c1__c2*nrm

    In [12]: check_mom1
    Out[12]: array([ 0.11815428,  0.        , -0.98039054])

    In [13]: mom1
    Out[13]: array([ 0.11815427,  0.        , -0.98039055])



::

    #C4OpBoundaryProcess::PostStepDoIt.Y.DiDi.0 PIDX 552
    mom0 = np.array([-0.02711790,0.00000000,-0.99963224])
    pol0 = np.array([0.00000000,-1.00000000,0.00000000])
    #C4OpBoundaryProcess::DielectricDielectric Rindex1 1.48426314 Rindex2 1.00000100
    #C4OpBoundaryProcess::DielectricDielectric.do polished YES
    theGlobalNormal = np.array([0.30020198,0.00000000,0.95387566])
    theFacetNormal = np.array([0.30020198,0.00000000,0.95387566])
    OldMomentum = np.array([-0.02711790,0.00000000,-0.99963224])
    OldPolarization = np.array([0.00000000,-1.00000000,0.00000000])
    #C4OpBoundaryProcess::DielectricDielectric.incident_ray_oblique.FresnelRefraction
    NewMomentum_0 = np.array([0.11403715,0.00000000,-0.99347649])
    #C4OpBoundaryProcess::DielectricDielectric.incident_ray_oblique ; sint1 = 0.27422447 ; cost1 = 0.96166571 ; cost2 = 0.91341886 ; Rindex2 = 1.00000100 ; Rindex1 = 1.48426314 ; alpha = 0.34626286
    #C4OpBoundaryProcess::PostStepDoIt.Y.DiDi.1 PIDX 552
    mom1 = np.array([0.11403715,0.00000000,-0.99347649])
    pol1 = np.array([0.00000000,-1.00000000,-0.00000000])
    nrm = np.array([0.30020198,0.00000000,0.95387566])


Looking like the normal is primary source of deviation::

    In [15]: theGlobalNormal
    Out[15]: array([0.30020198, 0.        , 0.95387566])

    In [16]: theGlobalNormal - nrm
    Out[16]: array([0.01190018, 0.        , 0.03781748])


1st dump positions with normals for certainty. 


::

    2023-08-08 20:38:11.978 INFO  [30026] [SEvt::hostside_running_resize_@1785] resizing photon 0 to evt.num_photon 10000
    #C4OpBoundaryProcess::DielectricDielectric Rindex1 1.35398554 Rindex2 1.48426314
    #C4OpBoundaryProcess::DielectricDielectric.do polished YES
    theGlobalPoint = np.array([100.00000000,0.00000000,169.14095242])   ## outside of Pyrex
    theGlobalNormal = np.array([0.29632217,0.00000000,0.95508804])
    theFacetNormal = np.array([0.29632217,0.00000000,0.95508804])
    theRecoveredNormal = np.array([0.29632217,0.00000000,0.95508804])
    OldMomentum = np.array([0.00000000,0.00000000,-1.00000000])
    OldPolarization = np.array([0.00000000,-1.00000000,0.00000000])
    #C4OpBoundaryProcess::DielectricDielectric.incident_ray_oblique.FresnelRefraction
    NewMomentum_0 = np.array([-0.02711790,0.00000000,-0.99963224])
    #C4OpBoundaryProcess::DielectricDielectric.incident_ray_oblique ; sint1 = 0.29632217 ; cost1 = 0.95508804 ; cost2 = 0.96277244 ; Rindex2 = 1.48426314 ; Rindex1 = 1.35398554 ; alpha = -0.10032030
    U4Recorder::UserSteppingAction_Optical PIDX 552 post U4StepPoint::DescPositionTime (    100.000      0.000    169.141      0.119) is_fastsim_flag 0 FAKES_SKIP 0 is_fake 0 fakemask 0

    C4OpBoundaryProcess::PostStepDoIt PIDX 552 m_custom_status Y
    #C4OpBoundaryProcess::PostStepDoIt.Y.DiDi.0 PIDX 552
    theGlobalPoint = np.array([99.85984668,0.00000000,163.97455814])
    mom0 = np.array([-0.02711790,0.00000000,-0.99963224])
    pol0 = np.array([0.00000000,-1.00000000,0.00000000])
    #C4OpBoundaryProcess::DielectricDielectric Rindex1 1.48426314 Rindex2 1.00000100
    #C4OpBoundaryProcess::DielectricDielectric.do polished YES
    theGlobalPoint = np.array([99.85984668,0.00000000,163.97455814])
    theGlobalNormal = np.array([0.30020198,0.00000000,0.95387566])
    theFacetNormal = np.array([0.30020198,0.00000000,0.95387566])
    theRecoveredNormal = np.array([0.30020198,0.00000000,0.95387566])
    OldMomentum = np.array([-0.02711790,0.00000000,-0.99963224])
    OldPolarization = np.array([0.00000000,-1.00000000,0.00000000])
    #C4OpBoundaryProcess::DielectricDielectric.incident_ray_oblique.FresnelRefraction
    NewMomentum_0 = np.array([0.11403715,0.00000000,-0.99347649])
    #C4OpBoundaryProcess::DielectricDielectric.incident_ray_oblique ; sint1 = 0.27422447 ; cost1 = 0.96166571 ; cost2 = 0.91341886 ; Rindex2 = 1.00000100 ; Rindex1 = 1.48426314 ; alpha = 0.34626286
    #C4OpBoundaryProcess::PostStepDoIt.Y.DiDi.1 PIDX 552
    mom1 = np.array([0.11403715,0.00000000,-0.99347649])
    pol1 = np.array([0.00000000,-1.00000000,-0.00000000])
    nrm = np.array([0.30020198,0.00000000,0.95387566])
    U4Recorder::UserSteppingAction_Optical PIDX 552 post U4StepPoint::DescPositionTime (     99.860      0.000    163.975      0.145) is_fastsim_flag 0 FAKES_SKIP 0 is_fake 0 fakemask 0
    C4OpBoundaryProcess::PostStepDoIt PIDX 552 m_custom_status Z
    U4Recorder::UserSteppingAction_Optical PIDX 552 post U4StepPoint::DescPositionTime (    135.899      0.000   -149.990      1.199) is_fastsim_flag 0 FAKES_SKIP 0 is_fake 0 fakemask 0
    U4Recorder::PostUserTrackingAction_Optical.fStopAndKill  ulabel.id    552 seq.brief TO BT BT SA
    2023-08-08 20:38:12.555 INFO  [30026] [SEvt::save@3243]  dir /tmp/blyth/opticks/GEOM/FewPMT/G4CXTest/ALL0/n001 index 1 instance 1 OPTICKS_SAVE_COMP  genstep,photon,record,seq,prd,hit,domain,inphoton,tag,flat,aux,sup
    2023-08-08 20:38:12.605 INFO  [30026] [SEvt::clear_except@1413] SEvt::clear_except
    2023-08-08 20:38:12.606 ERROR [30026] [G4CXApp::SaveMeta@256]  NULL savedir 
    2023-08-08 20:38:12.606 INFO  [30026] [G4CXApp::EndOfEventAction@231] not-(WITH_PMTSIM and POM_DEBUG)
    2023-08-08 20:38:12.606 INFO  [30026] [SEvt::clear@1392] SEvt::clear
    //qsim.propagate.head idx 552 : bnc 0 cosTheta -0.91923934 
    //qsim.propagate.head idx 552 : mom = np.array([0.00000000,0.00000000,-1.00000000]) 
    //qsim.propagate.head idx 552 : pos = np.array([ 100.00000,   0.00000, 195.00000]) 
    //qsim.propagate.head idx 552 : nrm = np.array([(0.28519988,0.00000000,0.91923934]) 
    //qsim.propagate_to_boundary.head idx 552 : u_absorption 0.33028582 logf(u_absorption) -1.10779667 absorption_length 37213.9219 absorption_distance 41225.457031 
    //qsim.propagate_to_boundary.head idx 552 : post = np.array([ 100.00000,   0.00000, 195.00000,   0.00000]) 
    //qsim.propagate_to_boundary.head idx 552 : distance_to_boundary    25.8590 absorption_distance 41225.4570 scattering_distance 96441.0859 u_scattering     0.5812 u_absorption     0.3303 
    //qsim.propagate idx 552 bounce 0 command 3 flag 0 s.optical.x 0 s.optical.y 1 
    //qsim.propagate.WITH_CUSTOM4 idx 552  BOUNDARY ems 1 lposcost   0.861 
    //qsim.propagate_at_boundary.head idx 552 : theTransmittance = -1.00000000 
    //qsim.propagate_at_boundary.head idx 552 : nrm = np.array([0.28519988,0.00000000,0.91923934]) 
                              ## cf theGlobalNormal = np.array([0.29632217,0.00000000,0.95508804])


    HUH: that nrm is not normalized   : SMOKING GUN 

    In [1]: nrm = np.array([0.28519988,0.00000000,0.91923934])

    In [2]: theGlobalNormal = np.array([0.29632217,0.00000000,0.95508804])


    In [8]: np.set_printoptions(precision=10)

    In [9]: n_nrm = nrm/np.sqrt(np.sum(nrm*nrm)) ; n_nrm       ## NORMALIZING GETS VERY CLOSE TO theGlobalNormal
    Out[9]: array([0.2963221695, 0.          , 0.955088044 ])

    In [5]: np.sum(nrm*nrm)
    Out[5]: 0.92633993575565

    In [6]: np.sum(theGlobalNormal*theGlobalNormal)
    Out[6]: 0.9999999925845506




    //qsim.propagate_at_boundary.head idx 552 : pos = np.array([ 100.00000,   0.00000, 169.14096])    
                               ## cf theGlobalPoint = np.array([100.00000000,0.00000000,169.14095242])   ## outside of Pyrex

    //qsim.propagate_at_boundary.head idx 552 : mom0 = np.array([0.00000000,0.00000000,-1.00000000]) 
    //qsim.propagate_at_boundary.head idx 552 : pol0 = np.array([-0.00000000,-1.00000000,-0.00000000]) 
    //qsim.propagate_at_boundary.head idx 552 : n1,n2,eta = (1.35398555,1.48426318,0.91222739) 
    //qsim.propagate_at_boundary.head idx 552 : c1 = 0.91923934 ; normal_incidence = 0 
    //qsim.propagate_at_boundary.body idx 552 : TransCoeff = 0.99714178 ; n1c1 = 1.24463677 ; n2c2 = 1.38523686 
    //qsim.propagate_at_boundary.body idx 552 : E2_t = np.array([0.94653732,0.00000000]) 
    //qsim.propagate_at_boundary.body idx 552 : A_trans = np.array([0.00000000,-1.00000000,0.00000000]) 
    //qsim.propagate_at_boundary.body idx 552 : u_reflect     0.1106 TransCoeff     0.9971 reflect 0 
    //qsim.propagate_at_boundary.body idx 552 : mom0 = np.array([0.00000000,0.00000000,-1.00000000])  
    //qsim.propagate_at_boundary.body idx 552 : pos = np.array([ 100.00000,   0.00000, 169.14096]) 
    //qsim.propagate_at_boundary.body idx 552 : nrm = np.array([0.28519988,0.00000000,0.91923934])  
    //qsim.propagate_at_boundary.body idx 552 : n1 = 1.35398555 ; n2 = 1.48426318 ; eta = 0.91222739  
    //qsim.propagate_at_boundary.body idx 552 : c1 = 0.91923934 ; eta_c1 = 0.83855534 ; c2 = 0.93328249 ; eta_c1__c2 = -0.09472716 
    //qsim.propagate_at_boundary.tail idx 552 : reflect 0 tir 0 TransCoeff     0.9971 u_reflect     0.1106 
    //qsim.propagate_at_boundary.tail idx 552 : mom1 = np.array([-0.02701617,0.00000000,-0.99930429]) 
    //qsim.propagate_at_boundary.tail idx 552 : pol1 = np.array([0.00000000,-1.00000000,0.00000000]) 
    //qsim.propagate.head idx 552 : bnc 1 cosTheta -0.92320967 
    //qsim.propagate.head idx 552 : mom = np.array([-0.02701617,0.00000000,-0.99930429]) 
    //qsim.propagate.head idx 552 : pos = np.array([ 100.00000,   0.00000, 169.14096]) 
    //qsim.propagate.head idx 552 : nrm = np.array([(0.28830180,0.00000000,0.91605818]) 
    //qsim.propagate_to_boundary.head idx 552 : u_absorption 0.56169021 logf(u_absorption) -0.57680476 absorption_length  1562.9586 absorption_distance 901.521973 
    //qsim.propagate_to_boundary.head idx 552 : post = np.array([ 100.00000,   0.00000, 169.14096,   0.11860]) 
    //qsim.propagate_to_boundary.head idx 552 : distance_to_boundary     5.1701 absorption_distance   901.5220 scattering_distance 3043071.0000 u_scattering     0.0477 u_absorption     0.5617 
    //qsim.propagate idx 552 bounce 1 command 3 flag 0 s.optical.x 7 s.optical.y 4 
    //qsim.propagate.WITH_CUSTOM4 idx 552  BOUNDARY ems 4 lposcost   0.854 
    //qsim::propagate_at_surface_CustomART idx     552 : mom = np.array([-0.02701617,0.00000000,-0.99930429]) 
    //qsim::propagate_at_surface_CustomART idx     552 : pol = np.array([0.00000000,-1.00000000,0.00000000]) 
    //qsim::propagate_at_surface_CustomART idx     552 : nrm = np.array([0.28830180,0.00000000,0.91605818]) 
    //qsim::propagate_at_surface_CustomART idx     552 : cross_mom_nrm = np.array([0.00000000,-0.26335284,-0.00000000]) 
    //qsim::propagate_at_surface_CustomART idx     552 : dot_pol_cross_mom_nrm = 0.26335284 
    //qsim::propagate_at_surface_CustomART idx     552 : minus_cos_theta = -0.92320967 
    //qsim::propagate_at_surface_CustomART idx 552 lpmtid 0 wl 420.000 mct  -0.923 dpcmn   0.263 ARTE (   0.650   0.079   0.921   0.537 ) 
    //qsim.propagate_at_surface_CustomART idx 552 lpmtid 0 ARTE (   0.650   0.079   0.921   0.537 ) u_theAbsorption    0.663 action 2 
    //qsim.propagate_at_boundary.head idx 552 : theTransmittance = 0.92073381 
    //qsim.propagate_at_boundary.head idx 552 : nrm = np.array([0.28830180,0.00000000,0.91605818]) 
    //qsim.propagate_at_boundary.head idx 552 : pos = np.array([  99.86032,   0.00000, 163.97441]) 
    //qsim.propagate_at_boundary.head idx 552 : mom0 = np.array([-0.02701617,0.00000000,-0.99930429]) 
    //qsim.propagate_at_boundary.head idx 552 : pol0 = np.array([0.00000000,-1.00000000,0.00000000]) 
    //qsim.propagate_at_boundary.head idx 552 : n1,n2,eta = (1.48426318,1.00000095,1.48426175) 
    //qsim.propagate_at_boundary.head idx 552 : c1 = 0.92320967 ; normal_incidence = 0 
    //qsim.propagate_at_boundary.body idx 552 : TransCoeff = 0.92073381 ; n1c1 = 1.37028611 ; n2c2 = 0.82137001 
    //qsim.propagate_at_boundary.body idx 552 : E2_t = np.array([1.25045729,0.00000000]) 
    //qsim.propagate_at_boundary.body idx 552 : A_trans = np.array([0.00000000,-1.00000000,-0.00000000]) 
    //qsim.propagate_at_boundary.body idx 552 : u_reflect     0.1287 TransCoeff     0.9207 reflect 0 
    //qsim.propagate_at_boundary.body idx 552 : mom0 = np.array([-0.02701617,0.00000000,-0.99930429])  
    //qsim.propagate_at_boundary.body idx 552 : pos = np.array([  99.86032,   0.00000, 163.97441]) 
    //qsim.propagate_at_boundary.body idx 552 : nrm = np.array([0.28830180,0.00000000,0.91605818])  
    //qsim.propagate_at_boundary.body idx 552 : n1 = 1.48426318 ; n2 = 1.00000095 ; eta = 1.48426175  
    //qsim.propagate_at_boundary.body idx 552 : c1 = 0.92320967 ; eta_c1 = 1.37028480 ; c2 = 0.82136923 ; eta_c1__c2 = 0.54891557 
    //qsim.propagate_at_boundary.tail idx 552 : reflect 0 tir 0 TransCoeff     0.9207 u_reflect     0.1287 
    //qsim.propagate_at_boundary.tail idx 552 : mom1 = np.array([0.11815427,0.00000000,-0.98039055]) 
    //qsim.propagate_at_boundary.tail idx 552 : pol1 = np.array([0.00000000,-1.00000000,-0.00000000]) 
    //qsim.propagate.head idx 552 : bnc 2 cosTheta 0.86403084 
    //qsim.propagate.head idx 552 : mom = np.array([0.11815427,0.00000000,-0.98039055]) 
    //qsim.propagate.head idx 552 : pos = np.array([  99.86032,   0.00000, 163.97441]) 
    //qsim.propagate.head idx 552 : nrm = np.array([(0.39726260,0.00000000,-0.83343577]) 
    //qsim.propagate_to_boundary.head idx 552 : u_absorption 0.47715482 logf(u_absorption) -0.73991418 absorption_length 1000000000.0000 absorption_distance 739914176.000000 
    //qsim.propagate_to_boundary.head idx 552 : post = np.array([  99.86032,   0.00000, 163.97441,   0.14495]) 
    //qsim.propagate_to_boundary.head idx 552 : distance_to_boundary   319.4231 absorption_distance 739914176.0000 scattering_distance 805802.8750 u_scattering     0.4467 u_absorption     0.4772 
    //qsim.propagate idx 552 bounce 2 command 3 flag 0 s.optical.x 7 s.optical.y 4 
    //qsim.propagate.WITH_CUSTOM4 idx 552  BOUNDARY ems 4 lposcost  -0.735 
    //qsim.propagate (lposcost < 0.f) idx 552 bounce 2 command 3 flag 0 ems 4 
    2023-08-08 20:38:12.725 INFO  [30026] [SEvt::save@3243]  dir /tmp/blyth/opticks/GEOM/FewPMT/G4CXTest/ALL0/p001 index 1 instance 0 OPTICKS_SAVE_COMP  genstep,photon,record,seq,prd,hit,domain,inphoton,tag,flat,aux,sup
    2023-08-08 20:38:12.755 INFO  [30026] [SEvt::clear_except@1413] SEvt::clear_except




Normalization of normals issue
---------------------------------

::

    2023-08-08 22:17:53.378 INFO  [50177] [G4CXApp::GeneratePrimaries@212] ]
    2023-08-08 22:17:53.383 INFO  [50177] [SEvt::hostside_running_resize_@1785] resizing photon 0 to evt.num_photon 10000
    #C4OpBoundaryProcess::DielectricDielectric Rindex1 1.35398554 Rindex2 1.48426314
    #C4OpBoundaryProcess::DielectricDielectric.do polished YES
    theGlobalPoint = np.array([100.00000000,0.00000000,169.14095242]) ; l_theGlobalPoint = 196.49086947
    theGlobalNormal = np.array([0.29632217,0.00000000,0.95508804]) ; l_theGlobalNormal = 1.00000000
    theFacetNormal = np.array([0.29632217,0.00000000,0.95508804]) ; l_theFacetNormal = 1.00000000
    theRecoveredNormal = np.array([0.29632217,0.00000000,0.95508804]) ; l_theRecoveredNormal = 1.00000000
    OldMomentum = np.array([0.00000000,0.00000000,-1.00000000]) ; l_OldMomentum = 1.00000000
    OldPolarization = np.array([0.00000000,-1.00000000,0.00000000]) ; l_OldPolarization = 1.00000000
    #C4OpBoundaryProcess::DielectricDielectric.incident_ray_oblique.FresnelRefraction
    NewMomentum0 = np.array([-0.02711790,0.00000000,-0.99963224]) ; l_NewMomentum0 = 1.00000000
    #C4OpBoundaryProcess::DielectricDielectric.incident_ray_oblique ; sint1 = 0.29632217 ; cost1 = 0.95508804 ; cost2 = 0.96277244 ; Rindex2 = 1.48426314 ; Rindex1 = 1.35398554 ; alpha = -0.10032030
    U4Recorder::UserSteppingAction_Optical PIDX 552 post U4StepPoint::DescPositionTime (    100.000      0.000    169.141      0.119) is_fastsim_flag 0 FAKES_SKIP 0 is_fake 0 fakemask 0

    C4OpBoundaryProcess::PostStepDoIt PIDX 552 m_custom_status Y
    #C4OpBoundaryProcess::PostStepDoIt.Y.DiDi.0 PIDX 552
    theGlobalPoint = np.array([99.85984668,0.00000000,163.97455814]) ; l_theGlobalPoint = 191.98865773
    mom0 = np.array([-0.02711790,0.00000000,-0.99963224]) ; l_OldMomentum = 1.00000000
    pol0 = np.array([0.00000000,-1.00000000,0.00000000]) ; l_OldPolarization = 1.00000000
    #C4OpBoundaryProcess::DielectricDielectric Rindex1 1.48426314 Rindex2 1.00000100
    #C4OpBoundaryProcess::DielectricDielectric.do polished YES
    theGlobalPoint = np.array([99.85984668,0.00000000,163.97455814]) ; l_theGlobalPoint = 191.98865773
    theGlobalNormal = np.array([0.30020198,0.00000000,0.95387566]) ; l_theGlobalNormal = 1.00000000
    theFacetNormal = np.array([0.30020198,0.00000000,0.95387566]) ; l_theFacetNormal = 1.00000000
    theRecoveredNormal = np.array([0.30020198,0.00000000,0.95387566]) ; l_theRecoveredNormal = 1.00000000
    OldMomentum = np.array([-0.02711790,0.00000000,-0.99963224]) ; l_OldMomentum = 1.00000000
    OldPolarization = np.array([0.00000000,-1.00000000,0.00000000]) ; l_OldPolarization = 1.00000000
    #C4OpBoundaryProcess::DielectricDielectric.incident_ray_oblique.FresnelRefraction
    NewMomentum0 = np.array([0.11403715,0.00000000,-0.99347649]) ; l_NewMomentum0 = 1.00000000
    #C4OpBoundaryProcess::DielectricDielectric.incident_ray_oblique ; sint1 = 0.27422447 ; cost1 = 0.96166571 ; cost2 = 0.91341886 ; Rindex2 = 1.00000100 ; Rindex1 = 1.48426314 ; alpha = 0.34626286
    #C4OpBoundaryProcess::PostStepDoIt.Y.DiDi.1 PIDX 552
    mom1 = np.array([0.11403715,0.00000000,-0.99347649]) ; l_NewMomentum = 1.00000000
    pol1 = np.array([0.00000000,-1.00000000,-0.00000000]) ; l_NewPolarization = 1.00000000
    nrm = np.array([0.30020198,0.00000000,0.95387566]) ; l_theRecoveredNormal = 1.00000000
    U4Recorder::UserSteppingAction_Optical PIDX 552 post U4StepPoint::DescPositionTime (     99.860      0.000    163.975      0.145) is_fastsim_flag 0 FAKES_SKIP 0 is_fake 0 fakemask 0
    C4OpBoundaryProcess::PostStepDoIt PIDX 552 m_custom_status Z
    U4Recorder::UserSteppingAction_Optical PIDX 552 post U4StepPoint::DescPositionTime (    135.899      0.000   -149.990      1.199) is_fastsim_flag 0 FAKES_SKIP 0 is_fake 0 fakemask 0
    U4Recorder::PostUserTrackingAction_Optical.fStopAndKill  ulabel.id    552 seq.brief TO BT BT SA
    2023-08-08 22:17:54.032 INFO  [50177] [SEvt::save@3243]  dir /tmp/blyth/opticks/GEOM/FewPMT/G4CXTest/ALL0/n001 index 1 instance 1 OPTICKS_SAVE_COMP  genstep,photon,record,seq,prd,hit,domain,inphoton,tag,flat,aux,sup
    2023-08-08 22:17:54.088 INFO  [50177] [SEvt::clear_except@1413] SEvt::clear_except
    2023-08-08 22:17:54.088 ERROR [50177] [G4CXApp::SaveMeta@256]  NULL savedir 
    2023-08-08 22:17:54.088 INFO  [50177] [G4CXApp::EndOfEventAction@231] not-(WITH_PMTSIM and POM_DEBUG)
    2023-08-08 22:17:54.088 INFO  [50177] [SEvt::clear@1392] SEvt::clear

    //qsim.propagate.head idx 552 : bnc 0 cosTheta -0.91923934 
    //qsim.propagate.head idx 552 : mom = np.array([0.00000000,0.00000000,-1.00000000]) ; lmom = 1.00000000  
    //qsim.propagate.head idx 552 : pos = np.array([ 100.00000,   0.00000, 195.00000]) ; lpos = 219.14607239 
    //qsim.propagate.head idx 552 : nrm = np.array([(0.28519988,0.00000000,0.91923934]) ; lnrm = 0.96246552  
    //qsim.propagate_to_boundary.head idx 552 : u_absorption 0.33028582 logf(u_absorption) -1.10779667 absorption_length 37213.9219 absorption_distance 41225.457031 
    //qsim.propagate_to_boundary.head idx 552 : post = np.array([ 100.00000,   0.00000, 195.00000,   0.00000]) 
    //qsim.propagate_to_boundary.head idx 552 : distance_to_boundary    25.8590 absorption_distance 41225.4570 scattering_distance 96441.0859 u_scattering     0.5812 u_absorption     0.3303 
    //qsim.propagate idx 552 bounce 0 command 3 flag 0 s.optical.x 0 s.optical.y 1 
    //qsim.propagate.WITH_CUSTOM4 idx 552  BOUNDARY ems 1 lposcost   0.861 
    //qsim.propagate_at_boundary.head idx 552 : theTransmittance = -1.00000000 
    //qsim.propagate_at_boundary.head idx 552 : nrm = np.array([0.28519988,0.00000000,0.91923934]) ; lnrm = 0.96246552  
    //qsim.propagate_at_boundary.head idx 552 : pos = np.array([ 100.00000,   0.00000, 169.14096]) ; lpos = 196.49087524 
    //qsim.propagate_at_boundary.head idx 552 : mom0 = np.array([0.00000000,0.00000000,-1.00000000]) ; lmom0 = 1.00000000 
    //qsim.propagate_at_boundary.head idx 552 : pol0 = np.array([-0.00000000,-1.00000000,-0.00000000]) ; lpol0 = 1.00000000 
    //qsim.propagate_at_boundary.head idx 552 : n1,n2,eta = (1.35398555,1.48426318,0.91222739) 
    //qsim.propagate_at_boundary.head idx 552 : c1 = 0.91923934 ; normal_incidence = 0 
    //qsim.propagate_at_boundary.body idx 552 : TransCoeff = 0.99714178 ; n1c1 = 1.24463677 ; n2c2 = 1.38523686 
    //qsim.propagate_at_boundary.body idx 552 : E2_t = np.array([0.94653732,0.00000000]) ; lE2_t = 0.94653732 
    //qsim.propagate_at_boundary.body idx 552 : A_trans = np.array([0.00000000,-1.00000000,0.00000000]) ; lA_trans = 1.00000000 
    //qsim.propagate_at_boundary.body idx 552 : u_reflect     0.1106 TransCoeff     0.9971 reflect 0 
    //qsim.propagate_at_boundary.body idx 552 : mom0 = np.array([0.00000000,0.00000000,-1.00000000]) ; lmom0 = 1.00000000 
    //qsim.propagate_at_boundary.body idx 552 : pos = np.array([ 100.00000,   0.00000, 169.14096]) ; lpos = 196.49087524 
    //qsim.propagate_at_boundary.body idx 552 : nrm = np.array([0.28519988,0.00000000,0.91923934]) ; lnrm = 0.96246552 
    //qsim.propagate_at_boundary.body idx 552 : n1 = 1.35398555 ; n2 = 1.48426318 ; eta = 0.91222739  
    //qsim.propagate_at_boundary.body idx 552 : c1 = 0.91923934 ; eta_c1 = 0.83855534 ; c2 = 0.93328249 ; eta_c1__c2 = -0.09472716 
    //qsim.propagate_at_boundary.tail idx 552 : reflect 0 tir 0 TransCoeff     0.9971 u_reflect     0.1106 
    //qsim.propagate_at_boundary.tail idx 552 : mom1 = np.array([-0.02701617,0.00000000,-0.99930429]) ; lmom1 = 0.99966937  
    //qsim.propagate_at_boundary.tail idx 552 : pol1 = np.array([0.00000000,-1.00000000,0.00000000]) ; lpol1 = 1.00000000 

    //qsim.propagate.head idx 552 : bnc 1 cosTheta -0.92320967 
    //qsim.propagate.head idx 552 : mom = np.array([-0.02701617,0.00000000,-0.99930429]) ; lmom = 0.99966937  
    //qsim.propagate.head idx 552 : pos = np.array([ 100.00000,   0.00000, 169.14096]) ; lpos = 196.49087524 
    //qsim.propagate.head idx 552 : nrm = np.array([(0.28830180,0.00000000,0.91605818]) ; lnrm = 0.96035433  
    //qsim.propagate_to_boundary.head idx 552 : u_absorption 0.56169021 logf(u_absorption) -0.57680476 absorption_length  1562.9586 absorption_distance 901.521973 
    //qsim.propagate_to_boundary.head idx 552 : post = np.array([ 100.00000,   0.00000, 169.14096,   0.11860]) 
    //qsim.propagate_to_boundary.head idx 552 : distance_to_boundary     5.1701 absorption_distance   901.5220 scattering_distance 3043071.0000 u_scattering     0.0477 u_absorption     0.5617 
    //qsim.propagate idx 552 bounce 1 command 3 flag 0 s.optical.x 7 s.optical.y 4 
    //qsim.propagate.WITH_CUSTOM4 idx 552  BOUNDARY ems 4 lposcost   0.854 
    //qsim::propagate_at_surface_CustomART idx     552 : mom = np.array([-0.02701617,0.00000000,-0.99930429]) ; lmom = 0.99966937 
    //qsim::propagate_at_surface_CustomART idx     552 : pol = np.array([0.00000000,-1.00000000,0.00000000]) ; lpol = 1.00000000 
    //qsim::propagate_at_surface_CustomART idx     552 : nrm = np.array([0.28830180,0.00000000,0.91605818]) ; lnrm = 0.96035433 
    //qsim::propagate_at_surface_CustomART idx     552 : cross_mom_nrm = np.array([0.00000000,-0.26335284,-0.00000000]) ; lcross_mom_nrm = 0.26335284  
    //qsim::propagate_at_surface_CustomART idx     552 : dot_pol_cross_mom_nrm = 0.26335284 
    //qsim::propagate_at_surface_CustomART idx     552 : minus_cos_theta = -0.92320967 
    //qsim::propagate_at_surface_CustomART idx 552 lpmtid 0 wl 420.000 mct  -0.923 dpcmn   0.263 ARTE (   0.650   0.079   0.921   0.537 ) 
    //qsim.propagate_at_surface_CustomART idx 552 lpmtid 0 ARTE (   0.650   0.079   0.921   0.537 ) u_theAbsorption    0.663 action 2 
    //qsim.propagate_at_boundary.head idx 552 : theTransmittance = 0.92073381 
    //qsim.propagate_at_boundary.head idx 552 : nrm = np.array([0.28830180,0.00000000,0.91605818]) ; lnrm = 0.96035433  
    //qsim.propagate_at_boundary.head idx 552 : pos = np.array([  99.86032,   0.00000, 163.97441]) ; lpos = 191.98878479 
    //qsim.propagate_at_boundary.head idx 552 : mom0 = np.array([-0.02701617,0.00000000,-0.99930429]) ; lmom0 = 0.99966937 
    //qsim.propagate_at_boundary.head idx 552 : pol0 = np.array([0.00000000,-1.00000000,0.00000000]) ; lpol0 = 1.00000000 
    //qsim.propagate_at_boundary.head idx 552 : n1,n2,eta = (1.48426318,1.00000095,1.48426175) 
    //qsim.propagate_at_boundary.head idx 552 : c1 = 0.92320967 ; normal_incidence = 0 
    //qsim.propagate_at_boundary.body idx 552 : TransCoeff = 0.92073381 ; n1c1 = 1.37028611 ; n2c2 = 0.82137001 
    //qsim.propagate_at_boundary.body idx 552 : E2_t = np.array([1.25045729,0.00000000]) ; lE2_t = 1.25045729 
    //qsim.propagate_at_boundary.body idx 552 : A_trans = np.array([0.00000000,-1.00000000,-0.00000000]) ; lA_trans = 1.00000000 
    //qsim.propagate_at_boundary.body idx 552 : u_reflect     0.1287 TransCoeff     0.9207 reflect 0 
    //qsim.propagate_at_boundary.body idx 552 : mom0 = np.array([-0.02701617,0.00000000,-0.99930429]) ; lmom0 = 0.99966937 
    //qsim.propagate_at_boundary.body idx 552 : pos = np.array([  99.86032,   0.00000, 163.97441]) ; lpos = 191.98878479 
    //qsim.propagate_at_boundary.body idx 552 : nrm = np.array([0.28830180,0.00000000,0.91605818]) ; lnrm = 0.96035433 
    //qsim.propagate_at_boundary.body idx 552 : n1 = 1.48426318 ; n2 = 1.00000095 ; eta = 1.48426175  
    //qsim.propagate_at_boundary.body idx 552 : c1 = 0.92320967 ; eta_c1 = 1.37028480 ; c2 = 0.82136923 ; eta_c1__c2 = 0.54891557 
    //qsim.propagate_at_boundary.tail idx 552 : reflect 0 tir 0 TransCoeff     0.9207 u_reflect     0.1287 
    //qsim.propagate_at_boundary.tail idx 552 : mom1 = np.array([0.11815427,0.00000000,-0.98039055]) ; lmom1 = 0.98748469  
    //qsim.propagate_at_boundary.tail idx 552 : pol1 = np.array([0.00000000,-1.00000000,-0.00000000]) ; lpol1 = 1.00000000 

    //qsim.propagate.head idx 552 : bnc 2 cosTheta 0.86403084 
    //qsim.propagate.head idx 552 : mom = np.array([0.11815427,0.00000000,-0.98039055]) ; lmom = 0.98748469  
    //qsim.propagate.head idx 552 : pos = np.array([  99.86032,   0.00000, 163.97441]) ; lpos = 191.98878479 
    //qsim.propagate.head idx 552 : nrm = np.array([(0.39726260,0.00000000,-0.83343577]) ; lnrm = 0.92327285  
    //qsim.propagate_to_boundary.head idx 552 : u_absorption 0.47715482 logf(u_absorption) -0.73991418 absorption_length 1000000000.0000 absorption_distance 739914176.000000 
    //qsim.propagate_to_boundary.head idx 552 : post = np.array([  99.86032,   0.00000, 163.97441,   0.14495]) 
    //qsim.propagate_to_boundary.head idx 552 : distance_to_boundary   319.4231 absorption_distance 739914176.0000 scattering_distance 805802.8750 u_scattering     0.4467 u_absorption     0.4772 
    //qsim.propagate idx 552 bounce 2 command 3 flag 0 s.optical.x 7 s.optical.y 4 
    //qsim.propagate.WITH_CUSTOM4 idx 552  BOUNDARY ems 4 lposcost  -0.735 
    //qsim.propagate (lposcost < 0.f) idx 552 bounce 2 command 3 flag 0 ems 4 
    2023-08-08 22:17:54.221 INFO  [50177] [SEvt::save@3243]  dir /tmp/blyth/opticks/GEOM/FewPMT/G4CXTest/ALL0/p001 index 1 instance 0 OPTICKS_SAVE_COMP  genstep,photon,record,seq,prd,hit,domain,inphoton,tag,flat,aux,sup
    2023-08-08 22:17:54.253 INFO  [50177] [SEvt::clear_except@1413] SEvt::clear_except
    2023-08-08 22:17:54.254 INFO  [50177] [G4CXApp::EndOfRunAction@182] 
    Python 3.7.7 (default, May  7 2020, 21:25:33) 



How to proceed  : Options 
----------------------------

1. revive CSG MOCK_CUDA intersect testing and examine intersect normals from ellipsoids and other shapes 
2. collect normals into aux in A and B and compare them 


CSG/tests/csg_intersect_leaf_test.sh
--------------------------------------

Checking ellipsoid intersects and normals::

    ~/opticks/CSG/tests/csg_intersect_leaf_test.sh 


CONFIRMED FIX
----------------

::

    N[blyth@localhost tests]$ PIDX=552 ./G4CXTest.sh 
    BASH_SOURCE                    : /data/blyth/junotop/opticks/u4/tests/FewPMT.sh 
    VERSION                        : 1 
    version_desc                   : N=1 natural geometry : CustomBoundary 
    POM                            : 1 
    pom_desc                       : POM:1 allow photons into PMT which has innards 
    GEOM                           : FewPMT 
    FewPMT_GEOMList                : nnvtLogicalPMT 
    LAYOUT                         : one_pmt 
    ./G4CXTest.sh : PIDX 552 is defined and APID BPID are both not defined so setting them to PIDX
    storch_FillGenstep_radius=0
    storch_FillGenstep_type=point
    storch_FillGenstep_pos=100,0,195
    storch_FillGenstep_mom=0,0,-1
             BASH_SOURCE : ./G4CXTest.sh 
                    SDIR : /data/blyth/junotop/opticks/g4cx/tests 
                  U4TDIR : /data/blyth/junotop/opticks/u4/tests 
                  BINDIR : /data/blyth/junotop/opticks/bin 
                    GEOM : FewPMT 
                     bin : G4CXTest 
                     ana : /data/blyth/junotop/opticks/g4cx/tests/G4CXTest.py 
                     tra : /data/blyth/junotop/opticks/g4cx/tests/G4CXSimtraceMinTest.py 
              geomscript : /data/blyth/junotop/opticks/u4/tests/FewPMT.sh 
                    BASE : /tmp/blyth/opticks/GEOM/FewPMT/G4CXTest 
                    FOLD :  
                   AFOLD : /tmp/blyth/opticks/GEOM/FewPMT/G4CXTest/ALL0/p001 
                   BFOLD : /tmp/blyth/opticks/GEOM/FewPMT/G4CXTest/ALL0/n001 
                   TFOLD : /tmp/blyth/opticks/GEOM/FewPMT/G4CXTest/0/p999 
    PMTSimParamData_BASE : /home/blyth/.opticks/GEOM/V1J009/CSGFoundry/SSim/extra/jpmt 
    2023-08-09 00:53:32.444 INFO  [68875] [G4CXApp::Create@281] U4Recorder::Switches
    WITH_CUSTOM4
    WITH_PMTSIM
    PMTSIM_STANDALONE
    NOT:PRODUCTION


    **************************************************************
     Geant4 version Name: geant4-10-04-patch-02 [MT]   (25-May-2018)
                           Copyright : Geant4 Collaboration
                          References : NIM A 506 (2003), 250-303
                                     : IEEE-TNS 53 (2006), 270-278
                                     : NIM A 835 (2016), 186-225
                                 WWW : http://geant4.org/
    **************************************************************

    2023-08-09 00:53:32.484 INFO  [68875] [SEvt::HighLevelCreate@939]  g4state_rerun_id -1 alldir ALL0 alldir0 ALL0 seldir SEL0 rerun 0
    SEvt::HighLevelCreate g4state_rerun_id -1 alldir ALL0 alldir0 ALL0 seldir SEL0 rerun 0
    2023-08-09 00:53:32.515 INFO  [68875] [G4CXApp::Construct@155] [
    U4VolumeMaker::PV name FewPMT
    U4VolumeMaker::PVG_ name FewPMT gdmlpath - sub - exists 0
    [ PMTSim::GetLV [nnvtLogicalPMT]
    PMTSim::init                   yielded chars :  cout  24774 cerr      0 : set VERBOSE to see them 
    PMTSim::getLV geom [nnvtLogicalPMT] mgr Y head [LogicalPMT]
    Option RealSurface is enabled in Central Detector.  Reduce the m_pmt_h from 570 to 357.225
     GetName() nnvt
    NNVT_MCPPMT_PMTSolid::NNVT_MCPPMT_PMTSolid
    G4Material::GetMaterial() WARNING: The material: PMT_Mirror does not exist in the table. Return NULL pointer.
    Warning: setting PMT mirror reflectivity to 0.9999 because no PMT_Mirror material properties defined
    [ ZSolid::ApplyZCutTree zcut    173.225 pmt_delta      0.001 body_delta     -4.999 inner_delta     -5.000 zcut+pmt_delta    173.226 zcut+body_delta    168.226 zcut+inner_delta    168.225
    ] ZSolid::ApplyZCutTree zcut 173.225
    Option RealSurface is enabed. Reduce the height of tube_hz from 60.000 to 21.112
    ] PMTSim::GetLV [nnvtLogicalPMT] lv Y
    U4VolumeMaker::Wrap [ name FewPMT GEOMWrap -
    [ items_lv.size 1
    U4VolumeMaker_WrapRockWater_Rock_HALFSIDE 210
    U4VolumeMaker_WrapRockWater_Water_HALFSIDE 200
    2023-08-09 00:53:32.578 INFO  [68875] [G4CXApp::Construct@162]  fPV Rock_lv_pv
    2023-08-09 00:53:32.578 INFO  [68875] [G4CXApp::Construct@164] ]
    [stree::postcreate
    stree::desc_sensor
     sensor_id.size 1
     sensor_count 1
     sensor_name.size 1
    sensor_name[
    nnvt_inner_phys
    ]
    [stree::desc_sensor_nd
     edge            0
     num_nd          8
     num_nd_sensor   1
     num_sid         1
    ...
    ]stree::desc_sensor_nd
    stree::desc_sensor_id sensor_id.size 1
    [
          0 sid        0
    ]]stree::postcreate
    2023-08-09 00:53:34.582 INFO  [68875] [GGeo::postDirectTranslation@648] NOT SAVING : TO ENABLE : export GGeo__postDirectTranslation_save=1 
    2023-08-09 00:53:34.616 INFO  [68875] [CSG_GGeo_Convert::init@95] CSG_GGeo_Convert::DescConsistent gg_all_sensor_index_num 1 st_all_sensor_id_num 1
    GGeoLib::descAllSensorIndex nmm 1
    ( 0 : 1) all[ 1]


    SPropMockup::Combination base $HOME/.opticks/GEOM/$GEOM relp GGeo/GScintillatorLib/LS_ori/RINDEX.npy spath::Resolve to path /home/blyth/.opticks/GEOM/FewPMT/GGeo/GScintillatorLib/LS_ori/RINDEX.npy
    SPropMockup::Combination path /home/blyth/.opticks/GEOM/FewPMT/GGeo/GScintillatorLib/LS_ori/RINDEX.npy exists NO 
    2023-08-09 00:53:35.431 ERROR [68875] [QSim::UploadComponents@151]  icdf null, snam::ICDF icdf.npy
    2023-08-09 00:53:41.360 INFO  [68875] [G4CXOpticks::saveGeometry@558] [ /home/blyth/.opticks/GEOM/FewPMT
    G4CXOpticks::saveGeometry [ /home/blyth/.opticks/GEOM/FewPMT
    2023-08-09 00:53:41.368 INFO  [68875] [U4GDML::write@186]  ekey U4GDML_GDXML_FIX_DISABLE U4GDML_GDXML_FIX_DISABLE 0 U4GDML_GDXML_FIX 1
    G4GDML: Writing '/home/blyth/.opticks/GEOM/FewPMT/origin_raw.gdml'...
    G4GDML: Writing definitions...
    G4GDML: Writing materials...
    G4GDML: Writing solids...
    G4GDML: Writing structure...
    G4GDML: Writing setup...
    G4GDML: Writing surfaces...
    G4GDML: Writing '/home/blyth/.opticks/GEOM/FewPMT/origin_raw.gdml' done !
    2023-08-09 00:53:41.385 INFO  [68875] [U4GDML::write@197]  Apply GDXML::Fix  rawpath /home/blyth/.opticks/GEOM/FewPMT/origin_raw.gdml dstpath /home/blyth/.opticks/GEOM/FewPMT/origin.gdml
    2023-08-09 00:53:41.417 ERROR [68875] [GGeo::save_to_dir@785]  default idpath : [/tmp/blyth/opticks/GGeo] is overridden : [/home/blyth/.opticks/GEOM/FewPMT/GGeo]
    2023-08-09 00:53:41.418 INFO  [68875] [GGeo::save@832]  idpath /home/blyth/.opticks/GEOM/FewPMT/GGeo
    Local_DsG4Scintillation::Local_DsG4Scintillation level 0 verboseLevel 0
    2023-08-09 00:53:41.524 INFO  [68875] [G4CXApp::G4CXApp@150] 
    U4Recorder::Desc
     U4Recorder_STATES                   : -1
     U4Recorder_RERUN                    : -1
     U4Recorder__PIDX_ENABLED            : YES
     U4Recorder__EndOfRunAction_Simtrace : NO 
     U4Recorder__REPLICA_NAME_SELECT     : PMT
     PIDX                                : 552
     EIDX                                : -1
     GIDX                                : -1
    U4Recorder__UserSteppingAction_Optical_ClearNumberOfInteractionLengthLeft:0
    U4Recorder::Switches
    WITH_CUSTOM4
    WITH_PMTSIM
    PMTSIM_STANDALONE
    NOT:PRODUCTION


    2023-08-09 00:53:42.025 INFO  [68875] [G4CXApp::BeginOfRunAction@177] 
    2023-08-09 00:53:42.025 INFO  [68875] [G4CXApp::GeneratePrimaries@198] [ fPrimaryMode T
    U4VPrimaryGenerator::GeneratePrimaries ph (10000, 4, 4, )
    2023-08-09 00:53:42.032 INFO  [68875] [G4CXApp::GeneratePrimaries@212] ]
    2023-08-09 00:53:42.035 INFO  [68875] [SEvt::hostside_running_resize_@1785] resizing photon 0 to evt.num_photon 10000
    #C4OpBoundaryProcess::DielectricDielectric Rindex1 1.35398554 Rindex2 1.48426314
    #C4OpBoundaryProcess::DielectricDielectric.do polished YES
    theGlobalPoint = np.array([100.00000000,0.00000000,169.14095242]) ; l_theGlobalPoint = 196.49086947
    theGlobalNormal = np.array([0.29632217,0.00000000,0.95508804]) ; l_theGlobalNormal = 1.00000000
    theFacetNormal = np.array([0.29632217,0.00000000,0.95508804]) ; l_theFacetNormal = 1.00000000
    theRecoveredNormal = np.array([0.29632217,0.00000000,0.95508804]) ; l_theRecoveredNormal = 1.00000000
    OldMomentum = np.array([0.00000000,0.00000000,-1.00000000]) ; l_OldMomentum = 1.00000000
    OldPolarization = np.array([0.00000000,-1.00000000,0.00000000]) ; l_OldPolarization = 1.00000000
    #C4OpBoundaryProcess::DielectricDielectric.incident_ray_oblique.FresnelRefraction
    NewMomentum0 = np.array([-0.02711790,0.00000000,-0.99963224]) ; l_NewMomentum0 = 1.00000000
    #C4OpBoundaryProcess::DielectricDielectric.incident_ray_oblique ; sint1 = 0.29632217 ; cost1 = 0.95508804 ; cost2 = 0.96277244 ; Rindex2 = 1.48426314 ; Rindex1 = 1.35398554 ; alpha = -0.10032030
    U4Recorder::UserSteppingAction_Optical PIDX 552 post U4StepPoint::DescPositionTime (    100.000      0.000    169.141      0.119) is_fastsim_flag 0 FAKES_SKIP 0 is_fake 0 fakemask 0
    C4CustomART::doIt
     pmtid 0
     _qe                      :     0.3475
     minus_cos_theta          :    -0.9617
     dot_pol_cross_mom_nrm    :     0.2742

     stack 
    Stack<double,4>
    idx 0
    Layr
      n:(    1.4843     0.0000)s  d:    0.0000
     st:(    0.2742     0.0000)s ct:(    0.9617    -0.0000)s
     rs:(   -0.1416     0.0000)s rp:(    0.1253     0.0000)s
     ts:(    0.8584     0.0000)s tp:(    0.8603     0.0000)s
    S
    | (    1.0000     0.0000)s (    0.0000     0.0000)s |
    | (    0.0000     0.0000)s (    1.0000     0.0000)s |

    P
    | (    1.0000     0.0000)s (    0.0000     0.0000)s |
    | (    0.0000     0.0000)s (    1.0000     0.0000)s |

    idx 1
    Layr
      n:(    1.9413     0.0000)s  d:   36.4900
     st:(    0.2097     0.0000)s ct:(    0.9778     0.0000)s
     rs:(   -0.1808    -0.2813)s rp:(    0.1616     0.2720)s
     ts:(    0.8192    -0.2813)s tp:(    0.8211    -0.2759)s
    S
    | (    0.5935    -1.0024)s (   -0.0840    -0.1419)s |
    | (   -0.0840     0.1419)s (    0.5935     1.0024)s |

    P
    | (    0.5922    -1.0002)s (    0.0742     0.1253)s |
    | (    0.0742    -0.1253)s (    0.5922     1.0002)s |

    idx 2
    Layr
      n:(    2.2735     1.4071)s  d:   21.1300
     st:(    0.1294    -0.0801)s ct:(    0.9949     0.0104)s
     rs:(    0.5195     0.2164)s rp:(   -0.4476    -0.2262)s
     ts:(    1.5195     0.2164)s tp:(    1.5741     0.2629)s
    S
    | (    1.6818    -0.6708)s (    0.1115    -0.2195)s |
    | (   -0.4928    -0.3518)s (    0.3719     0.6353)s |

    P
    | (    1.6772    -0.6819)s (   -0.1113     0.2045)s |
    | (    0.4565     0.3460)s (    0.3761     0.6328)s |

    idx 3
    Layr
      n:(    1.0000     0.0000)s  d:    0.0000
     st:(    0.4070     0.0000)s ct:(    0.9134     0.0000)s
     rs:(    0.0000     0.0000)s rp:(    0.0000     0.0000)s
     ts:(    0.0000     0.0000)s tp:(    0.0000     0.0000)s
    S
    | (    0.6450    -0.0919)s (    0.3550     0.0919)s |
    | (    0.3550     0.0919)s (    0.6450    -0.0919)s |

    P
    | (    0.6180    -0.1032)s (   -0.3000    -0.0936)s |
    | (   -0.3000    -0.0936)s (    0.6180    -0.1032)s |

    comp
    Layr
      n:(    0.0000     0.0000)s  d:    0.0000
     st:(    0.0000     0.0000)s ct:(    0.0000     0.0000)s
     rs:(    0.0130    -0.1669)s rp:(   -0.0412     0.1522)s
     ts:(    0.0099     0.6936)s tp:(   -0.0082     0.7286)s
    S
    | (    0.0206    -1.4415)s (    0.2017    -0.8911)s |
    | (   -0.2404    -0.0222)s (   -0.1398     0.3986)s |

    P
    | (   -0.0154    -1.3724)s (   -0.1899     0.7641)s |
    | (    0.2095     0.0542)s (   -0.1137     0.4058)s |


    ART_
      A_s      0.6641  A_p      0.6354  A_av     0.6497  A        0.6641
      R_s      0.0280  R_p      0.0249  R_av     0.0265  R        0.0280
      T_s      0.3079  T_p      0.3397  T_av     0.3238  T        0.3079
     SUM_s     1.0000 SUM_p     1.0000 SUM_a     1.0000 SUM_      1.0000
     SF        1.0000 wl      420.0000 ART_av    1.0000 mct     -0.9617


     theAbsorption    :     0.6641
     theReflectivity  :     0.0835
     theTransmittance :     0.9165
     theEfficiency    :     0.5371
    C4OpBoundaryProcess::PostStepDoIt PIDX 552 m_custom_status Y
    #C4OpBoundaryProcess::PostStepDoIt.Y.DiDi.0 PIDX 552
    theGlobalPoint = np.array([99.85984668,0.00000000,163.97455814]) ; l_theGlobalPoint = 191.98865773
    mom0 = np.array([-0.02711790,0.00000000,-0.99963224]) ; l_OldMomentum = 1.00000000
    pol0 = np.array([0.00000000,-1.00000000,0.00000000]) ; l_OldPolarization = 1.00000000
    #C4OpBoundaryProcess::DielectricDielectric Rindex1 1.48426314 Rindex2 1.00000100
    #C4OpBoundaryProcess::DielectricDielectric.do polished YES
    theGlobalPoint = np.array([99.85984668,0.00000000,163.97455814]) ; l_theGlobalPoint = 191.98865773
    theGlobalNormal = np.array([0.30020198,0.00000000,0.95387566]) ; l_theGlobalNormal = 1.00000000
    theFacetNormal = np.array([0.30020198,0.00000000,0.95387566]) ; l_theFacetNormal = 1.00000000
    theRecoveredNormal = np.array([0.30020198,0.00000000,0.95387566]) ; l_theRecoveredNormal = 1.00000000
    OldMomentum = np.array([-0.02711790,0.00000000,-0.99963224]) ; l_OldMomentum = 1.00000000
    OldPolarization = np.array([0.00000000,-1.00000000,0.00000000]) ; l_OldPolarization = 1.00000000
    #C4OpBoundaryProcess::DielectricDielectric.incident_ray_oblique.FresnelRefraction
    NewMomentum0 = np.array([0.11403715,0.00000000,-0.99347649]) ; l_NewMomentum0 = 1.00000000
    #C4OpBoundaryProcess::DielectricDielectric.incident_ray_oblique ; sint1 = 0.27422447 ; cost1 = 0.96166571 ; cost2 = 0.91341886 ; Rindex2 = 1.00000100 ; Rindex1 = 1.48426314 ; alpha = 0.34626286
    #C4OpBoundaryProcess::PostStepDoIt.Y.DiDi.1 PIDX 552
    mom1 = np.array([0.11403715,0.00000000,-0.99347649]) ; l_NewMomentum = 1.00000000
    pol1 = np.array([0.00000000,-1.00000000,-0.00000000]) ; l_NewPolarization = 1.00000000
    nrm = np.array([0.30020198,0.00000000,0.95387566]) ; l_theRecoveredNormal = 1.00000000
    U4Recorder::UserSteppingAction_Optical PIDX 552 post U4StepPoint::DescPositionTime (     99.860      0.000    163.975      0.145) is_fastsim_flag 0 FAKES_SKIP 0 is_fake 0 fakemask 0
    C4OpBoundaryProcess::PostStepDoIt PIDX 552 m_custom_status Z
    U4Recorder::UserSteppingAction_Optical PIDX 552 post U4StepPoint::DescPositionTime (    135.899      0.000   -149.990      1.199) is_fastsim_flag 0 FAKES_SKIP 0 is_fake 0 fakemask 0
    U4Recorder::PostUserTrackingAction_Optical.fStopAndKill  ulabel.id    552 seq.brief TO BT BT SA
    2023-08-09 00:53:42.615 INFO  [68875] [SEvt::save@3243]  dir /tmp/blyth/opticks/GEOM/FewPMT/G4CXTest/ALL0/n001 index 1 instance 1 OPTICKS_SAVE_COMP  genstep,photon,record,seq,prd,hit,domain,inphoton,tag,flat,aux,sup
    2023-08-09 00:53:42.666 INFO  [68875] [SEvt::clear_except@1413] SEvt::clear_except
    2023-08-09 00:53:42.666 ERROR [68875] [G4CXApp::SaveMeta@256]  NULL savedir 
    2023-08-09 00:53:42.666 INFO  [68875] [G4CXApp::EndOfEventAction@231] not-(WITH_PMTSIM and POM_DEBUG)
    2023-08-09 00:53:42.666 INFO  [68875] [SEvt::clear@1392] SEvt::clear

    //qsim.propagate.head idx 552 : bnc 0 cosTheta -0.95508808 
    //qsim.propagate.head idx 552 : mom = np.array([0.00000000,0.00000000,-1.00000000]) ; lmom = 1.00000000  
    //qsim.propagate.head idx 552 : pos = np.array([ 100.00000,   0.00000, 195.00000]) ; lpos = 219.14607239 
    //qsim.propagate.head idx 552 : nrm = np.array([(0.29632217,0.00000000,0.95508808]) ; lnrm = 1.00000000  
    //qsim.propagate_to_boundary.head idx 552 : u_absorption 0.33028582 logf(u_absorption) -1.10779667 absorption_length 37213.9219 absorption_distance 41225.457031 
    //qsim.propagate_to_boundary.head idx 552 : post = np.array([ 100.00000,   0.00000, 195.00000,   0.00000]) 
    //qsim.propagate_to_boundary.head idx 552 : distance_to_boundary    25.8590 absorption_distance 41225.4570 scattering_distance 96441.0859 
    //qsim.propagate_to_boundary.head idx 552 : u_scattering     0.5812 u_absorption     0.3303 
    //qsim.propagate idx 552 bounce 0 command 3 flag 0 s.optical.x 0 s.optical.y 1 
    //qsim.propagate.WITH_CUSTOM4 idx 552  BOUNDARY ems 1 lposcost   0.861 
    //qsim.propagate_at_boundary.head idx 552 : theTransmittance = -1.00000000 
    //qsim.propagate_at_boundary.head idx 552 : nrm = np.array([0.29632217,0.00000000,0.95508808]) ; lnrm = 1.00000000  
    //qsim.propagate_at_boundary.head idx 552 : pos = np.array([ 100.00000,   0.00000, 169.14096]) ; lpos = 196.49087524 
    //qsim.propagate_at_boundary.head idx 552 : mom0 = np.array([0.00000000,0.00000000,-1.00000000]) ; lmom0 = 1.00000000 
    //qsim.propagate_at_boundary.head idx 552 : pol0 = np.array([-0.00000000,-1.00000000,-0.00000000]) ; lpol0 = 1.00000000 
    //qsim.propagate_at_boundary.head idx 552 : n1,n2,eta = (1.35398555,1.48426318,0.91222739) 
    //qsim.propagate_at_boundary.head idx 552 : c1 = 0.95508808 ; normal_incidence = 0 
    //qsim.propagate_at_boundary.body idx 552 : TransCoeff = 0.99751025 ; n1c1 = 1.29317546 ; n2c2 = 1.42900777 
    //qsim.propagate_at_boundary.body idx 552 : E2_t = np.array([0.95010173,0.00000000]) ; lE2_t = 0.95010173 
    //qsim.propagate_at_boundary.body idx 552 : A_trans = np.array([0.00000000,-1.00000000,0.00000000]) ; lA_trans = 1.00000000 
    //qsim.propagate_at_boundary.body idx 552 : u_reflect     0.1106 TransCoeff     0.9975 reflect 0 
    //qsim.propagate_at_boundary.body idx 552 : mom0 = np.array([0.00000000,0.00000000,-1.00000000]) ; lmom0 = 1.00000000 
    //qsim.propagate_at_boundary.body idx 552 : pos = np.array([ 100.00000,   0.00000, 169.14096]) ; lpos = 196.49087524 
    //qsim.propagate_at_boundary.body idx 552 : nrm = np.array([0.29632217,0.00000000,0.95508808]) ; lnrm = 1.00000000 
    //qsim.propagate_at_boundary.body idx 552 : n1 = 1.35398555 ; n2 = 1.48426318 ; eta = 0.91222739  
    //qsim.propagate_at_boundary.body idx 552 : c1 = 0.95508808 ; eta_c1 = 0.87125748 ; c2 = 0.96277249 ; eta_c1__c2 = -0.09151500 
    //qsim.propagate_at_boundary.tail idx 552 : reflect 0 tir 0 TransCoeff     0.9975 u_reflect     0.1106 
    //qsim.propagate_at_boundary.tail idx 552 : mom1 = np.array([-0.02711792,0.00000000,-0.99963230]) ; lmom1 = 1.00000000  
    //qsim.propagate_at_boundary.tail idx 552 : pol1 = np.array([0.00000000,-1.00000000,0.00000000]) ; lpol1 = 1.00000000 

    //qsim.propagate.head idx 552 : bnc 1 cosTheta -0.96166575 
    //qsim.propagate.head idx 552 : mom = np.array([-0.02711792,0.00000000,-0.99963230]) ; lmom = 1.00000000  
    //qsim.propagate.head idx 552 : pos = np.array([ 100.00000,   0.00000, 169.14096]) ; lpos = 196.49087524 
    //qsim.propagate.head idx 552 : nrm = np.array([(0.30020195,0.00000000,0.95387566]) ; lnrm = 1.00000000  
    //qsim.propagate_to_boundary.head idx 552 : u_absorption 0.56169021 logf(u_absorption) -0.57680476 absorption_length  1562.9586 absorption_distance 901.521973 
    //qsim.propagate_to_boundary.head idx 552 : post = np.array([ 100.00000,   0.00000, 169.14096,   0.11860]) 
    //qsim.propagate_to_boundary.head idx 552 : distance_to_boundary     5.1683 absorption_distance   901.5220 scattering_distance 3043071.0000 
    //qsim.propagate_to_boundary.head idx 552 : u_scattering     0.0477 u_absorption     0.5617 
    //qsim.propagate idx 552 bounce 1 command 3 flag 0 s.optical.x 7 s.optical.y 4 
    //qsim.propagate.WITH_CUSTOM4 idx 552  BOUNDARY ems 4 lposcost   0.854 
    //qsim::propagate_at_surface_CustomART idx     552 : mom = np.array([-0.02711792,0.00000000,-0.99963230]) ; lmom = 1.00000000 
    //qsim::propagate_at_surface_CustomART idx     552 : pol = np.array([0.00000000,-1.00000000,0.00000000]) ; lpol = 1.00000000 
    //qsim::propagate_at_surface_CustomART idx     552 : nrm = np.array([0.30020195,0.00000000,0.95387566]) ; lnrm = 1.00000000 
    //qsim::propagate_at_surface_CustomART idx     552 : cross_mom_nrm = np.array([0.00000000,-0.27422443,-0.00000000]) ; lcross_mom_nrm = 0.27422443  
    //qsim::propagate_at_surface_CustomART idx     552 : dot_pol_cross_mom_nrm = 0.27422443 
    //qsim::propagate_at_surface_CustomART idx     552 : minus_cos_theta = -0.96166575 
    //qsim::propagate_at_surface_CustomART idx 552 lpmtid 0 wl 420.000 mct  -0.962 dpcmn   0.274 ARTE (   0.664   0.083   0.917   0.537 ) 
    //qsim.propagate_at_surface_CustomART idx 552 lpmtid 0 ARTE (   0.664   0.083   0.917   0.537 ) u_theAbsorption    0.663 action 1 
    2023-08-09 00:53:42.788 INFO  [68875] [SEvt::save@3243]  dir /tmp/blyth/opticks/GEOM/FewPMT/G4CXTest/ALL0/p001 index 1 instance 0 OPTICKS_SAVE_COMP  genstep,photon,record,seq,prd,hit,domain,inphoton,tag,flat,aux,sup
    2023-08-09 00:53:42.819 INFO  [68875] [SEvt::clear_except@1413] SEvt::clear_except
    2023-08-09 00:53:42.820 INFO  [68875] [G4CXApp::EndOfRunAction@182] 
    Python 3.7.7 (default, May  7 2020, 21:25:33) 
    Type 'copyright', 'credits' or 'license' for more information
    IPython 7.18.1 -- An enhanced Interactive Python. Type '?' for help.
    [from opticks.ana.p import * 
    CSGFoundry.CFBase returning [/home/blyth/.opticks/GEOM/FewPMT], note:[via GEOM] 
    ]from opticks.ana.p import * 
    detect fold.IsRemoteSession forcing MODE:0
    GLOBAL:0 MODE:0 SEL:0
    INFO:opticks.ana.pvplt:SEvt.Load NEVT:0 
    INFO:opticks.ana.fold:Fold.Load args ('$AFOLD',) quiet:1
    INFO:opticks.ana.fold:Fold.Load args ('/tmp/blyth/opticks/GEOM/FewPMT/G4CXTest/ALL0',) quiet:1
    INFO:opticks.ana.pvplt:init_ee with_photon_meta:1 with_ff:0
    INFO:opticks.ana.pvplt:SEvt.__init__  symbol a pid 552 opt  off [0. 0. 0.] 
    SEvt symbol a pid 552 opt  off [0. 0. 0.] a.f.base /tmp/blyth/opticks/GEOM/FewPMT/G4CXTest/ALL0/p001 
    INFO:opticks.ana.pvplt:SEvt.Load NEVT:0 
    INFO:opticks.ana.fold:Fold.Load args ('$BFOLD',) quiet:1
    INFO:opticks.ana.fold:Fold.Load args ('/tmp/blyth/opticks/GEOM/FewPMT/G4CXTest/ALL0',) quiet:1
    INFO:opticks.ana.pvplt:init_ee with_photon_meta:1 with_ff:0
    INFO:opticks.ana.pvplt:SEvt.__init__  symbol b pid 552 opt  off [0. 0. 0.] 
    SEvt symbol b pid 552 opt  off [0. 0. 0.] b.f.base /tmp/blyth/opticks/GEOM/FewPMT/G4CXTest/ALL0/n001 
    /home/blyth/.opticks/GEOM/FewPMT/CSGFoundry
    min_stamp:2023-08-09 00:53:41.385259
    max_stamp:2023-08-09 00:53:41.386259
    age_stamp:0:00:02.565700
         meshname :                 (8,)  : /home/blyth/.opticks/GEOM/FewPMT/CSGFoundry/meshname.txt 
         primname :                 (8,)  : /home/blyth/.opticks/GEOM/FewPMT/CSGFoundry/primname.txt 
          mmlabel :                 (1,)  : /home/blyth/.opticks/GEOM/FewPMT/CSGFoundry/mmlabel.txt 
             meta :                (15,)  : /home/blyth/.opticks/GEOM/FewPMT/CSGFoundry/meta.txt 
            solid :            (1, 3, 4)  : /home/blyth/.opticks/GEOM/FewPMT/CSGFoundry/solid.npy 
             prim :            (8, 4, 4)  : /home/blyth/.opticks/GEOM/FewPMT/CSGFoundry/prim.npy 
             node :           (14, 4, 4)  : /home/blyth/.opticks/GEOM/FewPMT/CSGFoundry/node.npy 
             tran :           (11, 4, 4)  : /home/blyth/.opticks/GEOM/FewPMT/CSGFoundry/tran.npy 
             itra :           (11, 4, 4)  : /home/blyth/.opticks/GEOM/FewPMT/CSGFoundry/itra.npy 
             inst :            (1, 4, 4)  : /home/blyth/.opticks/GEOM/FewPMT/CSGFoundry/inst.npy 
    at
     [[b'3547' b'4' b'TO BT SD                                                                                        ']
     [b'3042' b'1' b'TO BT SA                                                                                        ']
     [b'901' b'24' b'TO BT BT SR BT BT SA                                                                            ']
     [b'815' b'2' b'TO BT BT SR SA                                                                                  ']
     [b'323' b'58' b'TO BT BT SR BR SR BT BT SA                                                                      ']
     [b'318' b'27' b'TO BT BT SR BR SR SA                                                                            ']
     [b'283' b'0' b'TO BT BR BT SA                                                                                  ']
     [b'231' b'7' b'TO BT BT SA                                                                                     ']
     [b'111' b'165' b'TO BT BT SR BR SR BR SR BT BT SA                                                                ']
     [b'102' b'80' b'TO BT BT SR BR SR BR SR SA                                                                      ']
     [b'82' b'31' b'TO BT BT SR BR SA                                                                               ']
     [b'43' b'75' b'TO BT AB                                                                                        ']
     [b'36' b'104' b'TO BT BT SR BR SR BR SR BR SR SA                                                                ']
     [b'35' b'62' b'TO BT BT SR BR SR BR SA                                                                         ']
     [b'34' b'107' b'TO BT BT SR BR SR BR SR BR SR BT BT SA                                                          ']]
    bt
     [[b'3483' b'4' b'TO BT SD                                                                                        ']
     [b'3094' b'1' b'TO BT SA                                                                                        ']
     [b'915' b'0' b'TO BT BT SR BT BT SA                                                                            ']
     [b'864' b'8' b'TO BT BT SR SA                                                                                  ']
     [b'303' b'28' b'TO BT BT SR BR SR SA                                                                            ']
     [b'294' b'40' b'TO BT BT SR BR SR BT BT SA                                                                      ']
     [b'287' b'123' b'TO BT BR BT SA                                                                                  ']
     [b'243' b'33' b'TO BT BT SA                                                                                     ']
     [b'114' b'13' b'TO BT BT SR BR SR BR SR BT BT SA                                                                ']
     [b'113' b'14' b'TO BT BT SR BR SR BR SR SA                                                                      ']
     [b'79' b'228' b'TO BT BT SR BR SA                                                                               ']
     [b'39' b'173' b'TO BT BT SR BR SR BR SR BR SR SA                                                                ']
     [b'28' b'785' b'TO BT BT SR BR SR BR SA                                                                         ']
     [b'26' b'38' b'TO BR SA                                                                                        ']
     [b'22' b'357' b'TO BT AB                                                                                        ']]
    SAB
    SEvt symbol a pid 552 opt  off [0. 0. 0.] a.f.base /tmp/blyth/opticks/GEOM/FewPMT/G4CXTest/ALL0/p001 
    a

    CMDLINE:/data/blyth/junotop/opticks/g4cx/tests/G4CXTest.py
    a.base:/tmp/blyth/opticks/GEOM/FewPMT/G4CXTest/ALL0/p001

      : a.NPFold_index                                     :                 (9,) : 0:00:01.603141 
      : a.genstep                                          :            (1, 6, 4) : 0:00:01.602141 
      : a.photon                                           :        (10000, 4, 4) : 0:00:01.602141 
      : a.photon_meta                                      :                    3 : 0:00:01.602141 
      : a.record                                           :    (10000, 32, 4, 4) : 0:00:01.597141 
      : a.record_meta                                      :                    1 : 0:00:01.587141 
      : a.seq                                              :        (10000, 2, 2) : 0:00:01.584141 
      : a.prd                                              :    (10000, 32, 2, 4) : 0:00:01.581141 
      : a.hit                                              :         (3549, 4, 4) : 0:00:01.574141 
      : a.domain                                           :            (2, 4, 4) : 0:00:01.574141 
      : a.domain_meta                                      :                    4 : 0:00:01.574141 
      : a.tag                                              :           (10000, 4) : 0:00:01.574141 
      : a.flat                                             :          (10000, 64) : 0:00:01.573141 
      : a.NPFold_meta                                      :                   14 : 0:00:01.571141 
      : a.sframe                                           :            (4, 4, 4) : 0:00:01.571141 
      : a.sframe_meta                                      :                    5 : 0:00:01.571141 

     min_stamp : 2023-08-09 00:53:42.787271 
     max_stamp : 2023-08-09 00:53:42.819271 
     dif_stamp : 0:00:00.032000 
     age_stamp : 0:00:01.571141 
    SEvt symbol b pid 552 opt  off [0. 0. 0.] b.f.base /tmp/blyth/opticks/GEOM/FewPMT/G4CXTest/ALL0/n001 
    b

    CMDLINE:/data/blyth/junotop/opticks/g4cx/tests/G4CXTest.py
    b.base:/tmp/blyth/opticks/GEOM/FewPMT/G4CXTest/ALL0/n001

      : b.NPFold_index                                     :                (11,) : 0:00:01.775500 
      : b.genstep                                          :            (1, 6, 4) : 0:00:01.775500 
      : b.photon                                           :        (10000, 4, 4) : 0:00:01.775500 
      : b.photon_meta                                      :                    3 : 0:00:01.775500 
      : b.record                                           :    (10000, 32, 4, 4) : 0:00:01.770499 
      : b.record_meta                                      :                    1 : 0:00:01.760499 
      : b.seq                                              :        (10000, 2, 2) : 0:00:01.757499 
      : b.prd                                              :    (10000, 32, 2, 4) : 0:00:01.754499 
      : b.hit                                              :         (3487, 4, 4) : 0:00:01.748499 
      : b.domain                                           :            (2, 4, 4) : 0:00:01.747499 
      : b.domain_meta                                      :                    4 : 0:00:01.747499 
      : b.tag                                              :           (10000, 4) : 0:00:01.747499 
      : b.flat                                             :          (10000, 64) : 0:00:01.746499 
      : b.aux                                              :    (10000, 32, 4, 4) : 0:00:01.740499 
      : b.sup                                              :        (10000, 6, 4) : 0:00:01.727499 
      : b.NPFold_meta                                      :                   14 : 0:00:01.726499 
      : b.pho0                                             :           (10000, 4) : 0:00:01.725499 
      : b.pho                                              :           (10000, 4) : 0:00:01.725499 
      : b.gs                                               :               (1, 4) : 0:00:01.725499 
      : b.sframe                                           :            (4, 4, 4) : 0:00:01.724499 
      : b.sframe_meta                                      :                    5 : 0:00:01.724499 

     min_stamp : 2023-08-09 00:53:42.615269 
     max_stamp : 2023-08-09 00:53:42.666270 
     dif_stamp : 0:00:00.051001 
     age_stamp : 0:00:01.724499 
    qcf.aqu : np.c_[n,x,u][o][lim] : uniques in descending count order with first index x
    [[b'3547' b'4' b'TO BT SD                                                                                        ']
     [b'3042' b'1' b'TO BT SA                                                                                        ']
     [b'901' b'24' b'TO BT BT SR BT BT SA                                                                            ']
     [b'815' b'2' b'TO BT BT SR SA                                                                                  ']
     [b'323' b'58' b'TO BT BT SR BR SR BT BT SA                                                                      ']
     [b'318' b'27' b'TO BT BT SR BR SR SA                                                                            ']
     [b'283' b'0' b'TO BT BR BT SA                                                                                  ']
     [b'231' b'7' b'TO BT BT SA                                                                                     ']
     [b'111' b'165' b'TO BT BT SR BR SR BR SR BT BT SA                                                                ']
     [b'102' b'80' b'TO BT BT SR BR SR BR SR SA                                                                      ']]
    qcf.bqu : np.c_[n,x,u][o][lim] : uniques in descending count order with first index x
    [[b'3483' b'4' b'TO BT SD                                                                                        ']
     [b'3094' b'1' b'TO BT SA                                                                                        ']
     [b'915' b'0' b'TO BT BT SR BT BT SA                                                                            ']
     [b'864' b'8' b'TO BT BT SR SA                                                                                  ']
     [b'303' b'28' b'TO BT BT SR BR SR SA                                                                            ']
     [b'294' b'40' b'TO BT BT SR BR SR BT BT SA                                                                      ']
     [b'287' b'123' b'TO BT BR BT SA                                                                                  ']
     [b'243' b'33' b'TO BT BT SA                                                                                     ']
     [b'114' b'13' b'TO BT BT SR BR SR BR SR BT BT SA                                                                ']
     [b'113' b'14' b'TO BT BT SR BR SR BR SR SA                                                                      ']]
    a.CHECK : rain_point_xpositive_100 
    b.CHECK : rain_point_xpositive_100 
    QCF qcf :  
    a.q 10000 b.q 10000 lim slice(None, None, None) 
    c2sum :    16.7842 c2n :    17.0000 c2per:     0.9873  C2CUT:   30 
    c2sum/c2n:c2per(C2CUT)  16.78/17:0.987 (30)

    np.c_[siq,_quo,siq,sabo2,sc2,sabo1][0:25]  ## A-B history frequency chi2 comparison 
    [[' 0' 'TO BT SD                                                      ' ' 0' '  3547   3483' ' 0.5826' '     4      4']
     [' 1' 'TO BT SA                                                      ' ' 1' '  3042   3094' ' 0.4407' '     1      1']
     [' 2' 'TO BT BT SR BT BT SA                                          ' ' 2' '   901    915' ' 0.1079' '    24      0']
     [' 3' 'TO BT BT SR SA                                                ' ' 3' '   815    864' ' 1.4300' '     2      8']
     [' 4' 'TO BT BT SR BR SR BT BT SA                                    ' ' 4' '   323    294' ' 1.3630' '    58     40']
     [' 5' 'TO BT BT SR BR SR SA                                          ' ' 5' '   318    303' ' 0.3623' '    27     28']
     [' 6' 'TO BT BR BT SA                                                ' ' 6' '   283    287' ' 0.0281' '     0    123']
     [' 7' 'TO BT BT SA                                                   ' ' 7' '   231    243' ' 0.3038' '     7     33']
     [' 8' 'TO BT BT SR BR SR BR SR BT BT SA                              ' ' 8' '   111    114' ' 0.0400' '   165     13']
     [' 9' 'TO BT BT SR BR SR BR SR SA                                    ' ' 9' '   102    113' ' 0.5628' '    80     14']
     ['10' 'TO BT BT SR BR SA                                             ' '10' '    82     79' ' 0.0559' '    31    228']
     ['11' 'TO BT AB                                                      ' '11' '    43     22' ' 6.7846' '    75    357']
     ['12' 'TO BT BT SR BR SR BR SR BR SR SA                              ' '12' '    36     39' ' 0.1200' '   104    173']
     ['13' 'TO BT BT SR BR SR BR SA                                       ' '13' '    35     28' ' 0.7778' '    62    785']
     ['14' 'TO BT BT SR BR SR BR SR BR SR BT BT SA                        ' '14' '    34     22' ' 2.5714' '   107      9']
     ['15' 'TO BR SA                                                      ' '15' '    20     26' ' 0.7826' '   159     38']
     ['16' 'TO BT BT SR BR SR BR SR BR SR BR SR SA                        ' '16' '    15     19' ' 0.4706' '  1851    394']
     ['17' 'TO BT BT SR BR SR BR SR BR SR BR SR BT BT SA                  ' '17' '    10     11' ' 0.0000' '   996    405']
     ['18' 'TO BT BT SR BR SR BR SR BR SA                                 ' '18' '     7      7' ' 0.0000' '   119    459']
     ['19' 'TO BT BT SR BR SR BR SR BR SR BR SR BR SR SA                  ' '19' '     6      6' ' 0.0000' '  2930    627']
     ['20' 'TO AB                                                         ' '20' '     4      6' ' 0.0000' '   336   1814']
     ['21' 'TO BT BT SR BR SR BR SR BR SR BR SA                           ' '21' '     5      4' ' 0.0000' '   209   1270']
     ['22' 'TO BT BT SR BR SR BR SR BR SR BR SR BR SR BT BT SA            ' '22' '     4      5' ' 0.0000' '  7396   3808']
     ['23' 'TO BT BT SR BT AB                                             ' '23' '     4      2' ' 0.0000' '   534   3757']
     ['24' 'TO BT BT SR BR SR BT AB                                       ' '24' '     2      0' ' 0.0000' '   133     -1']]

    np.c_[siq,_quo,siq,sabo2,sc2,sabo1][bzero]  ## bzero: A histories not in B 
    []

    np.c_[siq,_quo,siq,sabo2,sc2,sabo1][azero]  ## azero: B histories not in A 
    []
    PICK=AB MODE=0 SEL=0 ./G4CXAppTest.sh ana 
    not plotting as MODE 0 in environ
    not plotting as MODE 0 in environ

    In [1]: 





