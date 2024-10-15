cxs_min_major_difference_between_TITAN_RTX_OptiX7p5_and_RTX_5000_Ada_OptiX8p0
==================================================================================

Comparing:

* P : NVIDIA TITAN RTX, OptiX 7.5
* A : NVIDIA RTX 5000 Ada Generation, OptiX 8.0


SRM_TORCH
-----------

Running cxs_min.sh in SRM_TORCH mode on A is unrealistically fast, 
and gives no hits. 


SRM_INPUT_PHOTON
-----------------

Comparing input photon targetting NNVT:0:1000 shows hits in both P and A BUT:

* A : all hits are onto the target PMT with no others
* P : lots of other PMTs hit from reflections off the target PMT 

It looks like no reflection off the target PMT are happening for A ?


HMM: is the A build without Custom4 ? That could explain it. 

* not so, simple the Custom4 external is configured



Do some PIDX comparison between P and A
-----------------------------------------

* note the same photon start position, but are getting different randoms ? 

::

    A[blyth@localhost CSGOptiX]$ PIDX=0 ./cxs_min.sh


    //qsim.propagate.head idx 0 : bnc 0 cosTheta -0.80563819 
    //qsim.propagate.head idx 0 : mom = np.array([-0.16308457,0.53761774,0.82726693]) ; lmom = 1.00000000  
    //qsim.propagate.head idx 0 : pos = np.array([-3191.91016,10522.31836,15746.38477]) ; lpos = 19205.62695312 
    //qsim.propagate.head idx 0 : nrm = np.array([(-0.01087651,0.03585108,-0.99929798]) ; lnrm = 1.00000000  
    //qsim.propagate_to_boundary.head idx 0 : u_absorption 0.00033755 logf(u_absorption) -7.99380875 absorption_length 41631.9062 absorption_distance 332797.500000 
    //qsim.propagate_to_boundary.head idx 0 : post = np.array([-3191.91016,10522.31836,15746.38477,   0.10000]) 


    P[blyth@localhost CSGOptiX]$ PIDX=0 ./cxs_min.sh 

    //qsim.propagate.head idx 0 : bnc 0 cosTheta -0.80563819 
    //qsim.propagate.head idx 0 : mom = np.array([-0.16308457,0.53761774,0.82726693]) ; lmom = 1.00000000  
    //qsim.propagate.head idx 0 : pos = np.array([-3191.91016,10522.31836,15746.38477]) ; lpos = 19205.62695312 
    //qsim.propagate.head idx 0 : nrm = np.array([(-0.01087651,0.03585108,-0.99929798]) ; lnrm = 1.00000000  
    //qsim.propagate_to_boundary.head idx 0 : u_absorption 0.15698862 logf(u_absorption) -1.85158193 absorption_length 41631.9062 absorption_distance 77084.882812 
    //qsim.propagate_to_boundary.head idx 0 : post = np.array([-3191.91016,10522.31836,15746.38477,   0.10000]) 
    //qsim.propagate_to_boundary.head idx 0 : distance_to_boundary   122.6315 absorption_distance 77084.8828 scattering_distance 142337.5469 
    //qsim.propagate_to_boundary.head idx 0 : u_scattering     0.5170 u_absorption     0.1570 
     



A ems 4
---------

::

    //qsim.propagate.body.WITH_CUSTOM4 idx 0  BOUNDARY ems 4 lposcost   0.118 
    //qsim::propagate_at_surface_CustomART idx       0 : mom = np.array([-0.11694922,0.38552967,0.91525394]) ; lmom = 1.00000000 
    //qsim::propagate_at_surface_CustomART idx       0 : pol = np.array([-0.95693922,-0.29028833,0.00000160]) ; lpol = 1.00000000 
    //qsim::propagate_at_surface_CustomART idx       0 : nrm = np.array([-0.19764146,0.65153337,-0.73242205]) ; lnrm = 0.99999994 
    //qsim::propagate_at_surface_CustomART idx       0 : cross_mom_nrm = np.array([-0.87868893,-0.26654831,0.00000033]) ; lcross_mom_nrm = 0.91822779  
    //qsim::propagate_at_surface_CustomART idx       0 : dot_pol_cross_mom_nrm = 0.91822773 
    //qsim::propagate_at_surface_CustomART idx       0 : minus_cos_theta = -0.39605269 
    //qsim::propagate_at_surface_CustomART idx 0 lpmtid 1425 wl 440.000 mct  -0.396 dpcmn   0.918 pre-ARTE 
    //qsim::propagate_at_surface_CustomART idx 0 lpmtid 1425 wl 440.000 mct  -0.396 dpcmn   0.918 ARTE (   0.818   1.000   0.000   0.541 ) 
    //qsim.propagate_at_surface_CustomART idx 0 lpmtid 1425 ARTE (   0.818   1.000   0.000   0.541 ) u_theAbsorption    0.002 action 1 
    //qsim.propagate.tail idx 0 bounce 4 command 1 flag 64 ctx.s.optical.y(ems) 4 

