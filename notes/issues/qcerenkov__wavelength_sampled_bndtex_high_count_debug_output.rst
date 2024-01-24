qcerenkov__wavelength_sampled_bndtex_high_count_debug_output
===============================================================

Input genstep running with most launch level logging placed behind SEvt__MINIMAL control
shows some logging from high count rindex sampling at low wavelengths.

TODO: check the RINDEX plot to understand this


::

    N[blyth@localhost opticks]$ ~/o/CSGOptiX/cxs_min_igs.sh

    .
                    GEOM : J23_1_0_rc3_ok0 
                  LOGDIR : /home/blyth/tmp/GEOM/J23_1_0_rc3_ok0/CSGOptiXSMTest/ALL1 
                 BINBASE : /home/blyth/tmp/GEOM/J23_1_0_rc3_ok0/CSGOptiXSMTest 
                     CVD :  
    CUDA_VISIBLE_DEVICES : 1 
                    SDIR : /data/blyth/junotop/opticks/CSGOptiX 
                    FOLD :  
                     LOG :  
                    NEVT :  
    /data/blyth/junotop/opticks/CSGOptiX/cxs_min.sh : run : delete prior LOGFILE CSGOptiXSMTest.log
    2024-01-24 18:07:01.455  455225482 : [/data/blyth/junotop/opticks/CSGOptiX/cxs_min.sh 
    //qcerenkov::wavelength_sampled_bndtex idx   7702 sampledRI   1.785 cosTheta   0.945 sin2Theta   0.106 wavelength 152.454 count 77 matline 35 
    //qcerenkov::wavelength_sampled_bndtex idx   5834 sampledRI   1.692 cosTheta   0.983 sin2Theta   0.033 wavelength 164.080 count 61 matline 35 
    //qcerenkov::wavelength_sampled_bndtex idx   6000 sampledRI   1.754 cosTheta   0.936 sin2Theta   0.125 wavelength 147.471 count 57 matline 35 
    //qcerenkov::wavelength_sampled_bndtex idx   1147 sampledRI   1.773 cosTheta   0.939 sin2Theta   0.117 wavelength 149.145 count 54 matline 35 
    //qcerenkov::wavelength_sampled_bndtex idx   5271 sampledRI   1.790 cosTheta   0.901 sin2Theta   0.187 wavelength 157.403 count 55 matline 35 
    //qcerenkov::wavelength_sampled_bndtex idx   4231 sampledRI   1.789 cosTheta   0.940 sin2Theta   0.115 wavelength 155.526 count 59 matline 35 
    //qcerenkov::wavelength_sampled_bndtex idx   4839 sampledRI   1.784 cosTheta   0.911 sin2Theta   0.169 wavelength 150.949 count 80 matline 35 
    //qcerenkov::wavelength_sampled_bndtex idx   4840 sampledRI   1.789 cosTheta   0.908 sin2Theta   0.175 wavelength 156.243 count 62 matline 35 
    //qcerenkov::wavelength_sampled_bndtex idx    598 sampledRI   1.772 cosTheta   0.927 sin2Theta   0.141 wavelength 148.979 count 57 matline 35 
    2024-01-24 18:07:12.040  040930314 : ]/data/blyth/junotop/opticks/CSGOptiX/cxs_min.sh 
    [sreport.main  argv0 sreport dirp /home/blyth/tmp/GEOM/J23_1_0_rc3_ok0/CSGOptiXSMTest/ALL1 is_executable_sibling_path NO 
    [sreport.main : CREATING REPORT 


::

    287 inline QCERENKOV_METHOD void qcerenkov::wavelength_sampled_bndtex(float& wavelength, float& cosTheta, float& sin2Theta, curandStateXORWOW& rng, cons    t scerenkov& gs, int idx, int gsid ) const
    288 {
    289     //printf("//qcerenkov::wavelength_sampled_bndtex bnd %p gs.matline %d \n", bnd, gs.matline ); 
    290     float u0 ;
    291     float u1 ;
    292     float w ;
    293     float sampledRI ;
    294     float u_maxSin2 ;
    295 
    296     unsigned count = 0 ;
    297 
    298     do {
    299         u0 = curand_uniform(&rng) ;
    300 
    301         w = gs.Wmin + u0*(gs.Wmax - gs.Wmin) ;
    302 
    303         wavelength = gs.Wmin*gs.Wmax/w ; // reciprocalization : arranges flat energy distribution, expressed in wavelength 
    304 
    305         float4 props = bnd->boundary_lookup(wavelength, gs.matline, 0u);
    306 
    307         sampledRI = props.x ;
    308 
    309         //printf("//qcerenkov::wavelength_sampled_bndtex count %d wavelength %10.4f sampledRI %10.4f \n", count, wavelength, sampledRI );  
    310 
    311         cosTheta = gs.BetaInverse / sampledRI ;
    312 
    313         sin2Theta = fmaxf( 0.f, (1.f - cosTheta)*(1.f + cosTheta));
    314 
    315         u1 = curand_uniform(&rng) ;
    316 
    317         u_maxSin2 = u1*gs.maxSin2 ;
    318 
    319         count += 1 ;
    320 
    321     } while ( u_maxSin2 > sin2Theta && count < 100 );
    322 
    323 #if !defined(PRODUCTION) && defined(DEBUG_PIDX)
    324     if(count > 50)
    325     printf("//qcerenkov::wavelength_sampled_bndtex idx %6d sampledRI %7.3f cosTheta %7.3f sin2Theta %7.3f wavelength %7.3f count %d matline %d \n",
    326               idx , sampledRI, cosTheta, sin2Theta, wavelength, count, gs.matline );
    327 #endif
    328 
    329 }



With lots of per launch logging for 1000 launches this warning output was made near invisible::

    2024-01-24 18:06:28.638 INFO  [257916] [QSim::simulate@377]  eventID 798 dt    0.006532 ph       8722 ph/M          0 ht          0 ht/M          0 reset_ YES
    2024-01-24 18:06:28.639 INFO  [257916] [SEvt::save@3967] /home/blyth/tmp/GEOM/J23_1_0_rc3_ok0/CSGOptiXSMTest/ALL1/A798 []
    2024-01-24 18:06:28.647 INFO  [257916] [QSim::simulate@377]  eventID 799 dt    0.006757 ph       8789 ph/M          0 ht          0 ht/M          0 reset_ YES
    2024-01-24 18:06:28.648 INFO  [257916] [SEvt::save@3967] /home/blyth/tmp/GEOM/J23_1_0_rc3_ok0/CSGOptiXSMTest/ALL1/A799 []
    //qcerenkov::wavelength_sampled_bndtex idx   4839 sampledRI   1.784 cosTheta   0.911 sin2Theta   0.169 wavelength 150.949 count 80 matline 35 
    //qcerenkov::wavelength_sampled_bndtex idx   4840 sampledRI   1.789 cosTheta   0.908 sin2Theta   0.175 wavelength 156.243 count 62 matline 35 
    2024-01-24 18:06:28.657 INFO  [257916] [QSim::simulate@377]  eventID 800 dt    0.007076 ph       8822 ph/M          0 ht          0 ht/M          0 reset_ YES
    2024-01-24 18:06:28.657 INFO  [257916] [SEvt::save@3967] /home/blyth/tmp/GEOM/J23_1_0_rc3_ok0/CSGOptiXSMTest/ALL1/A800 []
    2024-01-24 18:06:28.666 INFO  [257916] [QSim::simulate@377]  eventID 801 dt    0.006927 ph       8864 ph/M          0 ht          0 ht/M          0 reset_ YES
    2024-01-24 18:06:28.667 INFO  [257916] [SEvt::save@3967] /home/blyth/tmp/GEOM/J23_1_0_rc3_ok0/CSGOptiXSMTest/ALL1/A801 []



