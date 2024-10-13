story_of_opticks_review
==========================


EXPERIENCE
------------

* when innovating you never get to leap to the solution : development always iterative

* best can hope for is that stay headed in right general direction 
* must learn as go along : and course correct 
* sw dev skills continually improve as you gain experience 
* starting fresh, although painful, often quicker and easier 
  than working with old code 

  * pragmatic reality : have to do both, develop new and work with old 

* when learning new API need to not be constrained by existing code
* learn in unconstrained standalone tests but develop API usable from existing code 
* your skills and knowledge of the problem improve the more you develop 
* -> so 2nd/3rd/.. implementations much simpler and better than the 0th  

* develop from many standalone examples 


Thoughts
----------

* origin story : easy enough to convey
* development story : not so easy, aspects: 

  * moving target:

    * two optical simulations compared against each other 

      * => many many bugs (on both GPU and CPU sides) 
      * many many fixes, sometimes re-implementation needed 

    * geometry changing, getting more complicated, often unnecessarily 
    * PMT optical model changing -> first have to fix bugs on CPU before can bring to GPU    


STORY OF OPTICKS
-----------------

0. 1704 : Newton published "Opticks" 


env repo::

    * 2013-07-24 a9ae0e192 - opw- bash funcs for trying out optical photon weighting (11 years ago) <Simon C Blyth>
    * 2013-07-24 c6988de88 - looking into muon_simulation/optical_photon_weighting opw- (11 years ago) <Simon C Blyth>
    * 2013-12  purchased rMBP in HK, with NVIDIA Geforce 750M 2048 MB VRAM (after DYB shifts)
    * 2014-03-21 46de320c3 - take a look at pyopengl and glumpy (10 years ago) <Simon C Blyth>


Summary
~~~~~~~~~

1. (April 2014) Beginning : get Chroma GPU optical photon simulation to work with Dyb 
2. (July 2015) jump from Chroma -> NVIDIA OptiX (50x) and coined the name "Opticks" : flesh out impl
3. (July 2017) implement fully analytic CSG : needed for near perfect G4 match  
4. (August 2019) BOMBSHELL : NVIDIA 6.5->7.0 (like Geant3->4 ) 
5. (Jan 2021) Start transition from NVIDIA 6.5 to 7.0, "new" Opticks impl 
6. (June 2021 onwards) Hard slog, fixing many many issues with co-evolving CPU+GPU simulations


Brief History
~~~~~~~~~~~~~~~


1. (2014) Beginning : get Chroma GPU optical photon simulation to work with Dyb 

  * 200x http://localhost/env/presentation/gpu_optical_photon_simulation.html
  * 1st presention April 2014
  * my dyb infratructure experience made me comfortble with finding nd installing 
    many different open source projects, so getting Chroma installed was familiar 

  * Chroma claimed 200x Geant4 : with a triangulated geometry 
  * implemented trianglated geometry export from Geant4 with material/surface props using DAE/COLLADA (standard 3D file format)

    * liberated geometry from Geant4 : can use with many pkgs

  * pyopengl using single VBO for entire geometry 

    * fast/smooth vizualization on circa 2013 macbook pro with NVIDIA Geforce GPU 2GB VRAM  
    * first expts using pyopengl to make renders from the triangles took a few hours on 
      a saturday afternoon in NTU library : I was shocked by the performance 
    * personal experience of viz speed very powerful 

    * presented to Geant4 collab meeting in Okinawa in Sep 2014 
    * http://localhost/env/presentation/g4dae_geometry_exporter.html

2. jump from Chroma -> NVIDA OptiX (50x) and coined the name "Opticks" : flesh out impl

   * July 2015
   * http://localhost/env/presentation/optical_photon_simulation_with_nvidia_optix.html

   * Chroma issues (inconvenient python impl, no multi-GPU) made me look for alternatives
   * I found NVIDIA OptiX : trying it out, single GPU performance leap made be drop Chroma immediately 
   * and OptiX had near perfect performance scaling up to 4 GPUs 

   * http://localhost/env/presentation/optical_photon_simulation_with_nvidia_optix.html

     * p11 : synthesis of sources 

       * Chroma : high level structure
       * Geant4 : simulation details
       * Graphics : eg: textures for material/surface property lookup, instancing for big geom

   * curand split init issue
   
   * p33 ggeoview : http://localhost/env/presentation/optical_photon_simulation_with_nvidia_optix.html

   * following DYB meeting at IHEP, stayed a few extra days to work with Tao using 4 GPU, on early JUNO geometry  
   * instancing to handle large geometry like JUNO 
   * start development of analytic PMT geometry  
   
     * motivted by : g4 tesselation issues (fixed by mesh surgery) + PMT disco ball effect + need to match G4 

   * (october 2015)
   * http://localhost/env/presentation/optical_photon_simulation_progress.html
   * p24: rainbow


   * (jan 2016)
   * http://localhost/env/presentation/opticks_gpu_optical_photon_simulation_psroc.html
   * p12 : rinbow 

   * (mar 2016)
   * http://localhost/env/presentation/opticks_gpu_optical_photon_simulation_march2016.html
   * p7 : disco ball 

   * (april 2016, GTC) 
   * http://localhost/env/presentation/opticks_gpu_optical_photon_simulation_april2016_gtc.html

   * (oct 2016, CHEP) 
   * http://localhost/env/presentation/opticks_gpu_optical_photon_simulation_oct2016_chep.html
   * Aim : analytic on critical path, rest triangulated

   * (nov 2016, LLR)
   * http://localhost/env/presentation/opticks_gpu_optical_photon_simulation_nov2016_llr.html
   * https://bugzilla-geant4.kek.jp/show_bug.cgi?id=1275

   * (jan 2017, PSROC) 
   * http://localhost/env/presentation/opticks_gpu_optical_photon_simulation_jan2017_psroc.html
   * p23 : CSG expt 
   * p26 : DYB composite 

 3. (July 2017) Jump to fully analytic CSG : needed for near perfect G4 match  

   * (jul 2017, IHEP) 
   * http://localhost/env/presentation/opticks_gpu_optical_photon_simulation_jul2017_ihep.html
   * Auto translate for GPU 
   * CSG : bit twiddling 

     * looked for GPU impl : there were none so developed one myself starting from a CS paper 
     * emulating recursion
     * DYB CSG 

   * p16: SDF

   * (sep 2017, jinan)
   * http://localhost/env/presentation/opticks_gpu_optical_photon_simulation_sep2017_jinan.html
   * (sep 2017, uow)
   * http://localhost/env/presentation/opticks_gpu_optical_photon_simulation_sep2017_wollongong.html

   * (jul 2018, sof)
   * http://localhost/env/presentation/opticks_gpu_optical_photon_simulation_jul2018_chep.html

   * NVIDIA INTRODUCES RTX       

   * (sep 2018, qingdao) 
   * http://localhost/env/presentation/opticks_gpu_optical_photon_simulation_sep2018_qingdao.html
   * pmt torus neck 

   * (oct 2018, ihep)
   * http://localhost/env/presentation/opticks_gpu_optical_photon_simulation_oct2018_ihep.html
   * p11 parallel/simple/uncoupled 
   * p12 GPU constraints
   * p19 deciding history on way to boundary 
   * p29 curand : split init and use

   * (jan 2019, sjtu)
   * http://localhost/env/presentation/opticks_gpu_optical_photon_simulation_jan2019_sjtu.html
   * 5/40 JUNO solids with issues
   * profligate PMT modelling  
   * sAirTT CSG coincident face 


   * (jul 2019, ihep) 
   * http://localhost/env/presentation/opticks_gpu_optical_photon_simulation_jul2019_ihep.html
   * direct geometry workflow, not export/import 
   * p12 Optix 6.0 torus intersect issue
   * p12: Guide Tube Torus : removed (for now), AVOIDED : OptiX 6.0.0 NOT working with torus intersect
   * p12: PMT_20inch_body : simplified neck, FIXED : "cylinder - torus" -> polycone
   * p12: PMT_20inch_inner : simplified CSG modelling, FIXED : depth 4 tree (31 nodes) -> 1 primitive
   * p12: sAirTT : CSG modelling coincidence avoided, FIXED : "box - cylinder" : growing the subtracted 
   * p19 JUNO360 multiple-GPU benchmarking 
   * p36 simple test geom 1000x:


 4. Aug 2019 Pre-Pandemic BOMBSHELL : NVIDIA 6.5->7.0 (like Geant3->4 ) 

   * was busy with validation and optimization for RTX : when NV announced
   * NO WARNING : ALL NEW API : EFFECTIVELY HAVE TO START OVER 
   * DANGER OF DEPENDING ON CLOSED-CODE : BUT NO CHOICE THEN (OR YET) FOR HIGH PERF RAY-TRACE
   * NO TRANSPARENT MULTI-GPU 

   * HMM: I DIDNT IMMEDIATELY SWITCH TO DEV FOR 7 : PSYCHOLOGICALLY IMPOSSIBLE 
     TO DROP SO MANY YEARS OF WORK 

   * http://localhost/env/presentation/opticks_oct2019_dance.html   
   * with optix 7 need to develop multi-GPU load balancing 

   * http://localhost/env/presentation/opticks_nov2019_chep.html
 
   * (dec 2019, gtc, suzhou) 
   * http://localhost/env/presentation/opticks_gpu_optical_photon_simulation_dec2019_gtc_china_suzhou.html
   * NICE INTRO SLIDES 

   * (dec 2019, ihep epd/pifi seminar)
   * http://localhost/env/presentation/opticks_gpu_optical_photon_simulation_dec2019_ihep_epd_seminar.html

   * (may 2020, HSF)
   * http://localhost/env/presentation/opticks_may2020_hsf.html
   * p27 : Main operational problem : manpower
   * LOTS OF THOUGHTFUL SLIDES

   * (jul 2020, JUNO collab)
   * http://localhost/env/presentation/opticks_jul2020_juno.html
   * opticks junoenv integration
   * PMT shape simplification
   * --pmt20inch-polycone-neck 



   * (aug 2020)
   * http://localhost/env/presentation/opticks_aug2020_sjtu_neutrino_telescope_workshop.html
   * p44 : decades of CG research (milestones over 50 years) 

   * (jan 2021) 
   * http://localhost/env/presentation/opticks_jan2021_juno_sim_review.html
   * lots of "engagement" slides
   * because work over the past months distinctly non-interesting technical JUNOSW+Opticks  project integration work  

   * (feb 2021, first of series of 7 meetings)
   * http://localhost/env/presentation/lz_opticks_optix7_20210208.html
   * http://localhost/env/presentation/lz_opticks_optix7_20210225.html
   * HARD WORK OF MIGRATING TO 7  

   * (mar 2021, CAF)
   * http://localhost/env/presentation/opticks_detector_geometry_caf_mar2021.html

   * (mar 2021)
   * http://localhost/env/presentation/lz_opticks_optix7_20210315.html
   * GAS:BI:AABB 1NN issue elucidated

   * (apr 2021)
   * http://localhost/env/presentation/lz_opticks_optix7_20210406.html
   * FIRST MENTION OF THE CSGFoundry MODEL 

     * with OptiX 7, you have to BYO(GM) : bring-you-own-geometry-model    

   * (apr 2021)
   * http://localhost/env/presentation/juno_opticks_20210426.html
   * "bash junoenv opticks" 
   * mis-use + profligate use of G4Boolean
   * presention with one foot in old Opticks and one in new
   * p37 CSG boolean parade


   * (may 2021)
   * http://localhost/env/presentation/lz_opticks_optix7_20210504.html

   * GGeo -> CSGFoundry : was expedient, practicality is have to keep things working across transitions 
   * "LONGTERM POSSIBILITY : Populate CSGFoundry model direct from Geant4 geometry ? [Disruptive]"
   * started trying to keep pre-7 going with the new geometry model 

   * http://localhost/env/presentation/opticks_vchep_2021_may19.html
   * New "Foundry" Model : replaces pre-7 geometry context dropped in 6->7  
   * full geometry in GPU compatible form : simple serialization 
   * p12 : first OptiX 7 full JUNO raytrace
 
   * http://localhost/env/presentation/lz_opticks_optix7_20210518.html
   * p10 : missed repetitions
   * p13 : render in OptiX 5,6,7  

   * http://localhost/env/presentation/lz_opticks_optix7_20210518.html

   * http://localhost/env/presentation/juno_opticks_20210712.html
   * hard work of simulation matching 
   * G4Cerenkov_modified stale/undefined sin2Theta bug

   * http://localhost/env/presentation/lz_opticks_optix7_20210727.html
   * first mention of QUDARAP
   * Scint/Cerenkov matching
   * float/double Ck issue

   * http://localhost/env/presentation/juno_opticks_cerenkov_20210902.html
   * "For sanity : need to make the leap to OptiX 7 .."
   * s2 CK integration 
   * geometry detailed debug start
   * G4Cerenkov_modified GetAverageNumberOfPhotons_s2 ~2 photons diff from _asis for some BetaInverse due to poor split integral approx


   * http://localhost/env/presentation/opticks_autumn_20211019.html
   * CK inverse sampling  
   * p26 : Opticks updates for G4 1100
   * p29 : 2D sliced render technique
   * p36 : AdditionAcrylic pointless CSG hole subtraction, colocated sub-sub bug 
    


   * http://localhost/env/presentation/opticks_20211117.html
   * profligate Z-cut PMT : developed ZSolid solution : actually cut the tree  
   * p27 : spurious intersects from Geant4 torus neck  


   * http://localhost/env/presentation/opticks_20211223_pre_xmas.html
   * > 100x faster than times from July
   * mask tail cutting across PMT bulb ?
   * p19,20 demo fix for sub-sub bug with --additionacrylic-simplify-csg
   * p30-38 : MOI renders

   * http://localhost/env/presentation/opticks_20220115_innovation_in_hep_workshop_hongkong.html

   * (2022 jan)  
   * http://localhost/env/presentation/opticks_20220118_juno_collaboration_meeting.html
   * review of lots of geometry issues, interferences : RTP frame 

   * (2022 feb)
   * http://localhost/env/presentation/opticks_20220227_LHCbRich_UK_GPU_HACKATHON.html

   * (2022 mar)
   * http://localhost/env/presentation/opticks_20220307_fixed_global_leaf_placement_issue.html
   * complex solid reveals incompatibility of tree balancing and the CSG intersect alg 

   * http://localhost/env/presentation/opticks_20220329_progress_towards_production.html
   * mid-March : switch gears from geometry to physics
   * geometry changes can have big performance effects
   * dynamic prim selection


   * http://localhost/env/presentation/opticks_20220718_towards_production_use_juno_collab_meeting.html
   * COMPLETED : Full Simulation re-implementation for OptiX 7 API
   * systemtic random aligned sims


   * (2022 sep) 
   * http://localhost/env/presentation/opticks_202209XX_mask_spurious_debug.html
   * new geom -> CSG precision loss, spurious issues fixed
   * PMT overlap issues
   * first ART plots as look at PMT optical model 


   * (2022 nov)
   * http://localhost/env/presentation/opticks_20221117_mask_debug_and_tmm.html
   * p33: multi layer TMM  
   * p46: standlone test of single PMT with jPOM
   * p49: "Is fake Vacuum/Vacuum really needed ?"

   * (2022 dec)
   * http://localhost/env/presentation/opticks_20221220_junoPMTOpticalModel_FastSim_issues_and_CustomG4OpBoundaryProcess_fix.html
   * explain FastSim issues and custom fix


   * (2023 feb)
   * http://localhost/env/presentation/opticks_20230206_JUNO_PMT_Geometry_and_Optical_Model_Progress.html

   * (2023 apr)
   * http://localhost/env/presentation/opticks_20230428_More_junoPMTOpticalModel_issues_and_Validation_of_CustomG4OpBoundaryProcess_fix.html
   * Custom4 first mentioned 
   * more Fastsim bugs detailed  


   * (2023 may)
   * http://localhost/env/presentation/opticks_20230525_MR180_timestamp_analysis.html
   * apples-vs-oranges comparison

   * (2023 jun)
   * http://localhost/env/presentation/opticks_20230611_qingdao_sdu_workshop.html

   * (2023 jul)
   * http://localhost/env/presentation/opticks_20230726_kaiping_software_review.html

   * (2023 sep)
   * http://localhost/env/presentation/opticks_20230907_release.html
   * problem solids
   * huge code reduction
   * apex degenerate

   * (2023 oct)
   * http://localhost/env/presentation/opticks_20231027_nanjing_cepc_workshop.html

   * (2023 dec) 
   * http://localhost/env/presentation/opticks_20231211_profile.html
   * p8: chimney photons issue

 
   * http://localhost/env/presentation/opticks_20231219_using_junosw_plus_opticks_release.html
   * Using first release

   * (2024 feb)
   * http://localhost/env/presentation/opticks_20240224_offline_software_review.html
   * p2 : leak fixes






 

