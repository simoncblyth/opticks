fix : Brief Summaries of Recent Fixes/Additions 
==================================================

* G4OpticksRecorder/CRecorder machinery reworked to operate with dynamic(genstep-by-genstep) running by expanding 
  output arrays at each BeginOfGenstep  

* CPhotonInfo overhaul for cross reemission generation recording 

* suppress PMT Pyrex G4 0.001 mm "microsteps" 

* handle RINDEX-NoRINDEX "ImplicitSurface" transitions like Water->Tyvek by adding corresponding Opticks perfect absorber surfaces

* handle Geant4 special casing of material with name "Water" that has RINDEX property but lacks RAYLEIGH
  by grabbing the calulated RAYLEIGH and adding to material (Geant4 only changes G4OpRayleigh process physics table) 
  see X4MaterialWater

* suppress degenerate Pyrex///Pyrex +0.001mm boundary in GPU geometry that causes material inconsistencies 

* avoid 3BT/4BT discrepancy by skipping the sticks 




microStep mentions
--------------------

* :doc:`CRecorder_duplicated_point_BTBT`
* :doc:`check_innerwater_bulk_absorb`
* :doc:`ok_lacks_SI-4BT-SD`
* :doc:`ok_less_SA_more_AB`
* :doc:`tds3gun_nonaligned_comparison`
* :doc:`tds3ip_InwardsCubeCorners17699_paired_zeros_in_tail_G4:SR_BT_BT_SA_OK:SR_BT_SA`



recent iterations general ordering of fixes
----------------------------------------------



* :doc:`ok_less_SA_more_AB`  

  * confirmed due to degenerate -0.001mm Pyrex///Pyrex boundary

* :doc:`CRecorder_duplicated_point_BTBT` 

  * suppress G4 microStep, dx,bn buffers for looking into it    

* :doc:`tds3gun_nonaligned_comparison`  

  * FIXED OK not absorbing on Tyvek tds3gun_nonaligned_comparison 

* :doc:`GSurfaceLib__addImplicitBorderSurface_RINDEX_NoRINDEX`

  fixed X4PhysicalVolume::addBoundary surface finding to look at Opticks surfaces 
  not the G4 ones to find the added implicits  

* :doc:`ok_lacks_SI-4BT-SD`

* :doc:`ok_less_reemission`

  * wildcard selections show 15% less reemission in OK 

* :doc:`raw_scintillator_material_props`

  * 


* :doc:`tds3ip_InwardsCubeCorners17699_at_7_wavelengths`

  * compare first slot splits at 7 wavelengths : using the old coarse domain props
  * switched to 1 nm fine domain default using G4 Value interpolation



* :doc:`tds3ip_InwardsCubeCorners17699_paired_zeros_in_tail_G4:SR_BT_BT_SA_OK:SR_BT_SA`



recent issues
---------------


::

    epsilon:issues blyth$ ls -lt
    total 22528
    -rw-r--r--   1 blyth  staff   23689 Jul  1 10:43 CRecorder_duplicated_point_BTBT.rst
    -rw-r--r--   1 blyth  staff    1506 Jul  1 10:38 fix.rst
    -rw-r--r--   1 blyth  staff   41021 Jul  1 10:38 tds3ip_InwardsCubeCorners17699_paired_zeros_in_tail_G4:SR_BT_BT_SA_OK:SR_BT_SA.rst
    -rw-r--r--   1 blyth  staff   85562 Jul  1 10:34 check_innerwater_bulk_absorb.rst
    -rw-r--r--   1 blyth  staff    9190 Jun 29 21:42 tds3gun_OK_7pc_more_SI_AB.rst
    -rw-r--r--   1 blyth  staff    1827 Jun 29 16:18 geocache_inconsistency_between_machines.rst
    -rw-r--r--   1 blyth  staff   34703 Jun 29 13:08 tds3ip_InwardsCubeCorners17699_at_7_wavelengths.rst
    -rw-r--r--   1 blyth  staff   18526 Jun 29 11:20 OK_lacking_SD_SA_following_prop_shift_FIXED.rst
    -rw-r--r--   1 blyth  staff    4143 Jun 29 11:19 tds3ip_OK_pflags_mismatch_warning.rst
    -rw-r--r--   1 blyth  staff    4920 Jun 28 12:55 raw_scintillator_material_props.rst
    -rw-r--r--   1 blyth  staff   17629 Jun 28 10:48 opticks_t_44_of_486_after_switch_to_double_props.rst
    -rw-r--r--   1 blyth  staff   82992 Jun 27 11:47 ok_less_reemission.rst
    -rw-r--r--   1 blyth  staff   92357 Jun 25 20:22 ok_lacks_SI-4BT-SD.rst
    -rw-r--r--   1 blyth  staff   55625 Jun 24 17:54 ok_less_SA_more_AB.rst
    -rw-r--r--   1 blyth  staff   41268 Jun 23 16:16 CRecorder_record_id_ni_assert_CAUSED_BY_DsG4Scintillation_INSTRUMENTATION_REMOVED.rst
    -rw-r--r--   1 blyth  staff    9615 Jun 21 20:09 device_side_genstep_overhaul.rst
    -rw-r--r--   1 blyth  staff    2360 Jun 19 19:36 cuda-centric-new-DsG4Scintillation-genstep-with-mocking.rst
    -rw-r--r--   1 blyth  staff    3657 Jun 18 22:46 opticks-t-fails-jun-18-2021.rst
    -rw-r--r--   1 blyth  staff   16280 Jun 18 12:02 tds3ip_pflags_inconsistency.rst
    -rw-r--r--   1 blyth  staff      69 Jun 17 17:40 tds3gun_ab_mat_zeros.rst
    -rw-r--r--   1 blyth  staff   64404 Jun 16 12:04 tds3gun_nonaligned_comparison.rst
    -rw-r--r--   1 blyth  staff   22769 Jun 15 21:29 pflags_ana_BT_SD_SI_zero.rst
    -rw-r--r--   1 blyth  staff   10888 Jun 15 14:10 GSurfaceLib__addImplicitBorderSurface_RINDEX_NoRINDEX.rst
    -rw-r--r--   1 blyth  staff   20794 Jun 11 11:17 reemission_review.rst
    -rw-r--r--   1 blyth  staff   30215 Jun 10 17:47 analysis_shakedown.rst
    -rw-r--r--   1 blyth  staff    2608 Jun  9 17:56 G4Opticks_why_sudden_switch_to_pro_embedded_commandline.rst
    -rw-r--r--   1 blyth  staff     532 Jun  8 14:31 feasibility_of_cpu_running_opticks_propagation_for_testing_purposes.rst
    -rw-r--r--   1 blyth  staff     226 Jun  8 14:05 aligned_reemission_how_feasible.rst
    -rw-r--r--   1 blyth  staff    5190 Jun  8 13:16 alignment_kludge_simplification.rst
    -rw-r--r--   1 blyth  staff    2771 Jun  8 13:13 alignment_notes_index.rst
    -rw-r--r--   1 blyth  staff     374 Jun  8 10:43 rng_review.rst
    -rw-r--r--   1 blyth  staff    3587 Jun  7 22:26 gracious_no_gpu_handling.rst
    -rw-r--r--   1 blyth  staff    8950 Jun  7 12:31 output_event_directory_control_by_envvar.rst
    -rw-r--r--   1 blyth  staff   10475 Jun  7 11:20 set_input_photons_assert.rst
    -rw-r--r--   1 blyth  staff    2712 Jun  4 19:04 tboolean_box_fail.rst
    -rw-r--r--   1 blyth  staff   15811 Jun  4 18:49 opticks_t_1_jun_2021_12_of_471_fails.rst
    -rw-r--r--   1 blyth  staff   23314 May 24 11:27 CSGOptiXGGeo_9_TT_y_shift_transforms_not_applied.rst
    -rw-r--r--   1 blyth  staff    5990 May 23 20:13 G4OpticksRecorder_shakedown.rst
    -rw-r--r--   1 blyth  staff     238 May 14 20:35 Six-with-emm-t8,-hangs.rst
    -rw-r--r--   1 blyth  staff     893 May 12 22:51 optix7-csg-transform-issue.rst
    -rw-r--r--   1 blyth  staff   17695 Apr 26 16:19 opticks-t-april-26-test-fails-4-of-462.rst
    -rw-r--r--   1 blyth  staff   50534 Apr 25 00:55 OpSnapTest_debug_slowdown_with_new_geometry.rst
    -rw-r--r--   1 blyth  staff   11317 Apr 24 20:35 scan-revival-with-new-juno-geometry.rst
    -rw-r--r--   1 blyth  staff    8849 Apr 23 21:47 naming_compound_solids.rst
    -rw-r--r--   1 blyth  staff   38175 Apr 23 21:03 skipping_solids_by_name.rst
    -rw-r--r--   1 blyth  staff    6700 Apr 16 19:42 OpSnapTest_STimes_perplexing.rst
    -rw-r--r--   1 blyth  staff    2900 Apr 16 19:40 enabledmergedmesh-not-working-anymore.rst
    -rw-r--r--   1 blyth  staff    5869 Apr 16 15:12 GParts_ordering_difference_on_different_machine.rst

