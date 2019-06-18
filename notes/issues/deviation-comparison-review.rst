deviation-comparison-review
================================


context
---------


Proxy scanning bi-simulates photons incident on
each solid of a geometry individually.

Non aligned comparison

* :doc:`tboolean-proxy-scan`




review notes
-----------------------------


::

    [blyth@localhost issues]$ grep -l reflectcheat *.rst
    emitconfig-cfg4-chisq-too-good-as-not-indep-samples.rst
    OKG4Test_no_OK_hits_again.rst
    photon-polarization-testauto-SR.rst
    random_alignment.rst
    sc_ab_re_alignment.rst

    ls -1 *align*.rst
    alignment_kludge_simplification.rst
    BR_PhysicalStep_zero_misalignment.rst
    ckm_cerenkov_generation_align_g4_ok_deviations.rst
    ckm_cerenkov_generation_align.rst
    ckm_cerenkov_generation_align_small_quantized_deviation_g4_g4.rst
    emitconfig-aligned-comparison.rst
    quartic_solve_optix_600_misaligned_address_exception.rst
    random_alignment_iterating.rst
    random_alignment.rst
    review_alignment.rst
    revive_aligned_running.rst
    rng_aligned_cerenkov_generation.rst
    sc_ab_re_alignment.rst
    strategy_for_Cerenkov_Scintillation_alignment.rst
    tboolean_box_perfect_alignment.rst
    tboolean_box_perfect_alignment_small_deviations.rst




* :doc:`random_alignment`

  * deep into aligned comparison, in the guts of G4 

* :doc:`random_alignment_iterating`

  * instructions on masked running, ie reproducible running of chosen single photons  
  * g4lldb.py dumping
  * tboolean-box-ip outputs in aligned running  
  * Iteration Approach 1 : Directly select/dump non-history aligned records : maligned dumpline approach 
  * "TO AB" "TO BT AB" looks to be trying to do the same thing : velocity bug again perhaps ? NOPE log(double(u))

* :doc:`where_mask_running`

  * how to fast forward to debug single photon 
  * discusses how this was implemented 

