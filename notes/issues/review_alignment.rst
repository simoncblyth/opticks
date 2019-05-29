review_alignment
====================

State of play
---------------

* using input photons more advanced using "--align" option
* using input gensteps incomplete, only CerenkovGenerator
  and there are material consistency issues :doc:`ckm-okg4-material-rindex-mismatch`



OKG4Mgr alignment with input photons
---------------------------------------------

::

    102 /**
    103 OKG4Mgr::propagate_
    104 ---------------------
    105 
    106 Hmm propagate implies just photons to me, so this name 
    107 is misleading as it does a G4 beamOn with the hooked up 
    108 CSource subclass providing the primaries, which can be 
    109 photons but not necessarily. 
    110 
    111 Normally the G4 propagation is done first, because 
    112 gensteps eg from G4Gun can then be passed to Opticks.
    113 However with RNG-aligned testing using "--align" option
    114 which uses emitconfig CPU generated photons there is 
    115 no need to do G4 first. Actually it is more convenient
    116 for Opticks to go first in order to allow access to the ucf.py 
    117 parsed  kernel pindex log during lldb python scripted G4 debugging.
    118  
    119 Hmm it would be cleaner if m_gen was in charge if the branching 
    120 here as its kinda similar to initSourceCode.
    121 
    122 
    123 Notice the different genstep handling between this and OKMgr 
    124 because this has G4 available, so gensteps can come from the
    125 horses mouth.
    126 
    127 
    128 
    129 **/



ckm 
-----

* :doc:`OKG4Test_direct_two_executable_shakedown`

* :doc:`strategy_for_Cerenkov_Scintillation_alignment`

  * Apply CAlignEngine to CerenkovMinimal+G4Opticks 


* https://bitbucket.org/simoncblyth/opticks/commits/all?page=8


To find relevant commits::

    hg flog notes/issues/strategy_for_Cerenkov_Scintillation_alignment.rst

    hg flog cfg4/CAlignEngine.cc | grep rst


Stage have got to... 

* not wanting to instrument CerenkovMinimal with CFG4 complexity 
* but need the complexity for step by step comparison 



NEXT STEP : GET CFG4 TO RUN FROM GENSTEPS USING CCerenkovGenerator 


::

    blyth@localhost issues]$ l -1 *ckm*
    -rw-rw-r--. 1 blyth blyth 38363 May 29 16:59 ckm_cerenkov_generation_align_small_quantized_deviation_g4_g4.rst
    -rw-rw-r--. 1 blyth blyth 19015 May 28 21:53 ckm-analysis-shakedown.rst
    -rw-rw-r--. 1 blyth blyth  3903 May 27 23:09 ckm-viz-noshow-photon-first-steps.rst
    -rw-rw-r--. 1 blyth blyth  7445 Apr  1 18:38 ckm_revival_improve_error_handling_of_missing_rng_seq.rst
    -rw-rw-r--. 1 blyth blyth 15886 Mar 20 17:29 ckm-okg4-natural-fail.rst
    -rw-rw-r--. 1 blyth blyth  4960 Oct 15  2018 ckm_cerenkov_generation_align_g4_ok_deviations.rst
    -rw-rw-r--. 1 blyth blyth 24171 Oct 15  2018 ckm_cerenkov_generation_align.rst




