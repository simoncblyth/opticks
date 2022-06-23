U4RecorderTest_U4Stack_stag_enum_random_alignment
===================================================

Overview
---------

Opticks
   qsim.h stagr.h stag.h 
Geant4 
   U4Recorder U4Stack::Classify SBacktrace.h  

Machinery for enumerating and collecting random consumption records 
in both contexts is essentially complete following the ideas from prior. 

* :doc:`ideas_on_random_alignment_in_new_workflow`

Note that if there is a need to annotate the "simstreams" to help with
alignment : can do that simply by adding burns with suitable enum names. 

Need to apply the machinery to input_photons with a variety of
propagation histories to observe the consumption patterns
in order to decide how best to align. 



TODO : apply consumption enum collection machinery with storch_test.sh input photons
-----------------------------------------------------------------------------------------



TODO : observe how consumption changes when use cfg4/CPropLib reset tricks
----------------------------------------------------------------------------



DONE : checked storch_test.sh MOCK_CURAND input photons match on laptop and workstation
------------------------------------------------------------------------------------------

Confirmed perfect match with input photons generated on Linux workstation and Apple laptop::

    cd ~/opticks/sysrap/tests
    ./storch_test.sh       # remote  
    ./storch_test.sh       # local  
    ./storch_test.sh grab  # local  
    ./storch_test.sh cf  # local using sysrap/tests/storch_test_cf.py    





