how_to_validate_after_geometry_or_model_changes
=================================================


Validations at many different levels from the most basic unit tests
and standalone tests focussing on small parts of the code to integration
level tests making chi2 comparisons between the Opticks and Geant4 
simulations that depend on the full simulation. 


Unit tests, implemented using CMake/ctests
----------------------------------------------

::

    om-;om-test   ## single package tests 

    opticks-t     ## tests like the above for all active packages (2025/01/20 0/218 FAIL)


Expanded tests 
---------------

These tests expand on some the above ctests particularly for 
executables that encompass many tests selected via TEST envvar 
which are simpler to run and debug outside of ctests::

    ~/o/qudarap/tests/QSimTest_ALL.sh    ## QSimTest tests

       ## see notes/issues/QSimTest_ALL_initial_shakedown.rst  (2025/01/20 3/25 FAIL)

    ~/o/qudarap/tests/QEventTest_ALL.sh  ## QEventTest tests   (2023/01/20 0/6 FAIL)


Standalone tests of related repository
------------------------------------------

::

    ~/np/tests/go.sh    ## standalone NP tests (2025/01/20 8/57 FAIL)



~/o/cxs_min.sh (symbolic link to ~/o/CSGOptiX/cxs_min.sh)
-----------------------------------------------------------

This script is used for many purposes in development and testing 
making it require understanding and often editing before use,
plus the setting of the TEST envvar to select the type of 
test to perform.  

This script runs the CSGOptiXSMTest executable which has no Geant4 dependency,
so it is restricted to purely optical running and loads the persisted CSGFoundry
from ~/.opticks/GEOM/$GEOM/CSGFoundry using GEOM envvar 
set by ~/.opticks/GEOM/GEOM.sh 

This script is most commonly used for "torch" running where initial photons
are generated in simple patterns and with numbers of photons configured by the 
script. Input photons and input gensteps can also be configured. Small scans
simulating multiple events with varying numbers of photons can also be configured, 
often simply by selecting a TEST envvar value. 

::

    TEST=ref1 ~/o/cxs_min.sh

Example output directory::

    $TMP/GEOM/$GEOM/CSGOptiXSMTest/ALL${VERSION}_Debug_Philox_$TEST/A000/
    
The VERSION envvar controls the default opticks_event_mode which determines 
which arrays are saved for each event. 



~/o/G4CXTest_GEOM.sh (symbolic linked to ~/o/g4cx/tests/G4CXTest_GEOM.sh)
-----------------------------------------------------------------------------

This script does standalone optical only bi-simulation with G4CXApp::Main and the current GEOM
which is read from GDML and converted to Opticks.  

When the bi-simulations are configured to save the sseq.npy sequence histories it is 
then possible to compare those histories using a photon history frequency chi2 using::

    ~/o/G4CXTest_GEOM.sh chi2    

The chi2 sub-command uses the ~/o/sysrap/tests/sseq_index_test.sh script and a 
separate C++ executable to form and present the chi2, this replaces the old python "SAB" 
implementation of the same which was too slow. 

* ~/o/notes/issues/G4CXTest_GEOM_shakedown_2025.rst



