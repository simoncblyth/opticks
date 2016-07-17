Overview of Opticks Integration Tests 
======================================= 

Sourcing **opticks.bash** results in sourcing of the integration test
precursor bash functions.

.. code-block:: sh

    931 #### opticks top level tests ########
    932 
    933 tpmt-(){       . $(opticks-home)/tests/tpmt.bash     && tpmt-env $* ; }
    934 trainbow-(){   . $(opticks-home)/tests/trainbow.bash && trainbow-env $* ; }
    935 tnewton-(){    . $(opticks-home)/tests/tnewton.bash  && tnewton-env $* ; }
    936 tprism-(){     . $(opticks-home)/tests/tprism.bash   && tprism-env $* ; }
    937 tbox-(){       . $(opticks-home)/tests/tbox.bash     && tbox-env $* ; }
    938 treflect-(){   . $(opticks-home)/tests/treflect.bash && treflect-env $* ; }
    939 twhite-(){     . $(opticks-home)/tests/twhite.bash   && twhite-env $* ; }
    940 tlens-(){      . $(opticks-home)/tests/tlens.bash    && tlens-env $* ; }
    941 



    




