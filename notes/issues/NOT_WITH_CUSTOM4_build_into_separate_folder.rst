NOT_WITH_CUSTOM4_build_into_separate_folder
=============================================


Cheat via different user
---------------------------

* HMM: could cheat and use a different user to do this 
* eg simon@P which used for fresh_build.sh release testing a month ago

  * then should be as simple as commenting one line and running fresh_build.sh 


fresh_build.sh on simon@P : takes  5min 
-----------------------------------------------

* opticks-t had one SEQPATH fail from SEvtTest, now fixed
* have to flip between build and usage in .bashrc


cxs_min.sh scan
----------------

::

    ~/o/cxs_min.sh    ## 20 event TORCH scan 0.1 to 100M 


Keep cxs_min.sh outputs::

    [simon@localhost J23_1_0_rc3_ok0]$ pwd
    /data/simon/opticks/GEOM/J23_1_0_rc3_ok0
    [simon@localhost J23_1_0_rc3_ok0]$ mv CSGOptiXSMTest CSGOptiXSMTest_WITH_CUSTOM4


HMM : IS THERE GOING TO BE A CRASH ON HITTING PMT NOT WITH_CUSTOM4 : THE GEOMETRY HAS SPECIAL SURFACES
----------------------------------------------------------------------------------------------------------

NO CRASH

* TODO: rerun with seq to compare histories
* TODO: get switch metadata and t_Launch to appear in sreport 




Directory control
-------------------

HMM : because N is using the junosw environment 
and junosw uses Custom4 its not so simple

* HMM : TEMPTING TO SPLIT CUSTOM4 SOMEHOW TO AVOID THIS


::

    epsilon:opticks blyth$ echo $CMAKE_PREFIX_PATH | tr ":" "\n"
    /usr/local/opticks_externals/custom4/0.1.9
    /usr/local/opticks_externals/g4_1042
    /usr/local/opticks_externals/clhep
    /usr/local/opticks_externals/xercesc
    /usr/local/opticks_externals/boost
    /usr/local/opticks
    /usr/local/opticks/externals
    /usr/local/optix
    epsilon:opticks blyth$ 

