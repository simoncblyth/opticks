OPTICKS_MAX_BOUNCE_scanning
==============================

Overview
----------

Currently its set at initialization and never changed. Can it 
be changed from event to event to facilitate scanning ? 

* not easy to change this within a run of SEvt in debug mode as
  array sizes vary with it ?

  * HMM: is that true, most buffers get set to maxes : so maybe 
    can change the max_bounce without issue ?

* whatever : decided to cxs_min_scan.sh via separate invokations 
  which set OPTICKS_START_INDEX 


Observation
------------

* with SRM_TORCH from CD center see linear correspondence
  between MAX_BOUNCE and launch time up until about MAX_BOUNCE 16 

  * makes sense : the more ray traces per photon the longer it takes
  * BUT NOTE THE TIME STILL GOING UP EVEN IN THE EXTREME TAIL 


DONE : histogram of bounce counts
----------------------------------

* add HitPhotonSeq event mode : at switch that on with VERSION 4  

::

    ~/opticks/cxs_min.sh        ## workstation
    ~/opticks/cxs_min.sh grab   ## laptop


Trying to add just seq giving all "TO"::

    In [8]: a.qtab
    Out[8]: array([[b'100000', b'0', b'TO                                                                                              ']], dtype='|S96')


::
 
    VERSION=4 ~/o/cxs_min.sh       ## workstation 
    VERSION=4 ~/o/cxs_min.sh grab  ## laptop

    VERSION=4 ~/o/cxs_min.sh ana   ## laptop



    VERSION=4 OPTICKS_NUM_PHOTON=M1 ~/o/cxs_min.sh 
    VERSION=4 OPTICKS_NUM_PHOTON=M1 ~/o/cxs_min.sh grab 


Related
---------

* :doc:`is_seq_buffer_32x_too_large`


Workflow
----------

Workstation::

    ./cxs_min_scan.sh  
    
Laptop::

    ./cxs_min.sh grab 

    PLOT=Substamp_ONE_maxb_scan ~/opticks/sreport.sh 

