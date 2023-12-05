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


* whatever decided to cxs_min_scan.sh via separate invokations 
  which set OPTICKS_START_INDEX 


Observation
------------

* with SRM_TORCH from CD center see linear correspondence
  between MAX_BOUNCE and launch time up until about MAX_BOUNCE 16 

  * makes sense : the more ray traces per photon the longer it takes
  * BUT NOTE THE TIME STILL GOING UP EVEN IN THE EXTREME TAIL 



WIP : histogram of bounce counts
----------------------------------

* add HitPhotonSeq event mode : at switch that on with VERSION 4  

::

    ~/opticks/cxs_min.sh        ## workstation
    ~/opticks/cxs_min.sh grab   ## laptop


Trying to add just seq giving all "TO"::

    In [8]: a.qtab
    Out[8]: array([[b'100000', b'0', b'TO                                                                                              ']], dtype='|S96')


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



Does max_bounce change record size ?  YES 
-------------------------------------------

::

    epsilon:sysrap blyth$ opticks-f max_bounce
    ./CSGOptiX/CSGOptiX7.cu:    while( bounce < evt->max_bounce )
    ./sysrap/SEventConfig.hh:    static void SetMaxBounce( int max_bounce); 
    ./sysrap/SEventConfig.cc:void SEventConfig::SetMaxBounce( int max_bounce){  _MaxBounce  = max_bounce  ; Check() ; }
    ./sysrap/SEventConfig.cc:    int max_bounce = MaxBounce(); 
    ./sysrap/SEventConfig.cc:        SEventConfig::SetMaxRecord(max_bounce+1); 
    ./sysrap/SEventConfig.cc:        SEventConfig::SetMaxRec(max_bounce+1); 
    ./sysrap/SEventConfig.cc:        SEventConfig::SetMaxSeq(max_bounce+1); 
    ./sysrap/SEventConfig.cc:        SEventConfig::SetMaxPrd(max_bounce+1); 
    ./sysrap/SEventConfig.cc:        SEventConfig::SetMaxAux(max_bounce+1); 
    ./sysrap/tests/SEvtTest.cc:    unsigned max_bounce = 9 ; 
    ./sysrap/tests/SEvtTest.cc:    SEventConfig::SetMaxBounce(max_bounce); 
    ./sysrap/tests/SEvtTest.cc:    SEventConfig::SetMaxRecord(max_bounce+1); 
    ./sysrap/tests/SEvtTest.cc:    SEventConfig::SetMaxRec(max_bounce+1); 
    ./sysrap/tests/SEvtTest.cc:    SEventConfig::SetMaxSeq(max_bounce+1); 
    ./sysrap/sevent.h:    reads evt.seed evt.genstep evt.max_bounce
    ./sysrap/sevent.h:    int      max_bounce  ; // eg: 0:32  (not including 32)
    ./sysrap/sevent.h:    max_bounce   = SEventConfig::MaxBounce()  ; 
    ./sysrap/sevent.h:        << " evt.max_bounce    " << std::setw(w) << max_bounce   << std::endl 
    ./sysrap/sevent.h:   cfg.q0.u.z = max_bounce ; 
    ./sysrap/SEvt.cc:    setMeta<int>("MaxBounce", evt->max_bounce ); 
    ./sysrap/SEvt.cc:TODO: truncation : bounce < max_bounce 
    ./qudarap/qsim.h:    while( bounce < evt->max_bounce )
    ./qudarap/qsim.h:        ctx.prd = mock_prd + (evt->max_bounce*idx+bounce) ;  
    ./qudarap/qsim.h:        printf("//qsim.mock_propagate idx %d bounce %d evt.max_bounce %d prd.q0.f.xyzw (%10.4f %10.4f %10.4f %10.4f) \n", 
    ./qudarap/qsim.h:             idx, bounce, evt->max_bounce, ctx.prd->q0.f.x, ctx.prd->q0.f.y, ctx.prd->q0.f.z, ctx.prd->q0.f.w );  
    epsilon:opticks blyth$ 



    2084 void SEvt::setNumPhoton(unsigned num_photon)
    2085 {
    2086     //LOG_IF(info, LIFECYCLE) << id() << " num_photon " << num_photon ; 
    2087     bool num_photon_allowed = int(num_photon) <= evt->max_photon ;
    2088     const int M = 1000000 ;
    2089 
    2090     LOG_IF(fatal, !num_photon_allowed) << " num_photon/M " << num_photon/M << " evt.max_photon/M " << evt->max_photon/M ;
    2091     assert( num_photon_allowed );
    2092 
    2093     evt->num_photon = num_photon ;
    2094     evt->num_seq    = evt->max_seq   > 0 ? evt->num_photon : 0 ;
    2095     evt->num_tag    = evt->max_tag  == 1 ? evt->num_photon : 0 ;
    2096     evt->num_flat   = evt->max_flat == 1 ? evt->num_photon : 0 ;
    2097     evt->num_sup    = evt->max_sup   > 0 ? evt->num_photon : 0 ;
    2098 
    2099     evt->num_record = evt->max_record * evt->num_photon ;
    2100     evt->num_rec    = evt->max_rec    * evt->num_photon ;
    2101     evt->num_aux    = evt->max_aux    * evt->num_photon ;
    2102     evt->num_prd    = evt->max_prd    * evt->num_photon ;


Find
-------

::

    epsilon:qudarap blyth$ opticks-f max_bounce
    ./CSGOptiX/CSGOptiX7.cu:    while( bounce < evt->max_bounce )
    ./sysrap/SEventConfig.hh:    static void SetMaxBounce( int max_bounce); 
    ./sysrap/SEventConfig.cc:void SEventConfig::SetMaxBounce( int max_bounce){  _MaxBounce  = max_bounce  ; Check() ; }
    ./sysrap/tests/SEvtTest.cc:    unsigned max_bounce = 9 ; 
    ./sysrap/tests/SEvtTest.cc:    SEventConfig::SetMaxBounce(max_bounce); 
    ./sysrap/tests/SEvtTest.cc:    SEventConfig::SetMaxRecord(max_bounce+1); 
    ./sysrap/tests/SEvtTest.cc:    SEventConfig::SetMaxRec(max_bounce+1); 
    ./sysrap/tests/SEvtTest.cc:    SEventConfig::SetMaxSeq(max_bounce+1); 
    ./sysrap/sevent.h:    reads evt.seed evt.genstep evt.max_bounce
    ./sysrap/sevent.h:    int      max_bounce  ; // eg:  9 
    ./sysrap/sevent.h:    max_bounce   = SEventConfig::MaxBounce()  ; 
    ./sysrap/sevent.h:        << " evt.max_bounce    " << std::setw(w) << max_bounce   << std::endl 
    ./sysrap/sevent.h:   cfg.q0.u.z = max_bounce ; 
    ./sysrap/SEvt.cc:    setMeta<int>("MaxBounce", evt->max_bounce ); 
    ./sysrap/SEvt.cc:TODO: truncation : bounce < max_bounce 
    ./qudarap/qsim.h:    while( bounce < evt->max_bounce )
    ./qudarap/qsim.h:        ctx.prd = mock_prd + (evt->max_bounce*idx+bounce) ;  
    ./qudarap/qsim.h:        printf("//qsim.mock_propagate idx %d bounce %d evt.max_bounce %d prd.q0.f.xyzw (%10.4f %10.4f %10.4f %10.4f) \n", 
    ./qudarap/qsim.h:             idx, bounce, evt->max_bounce, ctx.prd->q0.f.x, ctx.prd->q0.f.y, ctx.prd->q0.f.z, ctx.prd->q0.f.w );  
    epsilon:opticks blyth$ 



