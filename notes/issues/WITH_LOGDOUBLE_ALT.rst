WITH_LOGDOUBLE_ALT
=======================






optixrap/cu/propagate.h::

     58 __device__ int propagate_to_boundary( Photon& p, State& s, curandState &rng)
     59 {
     60     //float speed = SPEED_OF_LIGHT/s.material1.x ;    // .x:refractive_index    (phase velocity of light in medium)
     61     float speed = s.m1group2.x ;  // .x:group_velocity  (group velocity of light in the material) see: opticks-find GROUPVEL
     62 
     63 #ifdef WITH_ALIGN_DEV
     64 #ifdef WITH_LOGDOUBLE
     65 
     66     float u_boundary_burn = curand_uniform(&rng) ;
     67     float u_scattering = curand_uniform(&rng) ;
     68     float u_absorption = curand_uniform(&rng) ;
     69 
     70     //  these two doubles brings about 100 lines of PTX with .f64
     71     //  see notes/issues/AB_SC_Position_Time_mismatch.rst      
     72     float scattering_distance = -s.material1.z*log(double(u_scattering)) ;   // .z:scattering_length
     73     float absorption_distance = -s.material1.y*log(double(u_absorption)) ;   // .y:absorption_length 
     74 
     75 #elif WITH_LOGDOUBLE_ALT
     76     float u_boundary_burn = curand_uniform(&rng) ;
     77     double u_scattering = curand_uniform_double(&rng) ;
     78     double u_absorption = curand_uniform_double(&rng) ;
     79 
     80     float scattering_distance = -s.material1.z*log(u_scattering) ;   // .z:scattering_length
     81     float absorption_distance = -s.material1.y*log(u_absorption) ;   // .y:absorption_length 
     82 
     83 #else
     84     float u_boundary_burn = curand_uniform(&rng) ;
     85     float u_scattering = curand_uniform(&rng) ;
     86     float u_absorption = curand_uniform(&rng) ;
     87     float scattering_distance = -s.material1.z*logf(u_scattering) ;   // .z:scattering_length
     88     float absorption_distance = -s.material1.y*logf(u_absorption) ;   // .y:absorption_length 
     89 #endif
     90 
     91 #else
     92     float scattering_distance = -s.material1.z*logf(curand_uniform(&rng));   // .z:scattering_length
     93     float absorption_distance = -s.material1.y*logf(curand_uniform(&rng));   // .y:absorption_length
     94 #endif
     95 



LV 39 (WITH_LOGDOUBLE)::


    absmry.py line 
    39 _FATAL_ 0x05   1M      2.991 29906    0.0195   123.0  6298.2       _FATAL_  7200.1499     1   WARNING     0.0079     0   _FATAL_     3.0234   782      sWorld0x4bc2350



    ta 39

    AB(1,torch,tboolean-proxy-39)  None 0     file_photons 1M   load_slice :1M:   loaded_photons 1M  
    ab.rpost_dv
    maxdvmax:7200.1499  ndvp:   1  level:FATAL  RC:1       skip:
                     :                                :                   :                       :     1     1     1 : 6.0426 8.7892 11.5359 :                                    
      idx        msg :                            sel :    lcu1     lcu2  :       nitem     nelem :  nwar  nerr  nfat :   fwar   ferr   ffat :        mx        mn       avg      
     0000            :                    TO BT BT SA :  544722   544717  :      544707   8715312 :     0     0     0 : 0.0000 0.0000 0.0000 :    5.4934    0.0000    0.0000   :                 INFO :  
     0001            :                       TO BT AB :   88662    88667  :       88658   1063896 :     0     0     0 : 0.0000 0.0000 0.0000 :    5.4934    0.0000    0.0002   :                 INFO :  
     0002            :                       TO SC SA :   74400    74405  :       74397    892764 :     0     0     0 : 0.0000 0.0000 0.0000 :    5.4934    0.0000    0.0029   :                 INFO :  
     0003            :                       TO BR SA :   48826    48830  :       48826    585912 :     0     0     0 : 0.0000 0.0000 0.0000 :    5.4934    0.0000    0.0000   :                 INFO :  
     0004            :                 TO BT BT SC SA :   45319    45319  :       45318    906360 :     0     0     0 : 0.0000 0.0000 0.0000 :    5.4934    0.0000    0.0018   :                 INFO :  
     0005            :                 TO BT SC BT SA :   31621    31620  :       31616    632320 :     0     0     0 : 0.0000 0.0000 0.0000 :    5.4934    0.0000    0.0041   :                 INFO :  
     0006            :                 TO BT BR BT SA :   26467    26466  :       26464    529280 :     0     0     0 : 0.0000 0.0000 0.0000 :    5.4934    0.0000    0.0000   :                 INFO :  
     0007            :  TO BT SC BR BR BR BR BR BR BR :   13423    13503  :        8960    358400 :     1     1     1 : 0.0000 0.0000 0.0000 : 7200.1499    0.0000    0.0316   :                FATAL :   > dvmax[2] 11.5359  
     0008            :                    TO SC SC SA :    9562     9562  :        9561    152976 :     0     0     0 : 0.0000 0.0000 0.0000 :    5.4934    0.0000    0.0052   :                 INFO :  
     0009            :                 TO SC BT BT SA :    6231     6229  :        6229    124580 :     0     0     0 : 0.0000 0.0000 0.0000 :    5.4934    0.0000    0.0027   :                 INFO :  
    .
    ab.rpol_dv
    maxdvmax:0.0079  ndvp:   0  level:WARNING  RC:0       skip:
                     :                                :                   :                       :     4     0     0 : 0.0078 0.0118 0.0157 :                                    
      idx        msg :                            sel :    lcu1     lcu2  :       nitem     nelem :  nwar  nerr  nfat :   fwar   ferr   ffat :        mx        mn       avg      
     0000            :                    TO BT BT SA :  544722   544717  :      544707   6536484 :     0     0     0 : 0.0000 0.0000 0.0000 :    0.0000    0.0000    0.0000   :                 INFO :  
     0001            :                       TO BT AB :   88662    88667  :       88658    797922 :     0     0     0 : 0.0000 0.0000 0.0000 :    0.0000    0.0000    0.0000   :                 INFO :  
     0002            :                       TO SC SA :   74400    74405  :       74397    669573 :     0     0     0 : 0.0000 0.0000 0.0000 :    0.0000    0.0000    0.0000   :                 INFO :  
     0003            :                       TO BR SA :   48826    48830  :       48826    439434 :     0     0     0 : 0.0000 0.0000 0.0000 :    0.0000    0.0000    0.0000   :                 INFO :  
     0004            :                 TO BT BT SC SA :   45319    45319  :       45318    679770 :     0     0     0 : 0.0000 0.0000 0.0000 :    0.0000    0.0000    0.0000   :                 INFO :  
     0005            :                 TO BT SC BT SA :   31621    31620  :       31616    474240 :     1     0     0 : 0.0000 0.0000 0.0000 :    0.0079    0.0000    0.0000   :              WARNING :   > dvmax[0] 0.0078  
     0006            :                 TO BT BR BT SA :   26467    26466  :       26464    396960 :     0     0     0 : 0.0000 0.0000 0.0000 :    0.0000    0.0000    0.0000   :                 INFO :  
     0007            :  TO BT SC BR BR BR BR BR BR BR :   13423    13503  :        8960    268800 :     0     0     0 : 0.0000 0.0000 0.0000 :    0.0000    0.0000    0.0000   :                 INFO :  
     0008            :                    TO SC SC SA :    9562     9562  :        9561    114732 :     3     0     0 : 0.0000 0.0000 0.0000 :    0.0079    0.0000    0.0000   :              WARNING :   > dvmax[0] 0.0078  
     0009            :                 TO SC BT BT SA :    6231     6229  :        6229     93435 :     0     0     0 : 0.0000 0.0000 0.0000 :    0.0000    0.0000    0.0000   :                 INFO :  
    .
    ab.ox_dv
    maxdvmax:3.0234  ndvp: 782  level:FATAL  RC:1       skip:
                     :                                :                   :                       : 10168   782   117 : 0.1000 0.2500 0.5000 :                                    
      idx        msg :                            sel :    lcu1     lcu2  :       nitem     nelem :  nwar  nerr  nfat :   fwar   ferr   ffat :        mx        mn       avg      
     0000            :                    TO BT BT SA :  544722   544717  :      544707  26145936 :     0     0     0 : 0.0000 0.0000 0.0000 :    0.0000    0.0000    0.0000   :                 INFO :  
     0001            :                       TO BT AB :   88662    88667  :       88658   3191688 :     0     0     0 : 0.0000 0.0000 0.0000 :    0.0039    0.0000    0.0000   :                 INFO :  
     0002            :                       TO SC SA :   74400    74405  :       74397   2678292 :  1062     3     2 : 0.0004 0.0000 0.0000 :    0.9219    0.0000    0.0007   :                FATAL :   > dvmax[2] 0.5000  
     0003            :                       TO BR SA :   48826    48830  :       48826   1757736 :     0     0     0 : 0.0000 0.0000 0.0000 :    0.0000    0.0000    0.0000   :                 INFO :  
     0004            :                 TO BT BT SC SA :   45319    45319  :       45318   2719080 :   666     5     1 : 0.0002 0.0000 0.0000 :    0.9219    0.0000    0.0004   :                FATAL :   > dvmax[2] 0.5000  
     0005            :                 TO BT SC BT SA :   31621    31620  :       31616   1896960 :  4829   443    52 : 0.0025 0.0002 0.0000 :    3.0234    0.0000    0.0009   :                FATAL :   > dvmax[2] 0.5000  
     0006            :                 TO BT BR BT SA :   26467    26466  :       26464   1587840 :     0     0     0 : 0.0000 0.0000 0.0000 :    0.0002    0.0002    0.0000   :                 INFO :  
     0007            :  TO BT SC BR BR BR BR BR BR BR :   13423    13503  :        8960   1075200 :  2903   290    57 : 0.0027 0.0003 0.0001 :    1.3574    0.0000    0.0008   :                FATAL :   > dvmax[2] 0.5000  
     0008            :                    TO SC SC SA :    9562     9562  :        9561    458928 :   697    41     5 : 0.0015 0.0001 0.0000 :    1.0312    0.0000    0.0009   :                FATAL :   > dvmax[2] 0.5000  
     0009            :                 TO SC BT BT SA :    6231     6229  :        6229    373740 :    11     0     0 : 0.0000 0.0000 0.0000 :    0.1719    0.0000    0.0003   :              WARNING :   > dvmax[0] 0.1000  
    .
    AB(1,torch,tboolean-proxy-39)  None 0     file_photons 1M   load_slice :1M:   loaded_photons 1M  
    RC 0x05
    ab.cfm
    nph: 1000000 A:    0.0195 B:    0.0000 B/A:       0.0 COMPUTE_MODE compute_requested  ALIGN non-reflectcheat non-utaildebug 
    ab.a.metadata:/home/blyth/local/opticks/tmp/tboolean-proxy-39/evt/tboolean-proxy-39/torch/1 ox:2b68ca92e0ed0ca176606562ff8f1340 rx:3749dfb4b374708d7a98dd7bdf3ebdcd np:1000000 pr:    0.0195 COMPUTE_MODE compute_requested 
    ab.b.metadata:/home/blyth/local/opticks/tmp/tboolean-proxy-39/evt/tboolean-proxy-39/torch/-1 ox:f883d2a7c3a7554e80940b54f819c4fe rx:78c4ae065df4f4777f00f90ef4b9acad np:1000000 pr:    0.0000 COMPUTE_MODE compute_requested 
    WITH_SEED_BUFFER WITH_RECORD WITH_SOURCE WITH_ALIGN_DEV WITH_ALIGN_DEV_DEBUG WITH_LOGDOUBLE WITH_KLUDGE_FLAT_ZERO_NOPEEK 
    {u'containerscale': 3.0, u'container': 1, u'ctrl': 0, u'verbosity': 0, u'containerautosize': 1, u'jsonLoadPath': u'/home/blyth/local/opticks/tmp/tboolean-proxy-39/0/meta.json', u'poly': u'IM', u'emitconfig': u'photons:10000,wavelength:380,time:0.0,posdelta:0.1,sheetmask:0x0,umin:0.35,umax:0.65,vmin:0.35,vmax:0.65', u'resolution': 20, u'emit': -1}
    .
    [2019-07-19 20:07:45,966] p457913 {<module>            :tboolean.py:38} CRITICAL -  RC 0x05 0b101 
    [2019-07-19 20:07:45,969] p457913 {check_utaildebug    :ab.py     :198} INFO     - requires both A and B to have been run with --utaildebug option




::

    ts 39 --generateoverride -1    ## --xanalytic now ON by default, adjusted sheetmask in tboolean-proxy to 0x1




