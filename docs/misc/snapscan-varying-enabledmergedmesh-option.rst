snapscan-varying-enabledmergedmesh-option
============================================

To identify performance problems with geometry it is useful to 
make render snapshots that vary the included geometry. See:

* https://simoncblyth.bitbucket.io/env/presentation/juno_opticks_20210426.html

This page describes how the renders and tables were created using 
Opticks executables and scripts.

On GPU Workstation::
 
    snapscan.sh --cvd 1 --rtx 1    ## runs snap.sh with sequence of -e options saving jpg/json/npy

CAUTION bin/snapscan.sh gets installed, so to update it::

    cd ~/opticks/bin
    om 

On laptop::

    snap.sh    ## rsyncs jpg,npy,json outputs for each snap from workstation to laptop
        
snap.py analysis script reads the json and presents a table sorted by render speed::

    epsilon:~ blyth$ snap.py  ## reads metadata and creates table
    idx         -e    time(s)   relative     enabled geometry description                                          
      0         5,     0.0020     0.1140     ONLY: 1:sStrutBallhead0x34be280                                       
      1         6,     0.0023     0.1363     ONLY: 1:uni10x3461bd0                                                 
      2         7,     0.0030     0.1739     ONLY: 1:base_steel0x35a1810                                           
      3         1,     0.0033     0.1939     ONLY: 5:PMT_3inch_pmt_solid0x43c0a40                                  
      4         9,     0.0041     0.2366     ONLY: 130:sPanel0x4e71750                                             
      5         4,     0.0042     0.2428     ONLY: 5:mask_PMT_20inch_vetosMask0x3c2e7c0                            
      6         3,     0.0068     0.3937     ONLY: 5:HamamatsuR12860sMask0x3c39130                                 
      7         2,     0.0075     0.4331     ONLY: 5:NNVTMCPPMTsMask0x3c2c750                                      
      8    1,2,3,4     0.0172     1.0000     ONLY PMT                                                              
      9        ~8,     0.0238     1.3806     EXCL: 1:uni_acrylic30x35932f0                                         
     10         0,     0.0510     2.9614     ONLY: 3084:sWorld0x33e3370                                            
     11        t8,     0.0875     5.0861     EXCL: 1:uni_acrylic30x35932f0                                         
     12         8,     1.1124    64.6409     ONLY: 1:uni_acrylic30x35932f0                                         
     13        t0,     1.4057    81.6813     EXCL: 3084:sWorld0x33e3370                                            
     14        t4,     1.6793    97.5841     EXCL: 5:mask_PMT_20inch_vetosMask0x3c2e7c0                            
     15        t6,     1.6910    98.2647     EXCL: 1:uni10x3461bd0                                                 
     16        t7,     1.6942    98.4478     EXCL: 1:base_steel0x35a1810                                           
     17        t5,     1.6942    98.4485     EXCL: 1:sStrutBallhead0x34be280                                       
     18         t0     1.6983    98.6845     3084:sWorld0x33e3370                                                  
     19        t9,     1.7003    98.8055     EXCL: 130:sPanel0x4e71750                                             
     20        t3,     1.7042    99.0309     EXCL: 5:HamamatsuR12860sMask0x3c39130                                 
     21        t2,     1.7261   100.3045     EXCL: 5:NNVTMCPPMTsMask0x3c2c750                                      
     22        t1,     1.7581   102.1626     EXCL: 5:PMT_3inch_pmt_solid0x43c0a40                                  
    idx         -e    time(s)   relative     enabled geometry description                                          
    epsilon:~ blyth$ 


For inclusion into s5 presentations the table can be output in RST format::

    epsilon:~ blyth$ snap.py --rst  
    +---+----------+----------+----------+--------------------------------------------------------------------------+
    |idx|        -e|   time(s)|  relative|    enabled geometry description                                          |
    +===+==========+==========+==========+==========================================================================+
    |  0|        5,|    0.0020|    0.1140|    ONLY: 1:sStrutBallhead0x34be280                                       |
    +---+----------+----------+----------+--------------------------------------------------------------------------+
    |  1|        6,|    0.0023|    0.1363|    ONLY: 1:uni10x3461bd0                                                 |
    ...


The commandlines can be dumped::

    epsilon:~ blyth$ snap.py --argline
    OpSnapTest --targetpvn lLowerChimney_phys --eye -1,-1,-1 -e 5, --snapoutdir /tmp/blyth/opticks/snap/lLowerChimney_phys --nameprefix lLowerChimney_phys__5,__ --cvd 1 --rtx 1 --tracer 
    OpSnapTest --targetpvn lLowerChimney_phys --eye -1,-1,-1 -e 6, --snapoutdir /tmp/blyth/opticks/snap/lLowerChimney_phys --nameprefix lLowerChimney_phys__6,__ --cvd 1 --rtx 1 --tracer 
    OpSnapTest --targetpvn lLowerChimney_phys --eye -1,-1,-1 -e 7, --snapoutdir /tmp/blyth/opticks/snap/lLowerChimney_phys --nameprefix lLowerChimney_phys__7,__ --cvd 1 --rtx 1 --tracer 
    OpSnapTest --targetpvn lLowerChimney_phys --eye -1,-1,-1 -e 1, --snapoutdir /tmp/blyth/opticks/snap/lLowerChimney_phys --nameprefix lLowerChimney_phys__1,__ --cvd 1 --rtx 1 --tracer 
    OpSnapTest --targetpvn lLowerChimney_phys --eye -1,-1,-1 -e 9, --snapoutdir /tmp/blyth/opticks/snap/lLowerChimney_phys --nameprefix lLowerChimney_phys__9,__ --cvd 1 --rtx 1 --tracer 
    ...


Also the source for presentations can be dumped::

    epsilon:~ blyth$ snap.py --refjpg          ## image references for inclusion into s5_background_image.txt 
        [0]lLowerChimney_phys__5,__00000
        /env/presentation/snap/lLowerChimney_phys/lLowerChimney_phys__5,__00000.jpg 1280px_720px

        [1]lLowerChimney_phys__6,__00000
        /env/presentation/snap/lLowerChimney_phys/lLowerChimney_phys__6,__00000.jpg 1280px_720px

        [2]lLowerChimney_phys__7,__00000
        /env/presentation/snap/lLowerChimney_phys/lLowerChimney_phys__7,__00000.jpg 1280px_720px
    ...

    epsilon:~ blyth$ snap.py --pagejpg          ## snap pages for inclusion into s5 presentations
    :blue:`[0]lLowerChimney_phys__5,__00000`
    ----------------------------------------

    :blue:`[1]lLowerChimney_phys__6,__00000`
    ----------------------------------------


To view the renders use "--jpg" option::

    epsilon:lLowerChimney_phys blyth$ snap.py --jpg                                ## list jpg snaps in speed of render order
    /tmp/blyth/opticks/snap/lLowerChimney_phys/lLowerChimney_phys__5,__00000.jpg
    /tmp/blyth/opticks/snap/lLowerChimney_phys/lLowerChimney_phys__6,__00000.jpg
    /tmp/blyth/opticks/snap/lLowerChimney_phys/lLowerChimney_phys__7,__00000.jpg
    ...
    epsilon:lLowerChimney_phys blyth$ open $(snap.py --jpg)   
    epsilon:lLowerChimney_phys blyth$ 

