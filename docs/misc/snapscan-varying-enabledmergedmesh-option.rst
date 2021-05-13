snapscan-varying-enabledmergedmesh-option
============================================



Classical GGeo OptiX 6 Geometry
------------------------------------

To identify performance problems with geometry it is useful to 
make render snapshots that vary the included geometry. See:

* https://simoncblyth.bitbucket.io/env/presentation/juno_opticks_20210426.html

This section describes how the renders and tables in the above 
presentation were created using Opticks executables and scripts.

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




OptiX 7 CSGOptiX Experimental Geometry
----------------------------------------

Currently the CSGFoundry geometry needs the --gparts-transform-offset option which
messes up the classical geometry.  But that is only needed during the conversion::

    cd ~/CSG_GGeo
    ./run.sh --gparts_transform_offset    
     ## CSGFoundry geometry is written to tmpdir

On GPU workstation::

    cd ~/CSGOptiX

    ./build.sh     # OptiX 6  
    ./build7.sh    # OptiX 7

    ./cxr.sh       # CSGOptiXRender at default inputs for EMM MOI EYE LOOK UP
                   # creating /tmp/blyth/opticks/CSGOptiX/CSGOptiXRender/70000/render/CSG_GGeo/1/cxr_t8,_sStrut.jpg

    MOI=ALL ./cxr.sh   # creates renders for each line of the arglist from a single geometry load 


Scan the included geometry::

    ./cxr_scan.sh 

On laptop::

    cd ~/CSGOptiX
    ./grab.sh 

    epsilon:CSGOptiX blyth$ ./cxr_table.sh 
    idx         -e    time(s)   relative     enabled geometry description                                          
      0         7,     0.0017     0.1042     ONLY: 1:base_steel0x35a1810                                           
      1         5,     0.0017     0.1049     ONLY: 1:sStrutBallhead0x34be280                                       
      2         9,     0.0018     0.1135     ONLY: 130:sPanel0x4e71750                                             
      3         1,     0.0021     0.1292     ONLY: 5:PMT_3inch_pmt_solid0x43c0a40                                  
      4         6,     0.0029     0.1824     ONLY: 1:uni10x3461bd0                                                 
      5         3,     0.0082     0.5075     ONLY: 5:HamamatsuR12860sMask0x3c39130                                 
      6         4,     0.0082     0.5084     ONLY: 5:mask_PMT_20inch_vetosMask0x3c2e7c0                            
      7         2,     0.0104     0.6408     ONLY: 5:NNVTMCPPMTsMask0x3c2c750                                      
      8    1,2,3,4     0.0162     1.0000     ONLY PMT                                                              
      9         0,     0.1163     7.1935     ONLY: 3084:sWorld0x33e3370                                            
     10        t8,     0.1203     7.4401     EXCL: 1:uni_acrylic30x35932f0                                         
     11         8,     0.5373    33.2399     ONLY: 1:uni_acrylic30x35932f0                                         
     12        t0,     0.5530    34.2133     EXCL: 3084:sWorld0x33e3370                                            
     13        t1,     0.6060    37.4915     EXCL: 5:PMT_3inch_pmt_solid0x43c0a40                                  
     14        t7,     0.6151    38.0565     EXCL: 1:base_steel0x35a1810                                           
     15        t2,     0.6168    38.1601     EXCL: 5:NNVTMCPPMTsMask0x3c2c750                                      
     16        t6,     0.6253    38.6862     EXCL: 1:uni10x3461bd0                                                 
     17        t3,     0.6279    38.8472     EXCL: 5:HamamatsuR12860sMask0x3c39130                                 
     18         t0     0.6334    39.1866     3084:sWorld0x33e3370                                                  
     19        t4,     0.6371    39.4137     EXCL: 5:mask_PMT_20inch_vetosMask0x3c2e7c0                            
     20        t5,     0.6470    40.0268     EXCL: 1:sStrutBallhead0x34be280                                       
     21        t9,     0.6471    40.0361     EXCL: 130:sPanel0x4e71750                                             
    idx         -e    time(s)   relative     enabled geometry description                                          
    epsilon:CSGOptiX blyth$ 


    epsilon:CSGOptiX blyth$ ./cxr_table.sh -h
    usage: 
    ::

        ggeo.py --mm > $TMP/mm.txt    # create list of mm names used for labels

        snap.py       # list the snaps in speed order with labels 

        open $(snap.py --jpg)         # open the jpg ordered by render speed

           [-h] [--level LEVEL] [--basedir BASEDIR] [--reldir RELDIR] [--jpg]
           [--refjpgpfx REFJPGPFX] [--s5base S5BASE] [--refjpg] [--pagejpg]
           [--mvjpg] [--cpjpg] [--argline] [--rst]

    optional arguments:
      -h, --help            show this help message and exit
      --level LEVEL         logging level
      --basedir BASEDIR     base
      --reldir RELDIR       Relative dir beneath $TMP/snap from which to load snap
                            .json
      --jpg                 List jpg paths in speed order
      --refjpgpfx REFJPGPFX
                            List jpg paths s5 background image presentation format
      --s5base S5BASE       Presentation repo base
      --refjpg              List jpg paths s5 background image presentation format
      --pagejpg             List jpg for inclusion into s5 presentation
      --mvjpg               List jpg for inclusion into s5 presentation
      --cpjpg               List cp commands to place into presentation repo
      --argline             List argline in speed order
      --rst                 Dump table in RST format




Comparing
---------------

::

    #!/bin/bash -l 

    adir=/tmp/$USER/opticks/snap 
    bdir=/tmp/$USER/opticks/CSGOptiX/CSGOptiXRender/70000/render/CSG_GGeo/1

    q=${1:-t8,}

    find $adir -name "lLowerChimney_phys__*${q}__00000.jpg" 
    find $bdir -name "*${q}_sWaterTube.jpg" 







