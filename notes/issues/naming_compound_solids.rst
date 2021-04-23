naming_compound_solids
========================

Identification of compound solids via index has staleness risk.

* could name after outer solid or find a common prefix 

::

    epsilon:opticks blyth$ ggeo.sh 1/ --names
    python3 /Users/blyth/opticks/ana/ggeo.py 1/ --names
    nrpo( 176632     1     0     0 )                        PMT_3inch_log_phys0x43c2530                             PMT_3inch_log0x43c1620  114 PMT_3inch_pmt_solid0x43c0a40 
    nrpo( 176633     1     0     1 )                       PMT_3inch_body_phys0x43c1a60                        PMT_3inch_body_log0x43c1510  112 PMT_3inch_body_solid_ell_ell_helper0x43c0d00 
    nrpo( 176634     1     0     2 )                     PMT_3inch_inner1_phys0x43c1ae0                      PMT_3inch_inner1_log0x43c1730  110 PMT_3inch_inner1_solid_ell_helper0x43c0d90 
    nrpo( 176635     1     0     3 )                     PMT_3inch_inner2_phys0x43c1b90                      PMT_3inch_inner2_log0x43c1840  111 PMT_3inch_inner2_solid_ell_helper0x43c0e70 
    nrpo( 176636     1     0     4 )                       PMT_3inch_cntr_phys0x43c1c40                        PMT_3inch_cntr_log0x43c1950  113 PMT_3inch_cntr_solid0x43c0f00 

    epsilon:opticks blyth$ ggeo.sh 2/ --names
    python3 /Users/blyth/opticks/ana/ggeo.py 2/ --names
    [2021-04-23 21:35:56,440] p77460 {/Users/blyth/opticks/ana/ggeo.py:438} INFO - mismatch oidx2:1(from nrpo/iid)  oidx:0(from range(num_volumes)) 
    [2021-04-23 21:35:56,441] p77460 {/Users/blyth/opticks/ana/ggeo.py:438} INFO - mismatch oidx2:2(from nrpo/iid)  oidx:1(from range(num_volumes)) 
    [2021-04-23 21:35:56,441] p77460 {/Users/blyth/opticks/ana/ggeo.py:438} INFO - mismatch oidx2:3(from nrpo/iid)  oidx:2(from range(num_volumes)) 
    [2021-04-23 21:35:56,441] p77460 {/Users/blyth/opticks/ana/ggeo.py:438} INFO - mismatch oidx2:4(from nrpo/iid)  oidx:3(from range(num_volumes)) 
    [2021-04-23 21:35:56,441] p77460 {/Users/blyth/opticks/ana/ggeo.py:438} INFO - mismatch oidx2:5(from nrpo/iid)  oidx:4(from range(num_volumes)) 
    nrpo(  70961     2     0     1 )                           NNVTMCPPMTpMask0x3c2cad0                           NNVTMCPPMTlMask0x3c2c950   98 NNVTMCPPMTsMask0x3c2c750 
    nrpo(  70962     2     0     2 )            NNVTMCPPMT_PMT_20inch_log_phys0x3c2cb50                 NNVTMCPPMT_PMT_20inch_log0x3c2a6b0  102 NNVTMCPPMT_PMT_20inch_pmt_solid0x3c21980 
    nrpo(  70963     2     0     3 )           NNVTMCPPMT_PMT_20inch_body_phys0x3c2aa10            NNVTMCPPMT_PMT_20inch_body_log0x3c2a5d0  101 NNVTMCPPMT_PMT_20inch_body_solid0x3c258a0 
    nrpo(  70964     2     0     4 )         NNVTMCPPMT_PMT_20inch_inner1_phys0x3c2aaa0          NNVTMCPPMT_PMT_20inch_inner1_log0x3c2a7d0   99 NNVTMCPPMT_PMT_20inch_inner1_solid_1_Ellipsoid0x3497520 
    nrpo(  70965     2     0     5 )         NNVTMCPPMT_PMT_20inch_inner2_phys0x3c2ab60          NNVTMCPPMT_PMT_20inch_inner2_log0x3c2a8f0  100 NNVTMCPPMT_PMT_20inch_inner2_solid0x3c2a360 

    epsilon:opticks blyth$ ggeo.sh 3/ --names
    python3 /Users/blyth/opticks/ana/ggeo.py 3/ --names
    [2021-04-23 21:36:17,843] p78569 {/Users/blyth/opticks/ana/ggeo.py:438} INFO - mismatch oidx2:1(from nrpo/iid)  oidx:0(from range(num_volumes)) 
    [2021-04-23 21:36:17,843] p78569 {/Users/blyth/opticks/ana/ggeo.py:438} INFO - mismatch oidx2:2(from nrpo/iid)  oidx:1(from range(num_volumes)) 
    [2021-04-23 21:36:17,843] p78569 {/Users/blyth/opticks/ana/ggeo.py:438} INFO - mismatch oidx2:3(from nrpo/iid)  oidx:2(from range(num_volumes)) 
    [2021-04-23 21:36:17,843] p78569 {/Users/blyth/opticks/ana/ggeo.py:438} INFO - mismatch oidx2:4(from nrpo/iid)  oidx:3(from range(num_volumes)) 
    [2021-04-23 21:36:17,843] p78569 {/Users/blyth/opticks/ana/ggeo.py:438} INFO - mismatch oidx2:5(from nrpo/iid)  oidx:4(from range(num_volumes)) 
    nrpo(  70967     3     0     1 )                      HamamatsuR12860pMask0x3c394b0                      HamamatsuR12860lMask0x3c39330  104 HamamatsuR12860sMask0x3c39130 
    nrpo(  70968     3     0     2 )       HamamatsuR12860_PMT_20inch_log_phys0x3c39540            HamamatsuR12860_PMT_20inch_log0x3c36c90  108 HamamatsuR12860_PMT_20inch_pmt_solid_1_90x3c4a970 
    nrpo(  70969     3     0     3 )      HamamatsuR12860_PMT_20inch_body_phys0x33eeec0       HamamatsuR12860_PMT_20inch_body_log0x3c36ba0  107 HamamatsuR12860_PMT_20inch_body_solid_1_90x3c28080 
    nrpo(  70970     3     0     4 )    HamamatsuR12860_PMT_20inch_inner1_phys0x3c373b0     HamamatsuR12860_PMT_20inch_inner1_log0x33eec60  105 HamamatsuR12860_PMT_20inch_inner1_solid_I0x3c32bc0 
    nrpo(  70971     3     0     5 )    HamamatsuR12860_PMT_20inch_inner2_phys0x3c37470     HamamatsuR12860_PMT_20inch_inner2_log0x33eed90  106 HamamatsuR12860_PMT_20inch_inner2_solid_1_90x3c36980 

    epsilon:opticks blyth$ ggeo.sh 4/ --names
    python3 /Users/blyth/opticks/ana/ggeo.py 4/ --names
    [2021-04-23 21:36:39,571] p79676 {/Users/blyth/opticks/ana/ggeo.py:438} INFO - mismatch oidx2:1(from nrpo/iid)  oidx:0(from range(num_volumes)) 
    [2021-04-23 21:36:39,572] p79676 {/Users/blyth/opticks/ana/ggeo.py:438} INFO - mismatch oidx2:2(from nrpo/iid)  oidx:1(from range(num_volumes)) 
    [2021-04-23 21:36:39,572] p79676 {/Users/blyth/opticks/ana/ggeo.py:438} INFO - mismatch oidx2:3(from nrpo/iid)  oidx:2(from range(num_volumes)) 
    [2021-04-23 21:36:39,572] p79676 {/Users/blyth/opticks/ana/ggeo.py:438} INFO - mismatch oidx2:4(from nrpo/iid)  oidx:3(from range(num_volumes)) 
    [2021-04-23 21:36:39,572] p79676 {/Users/blyth/opticks/ana/ggeo.py:438} INFO - mismatch oidx2:5(from nrpo/iid)  oidx:4(from range(num_volumes)) 
    nrpo( 304637     4     0     1 )                 mask_PMT_20inch_vetopMask0x3c2eb60                 mask_PMT_20inch_vetolMask0x3c2e9d0  121 mask_PMT_20inch_vetosMask0x3c2e7c0 
    nrpo( 304638     4     0     2 )                  PMT_20inch_veto_log_phys0x3c3e950                       PMT_20inch_veto_log0x3c3de20  125 PMT_20inch_veto_pmt_solid_1_20x3c305d0 
    nrpo( 304639     4     0     3 )                 PMT_20inch_veto_body_phys0x3c3e150                  PMT_20inch_veto_body_log0x3c3dd10  124 PMT_20inch_veto_body_solid_1_20x3c3cc50 
    nrpo( 304640     4     0     4 )               PMT_20inch_veto_inner1_phys0x3c3e1d0                PMT_20inch_veto_inner1_log0x3c3df30  122 PMT_20inch_veto_inner1_solid0x3c3d8c0 
    nrpo( 304641     4     0     5 )               PMT_20inch_veto_inner2_phys0x3c3e280                PMT_20inch_veto_inner2_log0x3c3e040  123 PMT_20inch_veto_inner2_solid0x3c3dae0 

    epsilon:opticks blyth$ ggeo.sh 5/ --names
    python3 /Users/blyth/opticks/ana/ggeo.py 5/ --names
    nrpo(  68488     5     0     0 )                               lSteel_phys0x34c07b0                                    lSteel0x34c0680   93 sStrutBallhead0x34be280 

    epsilon:opticks blyth$ ggeo.sh 6/ --names
    python3 /Users/blyth/opticks/ana/ggeo.py 6/ --names
    nrpo(  69078     6     0     0 )                           lFasteners_phys0x3461f60                                lFasteners0x3461e20   94 uni10x3461bd0 

    epsilon:opticks blyth$ ggeo.sh 7/ --names
    python3 /Users/blyth/opticks/ana/ggeo.py 7/ --names
    nrpo(  69668     7     0     0 )                               lUpper_phys0x35499e0                                    lUpper0x3549920   95 base_steel0x35a1810 

    epsilon:opticks blyth$ ggeo.sh 8/ --names
    python3 /Users/blyth/opticks/ana/ggeo.py 8/ --names
    nrpo(  70258     8     0     0 )                            lAddition_phys0x3593690                                 lAddition0x3593510   96 uni_acrylic30x35932f0 

    epsilon:opticks blyth$ ggeo.sh 9/ --names
    python3 /Users/blyth/opticks/ana/ggeo.py 9/ --names
    nrpo(     10     9     0     0 )                               pPanel_0_f_0x4e7c3c0                                    lPanel0x4e71970    7 sPanel0x4e71750 
    nrpo(     11     9     0     1 )                                pPanelTape0x4e7c6a0                                lPanelTape0x4e71b00    6 sPanelTape0x4e71a70 
    nrpo(     12     9     0     2 )                              pCoating_00_0x4e7c740                                  lCoating0x4e71c90    5 sBar0x4e71c00 
    nrpo(     13     9     0     3 )                                      pBar0x4e7f1c0                                      lBar0x4e71e20    4 sBar0x4e71d90 
    nrpo(     14     9     0     4 )                              pCoating_01_0x4e7c7e0                                  lCoating0x4e71c90    5 sBar0x4e71c00 
    nrpo(     15     9     0     5 )                                      pBar0x4e7f1c0                                      lBar0x4e71e20    4 sBar0x4e71d90 
    nrpo(     16     9     0     6 )                              pCoating_02_0x4e7c880                                  lCoating0x4e71c90    5 sBar0x4e71c00 
    ...

