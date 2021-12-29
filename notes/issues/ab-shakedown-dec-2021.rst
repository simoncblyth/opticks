ab-shakedown
=============

Related
---------

* ~/opticks/notes/issues/analysis_shakedown.rst


tds3gun.sh get : fails to grab as non-existing P:/tmp/blyth/opticks/tds3gun/evt/g4live/natural
-------------------------------------------------------------------------------------------------

::

    epsilon:opticks blyth$ tds3gun.sh get 
    PFX=tds3gun evtsync.sh
    evtsync from P:/tmp/blyth/opticks/tds3gun/evt/g4live/natural to /tmp/blyth/opticks/tds3gun/evt/g4live/natural
    receiving incremental file list
    rsync: change_dir "/tmp/blyth/opticks/tds3gun/evt/g4live/natural" failed: No such file or directory (2)

    sent 68 bytes  received 117 bytes  24.67 bytes/sec
    total size is 0  speedup is 0.00
    rsync error: some files/attrs were not transferred (see previous errors) (code 23) at main.c(1679) [Receiver=3.1.3]
    rsync: [Receiver] write error: Broken pipe (32)
    === evtsync : non-zero rc from rsync : start ssh tunnel with "tun" and ensure remote directory being grabbed exists
    epsilon:opticks blyth$ 


Only recent event dir from tds3 running is /tmp/blyth/opticks/evt/g4live/natural/1/ containing gensteps only
-------------------------------------------------------------------------------------------------------------

::

    N[blyth@localhost ~]$ l /tmp/blyth/opticks/evt/g4live/natural/1/
    total 16
     4 -rw-rw-r--. 1 blyth blyth    50 Dec 27 03:34 gs.json
    12 -rw-rw-r--. 1 blyth blyth 11600 Dec 27 03:34 gs.npy
     0 drwxrwxr-x. 4 blyth blyth    24 Dec 10 23:08 ..
     0 drwxrwxr-x. 2 blyth blyth    35 Dec 10 23:08 .
    N[blyth@localhost ~]$ date
    Mon Dec 27 18:19:39 CST 2021
    N[blyth@localhost ~]$ 


what does tds3gun do that tds3 does not ? what controls event writing and locations ?
--------------------------------------------------------------------------------------

Very little, just defines OPTICKS_EVENT_PFX and G4OpticksAnaMgr_outdir::

    2391 tds3gun(){
    2392    : unsets ctrl evars that may be defined from other funcs
    2393    export OPTICKS_EVENT_PFX=tds3gun
    2394    unset INPUT_PHOTON_PATH
    2395 
    2396    local outdir="/tmp/G4OpticksAnaMgr"
    2397    mkdir -p $outdir
    2398    export G4OpticksAnaMgr_outdir=$outdir
    2399 
    2400    tds3
    2401 }


OPTICKS_EVENT_PFX
---------------------

::

    epsilon:opticks blyth$ opticks-f OPTICKS_EVENT_PFX
    ./optickscore/OpticksCfg.cc:The envvars OPTICKS_EVENT_PFX and TESTNAME are checked in order, the first that
    ./optickscore/OpticksCfg.cc:    const char* pfx_envvar_default = SSys::getenvvar("OPTICKS_EVENT_PFX,TESTNAME" , NULL ); 
    epsilon:opticks blyth$ 


    1687 /**
    1688 OpticksCfg::getEventPfx
    1689 -------------------------
    1690 
    1691 Canonically used from Opticks::defineEventSpec to create Opticks::m_spec and m_nspec
    1692 that are the blueprints for all OpticksEvent.
    1693 
    1694 The envvars OPTICKS_EVENT_PFX and TESTNAME are checked in order, the first that
    1695 is found to be set defines the default pfx.  
    1696 This default is overridden by the commandline --pfx argument.
    1697 
    1698 **/
    1699 
    1700 template <class Listener>
    1701 const char* OpticksCfg<Listener>::getEventPfx() const
    1702 {
    1703     const char* pfx_envvar_default = SSys::getenvvar("OPTICKS_EVENT_PFX,TESTNAME" , NULL );
    1704     return m_event_pfx.empty() ? pfx_envvar_default : m_event_pfx.c_str() ;
    1705 }



    epsilon:opticks blyth$ opticks-f getEventPfx
    ./optickscore/OpticksAna.cc:         << "--pfx " << m_ok->getEventPfx() << " "
    ./optickscore/Opticks.hh:       const char*          getEventPfx() const ;
    ./optickscore/OpticksCfg.cc:OpticksCfg::getEventPfx
    ./optickscore/OpticksCfg.cc:const char* OpticksCfg<Listener>::getEventPfx() const 
    ./optickscore/OpticksCfg.hh:     const char* getEventPfx() const ;
    ./optickscore/Opticks.cc:    const char* resource_pfx = m_rsc->getEventPfx() ; 
    ./optickscore/Opticks.cc:    const char* config_pfx = m_cfg->getEventPfx() ; 
    ./optickscore/Opticks.cc:const char* Opticks::getEventPfx() const
    ./optickscore/Opticks.cc:    const char* pfx = getEventPfx();
    ./optickscore/Opticks.cc:    const char* pfx = getEventPfx();
    ./optickscore/Opticks.cc:    const char* pfx = getEventPfx();
    ./boostrap/BOpticksResource.hh:        const char* getEventPfx() const ; 
    ./boostrap/BOpticksResource.cc:const char* BOpticksResource::getEventPfx() const {  return m_evtpfx ; } 
    epsilon:opticks blyth$ 



    3046 void Opticks::defineEventSpec()
    3047 {
    3048     const char* cat = m_cfg->getEventCat(); // expected to be defined for tests and equal to the TESTNAME from bash functions like tboolean-
    3049     const char* udet = getInputUDet();
    3050     const char* tag = m_cfg->getEventTag();
    3051     const char* ntag = BStr::negate(tag) ;
    3052     const char* typ = getSourceType();
    3053 
    3054     const char* resource_pfx = m_rsc->getEventPfx() ;
    3055     const char* config_pfx = m_cfg->getEventPfx() ;
    3056     const char* pfx = config_pfx ? config_pfx : resource_pfx ;
    3057     if( !pfx )
    3058     {
    3059         pfx = DEFAULT_PFX ;
    3060         LOG(fatal)
    3061             << " resource_pfx " << resource_pfx
    3062             << " config_pfx " << config_pfx
    3063             << " pfx " << pfx
    3064             << " cat " << cat
    3065             << " udet " << udet
    3066             << " typ " << typ
    3067             << " tag " << tag
    3068             ;
    3069     }
    3070     //assert( pfx ); 
    3071 
    3072 
    3073     m_spec  = new OpticksEventSpec(pfx, typ,  tag, udet, cat );
    3074     m_nspec = new OpticksEventSpec(pfx, typ, ntag, udet, cat );
    3075 
    3076     LOG(LEVEL)
    3077          << " pfx " << pfx
    3078          << " typ " << typ
    3079          << " tag " << tag
    3080          << " ntag " << ntag
    3081          << " udet " << udet
    3082          << " cat " << cat
    3083          ;
    3084 
    3085 }


Logging from tds3gun.sh::

    0
        19 gp.x    3550.77 gp.y   -3828.58 gp.z   18657.51 gp.R   19374.43 pmt  314239          SI|SD|BT|EX otk     10 oti    3.61 bti    94.42 bp.x    3253.14 bp.y   -3530.91 bp.z   17161.07 bp.R   17820.00
    2021-12-27 18:33:56.404 INFO  [13051] [junoSD_PMT_v2_Opticks::EndOfEvent@180] ] num_hit 4887 merged_count  0 m_merged_total 0 m_opticksMode 3
    2021-12-27 18:33:56.404 INFO  [13051] [junoSD_PMT_v2_Opticks::TerminateEvent@227]  NOT invoking G4Opticks::reset as G4OpticksRecorder detected, should do reset in G4OpticksRecorder::TerminateEvent
    junoSD_PMT_v2::EndOfEvent m_opticksMode 3 hitCollection 6385 hitCollection_muon 0 hitCollection_opticks 0
    2021-12-27 18:33:56.404 INFO  [13051] [OpticksEvent::save@1972] /tmp/blyth/opticks/tds3gun/evt/g4live/natural/-2
    2021-12-27 18:33:56.427 FATAL [13051] [G4Opticks::reset@544]  m_way_enabled reset m_hiys 
    junotoptask:DetSimAlg.finalize  INFO: DetSimAlg finalized successfully
    junotoptask:DetSim0Svc.dumpOpticks  INFO: DetSim0Svc::finalizeOpticks m_opticksMode 3 WITH_G4OPTICKS 
    2021-12-27 18:33:56.428 INFO  [13051] [G4Opticks::Finalize@290] G4Opticks.desc ok 0x82ea240 opmgr 0x2c5989c0 


Now the events are being saved, what did tds3gun do that tds3 did not to switch that on ?::

    N[blyth@localhost ~]$ cd /tmp/blyth/opticks/tds3gun/evt/g4live/natural/-2
    N[blyth@localhost -2]$ l
    total 10012
       4 drwxrwxr-x. 3 blyth blyth    4096 Dec 27 18:33 .
       4 drwxrwxr-x. 2 blyth blyth    4096 Dec 27 18:33 20211227_183355
       4 -rw-rw-r--. 1 blyth blyth     144 Dec 27 18:33 OpticksProfileAccLabels.npy
       4 -rw-rw-r--. 1 blyth blyth      96 Dec 27 18:33 OpticksProfileAcc.npy
       4 -rw-rw-r--. 1 blyth blyth     144 Dec 27 18:33 OpticksProfileLisLabels.npy
       4 -rw-rw-r--. 1 blyth blyth      88 Dec 27 18:33 OpticksProfileLis.npy
       4 -rw-rw-r--. 1 blyth blyth    2068 Dec 27 18:33 report.txt
       0 -rw-rw-r--. 1 blyth blyth       0 Dec 27 18:33 DeltaTime.ini
       0 -rw-rw-r--. 1 blyth blyth       0 Dec 27 18:33 DeltaVM.ini
       4 -rw-rw-r--. 1 blyth blyth      96 Dec 27 18:33 idom.npy
       4 -rw-rw-r--. 1 blyth blyth      80 Dec 27 18:33 OpticksProfileLabels.npy
       4 -rw-rw-r--. 1 blyth blyth      80 Dec 27 18:33 OpticksProfile.npy
       4 -rw-rw-r--. 1 blyth blyth    1772 Dec 27 18:33 parameters.json
       0 -rw-rw-r--. 1 blyth blyth       0 Dec 27 18:33 Time.ini
       0 -rw-rw-r--. 1 blyth blyth       0 Dec 27 18:33 VM.ini
       4 -rw-rw-r--. 1 blyth blyth     128 Dec 27 18:33 fdom.npy
       4 -rw-rw-r--. 1 blyth blyth      80 Dec 27 18:33 bn.npy
       4 -rw-rw-r--. 1 blyth blyth      80 Dec 27 18:33 dg.npy
     184 -rw-rw-r--. 1 blyth blyth  184928 Dec 27 18:33 ph.npy
    7224 -rw-rw-r--. 1 blyth blyth 7394016 Dec 27 18:33 dx.npy
    1808 -rw-rw-r--. 1 blyth blyth 1848576 Dec 27 18:33 rx.npy
       4 -rw-rw-r--. 1 blyth blyth      80 Dec 27 18:33 ht.npy
       4 -rw-rw-r--. 1 blyth blyth      80 Dec 27 18:33 hy.npy
     724 -rw-rw-r--. 1 blyth blyth  739472 Dec 27 18:33 ox.npy
       0 drwxrwxr-x. 7 blyth blyth      53 Dec 27 18:33 ..
       4 -rw-rw-r--. 1 blyth blyth      28 Dec 27 18:33 so.json
       4 -rw-rw-r--. 1 blyth blyth      80 Dec 27 18:33 so.npy
    N[blyth@localhost -2]$ cd /tmp/blyth/opticks/tds3gun/evt/g4live/natural/2
    N[blyth@localhost 2]$ l
    total 4320
       0 drwxrwxr-x. 7 blyth blyth      53 Dec 27 18:33 ..
       4 drwxrwxr-x. 2 blyth blyth    4096 Dec 27 18:33 20211227_183355
       4 drwxrwxr-x. 3 blyth blyth    4096 Dec 27 18:33 .
       4 -rw-rw-r--. 1 blyth blyth     144 Dec 27 18:33 OpticksProfileAccLabels.npy
       4 -rw-rw-r--. 1 blyth blyth      96 Dec 27 18:33 OpticksProfileAcc.npy
       4 -rw-rw-r--. 1 blyth blyth      80 Dec 27 18:33 OpticksProfileLabels.npy
       4 -rw-rw-r--. 1 blyth blyth     144 Dec 27 18:33 OpticksProfileLisLabels.npy
       4 -rw-rw-r--. 1 blyth blyth      88 Dec 27 18:33 OpticksProfileLis.npy
       4 -rw-rw-r--. 1 blyth blyth      80 Dec 27 18:33 OpticksProfile.npy
       4 -rw-rw-r--. 1 blyth blyth    2022 Dec 27 18:33 report.txt
       0 -rw-rw-r--. 1 blyth blyth       0 Dec 27 18:33 DeltaTime.ini
       0 -rw-rw-r--. 1 blyth blyth       0 Dec 27 18:33 DeltaVM.ini
       4 -rw-rw-r--. 1 blyth blyth      96 Dec 27 18:33 idom.npy
       4 -rw-rw-r--. 1 blyth blyth    1748 Dec 27 18:33 parameters.json
       0 -rw-rw-r--. 1 blyth blyth       0 Dec 27 18:33 Time.ini
       0 -rw-rw-r--. 1 blyth blyth       0 Dec 27 18:33 VM.ini
       4 -rw-rw-r--. 1 blyth blyth     128 Dec 27 18:33 fdom.npy
       4 -rw-rw-r--. 1 blyth blyth     692 Dec 27 18:33 Boundary_IndexLocal.json
       4 -rw-rw-r--. 1 blyth blyth     696 Dec 27 18:33 Boundary_IndexSource.json
       4 -rw-rw-r--. 1 blyth blyth     636 Dec 27 18:33 History_SequenceLocal.json
       4 -rw-rw-r--. 1 blyth blyth     649 Dec 27 18:33 Material_SequenceLocal.json
       4 -rw-rw-r--. 1 blyth blyth     674 Dec 27 18:33 Material_SequenceSource.json
       4 -rw-rw-r--. 1 blyth blyth     663 Dec 27 18:33 History_SequenceSource.json
      48 -rw-rw-r--. 1 blyth blyth   46292 Dec 27 18:33 ps.npy
     452 -rw-rw-r--. 1 blyth blyth  462216 Dec 27 18:33 rs.npy
     184 -rw-rw-r--. 1 blyth blyth  184928 Dec 27 18:33 bn.npy
       4 -rw-rw-r--. 1 blyth blyth      80 Dec 27 18:33 dg.npy
       4 -rw-rw-r--. 1 blyth blyth      96 Dec 27 18:33 dx.npy
     184 -rw-rw-r--. 1 blyth blyth  184928 Dec 27 18:33 ph.npy
     364 -rw-rw-r--. 1 blyth blyth  369776 Dec 27 18:33 wy.npy
    1808 -rw-rw-r--. 1 blyth blyth 1848576 Dec 27 18:33 rx.npy
      12 -rw-rw-r--. 1 blyth blyth   11024 Dec 27 18:33 gs.npy
     308 -rw-rw-r--. 1 blyth blyth  312848 Dec 27 18:33 ht.npy
     156 -rw-rw-r--. 1 blyth blyth  156464 Dec 27 18:33 hy.npy
     724 -rw-rw-r--. 1 blyth blyth  739472 Dec 27 18:33 ox.npy
    N[blyth@localhost 2]$ 



Now "tds3gun.sh get" succeeds to grab
----------------------------------------

::

    epsilon:optickscore blyth$ tds3gun.sh get
    PFX=tds3gun evtsync.sh
    evtsync from P:/tmp/blyth/opticks/tds3gun/evt/g4live/natural to /tmp/blyth/opticks/tds3gun/evt/g4live/natural
    receiving incremental file list
    ./
    -1/
    -1/DeltaTime.ini
                  0 100%    0.00kB/s    0:00:00 (xfr#1, to-chk=164/171)
    -1/DeltaVM.ini
                  0 100%    0.00kB/s    0:00:00 (xfr#2, to-chk=163/171)



And "tds3gun.sh 1" succeeds to compare : bookkeeping mismatch presumably from the virtual hatboxes
-----------------------------------------------------------------------------------------------------
 

::

    tds3gun.sh 1

    #ab.ahis
    ab.ahis
    .    all_seqhis_ana  cfo:sum  1:g4live:tds3gun   -1:g4live:tds3gun 
    .             TOTALS:    11300    11300                  8372.99     8372.99/68 = 123.13   pvalue:P[C2>]:1.000  1-pvalue:P[C2<]:0.000 
       n             iseq        a        b      a-b     (a-b)^2/(a+b)        a/b                  b/a          [ns]   label
    0000               42     1727     1683       44            0.57        1.026 +- 0.025       0.975 +- 0.024 [2 ]   SI AB
    0001         7cccccc2       73     1795     -1722         1587.41        0.041 +- 0.005      24.589 +- 0.580 [8 ]   SI BT BT BT BT BT BT SD
    0002          7ccccc2     1836        1     1835         1833.00     1836.000 +- 42.849       0.001 +- 0.001 [7 ]   SI BT BT BT BT BT SD
    0003              452      515      505       10            0.10        1.020 +- 0.045       0.981 +- 0.044 [3 ]   SI RE AB
    0004         7ccccc62      809        1      808          806.00      809.000 +- 28.443       0.001 +- 0.001 [8 ]   SI SC BT BT BT BT BT SD
    0005        7cccccc62       48      713     -665          581.11        0.067 +- 0.010      14.854 +- 0.556 [9 ]   SI SC BT BT BT BT BT BT SD
    0006              462      379      377        2            0.01        1.005 +- 0.052       0.995 +- 0.051 [3 ]   SI SC AB
    0007        7cccccc52       23      518     -495          452.91        0.044 +- 0.009      22.522 +- 0.990 [9 ]   SI RE BT BT BT BT BT BT SD
    0008         7ccccc52      531        0      531          531.00        0.000 +- 0.000       0.000 +- 0.000 [8 ]   SI RE BT BT BT BT BT SD
    0009               41      256      262       -6            0.07        0.977 +- 0.061       1.023 +- 0.063 [2 ]   CK AB
    0010             4552      173      162       11            0.36        1.068 +- 0.081       0.936 +- 0.074 [4 ]   SI RE RE AB
    0011             8cc2      134      173      -39            4.95        0.775 +- 0.067       1.291 +- 0.098 [4 ]   SI BT BT SA
    0012        7ccccc662      267        0      267          267.00        0.000 +- 0.000       0.000 +- 0.000 [9 ]   SI SC SC BT BT BT BT BT SD
    0013       7cccccc662       15      249     -234          207.41        0.060 +- 0.016      16.600 +- 1.052 [10]   SI SC SC BT BT BT BT BT BT SD
    0014       7cccccc652       14      230     -216          191.21        0.061 +- 0.016      16.429 +- 1.083 [10]   SI RE SC BT BT BT BT BT BT SD
    0015             4652      119      120       -1            0.00        0.992 +- 0.091       1.008 +- 0.092 [4 ]   SI RE SC AB
    0016        7ccccc652      235        0      235          235.00        0.000 +- 0.000       0.000 +- 0.000 [9 ]   SI RE SC BT BT BT BT BT SD
    0017             4662      100      110      -10            0.48        0.909 +- 0.091       1.100 +- 0.105 [4 ]   SI SC SC AB
    0018             4cc2      109       87       22            2.47        1.253 +- 0.120       0.798 +- 0.086 [4 ]   SI BT BT AB
    .             TOTALS:    11300    11300                  8372.99     8372.99/68 = 123.13   pvalue:P[C2>]:1.000  1-pvalue:P[C2<]:0.000 




Clear difference in the number of BT before SD::

    G4 b : SI 6*BT SD 
    OK a : SI 5*BT SD
    .                            a        b
    0001         7cccccc2       73     1795     -1722         1587.41        0.041 +- 0.005      24.589 +- 0.580 [8 ]   SI BT BT BT BT BT BT SD
    0002          7ccccc2     1836        1     1835         1833.00     1836.000 +- 42.849       0.001 +- 0.001 [7 ]   SI BT BT BT BT BT SD


* I have fixed something like this before  : on that occasion it was the degenerate PMT surface causing microsteps that have to be skipped
  from G4 to align the bookkeeping  


Comparing with the histories from Jul, see that there are matched "SI 3*BT SD" so the issue is arising 
with the additional surfaces from the mask

* OR could be intersects onto the virtual hatboxes that are skipped from the GPU geometry 
  (that skipping is for pretty visuals cosmetics 

* TODO: set up some planar input photons so can visualize the photons 
  together with the geometry in my 2D cxs cross section scatter plots  



hatbox skipping of virtuals is for the cosmetics of the renders so when ab comparing need to not skip them
------------------------------------------------------------------------------------------------------------------

::

    2445 tds-skipsolidname(){ echo $(tds-skipsolidname-) | tr " " "," ; }
    2446 tds-skipsolidname-(){ cat << EON | grep -v ^#
    2447 
    2448 NNVTMCPPMTsMask_virtual
    2449 HamamatsuR12860sMask_virtual
    2450 mask_PMT_20inch_vetosMask_virtual
    2451 
    2452 NNVTMCPPMT_PMT_20inch_body_solid_1_2
    2453 HamamatsuR12860_PMT_20inch_body_solid_1_4
    2454 PMT_20inch_veto_body_solid_1_2
    2455 
    2456 EON
    2457 }
    2458 



Look back at matched histories from Jul 2021 (without mask)
-------------------------------------------------------------

* http://localhost/env/presentation/juno_opticks_20210712.html


::

    .. raw:: html 

        <pre class="mypretiny">
        epsilon:ana blyth$ tds3gun.sh 1    ## <b>seqhis: 64bit uint : 16x4bit step flags for each photon</b>
        In [1]: ab.his[:20]   ##  OK:1    G4:-1     OK-G4   "c2" deviation     

        .  n           seqhis        a        b      a-b     (a-b)^2/(a+b)   label                ## optickscore/OpticksPhoton.h enum 
                  ## hexstring
        0000               42     1822     1721      101            2.88     SI AB                ## <b>AB : BULK_ABSORB </b> 
        0001            7ccc2     1446     1406       40            0.56     SI BT BT BT SD       ## <b>SD : SURFACE_DETECT </b> 
        0002           7ccc62      672      666        6            0.03     SI SC BT BT BT SD    ## <b>SC : BULK_SCATTER </b> 
        0003            8ccc2      649      597       52            2.17     SI BT BT BT SA       ## <b>BT : BOUNDARY_TRANSMIT </b> 
        0004             8cc2      606      615       -9            0.07     SI BT BT SA          ## <b>SI : SCINTILLATION </b> 
        0005              452      538      536        2            0.00     SI RE AB             ## <b>RE : BULK_REEMIT </b> 
        0006           7ccc52      433      438       -5            0.03     SI RE BT BT BT SD    
        0007              462      397      405       -8            0.08     SI SC AB
        0008           8ccc62      269      262        7            0.09     SI SC BT BT BT SA
        0009          7ccc662      242      222       20            0.86     SI SC SC BT BT BT SD
        0010            8cc62      217      212        5            0.06     SI SC BT BT SA
        0011          7ccc652      211      205        6            0.09     SI RE SC BT BT BT SD
        0012           8ccc52      200      201       -1            0.00     SI RE BT BT BT SA
        0013            8cc52      158      192      -34            3.30     SI RE BT BT SA
        0014             4552      181      165       16            0.74     SI RE RE AB
        0015               41      164      145       19            1.17     CK AB                ## <b>CK : CERENKOV</b> 
        0016          7ccc552      135      160      -25            2.12     SI RE RE BT BT BT SD
        0017             4cc2      130      115       15            0.92     SI BT BT AB
        0018             4652      120      117        3            0.04     SI RE SC AB

        .             TOTALS:    11684    11684                    52.92     52.92/63 =  0.84   pvalue:P[C2>]:0.814  1-pvalue:P[C2<]:0.186 





Without the hatbox virtuals skipped : the bookkeeping matches much better (Tue Dec 28, 2021)
----------------------------------------------------------------------------------------------

::

    #ab.ahis
    ab.ahis
    .    all_seqhis_ana  cfo:sum  1:g4live:tds3gun   -1:g4live:tds3gun 
    .             TOTALS:    11300    11300                    94.69     94.69/62 =  1.53   pvalue:P[C2>]:1.000  1-pvalue:P[C2<]:0.000 
       n             iseq        a        b      a-b     (a-b)^2/(a+b)        a/b                  b/a          [ns]   label
    0000         7cccccc2     1832     1795       37            0.38        1.021 +- 0.024       0.980 +- 0.023 [8 ]   SI BT BT BT BT BT BT SD
    0001               42     1727     1683       44            0.57        1.026 +- 0.025       0.975 +- 0.024 [2 ]   SI AB
    0002        7cccccc62      766      713       53            1.90        1.074 +- 0.039       0.931 +- 0.035 [9 ]   SI SC BT BT BT BT BT BT SD
    0003        7cccccc52      522      518        4            0.02        1.008 +- 0.044       0.992 +- 0.044 [9 ]   SI RE BT BT BT BT BT BT SD
    0004              452      515      505       10            0.10        1.020 +- 0.045       0.981 +- 0.044 [3 ]   SI RE AB
    0005              462      379      377        2            0.01        1.005 +- 0.052       0.995 +- 0.051 [3 ]   SI SC AB
    0006               41      256      262       -6            0.07        0.977 +- 0.061       1.023 +- 0.063 [2 ]   CK AB
    0007       7cccccc662      246      249       -3            0.02        0.988 +- 0.063       1.012 +- 0.064 [10]   SI SC SC BT BT BT BT BT BT SD
    0008       7cccccc652      212      230      -18            0.73        0.922 +- 0.063       1.085 +- 0.072 [10]   SI RE SC BT BT BT BT BT BT SD
    0009       7cccccc552      167      168       -1            0.00        0.994 +- 0.077       1.006 +- 0.078 [10]   SI RE RE BT BT BT BT BT BT SD
    0010             4552      173      162       11            0.36        1.068 +- 0.081       0.936 +- 0.074 [4 ]   SI RE RE AB
    0011             8cc2      118      173      -55           10.40        0.682 +- 0.063       1.466 +- 0.111 [4 ]   *SI BT BT SA*
    0012             4652      119      120       -1            0.00        0.992 +- 0.091       1.008 +- 0.092 [4 ]   SI RE SC AB
    0013       cccccc6662      124      104       20            1.75        1.192 +- 0.107       0.839 +- 0.082 [10]   SI SC SC SC BT BT BT BT BT BT
    0014       cccccc6652      111      114       -3            0.04        0.974 +- 0.092       1.027 +- 0.096 [10]   SI RE SC SC BT BT BT BT BT BT
    0015             4662      100      110      -10            0.48        0.909 +- 0.091       1.100 +- 0.105 [4 ]   SI SC SC AB
    0016             4cc2      101       87       14            1.04        1.161 +- 0.116       0.861 +- 0.092 [4 ]   SI BT BT AB
    0017       cccccc6552       91       87        4            0.09        1.046 +- 0.110       0.956 +- 0.102 [10]   SI RE RE SC BT BT BT BT BT BT
    0018        7ccccccc2       70      100      -30            5.29        0.700 +- 0.084       1.429 +- 0.143 [9 ]   SI BT BT BT BT BT BT BT SD
    .             TOTALS:    11300    11300                    94.69     94.69/62 =  1.53   pvalue:P[C2>]:1.000  1-pvalue:P[C2<]:0.000 



Largest discrep visible, from "SI BT BT SA" which Opticks is doing less than Geant4::

    In [6]: b.sel = "SI BT BT SA"

    In [12]: b.dx.shape
    Out[12]: (173, 10, 2, 4)


Looking for the counterpart discrepancy the other way, see that its "SI BT BT BT AB" which Opticks is doing more::

    In [4]: np.set_printoptions(linewidth=100)

    In [5]: ab.his.ss[ab.his.c2 > 2]
    Out[5]: 
    array(['0011             8cc2      118      173      -55           10.40     SI BT BT SA',
           '0018        7ccccccc2       70      100      -30            5.29     SI BT BT BT BT BT BT BT SD',
           '0019       cccccccc62       77       59       18            2.38     SI SC BT BT BT BT BT BT BT BT',
           '0031       ccccc66652       35       54      -19            4.06     SI RE SC SC SC BT BT BT BT BT',
           '0042            4ccc2       54        8       46           34.13     SI BT BT BT AB',
           '0045       ccccc65552       21       36      -15            3.95     SI RE RE RE SC BT BT BT BT BT',
           '0059       cccbcccc52       24       13       11            3.27     SI RE BT BT BT BT BR BT BT BT'],
          dtype='|S98')

    In [6]: 


    In [7]: ab.his[:50]
    Out[7]: 
    ab.his
    .       seqhis_ana  cfo:sum  1:g4live:tds3gun   -1:g4live:tds3gun 
    .             TOTALS:    11300    11300                    94.69     94.69/62 =  1.53   pvalue:P[C2>]:1.000  1-pvalue:P[C2<]:0.000 
       n             iseq        a        b      a-b     (a-b)^2/(a+b)        a/b                  b/a          [ns]   label
    0000         7cccccc2     1832     1795       37            0.38        1.021 +- 0.024       0.980 +- 0.023 [8 ]   SI BT BT BT BT BT BT SD
    0001               42     1727     1683       44            0.57        1.026 +- 0.025       0.975 +- 0.024 [2 ]   SI AB
    0002        7cccccc62      766      713       53            1.90        1.074 +- 0.039       0.931 +- 0.035 [9 ]   SI SC BT BT BT BT BT BT SD
    0003        7cccccc52      522      518        4            0.02        1.008 +- 0.044       0.992 +- 0.044 [9 ]   SI RE BT BT BT BT BT BT SD
    0004              452      515      505       10            0.10        1.020 +- 0.045       0.981 +- 0.044 [3 ]   SI RE AB
    0005              462      379      377        2            0.01        1.005 +- 0.052       0.995 +- 0.051 [3 ]   SI SC AB
    0006               41      256      262       -6            0.07        0.977 +- 0.061       1.023 +- 0.063 [2 ]   CK AB
    0007       7cccccc662      246      249       -3            0.02        0.988 +- 0.063       1.012 +- 0.064 [10]   SI SC SC BT BT BT BT BT BT SD
    0008       7cccccc652      212      230      -18            0.73        0.922 +- 0.063       1.085 +- 0.072 [10]   SI RE SC BT BT BT BT BT BT SD
    0009       7cccccc552      167      168       -1            0.00        0.994 +- 0.077       1.006 +- 0.078 [10]   SI RE RE BT BT BT BT BT BT SD
    0010             4552      173      162       11            0.36        1.068 +- 0.081       0.936 +- 0.074 [4 ]   SI RE RE AB
    0011             8cc2    **118      173      -55           10.40**      0.682 +- 0.063       1.466 +- 0.111 [4 ]   SI BT BT SA
    0012             4652      119      120       -1            0.00        0.992 +- 0.091       1.008 +- 0.092 [4 ]   SI RE SC AB
    0013       cccccc6662      124      104       20            1.75        1.192 +- 0.107       0.839 +- 0.082 [10]   SI SC SC SC BT BT BT BT BT BT
    0014       cccccc6652      111      114       -3            0.04        0.974 +- 0.092       1.027 +- 0.096 [10]   SI RE SC SC BT BT BT BT BT BT
    0015             4662      100      110      -10            0.48        0.909 +- 0.091       1.100 +- 0.105 [4 ]   SI SC SC AB
    0016             4cc2      101       87       14            1.04        1.161 +- 0.116       0.861 +- 0.092 [4 ]   SI BT BT AB
    0017       cccccc6552       91       87        4            0.09        1.046 +- 0.110       0.956 +- 0.102 [10]   SI RE RE SC BT BT BT BT BT BT
    0018        7ccccccc2       70      100      -30            5.29        0.700 +- 0.084       1.429 +- 0.143 [9 ]   SI BT BT BT BT BT BT BT SD
    0019       cccccccc62       77       59       18            2.38        1.305 +- 0.149       0.766 +- 0.100 [10]   SI SC BT BT BT BT BT BT BT BT
    0020       ccccbcccc2       56       67      -11            0.98        0.836 +- 0.112       1.196 +- 0.146 [10]   SI BT BT BT BT BR BT BT BT BT
    0021       cccccc5552       63       58        5            0.21        1.086 +- 0.137       0.921 +- 0.121 [10]   SI RE RE RE BT BT BT BT BT BT
    0022       7cccccc562       59       57        2            0.03        1.035 +- 0.135       0.966 +- 0.128 [10]   SI SC RE BT BT BT BT BT BT SD
    0023       ccccccc662       52       59       -7            0.44        0.881 +- 0.122       1.135 +- 0.148 [10]   SI SC SC BT BT BT BT BT BT BT
    0024           7cccc2       53       56       -3            0.08        0.946 +- 0.130       1.057 +- 0.141 [6 ]   SI BT BT BT BT SD
    0025           8c9cc2       49       56       -7            0.47        0.875 +- 0.125       1.143 +- 0.153 [6 ]   SI BT BT DR BT SA
    0026            8ccc2       49       51       -2            0.04        0.961 +- 0.137       1.041 +- 0.146 [5 ]   SI BT BT BT SA
    0027            4cc62       42       54      -12            1.50        0.778 +- 0.120       1.286 +- 0.175 [5 ]   SI SC BT BT AB
    0028              4c2       40       53      -13            1.82        0.755 +- 0.119       1.325 +- 0.182 [3 ]   SI BT AB
    0029       7ccccccc62       45       47       -2            0.04        0.957 +- 0.143       1.044 +- 0.152 [10]   SI SC BT BT BT BT BT BT BT SD
    0030             4562       39       52      -13            1.86        0.750 +- 0.120       1.333 +- 0.185 [4 ]   SI SC RE AB
    0031       ccccc66652       35       54      -19            4.06        0.648 +- 0.110       1.543 +- 0.210 [10]   SI RE SC SC SC BT BT BT BT BT
    0032       ccccc66662       42       45       -3            0.10        0.933 +- 0.144       1.071 +- 0.160 [10]   SI SC SC SC SC BT BT BT BT BT
    0033        7cccccc51       40       45       -5            0.29        0.889 +- 0.141       1.125 +- 0.168 [9 ]   CK RE BT BT BT BT BT BT SD
    0034       ccccccc652       44       35        9            1.03        1.257 +- 0.190       0.795 +- 0.134 [10]   SI RE SC BT BT BT BT BT BT BT
    0035            45552       38       41       -3            0.11        0.927 +- 0.150       1.079 +- 0.169 [5 ]   SI RE RE RE AB
    0036            8cc62       34       44      -10            1.28        0.773 +- 0.133       1.294 +- 0.195 [5 ]   SI SC BT BT SA
    0037            46662       36       39       -3            0.12        0.923 +- 0.154       1.083 +- 0.173 [5 ]   SI SC SC SC AB
    0038              451       35       38       -3            0.12        0.921 +- 0.156       1.086 +- 0.176 [3 ]   CK RE AB
    0039            46552       31       41      -10            1.39        0.756 +- 0.136       1.323 +- 0.207 [5 ]   SI RE RE SC AB
    0040            8cc52       30       39       -9            1.17        0.769 +- 0.140       1.300 +- 0.208 [5 ]   SI RE BT BT SA
    0041            46652       37       29        8            0.97        1.276 +- 0.210       0.784 +- 0.146 [5 ]   SI RE SC SC AB
    0042            4ccc2     **54        8       46           34.13**      6.750 +- 0.919       0.148 +- 0.052 [5 ]   SI BT BT BT AB
    0043           8cccc2       32       27        5            0.42        1.185 +- 0.210       0.844 +- 0.162 [6 ]   SI BT BT BT BT SA
    0044            4cc52       30       29        1            0.02        1.034 +- 0.189       0.967 +- 0.180 [5 ]   SI RE BT BT AB
    0045       ccccc65552       21       36      -15            3.95        0.583 +- 0.127       1.714 +- 0.286 [10]   SI RE RE RE SC BT BT BT BT BT
    0046       ccccc66552       24       27       -3            0.18        0.889 +- 0.181       1.125 +- 0.217 [10]   SI RE RE SC SC BT BT BT BT BT
    0047       cccbcccc62       21       28       -7            1.00        0.750 +- 0.164       1.333 +- 0.252 [10]   SI SC BT BT BT BT BR BT BT BT
    0048       cccccc6562       24       25       -1            0.02        0.960 +- 0.196       1.042 +- 0.208 [10]   SI SC RE SC BT BT BT BT BT BT

    .             TOTALS:    11300    11300                    94.69     94.69/62 =  1.53   pvalue:P[C2>]:1.000  1-pvalue:P[C2<]:0.000 

    In [8]: 

        
Taking out that one pair of discrepancies would put the chi2/df below 1. 

Where are the below endpoints on the geometry::

   SI BT BT SA           ##   OK lack
   SI BT BT BT AB        ##   OK excess



Look for similar issues and tips for debugging techniques
-------------------------------------------------------------

::


    epsilon:issues blyth$ grep excess *.rst
    ab-shakedown-dec-2021.rst:   SI BT BT BT AB        ##   OK excess
    check_innerwater_bulk_absorb.rst:  * OK has "TO SA" sail to boundary excess of 989/100,000 (1%) 
    check_innerwater_bulk_absorb.rst:    ^^^^^^^^^^  G4 has excess of scatters that get back into LS ? ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ckm_cerenkov_generation_align.rst:The excessive bookeeping with lots of different paths above, motivated
    csg_complement.rst:* Translating DYB Near site geometry yields 22/249 excessively deep(greater that height 3) CSG trees
    g4ok-geometry-identity-caching.rst:and compute a hash of the file.  That is excessively slow for large geometries, also not 
    malloc.rst:         to 15 bytes at the end of an allocated block may be excess at the end of the page, and libgmalloc will not detect buffer overruns into that area by default.  This default
    ok_less_SA_more_AB.rst:* excess OK AB seems fixed
    ok_less_SA_more_AB.rst:                                               ^^^^^^ OK excess bulk AB
    ok_less_SA_more_AB.rst:                                                ^^^^^ OK excess bulk AB
    ok_less_SA_more_AB.rst:Pyrex ABSLEN is much shorter than water... this might explain the excess AB in the "Water" 
    ok_less_SA_more_AB.rst:Look at the excess AB in "Water"
    tds3gun_nonaligned_comparison.rst:And with tds3gun have removed the zeros, next issue looks to be excess AB in the water with OK::
    epsilon:issues blyth$ 


Some old debug tips from :doc:`ok_less_SA_more_AB`::

    In [7]: a.bn.view(np.int8).shape
    Out[7]: (11142, 1, 16)

    In [9]: als[10:11]
    Out[9]: SI BT BT SA

    In [10]: print(a.blib.format(a.bn[10]))
     18 : Acrylic///LS
     17 : Water///Acrylic
     16 : Tyvek//Implicit_RINDEX_NoRINDEX_pInnerWater_pCentralDetector/Water

    



After a reboot have to grab again as event in tmp folders::

    tds3gun.sh get   # grab from remote 


Select category with Opticks excess "SI BT BT BT AB"::


    tds3gun.sh 1     # load into python
    ...
    In [7]: ab.his[:50]
    ...
    0037            46662       36       39       -3            0.12        0.923 +- 0.154       1.083 +- 0.173 [5 ]   SI SC SC SC AB
    0038              451       35       38       -3            0.12        0.921 +- 0.156       1.086 +- 0.176 [3 ]   CK RE AB
    0039            46552       31       41      -10            1.39        0.756 +- 0.136       1.323 +- 0.207 [5 ]   SI RE RE SC AB
    0040            8cc52       30       39       -9            1.17        0.769 +- 0.140       1.300 +- 0.208 [5 ]   SI RE BT BT SA
    0041            46652       37       29        8            0.97        1.276 +- 0.210       0.784 +- 0.146 [5 ]   SI RE SC SC AB
    0042            4ccc2       54        8       46           34.13        6.750 +- 0.919       0.148 +- 0.052 [5 ]   SI BT BT BT AB
    0043           8cccc2       32       27        5            0.42        1.185 +- 0.210       0.844 +- 0.162 [6 ]   SI BT BT BT BT SA
    0044            4cc52       30       29        1            0.02        1.034 +- 0.189       0.967 +- 0.180 [5 ]   SI RE BT BT AB
    0045       ccccc65552       21       36      -15            3.95        0.583 +- 0.127       1.714 +- 0.286 [10]   SI RE RE RE SC BT BT BT BT BT
    0046       ccccc66552       24       27       -3            0.18        0.889 +- 0.181       1.125 +- 0.217 [10]   SI RE RE SC SC BT BT BT BT BT
    0047       cccbcccc62       21       28       -7            1.00        0.750 +- 0.164       1.333 +- 0.252 [10]   SI SC BT BT BT BT BR BT BT BT
    0048       cccccc6562       24       25       -1            0.02        0.960 +- 0.196       1.042 +- 0.208 [10]   SI SC RE SC BT BT BT BT BT BT
    .             TOTALS:    11300    11300                    94.69     94.69/62 =  1.53   pvalue:P[C2>]:1.000  1-pvalue:P[C2<]:0.000 


    In [2]: 54./11300.                                                                                                                                                                                        
    Out[2]: 0.004778761061946903



    In [8]: a.sel = "SI BT BT BT AB"

    In [9]: a.bn.shape
    Out[9]: (54, 1, 4)


    In [13]: a.bn.view(np.int8).shape
    Out[13]: (54, 1, 16)

    In [14]: a.bn.view(np.int8)
    Out[14]: 
    A([[[ 18, -17,  17,  24,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]],

       [[ 18, -17, -24,  24,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]],

       [[ 18, -17, -24,  24,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]],

       [[ 18, -17, -24,  24,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]],

       [[ 18, -17, -24,  24,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]],

       [[ 18, -17, -24,  24,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]],

       [[ 18,  17, -24,  24,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]],

       [[ 18, -17, -24,  24,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]],

       [[ 18, -17, -24,  24,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]],

       [[ 18, -17, -24,  24,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]],

       [[ 18,  17, -24,  24,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]],

       [[ 18, -17, -24,  24,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]],


    In [24]: a.blib.bname(17-1)
    Out[24]: 'Water///Acrylic'

    In [25]: a.blib.bname(18-1)
    Out[25]: 'Acrylic///LS'

    In [26]: a.blib.bname(17-1)
    Out[26]: 'Water///Acrylic'

    In [27]: a.blib.bname(24-1)
    Out[27]: 'Water///PE_PA'


    In [41]: print(a.blib.format(list(map(int,a.bn.view(np.int8)[0][0]))))
     18 : Acrylic///LS
    -17 : Water///Acrylic
     17 : Water///Acrylic
     24 : Water///PE_PA

    In [42]: print(a.blib.format(list(map(int,a.bn.view(np.int8)[1][0]))))
     18 : Acrylic///LS
    -17 : Water///Acrylic
    -24 : Water///PE_PA
     24 : Water///PE_PA


::

    epsilon:issues blyth$ jgr PE_PA
    ...
    ./Simulation/DetSimV2/CentralDetector/include/XJfixtureConstruction.hh:    G4Material* PE_PA ;
    ./Simulation/DetSimV2/CentralDetector/src/XJfixtureConstruction.cc:    PE_PA = G4Material::GetMaterial("PE_PA");
    ./Simulation/DetSimV2/CentralDetector/src/XJfixtureConstruction.cc:		PE_PA,


PE_PA seems to only be used with XJfixtureConstruction so take a look at that geometry for Opticks/Geant4 discrepancies.


1. add X4SolidMaker::XJfixtureConstruction grabbing the definition from "jcv XJfixtureConstruction"
2. look at the mesh from Geant4 polygonization::

    x4 ; GEOM=XJfixtureConstruction EYE=1,1,1 ./X4MeshTest.sh      # looks like some piece of church architecture/christian symbol 

3. convert G4VSolid via GeoChain into CSGFoundry::

    gc ; GEOM=XJfixtureConstruction ./run.sh 

4. render the CSGFoundry::

    cx ; GEOM=XJfixtureConstruction EYE=1,1,1 TMIN=0.1 ZOOM=2 ./cxr_geochain.sh

    * and voila : another presumably coincidence issue, the upper surface is not subtracted 



Two solids with the same name::

    N[blyth@localhost CSGFoundry]$ grep XJ meshname.txt 
    solidXJfixture
    solidXJanchor
    solidXJfixture
    N[blyth@localhost CSGFoundry]$ 

Not sure if the MOI distinction via ordinal is working as the renders look the same
Actually I think that name duplication is because the CSG is complex and was balanced resulting 
in both balanced and unbalanced CSG trees being written.::

    cx ; MOI=XJfixtureConstruction:0 ./cxr_view.sh
    cx ; MOI=XJfixtureConstruction:1 ./cxr_view.sh

    cx ; ./cxr_grab.sh jpg


    cx ; MOI=solidXJfixture:1 ../bin/flight7.sh 
    cx ; ./cxr_grab.sh mp4

    cx ; MOI=solidXJfixture:1 SCALE0=20 ../bin/flight7.sh     ## HUH SCALE0 seems to make no difference
    cx ; ./cxr_grab.sh mp4


The rendered jpg and mp4 show a coincidence problem but also look like an 
overlap of volumes. There is a cross piece that is not in solidXJfixture.

The SCALE0 control seems to make no difference. 

View 3D positions of the excess OK points::

    In [3]: import pyvista as pv
    In [4]: pl = pv.Plotter(window_size=2*np.array([1280, 720]))
    In [5]: a.sel = "SI BT BT BT AB"
    In [6]: pos = a.ox[:,0,:3]
    In [7]: pos.shape
    Out[7]: (54, 3)
    In [8]: pl.add_points(pos)
    Out[8]: (vtkRenderingOpenGL2Python.vtkOpenGLActor)0x168ba0c
    In [9]: cp = pl.show()


::

    epsilon:GNodeLib blyth$ grep fixture all_volume_LVNames.txt
    lXJfixture0x592bc30
    lXJfixture0x592bc30
    lXJfixture0x592bc30
    lXJfixture0x592bc30
    ...

    epsilon:GNodeLib blyth$ grep fixture all_volume_LVNames.txt | wc -l 
          64

    epsilon:GNodeLib blyth$  grep fixture all_volume_PVNames.txt
    lXJfixture_phys0x59354c0
    lXJfixture_phys0x5937900
    lXJfixture_phys0x59379e0
    lXJfixture_phys0x5937ac0
    lXJfixture_phys0x5935640
    ...

    epsilon:GNodeLib blyth$ grep fixture all_volume_PVNames.txt | wc -l 
          64


Look at counts for unique LVNames::

    epsilon:ana blyth$ ./lvn.sh 
     lvn.shape (336648,)  path /usr/local/opticks/geocache/OKX4Test_lWorld0x574e7f0_PV_g4live/g4ok_gltf/f65f5cd1a197e3a0c9fe55975ff2c7a7/1/GNodeLib/all_volume_LVNames.txt 
         0 :     30 :                           GLb1.bt02_HBeam0x57bbe70 
         1 :     30 :                           GLb1.bt05_HBeam0x57c8f80 
         2 :     30 :                           GLb1.bt06_HBeam0x57cb530 
         3 :     30 :                           GLb1.bt07_HBeam0x57cdae0 
         4 :     30 :                           GLb1.bt08_HBeam0x57d0090 
        ... 
        55 :     30 :                          GZ1.B05_06_HBeam0x57a9b90 
        56 :     30 :                          GZ1.B06_07_HBeam0x57ac170 
        57 :   5000 :       HamamatsuR12860_PMT_20inch_body_log0x5f1bef0 
        58 :   5000 :     HamamatsuR12860_PMT_20inch_inner1_log0x5f19570 
        59 :   5000 :     HamamatsuR12860_PMT_20inch_inner2_log0x5f19870 
        60 :   5000 :            HamamatsuR12860_PMT_20inch_log0x5f1b1b0 
        61 :   5000 :                      HamamatsuR12860lMask0x5f1df00 
        62 :   5000 :                  HamamatsuR12860lMaskTail0x5f1f370 
        63 :   5000 :               HamamatsuR12860lMaskVirtual0x5f1d410 
        64 :  12612 :            NNVTMCPPMT_PMT_20inch_body_log0x5f2b890 
        65 :  12612 :          NNVTMCPPMT_PMT_20inch_inner1_log0x5f2bce0 
        66 :  12612 :          NNVTMCPPMT_PMT_20inch_inner2_log0x5f2c520 
        67 :  12612 :                 NNVTMCPPMT_PMT_20inch_log0x5f2bc30 
        68 :  12612 :                           NNVTMCPPMTlMask0x5f2dd90 
        69 :  12612 :                       NNVTMCPPMTlMaskTail0x5f2f0d0 
        70 :  12612 :                    NNVTMCPPMTlMaskVirtual0x5f2d2f0 
        71 :   2400 :                  PMT_20inch_veto_body_log0x5f348f0 
        72 :   2400 :                PMT_20inch_veto_inner1_log0x5f34ab0 
        73 :   2400 :                PMT_20inch_veto_inner2_log0x5f34bd0 
        74 :   2400 :                       PMT_20inch_veto_log0x5f349d0 
        75 :  25600 :                        PMT_3inch_body_log0x66b3630 
       ...
       125 :      1 :                        lUpperChimneySteel0x7171300 
       126 :      1 :                        lUpperChimneyTyvek0x71713f0 
       127 :     63 :                                  lWallff_0x71761e0 
       128 :      1 :                                    lWorld0x574e7f0 
       129 :     56 :                                 lXJanchor0x58ff9c0 
       130 :     64 :                                lXJfixture0x592bc30 
       131 :   2400 :                 mask_PMT_20inch_vetolMask0x5f31260 
       132 :   2400 :          mask_PMT_20inch_vetolMaskVirtual0x5f306f0 

    In [1]:                                                                                                                                                                                                  
   
::
     
    epsilon:GNodeLib blyth$ jgr lXJanchor
    ./Simulation/DetSimV2/CentralDetector/src/XJanchorConstruction.cc:    if(motherName == "lXJanchor")
    ./Simulation/DetSimV2/CentralDetector/src/XJanchorConstruction.cc:		"lXJanchor",
    epsilon:offline blyth$ 


::    

    epsilon:extg4 blyth$ MOI=solidXJfixture:0,solidXJfixture:1,solidXJfixture:2,solidXJfixture:3,solidXJfixture:4,solidXJfixture:5,solidXJfixture:6 CSGTargetTest 
    2021-12-28 19:39:33.590 INFO  [429671] [main@38] cfbase /usr/local/opticks/geocache/OKX4Test_lWorld0x574e7f0_PV_g4live/g4ok_gltf/f65f5cd1a197e3a0c9fe55975ff2c7a7/1/CSG_GGeo
    2021-12-28 19:39:33.590 INFO  [429671] [CSGFoundry::load@1150] /usr/local/opticks/geocache/OKX4Test_lWorld0x574e7f0_PV_g4live/g4ok_gltf/f65f5cd1a197e3a0c9fe55975ff2c7a7/1/CSG_GGeo/CSGFoundry
    2021-12-28 19:39:33.590 INFO  [429671] [CSGFoundry::loadArray@1214]  ni    10 nj 3 nk 4 solid.npy
    2021-12-28 19:39:33.591 INFO  [429671] [CSGFoundry::loadArray@1214]  ni  3243 nj 4 nk 4 prim.npy
    2021-12-28 19:39:33.597 INFO  [429671] [CSGFoundry::loadArray@1214]  ni 17671 nj 4 nk 4 node.npy
    2021-12-28 19:39:33.600 INFO  [429671] [CSGFoundry::loadArray@1214]  ni  8205 nj 4 nk 4 tran.npy
    2021-12-28 19:39:33.603 INFO  [429671] [CSGFoundry::loadArray@1214]  ni  8205 nj 4 nk 4 itra.npy
    2021-12-28 19:39:33.617 INFO  [429671] [CSGFoundry::loadArray@1214]  ni 48477 nj 4 nk 4 inst.npy
    2021-12-28 19:39:33.655 INFO  [429671] [main@41] foundry CSGFoundry  total solids 10 STANDARD 10 ONE_PRIM 0 ONE_NODE 0 DEEP_COPY 0 KLUDGE_BBOX 0 num_prim 3243 num_node 17671 num_plan 0 num_tran 8205 num_itra 8205 num_inst 48477 ins 0 gas 0 ias 0 meshname 136 mmlabel 0
    2021-12-28 19:39:33.656 INFO  [429671] [main@49]  MOI solidXJfixture:0,solidXJfixture:1,solidXJfixture:2,solidXJfixture:3,solidXJfixture:4,solidXJfixture:5,solidXJfixture:6 vmoi.size 7
     moi solidXJfixture:0 midx    88 mord     0 iidx      0 name solidXJfixture0x592ba10
     moi solidXJfixture:1 midx    88 mord     1 iidx      0 name solidXJfixture0x592ba10
     moi solidXJfixture:2 midx    88 mord     2 iidx      0 name solidXJfixture0x592ba10
     moi solidXJfixture:3 midx    88 mord     3 iidx      0 name solidXJfixture0x592ba10
     moi solidXJfixture:4 midx    88 mord     4 iidx      0 name solidXJfixture0x592ba10
     moi solidXJfixture:5 midx    88 mord     5 iidx      0 name solidXJfixture0x592ba10
     moi solidXJfixture:6 midx    88 mord     6 iidx      0 name solidXJfixture0x592ba10
     moi solidXJfixture:0 ce ( 0.000, 0.000,17696.938,63.792) 
     moi solidXJfixture:1 ce ( 0.000, 0.000,17696.938,63.792)        ## huh looks like two on top of each other 
     moi solidXJfixture:2 ce (12989.057,7500.323,9370.795,65.142) 
     moi solidXJfixture:3 ce ( 0.000,14996.406,9370.795,65.000) 
     moi solidXJfixture:4 ce (-12989.057,7500.323,9370.795,65.142) 
     moi solidXJfixture:5 ce (-12989.057,-7500.323,9370.795,65.142) 
     moi solidXJfixture:6 ce ( 0.000,-14996.406,9370.795,65.000) 
     moi solidXJfixture:0 q0 ( 1.000, 0.000, 0.000, 0.000) ( 0.000, 1.000, 0.000, 0.000) ( 0.000, 0.000, 1.000, 0.000) ( 0.000, 0.000, 0.000, 1.000) 
     moi solidXJfixture:1 q0 ( 1.000, 0.000, 0.000, 0.000) ( 0.000, 1.000, 0.000, 0.000) ( 0.000, 0.000, 1.000, 0.000) ( 0.000, 0.000, 0.000, 1.000) 
     moi solidXJfixture:2 q0 ( 1.000, 0.000, 0.000, 0.000) ( 0.000, 1.000, 0.000, 0.000) ( 0.000, 0.000, 1.000, 0.000) ( 0.000, 0.000, 0.000, 1.000) 
     moi solidXJfixture:3 q0 ( 1.000, 0.000, 0.000, 0.000) ( 0.000, 1.000, 0.000, 0.000) ( 0.000, 0.000, 1.000, 0.000) ( 0.000, 0.000, 0.000, 1.000) 
     moi solidXJfixture:4 q0 ( 1.000, 0.000, 0.000, 0.000) ( 0.000, 1.000, 0.000, 0.000) ( 0.000, 0.000, 1.000, 0.000) ( 0.000, 0.000, 0.000, 1.000) 
     moi solidXJfixture:5 q0 ( 1.000, 0.000, 0.000, 0.000) ( 0.000, 1.000, 0.000, 0.000) ( 0.000, 0.000, 1.000, 0.000) ( 0.000, 0.000, 0.000, 1.000) 
     moi solidXJfixture:6 q0 ( 1.000, 0.000, 0.000, 0.000) ( 0.000, 1.000, 0.000, 0.000) ( 0.000, 0.000, 1.000, 0.000) ( 0.000, 0.000, 0.000, 1.000) 
    epsilon:extg4 blyth$ 




solidXJfixture geometry deep dive
-------------------------------------

::

    epsilon:opticks blyth$ git add . 
    epsilon:opticks blyth$ git commit -m "examine solidXJfixture geometry for potential coincidence issues, note that first impression of issue with Tubs was because of stray --x4tubsnudgeskip 0 in GeoChain run.sh, there is however coincident internal union face between the celtic-cross and the altar that is not showing spurious intersects in xxs.sh testing, but may do so in cxs.sh "
    [master a1ed6e6c1] examine solidXJfixture geometry for potential coincidence issues, note that first impression of issue with Tubs was because of stray --x4tubsnudgeskip 0 in GeoChain run.sh, there is however coincident internal union face between the celtic-cross and the altar that is not showing spurious intersects in xxs.sh testing, but may do so in cxs.sh
     9 files changed, 178 insertions(+), 28 deletions(-)
     create mode 100644 extg4/XJfixtureConstruction.sh
    epsilon:opticks blyth$ git push 



Hmm GeoChain was "--x4tubsnudgeskip 0" skipping inner nudge which was causing the coincidence at the top of the tubs::

    171 opts=""
    172 opts="$opts --x4tubsnudgeskip 0"
    173 #opts="$opts --skipsolidname ${GEOM}_body_solid_1_9   " 
    174 


::

    2021-12-29 10:34:14.569 INFO  [722870] [*X4Solid::convertTubs_cylinder@682]  rmin 25 rmax 45 hz 6.5 has_inner 1 do_nudge_inner 0
    2021-12-29 10:34:14.569 INFO  [722870] [X4Solid::convertTubs@779]  has_deltaPhi 0 pick_disc 0 deltaPhi_segment_enabled 1 is_x4tubsnudgeskip 1 do_nudge_inner 0
    2021-12-29 10:34:14.569 INFO  [722870] [X4Solid::init@199] ]


Removing that and the x4 xxs cross section render looks OK::

    x4 ; ./xxs.sh 

Checking the geometry note that the z-underface of the "celtic-cross" is coincident with the z-upperface for the altar.
But that seems to not be causing spurious intersects in xxs standalone testing (but that is Geant4 intersection 
with the luxury of double precision, still potential for cxs spurious intersects on that inner face). 

 Need to make some more insitu cxs.sh and cxr_view.sh  






cxr_view.sh shows can select between the global 64 with the mesh-ordinal "mord"
-----------------------------------------------------------------------------------

::

    cx
    MOI=solidXJfixture:10 ./cxr_view.sh 

    cx
    ./cxr_grab.sh jpg 




Get scale to work for flight7.sh mp4
----------------------------------------

::


    cx
    MOI=solidXJfixture:10 ../bin/flight7.sh 
    ./cxr_grab.sh mp4

    MOI=solidXJfixture:20 ../bin/flight7.sh 
    ./cxr_grab.sh mp4

    MOI=solidXJfixture:20 FlightPath_scale=10 ../bin/flight7.sh 
    ./cxr_grab.sh mp4

    MOI=solidXJfixture:20 FlightPath_scale=4 PERIOD=8 ../bin/flight7.sh  


cxr_arglist
-------------

Note that with repeated global geometry that 
is currently not instanced such as solidXJfixture
the apparent viewpoints are all over the place despite a fixed
eye, look, up because there is no instance transform
dedicated to the target geometry instead there is
only the global identity transform.  

This may explain the issue with cxs cross sections of 
global repeated non-instanced geometry. 



cxs for instanced
-------------------

Check for instanced, it works as expected with virtual hatboxes present::

    cx
    GEOM=Hama_1 ./cxs.sh
    cx
    ./cxs_grab.sh png 


With ce_offset true in SEvent get no intersects and a grid not 
in local frame. Using both ce_offset and the instance transform
is kinda the same info in two different ways for instanced geom.



cxs for global repeated XJfixtureConstruction_0 getting a blank 
------------------------------------------------------------------------------

::

    cx
    GEOM=XJfixtureConstruction_0 ./cxs.sh 

    cx
    ./cxs_grab.sh png 


Hmm the gensteps might be in totally the wrong place::

    2021-12-30 00:18:34.289 INFO  [100944] [SBT::checkHitgroup@819]  num_sbt (sbt.hitgroupRecordCount) 3240 num_solid 10 num_prim 3240
    2021-12-30 00:18:34.289 INFO  [100944] [SBT::createGeom@101] ]
    2021-12-30 00:18:34.289 INFO  [100944] [SBT::getAS@523]  spec i0 c i idx 0
    2021-12-30 00:18:34.289 INFO  [100944] [main@128]  moi solidXJfixture:10 midx 88 mord 10 iidx 0
    2021-12-30 00:18:34.290 INFO  [100944] [main@141]  rc 0 MOI.ce (-17336 -4160.73 -809.117 66.0447)
    2021-12-30 00:18:34.290 INFO  [100944] [main@144] 
    qt( 1.000, 0.000, 0.000, 0.000) ( 0.000, 1.000, 0.000, 0.000) ( 0.000, 0.000, 1.000, 0.000) ( 0.000, 0.000, 0.000, 1.000) 
    2021-12-30 00:18:34.290 INFO  [100944] [SEvent::StandardizeCEGS@93]  CXS_CEGS  ix0 ix1 0 0 iy0 iy1 -16 16 iz0 iz1 -9 9 photons_per_genstep 100
    2021-12-30 00:18:34.290 INFO  [100944] [SEvent::StandardizeCEGS@108]  CXS_CEGS  x0      0.000 x1      0.000 y0    -52.836 y1     52.836 z0    -29.720 z1     29.720 photons_per_genstep 100 gridscale      0.050 ce.w(extent)     66.045
    2021-12-30 00:18:34.294 INFO  [100944] [CSGOptiX::setCE@257]  ce [ -17336 -4160.73 -809.117 66.0447] tmin_model 0.1 tmin 6.60447
    2021-12-30 00:18:34.294 INFO  [100944] [Composition::setNear@2424]  intended 6.60447 result 6.60447
    2021-12-30 00:18:34.294 INFO  [100944] [QEvent::setGensteps@55]  num_gs 627
    //QSeed_create_photon_seeds 
    2021-12-30 00:18:34.295 INFO  [100944] [CSGOptiX::prepareSimulateParam@208] [



Yes, all near origin. The SEvent was not using ce.xyz::


    In [1]: a = np.load("/Users/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/3dbec4dc3bdef47884fe48af781a179d/1/CSG_GGeo/CSGOptiXSimulateTest/cvd0/70000/XJfixtureConstruction_0/genstep.npy")


    In [3]: a.shape
    Out[3]: (627, 6, 4)

    In [4]: a[:,5]
    Out[4]: 
    array([[  0.   , -52.836, -29.72 ,   1.   ],
           [  0.   , -52.836, -26.418,   1.   ],
           [  0.   , -52.836, -23.116,   1.   ],
           ...,
           [  0.   ,  52.836,  23.116,   1.   ],
           [  0.   ,  52.836,  26.418,   1.   ],
           [  0.   ,  52.836,  29.72 ,   1.   ]], dtype=float32)

    In [5]: a[:100,5]
    Out[5]: 
    array([[  0.   , -52.836, -29.72 ,   1.   ],
           [  0.   , -52.836, -26.418,   1.   ],
           [  0.   , -52.836, -23.116,   1.   ],
           [  0.   , -52.836, -19.813,   1.   ],
           [  0.   , -52.836, -16.511,   1.   ],
           [  0.   , -52.836, -13.209,   1.   ],
           [  0.   , -52.836,  -9.907,   1.   ],
           [  0.   , -52.836,  -6.604,   1.   ],
           [  0.   , -52.836,  -3.302,   1.   ],
           [  0.   , -52.836,   0.   ,   1.   ],




review cxs cegs generation
---------------------------

Is this qudarap/qsim.h whats doing the generation? It only handles XZ::


    670 template <typename T>
    671 inline QSIM_METHOD void qsim<T>::generate_photon_torch(quad4& p, curandStateXORWOW& rng, const quad6& gs, unsigned photon_id, unsigned genstep_id )
    672 {   
    673     p.q0.f = gs.q1.f ;  // start with local frame position, eg (0,0,0)   
    674     
    675     float u = curand_uniform(&rng);
    676     float sinPhi, cosPhi;
    677     sincosf(2.f*M_PIf*u,&sinPhi,&cosPhi);
    678     
    679     //  local frame XZ plane directions
    680     p.q1.f.x = cosPhi ;  p.q1.f.y = 0.f    ;  p.q1.f.z = sinPhi ;  p.q1.f.w = 0.f ;
    681 
    682     qat4 qt(gs) ; // copy 4x4 transform from last 4 quads of genstep 
    683     qt.right_multiply_inplace( p.q0.f, 1.f );   // position 
    684     qt.right_multiply_inplace( p.q1.f, 0.f );   // direction 
    685 }   
    686 




