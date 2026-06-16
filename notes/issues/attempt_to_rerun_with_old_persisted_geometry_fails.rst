attempt_to_rerun_with_old_persisted_geometry_fails
===================================================

overview
---------

1. trying to use current stree.h code to read a two year old persisted geometry fails

   * HMM: SOME OF THE WARNINGS MAY BE BENIGN

2. instead recreate the geometry from the old GDML file, that works

   * YES: but the geometry misses sensors as SD status does not survive GDML
   * added "U4SensorIdentifierDefault__MODE=lax" to mint sensors just based on volume name, that has
     to be set during the GDML conversion



3. BUT: running (this means cxs_min.sh ?)  on the recreated geometry gives very different hits and lots of sensor_id -1 NOT_A_SENSOR
4. WIP: look into stree.h sensor_id reporting

   * potentially s_pmt.h changes need to be versioned to work with the old geometry




stree serialization changes prevent use of "J_2024aug27" CSGFoundry GEOM with current code
---------------------------------------------------------------------------------------------





::

    [lo] A[blyth@localhost ~]$ cxs_min.sh 
    /data1/blyth/local/opticks_Debug/bin/cxs_min.sh - Internal GEOM setup detected
    BASH_SOURCE                    : /data1/blyth/local/opticks_Debug/bin/cxs_min.sh 
    SDIR                           : /data1/blyth/local/opticks_Debug/bin 
    SDIR_NOTE                      : Think of SDIR as sibling-dir not source-dir as this script is used from release bin dir too 
    defarg                         : run_report_info 
    arg                            : run_report_info 
    allarg                         : info_env_fold_run_dbg_meta_report_grab_grep_gevt_du_pdb1_pdb0_AB_ana_pvcap_pvpub_mpcap_mppub 
    bin                            : CSGOptiXSMTest 
    script                         : /data1/blyth/local/opticks_Debug/bin/cxs_min.py 
    script_AB                      : /data1/blyth/local/opticks_Debug/bin/cxs_min_AB.py 
    script_lite                    : /data1/blyth/local/opticks_Debug/bin/cxs_min_lite.py 
    script_hlm                     : /data1/blyth/local/opticks_Debug/bin/cxs_min_hlm.py 
    GEOM                           : J_2024aug27 
    ...

    /data1/blyth/local/opticks_Debug/bin/cxs_min.sh : run : delete prior LOGFILE CSGOptiXSMTest.log
    2026-06-08 15:51:46.810  810023430 : [/data1/blyth/local/opticks_Debug/bin/cxs_min.sh 
    stree::ImportArray array is null, label[prim_nidx.npy]
    stree::ImportArray array is null, label[nidx_prim.npy]
    stree::ImportNames array is null, label[prname.txt]
    (typically this means the stree.h serialization code has changed compared to the version used for saving the tree)
    stree::ImportNames array is null, label[soname.txt]
    (typically this means the stree.h serialization code has changed compared to the version used for saving the tree)
    CSGOptiXSMTest: /home/blyth/opticks/sysrap/SPMT.h:619: void SPMT::init_total(): Assertion `pmtTotal->shape.size() == 1 && pmtTotal->shape[0] == SPMT_Total::FIELDS' failed.
    /data1/blyth/local/opticks_Debug/bin/cxs_min.sh: line 826: 674385 Aborted                 (core dumped) $bin
    /data1/blyth/local/opticks_Debug/bin/cxs_min.sh run error
    [lo] A[blyth@localhost ~]$ 


Try to convert old GDML file with current code using ~/o/g4cx/tests/G4CXOpticks_setGeometry_Test.sh
-----------------------------------------------------------------------------------------------------

Geometry folders standardly include origin GDML::

    0 lrwxrwxrwx.  1 blyth blyth    51 Aug 29  2024 J_2024aug27 -> /cvmfs/opticks.ihep.ac.cn/.opticks/GEOM/J_2024aug27
    A[blyth@localhost GEOM]$ ls J_2024aug27/
    CSGFoundry  origin.gdml  origin_gdxml_report.txt  origin_raw.gdml
    A[blyth@localhost GEOM]$ ls -alst J_2024aug27/
    total 41004
        1 drwxrwxr-x. 4 cvmfs cvmfs       33 Dec 15  2024 ..
        1 drwxrwxr-x. 3 cvmfs cvmfs      117 Sep 11  2024 .
        1 drwxr-xr-x. 3 cvmfs cvmfs      238 Sep 10  2024 CSGFoundry
    20501 -rw-rw-r--. 1 cvmfs cvmfs 20992027 Sep 10  2024 origin.gdml
        1 -rw-rw-r--. 1 cvmfs cvmfs      197 Sep 10  2024 origin_gdxml_report.txt
    20501 -rw-rw-r--. 1 cvmfs cvmfs 20992947 Sep 10  2024 origin_raw.gdml
    A[blyth@localhost GEOM]$ 
     


Using gxt

1. set ~/.opticks/GEOM/GEOM.sh  to export GEOM=J_2024aug27_recreated
2. create directory for the recreated geometry and copy GDML into it::

   mkdir ~/.opticks/GEOM/J_2024aug27_recreated
   cp ~/.opticks/GEOM/J_2024aug27/origin.gdml ~/.opticks/GEOM/J_2024aug27_recreated/


3. invoke the conversion after first using "info" to check GEOM is "J_2024aug27_recreated"::

   ~/o/g4cx/tests/G4CXOpticks_setGeometry_Test.sh info
   ~/o/g4cx/tests/G4CXOpticks_setGeometry_Test.sh info_run



FAILS : no surprise did not do GDML conversion in long time
---------------------------------------------------------------


::

    ~/o/g4cx/tests/G4CXOpticks_setGeometry_Test.sh info_dbg


    Reading symbols from G4CXOpticks_setGeometry_Test...
    Starting program: /data1/blyth/local/opticks_Debug/lib/G4CXOpticks_setGeometry_Test 
    [Thread debugging using libthread_db enabled]
    Using host libthread_db library "/lib64/libthread_db.so.1".
    [Detaching after vfork from child process 676438]
    2026-06-08 16:16:21.221 INFO  [676432] [main@16] [SetGeometry
    [New Thread 0x7fffe81ff000 (LWP 676439)]
    2026-06-08 16:16:21.324 FATAL [676432] [G4CXOpticks::setGeometry@233]  failed to setGeometry 
    G4CXOpticks_setGeometry_Test: /home/blyth/opticks/g4cx/G4CXOpticks.cc:234: void G4CXOpticks::setGeometry(): Assertion `0' failed.

    Thread 1 "G4CXOpticks_set" received signal SIGABRT, Aborted.
    (gdb) bt
    #0  0x00007ffff148bedc in __pthread_kill_implementation () from /lib64/libc.so.6
    #1  0x00007ffff143eb46 in raise () from /lib64/libc.so.6
    #2  0x00007ffff1428833 in abort () from /lib64/libc.so.6
    #3  0x00007ffff142875b in __assert_fail_base.cold () from /lib64/libc.so.6
    #4  0x00007ffff1437886 in __assert_fail () from /lib64/libc.so.6
    #5  0x00007ffff7ececd5 in G4CXOpticks::setGeometry (this=0x491360) at /home/blyth/opticks/g4cx/G4CXOpticks.cc:234
    #6  0x00007ffff7ecda22 in G4CXOpticks::SetGeometry () at /home/blyth/opticks/g4cx/G4CXOpticks.cc:62
    #7  0x00000000004038c9 in main (argc=1, argv=0x7fffffffb8c8) at /home/blyth/opticks/g4cx/tests/G4CXOpticks_setGeometry_Test.cc:17
    (gdb) f 6
    #6  0x00007ffff7ecda22 in G4CXOpticks::SetGeometry () at /home/blyth/opticks/g4cx/G4CXOpticks.cc:62
    62	    gx->setGeometry();
    (gdb) f 5
    #5  0x00007ffff7ececd5 in G4CXOpticks::setGeometry (this=0x491360) at /home/blyth/opticks/g4cx/G4CXOpticks.cc:234
    234	        assert(0);
    (gdb) 



Looks like missing gdml route::

    219 void G4CXOpticks::setGeometry()
    220 {
    221     if(spath::has_CFBaseFromGEOM())
    222     {
    223         LOG(LEVEL) << "[ CSGFoundry::Load " ;
    224         CSGFoundry* cf = CSGFoundry::Load() ;
    225         LOG(LEVEL) << "] CSGFoundry::Load " ;
    226 
    227         LOG(LEVEL) << "[ setGeometry(cf)  " ;
    228         setGeometry(cf);
    229         LOG(LEVEL) << "] setGeometry(cf)  " ;
    230     }
    231     else
    232     {
    233         LOG(fatal) << " failed to setGeometry " ;
    234         assert(0);
    235     }
    236 }


::

    Resolve_GDMLPathFromGEOM()
    {   
       local origin=$HOME/.opticks/GEOM/$GEOM/origin.gdml 
       if [ -f "$origin" ]; then 
            export ${GEOM}_GDMLPathFromGEOM=$origin
            echo $BASH_SOURCE : FOUND origin $origin
       else 
            echo $BASH_SOURCE : NOT-FOUND origin $origin
       fi  
    }


Restored that functionality by fallback to doing geometry conversion from gdml::

    222 void G4CXOpticks::setGeometry()
    223 {
    224     if(spath::has_CFBaseFromGEOM())
    225     {
    226         LOG(LEVEL) << "[ CSGFoundry::Load " ;
    227         CSGFoundry* cf = CSGFoundry::Load() ;
    228         LOG(LEVEL) << "] CSGFoundry::Load " ;
    229 
    230         LOG(LEVEL) << "[ setGeometry(cf)  " ;
    231         setGeometry(cf);
    232         LOG(LEVEL) << "] setGeometry(cf)  " ;
    233     }
    234     else if(spath::has_GDMLPathFromGEOM())
    235     {
    236         const char* gdmlpath = spath::GDMLPathFromGEOM();
    237         LOG(LEVEL) << " has_GDMLPathFromGEOM gdmlpath[" << ( gdmlpath ? gdmlpath : "-" ) << "]" ;
    238         setGeometry(gdmlpath);
    239     }
    240     else
    241     {
    242         LOG(fatal) << " failed to setGeometry " ;
    243         assert(0);
    244     }
    245 }
    246 



Issues with a recreated old geometry "J_2024aug27_recreated"
---------------------------------------------------------------


::

    //qsim::propagate_at_surface_CustomART pidx 9678791 lpmtid -1 : ERROR UNEXPECTED LPMTID : NAN_ABORT 
    //qsim::propagate_at_surface_CustomART pidx 9678793 lpmtid -1 : ERROR UNEXPECTED LPMTID : NAN_ABORT 
    //qsim::propagate_at_surface_CustomART pidx 9678794 lpmtid -1 : ERROR UNEXPECTED LPMTID : NAN_ABORT 
    //qsim::propagate_at_surface_CustomART pidx 9678797 lpmtid -1 : ERROR UNEXPECTED LPMTID : NAN_ABORT 
    //qsim::propagate_at_surface_CustomART pidx 9678804 lpmtid -1 : ERROR UNEXPECTED LPMTID : NAN_ABORT 
    //qsim::propagate_at_surface_CustomART pidx 9678805 lpmtid -1 : ERROR UNEXPECTED LPMTID : NAN_ABORT 
    //qsim::propagate_at_surface_CustomART pidx 9678811 lpmtid -1 : ERROR UNEXPECTED LPMTID : NAN_ABORT 
    //qsim::propagate_at_surface_CustomART pidx 9678722 lpmtid -1 : ERROR UNEXPECTED LPMTID : NAN_ABORT 


The old gdml geometry has old convention copyno, maybe ?::

    [lo] A[blyth@localhost J_2024aug27_recreated]$ grep copynumber origin.gdml | grep PMT_8inch
    [lo] A[blyth@localhost J_2024aug27_recreated]$ 



::

    1788     int lpmtid = ctx.prd->identity() - 1 ;  // identity comes from optixInstance.instanceId where 0 means not-a-sensor
    1789     const float lposcost = ctx.prd->lposcost() ;  // local frame intersect position cosine theta
    ...
    1812 
    1813     // formerly excluded Custom4 hits onto WP PMTs see ~/j/issues/jok-tds-mu-running-NOT-A-SENSOR-warnings.rst
    1814     //if(lpmtid < s_pmt::OFFSET_CD_LPMT || lpmtid >= s_pmt::OFFSET_WP_PMT_END )
    1815     //if(lpmtid < s_pmt::OFFSET_CD_LPMT || lpmtid >= s_pmt::OFFSET_WP_ATM_LPMT_END )
    1816     if(lpmtid < s_pmt::OFFSET_CD_LPMT || lpmtid >=   s_pmt::OFFSET_WP_WAL_PMT_END )
    1817     {
    1818         flag = NAN_ABORT ;
    1819 #if !defined(PRODUCTION) && defined(DEBUG_PIDX)
    1820         printf("//qsim::propagate_at_surface_CustomART pidx %7lld lpmtid %d : ERROR UNEXPECTED LPMTID : NAN_ABORT \n", ctx.pidx, lpmtid );
    1821 #endif
    1822         return BREAK ;
    1823     }
    1824 




Where is the lpmtid -1 coming from when running on old geometry ?
--------------------------------------------------------------------




::

     505     for(unsigned i=0 ; i < num_ias_inst ; i++)
     506     {
     507         const qat4& q = ias_inst[i] ;
     508         int ins_idx,  gasIdx, sensor_identifier, sensor_index ;
     509         q.getIdentity(ins_idx, gasIdx, sensor_identifier, sensor_index );
     510 
     511         unsigned instanceId = q.get_IAS_OptixInstance_instanceId() ;
     512         assert( int(instanceId) == sensor_identifier );
     513 
     514         bool instanceId_is_allowed = instanceId < properties->limitMaxInstanceId ;
     515         LOG_IF(fatal, !instanceId_is_allowed)
     516             << " instanceId " << instanceId
     517             << " sbt->properties->limitMaxInstanceId " << properties->limitMaxInstanceId
     518             << " instanceId_is_allowed " << ( instanceId_is_allowed ? "YES" : "NO " )
     519             ;
     520         assert( instanceId_is_allowed  ) ;
     521 
     522         OptixTraversableHandle handle = getGASHandle(gasIdx);
     523 
     524         bool found = gasIdx_sbtOffset.count(gasIdx) == 1 ;
     525         unsigned sbtOffset = found ? gasIdx_sbtOffset.at(gasIdx) : getOffset(gasIdx, prim_idx ) ;
     526         if(!found)
     527         {
     528             gasIdx_sbtOffset[gasIdx] = sbtOffset ;
     529             LOG(LEVEL)
     530                 << " i " << std::setw(7) << i
     531                 << " gasIdx " << std::setw(3) << gasIdx
     532                 << " sbtOffset " << std::setw(6) << sbtOffset
     533                 << " gasIdx_sbtOffset.size " << std::setw(3) << gasIdx_sbtOffset.size()
     534                 << " instanceId " << instanceId
     535                 ;
     536         }
     537 
     538         //unsigned visibilityMask = 255;  // cf SOPTIX_Scene::init_Instances
     539         unsigned visibilityMask = properties->visibilityMask(gasIdx);
     540 
     541         OptixInstance instance = {} ;
     542         q.copy_columns_3x4( instance.transform );
     543         instance.instanceId = instanceId ;
     544         instance.sbtOffset = sbtOffset ;
     545         instance.visibilityMask = visibilityMask ;
     546 


Q: Where does the sensor_identifier come from ?
------------------------------------------------


A: U4Tree uses U4SensorIdentifierDefault to set the sensor_identifiers using the copyNo
------------------------------------------------------------------------------------------

::

    108 inline int U4SensorIdentifierDefault::getInstanceIdentity( const G4VPhysicalVolume* instance_outer_pv ) const
    109 {
    110     const char* pvn = instance_outer_pv ? instance_outer_pv->GetName().c_str() : "-" ;
    111     bool has_PMT_pvn = strstr(pvn, "PMT") != nullptr  ;
    112 
    113     const G4PVPlacement* pvp = dynamic_cast<const G4PVPlacement*>(instance_outer_pv) ;
    114     int copyno = pvp ? pvp->GetCopyNo() : -1 ;
    115 
    116     std::vector<const G4VPhysicalVolume*> sdpv ;
    117     FindSD_r(sdpv, instance_outer_pv, 0 );
    118 
    119     unsigned num_sd = sdpv.size() ;
    120     bool is_sensor = num_sd > 0 && has_PMT_pvn  ;
    121 
    122     int identifier = is_sensor ? copyno : -1  ;
    123 
    124     //bool is_interesting_copyno = IsInterestingCopyNo(copyno) ;
    125     //bool dump = is_sensor && is_interesting_copyno ;
    126     //bool dump = false ;
    127     //bool dump = true ;
    128     //bool dump = num_sd > 0 ;
    129 
    130     if(level > 0) std::cout
    131         << "U4SensorIdentifierDefault::getIdentity"
    132         << " level " << level
    133         << " copyno " << copyno
    134         << " num_sd " << num_sd
    135         << " is_sensor " << is_sensor
    136         << " pvn " << ( pvn ? pvn : "-" )
    137         << " has_PMT_pvn " << ( has_PMT_pvn ? "YES" : "NO " )
    138         << " identifier " << identifier
    139         << std::endl
    140         ;
    141 
    142 
    143     return identifier ;
    144 }


Look into stree.h reporting of sensor info
---------------------------------------------

::

    1127             int nidx = outer[j] ;
    1128             const G4VPhysicalVolume* pv = get_pv_(nidx) ;
    1129             const char* pvn = pv->GetName().c_str() ;
    1130 
    1131             int sensor_id = sid->getInstanceIdentity(pv) ;
    1132             assert( sensor_id >= -1 );  // sensor_id:-1 signifies "not-a-sensor"
    1133 
    1134             int sensor_index = sensor_id > -1 ? st->sensor_count : -1 ;
    1135             int sensor_name = -1 ;
    1136 
    1137             if(sensor_id > -1 )
    1138             {
    1139                 st->sensor_count += 1 ;  // count over all factors
    1140                 fac.sensors += 1 ;       // count sensors for each factor
    1141                 sensor_name = suniquename::Add(pvn, st->sensor_name ) ;
    1142             }
    1143 
    1144             snode& nd = st->nds[nidx] ;
    1145             nd.sensor_id = sensor_id ;  // -1:not-a-sensor-at-this-juncture
    1146             nd.sensor_index = sensor_index ;
    1147             nd.sensor_name = sensor_name ;







nd.sensor_index are canonicalized into zero based index in preorder traveral order of all nodes
------------------------------------------------------------------------------------------------

::

    1564 inline void stree::reorderSensors()
    1565 {
    1566     if(level > 0) std::cout
    1567         << "[ stree::reorderSensors"
    1568         << std::endl
    1569         ;
    1570 
    1571     sensor_count = 0 ;
    1572     reorderSensors_r(0);
    1573 
    1574     if(level > 0) std::cout
    1575         << "] stree::reorderSensors"
    1576         << " sensor_count " << sensor_count
    1577         << std::endl
    1578         ;
    1579 
    1580     // change sensor_id vector by looping over
    1581     // all nodes collecting it when > -1
    1582     get_sensor_id(sensor_id);
    1583 
    1584     assert( sensor_count == sensor_id.size() );
    1585 }
    1586 
    1587 /**
    1588 stree::reorderSensors_r
    1589 ------------------------
    1590 
    1591 For nodes with sensor_id > -1 change the sensor_index
    1592 into a 0-based preorder traversal count index.
    1593 
    1594 **/
    1595 
    1596 inline void stree::reorderSensors_r(int nidx)
    1597 {
    1598     snode& nd = nds[nidx] ;
    1599 
    1600     if( nd.sensor_id > -1 )
    1601     {
    1602         nd.sensor_index = sensor_count ;
    1603         sensor_count += 1 ;
    1604     }
    1605 
    1606     std::vector<int> children ;
    1607     get_children(children, nidx);
    1608     for(unsigned i=0 ; i < children.size() ; i++) reorderSensors_r(children[i]);
    1609 }
    1610 




KLUDGE : AVOID BEING DROWNED IN -1
--------------------------------------


qudarap/qsim.h::

    1816     if(lpmtid == s_pmt::NOT_A_SENSOR )
    1817     {
    1818         flag = NAN_ABORT ;  // SEPARATELY HANDLE NOT_A_SENSOR -1 TO AVOID BEING SWAPPED IN OUTPUT
    1819         return BREAK ;
    1820     }
    1821     else if(lpmtid < s_pmt::OFFSET_CD_LPMT || lpmtid >=   s_pmt::OFFSET_WP_WAL_PMT_END )
    1822     {
    1823         flag = NAN_ABORT ;
    1824 #if !defined(PRODUCTION) && defined(DEBUG_PIDX)
    1825         printf("//qsim::propagate_at_surface_CustomART pidx %7lld lpmtid %d : ERROR UNEXPECTED LPMTID : NAN_ABORT \n", ctx.pidx, lpmtid );
    1826 #endif
    1827         return BREAK ;
    1828     }



Compare current geom with recreated old geom
-----------------------------------------------


::

    REPORT_GEOM=J26_1_1_opticks_Debug cxs_min.sh report
    REPORT_GEOM=J_2024aug27_recreated cxs_min.sh report



REPORT_GEOM=J26_1_1_opticks_Debug cxs_min.sh report
------------------------------------------------------

::

    -[NPFold::compare_subarray.a_subcount
    [NP::descTable_ (12, 2, )
                      genstep      hit
              //A000        1   213947 
              //A001        1   212942 
              //A002        1  2132801 
              //A003        1  4263188 
              //A004        1  6392926 
              //A005        1  8532222 
              //A006        1 10659210 
              //A007        1 12786810 
              //A008        1 14925978 
              //A009        1 17060820 
              //A010        1 19192829 
              //A011        1 21325696 
    num_timestamp 0 auto-offset from t0 0
              TOTAL:       12 117699369
    ]NP::descTable_ (12, 2, )
    -]NPFold::compare_subarray.a_subcount
    -[NPFold::compare_subarray.b_subcount
    -
    -]NPFold::compare_subarray.b_subcount
    -[NPFold::compare_subarray.a
    [NP::descTable_ (12, 13, )
                        SbOE0    SbOE1    SeOE0     tBOE     tsG3     tsG4     tsG5     tsG6     tsG7     tsG8     tPrL     tPoL     tEOE
              //A000        0      109   242488       12      166      334      347      363     1538    14482    14513   225079   242495 
              //A001        0       52   222679        6       89       89      106      116      955      959      967   203812   222684 
              //A002        0       46  2126711        6       81       81       98      108     1088     1092     1100  1972002  2126716 
              //A003        0       60  4271469        6       99       99      118      131     1271     1275     1284  3955263  4271476 
              //A004        0       78  6450629       11      125      125      156      175     1509     1514     1522  5981774  6450636 
              //A005        0       66  8657155        8      112      112      145      165     1661     1667     1675  8029542  8657162 
              //A006        0       60 10863259        7      106      106      139      160     1833     1838     1846 10087314 10863266 
              //A007        0       72 13124090        9      123      124      160      182     2021     2026     2036 12182885 13124097 
              //A008        0       60 15497381        7      106      106      140      161     2171     2176     2186 14294108 15497388 
              //A009        0       61 17661170        9      107      108      142      163     2331     2336     2345 16408395 17661177 
              //A010        0       61 19950610        8      106      107      142      163     2518     2523     2532 18532412 19950616 
              //A011        0       61 22234343        8      105      106      140      161     2655     2660     2669 20587465 22234353 
    num_timestamp 0 auto-offset from t0 0
              TOTAL:        0      786 121301984       97     1325     1497     1833     2048    21551    34548    34675 112460051 121302066




Hugely less hits when using the recreated geometry, also 3x faster
--------------------------------------------------------------------


REPORT_GEOM=J_2024aug27_recreated cxs_min.sh report::
      

    -[NPFold::compare_subarray.a_subcount
    [NP::descTable_ (12, 2, )
                      genstep      hit
              //A000        1    15857 
              //A001        1    15660 
              //A002        1   158742 
              //A003        1   316993 
              //A004        1   475661 
              //A005        1   635181 
              //A006        1   793012 
              //A007        1   952893 
              //A008        1  1110645 
              //A009        1  1269158 
              //A010        1  1430586 
              //A011        1  1587841 
    num_timestamp 0 auto-offset from t0 0
              TOTAL:       12  8762229
    ]NP::descTable_ (12, 2, )
    -]NPFold::compare_subarray.a_subcount
    -[NPFold::compare_subarray.b_subcount
    -
    -]NPFold::compare_subarray.b_subcount
    -[NPFold::compare_subarray.a
    [NP::descTable_ (12, 13, )
                        SbOE0    SbOE1    SeOE0     tBOE     tsG3     tsG4     tsG5     tsG6     tsG7     tsG8     tPrL     tPoL     tEOE
              //A000        0      113    94960       13      173      343      358      375     1513    14221    14264    92029    94967 
              //A001        0       49    78230        5       86       86      101      111      961      965      972    76012    78235 
              //A002        0       41   709275        6       77       77       90       99     1024     1028     1036   691783   709281 
              //A003        0       49  1411094       11       81       81       97      106     1244     1248     1256  1386867  1411100 
              //A004        0       47  2132101        5       79       79       95      105     1360     1364     1373  2088859  2132107 
              //A005        0       50  2831796        6       85       86      103      114     1600     1605     1613  2774761  2831802 
              //A006        0       57  3542734        7       97       97      115      126     1787     1791     1799  3477466  3542739 
              //A007        0       55  4274631        6       93       94      111      122     1948     1953     1961  4187909  4274637 
              //A008        0       47  5013306        5       80       80       95      106     2098     2102     2110  4914148  5013312 
              //A009        0       61  5749986        7      106      106      135      151     2307     2313     2323  5628018  5749992 
              //A010        0       52  6510912        6       97       98      116      130     2450     2454     2462  6379280  6510918 
              //A011        0       54  7277128        7       93       93      112      125     2668     2673     2682  7140004  7277135 
    num_timestamp 0 auto-offset from t0 0
              TOTAL:        0      675 39626153       84     1147     1320     1528     1670    20960    33717    33851 38837136 39626225



desc_sensor on the old geometry
---------------------------------


::

    TEST=desc_sensor ~/o/sysrap/tests/stree_load_test.sh 

    [lo] A[blyth@localhost tests]$ TEST=desc_sensor ~/o/sysrap/tests/stree_load_test.sh
                       BASH_SOURCE : /home/blyth/o/sysrap/tests/stree_load_test.sh 
                               opt : -DWITH_PLACEHOLDER -DWITH_CHILD 
                              GEOM : J_2024aug27 
                               CFB : J_2024aug27_CFBaseFromGEOM 
                              FOLD : /home/blyth/.opticks/GEOM/J_2024aug27/CSGFoundry/SSim/stree 
                               MOI :  
                              TEST : desc_sensor 
                           TMPFOLD : /data1/blyth/tmp/stree_load_test 
    stree::ImportArray array is null, label[prim_nidx.npy]
    stree::ImportArray array is null, label[nidx_prim.npy]
    stree::ImportNames array is null, label[prname.txt]
    (typically this means the stree.h serialization code has changed compared to the version used for saving the tree)
    [stree_load_test::init
    [stree::desc_id
     loaddir /home/blyth/.opticks/GEOM/J_2024aug27/CSGFoundry/SSim/stree
    ]stree::desc_id

    ]stree_load_test::init
    stree::desc_sensor
     sensor_id.size 45612
     sensor_count 45612
     num_sensor_name 4 (expect small number of unique names)
    sensor_name[
    PMT_3inch_log_phys
    pLPMT_NNVT_MCPPMT
    pLPMT_Hamamatsu_R12860
    mask_PMT_20inch_vetolMaskVirtual_phys
    ]
    [lo] A[blyth@localhost tests]$ 





::

    [lo] A[blyth@localhost issues]$ TEST=desc_named:sensor_nd  ~/o/sysrap/tests/stree_load_test.sh run
    stree::ImportArray array is null, label[prim_nidx.npy]
    stree::ImportArray array is null, label[nidx_prim.npy]
    stree::ImportNames array is null, label[prname.txt]
    (typically this means the stree.h serialization code has changed compared to the version used for saving the tree)
    [stree_load_test::init
    [stree::desc_id
     loaddir /home/blyth/.opticks/GEOM/J_2024aug27/CSGFoundry/SSim/stree
    ]stree::desc_id

    ]stree_load_test::init
    stree::desc_named key[sensor_nd]
    [stree::desc_sensor_nd
     edge            10
     num_nd          382197
     num_nd_sensor   45612
     num_sid         45612
     num_sensor_name 4
     nidx  70938 i      0 sensor_id      0 sensor_index      0 sensor_name      2 [pLPMT_Hamamatsu_R12860]
     nidx  70950 i      1 sensor_id      1 sensor_index      1 sensor_name      2 [pLPMT_Hamamatsu_R12860]
     nidx  70962 i      2 sensor_id      2 sensor_index      2 sensor_name      1 [pLPMT_NNVT_MCPPMT]
     nidx  70971 i      3 sensor_id      3 sensor_index      3 sensor_name      2 [pLPMT_Hamamatsu_R12860]
     nidx  70983 i      4 sensor_id      4 sensor_index      4 sensor_name      1 [pLPMT_NNVT_MCPPMT]
     nidx  70992 i      5 sensor_id      5 sensor_index      5 sensor_name      2 [pLPMT_Hamamatsu_R12860]
     nidx  71004 i      6 sensor_id      6 sensor_index      6 sensor_name      1 [pLPMT_NNVT_MCPPMT]
     nidx  71013 i      7 sensor_id      7 sensor_index      7 sensor_name      2 [pLPMT_Hamamatsu_R12860]
     nidx  71025 i      8 sensor_id      8 sensor_index      8 sensor_name      2 [pLPMT_Hamamatsu_R12860]
     nidx  71037 i      9 sensor_id      9 sensor_index      9 sensor_name      2 [pLPMT_Hamamatsu_R12860]
    ...
     nidx 244425 i  17611 sensor_id  17611 sensor_index  17611 sensor_name      2 [pLPMT_Hamamatsu_R12860]
     nidx 244437 i  17612 sensor_id  20000 sensor_index  17612 sensor_name      0 [PMT_3inch_log_phys]
     nidx 244442 i  17613 sensor_id  20001 sensor_index  17613 sensor_name      0 [PMT_3inch_log_phys]
     nidx 244447 i  17614 sensor_id  20002 sensor_index  17614 sensor_name      0 [PMT_3inch_log_phys]
     nidx 372432 i  43211 sensor_id  45599 sensor_index  43211 sensor_name      0 [PMT_3inch_log_phys]
     nidx 372597 i  43212 sensor_id  50000 sensor_index  43212 sensor_name      3 [mask_PMT_20inch_vetolMaskVirtual_phys]
     nidx 372601 i  43213 sensor_id  50001 sensor_index  43213 sensor_name      3 [mask_PMT_20inch_vetolMaskVirtual_phys]
     nidx 372605 i  43214 sensor_id  50002 sensor_index  43214 sensor_name      3 [mask_PMT_20inch_vetolMaskVirtual_phys]
     nidx 382161 i  45603 sensor_id  52391 sensor_index  45603 sensor_name      3 [mask_PMT_20inch_vetolMaskVirtual_phys]
     nidx 382165 i  45604 sensor_id  52392 sensor_index  45604 sensor_name      3 [mask_PMT_20inch_vetolMaskVirtual_phys]
     nidx 382169 i  45605 sensor_id  52393 sensor_index  45605 sensor_name      3 [mask_PMT_20inch_vetolMaskVirtual_phys]
     nidx 382173 i  45606 sensor_id  52394 sensor_index  45606 sensor_name      3 [mask_PMT_20inch_vetolMaskVirtual_phys]
     nidx 382177 i  45607 sensor_id  52395 sensor_index  45607 sensor_name      3 [mask_PMT_20inch_vetolMaskVirtual_phys]
     nidx 382181 i  45608 sensor_id  52396 sensor_index  45608 sensor_name      3 [mask_PMT_20inch_vetolMaskVirtual_phys]
     nidx 382185 i  45609 sensor_id  52397 sensor_index  45609 sensor_name      3 [mask_PMT_20inch_vetolMaskVirtual_phys]
     nidx 382189 i  45610 sensor_id  52398 sensor_index  45610 sensor_name      3 [mask_PMT_20inch_vetolMaskVirtual_phys]
     nidx 382193 i  45611 sensor_id  52399 sensor_index  45611 sensor_name      3 [mask_PMT_20inch_vetolMaskVirtual_phys]
    ]stree::desc_sensor_nd




Recreated lacks sensors
--------------------------

::

    [lo] A[blyth@localhost u4]$ TEST=desc_named:sensor_nd  ~/o/sysrap/tests/stree_load_test.sh run
    [stree_load_test::init
    [stree::desc_id
     loaddir /home/blyth/.opticks/GEOM/J_2024aug27_recreated/CSGFoundry/SSim/stree
    ]stree::desc_id

    ]stree_load_test::init
    stree::desc_named key[sensor_nd]
    [stree::desc_sensor_nd
     edge            10
     num_nd          382197
     num_nd_sensor   0
     num_sid         0
     num_sensor_name 0
    ]stree::desc_sensor_nd
    [lo] A[blyth@localhost u4]$ TEST=desc_named:sensor  ~/o/sysrap/tests/stree_load_test.sh run
    [stree_load_test::init
    [stree::desc_id
     loaddir /home/blyth/.opticks/GEOM/J_2024aug27_recreated/CSGFoundry/SSim/stree
    ]stree::desc_id

    ]stree_load_test::init
    stree::desc_named key[sensor]
    stree::desc_sensor
     sensor_id.size 0
     sensor_count 0
     num_sensor_name 0 (expect small number of unique names)
    sensor_name[
    ]



Try contemporary recreation
----------------------------

::

    A[blyth@localhost u4]$ mkdir ~/.opticks/GEOM/J26_1_1_opticks_Debug_recreated
    A[blyth@localhost u4]$ cp /home/blyth/junosw/InstallArea/blyth-revive-opticksMode-zero-fixing-cleanup-SIGSEGV/.opticks/GEOM/J26_1_1_opticks_Debug/origin.gdml ~/.opticks/GEOM/J26_1_1_opticks_Debug_recreated/
    A[blyth@localhost u4]$ 



HMM need special settings for EMF::

    [lo] A[blyth@localhost tests]$ ~/o/g4cx/tests/G4CXOpticks_setGeometry_Test.sh
    /home/blyth/o/g4cx/tests/G4CXOpticks_setGeometry_Test.sh : GEOM J26_1_1_opticks_Debug_recreated : no geomscript
                       BASH_SOURCE : /home/blyth/o/g4cx/tests/G4CXOpticks_setGeometry_Test.sh 
                            defarg : info_run_ana 
                               arg : info_run_ana 
                              SDIR :  
                              GEOM : J26_1_1_opticks_Debug_recreated 
                           savedir : /home/blyth/.opticks/GEOM/J26_1_1_opticks_Debug_recreated 
                              FOLD : /home/blyth/.opticks/GEOM/J26_1_1_opticks_Debug_recreated 
                               bin : G4CXOpticks_setGeometry_Test 
                        geomscript :  
                            script : G4CXOpticks_setGeometry_Test.py 
                            origin : /home/blyth/.opticks/GEOM/J26_1_1_opticks_Debug_recreated/origin.gdml 
    ./GXTestRunner.sh - use externaly set GEOM CFBaseFromGEOM
                    HOME : /home/blyth
                     PWD : /home/blyth/opticks/g4cx/tests
                    GEOM : J26_1_1_opticks_Debug_recreated
    J26_1_1_opticks_Debug_recreated_GDMLPathFromGEOM : /home/blyth/.opticks/GEOM/J26_1_1_opticks_Debug_recreated/origin.gdml
             BASH_SOURCE : ./GXTestRunner.sh
              EXECUTABLE : G4CXOpticks_setGeometry_Test
                    ARGS : 
    2026-06-09 17:28:50.796 INFO  [797049] [main@16] [SetGeometry
    G4GDML: Reading '/home/blyth/.opticks/GEOM/J26_1_1_opticks_Debug_recreated/origin.gdml'...
    G4GDML: Reading definitions...
    G4GDML: Reading materials...
    G4GDML: Reading solids...
    G4GDML: Reading structure...
    G4GDML: Reading setup...
    G4GDML: Reading '/home/blyth/.opticks/GEOM/J26_1_1_opticks_Debug_recreated/origin.gdml' done!
    U4GDML::read                   yielded chars :  cout      0 cerr      0 : set VERBOSE to see them 
    2026-06-09 17:28:51.967 INFO  [797049] [U4Tree::Create@225] [new U4Tree
    2026-06-09 17:28:51.984 INFO  [797049] [U4Tree::init@283] -initSid
    2026-06-09 17:28:51.984 INFO  [797049] [U4Tree::init@285] -initRayleigh
    2026-06-09 17:28:51.984 INFO  [797049] [U4Tree::init@287] -initMaterials
    2026-06-09 17:28:52.034 INFO  [797049] [U4Tree::init@289] -initMaterials_NoRINDEX
    2026-06-09 17:28:52.034 INFO  [797049] [U4Tree::init@292] -initScint
    2026-06-09 17:28:52.035 INFO  [797049] [U4Tree::init@295] -initSurfaces
    2026-06-09 17:28:52.041 INFO  [797049] [U4Tree::init@298] -initSolids
    U4Polycone::init FATAL geometry with unsupported phicut :  enable experimental support with envvar  [U4Polycone__ENABLE_PHICUT] G4Polycone.GetName [s_EMFcoil_holder_ring8_seg10xb28a990]
    G4CXOpticks_setGeometry_Test: /data1/blyth/local/opticks_Debug/include/U4/U4Polycone.h:347: void U4Polycone::init(): Assertion `0' failed.
    ./GXTestRunner.sh: line 51: 797049 Aborted                 (core dumped) $EXECUTABLE $@
    ./GXTestRunner.sh : FAIL from G4CXOpticks_setGeometry_Test
    /home/blyth/o/g4cx/tests/G4CXOpticks_setGeometry_Test.sh : run error
    [lo] A[blyth@localhost tests]$ 



J26_1_1_opticks_Debug_recreated
---------------------------------

::
   
    export U4Polycone__ENABLE_PHICUT=1 
    export sn__PhiCut_PACMAN_ALLOWED=1
    ~/o/g4cx/tests/G4CXOpticks_setGeometry_Test.sh


YEP, contemporary GDML recreated geom lacks sensors::

    [lo] A[blyth@localhost tests]$ TEST=desc_named:sensor_nd  ~/o/sysrap/tests/stree_load_test.sh run
    [stree_load_test::init
    [stree::desc_id
     loaddir /home/blyth/.opticks/GEOM/J26_1_1_opticks_Debug_recreated/CSGFoundry/SSim/stree
    ]stree::desc_id

    ]stree_load_test::init
    stree::desc_named key[sensor_nd]
    [stree::desc_sensor_nd
     edge            10
     num_nd          386569
     num_nd_sensor   0
     num_sid         0
     num_sensor_name 0
    ]stree::desc_sensor_nd
    [lo] A[blyth@localhost tests]$ 


This stymies attempts to use the old geom ... so try adding lax sensor identification::

    export U4SensorIdentifierDefault__MODE=lax
    

J26_1_1_opticks_Debug_recreated try in fresh env with lax
-----------------------------------------------------------

::

    export U4Polycone__ENABLE_PHICUT=1 
    export sn__PhiCut_PACMAN_ALLOWED=1
    export U4SensorIdentifierDefault__MODE=lax
    ~/o/g4cx/tests/G4CXOpticks_setGeometry_Test.sh










Where is ctx.prd->identity populated ? CSGOptiX7.cu:__closesthit__ch
-------------------------------------------------------------------------


::

    749 extern "C" __global__ void __closesthit__ch()
    750 {
    751     unsigned iindex = optixGetInstanceIndex() ;
    752     unsigned identity = optixGetInstanceId() ;
    753     unsigned iindex_identity = (( iindex & 0xffffu ) << 16 ) | ( identity & 0xffffu ) ;


::

    1760 /**
    1761 qsim::propagate_at_surface_CustomART
    1762 -------------------------------------
    1763 
    1764 lpmtid:-1
    1765    indicates "not-a-sensor", that occurring at this juncture would
    1766    indicate a sensor_identity issue as CustomART special surfaces
    1767    are expected to always have sensor identities
    1768 
    1769 
    1770 Where ctx.prd->identity() comes from ? Where is the "+ 1" done ?
    1771 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    1772 
    1773 1. SBT::collectInstances sets OptixInstance::instanceId from sqat4::get_IAS_OptixInstance_instanceId
    1774    aka "sensor_identifier"
    1775 
    1776 2. CSGFoundry::addInstance for firstcall:true does the "+1" as done by CSGFoundry::addInstanceVector
    1777 
    1778 3. original access to the copyno from Geant4 in U4SensorIdentifierDefault::getInstanceIdentity
    1779    which is used from U4Tree::identifySensitiveInstances to populate the stree.h snode::sensor_id
    1780 
    1781 **/
    1782 
    1783 inline QSIM_METHOD int qsim::propagate_at_surface_CustomART(unsigned& flag, RNG& rng, sctx& ctx) const
    1784 {
    1785 
    1786     sphoton& p = ctx.p ;
    1787     const float3* normal = (float3*)&ctx.prd->q0.f.x ;  // geometrical outwards normal
    1788     int lpmtid = ctx.prd->identity() - 1 ;  // identity comes from optixInstance.instanceId where 0 means not-a-sensor
    1789     const float lposcost = ctx.prd->lposcost() ;  // local frame intersect position cosine theta
    1790 
    1791 




   /** 
    sqat4::get_IAS_OptixInstance_instanceId
    ----------------------------------------

    Canonical use by SBT::collectInstances

    * July 2023 : QPMT needs lpmtid GPU side so changed to sensor_identifier from ins_idx
    * FORMER BUG: -1 identifier when unsigned became ~0u exceeding permissable bits,
      causing change to +1 and using zero to mean not-a-sensor

    YEP: that ~0u caused notes/issues/unspecified_launch_failure_with_simpleLArTPC.rst

    **/

    QAT4_METHOD int get_IAS_OptixInstance_instanceId() const
    {   
        const int& sensor_identifier_1 = q2.i.w ;
        assert( sensor_identifier_1 >= 0 );  // 0 means not a sensor GPU side
        return sensor_identifier_1 ;    // NB this is +1,  subtract 1 to get original sensorId of sensors
    }   

    /** 
    sqat4::setIdentity
    -------------------

    Canonical usage from CSGFoundry::addInstance  where sensor_identifier gets +1
    with 0 meaning not a sensor.
    **/

    QAT4_METHOD void setIdentity(int ins_idx, int gas_idx, int sensor_identifier_1, int sensor_index )
    {   
        assert( sensor_identifier_1 >= 0 );

        q0.i.w = ins_idx ;               // formerly unsigned and "+ 1"
        q1.i.w = gas_idx ;
        q2.i.w = sensor_identifier_1 ;   // now +1 with 0 meaning not-a-sensor
        q3.i.w = sensor_index ;
    }   




::

    2134 /**
    2135 CSGFoundry::addInstanceVector
    2136 ------------------------------
    2137 
    2138 Canonical stack::
    2139 
    2140     G4CXOpticks::setGeometry
    2141     CSGFoundry::CreateFromSim  (after U4Tree::Create within G4CXOpticks::setGeometry)
    2142     CSGFoundry::importSim
    2143     CSGImport::import
    2144     CSGImport::importInst     (with argument stree::inst_f4 populated by stree::add_inst from snode::sensor_id)
    2145     CSGFoundry::addInstanceVector
    2146 
    2147 stree.h/snode.h uses sensor_identifier -1 to indicate not-a-sensor, but
    2148 that is not convenient on GPU due to OptixInstance.instanceId limits.
    2149 Hence here make transition by adding 1 and treating 0 as not-a-sensor,
    2150 with the sqat4::incrementSensorIdentifier method
    2151 
    2152 The stree::inst_f4 is formed from the stree globals and factors by stree::add_inst
    2153 
    2154 **/
    2155 
    2156 void CSGFoundry::addInstanceVector( const std::vector<glm::tmat4x4<float>>& v_inst_f4 )
    2157 {
    2158     assert( inst.size() == 0 );
    2159     int num_inst = v_inst_f4.size() ;
    2160 
    2161     for(int i=0 ; i < num_inst ; i++)
    2162     {
    2163         const glm::tmat4x4<float>& inst_f4 = v_inst_f4[i] ;
    2164         const float* tr16 = glm::value_ptr(inst_f4) ;
    2165         qat4 instance(tr16) ;
    2166         instance.incrementSensorIdentifier() ; // GPU side needs 0 to mean "not-a-sensor"
    2167         inst.push_back( instance );
    2168     }
    2169 }




::

    6722 inline void stree::add_inst(
    6723     glm::tmat4x4<double>& tr_m2w,
    6724     glm::tmat4x4<double>& tr_w2m,
    6725     int gas_idx,
    6726     int nidx )
    6727 {
    6728     assert( nidx > -1 && nidx < int(nds.size()) );
    6729     const snode& nd = nds[nidx];    // structural volume node
    6730 
    6731     int ins_idx = int(inst.size()); // 0-based index follow sqat4.h::setIdentity
    6732 
    6733     glm::tvec4<int64_t> col3 ;   // formerly uint64_t
    6734 
    6735     col3.x = ins_idx ;            // formerly  +1
    6736     col3.y = gas_idx ;            // formerly  +1
    6737     col3.z = nd.sensor_id ;       // formerly ias_idx + 1 (which was always 1)
    6738     col3.w = nd.sensor_index ;
    6739 
    6740     strid::Encode(tr_m2w, col3 );
    6741     strid::Encode(tr_w2m, col3 );
    6742 
    6743     inst.push_back(tr_m2w);
    6744     iinst.push_back(tr_w2m);
    6745 
    6746     inst_nidx.push_back(nidx);
    6747 }



::

    1106 inline void U4Tree::identifySensitiveInstances()
    1107 {
    1108     unsigned num_factor = st->get_num_factor();
    1109     if(level > 0) std::cerr
    1110         << "[ U4Tree::identifySensitiveInstances"
    1111         << " num_factor " << num_factor
    1112         << " st.sensor_count " << st->sensor_count
    1113         << std::endl
    1114         ;
    1115 
    1116     for(unsigned i=0 ; i < num_factor ; i++)
    1117     {
    1118         std::vector<int> outer ;
    1119         st->get_factor_nodes(outer, i);
    1120         // nidx of outer volumes of the instances for each factor
    1121 
    1122         sfactor& fac = st->get_factor_(i);
    1123         fac.sensors = 0  ;
    1124 
    1125         for(unsigned j=0 ; j < outer.size() ; j++)
    1126         {
    1127             int nidx = outer[j] ;
    1128             const G4VPhysicalVolume* pv = get_pv_(nidx) ;
    1129             const char* pvn = pv->GetName().c_str() ;
    1130 
    1131             int sensor_id = sid->getInstanceIdentity(pv) ;
    1132             assert( sensor_id >= -1 );  // sensor_id:-1 signifies "not-a-sensor"
    1133 
    1134             int sensor_index = sensor_id > -1 ? st->sensor_count : -1 ;
    1135             int sensor_name = -1 ;
    1136 
    1137             if(sensor_id > -1 )
    1138             {
    1139                 st->sensor_count += 1 ;  // count over all factors
    1140                 fac.sensors += 1 ;       // count sensors for each factor
    1141                 sensor_name = suniquename::Add(pvn, st->sensor_name ) ;
    1142             }
    1143 
    1144             snode& nd = st->nds[nidx] ;
    1145             nd.sensor_id = sensor_id ;  // -1:not-a-sensor-at-this-juncture
    1146             nd.sensor_index = sensor_index ;
    1147             nd.sensor_name = sensor_name ;







