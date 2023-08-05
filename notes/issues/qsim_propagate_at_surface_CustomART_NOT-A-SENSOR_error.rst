qsim_propagate_at_surface_CustomART_NOT-A-SENSOR_error
=========================================================


DONE : Added qsim.h handling for lpmtid < 0 
-------------------------------------------------

::

    1495     if(lpmtid < 0 )
    1496     {
    1497         flag = NAN_ABORT ;
    1498 #ifdef DEBUG_PIDX
    1499         //if( ctx.idx == base->pidx ) 
    1500         printf("//qsim::propagate_at_surface_CustomART idx %d lpmtid %d : ERROR NOT-A-SENSOR : NAN_ABORT \n", ctx.idx, lpmtid );
    1501 #endif
    1502         return BREAK ;
    1503     }
    1504 


::

    //qsim::propagate_at_surface_CustomART idx 5114 lpmtid -1 : ERROR NOT-A-SENSOR : NAN_ABORT 
    //qsim::propagate_at_surface_CustomART idx 5115 lpmtid -1 : ERROR NOT-A-SENSOR : NAN_ABORT 
    //qsim::propagate_at_surface_CustomART idx 5116 lpmtid -1 : ERROR NOT-A-SENSOR : NAN_ABORT 
    //qsim::propagate_at_surface_CustomART idx 5117 lpmtid -1 : ERROR NOT-A-SENSOR : NAN_ABORT 
    //qsim::propagate_at_surface_CustomART idx 5118 lpmtid -1 : ERROR NOT-A-SENSOR : NAN_ABORT 
    //qsim::propagate_at_surface_CustomART idx 5119 lpmtid -1 : ERROR NOT-A-SENSOR : NAN_ABORT 
    //qsim.propagate_to_boundary[ idx 5001 u_absorption 0.81189805 logf(u_absorption) -0.20838045 absorption_length  1562.9586 absorption_distance 325.690002 
    //qsim.propagate idx 5001 bounce 1 command 3 flag 0 s.optical.x 7 s.optical.y 4 
    //qsim.propagate.WITH_CUSTOM4 idx 5001  BOUNDARY ems 4 lposcost   1.000 
    //qsim::propagate_at_surface_CustomART p.mom                 (  0.000   0.000  -1.000) 
    //qsim::propagate_at_surface_CustomART p.pol                 (  0.000  -1.000   0.000) 
    //qsim::propagate_at_surface_CustomART normal                ( -0.000   0.000   1.000) 
    //qsim::propagate_at_surface_CustomART cross_mom_nrm         (  0.000   0.000   0.000) 
    //qsim::propagate_at_surface_CustomART dot_pol_cross_mom_nrm:  -0.000 
    //qsim::propagate_at_surface_CustomART minus_cos_theta         -1.000 
    //qsim::propagate_at_surface_CustomART idx 4992 lpmtid -1 : ERROR NOT-A-SENSOR : NAN_ABORT 
    //qsim::propagate_at_surface_CustomART idx 4993 lpmtid -1 : ERROR NOT-A-SENSOR : NAN_ABORT 
    //qsim::propagate_at_surface_CustomART idx 4994 lpmtid -1 : ERROR NOT-A-SENSOR : NAN_ABORT 
    //qsim::propagate_at_surface_CustomART idx 4995 lpmtid -1 : ERROR NOT-A-SENSOR : NAN_ABORT 
    //qsim::propagate_at_surface_CustomART idx 4996 lpmtid -1 : ERROR NOT-A-SENSOR : NAN_ABORT 
    //qsim::propagate_at_surface_CustomART idx 4997 lpmtid -1 : ERROR NOT-A-SENSOR : NAN_ABORT 
    //qsim::propagate_at_surface_CustomART idx 4998 lpmtid -1 : ERROR NOT-A-SENSOR : NAN_ABORT 
    //qsim::propagate_at_surface_CustomART idx 4999 lpmtid -1 : ERROR NOT-A-SENSOR : NAN_ABORT 





A has no SD, maybe lacking Sensor setup ?
-------------------------------------------

Setting the below is not enough::

   export GBndLib__SENSOR_BOUNDARY_LIST=$(cat << EOL
    Pyrex/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/Vacuum
    Pyrex/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/Vacuum
    Pyrex/PMT_3inch_photocathode_logsurf2/PMT_3inch_photocathode_logsurf1/Vacuum
    Pyrex/PMT_20inch_veto_photocathode_logsurf2/PMT_20inch_veto_photocathode_logsurf1/Vacuum
   EOL
   )


Look into sensor setup::

    epsilon:ggeo blyth$ opticks-f getSensorBoundaryReport
    ./extg4/X4PhysicalVolume.cc:        << " GGeo::getSensorBoundaryReport() "
    ./extg4/X4PhysicalVolume.cc:        << m_ggeo->getSensorBoundaryReport()
    ./ggeo/GBndLib.cc:    ss << "GBndLib::getSensorBoundaryReport" << std::endl ; 
    ./ggeo/GBndLib.cc:    ss << getSensorBoundaryReport() ; 
    ./ggeo/GBndLib.cc:std::string GBndLib::getSensorBoundaryReport() const 
    ./ggeo/GBndLib.cc:    ss << "GBndLib::getSensorBoundaryReport" << std::endl ; 
    ./ggeo/GGeo.hh:        std::string  getSensorBoundaryReport() const ; 
    ./ggeo/GGeo.cc:std::string  GGeo::getSensorBoundaryReport() const { return m_bndlib->getSensorBoundaryReport() ; }
    ./ggeo/GBndLib.hh:       std::string getSensorBoundaryReport() const ;
    epsilon:opticks blyth$ 


::

     223 void X4PhysicalVolume::postConvert() const
     224 {
     225     LOG(LEVEL)
     226         << " GGeo::getNumVolumes() " << m_ggeo->getNumVolumes()
     227         << " GGeo::getNumSensorVolumes() " << m_ggeo->getNumSensorVolumes()
     228         << std::endl
     229         << " GGeo::getSensorBoundaryReport() "
     230         << std::endl
     231         << m_ggeo->getSensorBoundaryReport()
     232         ;
     233 
     234 
     235     LOG(LEVEL) << m_blib->getAddBoundaryReport() ;
     236 
     237     // too soon for sensor dumping as instances not yet formed, see GGeo::postDirectTranslationDump 
     238     //m_ggeo->dumpSensorVolumes("X4PhysicalVolume::postConvert"); 
     239 
     240     //m_ggeo->dumpSurfaces("X4PhysicalVolume::postConvert" ); 
     241 
     242 
     243     LOG(LEVEL) << m_blib->descSensorBoundary() ;
     244 
     245 
     246 }



AHHA the sensor boundary names need to be chosen according to the geometry::


    GBndLib__SENSOR_BOUNDARY_LIST YES
     num_SENSOR_BOUNDARY_LIST 4
      0 : [Pyrex/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/Vacuum]
      1 : [Pyrex/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/Vacuum]
      2 : [Pyrex/PMT_3inch_photocathode_logsurf2/PMT_3inch_photocathode_logsurf1/Vacuum]
      3 : [Pyrex/PMT_20inch_veto_photocathode_logsurf2/PMT_20inch_veto_photocathode_logsurf1/Vacuum]
                          sensor_total      0

    2023-08-05 22:57:54.981 INFO  [212656] [X4PhysicalVolume::postConvert@235] GBndLib::getAddBoundaryReport edgeitems 100 num_boundary_add 8
     boundary   0 b+1   1 add_count      1 Rock///Rock
     boundary   1 b+1   2 add_count      1 Rock//water_rock_bs/Water
     boundary   2 b+1   3 add_count      1 Water///Pyrex
     boundary   3 b+1   4 add_count      1 Pyrex/nnvt_photocathode_mirror_logsurf/nnvt_photocathode_mirror_logsurf/Vacuum
     boundary   4 b+1   5 add_count      1 Vacuum/nnvt_mcp_edge_opsurface/nnvt_photocathode_mirror_logsurf/Steel
     boundary   5 b+1   6 add_count      1 Vacuum/nnvt_mcp_plate_opsurface/nnvt_photocathode_mirror_logsurf/Steel
     boundary   6 b+1   7 add_count      1 Vacuum/nnvt_mcp_tube_opsurface/nnvt_photocathode_mirror_logsurf/Steel
     boundary   7 b+1   8 add_count      1 Vacuum/nnvt_mcp_opsurface/nnvt_photocathode_mirror_logsurf/Steel
                          add_total      8

    2023-08-05 22:57:54.981 INFO  [212656] [X4PhysicalVolume::postConvert@243] GBndLib::descSensorBoundary ni 8 sensor_count 0
      0 ( 4,-1,-1, 4) isb 0
      1 ( 4,-1, 5, 3) isb 0
      2 ( 3,-1,-1, 2) isb 0
      3 ( 2, 6, 6, 1) isb 0
      4 ( 1, 2, 6, 0) isb 0
      5 ( 1, 1, 6, 0) isb 0
      6 ( 1, 3, 6, 0) isb 0
      7 ( 1, 4, 6, 0) isb 0
    GBndLib::getSensorBoundaryReport
    GBndLib::getSensorBoundaryReport
    GBndLib__SENSOR_BOUNDARY_LIST.eval
    [    Pyrex/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/Vacuum
        Pyrex/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/Vacuum
        Pyrex/PMT_3inch_photocathode_logsurf2/PMT_3inch_photocathode_logsurf1/Vacuum
        Pyrex/PMT_20inch_veto_photocathode_logsurf2/PMT_20inch_veto_photocathode_logsurf1/Vacuum]
    GBndLib__SENSOR_BOUNDARY_LIST YES
     num_SENSOR_BOUNDARY_LIST 4
      0 : [Pyrex/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/Vacuum]
      1 : [Pyrex/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/Vacuum]
      2 : [Pyrex/PMT_3inch_photocathode_logsurf2/PMT_3inch_photocathode_logsurf1/Vacuum]
      3 : [Pyrex/PMT_20inch_veto_photocathode_logsurf2/PMT_20inch_veto_photocathode_logsurf1/Vacuum]
                          sensor_total      0



After doing that get sensor consistency error::

    2023-08-05 23:01:08.565 INFO  [212863] [X4PhysicalVolume::postConvert@225]  GGeo::getNumVolumes() 0 GGeo::getNumSensorVolumes() 0
     GGeo::getSensorBoundaryReport() 
    GBndLib::getSensorBoundaryReport
    GBndLib__SENSOR_BOUNDARY_LIST.eval
    [    Pyrex/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/Vacuum
        Pyrex/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/Vacuum
        Pyrex/PMT_3inch_photocathode_logsurf2/PMT_3inch_photocathode_logsurf1/Vacuum
        Pyrex/PMT_20inch_veto_photocathode_logsurf2/PMT_20inch_veto_photocathode_logsurf1/Vacuum

        Pyrex/nnvt_photocathode_mirror_logsurf/nnvt_photocathode_mirror_logsurf/Vacuum]
    GBndLib__SENSOR_BOUNDARY_LIST YES
     num_SENSOR_BOUNDARY_LIST 5
      0 : [Pyrex/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/Vacuum]
      1 : [Pyrex/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/Vacuum]
      2 : [Pyrex/PMT_3inch_photocathode_logsurf2/PMT_3inch_photocathode_logsurf1/Vacuum]
      3 : [Pyrex/PMT_20inch_veto_photocathode_logsurf2/PMT_20inch_veto_photocathode_logsurf1/Vacuum]
      4 : [Pyrex/nnvt_photocathode_mirror_logsurf/nnvt_photocathode_mirror_logsurf/Vacuum]
     boundary   3 b+1   4 sensor_count      1 Pyrex/nnvt_photocathode_mirror_logsurf/nnvt_photocathode_mirror_logsurf/Vacuum
                          sensor_total      1

    2023-08-05 23:01:08.565 INFO  [212863] [X4PhysicalVolume::postConvert@235] GBndLib::getAddBoundaryReport edgeitems 100 num_boundary_add 8
     boundary   0 b+1   1 add_count      1 Rock///Rock
     boundary   1 b+1   2 add_count      1 Rock//water_rock_bs/Water
     boundary   2 b+1   3 add_count      1 Water///Pyrex
     boundary   3 b+1   4 add_count      1 Pyrex/nnvt_photocathode_mirror_logsurf/nnvt_photocathode_mirror_logsurf/Vacuum
     boundary   4 b+1   5 add_count      1 Vacuum/nnvt_mcp_edge_opsurface/nnvt_photocathode_mirror_logsurf/Steel
     boundary   5 b+1   6 add_count      1 Vacuum/nnvt_mcp_plate_opsurface/nnvt_photocathode_mirror_logsurf/Steel
     boundary   6 b+1   7 add_count      1 Vacuum/nnvt_mcp_tube_opsurface/nnvt_photocathode_mirror_logsurf/Steel
     boundary   7 b+1   8 add_count      1 Vacuum/nnvt_mcp_opsurface/nnvt_photocathode_mirror_logsurf/Steel
                          add_total      8

    2023-08-05 23:01:08.565 INFO  [212863] [X4PhysicalVolume::postConvert@243] GBndLib::descSensorBoundary ni 8 sensor_count 1
      0 ( 4,-1,-1, 4) isb 0
      1 ( 4,-1, 5, 3) isb 0
      2 ( 3,-1,-1, 2) isb 0
      3 ( 2, 6, 6, 1) isb 1
      4 ( 1, 2, 6, 0) isb 0
      5 ( 1, 1, 6, 0) isb 0
      6 ( 1, 3, 6, 0) isb 0
      7 ( 1, 4, 6, 0) isb 0
    GBndLib::getSensorBoundaryReport
    GBndLib::getSensorBoundaryReport
    GBndLib__SENSOR_BOUNDARY_LIST.eval
    [    Pyrex/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/Vacuum
        Pyrex/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/Vacuum
        Pyrex/PMT_3inch_photocathode_logsurf2/PMT_3inch_photocathode_logsurf1/Vacuum
        Pyrex/PMT_20inch_veto_photocathode_logsurf2/PMT_20inch_veto_photocathode_logsurf1/Vacuum

        Pyrex/nnvt_photocathode_mirror_logsurf/nnvt_photocathode_mirror_logsurf/Vacuum]
    GBndLib__SENSOR_BOUNDARY_LIST YES
     num_SENSOR_BOUNDARY_LIST 5
      0 : [Pyrex/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/Vacuum]
      1 : [Pyrex/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/Vacuum]
      2 : [Pyrex/PMT_3inch_photocathode_logsurf2/PMT_3inch_photocathode_logsurf1/Vacuum]
      3 : [Pyrex/PMT_20inch_veto_photocathode_logsurf2/PMT_20inch_veto_photocathode_logsurf1/Vacuum]
      4 : [Pyrex/nnvt_photocathode_mirror_logsurf/nnvt_photocathode_mirror_logsurf/Vacuum]
     boundary   3 b+1   4 sensor_count      1 Pyrex/nnvt_photocathode_mirror_logsurf/nnvt_photocathode_mirror_logsurf/Vacuum
                          sensor_total      1

    2023-08-05 23:01:08.565 INFO  [212863] [X4PhysicalVolume::init@218] ]
    2023-08-05 23:01:08.583 INFO  [212863] [GGeo::postDirectTranslation@648] NOT SAVING : TO ENABLE : export GGeo__postDirectTranslation_save=1 
    2023-08-05 23:01:08.585 INFO  [212863] [G4CXOpticks::setGeometry@273] 
    2023-08-05 23:01:08.620 FATAL [212863] [CSG_GGeo_Convert::init@94] CSG_GGeo_Convert::DescConsistent gg_all_sensor_index_num 1 st_all_sensor_id_num 0
    GGeoLib::descAllSensorIndex nmm 1
    ( 0 : 1) all[ 1]


    2023-08-05 23:01:08.621 INFO  [212863] [CSG_GGeo_Convert::init@95] CSG_GGeo_Convert::DescConsistent gg_all_sensor_index_num 1 st_all_sensor_id_num 0
    GGeoLib::descAllSensorIndex nmm 1
    ( 0 : 1) all[ 1]


    G4CXAppTest: /data/blyth/junotop/opticks/CSG_GGeo/CSG_GGeo_Convert.cc:96: void CSG_GGeo_Convert::init(): Assertion `consistent == 0' failed.
    ./G4CXAppTest.sh: line 100: 212863 Aborted                 (core dumped) $bin
    ./G4CXAppTest.sh : run error
    N[blyth@localhost tests]$ 



::

    (gdb) bt
    #0  0x00007fffeb094387 in raise () from /lib64/libc.so.6
    #1  0x00007fffeb095a78 in abort () from /lib64/libc.so.6
    #2  0x00007fffeb08d1a6 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007fffeb08d252 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007fffeff5838a in CSG_GGeo_Convert::init (this=0x7fffffff34b0) at /data/blyth/junotop/opticks/CSG_GGeo/CSG_GGeo_Convert.cc:96
    #5  0x00007fffeff5810b in CSG_GGeo_Convert::CSG_GGeo_Convert (this=0x7fffffff34b0, foundry_=0xf7a6d0, ggeo_=0xa58d50)
        at /data/blyth/junotop/opticks/CSG_GGeo/CSG_GGeo_Convert.cc:88
    #6  0x00007fffeff57c86 in CSG_GGeo_Convert::Translate (ggeo=0xa58d50) at /data/blyth/junotop/opticks/CSG_GGeo/CSG_GGeo_Convert.cc:49
    #7  0x00007ffff7b14cc1 in G4CXOpticks::setGeometry (this=0x9f9c90, gg_=0xa58d50) at /data/blyth/junotop/opticks/g4cx/G4CXOpticks.cc:276
    #8  0x00007ffff7b14bb4 in G4CXOpticks::setGeometry (this=0x9f9c90, world=0x99d2a0) at /data/blyth/junotop/opticks/g4cx/G4CXOpticks.cc:269
    #9  0x00007ffff7b1339f in G4CXOpticks::SetGeometry (world=0x99d2a0) at /data/blyth/junotop/opticks/g4cx/G4CXOpticks.cc:70
    #10 0x000000000041786f in G4CXApp::Construct (this=0x8d7e50) at /data/blyth/junotop/opticks/g4cx/tests/G4CXApp.h:168
    #11 0x00007ffff3ef5cbe in G4RunManager::InitializeGeometry() () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4run.so
    #12 0x00007ffff3ef5b2c in G4RunManager::Initialize() () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4run.so
    #13 0x000000000041744b in G4CXApp::G4CXApp (this=0x8d7e50, runMgr=0x87b240) at /data/blyth/junotop/opticks/g4cx/tests/G4CXApp.h:148
    #14 0x0000000000418373 in G4CXApp::Create () at /data/blyth/junotop/opticks/g4cx/tests/G4CXApp.h:283
    #15 0x0000000000418431 in G4CXApp::Main () at /data/blyth/junotop/opticks/g4cx/tests/G4CXApp.h:294
    #16 0x00000000004185d1 in main (argc=1, argv=0x7fffffff4a58) at /data/blyth/junotop/opticks/g4cx/tests/G4CXAppTest.cc:18
    (gdb) 



::

    (gdb) f 7 
    #7  0x00007ffff7b14cc1 in G4CXOpticks::setGeometry (this=0x9f9c90, gg_=0xa58d50) at /data/blyth/junotop/opticks/g4cx/G4CXOpticks.cc:276
    276	    CSGFoundry* fd_ = CSG_GGeo_Convert::Translate(gg) ; 
    (gdb) f 6
    #6  0x00007fffeff57c86 in CSG_GGeo_Convert::Translate (ggeo=0xa58d50) at /data/blyth/junotop/opticks/CSG_GGeo/CSG_GGeo_Convert.cc:49
    49	    CSG_GGeo_Convert conv(fd, ggeo ) ; 
    (gdb) f 5
    #5  0x00007fffeff5810b in CSG_GGeo_Convert::CSG_GGeo_Convert (this=0x7fffffff34b0, foundry_=0xf7a6d0, ggeo_=0xa58d50)
        at /data/blyth/junotop/opticks/CSG_GGeo/CSG_GGeo_Convert.cc:88
    88	    init(); 
    (gdb) f 4
    #4  0x00007fffeff5838a in CSG_GGeo_Convert::init (this=0x7fffffff34b0) at /data/blyth/junotop/opticks/CSG_GGeo/CSG_GGeo_Convert.cc:96
    96	    assert( consistent == 0 ); 
    (gdb) list
    91	void CSG_GGeo_Convert::init()
    92	{
    93	    int consistent = CheckConsistent(ggeo, tree) ; 
    94	    LOG_IF(fatal, consistent != 0 ) << DescConsistent(ggeo, tree); 
    95	    LOG(info) << DescConsistent(ggeo, tree); 
    96	    assert( consistent == 0 ); 
    97	
    98	    ggeo->getMeshNames(foundry->meshname); 
    99	    ggeo->getMergedMeshLabels(foundry->mmlabel); 
    100	    // boundary names now travel with the NP bnd.names 
    (gdb) 



::

     298 int CSG_GGeo_Convert::CheckConsistent(const GGeo* gg, const stree* st ) // static 
     299 {
     300     bool one_based_index = true ;
     301     std::vector<int> gg_all_sensor_index ;
     302     gg->getAllSensorIndex(gg_all_sensor_index, one_based_index );
     303 
     304     int gg_all_sensor_index_num = gg_all_sensor_index.size() ;
     305     int st_all_sensor_id_num = st ? st->sensor_id.size() : -1 ;
     306 
     307     int rc = 0 ;
     308     if( gg_all_sensor_index_num != st_all_sensor_id_num ) rc += 1000 ;
     309     return rc ;
     310 }
     311 
     312 std::string CSG_GGeo_Convert::DescConsistent(const GGeo* gg, const stree* st ) // static 
     313 {
     314     bool one_based_index = true ;
     315     std::vector<int> gg_all_sensor_index ;
     316     gg->getAllSensorIndex(gg_all_sensor_index, one_based_index );
     317 
     318     int gg_all_sensor_index_num = gg_all_sensor_index.size() ;
     319     int st_all_sensor_id_num = st ? st->sensor_id.size() : -1 ;
     320 
     321     std::stringstream ss ;
     322     ss << "CSG_GGeo_Convert::DescConsistent"
     323         << " gg_all_sensor_index_num " << gg_all_sensor_index_num
     324         << " st_all_sensor_id_num " << st_all_sensor_id_num
     325         << std::endl 
     326         ;
     327 
     328     ss << gg->descAllSensorIndex() << std::endl ;
     329 
     330     std::string str = ss.str();
     331     return str ;
     332 }


HMM probably need some work on U4Tree::identifySensitiveGlobals
-------------------------------------------------------------------------

::


     871 NB changes made to U4Tree::identifySensitiveInstances should
     872 usually be made in tandem with U4Tree::identifySensitiveGlobals
     873 
     874 **/
     875 
     876 inline void U4Tree::identifySensitiveInstances()
     877 {
     878     unsigned num_factor = st->get_num_factor();
     879     if(level > 0) std::cerr
     880         << "[ U4Tree::identifySensitiveInstances"
     881         << " num_factor " << num_factor
     882         << " st.sensor_count " << st->sensor_count
     883         << std::endl
     884         ;
     885 


Note that the error can be looked at on Darwin as no need for OptiX7. 

::

    [ stree::labelFactorSubtrees num_factor 0
    ] stree::labelFactorSubtrees 
    stree::collectRemainderNodes rem.size 8
    stree::desc_factor
    sfactor::Desc num_factor 0
     tot_freq_subtree       0

    ] stree::factorize 
    [ U4Tree::identifySensitive 
    [ U4Tree::identifySensitiveInstances num_factor 0 st.sensor_count 0
    ] U4Tree::identifySensitiveInstances num_factor 0 st.sensor_count 0
    [ U4Tree::identifySensitiveGlobals st.sensor_count 0 remainder.size 8
    ] U4Tree::identifySensitiveGlobals  st.sensor_count 0 remainder.size 8
    [ stree::reorderSensors
    ] stree::reorderSensors sensor_count 0
    ] U4Tree::identifySensitive st.sensor_count 0
    ] U4Tree::Create 
    [stree::postcreate
    stree::desc_sensor
     sensor_id.size 0
     sensor_count 0
     sensor_name.size 0
    sensor_name[
    ]
    [stree::desc_sensor_nd
     edge            0
     num_nd          8
     num_nd_sensor   0
     num_sid         0
    ]stree::desc_sensor_nd
    stree::desc_sensor_id sensor_id.size 0



