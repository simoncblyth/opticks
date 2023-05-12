U4SimtraceTest_getting_num_simtrace_zero
===========================================

::

    epsilon:tests blyth$ ./U4SimtraceTest.sh 
    BASH_SOURCE                    : ./FewPMT.sh 
    VERSION                        : 0 
    version_desc                   : N=0 unnatural geometry : FastSim/jPOM 
    POM                            :  
    pom_desc                       :  
    GEOM                           : FewPMT 
    FewPMT_GEOMList                : nnvtLogicalPMT 
    LAYOUT                         : one_pmt 
    ./U4SimtraceTest.sh GEOMList : nnvtLogicalPMT
    SLOG::EnvLevel adjusting loglevel by envvar   key U4VolumeMaker level INFO fallback DEBUG upper_level INFO
    U4VolumeMaker::PV name FewPMT
    U4VolumeMaker::PVG_ name FewPMT gdmlpath - sub - exists 0
    [ PMTSim::GetLV [nnvtLogicalPMT]
    PMTSim::init                   yielded chars :  cout  24933 cerr      0 : set VERBOSE to see them 
    PMTSim::getLV geom [nnvtLogicalPMT] mgr Y head [LogicalPMT]
    Option RealSurface is enabled in Central Detector.  Reduce the m_pmt_h from 570 to 357.225
     GetName() nnvt
    NNVT_MCPPMT_PMTSolid::NNVT_MCPPMT_PMTSolid
    G4Material::GetMaterial() WARNING: The material: PMT_Mirror does not exist in the table. Return NULL pointer.
    Warning: setting PMT mirror reflectivity to 0.9999 because no PMT_Mirror material properties defined
    [ ZSolid::ApplyZCutTree zcut    173.225 pmt_delta      0.001 body_delta     -4.999 inner_delta     -5.000 zcut+pmt_delta    173.226 zcut+body_delta    168.226 zcut+inner_delta    168.225
    ] ZSolid::ApplyZCutTree zcut 173.225
    Option RealSurface is enabed. Reduce the height of tube_hz from 60.000 to 21.112
    ] PMTSim::GetLV [nnvtLogicalPMT] lv Y
    main@63:  U4VolumeMaker::Desc() U4VolumeMaker::Desc GEOM FewPMT METH PVP_ WITH_PMTSIM 
    U4Tree::simtrace_scan@790: [ simtrace
    U4Tree::simtrace_scan@800:  i   0 RGMode: simtrace
    SSimtrace::simtrace@114:  num_simtrace 0
    SEvt::save@2444:  dir /tmp/blyth/opticks/GEOM/FewPMT/U4SimtraceTest/0/nnvt_inner1_solid_head
    U4Tree::simtrace_scan@800:  i   1 RGMode: simtrace
    SSimtrace::simtrace@114:  num_simtrace 0
    SEvt::save@2444:  dir /tmp/blyth/opticks/GEOM/FewPMT/U4SimtraceTest/0/nnvt_edge_solid
    U4Tree::simtrace_scan@800:  i   2 RGMode: simtrace
    SSimtrace::simtrace@114:  num_simtrace 0



