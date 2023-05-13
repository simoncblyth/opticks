FIXED : U4SimtraceTest_getting_num_simtrace_zero 
=====================================================

Confirmed that the issues was caused by frame rejig
for input photons breaking simtrace. Together with 
an analysis folder inconsistency with the event index.



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


::

    epsilon:sysrap blyth$ opticks-f setNumSimtrace
    ./sysrap/SEvt.hh:    void setNumSimtrace(unsigned num_simtrace);
    ./sysrap/SEvt.cc:    setNumSimtrace( num_photon_gs ); 
    ./sysrap/SEvt.cc:SEvt::setNumSimtrace
    ./sysrap/SEvt.cc:void SEvt::setNumSimtrace(unsigned num_simtrace)
    ./qudarap/QEvent.hh:    void     setNumSimtrace(unsigned num_simtrace) ;  
    ./qudarap/QEvent.cc:        setNumSimtrace( evt->num_seed ); 
    ./qudarap/QEvent.cc:void QEvent::setNumSimtrace(unsigned num_simtrace)
    ./qudarap/QEvent.cc:    sev->setNumSimtrace(num_simtrace); 
    epsilon:opticks blyth$ 



Probably the frame rejig needed for input photons has broken simtrace
-----------------------------------------------------------------------

::

     509 /**
     510 SEvt::setFrame_HostsideSimtrace
     511 ---------------------------------
     512 
     513 Called from SEvt::setFrame when sframe::is_hostside_simtrace, eg at behest of X4Simtrace::simtrace
     514 
     515 **/
     516 
     517 void SEvt::setFrame_HostsideSimtrace()
     518 {
     519     unsigned num_photon_gs = getNumPhotonFromGenstep();
     520     unsigned num_photon_evt = evt->num_photon ;
     521     LOG(LEVEL)
     522          << "frame.is_hostside_simtrace"
     523          << " num_photon_gs " << num_photon_gs
     524          << " num_photon_evt " << num_photon_evt
     525          ;
     526 
     527     assert( num_photon_gs == num_photon_evt );
     528     setNumSimtrace( num_photon_gs );
     529 
     530     LOG(LEVEL) << " before hostside_running_resize simtrace.size " << simtrace.size() ;
     531 
     532     hostside_running_resize();
     533 
     534     LOG(LEVEL) << " after hostside_running_resize simtrace.size " << simtrace.size() ;
     535 
     536     SFrameGenstep::GenerateSimtracePhotons( simtrace, genstep );
     537 }


     400 /**
     401 SEvt::setFrame
     402 ------------------
     403 
     404 As it is necessary to have the geometry to provide the frame this 
     405 is now split from eg initInputPhotons.  
     406 
     407 For simtrace and input photon running with or without a transform 
     408 it is necessary to call this for every event. 
     409 
     410 **simtrace running**
     411     MakeCenterExtentGensteps based on the given frame. 
     412 
     413 **simulate inputphoton running**
     414     MakeInputPhotonGenstep and m2w (model-2-world) 
     415     transforms the photons using the frame transform
     416 
     417 **/
     418 
     419 
     420 void SEvt::setFrame(const sframe& fr )
     421 {
     422     frame = fr ;
     423     // addFrameGenstep(); // relocated to SEvt::BeginOfEvent
     424     transformInputPhoton();
     425 }




Looks to be a path inconsistency
----------------------------------

Are writing into the below folder, from where the low level check shows expected scatter plot::

    epsilon:000 blyth$ pwd
    /tmp/blyth/opticks/GEOM/FewPMT/U4SimtraceTest/0/nnvt_inner1_solid_head/000
    epsilon:000 blyth$ ~/opticks/sysrap/tests/SSimtrace_check.sh 


But the U4SimtraceTest.py script is looking elsewhere. HMM it should be looking inside the 000::

    In [3]: s.sfs["nnvt_inner1_solid_head"]                                                                                                                                    
    Out[3]: 
    s04

    CMDLINE:/Users/blyth/opticks/u4/tests/U4SimtraceTest.py
    s04.base:/tmp/blyth/opticks/GEOM/FewPMT/U4SimtraceTest/0/nnvt_inner1_solid_head

      : s04.sframe_meta                                    :                    1 : 0:33:35.936112 
      : s04.000                                            :                 None : 0:22:21.181570 
      : s04.sframe                                         :            (4, 4, 4) : 0:33:35.936043 
      : s04.NPFold_index                                   :                    0 : 0:33:35.936397 

     min_stamp : 2023-05-13 09:36:16.191078 
     max_stamp : 2023-05-13 09:47:30.945905 
     dif_stamp : 0:11:14.754827 
     age_stamp : 0:22:21.181570 
         









    




