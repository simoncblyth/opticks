metadata review
======================

context
----------

* :doc:`tboolean_box_perfect_alignment_revisted`


future ab.perf module needs run metadata
-------------------------------------------------------------------------------------------

* for metadata that is not specific to an event 

* hostname 
* number and types of GPUs 
* RTX mode setting
* versions of OptiX, Geant4, Opticks(hmm probably just a commit hash for now, prior to releases)  

These are things that belong in Opticks m_parameters(NMeta) 
and need to be passed to and persisted with the event.  

* Actually they apply to multiple events, no need to duplicate to all.
  But need a convention of where to persist ? tag 0 : is the obvious answer  

* Better to initiate metadata persisting of OpticksRun.  
  Actully simpler to do this from Opticks ?  


::

     874 void Opticks::saveParameters() const
     875 {
     876     const char* dir = getRunResultsDir();
     877     const char* name = "parameters.json" ;
     878     LOG(info) << name << " into " << dir ;
     879     m_parameters->save( dir, name);
     880 }
     881 




where to persist run metadata
------------------------------

::

    [blyth@localhost tboolean-box]$ pwd
    /tmp/tboolean-box
    [blyth@localhost tboolean-box]$ l evt/tboolean-box/torch/
    total 44
    -rw-rw-r--. 1 blyth blyth 1781 Jun 18 21:57 DeltaVM.ini
    -rw-rw-r--. 1 blyth blyth  976 Jun 18 21:57 Opticks.npy
    -rw-rw-r--. 1 blyth blyth 2319 Jun 18 21:57 VM.ini
    -rw-rw-r--. 1 blyth blyth 1819 Jun 18 21:57 DeltaTime.ini
    -rw-rw-r--. 1 blyth blyth 1983 Jun 18 21:57 Time.ini
    drwxrwxr-x. 3 blyth blyth 4096 Jun 18 21:56 4
    drwxrwxr-x. 3 blyth blyth 4096 Jun 18 21:56 -4
    drwxrwxr-x. 5 blyth blyth 4096 Jun 18 16:34 3
    drwxrwxr-x. 5 blyth blyth 4096 Jun 18 16:34 -3
    drwxrwxr-x. 6 blyth blyth 4096 Jun 17 20:58 1
    drwxrwxr-x. 6 blyth blyth 4096 Jun 17 20:58 -1
    [blyth@localhost tboolean-box]$ 




current event metadata, from python
--------------------------------------

Work out where its coming from, by splitting into sources which are setting it.


* mostly from Opticks::makeEvent and OpticksEvent::init 


::

    tboolean-;tboolean-box-ip --tag 4


    In [2]: b.metadata.parameters
    Out[2]: 
    {
     u'Dynamic': 0,

     u'Note': u'recstp',

     u'NumG4Event': 10,
     u'NumPhotonsPerG4Event': 10000,

     u'genstepDigest': u'9587445e698bb77d8e895ff238715fec',
     u'photonData': u'dfa123a7382e32bfebb7eb741ccaa749',
     u'recordData': u'e1c46ce4b32c1c7e00f1378e807aa972',
     u'sequenceData': u'cd36790fbad96bc9edfbff51c41648d0'}

     u'jsonLoadPath': u'/tmp/tboolean-box/evt/tboolean-box/torch/-4/parameters.json',




OpticksEvent::init
~~~~~~~~~~~~~~~~~~~~~~~~~~

* these are event specific OR they deserve to be duplicated to all events for addressing purposes
  (they are the event spec) 

::

     u'TimeStamp': u'20190618_215604',
     u'Type': u'torch',
     u'Tag': u'-4',
     u'Detector': u'tboolean-box',
     u'Cat': u'tboolean-box',
     u'UDet': u'tboolean-box',


Switches is run level::

     u'Switches': u'WITH_SEED_BUFFER WITH_RECORD WITH_SOURCE WITH_ALIGN_DEV WITH_LOGDOUBLE ',

::

     569 void OpticksEvent::init()
     570 {
     571     m_versions = new NMeta ;
     572     m_parameters = new NMeta ;
     573     m_report = new Report ;
     574     m_domain = new OpticksDomain ;
     575 
     576     m_versions->add<int>("OptiXVersion",  OKConf::OptiXVersionInteger() );
     577     m_versions->add<int>("CUDAVersion",   OKConf::CUDAVersionInteger() );
     578     m_versions->add<int>("ComputeVersion", OKConf::ComputeCapabilityInteger() );
     579     m_versions->add<int>("Geant4Version",  OKConf::Geant4VersionInteger() );
     580 
     581     m_parameters->add<std::string>("TimeStamp", timestamp() );
     582     m_parameters->add<std::string>("Type", m_typ );
     583     m_parameters->add<std::string>("Tag", m_tag );
     584     m_parameters->add<std::string>("Detector", m_det );
     585     if(m_cat) m_parameters->add<std::string>("Cat", m_cat );
     586     m_parameters->add<std::string>("UDet", getUDet() );
     587 
     588     std::string switches = OpticksSwitches();
     589     m_parameters->add<std::string>("Switches", switches );
     590 



::


     u'Id': 0,
     u'Creator': u'/home/blyth/local/opticks/lib/OKG4Test',

     u'TestCSGPath': u'tboolean-box',
     u'TestConfig': u'autoseqmap=TO:0,SR:1,SA:0_name=tboolean-box_outerfirst=1_analytic=1_csgpath=tboolean-box_mode=PyCsgInBox_autoobject=Vacuum/perfectSpecularSurface//GlassSchottF2_autoemitconfig=photons:600000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x1,umin:0.45,umax:0.55,vmin:0.45,vmax:0.55,diffuse:1,ctmindiffuse:0.5,ctmaxdiffuse:1.0_autocontainer=Rock//perfectAbsorbSurface/Vacuum',

     u'NumGensteps': 1,
     u'NumPhotons': 100000,
     u'NumRecords': 1000000,



::

    0667 void OpticksEvent::setId(int id)
     668 {
     669     m_parameters->add<int>("Id", id);
     670 }
     671 
     672 void OpticksEvent::setCreator(const char* executable)
     673 {
     674     m_parameters->add<std::string>("Creator", executable ? executable : "NULL" );
     675 }

     682 void OpticksEvent::setTestCSGPath(const char* testcsgpath)
     683 {
     684     m_parameters->add<std::string>("TestCSGPath", testcsgpath ? testcsgpath : "" );
     685 }

     690 
     691 void OpticksEvent::setTestConfigString(const char* testconfig)
     692 {
     693     m_parameters->add<std::string>("TestConfig", testconfig ? testconfig : "" );
     694 }


    1036 void OpticksEvent::resize()
    1037 {
    ....
    1077     m_parameters->add<unsigned int>("NumGensteps", getNumGensteps());
    1078     m_parameters->add<unsigned int>("NumPhotons",  getNumPhotons());
    1079     m_parameters->add<unsigned int>("NumRecords",  getNumRecords());
    1080 
    1081 }




Opticks::makeEvent
~~~~~~~~~~~~~~~~~~~~~

Hmm all of these are not event specific, they are run specific.

::

     u'RngMax': 3000000,
     u'BounceMax': 9,
     u'RecordMax': 10,
     u'mode': u'INTEROP_MODE',
     u'cmdline': u'--okg4 --align --dbgskipclearzero --dbgnojumpzero --dbgkludgeflatzero --envkey --rendermode +global,+axis --animtimemax 20 --timemax 20 --geocenter --stack 2180 --eye 1,0,0 --test --testconfig autoseqmap=TO:0,SR:1,SA:0_name=tboolean-box_outerfirst=1_analytic=1_csgpath=tboolean-box_mode=PyCsgInBox_autoobject=Vacuum/perfectSpecularSurface//GlassSchottF2_autoemitconfig=photons:600000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x1,umin:0.45,umax:0.55,vmin:0.45,vmax:0.55,diffuse:1,ctmindiffuse:0.5,ctmaxdiffuse:1.0_autocontainer=Rock//perfectAbsorbSurface/Vacuum --torch --torchconfig type=disc_photons=100000_mode=fixpol_polarization=1,1,0_frame=-1_transform=1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000_source=0,0,599_target=0,0,0_time=0.1_radius=300_distance=200_zenithazimuth=0,1,0,1_material=Vacuum_wavelength=500 --torchdbg --tag 4 --pfx tboolean-box --cat tboolean-box --anakey tboolean --args --save ',

     u'EntryCode': u'G',
     u'EntryName': u'GENERATE',

     u'KEY': u'OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.f6cc352e44243f8fa536ab483ad390ce',
     u'GEOCACHE': u'/home/blyth/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/f6cc352e44243f8fa536ab483ad390ce/1',

::

    2353     NMeta*       parameters = evt->getParameters();
    2354     parameters->add<unsigned int>("RngMax",    rng_max );
    2355     parameters->add<unsigned int>("BounceMax", bounce_max );
    2356     parameters->add<unsigned int>("RecordMax", record_max );
    2357 
    2358     parameters->add<std::string>("mode", m_mode->description());
    2359     parameters->add<std::string>("cmdline", m_cfg->getCommandLine() );
    2360 
    2361     parameters->add<std::string>("EntryCode", BStr::ctoa(getEntryCode()) );
    2362     parameters->add<std::string>("EntryName", getEntryName() );
    2363 
    2364     parameters->add<std::string>("KEY",  getKeySpec() );
    2365     parameters->add<std::string>("GEOCACHE",  getIdPath() );
    2366     // formerly would have called this IDPATH, now using GEOCACHE to indicate new approach 
    2367 
    2368     evt->setCreator(SProc::ExecutablePath()) ; // no access to argv[0] for embedded running 
    2369 
    2370     assert( parameters->get<unsigned int>("RngMax") == rng_max );
    2371     assert( parameters->get<unsigned int>("BounceMax") == bounce_max );
    2372     assert( parameters->get<unsigned int>("RecordMax") == record_max );




