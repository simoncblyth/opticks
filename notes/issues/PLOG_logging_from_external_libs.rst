PLOG_logging_from_external_libs always DEBUG, never suppressed
=================================================================

Overview
----------

Getting PLOG logging to work properly in an external lib 
in a project that does not use " -fvisibility=hidden " 
is proving difficult. 

Despite numerous attempts its still not working. 


The way things go wrong:

1. logging not suppressed by the level 
2. all logging suppressed 


DONE 
-----

1. setup a test examples/UseFindOpticks/CMakeLists.txt which uses FindOpticks.cmake and that reproduces the logging mis-behaviour to allow investigation 
2. added API to use the integer template argument for logging 



AHHA : the normal LOG(LEVEL) invokation is using the PLOG_DEFAULT_INSTANCE which is 0 
------------------------------------------------------------------------------------------

::

    #define LOG_(instance, severity)        IF_LOG_(instance, severity) (*plog::get<instance>()) += plog::Record(severity, PLOG_GET_FUNC(), __LINE__, PLOG_GET_FILE(), PLOG_GET_THIS())
    #define LOG(severity)                   LOG_(PLOG_DEFAULT_INSTANCE, severity)


So try changing the PMTSim logging to use the appropriate index::

    LOG_(1, LEVEL) 


Expt
------

::

    091 #include "DemoLib.hh"
    092 #include "DEMO_LOG.hh"
     93 #include "OPTICKS_LOG.hh"
     94 
     95 #define LOCAL_OPTICKS_LOG(argc, argv) {  PLOG_COLOR(argc, argv); OPTICKS_LOG_::Initialize(PLOG::instance, plog::get(), NULL ); } 
     96 
    099 int main(int argc, char** argv)
    100 {
    101     LOCAL_OPTICKS_LOG(argc, argv);
    102     DEMO_LOG::Initialize(info, plog::get(), nullptr );
    103     
    104     
    105     LOG(error) << "[" << argv[0] ;
    106     DemoLib::Dump();  
    107     LOG(error) << "]" << argv[0] ;
    108     return 0 ; 
    109     
    110 }   


Adding NAME_LOG.hh/NAME_LOG.cc to each of the external libs integrated with looks like
it might be workable, eg::

    OK_PMTSIM_LOG.hh
    OK_PHYSISIM_LOG.hh
    OK_DETSIMOPTIONS_LOG.hh

That covers the most inportant three, but there are more::

    Simulation/DetSimV2/PhysiSim
    Simulation/GenTools
    Simulation/DetSimV2/PMTSim
    Simulation/DetSimV2/AnalysisCode
    Simulation/DetSimV2/DetSimOptions

HMM : are the libs actually split like that ? YES::

    N[blyth@localhost build]$ cd lib
    N[blyth@localhost lib]$ l
    total 276696
     7672 -rwxrwxr-x.  1 blyth blyth  7853248 Sep 30 19:59 libPhysiSim.so
    13312 -rwxrwxr-x.  1 blyth blyth 13629304 Sep 30 19:32 libPMTSim.so
    10572 -rwxrwxr-x.  1 blyth blyth 10821920 Sep 30 19:32 libGenTools.so
     5268 -rwxrwxr-x.  1 blyth blyth  5393968 Sep 28 00:58 libDetSimOptions.so
    12868 -rwxrwxr-x.  1 blyth blyth 13176632 Sep 28 00:58 libAnalysisCode.so


HMM that is kinda heavy. Could add a static method to a suitable class from each shared lib ?
YES but its cleaner and more understandable to use separate struct for logging setup.::

    N[blyth@localhost junosw]$ jgr ELOG
    ./Simulation/DetSimV2/DetSimMTUtil/src/DetFactorySvc.cc:    OPTICKS_ELOG("DetFactorySvc"); 
    ./Simulation/DetSimV2/DetSimOptions/src/DetSim0Svc.cc:    OPTICKS_ELOG("DetSim0Svc_CXOK"); 
    ./Simulation/DetSimV2/DetSimOptions/src/DetSim0Svc.cc:    OPTICKS_ELOG("DetSim0Svc_OK"); 
    N[blyth@localhost junosw]$ 





    304 bool DetSim0Svc::initializeOpticks()
    305 {
    306     dumpOpticks("DetSim0Svc::initializeOpticks");
    307     assert( m_opticksMode > 0);
    ^^^^^^^^ THATS NOT CORRECT : COULD USE OPTICKS LOGGING WITH m_opticksMode 0  ^^^^^^^^^^^^
    308 
    309 #ifdef WITH_G4CXOPTICKS
    310     OPTICKS_ELOG("DetSim0Svc_CXOK");
    311 #elif WITH_G4OPTICKS
    312     OPTICKS_ELOG("DetSim0Svc_OK");
    313 #else
    314     LogError << " FATAL : non-zero opticksMode **NOT** WITH_G4CXOPTICKS or WITH_G4OPTICKS  " << std::endl ;
    315     assert(0);
    316 #endif
    317     return true ;
    318 }


::

    459 #define OPTICKS_ELOG(name) {           PLOG_ECOLOR(name);     OPTICKS_LOG_::Initialize(PLOG::instance, plog::get(), NULL ); } 


Logging mis-behaviour
------------------------

::

    epsilon:opticks blyth$ jcv junoSD_PMT_v2_Opticks
    2 files to edit
    ./Simulation/DetSimV2/PMTSim/include/junoSD_PMT_v2_Opticks.hh
    ./Simulation/DetSimV2/PMTSim/src/junoSD_PMT_v2_Opticks.cc

::

     44 
     45 #if (defined WITH_G4CXOPTICKS) || (defined WITH_G4OPTICKS)
     46 const plog::Severity junoSD_PMT_v2_Opticks::LEVEL = PLOG::EnvLevel("junoSD_PMT_v2_Opticks", "DEBUG") ;
     47 #endif
     48 


LOG(LEVEL) outputs from external libs always DEBUG, when would expect those would be suppressed.
The LOG(info) outputs appear as expected:: 

    dir /tmp/u4debug/ntds3/000 num_record 47
    2022-09-30 03:05:45.963 INFO  [178202] [U4Hit_Debug::Save@11]  dir /tmp/u4debug/ntds3/000 num_record 14
    dir /tmp/u4debug/ntds3/000 num_record 14
    [ junoSD_PMT_v2::EndOfEvent m_opticksMode  3
    2022-09-30 03:05:45.963 DEBUG [178202] [junoSD_PMT_v2_Opticks::EndOfEvent@169] [ eventID 0 m_opticksMode 3
    2022-09-30 03:05:45.995 INFO  [178202] [junoSD_PMT_v2_Opticks::EndOfEvent@190]  eventID 0 num_hit 27 way_enabled 0
         0 gp.x  -13840.08 gp.y   -8162.24 gp.z  -10659.09 gp.R   19281.76 pmt   13743          CK|RE|SD|BT
         1 gp.x  -13331.45 gp.y   -7860.98 gp.z  -11652.90 gp.R   19372.99 pmt   14076          CK|RE|SD|BT
         2 gp.x   -7827.26 gp.y  -16841.33 gp.z    5141.73 gp.R   19270.02 pmt    6269          CK|RE|SD|BT






Planting the LOG header and getting it to be exported works::

    Untracked files:
      (use "git add <file>..." to include in what will be committed)
        Simulation/DetSimV2/PMTSim/PMTSim/
        Simulation/DetSimV2/PMTSim/src/OK_PMTSIM_LOG.cc

    no changes added to commit (use "git add" and/or "git commit -a")
    N[blyth@localhost junosw]$ l Simulation/DetSimV2/PMTSim/PMTSim/
    total 4
    0 drwxrwxr-x. 2 blyth blyth  30 Sep 30 23:11 .
    0 drwxrwxr-x. 5 blyth blyth  68 Sep 30 23:10 ..
    4 -rw-rw-r--. 1 blyth blyth 365 Sep 30 23:00 OK_PMTSIM_LOG.hh
    N[blyth@localhost junosw]$ 



But then run into symbol visibility issue::

    junotoptask:MCParamsSvc.GetPath  INFO: Optical parameters will be used from: /data/blyth/junotop/data/Simulation/DetSim
    junotoptask:PMTSimParamSvc.init_file  INFO: Loading parameters from file: /data/blyth/junotop/data/Simulation/SimSvc/PMTSimParamSvc/PMTParam_CD_LPMT.root
    Detaching after fork from child process 222920.
    junotoptask:PMTSimParamSvc.init_file_SPMT  INFO: Loading parameters from file: /data/blyth/junotop/data/Simulation/SimSvc/PMTSimParamSvc/PMTParam_CD_SPMT.root
     m_all_pmtID.size = 45612
    junotoptask:DetSim0Svc.dumpOpticks  INFO: DetSim0Svc::initializeOpticks m_opticksMode 3 WITH_G4CXOPTICKS 
    python: /data/blyth/junotop/ExternalLibs/opticks/head/externals/plog/include/plog/Logger.h:22: plog::Logger<instance>& plog::Logger<instance>::addAppender(plog::IAppender*) [with int instance = 0]: Assertion `appender != this' failed.

    Program received signal SIGABRT, Aborted.
    0x00007ffff696e387 in raise () from /lib64/libc.so.6


The distinct loggers in main and in shared libs relies on not having global symbol visibility. 
This works in Opticks because are using " -fvisibility=hidden"

Question ? C++ How to arrange distinct symbols in main and shared lib without visibility hidden 


* https://stackoverflow.com/questions/69088562/hiding-symbols-of-the-derived-class-in-shared-library
* https://stackoverflow.com/questions/435352/limiting-visibility-of-symbols-when-linking-shared-libraries

Hmm maybe can use a namespace to avoid the symbol clash whilst not using " -fvisibility=hidden" 

::


    #pragma GCC visibility push(visibility)
    #pragma GCC visibility pop
        This pragma allows the user to set the visibility for multiple
        declarations without having to give each a visibility attribute See Function
        Attributes, for more information about visibility and the attribute syntax.

        In C++, ‘#pragma GCC visibility’ affects only namespace-scope
        declarations. Class members and template specializations are not affected; if
        you want to override the visibility for a particular member or instantiation,
        you must use an attribute. 




Actually plog has integer template argument that perhaps can handle this

/usr/local/opticks/externals/plog/include/plog/Logger.h::


     06 #ifndef PLOG_DEFAULT_INSTANCE
      7 #   define PLOG_DEFAULT_INSTANCE 0
      8 #endif
      9 
     10 namespace plog
     11 {
     12     template<int instance>
     13     class Logger : public util::Singleton<Logger<instance> >, public IAppender
     14     {
     15     public:
     16         Logger(Severity maxSeverity = none) : m_maxSeverity(maxSeverity)
     17         {
     18         }
     19 
     20         Logger& addAppender(IAppender* appender)
     21         {
     22             assert(appender != this);
     23             m_appenders.push_back(appender);
     24             return *this;
     25         }
     ..
     63     template<int instance>
     64     inline Logger<instance>* get()
     65     {
     66         return Logger<instance>::getInstance();
     67     }
     68 
     69     inline Logger<PLOG_DEFAULT_INSTANCE>* get()
     70     {
     71         return Logger<PLOG_DEFAULT_INSTANCE>::getInstance();
     72     }
     73 }

Try using the template argument. 

Simulation/DetSimV2/PMTSim/PMTSim/OK_PMTSIM_LOG.hh::

     01 #pragma once
      2 
      3 #ifdef WITH_G4CXOPTICKS
      4 
      5 #define OK_PMTSIM_LOG_( IDX ) { OK_PMTSIM_LOG::Initialize(plog::get<IDX>()->getMaxSeverity(), plog::get<IDX>(), nullptr ); }  
      6 #define OK_PMTSIM_API  __attribute__ ((visibility ("default")))
      7 
      8 struct OK_PMTSIM_API OK_PMTSIM_LOG
      9 {
     10     static void Initialize(int level, void* app1, void* app2 );
     11     static void Check(const char* msg);
     12 };  
     13 
     14 #endif




::

    junoSD_PMT_v2::EndOfEvent m_opticksMode 3 hitCollection 41 hitCollection_muon 0 hitCollection_opticks 0
    junotoptask:DetSimAlg.execute   INFO: DetSimAlg Simulate An Event (1) 
    junoSD_PMT_v2::Initialize
    2022-10-01 01:53:58.628 DEBUG [226536] [junoSD_PMT_v2_Opticks::Initialize@119]  eventID 1 wavelength (null) tool 0 input_photons 0 input_photon_repeat 0 LEVEL 5:DEBUG
    Begin of Event --> 1
    [ junoSD_PMT_v2::EndOfEvent m_opticksMode  3
    2022-10-01 01:53:58.645 DEBUG [226536] [junoSD_PMT_v2_Opticks::EndOfEvent@169] [ eventID 1 m_opticksMode 3
    2022-10-01 01:53:58.655 INFO  [226536] [junoSD_PMT_v2_Opticks::EndOfEvent@190]  eventID 1 num_hit 28 way_enabled 0
         0 gp.x     840.38 gp.y   19245.69 gp.z    1502.42 gp.R   19322.53 pmt 





Need to use consistent integer template argument for creation in the shared lib as well as hookup in the main::


     21 #pragma once
     22 #include "SYSRAP_API_EXPORT.hh"
     23 
     24 #define SYSRAP_LOG__  {       SYSRAP_LOG::Initialize(PLOG::instance->prefixlevel_parse( info, "SYSRAP"), plog::get(), NULL );  } 
     25 #define SYSRAP_LOG_ {         SYSRAP_LOG::Initialize(plog::get()->getMaxSeverity(), plog::get(), NULL ); } 
     26 #define _SYSRAP_LOG( IDX ) {  SYSRAP_LOG::Init<IDX>( info, plog::get<IDX>(), nullptr ) ; }
     27 
     28 
     29 struct SYSRAP_API SYSRAP_LOG 
     30 {
     31     static void Initialize(int level, void* app1, void* app2 );
     32     static void Check(const char* msg);
     33     
     34     template<int instance>
     35     static void Init(int level, void* app1, void* app2 );
     36 };


     21 #include <plog/Log.h>
     22 
     23 #include "SYSRAP_LOG.hh"
     24 #include "PLOG_INIT.hh"
     25 #include "PLOG.hh"
     26        
     27 void SYSRAP_LOG::Initialize(int level, void* app1, void* app2 )
     28 {
     29     PLOG_INIT(level, app1, app2);
     30 }
     31 void SYSRAP_LOG::Check(const char* msg)
     32 {   
     33     PLOG_CHECK(msg);
     34 }   
     35     
     36 
     37 template<int IDX>
     38 void SYSRAP_LOG::Init(int level, void* app1, void* app2 )
     39 {
     40     PLOG_INIT_(level, app1, app2, IDX ); 
     41 }
     42 


