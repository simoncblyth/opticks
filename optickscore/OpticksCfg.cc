
#include <string>
#include <vector>


#include "NGLM.hpp"
#include "OpticksCfg.hh"
#include "Opticks.hh"

class Opticks ; 

template OKCORE_API void BCfg::addOptionF<Opticks>(Opticks*, const char*, const char* );
template OKCORE_API void BCfg::addOptionI<Opticks>(Opticks*, const char*, const char* );
template OKCORE_API void BCfg::addOptionS<Opticks>(Opticks*, const char*, const char* );


#include "PLOG.hh"


template <class Listener>
OpticksCfg<Listener>::OpticksCfg(const char* name, Listener* listener, bool live) 
       : 
       BCfg(name, live),
       m_listener(listener),
       m_size(""),
       m_position(""),
       m_logname(""),
       m_exportconfig(""),
       m_torchconfig(""),
       m_g4gunconfig(""),
       m_g4inimac(""),
       m_g4runmac(""),
       m_g4finmac(""),
       m_testconfig(""),
       m_state_tag(""),
       m_materialprefix("/dd/Materials/"),
       m_zexplodeconfig("-5564.975,1000."),  // -(5564.950 + 5565.000)/2.0 = -5564.975
       m_meshversion(""),
       m_islice(""),
       m_fslice(""),
       m_pslice(""),
       m_pindex(""),
       m_builder(""),
       m_traverser(""),
       m_epsilon(0.1f),     
       m_rngmax(3000000),     
       m_bouncemax(9),     
       m_recordmax(10),
       m_timemax(200),
       m_animtimemax(50),
       m_animator_period(200),
       m_ivperiod(100),
       m_ovperiod(180),
       m_tvperiod(100),
       m_repeatidx(-1),
       m_multievent(1),
       m_restrictmesh(-1),
       m_analyticmesh(-1),
       m_modulo(-1),
       m_override(-1),
       m_debugidx(0),
       m_stack(2180),
       m_num_photons_per_g4event(10000),
       m_loaderverbosity(0),
       m_meshverbosity(0)
{   
   init();  
   m_listener->setCfg(this); 
}


template <class Listener>
void OpticksCfg<Listener>::init()
{
   m_desc.add_options()
       ("cfg4", "use Geant4 for generation/propagation, only supports TORCH source type with test geometries") ;

   m_desc.add_options()
       ("version,v", "print version string") ;

   m_desc.add_options()
       ("help,h",    "print help message") ;

   m_desc.add_options()
       ("idpath,i",  "print idpath based on input envvars") ;

   m_desc.add_options()
       ("nogeocache,G",  "inhibit use of the geocache") ;

   m_desc.add_options()
       ("nopropagate,P",  "inhibit generation/propagation") ;

   m_desc.add_options()
       ("noevent,E",  "inhibit event handling") ;

   m_desc.add_options()
       ("nostep,S",  "inhibit step by step recording") ;

   m_desc.add_options()
       ("nooptix,O",  "inhibit use of OptiX") ;

   m_desc.add_options()
       ("nonet,N",  "inhibit use of network") ;


   m_desc.add_options()
       ("jwire",  "enable wire frame view, use only with workstation GPUs") ;




   m_desc.add_options()
       ("optixviz",  "Enable OptiXViz, needed in load mode where ordinarily OptiX is not enabled as no propagation is done.") ;

   m_desc.add_options()
       ("noviz,V",  "just generate, propagate and save : no visualization") ;

   m_desc.add_options()
       ("noindex,I",  "no photon/record indexing") ;

   m_desc.add_options()
       ("noinstanced",  "inhibit instancing, use when debugging few volumes") ;

   m_desc.add_options()
       ("zbuf",  "experimental: enable OptiX Z-buffer rendering with ORenderer, OFrame ... ") ;



   m_desc.add_options()
       ("trivial",  "swap OptiX generate program with trivial standin for debugging") ;

   m_desc.add_options()
       ("nothing",  "swap OptiX generate program with do nothing standin for debugging, NB this is different from --nopropagate as a launch is done") ;

   m_desc.add_options()
       ("dumpseed",  "swap OptiX generate program with standin for debugging, that just dumps the seed ");

   m_desc.add_options()
       ("seedtest",  "interop buffer overwrite debugging") ;



   m_desc.add_options()
       ("simplify",  "simplify OptiX geometry for debugging") ;

   m_desc.add_options()
       ("qe1",  "Perfect quantum efficiency of cathodes") ;

   m_desc.add_options()
       ("geocenter",  "center view on geometry rather than the default genstep centering") ;

   m_desc.add_options()
       ("g4ui",  "cfg4: start g4ui session") ;


   m_desc.add_options()("trace",  "loglevel");
   m_desc.add_options()("debug",  "loglevel");
   m_desc.add_options()("info",  "loglevel");
   m_desc.add_options()("warning",  "loglevel");
   m_desc.add_options()("error",  "loglevel");
   m_desc.add_options()("fatal",  "loglevel");


   m_desc.add_options()
       ("compute",  "COMPUTE mode, ie not INTEROP") ; 

   m_desc.add_options()
       ("scintillation,s",  "load scintillation gensteps") ;

   m_desc.add_options()
       ("cerenkov,c",  "load cerenkov gensteps") ;

   m_desc.add_options()
       ("natural,n",  "load natural gensteps containing a mixture of scintillation and cerenkov steps") ;

   m_desc.add_options()
       ("machinery",  "machinery testing gensteps/event potentially non-indexable  ");



   m_desc.add_options()
       ("save",  "download generated/propagated event data from GPU and save to file") ;

   m_desc.add_options()
       ("load",  "load event data from file and upload to GPU for visualization") ;

   m_desc.add_options()
       ("torch",  "fabricate torch genstep using torch config settings") ;

   m_desc.add_options()
       ("torchconfig",   boost::program_options::value<std::string>(&m_torchconfig), "torch configuration" );

   m_desc.add_options()
       ("torchdbg",  "dump details of torch config") ;

   m_desc.add_options()
       ("dbgseed",  "save empty interop mode photons buffer after seeding to $TMP/dbgseed.npy "); 
   m_desc.add_options()
       ("dbginterop", "used Scene::initRenderersDebug with subset of renderers "); 
   m_desc.add_options()
       ("dbguploads", "used in OpticksViz::uploadEvent to dump the uploads table "); 



   m_desc.add_options()
       ("onlyseed", "exit App after seeding "); 

   m_desc.add_options()
       ("nogeometry", "skip loading of geometry, for debugging only "); 





   m_desc.add_options()
       ("g4gun",  "enable cfg4- geant4 only particle gun ") ;

   m_desc.add_options()
       ("g4gunconfig",   boost::program_options::value<std::string>(&m_g4gunconfig), "g4gun configuration" );

   m_desc.add_options()
       ("g4gundbg",  "dump details of g4gun config") ;


   m_desc.add_options()
       ("g4inimac",   boost::program_options::value<std::string>(&m_g4inimac), "path to g4 initialization .mac file, see cfg4-/CG4" );

   m_desc.add_options()
       ("g4runmac",   boost::program_options::value<std::string>(&m_g4runmac), "path to g4 run .mac file, see cfg4-/CG4" );

   m_desc.add_options()
       ("g4finmac",   boost::program_options::value<std::string>(&m_g4finmac), "path to g4 final .mac file, see cfg4-/CG4" );





   m_desc.add_options()
       ("steppingdbg",  "dump details of cfg4 stepping") ;

   m_desc.add_options()
       ("indexdbg",  "dump details of indexing") ;

   m_desc.add_options()
       ("forceindex",  "force indexing even when index exists already") ;



   m_desc.add_options()
       ("meshfixdbg",  "dump details of meshfixing, only active when rebuilding geocache with -G option") ;


   m_desc.add_options()
       ("test",  "fabricate dynamic test geometry, materials and surfaces configured via testconfig settings") ;

   m_desc.add_options()
       ("testconfig",   boost::program_options::value<std::string>(&m_testconfig), "dynamic test geometry configuration" );



   m_desc.add_options()
       ("export",  "cfg4: write geometry to file using exportconfig settings") ;

   m_desc.add_options()
       ("exportconfig",   boost::program_options::value<std::string>(&m_exportconfig), "export configuration" );




   m_desc.add_options()
       ("primary",  "Enable recording of primary vertices in cfg4-, stored in NumpyEvt primary ") ;



   m_desc.add_options()
       ("zexplode",  "explode mesh in z for debugging split unions") ;

   m_desc.add_options()
       ("zexplodeconfig",   boost::program_options::value<std::string>(&m_zexplodeconfig), "zexplode configuration" );



   char modulo[128];
   snprintf(modulo,128, "Modulo scaledown input gensteps for speed/debugging, eg 10 to decimate. Values less than 1 disable scaledown. Default %d", m_modulo);
   m_desc.add_options()
       ("modulo",  boost::program_options::value<int>(&m_modulo), modulo );

   char override[128];
   snprintf(override,128, "Override photons to generate/propagate for debugging, eg 1 for a single photon. Values less than 1 disable any override. Default %d", m_override);
   m_desc.add_options()
       ("override",  boost::program_options::value<int>(&m_override), override );

   char debugidx[128];
   snprintf(debugidx,128, "Index of item eg Photon for debugging. Default %d", m_debugidx);
   m_desc.add_options()
       ("debugidx",  boost::program_options::value<int>(&m_debugidx), debugidx );

   char stack[128];
   snprintf(stack,128, "OptiX stack size, smaller the faster util get overflows. Default %d", m_stack);
   m_desc.add_options()
       ("stack",  boost::program_options::value<int>(&m_stack), stack );

   char epsilon[128];
   snprintf(epsilon,128, "OptiX propagate epsilon. Default %10.4f", m_epsilon);
   m_desc.add_options()
       ("epsilon",  boost::program_options::value<float>(&m_epsilon), epsilon );


   char g4ppe[256];
   snprintf(g4ppe,256, 
"Number of torch photons to generate/propagate per event with Geant4 cfg4.sh,"
" changing the number of events to meet the photon total. Default %d", m_num_photons_per_g4event);
   m_desc.add_options()
       ("g4ppe",  boost::program_options::value<int>(&m_num_photons_per_g4event), g4ppe );

   m_desc.add_options()
       ("alt,a",  "use alternative record renderer") ;

   m_desc.add_options()
       ("fullscreen,f",  "start in fullscreen mode") ;

   m_desc.add_options()
       ("tag",   boost::program_options::value<std::string>(&m_event_tag), "eventtag to load/save" );

   m_desc.add_options()
       ("itag",   boost::program_options::value<std::string>(&m_integrated_event_tag), "integrated eventtag to load/save, used from OPG4 package" );


   m_desc.add_options()
       ("cat",   boost::program_options::value<std::string>(&m_event_cat), "event category for organization of event files, typically used instead of detector for test geometries such as prism and lens" );


   m_desc.add_options()
       ("state",   boost::program_options::value<std::string>(&m_state_tag), "Bookmarks state tag, allowing use of multiple collections of bookmarks." );

   m_desc.add_options()
       ("materialprefix",   boost::program_options::value<std::string>(&m_materialprefix), "Materials prefix string eg /dd/Materials/ " );



   m_desc.add_options()
       ("meshversion",   boost::program_options::value<std::string>(&m_meshversion), "debug only option for testing alternate mesh versions" );

   m_desc.add_options()
       ("islice",        boost::program_options::value<std::string>(&m_islice), "debug only option for use of partial instanced geometry, specified by python slice style colon delimited ints " );

   m_desc.add_options()
       ("fslice",        boost::program_options::value<std::string>(&m_fslice), "debug only option for use of partial face geometry, specified by python slice style colon delimited ints " );

   m_desc.add_options()
       ("pslice",        boost::program_options::value<std::string>(&m_pslice), "debug only option for selecting parts of analytic geometry, specified by python slice style colon delimited ints " );



   m_desc.add_options()
       ("pindex",        boost::program_options::value<std::string>(&m_pindex), "debug OptiX launch print index specified by up to three comma delimited ints " );


   m_desc.add_options()
       ("builder",        boost::program_options::value<std::string>(&m_builder), "OptiX Accel structure builder, CAUTION case sensitive ");

   m_desc.add_options()
       ("traverser",      boost::program_options::value<std::string>(&m_traverser), "OptiX Accel structure traverser, CAUTION case sensitive ");

   char rngmax[128];
   snprintf(rngmax,128, 
"Maximum number of photons that can be generated/propagated as limited by the number of pre-persisted curand streams. "
"Value must match envvar CUDAWRAP_RNG_MAX and corresponding pre-cooked seeds, see cudawrap- for details. "
"Default %d ", m_rngmax);

   m_desc.add_options()
       ("rngmax",  boost::program_options::value<int>(&m_rngmax), rngmax );

   char bouncemax[128];
   snprintf(bouncemax,128, 
"Maximum number of boundary bounces, 0:prevents any propagation leaving generated photons"
"Default %d ", m_bouncemax);
   m_desc.add_options()
       ("bouncemax,b",  boost::program_options::value<int>(&m_bouncemax), bouncemax );


   // keeping bouncemax one less than recordmax is advantageous 
   // as bookeeping is then consistent between the photons and the records 
   // as this avoiding truncation of the records

   char recordmax[128];
   snprintf(recordmax,128, 
"Maximum number of photon step records per photon, 1:to minimize without breaking machinery. Default %d ", m_recordmax);
   m_desc.add_options()
       ("recordmax,r",  boost::program_options::value<int>(&m_recordmax), recordmax );

   char timemax[128];
   snprintf(timemax,128, "Maximum time in nanoseconds. Default %d ", m_timemax);
   m_desc.add_options()
       ("timemax",  boost::program_options::value<int>(&m_timemax), timemax );


   char animtimemax[128];
   snprintf(animtimemax,128, "Maximum animation time in nanoseconds. Default %d ", m_animtimemax);
   m_desc.add_options()
       ("animtimemax",  boost::program_options::value<int>(&m_animtimemax), animtimemax );

   char animator_period[128];
   snprintf(animator_period,128, "Event Animator Period, typically in range 200 to 400 controlling the number of steps of the animation. Default %d ", m_animator_period);
   m_desc.add_options()
       ("evperiod",  boost::program_options::value<int>(&m_animator_period), animator_period);

   char ivperiod[128];
   snprintf(ivperiod,128, "Interpolated View Period, typically in range 50 to 400 controlling the number of steps of the animation. Default %d ", m_ivperiod);
   m_desc.add_options()
       ("ivperiod",  boost::program_options::value<int>(&m_ivperiod), ivperiod);

   char ovperiod[128];
   snprintf(ovperiod,128, "Orbital View Period, typically in range 180 to 360 controlling the number of steps of the animation. Default %d ", m_ovperiod);
   m_desc.add_options()
       ("ovperiod",  boost::program_options::value<int>(&m_ovperiod), ovperiod);

   char tvperiod[128];
   snprintf(tvperiod,128, "Track View Period, typically in range 50 to 200 controlling the number of steps of the animation. Default %d ", m_tvperiod);
   m_desc.add_options()
       ("tvperiod",  boost::program_options::value<int>(&m_tvperiod), tvperiod);


   char repeatidx[128];
   snprintf(repeatidx,128, "Repeat index used in development of instanced geometry, -1:flat full geometry. Default %d ", m_repeatidx);
   m_desc.add_options()
       ("repeatidx",  boost::program_options::value<int>(&m_repeatidx), repeatidx );


   char multievent[128];
   snprintf(multievent,128, "Multievent count used in development of multiple event propagation. Default %d ", m_multievent);
   m_desc.add_options()
       ("multievent",  boost::program_options::value<int>(&m_multievent), multievent);



   char restrictmesh[128];
   snprintf(restrictmesh,128, "Restrict meshes converted to OptiX geometry to the one identitied by index eg 0,1,2. Or -1 for no restriction. Default %d ", m_restrictmesh);
   m_desc.add_options()
       ("restrictmesh",  boost::program_options::value<int>(&m_restrictmesh), restrictmesh );

   char analyticmesh[128];
   snprintf(analyticmesh,128, "Index of instanced mesh with which to attempt analytic OptiX geometry eg 1,2. Or -1 for no analytic geometry. Default %d ", m_analyticmesh);
   m_desc.add_options()
       ("analyticmesh",  boost::program_options::value<int>(&m_analyticmesh), analyticmesh );

   char loaderverbosity[128];
   snprintf(loaderverbosity,128, "Geometry Loader Verbosity eg AssimpGGeo.  Default %d ", m_loaderverbosity);
   m_desc.add_options()
       ("loaderverbosity",  boost::program_options::value<int>(&m_loaderverbosity), loaderverbosity );

   char meshverbosity[128];
   snprintf(meshverbosity,128, "Mesh Operation Verbosity eg GMergedMesh::Create.  Default %d ", m_meshverbosity);
   m_desc.add_options()
       ("meshverbosity",  boost::program_options::value<int>(&m_meshverbosity), meshverbosity );



   ///////////////

   m_desc.add_options()
       ("size",  boost::program_options::value<std::string>(&m_size),
            "Comma delimited screen window coordinate width,height,window2pixel eg 1024,768,2  ");

   m_desc.add_options()
       ("position",  boost::program_options::value<std::string>(&m_position),
            "Comma delimited screen window upper left coordinate x,y,-,- eg 100,100  "
            "NB although 0,0 is screen top left the application title bar prevents positioning of the window over pixels 0:20 (approx) in y. " 
            "Also when the frame size is large positioning is constrained"
       );


   m_desc.add_options()
       ("logname",   boost::program_options::value<std::string>(&m_logname),
         "name of logfile");

   m_desc.add_options()
       ("config",   boost::program_options::value<std::string>(&m_configpath),
         "name of a file of a configuration.");

   m_desc.add_options()
       ("liveline",  boost::program_options::value<std::string>(&m_liveline),
           "string with spaces to be live parsed, as test of composed overrides");



    // the below formerly called size seems not to be working, so use simpler size above 
   addOptionS<Listener>(m_listener, 
            "livesize", 
            "Comma delimited screen window coordinate width,height,window2pixel eg 1024,768,2  ");

   // this size is being overriden: 
   // the screen size is set by Opticks::init using size from composition 

   addOptionI<Listener>(m_listener, 
            "dumpevent", 
            "Control GLFW event dumping ");
}   




template <class Listener>
const std::string& OpticksCfg<Listener>::getLogName()
{
    return m_logname ;
}

template <class Listener>
const std::string& OpticksCfg<Listener>::getSize()
{
    return m_size ;
}


template <class Listener>
const std::string& OpticksCfg<Listener>::getPosition()
{
    return m_position ;
}



template <class Listener>
const std::string& OpticksCfg<Listener>::getConfigPath()
{
    return m_configpath ;
}

template <class Listener>
const std::string& OpticksCfg<Listener>::getEventTag()
{
    if(m_event_tag.empty()) m_event_tag = "1" ;
    return m_event_tag ;
}


template <class Listener>
const std::string& OpticksCfg<Listener>::getIntegratedEventTag()
{
    if(m_integrated_event_tag.empty()) m_integrated_event_tag = "100" ;
    return m_integrated_event_tag ;
}




template <class Listener>
const std::string& OpticksCfg<Listener>::getEventCat()
{
    if(m_event_cat.empty()) m_event_cat = "" ;
    return m_event_cat ;
}


template <class Listener>
const std::string& OpticksCfg<Listener>::getStateTag()
{
    return m_state_tag ;
}

template <class Listener>
const std::string& OpticksCfg<Listener>::getMaterialPrefix()
{
    return m_materialprefix ;
}


template <class Listener>
const std::string& OpticksCfg<Listener>::getLiveLine()
{
    return m_liveline ;
}



template <class Listener>
const std::string& OpticksCfg<Listener>::getTorchConfig()
{
    return m_torchconfig ;
}

template <class Listener>
const std::string& OpticksCfg<Listener>::getG4GunConfig()
{
    return m_g4gunconfig ;
}

template <class Listener>
const std::string& OpticksCfg<Listener>::getG4IniMac()
{
    return m_g4inimac ;
}
template <class Listener>
const std::string& OpticksCfg<Listener>::getG4RunMac()
{
    return m_g4runmac ;
}
template <class Listener>
const std::string& OpticksCfg<Listener>::getG4FinMac()
{
    return m_g4finmac ;
}



template <class Listener>
const std::string& OpticksCfg<Listener>::getTestConfig()
{
    return m_testconfig ;
}
template <class Listener>
const std::string& OpticksCfg<Listener>::getExportConfig()
{
    return m_exportconfig ;
}



template <class Listener>
const std::string& OpticksCfg<Listener>::getZExplodeConfig()
{
    return m_zexplodeconfig ;
}
template <class Listener>
const std::string& OpticksCfg<Listener>::getMeshVersion()
{
    return m_meshversion ;
}
template <class Listener>
const std::string& OpticksCfg<Listener>::getISlice()
{
    return m_islice ;
}

template <class Listener>
const std::string& OpticksCfg<Listener>::getFSlice()
{
    return m_fslice ;
}
template <class Listener>
const std::string& OpticksCfg<Listener>::getPSlice()
{
    return m_pslice ;
}



template <class Listener>
const std::string& OpticksCfg<Listener>::getPrintIndex()
{
    return m_pindex ;
}



template <class Listener>
const std::string& OpticksCfg<Listener>::getBuilder()
{
    return m_builder ;
}
template <class Listener>
const std::string& OpticksCfg<Listener>::getTraverser()
{
    return m_traverser ;
}



template <class Listener>
float OpticksCfg<Listener>::getEpsilon()
{
    return m_epsilon ; 
}

template <class Listener>
int OpticksCfg<Listener>::getRngMax()
{
    return m_rngmax ; 
}

template <class Listener>
int OpticksCfg<Listener>::getBounceMax()
{
    return m_bouncemax ; 
}

template <class Listener>
int OpticksCfg<Listener>::getRecordMax()
{
    return m_recordmax ; 
}
template <class Listener>
int OpticksCfg<Listener>::getTimeMax()
{
    return m_timemax ; 
}

template <class Listener>
int OpticksCfg<Listener>::getAnimTimeMax()
{
    return m_animtimemax ; 
}

template <class Listener>
int OpticksCfg<Listener>::getInterpolatedViewPeriod()
{
    return m_ivperiod ; 
}

template <class Listener>
int OpticksCfg<Listener>::getOrbitalViewPeriod()
{
    return m_ovperiod ; 
}

template <class Listener>
int OpticksCfg<Listener>::getTrackViewPeriod()
{
    return m_tvperiod ; 
}
template <class Listener>
int OpticksCfg<Listener>::getAnimatorPeriod()
{
    return m_animator_period ; 
}



template <class Listener>
int OpticksCfg<Listener>::getRepeatIndex()
{
    return m_repeatidx ; 
}
template <class Listener>
int OpticksCfg<Listener>::getMultiEvent()
{
    return m_multievent ; 
}


template <class Listener>
int OpticksCfg<Listener>::getRestrictMesh()
{
    return m_restrictmesh ; 
}
template <class Listener>
int OpticksCfg<Listener>::getAnalyticMesh()
{
    return m_analyticmesh ; 
}





template <class Listener>
int OpticksCfg<Listener>::getModulo()
{
    return m_modulo ; 
}
template <class Listener>
int OpticksCfg<Listener>::getOverride()
{
    return m_override ; 
}
template <class Listener>
int OpticksCfg<Listener>::getDebugIdx()
{
    return m_debugidx ; 
}

template <class Listener>
int OpticksCfg<Listener>::getStack()
{
    return m_stack ; 
}

template <class Listener>
int OpticksCfg<Listener>::getNumPhotonsPerG4Event()
{
    return m_num_photons_per_g4event ; 
}

template <class Listener>
int OpticksCfg<Listener>::getLoaderVerbosity()
{
    return m_loaderverbosity ; 
}


template <class Listener>
int OpticksCfg<Listener>::getMeshVerbosity()
{
    return m_meshverbosity ; 
}





template <class Listener>
void OpticksCfg<Listener>::dump(const char* msg)
{
    LOG(info) << msg ;
    
    std::vector<std::string> names ; 
    names.push_back("compute");
    names.push_back("save");
    names.push_back("load");
    names.push_back("test");

    names.push_back("g4gun");

    for(unsigned int i=0 ; i < names.size() ; i++)
    {
        LOG(info) << std::setw(10) << names[i] << " " <<  hasOpt(names[i].c_str()) ;
    }
}


template class OKCORE_API OpticksCfg<Opticks> ;

