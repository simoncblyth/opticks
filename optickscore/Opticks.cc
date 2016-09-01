
#ifdef _MSC_VER
// object allocated on the heap may not be aligned 16
// https://github.com/g-truc/glm/issues/235
// apparently fixed by 0.9.7.1 Release : currently on 0.9.6.3

#pragma warning( disable : 4316 )
#endif


#include "SSys.hh"
// brap-
#include "BDynamicDefine.hh"
#include "BStr.hh"
#include "PLOG.hh"
#include "Map.hh"


// npy-
#include "Timer.hpp"
#include "Parameters.hpp"
#include "TorchStepNPY.hpp"
#include "GLMFormat.hpp"
#include "NState.hpp"
#include "NPropNames.hpp"
#include "NLoad.hpp"

// okc-
#include "OpticksPhoton.h"
#include "OpticksFlags.hh"
#include "Opticks.hh"
#include "OpticksResource.hh"
#include "OpticksColors.hh"
#include "OpticksEvent.hh"
#include "OpticksMode.hh"


#include "OpticksCfg.hh"


NPropNames* Opticks::G_MATERIAL_NAMES = NULL ; 

const float Opticks::F_SPEED_OF_LIGHT = 299.792458f ;  // mm/ns

const char* Opticks::COMPUTE_ARG_ = "--compute" ; 

// formerly of GPropertyLib, now booted upstairs
float        Opticks::DOMAIN_LOW  = 60.f ;
float        Opticks::DOMAIN_HIGH = 820.f ;  // has been 810.f for a long time  
float        Opticks::DOMAIN_STEP = 20.f ; 
unsigned int Opticks::DOMAIN_LENGTH = 39  ;

/*

In [12]: np.linspace(60,820,39)
Out[12]: 
array([  60.,   80.,  100.,  120.,  140.,  160.,  180.,  200.,  220.,
        240.,  260.,  280.,  300.,  320.,  340.,  360.,  380.,  400.,
        420.,  440.,  460.,  480.,  500.,  520.,  540.,  560.,  580.,
        600.,  620.,  640.,  660.,  680.,  700.,  720.,  740.,  760.,
        780.,  800.,  820.])

In [13]: np.linspace(60,820,39).shape
Out[13]: (39,)

*/


glm::vec4 Opticks::getDefaultDomainSpec()
{
    glm::vec4 bd ;

    bd.x = DOMAIN_LOW ;
    bd.y = DOMAIN_HIGH ;
    bd.z = DOMAIN_STEP ;
    bd.w = DOMAIN_HIGH - DOMAIN_LOW ;

    return bd ; 
}


Opticks::Opticks(int argc, char** argv, bool integrated )
     :
       m_argc(argc),
       m_argv(argv),
       m_envprefix(strdup("OPTICKS_")),
       m_materialprefix(NULL),

       m_resource(NULL),
       m_state(NULL),

       m_exit(false),
       m_compute(false),
       m_geocache(false),
       m_instanced(true),
       m_integrated(integrated),

       m_lastarg(NULL),

       m_cfg(NULL),
       m_timer(NULL),
       m_parameters(NULL),
       m_detector(NULL),
       m_tag(NULL),
       m_cat(NULL),
       m_event_count(0),
       m_domains_configured(false),
       m_mode(NULL)
{
       init();
}

int Opticks::getArgc()
{
    return m_argc ; 
}
char** Opticks::getArgv()
{
    return m_argv ; 
}


bool Opticks::hasArg(const char* arg)
{
    bool has = false ; 
    for(int i=1 ; i < m_argc ; i++ ) if(strcmp(m_argv[i], arg) == 0) has = true ; 
    return has ; 
}

void Opticks::setCfg(OpticksCfg<Opticks>* cfg)
{
    m_cfg = cfg ; 
}
OpticksCfg<Opticks>* Opticks::getCfg()
{
    return m_cfg ; 
}

Timer* Opticks::getTimer()
{
    return m_timer ; 
}
Parameters* Opticks::getParameters()
{
    return m_parameters ; 
}

OpticksResource* Opticks::getResource()
{
    return m_resource  ; 
}
NState* Opticks::getState()
{
    return m_state  ; 
}

const char* Opticks::getLastArg()
{
   return m_lastarg ; 
}





void Opticks::setModeOverride(unsigned int mode)
{
    m_mode->setOverride(mode) ; 
}
bool Opticks::isRemoteSession()
{
    return SSys::IsRemoteSession();
}
bool Opticks::isCompute()
{
    return m_mode->isCompute() ;
}
bool Opticks::isInterop()
{
    return m_mode->isInterop() ;
}
bool Opticks::isCfG4()
{
    return m_mode->isCfG4(); 
}



void Opticks::setGeocache(bool geocache)
{
    m_geocache = geocache ; 
}
bool Opticks::isGeocache()
{
    return m_geocache ;
}

void Opticks::setInstanced(bool instanced)
{
   m_instanced = instanced ;
}
bool Opticks::isInstanced()
{
   return m_instanced ; 
}


void Opticks::setIntegrated(bool integrated)
{
   m_integrated = integrated ;
}
bool Opticks::isIntegrated()
{
   return m_integrated ; 
}





const glm::vec4& Opticks::getTimeDomain()
{
    return m_time_domain ; 
}
const glm::vec4& Opticks::getSpaceDomain()
{
    return m_space_domain ; 
}
const glm::vec4& Opticks::getWavelengthDomain()
{
    return m_wavelength_domain ; 
}
const glm::ivec4& Opticks::getSettings()
{
    return m_settings ; 
}


const glm::uvec4& Opticks::getSize()
{
    return m_size ; 
}
const glm::uvec4& Opticks::getPosition()
{
    return m_position ; 
}




void Opticks::setDetector(const char* detector)
{
    m_detector = detector ? strdup(detector) : NULL ; 
}


void Opticks::configureS(const char* , std::vector<std::string> )
{
}

void Opticks::configureI(const char* , std::vector<int> )
{
}

bool Opticks::isExit()
{
    return m_exit ; 
}
void Opticks::setExit(bool exit_)
{
    m_exit = exit_  ;   
    if(m_exit)
    {
        LOG(info) << "Opticks::setExit EXITING " ; 
        exit(EXIT_SUCCESS) ;
    }
}
 

void Opticks::init()
{
    m_mode = new OpticksMode(hasArg(COMPUTE_ARG_)) ; 

    m_cfg = new OpticksCfg<Opticks>("opticks", this,false);

    m_timer = new Timer("Opticks::");

    m_timer->setVerbose(true);

    m_timer->start();

    m_parameters = new Parameters ;  

    m_lastarg = m_argc > 1 ? strdup(m_argv[m_argc-1]) : NULL ;

    m_resource = new OpticksResource(this, m_envprefix, m_lastarg);

    setDetector( m_resource->getDetector() );

    LOG(trace) << "Opticks::init DONE " ;
}


void Opticks::defineEventSpec()
{
    const char* typ = getSourceType(); 
    const char* tag = getEventTag();
    std::string det = m_detector ? m_detector : "" ;
    std::string cat = m_cfg->getEventCat();   // overrides det for categorization of test events eg "rainbow" "reflect" "prism" "newton"

    const char* ntag = BStr::negate(tag) ; 

    m_spec  = new OpticksEventSpec(typ,  tag, det.c_str(), cat.c_str() );
    m_nspec = new OpticksEventSpec(typ, ntag, det.c_str(), cat.c_str() );
}

void Opticks::dumpArgs(const char* msg)
{
    LOG(info) << msg << " argc " << m_argc ;
    for(int i=0 ; i < m_argc ; i++) 
          std::cout << std::setw(3) << i << " : " << m_argv[i] << std::endl ;

   // PLOG by default writes to stdout so for easy splitting write 
   // mostly to stdout and just messages to stderr

}

void Opticks::configure()
{
    dumpArgs("Opticks::configure");  

    m_cfg->commandline(m_argc, m_argv);

    defineEventSpec();

    const std::string& ssize = m_cfg->getSize();

    if(!ssize.empty()) 
    {
        m_size = guvec4(ssize);
    }
    else if(m_cfg->hasOpt("fullscreen"))
    {
        m_size = glm::uvec4(2880,1800,2,0) ;
    } 
    else
    {
        m_size = glm::uvec4(2880,1704,2,0) ;  // 1800-44-44px native height of menubar  
    }


    const std::string& sposition = m_cfg->getPosition();
    if(!sposition.empty()) 
    {
        m_position = guvec4(sposition);
    }
    else
    {
        m_position = glm::uvec4(200,200,0,0) ;  // top left
    }



    const char* type = "State" ; 
    const std::string& stag = m_cfg->getStateTag();
    const char* subtype = stag.empty() ? NULL : stag.c_str() ; 

    std::string prefdir = getPreferenceDir(type, subtype);  

    LOG(debug) << "Opticks::configure " 
               << " m_size " << gformat(m_size)
               << " m_position " << gformat(m_position)
               << " prefdir " << prefdir
               ;
 

    // Below "state" is a placeholder name of the current state that never gets persisted, 
    // names like 001 002 are used for persisted states : ie the .ini files within the prefdir

    m_state = new NState(prefdir.c_str(), "state")  ;


    const std::string& mpfx = m_cfg->getMaterialPrefix();
    m_materialprefix = ( mpfx.empty() || isJuno()) ? NULL : strdup(mpfx.c_str()) ;

    LOG(debug) << "Opticks::configure DONE " ;
}





void Opticks::Summary(const char* msg)
{
    LOG(info) << msg 
              << " sourceCode " << getSourceCode() 
              << " sourceType " << getSourceType() 
              << " mode " << m_mode->description()
              ; 

    m_resource->Summary(msg);

    LOG(info) << msg << "DONE" ; 
}


int Opticks::getLastArgInt()
{
    return BStr::atoi(m_lastarg, -1 );
}

int Opticks::getInteractivityLevel()
{
    int interactivity = SSys::GetInteractivityLevel() ;
    if(hasOpt("noviz|compute")) interactivity = 0 ; 
    return interactivity  ;
}


void Opticks::setSpaceDomain(const glm::vec4& sd)
{
    setSpaceDomain(sd.x, sd.y, sd.z, sd.w )  ; 
}

void Opticks::setSpaceDomain(float x, float y, float z, float w)
{
    m_space_domain.x = x  ; 
    m_space_domain.y = y  ; 
    m_space_domain.z = z  ; 
    m_space_domain.w = w  ; 

    configureDomains();
}


void Opticks::configureDomains()
{
   // this is triggered by setSpaceDomain which is 
   // invoked when geometry is loaded 
   m_domains_configured = true ; 

   m_time_domain.x = 0.f  ;
   m_time_domain.y = m_cfg->getTimeMax() ;
   m_time_domain.z = m_cfg->getAnimTimeMax() ;
   m_time_domain.w = 0.f  ;

   m_wavelength_domain = getDefaultDomainSpec() ;  

   int e_rng_max = SSys::getenvint("CUDAWRAP_RNG_MAX",-1); 

   int x_rng_max = getRngMax() ;

   if(e_rng_max != x_rng_max)
       LOG(fatal) << "Opticks::configureDomains"
                  << " CUDAWRAP_RNG_MAX " << e_rng_max 
                  << " x_rng_max " << x_rng_max 
                  ;

   //assert(e_rng_max == x_rng_max && "Configured RngMax must match envvar CUDAWRAP_RNG_MAX and corresponding files, see cudawrap- ");    
}

std::string Opticks::description()
{
    std::stringstream ss ; 
    ss << "Opticks"
       << " time " << gformat(m_time_domain)  
       << " space " << gformat(m_space_domain) 
       << " wavelength " << gformat(m_wavelength_domain) 
       ;
    return ss.str();
}


const char* Opticks::getUDet()
{
    const char* det = m_detector ? m_detector : "" ;
    const std::string& cat = m_cfg->getEventCat();   // overrides det for categorization of test events eg "rainbow" "reflect" "prism" "newton"
    const char* cat_ = cat.c_str();
    return strlen(cat_) > 0 ? cat_ : det ;  
}


unsigned int Opticks::getSourceCode()
{
    unsigned int code ;
    if(     m_cfg->hasOpt("natural"))       code = NATURAL ;     // doing (CERENKOV | SCINTILLATION) would entail too many changes 
    else if(m_cfg->hasOpt("cerenkov"))      code = CERENKOV ;
    else if(m_cfg->hasOpt("scintillation")) code = SCINTILLATION ;
    else if(m_cfg->hasOpt("torch"))         code = TORCH ;
    else if(m_cfg->hasOpt("g4gun"))         code = G4GUN ;
    else                                    code = TORCH ;
    return code ;
}

const char* Opticks::getSourceType()
{
    unsigned int code = getSourceCode();
    return OpticksFlags::SourceTypeLowercase(code) ; 
}

const char* Opticks::getEventTag()
{
    if(!m_tag)
    {
        std::string tag = m_integrated ? m_cfg->getIntegratedEventTag() : m_cfg->getEventTag();
        m_tag = strdup(tag.c_str());
    }
    return m_tag ; 
}

const char* Opticks::getEventCat()
{
    if(!m_cat)
    {
        std::string cat = m_cfg->getEventCat();
        m_cat = strdup(cat.c_str());
    }
    return m_cat ; 
}


Index* Opticks::loadHistoryIndex()
{
    const char* typ = getSourceType();
    const char* tag = getEventTag();
    const char* udet = getUDet();
    return OpticksEvent::loadHistoryIndex(typ, tag, udet) ;
}
Index* Opticks::loadMaterialIndex()
{
    const char* typ = getSourceType();
    const char* tag = getEventTag();
    const char* udet = getUDet();
    return OpticksEvent::loadMaterialIndex(typ, tag, udet ) ;
}
Index* Opticks::loadBoundaryIndex()
{
    const char* typ = getSourceType();
    const char* tag = getEventTag();
    const char* udet = getUDet();
    return OpticksEvent::loadBoundaryIndex(typ, tag, udet ) ;
}


BDynamicDefine* Opticks::makeDynamicDefine()
{
    BDynamicDefine* dd = new BDynamicDefine();   // configuration used in oglrap- shaders
    dd->add("MAXREC",m_cfg->getRecordMax());    
    dd->add("MAXTIME",m_cfg->getTimeMax());    
    dd->add("PNUMQUAD", 4);  // quads per photon
    dd->add("RNUMQUAD", 2);  // quads per record 
    dd->add("MATERIAL_COLOR_OFFSET", (unsigned int)OpticksColors::MATERIAL_COLOR_OFFSET );
    dd->add("FLAG_COLOR_OFFSET", (unsigned int)OpticksColors::FLAG_COLOR_OFFSET );
    dd->add("PSYCHEDELIC_COLOR_OFFSET", (unsigned int)OpticksColors::PSYCHEDELIC_COLOR_OFFSET );
    dd->add("SPECTRAL_COLOR_OFFSET", (unsigned int)OpticksColors::SPECTRAL_COLOR_OFFSET );

    return dd ; 
}


OpticksEventSpec* Opticks::getEventSpec()
{
    return m_spec ; 
}


OpticksEvent* Opticks::makeEvent(bool ok)
{
    m_event_count += 1 ; 

    OpticksEvent* evt = new OpticksEvent(ok ? m_spec : m_nspec);
    evt->setId(m_event_count) ; 


    const char* x_udet = getUDet();
    const char* e_udet = evt->getUDet();

    bool match = strcmp(e_udet, x_udet) == 0 ;
    if(!match)
    {
        LOG(fatal) << "Opticks::makeEvent"
                   << " MISMATCH "
                   << " x_udet " << x_udet 
                   << " e_udet " << e_udet 
                   ;
    }
    assert(match);


    evt->setMode(m_mode);

    // formerly did configureDomains here, but thats confusing 
    // configureDomains now invoked when setSpaceDomain is called
    if(!m_domains_configured)
         LOG(fatal) << "Opticks::makeEvent"
                    << " domains MUST be configured by calling setSpaceDomain "
                    << " prior to makeEvent being possible "
                    << " description " << description()
                    ;

    assert(m_domains_configured);

    evt->setTimeDomain(getTimeDomain());
    evt->setSpaceDomain(getSpaceDomain());  
    evt->setWavelengthDomain(getWavelengthDomain());

    evt->setMaxRec(m_cfg->getRecordMax());
    evt->createSpec();   
    evt->createBuffers();  // not-allocated and with itemcount 0 
 
    // ctor args define the identity of the Evt, coming in from config
    // other params are best keep in m_parameters where they get saved/loaded  
    // with the evt 

    Parameters* parameters = evt->getParameters();

    unsigned int rng_max = getRngMax() ;
    unsigned int bounce_max = getBounceMax() ;
    unsigned int record_max = getRecordMax() ;

    parameters->add<unsigned int>("RngMax",    rng_max );
    parameters->add<unsigned int>("BounceMax", bounce_max );
    parameters->add<unsigned int>("RecordMax", record_max );

    parameters->add<std::string>("mode", m_mode->description());
    parameters->add<std::string>("cmdline", m_cfg->getCommandLine() );

    assert( parameters->get<unsigned int>("RngMax") == rng_max );
    assert( parameters->get<unsigned int>("BounceMax") == bounce_max );
    assert( parameters->get<unsigned int>("RecordMax") == record_max );

    // TODO: use these parameters from here, instead of from config again ?

    m_settings.x = bounce_max ;   
    m_settings.y = rng_max ;   
    m_settings.z = 0 ;   
    m_settings.w = record_max ;   

    return evt ; 
}


NPY<float>* Opticks::loadGenstep()
{
    const char* det = m_spec->getDet();
    const char* typ = m_spec->getTyp();
    const char* tag = m_spec->getTag();

    std::string path = NLoad::GenstepsPath(det, typ, tag);
    NPY<float>* gs = NPY<float>::load(path.c_str());
    if(!gs)
    {
        LOG(warning) << "Opticks::loadGenstep"
                     << " FAILED TO LOAD GENSTEPS FROM "
                     << " path " << path 
                     << " typ " << typ
                     << " tag " << tag
                     << " det " << det
                     ; 
        return NULL ;
    }
    return gs ; 
}





const char* Opticks::getMaterialPrefix()
{
    return m_materialprefix ; 
}

const char* Opticks::Material(const unsigned int mat)
{
    if(G_MATERIAL_NAMES == NULL)
    {
        LOG(info) << "Opticks::Material populating global G_MATERIAL_NAMES " ;
        G_MATERIAL_NAMES = new NPropNames("GMaterialLib") ;
    }
    return G_MATERIAL_NAMES ? G_MATERIAL_NAMES->getLine(mat) : "Opticks::Material-ERROR-NO-GMaterialLib" ;
}

std::string Opticks::MaterialSequence(const unsigned long long seqmat)
{
    LOG(info) << "Opticks::MaterialSequence"
              << " seqmat " << std::hex << seqmat << std::dec ; 

    std::stringstream ss ;
    assert(sizeof(unsigned long long)*8 == 16*4);
    for(unsigned int i=0 ; i < 16 ; i++)
    {
        unsigned long long m = (seqmat >> i*4) & 0xF ; 

        const char* mat = Opticks::Material(m)  ; 

        ss << ( mat ? mat : "NULL" ) << " " ;
    }
    return ss.str();
}


TorchStepNPY* Opticks::makeSimpleTorchStep()
{
    TorchStepNPY* torchstep = new TorchStepNPY(TORCH, 1);

    std::string config = m_cfg->getTorchConfig() ;

    if(!config.empty()) torchstep->configure(config.c_str());

    unsigned int photons_per_g4event = m_cfg->getNumPhotonsPerG4Event() ;  // only used for cfg4-
    torchstep->setNumPhotonsPerG4Event(photons_per_g4event);

    return torchstep ; 
}


unsigned int Opticks::getRngMax(){       return m_cfg->getRngMax(); }
unsigned int Opticks::getBounceMax() {   return m_cfg->getBounceMax(); }
unsigned int Opticks::getRecordMax() {   return m_cfg->getRecordMax() ; }
float Opticks::getEpsilon() {            return m_cfg->getEpsilon()  ; }
bool Opticks::hasOpt(const char* name) { return m_cfg->hasOpt(name); }






const char* Opticks::getDetector() { return m_resource->getDetector(); }
bool Opticks::isJuno() {    return m_resource->isJuno(); }
bool Opticks::isDayabay() { return m_resource->isDayabay(); }
bool Opticks::isPmtInBox(){ return m_resource->isPmtInBox(); }
bool Opticks::isOther() {   return m_resource->isOther(); }
bool Opticks::isValid() {   return m_resource->isValid(); }

const char* Opticks::getRNGInstallCacheDir() { return m_resource->getRNGInstallCacheDir(); } 

std::string Opticks::getPreferenceDir(const char* type, const char* subtype)
{
    const char* udet = getUDet();
    return m_resource->getPreferenceDir(type, udet, subtype);
}

std::string Opticks::getObjectPath(const char* name, unsigned int ridx, bool relative) 
{
   return relative ?
                     m_resource->getRelativePath(name, ridx)
                   :
                     m_resource->getObjectPath(name, ridx)
                   ; 
}
std::string Opticks::getRelativePath(const char* path) { return m_resource->getRelativePath(path); }

OpticksQuery*   Opticks::getQuery() {     return m_resource->getQuery(); }
OpticksColors*  Opticks::getColors() {    return m_resource->getColors(); }
OpticksFlags*   Opticks::getFlags() {     return m_resource->getFlags(); }
OpticksAttrSeq* Opticks::getFlagNames() { return m_resource->getFlagNames(); }

std::map<unsigned int, std::string> Opticks::getFlagNamesMap()
{   
    return m_resource->getFlagNamesMap() ;
}


Types*          Opticks::getTypes() {     return m_resource->getTypes(); }
Typ*            Opticks::getTyp() {       return m_resource->getTyp(); }
const char*     Opticks::getIdPath() {    return m_resource ? m_resource->getIdPath() : NULL ; }
const char*     Opticks::getIdFold() {    return m_resource ? m_resource->getIdFold() : NULL ; }
const char*     Opticks::getDetectorBase() {    return m_resource ? m_resource->getDetectorBase() : NULL ; }
const char*     Opticks::getMaterialMap() {  return m_resource ? m_resource->getMaterialMap() : NULL ; }
const char*     Opticks::getGDMLPath() {  return m_resource ? m_resource->getGDMLPath() : NULL ; }
const char*     Opticks::getDAEPath() {   return m_resource ? m_resource->getDAEPath() : NULL ; }
const char*     Opticks::getInstallPrefix() { return m_resource ? m_resource->getInstallPrefix() : NULL ; }

void Opticks::prepareInstallCache(const char* dir)
{
    // Moved save directory from IdPath to ResourceDir as
    // the IdPath is not really appropriate  
    // for things such as the flags that are a feature of an 
    // Opticks installation, not a feature of the geometry.
    // 
    // But ResourceDir is not appropriate either as that requires 
    // manual management via opticksdata repo.
    //
    //  So 
    // 
    //  TODO:
    //     incorporate resources saving into 
    //     the build process 
    //     ... currently this is done manually by 
    //
    //         OpticksSaveResources 
    //

    if(dir == NULL) dir = m_resource->getOKCInstallCacheDir() ;
    LOG(info) << "Opticks::saveResources " << ( dir ? dir : "NULL" )  ; 
    m_resource->saveFlags(dir);
    m_resource->saveTypes(dir);
}


void Opticks::setIdPathOverride(const char* idpath_tmp) // used for saves into non-standard locations whilst testing
{
    m_resource->setIdPathOverride(idpath_tmp);
}


void Opticks::cleanup()
{
    LOG(info) << "Opticks::cleanup" ;
}


void Opticks::configureF(const char* name, std::vector<float> values)
{
     if(values.empty())
     {   
         printf("Opticks::parameter_set %s no values \n", name);
     }   
     else    
     {   
         float vlast = values.back() ;

         LOG(info) << "Opticks::configureF"
                   << " name " << name 
                   << " vals " << values.size()
                   ;

         for(size_t i=0 ; i < values.size() ; i++ ) printf("%10.3f ", values[i]);
         printf(" : vlast %10.3f \n", vlast );

         //configure(name, vlast);  
     }   
}
 

