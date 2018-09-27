#ifdef _MSC_VER
// object allocated on the heap may not be aligned 16
// https://github.com/g-truc/glm/issues/235
// apparently fixed by 0.9.7.1 Release : currently on 0.9.6.3

#pragma warning( disable : 4316 )
#endif


#include "SLog.hh"
#include "SArgs.hh"
#include "SSys.hh"
// brap-
#include "BDynamicDefine.hh"
#include "BOpticksEvent.hh"
#include "BOpticksKey.hh"
#include "BFile.hh"
#include "BHex.hh"
#include "BStr.hh"
#include "BEnv.hh"
#include "PLOG.hh"
#include "Map.hh"


// npy-
#include "Timer.hpp"
#include "NParameters.hpp"
#include "TorchStepNPY.hpp"
#include "GLMFormat.hpp"
#include "NState.hpp"
#include "NPropNames.hpp"
#include "NGLTF.hpp"
#include "NScene.hpp"
#include "NLoad.hpp"
#include "NSlice.hpp"
#include "NSceneConfig.hpp"
#include "NLODConfig.hpp"
#include "NSnapConfig.hpp"

// okc-
#include "OpticksPhoton.h"
#include "OpticksFlags.hh"
#include "Opticks.hh"
#include "OpticksResource.hh"
#include "OpticksColors.hh"
#include "OpticksEvent.hh"
#include "OpticksRun.hh"
#include "OpticksMode.hh"
#include "OpticksEntry.hh"
#include "OpticksProfile.hh"
#include "OpticksAna.hh"
#include "OpticksDbg.hh"


#include "OpticksCfg.hh"




NPropNames* Opticks::G_MATERIAL_NAMES = NULL ; 



const float Opticks::F_SPEED_OF_LIGHT = 299.792458f ;  // mm/ns

const char* Opticks::COMPUTE_ARG_ = "--compute" ; 

// formerly of GPropertyLib, now booted upstairs
float        Opticks::DOMAIN_LOW  = 60.f ;
float        Opticks::DOMAIN_HIGH = 820.f ;  // has been 810.f for a long time  
float        Opticks::DOMAIN_STEP = 20.f ; 
unsigned int Opticks::DOMAIN_LENGTH = 39  ;

float        Opticks::FINE_DOMAIN_STEP = 1.f ; 
unsigned int Opticks::FINE_DOMAIN_LENGTH = 761  ;






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


In [10]: np.arange(60., 820.1, 20. )
Out[10]: 
array([  60.,   80.,  100.,  120.,  140.,  160.,  180.,  200.,  220.,
        240.,  260.,  280.,  300.,  320.,  340.,  360.,  380.,  400.,
        420.,  440.,  460.,  480.,  500.,  520.,  540.,  560.,  580.,
        600.,  620.,  640.,  660.,  680.,  700.,  720.,  740.,  760.,
        780.,  800.,  820.])

In [17]: np.arange(60., 820.1, 20. ).shape
Out[17]: (39,)

In [18]: np.arange(60., 820.1, 1. ).shape
Out[18]: (761,)



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

glm::vec4 Opticks::getDomainSpec(bool fine)
{
    glm::vec4 bd ;

    bd.x = DOMAIN_LOW ;
    bd.y = DOMAIN_HIGH ;
    bd.z = fine ? FINE_DOMAIN_STEP : DOMAIN_STEP ;
    bd.w = DOMAIN_HIGH - DOMAIN_LOW ;

    return bd ; 
}




glm::vec4 Opticks::getDefaultDomainReciprocalSpec()
{
    glm::vec4 rd ;
    rd.x = 1.f/DOMAIN_LOW ;
    rd.y = 1.f/DOMAIN_HIGH ;
    rd.z = 0.f ;
    rd.w = 0.f ;
    // not flipping order, only endpoints used for sampling, not the step 

    return rd ; 
}

glm::vec4 Opticks::getDomainReciprocalSpec(bool /*fine*/)
{
    glm::vec4 rd ;
    rd.x = 1.f/DOMAIN_LOW ;
    rd.y = 1.f/DOMAIN_HIGH ;
    rd.z = 0.f ;
    rd.w = 0.f ;
    // not flipping order, only endpoints used for sampling, not the step 

    return rd ; 
}




int Opticks::getRC() const
{
    return m_rc ; 
}
void Opticks::setRC(int rc, const char* rcmsg)
{
    m_rc = rc ; 
    m_rcmsg = rcmsg ? strdup(rcmsg) : NULL ; 
    dumpRC();
}

const char* Opticks::getRCMessage() const 
{
    return m_rcmsg ; 
}

int Opticks::rc() const 
{
    dumpRC();
    return m_rc ; 
}


void Opticks::dumpRC() const 
{
    LOG( m_rc == 0 ? info : fatal) 
           << " rc " << m_rc 
           << " rcmsg : " << ( m_rcmsg ? m_rcmsg : "-" ) 
           ;
}



Opticks*    Opticks::fInstance = NULL ; 

bool Opticks::HasInstance() 
{
    return fInstance != NULL ; 
}

bool Opticks::HasKey()
{
    assert( fInstance ) ; 
    return fInstance->hasKey() ; 
}

Opticks* Opticks::GetInstance()
{
     if(fInstance == NULL )
     {
         const char* argforced = SSys::getenvvar("OPTICKS_INTERNAL_ARGS") ; 
         Opticks* ok = new Opticks(0,0, argforced);  
         ok->setInternal(true);   // internal means was instanciated within Opticks::GetInstance
     }
     assert( fInstance != NULL ) ; // Opticks ctor should have defined THE instance
     return fInstance ; 
}


Opticks::Opticks(int argc, char** argv, const char* argforced )
     :
       m_log(new SLog("Opticks::Opticks")),
       m_ok(this),
       m_sargs(new SArgs(argc, argv, argforced)), 
       m_argc(m_sargs->argc),
       m_argv(m_sargs->argv),
       m_dumpenv(m_sargs->hasArg("--dumpenv")),
       m_envkey(m_sargs->hasArg("--envkey") ? BOpticksKey::SetKey(NULL) : false),  // see tests/OpticksEventDumpTest.cc makes sensitive to OPTICKS_KEY
       m_production(m_sargs->hasArg("--production")),
       m_profile(new OpticksProfile("Opticks",m_sargs->hasArg("--stamp"))),
       m_materialprefix(NULL),
       m_photons_per_g4event(0), 

       m_spec(NULL),
       m_nspec(NULL),
       m_resource(NULL),
       m_state(NULL),
       m_apmtslice(NULL),
       m_apmtmedium(NULL),

       m_exit(false),
       m_compute(false),
       m_geocache(false),
       m_instanced(true),

       m_lastarg(NULL),

       m_configured(false),
       m_cfg(NULL),
       m_timer(NULL),
       m_parameters(NULL),
       m_scene_config(NULL),
       m_lod_config(NULL),
       m_snap_config(NULL),
       m_detector(NULL),
       m_event_count(0),
       m_domains_configured(false),
       m_mode(NULL),
       m_run(new OpticksRun(this)),
       m_evt(NULL),
       m_ana(new OpticksAna(this)),
       m_dbg(new OpticksDbg(this)),
       m_rc(0),
       m_rcmsg(NULL),
       m_tagoffset(0),
       m_verbosity(0),
       m_internal(false)
{
       OK_PROFILE("Opticks::Opticks");


       if(fInstance != NULL)
       {
          LOG(fatal) << " SECOND OPTICKS INSTANCE " ;  
       }
       //assert( fInstance == NULL ); // should only ever be one instance 

       fInstance = this ; 

       init();
       (*m_log)("DONE");
}


bool Opticks::isDumpEnv() const 
{
    return m_dumpenv ; 
}
bool Opticks::isInternal() const 
{
    return m_internal ; 
}
void Opticks::setInternal(bool internal)  
{
    m_internal = internal ; 
}



std::string Opticks::getArgLine()
{
    return m_sargs->getArgLine();
}


template <typename T>
void Opticks::profile(T label)
{
    m_profile->stamp<T>(label, m_tagoffset);
   // m_tagoffset is set by Opticks::makeEvent
}
void Opticks::dumpProfile(const char* msg, const char* startswith, const char* spacewith, double tcut)
{
   m_profile->dump(msg, startswith, spacewith, tcut);
}
void Opticks::saveProfile()
{
   m_profile->save();
}

void Opticks::postpropagate()
{
   saveProfile();
   dumpProfile("Opticks::postpropagate", NULL, "OpticksRun::createEvent.BEG", 0.0001 );  // spacwith spacing at start if each evt

   // startswith filtering 
   dumpProfile("Opticks::postpropagate", "OPropagator::launch");  
   dumpProfile("Opticks::postpropagate", "CG4::propagate");  

   dumpParameters("Opticks::postpropagate");
}

void Opticks::ana()
{
   LOG(error) << "Opticks::ana START" ; 
   m_ana->run();
   LOG(error) << "Opticks::ana DONE" ; 
}



NPY<unsigned>* Opticks::getMaskBuffer() const 
{
    return m_dbg->getMaskBuffer() ;  
}
const std::vector<unsigned>&  Opticks::getMask() const 
{
    return m_dbg->getMask();
}
unsigned Opticks::getMaskIndex(unsigned idx) const
{
    bool mask = hasMask();  
    if(!mask)
        LOG(warning) << "Opticks::getMaskIndex BUT there is no mask " ; 

    return mask ? m_dbg->getMaskIndex(idx) : idx ;
}
bool Opticks::hasMask() const 
{
    return m_dbg->getMask().size() > 0 ; 
}


bool Opticks::isDbgPhoton(unsigned record_id) const 
{
   return m_dbg->isDbgPhoton(record_id);
}
bool Opticks::isOtherPhoton(unsigned photon_id) const 
{
   return m_dbg->isOtherPhoton(photon_id);
}
bool Opticks::isMaskPhoton(unsigned photon_id) const 
{
   return m_dbg->isMaskPhoton(photon_id);
}
bool Opticks::isX4PolySkip(unsigned lvIdx) const 
{
   return m_dbg->isX4PolySkip(lvIdx);
}






bool Opticks::isDbgPhoton(int event_id, int track_id)
{
    unsigned record_id = event_id*m_photons_per_g4event + track_id ; 
    return m_dbg->isDbgPhoton(record_id);
}
bool Opticks::isOtherPhoton(int event_id, int track_id)
{
    unsigned record_id = event_id*m_photons_per_g4event + track_id ; 
    return m_dbg->isOtherPhoton(record_id);
}
bool Opticks::isMaskPhoton(int event_id, int track_id)
{
    unsigned record_id = event_id*m_photons_per_g4event + track_id ; 
    return m_dbg->isMaskPhoton(record_id);
}





unsigned Opticks::getNumDbgPhoton() const 
{
    return m_dbg->getNumDbgPhoton();
}
unsigned Opticks::getNumOtherPhoton() const 
{
    return m_dbg->getNumOtherPhoton();
}
unsigned Opticks::getNumMaskPhoton() const 
{
    return m_dbg->getNumMaskPhoton();
}

const std::vector<unsigned>&  Opticks::getDbgIndex()
{
    return m_dbg->getDbgIndex();
}
const std::vector<unsigned>&  Opticks::getOtherIndex()
{
    return m_dbg->getOtherIndex();
}


int Opticks::getDebugIdx() const 
{
   return m_cfg->getDebugIdx();
}
int Opticks::getDbgNode() const 
{
   return m_cfg->getDbgNode();
}
int Opticks::getStack() const 
{
   return m_cfg->getStack();
}
int Opticks::getMeshVerbosity() const 
{
   return m_cfg->getMeshVerbosity();
}




const char* Opticks::getDbgMesh() const 
{
   const std::string& dbgmesh = m_cfg->getDbgMesh();
   return dbgmesh.empty() ? NULL : dbgmesh.c_str() ;
}





void Opticks::init()
{
    m_mode = new OpticksMode(hasArg(COMPUTE_ARG_)) ; 

    m_cfg = new OpticksCfg<Opticks>("opticks", this,false);

    m_timer = new Timer("Opticks::");

    m_timer->setVerbose(true);

    m_timer->start();

    m_parameters = new NParameters ;  

    m_lastarg = m_argc > 1 ? strdup(m_argv[m_argc-1]) : NULL ;


    LOG(verbose) << " Opticks::init start instanciate resource " ;
    m_resource = new OpticksResource(this, m_lastarg);
    LOG(verbose) << " Opticks::init done instanciate resource " ;
    setDetector( m_resource->getDetector() );

    LOG(debug) << "Opticks::init DONE " << m_resource->desc()  ;

    //configure(); 
}

/*
void Opticks::setupResource()
{


}
*/



int Opticks::getArgc()
{
    return m_argc ; 
}
char** Opticks::getArgv()
{
    return m_argv ; 
}
char* Opticks::getArgv0()
{
    return m_argc > 0 && m_argv ? m_argv[0] : NULL ; 
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
OpticksCfg<Opticks>* Opticks::getCfg() const 
{
    return m_cfg ; 
}

const char* Opticks::getRenderMode() const 
{
    const std::string& s = m_cfg->getRenderMode();
    return s.c_str();
}

const char* Opticks::getLVSDName() const 
{
    const std::string& s = m_cfg->getLVSDName();
    return s.empty() ? NULL : s.c_str();
}

const char* Opticks::getCathode() const 
{
    const std::string& s = m_cfg->getCathode();
    return s.c_str();
}

const char* Opticks::getCerenkovClass() const 
{
    const std::string& s = m_cfg->getCerenkovClass();
    return s.c_str();
}

const char* Opticks::getScintillationClass() const 
{
    const std::string& s = m_cfg->getScintillationClass();
    return s.c_str();
}








bool Opticks::isG4CodeGen() const 
{
    return m_cfg->hasOpt("g4codegen") ;
}

bool Opticks::isPrintEnabled() const 
{
    return m_cfg->hasOpt("printenabled") ;
}

bool Opticks::isPrintIndexLog() const 
{
    return m_cfg->hasOpt("pindexlog") ;
}

bool Opticks::isXAnalytic() const 
{
    return m_cfg->hasOpt("xanalytic") ;
}



int Opticks::getPrintIndex(unsigned dim) const 
{
    glm::ivec3 idx ; 
    int pindex = -1 ; 
    if(getPrintIndex(idx)) 
    {
        switch(dim)
        {
            case 0: pindex = idx.x ; break ; 
            case 1: pindex = idx.y ; break ; 
            case 2: pindex = idx.z ; break ; 
        }
    }
    return pindex ; 
}

bool Opticks::getPrintIndex(glm::ivec3& idx) const 
{
    const char* pindex = getPrintIndexString();
    if(!pindex) return false ; 
    idx = givec3(pindex);
    return true ; 
}

const char* Opticks::getPrintIndexString() const 
{
    const std::string& printIndex = m_cfg->getPrintIndex();
    return printIndex.empty() ? NULL : printIndex.c_str();
}
const char* Opticks::getDbgIndex() const 
{
    const std::string& dbgIndex = m_cfg->getDbgIndex();
    return dbgIndex.empty() ? NULL : dbgIndex.c_str();
}






const char* Opticks::getDbgCSGPath()
{
    const std::string& dbgcsgpath = m_cfg->getDbgCSGPath();
    return dbgcsgpath.empty() ? NULL : dbgcsgpath.c_str();
}

unsigned Opticks::getSeed() const 
{
    return m_cfg->getSeed();
}

int Opticks::getLoadVerbosity() const 
{
    return m_cfg->getLoadVerbosity();
}
int Opticks::getImportVerbosity() const 
{
    return m_cfg->getImportVerbosity();
}





OpticksRun* Opticks::getRun()
{
    return m_run ;  
}
OpticksEvent* Opticks::getEvent() const 
{
    return m_run->getEvent()  ; 
}

Timer* Opticks::getTimer()
{
    OpticksEvent* evt = m_run->getEvent();
    return evt ? evt->getTimer() : m_timer ; 
}



NParameters* Opticks::getParameters()
{
    return m_parameters ; 
}
void Opticks::dumpParameters(const char* msg)
{
    m_parameters->dump(msg);
}


OpticksResource* Opticks::getResource()
{
    return m_resource  ; 
}
void Opticks::dumpResource() const 
{
    return m_resource->Dump()  ; 
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




bool Opticks::isAlign() const  // --align
{
   return m_cfg->hasOpt("align");
}
bool Opticks::isDbgNoJumpZero() const  // --dbgnojumpzero
{
   return m_cfg->hasOpt("dbgnojumpzero");
}
bool Opticks::isDbgSkipClearZero() const  // --dbgskipclearzero
{
   return m_cfg->hasOpt("dbgskipclearzero");
}
bool Opticks::isDbgKludgeFlatZero() const  // --dbgkludgeflatzero
{
   return m_cfg->hasOpt("dbgkludgeflatzero");
}
bool Opticks::isDbgTex() const  // --dbgtex
{
   return m_cfg->hasOpt("dbgtex");
}








bool Opticks::isReflectCheat() const  // --reflectcheat
{
   return m_cfg->hasOpt("reflectcheat");
}
bool Opticks::isSave() const 
{
    return m_cfg->hasOpt("save");  
}
bool Opticks::isLoad() const
{
   // --noload trumps --load
    return m_cfg->hasOpt("load") && !m_cfg->hasOpt("noload"); 
}
bool Opticks::isTracer() const
{
    return m_cfg->hasOpt("tracer") ;
}

bool Opticks::isRayLOD() const
{
    return m_cfg->hasOpt("raylod") ;
}

bool Opticks::isMaterialDbg() const
{
    return m_cfg->hasOpt("materialdbg") ;
}

bool Opticks::isDbgAnalytic() const
{
    return m_cfg->hasOpt("dbganalytic") ;
}

bool Opticks::isDbgSurf() const
{
    return m_cfg->hasOpt("dbgsurf") ;
}
bool Opticks::isDbgBnd() const
{
    return m_cfg->hasOpt("dbgbnd") ;
}

bool Opticks::isDbgRec() const
{
    return m_cfg->hasOpt("dbgrec") ;
}
bool Opticks::isDbgZero() const
{
    return m_cfg->hasOpt("dbgzero") ;
}
bool Opticks::isRecPoi() const
{
    return m_cfg->hasOpt("recpoi") ;
}
bool Opticks::isRecPoiAlign() const
{
    return m_cfg->hasOpt("recpoialign") ;
}


bool Opticks::isRecCf() const
{
    return m_cfg->hasOpt("reccf") ;
}





bool Opticks::isDbgTorch() const
{
    return m_cfg->hasOpt("torchdbg") ;
}
bool Opticks::isDbgSource() const
{
    return m_cfg->hasOpt("sourcedbg") ;
}
bool Opticks::isDbgClose() const
{
    return m_cfg->hasOpt("dbgclose") ;
}







std::string Opticks::brief()
{
    std::stringstream ss ; 
    ss << "OK" ;
    ss << ( isCompute() ? " COMPUTE" : " INTEROP" ) ;
    ss << ( isProduction() ? " PRODUCTION" : " DEVELOPMENT" ) ;
    return ss.str();
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
bool Opticks::isProduction()
{
   return m_production ; 
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


unsigned long long Opticks::getDbgSeqmat()
{
    const std::string& seqmat = m_cfg->getDbgSeqmat();
    return BHex<unsigned long long>::hex_lexical_cast( seqmat.c_str() );
}
unsigned long long Opticks::getDbgSeqhis()
{
    const std::string& seqhis = m_cfg->getDbgSeqhis();
    return BHex<unsigned long long>::hex_lexical_cast( seqhis.c_str() );
}

const std::string& Opticks::getSeqMapString() const 
{
    return m_cfg->getSeqMap() ;
}

void Opticks::setSeqMapString(const char* seqmap)
{
    m_cfg->setSeqMap(seqmap);
}


bool Opticks::getSeqMap(unsigned long long& seqhis, unsigned long long& seqval)
{
    const std::string& seqmap = m_cfg->getSeqMap();
    if(seqmap.empty()) return false ; 
    char edelim = BStr::HasChar(seqmap, ',') ? ',' : ' ' ; 
    OpticksFlags::AbbrevToFlagValSequence(seqhis, seqval, seqmap.c_str(), edelim );
    return true ; 
}



float Opticks::getFxRe()
{
    std::string fxre = m_cfg->getFxReConfig();
    return BStr::atof(fxre.c_str(), 0);
}
float Opticks::getFxAb()
{
    std::string fxab = m_cfg->getFxAbConfig();
    return BStr::atof(fxab.c_str(), 0);
}
float Opticks::getFxSc()
{
    std::string fxsc = m_cfg->getFxScConfig();
    return BStr::atof(fxsc.c_str(), 0);
}


bool Opticks::isAnalyticPMTLoad()
{
    return m_cfg->hasOpt("apmtload");
}




unsigned Opticks::getAnalyticPMTIndex()
{
    return m_cfg->getAnalyticPMTIndex();
}

const char* Opticks::getAnalyticPMTMedium()
{
    if(m_apmtmedium == NULL)
    {
        std::string cmed = m_cfg->getAnalyticPMTMedium() ;
        std::string dmed = m_resource->getDefaultMedium()  ; 
        LOG(verbose) 
            << " cmed " << cmed 
            << " cmed.empty " << cmed.empty()
            << " dmed " << dmed 
            << " dmed.empty " << dmed.empty()
            ;

        m_apmtmedium = !cmed.empty() ? strdup(cmed.c_str()) : strdup(dmed.c_str()) ;
    }
    return m_apmtmedium ;
}

const int Opticks::getDefaultFrame() const 
{
    return m_resource->getDefaultFrame() ; 
}


NSlice* Opticks::getAnalyticPMTSlice()
{
    if(m_apmtslice == 0)
    {
        std::string sli = m_cfg->getAnalyticPMTSlice() ; 
        if(!sli.empty()) m_apmtslice = new NSlice(sli.c_str());
    }
    return m_apmtslice ; 
}


const char* Opticks::getSensorSurface()
{
    return m_resource->getSensorSurface() ;
}







int  Opticks::getGLTF() const 
{
    return m_cfg->getGLTF(); 
}
int  Opticks::getGLTFTarget() const 
{
    return m_cfg->getGLTFTarget(); 
}

bool Opticks::isGLTF() const 
{
    return getGLTF() > 0 ; 
}

const char* Opticks::getGLTFPath() const { return m_resource->getGLTFPath() ; }
const char* Opticks::getSrcGLTFPath() const { return m_resource->getSrcGLTFPath() ; }
const char* Opticks::getG4CodeGenDir() const { return m_resource->getG4CodeGenDir() ; }
const char* Opticks::getCacheMetaPath() const { return m_resource->getCacheMetaPath() ; } 




const char* Opticks::getSrcGLTFBase() const  // config base and name only used whilst testing with gltf >= 100
{
    int gltf = getGLTF();
    const char* path = getSrcGLTFPath() ;
    if(!path) return NULL ; 
    std::string base = gltf < 100 ? BFile::ParentDir(path) : m_cfg->getSrcGLTFBase() ;
    return strdup(base.c_str()) ; 
}


const char* Opticks::getSrcGLTFName() const 
{
    int gltf = getGLTF();
    const char* path = getSrcGLTFPath() ;
    if(!path) return NULL ; 
    std::string name = gltf < 100 ? BFile::Name(path) : m_cfg->getSrcGLTFName()  ;
    return strdup(name.c_str()) ; 
}


bool Opticks::hasSrcGLTF() const 
{
    // lookahead to what GScene::GScene will do
    return NGLTF::Exists(getSrcGLTFBase(), getSrcGLTFName()) ;
}


void Opticks::configureCheckGeometryFiles() 
{
    if(isGLTF() && !hasSrcGLTF())
    {
        LOG(fatal) << "gltf option is selected but there is no gltf file " ;
        LOG(fatal) << " SrcGLTFBase " << getSrcGLTFBase() ;
        LOG(fatal) << " SrcGLTFName " << getSrcGLTFName() ;
        LOG(fatal) << "Try to create the GLTF from GDML with eg:  op --j1707 --gdml2gltf  "  ;
        
        //setExit(true); 
        //assert(0);
    }
} 



const char* Opticks::getGLTFConfig()
{
    return m_cfg->getGLTFConfig().c_str() ; 
}

int  Opticks::getLayout() const 
{
    return m_cfg->getLayout(); 
}




const char* Opticks::getGPUMonPath() const 
{
    const std::string& gpumonpath = m_cfg->getGPUMonPath() ;
    return gpumonpath.c_str() ;
}
bool Opticks::isGPUMon() const 
{
    return m_cfg->hasOpt("gpumon");
}


bool Opticks::isTest() const 
{
    return m_cfg->hasOpt("test");
}
bool Opticks::isTestAuto() const 
{
    return m_cfg->hasOpt("testauto");
}

const char* Opticks::getTestConfig() const 
{
    const std::string& tc = m_cfg->getTestConfig() ;
    return tc.empty() ? NULL : tc.c_str() ; 
}




bool Opticks::isG4Snap() const 
{
    return m_cfg->hasOpt("g4snap");
}
const char* Opticks::getG4SnapConfigString()  const 
{
    return m_cfg->getG4SnapConfig().c_str() ; 
}

const char* Opticks::getSnapConfigString()
{
    return m_cfg->getSnapConfig().c_str() ; 
}

const char* Opticks::getLODConfigString()
{
    return m_cfg->getLODConfig().c_str() ; 
}
int  Opticks::getLOD()
{
    return m_cfg->getLOD(); 
}

NLODConfig* Opticks::getLODConfig()
{
    if(m_lod_config == NULL)
    {
        m_lod_config = new NLODConfig(getLODConfigString());
    }
    return m_lod_config ; 
}


NSnapConfig* Opticks::getSnapConfig()
{
    if(m_snap_config == NULL)
    {
        m_snap_config = new NSnapConfig(getSnapConfigString());
    }
    return m_snap_config ; 
}







NSceneConfig* Opticks::getSceneConfig()
{
    if(m_scene_config == NULL)
    {
        m_scene_config = new NSceneConfig(getGLTFConfig());
    }
    return m_scene_config ; 
}



int  Opticks::getTarget() const
{
    return m_cfg->getTarget(); 
}
int  Opticks::getAlignLevel() const 
{
    return m_cfg->getAlignLevel(); 
}




 
unsigned Opticks::getVerbosity() const 
{
    return m_verbosity ; 
}
void  Opticks::setVerbosity(unsigned verbosity)
{
    m_verbosity = verbosity ; 
}





/**
Opticks::defineEventSpec
-------------------------


**/


void Opticks::defineEventSpec()
{
    const char* typ = getSourceType(); 

    //std::string tag_ = m_integrated ? m_cfg->getIntegratedEventTag() : m_cfg->getEventTag();
    std::string tag_ = m_cfg->getEventTag();
    const char* tag = tag_.c_str();
    const char* ntag = BStr::negate(tag) ; 

    std::string det = m_detector ? m_detector : "" ;
    std::string cat = m_cfg->getEventCat();   // overrides det for categorization of test events eg "rainbow" "reflect" "prism" "newton"

    m_spec  = new OpticksEventSpec(typ,  tag, det.c_str(), cat.c_str() );
    m_nspec = new OpticksEventSpec(typ, ntag, det.c_str(), cat.c_str() );


    LOG(info) 
         << " typ " << typ
         << " tag " << tag
         << " det " << det 
         << " cat " << cat 
         ;

}

void Opticks::dumpArgs(const char* msg)
{
    LOG(info) << msg << " argc " << m_argc ;
    for(int i=0 ; i < m_argc ; i++) 
          std::cout << std::setw(3) << i << " : " << m_argv[i] << std::endl ;

   // PLOG by default writes to stdout so for easy splitting write 
   // mostly to stdout and just messages to stderr

}


void Opticks::checkOptionValidity()
{
    if(isInterop() && getMultiEvent() > 1)
    {
        LOG(fatal) << "INTEROP mode with --multievent greater than 1 is not supported " ;  
        setExit(true);
    }
}



void Opticks::configure()
{
    if(m_configured) 
    {
        LOG(fatal) << " configured already " ; 
        return ; 
    }
    m_configured = true ; 

    dumpArgs("Opticks::configure");  


    m_cfg->commandline(m_argc, m_argv);

    checkOptionValidity();

    defineEventSpec();

    m_profile->setDir(getEventFold());

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
#ifdef __APPLE__
        m_size = glm::uvec4(2880,1704,2,0) ;  // 1800-44-44px native height of menubar  
#else
        m_size = glm::uvec4(1920,1080,1,0) ;
#endif
    }


    const std::string& sposition = m_cfg->getPosition();
    if(!sposition.empty()) 
    {
        m_position = guvec4(sposition);
    }
    else
    {
#ifdef __APPLE__
        m_position = glm::uvec4(200,200,0,0) ;  // top left
#else
        m_position = glm::uvec4(100,100,0,0) ;  // top left
#endif
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


    m_photons_per_g4event = m_cfg->getNumPhotonsPerG4Event();
    m_dbg->postconfigure();

    m_verbosity = m_cfg->getVerbosity(); 

    configureCheckGeometryFiles();

    configureGeometryHandling();


    if(hasOpt("dumpenv")) 
         BEnv::dumpEnvironment("Opticks::configure --dumpenv", "G4,OPTICKS,DAE,IDPATH") ; 


    LOG(debug) << "Opticks::configure DONE " 
              << " verbosity " << m_verbosity 
              ;

}




void Opticks::configureGeometryHandling()
{
    bool geocache = !m_cfg->hasOpt("nogeocache") ;
    bool instanced = !m_cfg->hasOpt("noinstanced") ; // find repeated geometry 

    LOG(debug) << "Opticks::configureGeometryHandling"
              << " geocache " << geocache 
              << " instanced " << instanced
              ;   

    setGeocache(geocache);
    setInstanced(instanced); // find repeated geometry 
}








void Opticks::dump(const char* msg) 
{
    LOG(info) << msg  ;

    const char* dbgmesh = getDbgMesh();

    std::cout
         << " argline " << std::setw(30) << getArgLine() << std::endl 
         << " dbgnode " << std::setw(30) << getDbgNode() << std::endl 
         << " dbgmesh " << std::setw(30) << ( dbgmesh ? dbgmesh : "-" ) << std::endl
         ;

}


void Opticks::Summary(const char* msg)
{
    LOG(info) << msg 
              << " sourceCode " << getSourceCode() 
              << " sourceType " << getSourceType() 
              << " mode " << m_mode->description()
              ; 

    m_resource->Summary(msg);

    const char* srcgltfbase = getSrcGLTFBase() ;
    const char* srcgltfname = getSrcGLTFName() ;


    std::cout
        << std::setw(40) << " isInternal "
        << std::setw(40) << isInternal()
        << std::endl
        << std::setw(40) << " Verbosity "
        << std::setw(40) << getVerbosity()
        << std::endl
        << std::setw(40) << " AnalyticPMTMedium "
        << std::setw(40) << getAnalyticPMTMedium()
        << std::endl
        << std::setw(40) << " SrcGLTFBase "
        << std::setw(40) << ( srcgltfbase ? srcgltfbase : "-" )
        << std::endl
        << std::setw(40) << " SrcGLTFName "
        << std::setw(40) << ( srcgltfname ? srcgltfname : "-" )
        << std::endl
        ;

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

int Opticks::getMultiEvent()
{    
    return m_cfg->getMultiEvent();
}
int Opticks::getRestrictMesh()
{    
    return m_cfg->getRestrictMesh();
}




float Opticks::getTimeMin()
{
    return m_time_domain.x ; 
}
float Opticks::getTimeMax()
{
    return m_time_domain.y ; 
}
float Opticks::getAnimTimeMax()
{
    return m_time_domain.z ; 
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
       LOG(verbose) << "Opticks::configureDomains"
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





/**
Opticks::getSourceCode
-------------------------

This is not the final word, see OpticksGen 

*live-gensteps* 
    G4GUN: collected from a live CG4 instance  

*loaded-from-file*
    CERENKOV SCINTILLATION NATURAL

*fabricated-from-config*
    TORCH  

**/

unsigned int Opticks::getSourceCode() const
{
    unsigned int code ;
    if(     m_cfg->hasOpt("natural"))       code = NATURAL ;     // doing (CERENKOV | SCINTILLATION) would entail too many changes 
    else if(m_cfg->hasOpt("cerenkov"))      code = CERENKOV ;
    else if(m_cfg->hasOpt("scintillation")) code = SCINTILLATION ;
    else if(m_cfg->hasOpt("torch"))         code = TORCH ;
    else if(m_cfg->hasOpt("machinery"))     code = MACHINERY ;
    else if(m_cfg->hasOpt("g4gun"))         code = G4GUN ;           // <-- dynamic : photon count not known ahead of time
    else if(m_cfg->hasOpt("emitsource"))    code = EMITSOURCE ;      
    else if(m_cfg->hasOpt("primarysource")) code = PRIMARYSOURCE ;   // <-- dynamic : photon count not known ahead of time
    else                                    code = TORCH ;             
    return code ;
}

// not-definitive see OpticksGen CGenerator
const char* Opticks::getSourceType() const
{
    unsigned int code = getSourceCode();
    return OpticksFlags::SourceTypeLowercase(code) ; 
}

bool Opticks::isFabricatedGensteps() const
{
    unsigned int code = getSourceCode() ;
    return code == TORCH || code == MACHINERY ;  
}

bool Opticks::isEmbedded() const { return hasOpt("embedded"); }
bool Opticks::isLiveGensteps() const {  return hasOpt("live"); }
bool Opticks::isNoInputGensteps() const { return hasOpt("load|nopropagate") ; } 




char Opticks::getEntryCode() const  // debug switching of OptiX launch program  
{
   return OpticksEntry::CodeFromConfig(m_cfg);
}
const char* Opticks::getEntryName() const
{  
    char code = getEntryCode();
    return OpticksEntry::Name(code);
}
bool Opticks::isTrivial() const
{
   char code = getEntryCode();
   return  code == 'T' ; 
}
bool Opticks::isSeedtest() const
{
   char code = getEntryCode();
   return  code == 'S' ; 
}




const char* Opticks::getEventFold() const
{
    return m_spec ? m_spec->getFold() : NULL ;
}

const char* Opticks::getEventDir() const 
{
    return m_spec ? m_spec->getDir() : NULL ;
}
const char* Opticks::getEventTag() const
{
    return m_spec->getTag();
}
int Opticks::getEventITag() const
{
    return m_spec->getITag() ; 
}
const char* Opticks::getEventCat() const
{
    return m_spec->getCat();
}






Index* Opticks::loadHistoryIndex()
{
    const char* typ = getSourceType();
    const char* tag = getEventTag();
    const char* udet = getUDet();

    Index* index = OpticksEvent::loadHistoryIndex(typ, tag, udet) ;

    return index ; 
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




OpticksEvent* Opticks::loadEvent(bool ok, unsigned tagoffset)
{
    OpticksEvent* evt = OpticksEvent::make(ok ? m_spec : m_nspec, tagoffset);

    evt->setOpticks(this);

    bool verbose = false ; 
    evt->loadBuffers(verbose);


    LOG(info) << "Opticks::loadEvent"
              << " tagdir " << evt->getTagDir() 
              << " " << ( evt->isNoLoad() ? "FAILED" : "SUCEEDED" )
              ; 


    return evt ; 
}

void Opticks::setTagOffset(unsigned tagoffset)
{
    m_tagoffset = tagoffset ; 
}
unsigned Opticks::getTagOffset()
{
    return m_tagoffset ; 
}
OpticksEvent* Opticks::makeEvent(bool ok, unsigned tagoffset)
{
    setTagOffset(tagoffset) ; 

    OpticksEvent* evt = OpticksEvent::make(ok ? m_spec : m_nspec, tagoffset);

    evt->setId(m_event_count) ;   // starts from id 0 
    evt->setOpticks(this);
    evt->setEntryCode(getEntryCode());


    LOG(debug) << "Opticks::makeEvent" 
               << ( ok ? " OK " : " G4 " )
               << " tagoffset " << tagoffset 
               << " id " << evt->getId() 
               ;

    m_event_count += 1 ; 


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




    unsigned int rng_max = getRngMax() ;
    unsigned int bounce_max = getBounceMax() ;
    unsigned int record_max = getRecordMax() ;
    
    evt->setTimeDomain(getTimeDomain());
    evt->setSpaceDomain(getSpaceDomain());  
    evt->setWavelengthDomain(getWavelengthDomain());

    evt->setMaxRng(rng_max);
    evt->setMaxRec(record_max);
    evt->setMaxBounce(bounce_max);

    evt->createSpec();   
    evt->createBuffers();  // not-allocated and with itemcount 0 
 
    // ctor args define the identity of the Evt, coming in from config
    // other params are best keep in m_parameters where they get saved/loaded  
    // with the evt 

    NParameters* parameters = evt->getParameters();
    parameters->add<unsigned int>("RngMax",    rng_max );
    parameters->add<unsigned int>("BounceMax", bounce_max );
    parameters->add<unsigned int>("RecordMax", record_max );

    parameters->add<std::string>("mode", m_mode->description());
    parameters->add<std::string>("cmdline", m_cfg->getCommandLine() );

    parameters->add<std::string>("EntryCode", BStr::ctoa(getEntryCode()) );
    parameters->add<std::string>("EntryName", getEntryName() );

    evt->setCreator(getArgv0()) ;  

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


void Opticks::setOptiXVersion(unsigned version)
{
    m_parameters->add<unsigned>("OptiXVersion",version);
}
void Opticks::setGeant4Version(unsigned version)
{
    m_parameters->add<unsigned>("Geant4Version",version);
}

unsigned Opticks::getOptiXVersion()
{
    return m_parameters->get<unsigned>("OptiXVersion",0);
}
unsigned Opticks::getGeant4Version()
{
    return m_parameters->get<unsigned>("Geant4Version",0);
}



const char* Opticks::getDirectGenstepPath() const 
{
    const char* det = m_spec->getDet();
    const char* typ = m_spec->getTyp();
    const char* tag = m_spec->getTag();

    const char* srctagdir = BOpticksEvent::srctagdir(det, typ, tag ); 

    LOG(info) << "Opticks::getDirectGenstepPath"
              << " det " << det 
              << " typ " << typ 
              << " tag " << tag
              << " srctagdir " << srctagdir
              ; 

    std::string path = BFile::FormPath( srctagdir, "gs.npy" ); 
    return strdup(path.c_str())  ; 
}


const char* Opticks::getLegacyGenstepPath() const 
{
    const char* det = m_spec->getDet();
    const char* typ = m_spec->getTyp();
    const char* tag = m_spec->getTag();

    std::string path = NLoad::GenstepsPath(det, typ, tag);

    LOG(info) << "Opticks::getLegacyGenstepPath"
              << " det " << det 
              << " typ " << typ 
              << " tag " << tag
              << " path " << path
              ; 

    return strdup(path.c_str()) ; 
}



bool Opticks::hasKey() const { return m_resource->hasKey() ; }


/**
Opticks::getGenstepPath
-------------------------

Legacy genstep paths carry the tag in their stems::

    /usr/local/opticks/opticksdata/gensteps/dayabay/scintillation/./1.npy 


const char* Opticks::getGenstepPath() const 
{
    return hasKey() ? getDirectGenstepPath() : getLegacyGenstepPath() ; 
}

**/


bool Opticks::existsDirectGenstepPath() const 
{
    const char* path = getDirectGenstepPath();
    bool exists = path ? BFile::ExistsFile(path) : false ;
    LOG(error) 
       << " path " << path 
       << " exists " << exists 
       ;

    return exists ; 
}

bool Opticks::existsLegacyGenstepPath() const 
{
    const char* path = getLegacyGenstepPath();
    bool exists = path ? BFile::ExistsFile(path) : false ;
    LOG(error) 
       << " path " << path 
       << " exists " << exists 
       ;

    return exists ; 
}






NPY<float>* Opticks::load(const char* path) const 
{
    NPY<float>* a = NPY<float>::load(path);
    if(!a)
    {
        LOG(warning) << "Opticks::load"
                     << " FAILED TO LOAD FROM "
                     << " path " << path 
                     ; 
        return NULL ;
    }
    return a ; 
}

NPY<float>* Opticks::loadDirectGenstep() const 
{
    std::string path = getDirectGenstepPath();
    return load(path.c_str()); 
}
NPY<float>* Opticks::loadLegacyGenstep() const 
{
    std::string path = getLegacyGenstepPath();
    return load(path.c_str()); 
}




/*
bool Opticks::existsDirectGenstepPath() const 
{
    const char* path = getDirectGenstepPath();
    bool exists = path ? BFile::ExistsFile(path) : false ;
    LOG(error) 
       << " path " << path 
       << " exists " << exists 
       ;

    return exists ; 
}

NPY<float>* Opticks::loadDirectGenstep() const 
{
    std::string path = getDirectGenstepPath();
    return load(path.c_str()); 
}

const char* Opticks::getPrimariesPath() const { return m_resource->getPrimariesPath() ; } 

bool Opticks::existsPrimariesPath() const 
{
    const char* path = getPrimariesPath();
    return path ? BFile::ExistsFile(path) : false ; 
}


NPY<float>* Opticks::loadPrimaries() const 
{
    const char* path = getPrimariesPath();
    return load(path); 
}

*/



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
    const std::string& config = m_cfg->getTorchConfig() ;

    const char* cfg = config.empty() ? NULL : config.c_str() ;

    LOG(info) << "Opticks::makeSimpleTorchStep" 
              << " config " << config 
              << " cfg " << ( cfg ? cfg : "NULL" )
              ;

    TorchStepNPY* torchstep = new TorchStepNPY(TORCH, 1, cfg );

    unsigned int photons_per_g4event = m_cfg->getNumPhotonsPerG4Event() ;  // only used for cfg4-

    torchstep->setNumPhotonsPerG4Event(photons_per_g4event);

    return torchstep ; 
}


unsigned Opticks::getNumPhotonsPerG4Event(){ return m_cfg->getNumPhotonsPerG4Event() ; }
unsigned Opticks::getRngMax(){       return m_cfg->getRngMax(); }
unsigned Opticks::getBounceMax() {   return m_cfg->getBounceMax(); }
unsigned Opticks::getRecordMax() {   return m_cfg->getRecordMax() ; }

float Opticks::getEpsilon() {            return m_cfg->getEpsilon()  ; }
bool Opticks::hasOpt(const char* name) const { return m_cfg->hasOpt(name); }

bool Opticks::operator()(const char* name) const 
{
    return m_cfg->hasOpt(name) ;
} 


const char* Opticks::getAnaKey() const 
{
    const std::string& s = m_cfg->getAnaKey();
    return s.empty() ? NULL : s.c_str() ; 
}
const char* Opticks::getAnaKeyArgs() const 
{
    std::string s = m_cfg->getAnaKeyArgs();
    if(s.empty()) return NULL ; 
    BStr::replace_all(s, "_", " ");
    return strdup(s.c_str()) ; 
}
const char* Opticks::getG4GunConfig() const 
{
    const std::string& s = m_cfg->getG4GunConfig();
    return s.empty() ? NULL : s.c_str() ; 
}


const char* Opticks::getExampleMaterialNames() { return m_resource->getExampleMaterialNames(); }
const char* Opticks::getDefaultMaterial() { return m_resource->getDefaultMaterial(); }
const char* Opticks::getDetector() { return m_resource->getDetector(); }
bool Opticks::isJuno() {    return m_resource->isJuno(); }
bool Opticks::isDayabay() { return m_resource->isDayabay(); }
bool Opticks::isPmtInBox(){ return m_resource->isPmtInBox(); }
bool Opticks::isOther() {   return m_resource->isOther(); }
bool Opticks::isValid() {   return m_resource->isValid(); }
bool Opticks::hasCtrlKey(const char* key) const  { return m_resource->hasCtrlKey(key); }
bool Opticks::hasVolnames() const { return !hasCtrlKey("novolnames") ; }

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


NSensorList*    Opticks::getSensorList(){ return m_resource ? m_resource->getSensorList() : NULL ; }
const char*     Opticks::getIdPath() {    return m_resource ? m_resource->getIdPath() : NULL ; }
const char*     Opticks::getIdFold() {    return m_resource ? m_resource->getIdFold() : NULL ; }
const char*     Opticks::getDetectorBase() {    return m_resource ? m_resource->getDetectorBase() : NULL ; }
const char*     Opticks::getMaterialMap() {  return m_resource ? m_resource->getMaterialMap() : NULL ; }
const char*     Opticks::getDAEPath() {   return m_resource ? m_resource->getDAEPath() : NULL ; }
const char*     Opticks::getInstallPrefix() { return m_resource ? m_resource->getInstallPrefix() : NULL ; }

bool             Opticks::SetKey(const char* spec) { return BOpticksKey::SetKey(spec) ; }
BOpticksKey*     Opticks::GetKey() {                 return BOpticksKey::GetKey() ; }
BOpticksKey*     Opticks::getKey() {                 return m_resource->getKey() ; }

const char*     Opticks::getSrcGDMLPath() const {  return m_resource ? m_resource->getSrcGDMLPath() : NULL ; }
const char*     Opticks::getGDMLPath()    const {  return m_resource ? m_resource->getGDMLPath() : NULL ; }
const char*     Opticks::getCurrentGDMLPath() const 
{
    bool is_embedded = isEmbedded() ;   
    return is_embedded ? getGDMLPath() : getSrcGDMLPath() ;
    // GDML path for embedded Opticks (ie direct from Geant4) is within the geocache directory 
}


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
 


template OKCORE_API void Opticks::profile<unsigned>(unsigned);
template OKCORE_API void Opticks::profile<int>(int);
template OKCORE_API void Opticks::profile<char*>(char*);
template OKCORE_API void Opticks::profile<const char*>(const char*);



