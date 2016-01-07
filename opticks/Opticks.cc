#include "Opticks.hh"
#include "OpticksResource.hh"
#include "OpticksCfg.hh"
#include "OpticksPhoton.h"

// npy-
#include "Map.hpp"
#include "stringutil.hpp"
#include "Parameters.hpp"
#include "NumpyEvt.hpp"
#include "TorchStepNPY.hpp"
#include "GLMFormat.hpp"
#include "NLog.hpp"


const char* Opticks::COMPUTE = "--compute" ; 


const char* Opticks::ZERO_              = "." ;
const char* Opticks::CERENKOV_          = "CERENKOV" ;
const char* Opticks::SCINTILLATION_     = "SCINTILLATION" ;
const char* Opticks::MISS_              = "MISS" ;
const char* Opticks::OTHER_             = "OTHER" ;
const char* Opticks::BULK_ABSORB_       = "BULK_ABSORB" ;
const char* Opticks::BULK_REEMIT_       = "BULK_REEMIT" ;
const char* Opticks::BULK_SCATTER_      = "BULK_SCATTER" ; 
const char* Opticks::SURFACE_DETECT_    = "SURFACE_DETECT" ;
const char* Opticks::SURFACE_ABSORB_    = "SURFACE_ABSORB" ; 
const char* Opticks::SURFACE_DREFLECT_  = "SURFACE_DREFLECT" ; 
const char* Opticks::SURFACE_SREFLECT_  = "SURFACE_SREFLECT" ; 
const char* Opticks::BOUNDARY_REFLECT_  = "BOUNDARY_REFLECT" ; 
const char* Opticks::BOUNDARY_TRANSMIT_ = "BOUNDARY_TRANSMIT" ; 
const char* Opticks::TORCH_             = "TORCH" ; 
const char* Opticks::NAN_ABORT_         = "NAN_ABORT" ; 
const char* Opticks::BAD_FLAG_          = "BAD_FLAG" ; 

const char* Opticks::cerenkov_          = "cerenkov" ;
const char* Opticks::scintillation_     = "scintillation" ;
const char* Opticks::torch_             = "torch" ; 
const char* Opticks::other_             = "other" ;

const char* Opticks::BNDIDX_NAME_  = "Boundary_Index" ;
const char* Opticks::SEQHIS_NAME_  = "History_Sequence" ;
const char* Opticks::SEQMAT_NAME_  = "Material_Sequence" ;


const char* Opticks::COMPUTE_MODE_  = "Compute" ;
const char* Opticks::INTEROP_MODE_  = "Interop" ;
const char* Opticks::CFG4_MODE_  = "CfG4" ;



// formerly of GPropertyLib, now booted upstairs
float        Opticks::DOMAIN_LOW  = 60.f ;
float        Opticks::DOMAIN_HIGH = 820.f ;  // has been 810.f for a long time  
float        Opticks::DOMAIN_STEP = 20.f ; 
unsigned int Opticks::DOMAIN_LENGTH = 39  ;





unsigned int Opticks::getRngMax()
{
    return m_cfg->getRngMax(); 
}
unsigned int Opticks::getBounceMax()
{
    return m_cfg->getBounceMax();
}
unsigned int Opticks::getRecordMax()
{
    return m_cfg->getRecordMax() ;
}
float Opticks::getEpsilon()
{
    return m_cfg->getEpsilon()  ;
}


std::string Opticks::getRelativePath(const char* path)
{
    return m_resource->getRelativePath(path);
}


glm::vec4 Opticks::getDefaultDomainSpec()
{
    glm::vec4 bd ;

    bd.x = DOMAIN_LOW ;
    bd.y = DOMAIN_HIGH ;
    bd.z = DOMAIN_STEP ;
    bd.w = DOMAIN_HIGH - DOMAIN_LOW ;

    return bd ; 
}




const char* Opticks::SourceType( int code )
{
    const char* name = 0 ; 
    switch(code)
    {
       case CERENKOV     :name = CERENKOV_      ;break;
       case SCINTILLATION:name = SCINTILLATION_ ;break;
       case TORCH        :name = TORCH_         ;break;
       default           :name = OTHER_         ;break; 
    }
    return name ; 
}

const char* Opticks::SourceTypeLowercase( int code )
{
    const char* name = 0 ; 
    switch(code)
    {
       case CERENKOV     :name = cerenkov_      ;break;
       case SCINTILLATION:name = scintillation_ ;break;
       case TORCH        :name = torch_         ;break;
       default           :name = other_         ;break; 
    }
    return name ; 
}

unsigned int Opticks::SourceCode(const char* type)
{
    unsigned int code = 0 ; 
    if(     strcmp(type,torch_)==0)         code = TORCH ;
    else if(strcmp(type,cerenkov_)==0)      code = CERENKOV ;
    else if(strcmp(type,scintillation_)==0) code = SCINTILLATION ;
    return code ; 
}


void Opticks::init()
{
   m_cfg = new OpticksCfg<Opticks>("opticks", this,false);
   m_parameters = new Parameters ;  
   m_resource = new OpticksResource(m_envprefix);
   m_log = new NLog(m_logname, m_loglevel);
}


const char* Opticks::getIdPath()
{
   return m_resource ? m_resource->getIdPath() : NULL ; 
}


void Opticks::preconfigure(int argc, char** argv)
{
    // need to know whether compute mode is active prior to standard configuration is done, 
    // in order to skip the Viz methods, so do in the pre-configure here 

    bool compute = false ;
    for(unsigned int i=1 ; i < argc ; i++ )
    {
        //printf("Opticks::preconfigure  %2d : %s \n", i, argv[i] );
        if(strcmp(argv[i], COMPUTE) == 0) 
        {
            //printf("Opticks::configure setting compute \n");
            compute = true ; 
        }
    }

    setMode( compute ? COMPUTE_MODE : INTEROP_MODE );
    setDetector( m_resource->getDetector() );

    LOG(info) << "Opticks::preconfigure" 
              << " mode " << getModeString() 
              << " detector " << m_resource->getDetector()
              ;
}


void Opticks::configure(int argc, char** argv)
{
    preconfigure(argc, argv);

    m_log->configure(argc, argv);
    m_log->init(getIdPath());

    m_lastarg = argc > 1 ? strdup(argv[argc-1]) : NULL ;
}

void Opticks::Summary(const char* msg)
{
    LOG(info) << msg ; 
    m_resource->Summary(msg);
}


int Opticks::getLastArgInt()
{
    int index(-1);
    if(!m_lastarg) return index ;
 
    try{ 
        index = boost::lexical_cast<int>(m_lastarg) ;
    }
    catch (const boost::bad_lexical_cast& e ) {
        LOG(warning)  << "Caught bad lexical cast with error " << e.what() ;
    }
    catch( ... ){
        LOG(warning) << "Unknown exception caught!" ;
    }
    return index;
}










void Opticks::configureDomains()
{
   m_time_domain.x = 0.f  ;
   m_time_domain.y = m_cfg->getTimeMax() ;
   m_time_domain.z = m_cfg->getAnimTimeMax() ;
   m_time_domain.w = 0.f  ;


   // space domain is updated once geometry is loaded
   m_space_domain.x = 0.f ; 
   m_space_domain.y = 0.f ; 
   m_space_domain.z = 0.f ; 
   m_space_domain.w = 1000.f ; 

   m_wavelength_domain = getDefaultDomainSpec() ;  

   int rng_max = getenvint("CUDAWRAP_RNG_MAX",-1); 

   int x_rng_max = getRngMax() ;

   if(rng_max != x_rng_max)
       LOG(fatal) << "Opticks::configureDomains"
                  << " CUDAWRAP_RNG_MAX " << rng_max 
                  << " x_rng_max " << x_rng_max 
                  ;

   assert(rng_max == x_rng_max && "Configured RngMax must match envvar CUDAWRAP_RNG_MAX and corresponding files, see cudawrap- ");    
}

void Opticks::dumpDomains(const char* msg)
{
    LOG(info) << msg << std::endl 
              << " time " << gformat(m_time_domain)  
              << " space " << gformat(m_space_domain) 
              << " wavelength " << gformat(m_wavelength_domain) 
              ;
}




std::string Opticks::getModeString()
{
    std::stringstream ss ; 

    if(isCompute()) ss << COMPUTE_MODE_ ; 
    if(isInterop()) ss << INTEROP_MODE_ ; 
    if(isCfG4())    ss << CFG4_MODE_ ; 

    return ss.str();
}


NumpyEvt* Opticks::makeEvt()
{

    unsigned int code = getSourceCode();
    std::string typ = SourceTypeLowercase(code); // cerenkov, scintillation, torch
    std::string tag = m_cfg->getEventTag();

    std::string det = m_detector ? m_detector : "" ;
    std::string cat = m_cfg->getEventCat();   // overrides det for categorization of test events eg "rainbow" "reflect" "prism" "newton"


    NumpyEvt* evt = new NumpyEvt(typ.c_str(), tag.c_str(), det.c_str(), cat.c_str() );

    configureDomains();

    evt->setTimeDomain(getTimeDomain());
    evt->setSpaceDomain(getSpaceDomain());   // default, will be updated in App:registerGeometry following geometry loading
    evt->setWavelengthDomain(getWavelengthDomain());

    bool nostep = m_cfg->hasOpt("nostep") ;
    evt->setStep(!nostep);
    evt->setMaxRec(m_cfg->getRecordMax());
 
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

    parameters->add<std::string>("mode", getModeString() ); 

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



unsigned int Opticks::getSourceCode()
{
    unsigned int code ;
    if(     m_cfg->hasOpt("cerenkov"))      code = CERENKOV ;
    else if(m_cfg->hasOpt("scintillation")) code = SCINTILLATION ;
    else if(m_cfg->hasOpt("torch"))         code = TORCH ;
    else                                    code = TORCH ;
    return code ;
}


std::string Opticks::getSourceType()
{
    unsigned int code = getSourceCode();
    std::string typ = SourceType(code) ; 
    boost::algorithm::to_lower(typ);
    return typ ; 
}

const char* Opticks::Flag(const unsigned int flag)
{
    const char* s = 0 ; 
    switch(flag)
    {
        case 0:                s=ZERO_;break;
        case CERENKOV:         s=CERENKOV_;break;
        case SCINTILLATION:    s=SCINTILLATION_ ;break; 
        case MISS:             s=MISS_ ;break; 
        case BULK_ABSORB:      s=BULK_ABSORB_ ;break; 
        case BULK_REEMIT:      s=BULK_REEMIT_ ;break; 
        case BULK_SCATTER:     s=BULK_SCATTER_ ;break; 
        case SURFACE_DETECT:   s=SURFACE_DETECT_ ;break; 
        case SURFACE_ABSORB:   s=SURFACE_ABSORB_ ;break; 
        case SURFACE_DREFLECT: s=SURFACE_DREFLECT_ ;break; 
        case SURFACE_SREFLECT: s=SURFACE_SREFLECT_ ;break; 
        case BOUNDARY_REFLECT: s=BOUNDARY_REFLECT_ ;break; 
        case BOUNDARY_TRANSMIT:s=BOUNDARY_TRANSMIT_ ;break; 
        case TORCH:            s=TORCH_ ;break; 
        case NAN_ABORT:        s=NAN_ABORT_ ;break; 
        default:               s=BAD_FLAG_  ;
                               LOG(warning) << "Opticks::Flag BAD_FLAG [" << flag << "]" << std::hex << flag << std::dec ;             
    }
    return s;
}


std::string Opticks::FlagSequence(const unsigned long long seqhis)
{
    std::stringstream ss ;
    assert(sizeof(unsigned long long)*8 == 16*4);
    for(unsigned int i=0 ; i < 16 ; i++)
    {
        unsigned long long f = (seqhis >> i*4) & 0xF ; 
        unsigned int flg = f == 0 ? 0 : 0x1 << (f - 1) ; 
        ss << Flag(flg) << " " ;
    }
    return ss.str();
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

         printf("Opticks::parameter_set %s : %lu values : ", name, values.size());
         for(size_t i=0 ; i < values.size() ; i++ ) printf("%10.3f ", values[i]);
         printf(" : vlast %10.3f \n", vlast );

         //configure(name, vlast);  
     }   
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





