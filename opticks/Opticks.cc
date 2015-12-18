#include "Opticks.hh"
#include "OpticksCfg.hh"
#include "OpticksPhoton.h"

#include "TorchStepNPY.hpp"
#include "NLog.hpp"

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

    return torchstep ; 
}




