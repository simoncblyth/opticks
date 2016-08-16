#include "OpticksBufferControl.hh"
#include <sstream>
#include "BStr.hh"
#include "PLOG.hh"

const char* OpticksBufferControl::OPTIX_SETSIZE_ = "OPTIX_SETSIZE" ; 
const char* OpticksBufferControl::OPTIX_NON_INTEROP_ = "OPTIX_NON_INTEROP" ;   // use optix buffer even in in interop mode
const char* OpticksBufferControl::OPTIX_INPUT_OUTPUT_ = "OPTIX_INPUT_OUTPUT" ; 
const char* OpticksBufferControl::OPTIX_INPUT_ONLY_ = "OPTIX_INPUT_ONLY" ; 
const char* OpticksBufferControl::OPTIX_OUTPUT_ONLY_ = "OPTIX_OUTPUT_ONLY" ; 
const char* OpticksBufferControl::PTR_FROM_OPTIX_ = "PTR_FROM_OPTIX" ; 
const char* OpticksBufferControl::PTR_FROM_OPENGL_ = "PTR_FROM_OPENGL" ; 
const char* OpticksBufferControl::UPLOAD_WITH_CUDA_ = "UPLOAD_WITH_CUDA" ; 
const char* OpticksBufferControl::BUFFER_COPY_ON_DIRTY_ = "BUFFER_COPY_ON_DIRTY" ; 
const char* OpticksBufferControl::BUFFER_GPU_LOCAL_ = "BUFFER_GPU_LOCAL" ; 

std::vector<const char*> OpticksBufferControl::Tags()
{
    std::vector<const char*> tags ; 
    tags.push_back(OPTIX_SETSIZE_);
    tags.push_back(OPTIX_NON_INTEROP_);
    tags.push_back(OPTIX_INPUT_OUTPUT_);
    tags.push_back(OPTIX_INPUT_ONLY_);
    tags.push_back(OPTIX_OUTPUT_ONLY_);
    tags.push_back(PTR_FROM_OPTIX_);
    tags.push_back(PTR_FROM_OPENGL_);
    tags.push_back(UPLOAD_WITH_CUDA_);
    tags.push_back(BUFFER_COPY_ON_DIRTY_);
    tags.push_back(BUFFER_GPU_LOCAL_);
    return tags  ;
}

std::string OpticksBufferControl::Description(unsigned long long ctrl)
{
   std::stringstream ss ;
   if( ctrl & OPTIX_SETSIZE )       ss << OPTIX_SETSIZE_ << " "; 
   if( ctrl & OPTIX_NON_INTEROP  )  ss << OPTIX_NON_INTEROP_ << " "; 
   if( ctrl & OPTIX_INPUT_OUTPUT )  ss << OPTIX_INPUT_OUTPUT_ << " "; 
   if( ctrl & OPTIX_INPUT_ONLY   )  ss << OPTIX_INPUT_ONLY_ << " "; 
   if( ctrl & OPTIX_OUTPUT_ONLY   ) ss << OPTIX_OUTPUT_ONLY_ << " "; 
   if( ctrl & PTR_FROM_OPTIX      ) ss << PTR_FROM_OPTIX_ << " "; 
   if( ctrl & PTR_FROM_OPENGL     ) ss << PTR_FROM_OPENGL_ << " "; 
   if( ctrl & UPLOAD_WITH_CUDA    ) ss << UPLOAD_WITH_CUDA_ << " "; 
   if( ctrl & BUFFER_COPY_ON_DIRTY ) ss << BUFFER_COPY_ON_DIRTY_ << " "; 
   if( ctrl & BUFFER_GPU_LOCAL ) ss << BUFFER_GPU_LOCAL_ << " "; 
   return ss.str();
}

unsigned long long OpticksBufferControl::ParseTag(const char* k)
{
    unsigned long long tag = 0 ;
    if(     strcmp(k,OPTIX_SETSIZE_)==0)      tag = OPTIX_SETSIZE ;
    else if(strcmp(k,OPTIX_NON_INTEROP_)==0)  tag = OPTIX_NON_INTEROP ;
    else if(strcmp(k,OPTIX_INPUT_OUTPUT_)==0) tag = OPTIX_INPUT_OUTPUT ;
    else if(strcmp(k,OPTIX_INPUT_ONLY_)==0)   tag = OPTIX_INPUT_ONLY ;
    else if(strcmp(k,OPTIX_OUTPUT_ONLY_)==0)  tag = OPTIX_OUTPUT_ONLY ;
    else if(strcmp(k,PTR_FROM_OPTIX_)==0)     tag = PTR_FROM_OPTIX ;
    else if(strcmp(k,PTR_FROM_OPENGL_)==0)    tag = PTR_FROM_OPENGL ;
    else if(strcmp(k,UPLOAD_WITH_CUDA_)==0)   tag = UPLOAD_WITH_CUDA ;
    else if(strcmp(k,BUFFER_COPY_ON_DIRTY_)==0) tag = BUFFER_COPY_ON_DIRTY ;
    else if(strcmp(k,BUFFER_GPU_LOCAL_)==0)     tag = BUFFER_GPU_LOCAL ;
    return tag ;
}



unsigned long long OpticksBufferControl::Parse(const char* ctrl_, char delim)
{
    unsigned long long ctrl(0) ; 
    if(ctrl_)
    {
        std::vector<std::string> elems ; 
        BStr::split(elems,ctrl_,delim);
        for(unsigned i=0 ; i < elems.size() ; i++)
        {
            const char* tag_ = elems[i].c_str() ;
            unsigned long long tag = ParseTag(tag_) ;
            if(tag == 0)
                 LOG(fatal) << "OpticksBufferControl::Parse BAD TAG " << tag_ ;
            assert(tag);
            ctrl |= tag ;
        }
    }
    return ctrl ; 
}
bool OpticksBufferControl::isSet(unsigned long long ctrl, const char* mask_)  
{
    unsigned long long mask = Parse(mask_) ;   
    bool match = (ctrl & mask) != 0 ; 
    return match ; 
}

OpticksBufferControl::OpticksBufferControl(unsigned long long ctrl)
    :
    m_ctrl(ctrl)
{
}
OpticksBufferControl::OpticksBufferControl(const char* ctrl)
    :
    m_ctrl(Parse(ctrl))
{
}

bool OpticksBufferControl::isSet(const char* mask) const 
{
    return isSet(m_ctrl, mask );
}
std::string OpticksBufferControl::description(const char* msg) const
{
   std::stringstream ss ;
   ss << msg << " : " ;
   ss << Description(m_ctrl) ;
   return ss.str();
}

 

