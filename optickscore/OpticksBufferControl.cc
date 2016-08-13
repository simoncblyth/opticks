#include "OpticksBufferControl.hh"
#include <sstream>
#include "BStr.hh"

const char* OpticksBufferControl::OPTIX_SETSIZE_ = "OPTIX_SETSIZE" ; 
const char* OpticksBufferControl::OPTIX_NON_INTEROP_ = "OPTIX_NON_INTEROP" ; 
const char* OpticksBufferControl::OPTIX_INPUT_OUTPUT_ = "OPTIX_INPUT_OUTPUT" ; 
const char* OpticksBufferControl::OPTIX_INPUT_ONLY_ = "OPTIX_INPUT_ONLY" ; 
const char* OpticksBufferControl::OPTIX_OUTPUT_ONLY_ = "OPTIX_OUTPUT_ONLY" ; 


std::string OpticksBufferControl::Description(unsigned long long ctrl)
{
   std::stringstream ss ;
   if( ctrl & OPTIX_SETSIZE )       ss << OPTIX_SETSIZE_ << " "; 
   if( ctrl & OPTIX_NON_INTEROP  )  ss << OPTIX_NON_INTEROP_ << " "; 
   if( ctrl & OPTIX_INPUT_OUTPUT )  ss << OPTIX_INPUT_OUTPUT_ << " "; 
   if( ctrl & OPTIX_INPUT_ONLY   )  ss << OPTIX_INPUT_ONLY_ << " "; 
   if( ctrl & OPTIX_OUTPUT_ONLY   ) ss << OPTIX_OUTPUT_ONLY_ << " "; 
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
    return tag ;
}


unsigned long long OpticksBufferControl::Parse(const char* ctrl_, char delim)
{
    std::vector<std::string> elems ; 
    BStr::split(elems,ctrl_,delim);

    unsigned long long ctrl = 0 ; 
    for(unsigned i=0 ; i < elems.size() ; i++)
    {
        ctrl |= ParseTag(elems[i].c_str()) ;
    }    
    return ctrl ; 
}




