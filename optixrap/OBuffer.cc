#include <sstream>

#include "PLOG.hh"
#include "OBuffer.hh"
#include "OXPPNS.hh"

const char* OBuffer::RT_BUFFER_INPUT_ = "INPUT" ; 
const char* OBuffer::RT_BUFFER_OUTPUT_ = "OUTPUT" ; 
const char* OBuffer::RT_BUFFER_INPUT_OUTPUT_ = "INPUT_OUTPUT" ; 
const char* OBuffer::RT_BUFFER_PROGRESSIVE_STREAM_ = "PROGRESSIVE_STREAM" ; 


const char* OBuffer::RT_BUFFER_LAYERED_ = "LAYERED" ; 
const char* OBuffer::RT_BUFFER_CUBEMAP_ = "CUBEMAP" ; 
const char* OBuffer::RT_BUFFER_GPU_LOCAL_ = "GPU_LOCAL" ; 
const char* OBuffer::RT_BUFFER_COPY_ON_DIRTY_ = "COPY_ON_DIRTY" ;  


std::string OBuffer::BufferDesc(unsigned buffer_desc)  // static 
{
    std::stringstream ss ; 

    if( buffer_desc & RT_BUFFER_INPUT )              ss << RT_BUFFER_INPUT_ << " " ; 
    if( buffer_desc & RT_BUFFER_OUTPUT )             ss << RT_BUFFER_OUTPUT_ << " " ; 
    if( buffer_desc & RT_BUFFER_INPUT_OUTPUT )       ss << RT_BUFFER_INPUT_OUTPUT_ << " " ; 
    if( buffer_desc & RT_BUFFER_PROGRESSIVE_STREAM ) ss << RT_BUFFER_PROGRESSIVE_STREAM_ << " " ; 

    if( buffer_desc & RT_BUFFER_LAYERED )            ss << RT_BUFFER_LAYERED_ << " " ; 
    if( buffer_desc & RT_BUFFER_CUBEMAP )            ss << RT_BUFFER_CUBEMAP_ << " " ; 
    if( buffer_desc & RT_BUFFER_GPU_LOCAL )          ss << RT_BUFFER_GPU_LOCAL_ << " " ; 
    if( buffer_desc & RT_BUFFER_COPY_ON_DIRTY )      ss << RT_BUFFER_COPY_ON_DIRTY_ << " " ; 

    return ss.str();
}


void OBuffer::Dump(const char* msg) // static 
{
    LOG(info) << msg ; 
    std::cout 
        << std::setw(5) << RT_BUFFER_INPUT << std::setw(30) << RT_BUFFER_INPUT_  << std::endl 
        << std::setw(5) << RT_BUFFER_OUTPUT << std::setw(30) << RT_BUFFER_OUTPUT_  << std::endl 
        << std::setw(5) << RT_BUFFER_INPUT_OUTPUT << std::setw(30) << RT_BUFFER_INPUT_OUTPUT_  << std::endl 
        << std::setw(5) << RT_BUFFER_GPU_LOCAL << std::setw(30) << RT_BUFFER_GPU_LOCAL_  << std::endl 
        << std::setw(5) << RT_BUFFER_COPY_ON_DIRTY << std::setw(30) << RT_BUFFER_COPY_ON_DIRTY_  << std::endl 
        << std::setw(5) << RT_BUFFER_PROGRESSIVE_STREAM << std::setw(30) << RT_BUFFER_PROGRESSIVE_STREAM_  << std::endl 
        << std::setw(5) << RT_BUFFER_LAYERED << std::setw(30) << RT_BUFFER_LAYERED_  << std::endl 
        << std::setw(5) << RT_BUFFER_CUBEMAP << std::setw(30) << RT_BUFFER_CUBEMAP_  << std::endl 
        ;

}

