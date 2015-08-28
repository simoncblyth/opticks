#include "GBuffer.hh"

#include <iomanip>
#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


void GBuffer::Summary(const char* msg)
{
    LOG(info) << std::left << std::setw(30) << msg << std::right
              << " BufferId " << std::setw(4) << getBufferId()
              << " BufferTarget " << std::setw(4) << getBufferTarget()
              << " NumBytes " << std::setw(7) << getNumBytes()
              << " ItemSize " << std::setw(7) << getItemSize()
              << " NumElements " << std::setw(7) << getNumElements()
              << " NumItems " << std::setw(7) << getNumItems()
              << " NumElementsTotal " << std::setw(7) << getNumElementsTotal()
              ;

}




/*
const char* GBuffer::getBufferTargetName()
{
    return getBufferTargetName(m_buffer_target);
}

const char* GBuffer::getBufferTargetName(int buffer_target)
{
    switch(buffer_target)
    {
        case 
    }
}
*/

