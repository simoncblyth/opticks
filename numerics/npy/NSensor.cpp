#include "NSensor.hpp"
#include <sstream>
#include <iomanip>

const unsigned int NSensor::UNSET_INDEX = 0  ;

std::string NSensor::description()
{
    std::stringstream ss ; 

    ss << "NSensor "
       << " index " << std::setw(6) << m_index 
       << " idhex " << std::setw(6) << std::hex << m_id  
       << " iddec " << std::setw(6) << std::dec << m_id  
       << " node_index " << std::setw(6) << m_node_index 
       << " name " << m_node_name  
       ;

    return ss.str();
}
