#include "NSensor.hpp"
#include <sstream>
#include <iomanip>

// kludge until fix the "csv" idmap to only put sensor labels on cathodes
const char* NSensor::CATHODE_NODE_NAME = "/dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode" ;

const unsigned int NSensor::UNSET_INDEX = 0  ;

bool NSensor::isCathode()
{
    return m_node_name && strcmp(m_node_name, CATHODE_NODE_NAME) == 0 ; 
}

std::string NSensor::description()
{
    std::stringstream ss ; 

    ss << "NSensor "
       << " index " << std::setw(6) << m_index 
       << " idhex " << std::setw(6) << std::hex << m_id  
       << " iddec " << std::setw(6) << std::dec << m_id  
       << " node_index " << std::setw(6) << m_node_index 
       << " name " << m_node_name  
       << " " << ( isCathode() ? "CATHODE" : "NOT-CATHODE" ) 
       ;

    return ss.str();
}
