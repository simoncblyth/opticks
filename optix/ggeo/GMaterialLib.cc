#include "GMaterialLib.hh"
#include "GMaterial.hh"

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


const char* GMaterialLib::refractive_index  = "refractive_index" ;
const char* GMaterialLib::absorption_length = "absorption_length" ;
const char* GMaterialLib::scattering_length = "scattering_length" ;
const char* GMaterialLib::reemission_prob   = "reemission_prob" ;


void GMaterialLib::Summary(const char* msg)
{
    LOG(info) << msg  
              << " NumMaterials " << getNumMaterials() 
              << " NumRawMaterials " << getNumRawMaterials() 
              ;
}

