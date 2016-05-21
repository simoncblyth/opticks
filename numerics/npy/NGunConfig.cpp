#include "NGunConfig.hpp"
#include "NLog.hpp"

void NGunConfig::init()
{
    LOG(info) << "NGunConfig::init" << ( m_config ? m_config : "NULL" ) ; 
}


