
//  /opt/local/share/doc/boost/libs/program_options/example/multiple_sources.cpp 

#include "Config.hh"

Config::Config()
{
}

Config::~Config()
{
}

unsigned int Config::getUdpPort()
{
    return m_udpPort ;
}

void Config::setUdpPort(unsigned int udpPort)
{
    m_udpPort = udpPort ;
}


