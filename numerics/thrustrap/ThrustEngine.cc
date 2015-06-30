#include "ThrustEngine.hh"

#include "stdio.h"
#include <iostream>

#include <thrust/version.h>

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


void ThrustEngine::version()
{
    LOG(info) << "ThrustEngine::version with Thrust v" << THRUST_MAJOR_VERSION << "." << THRUST_MINOR_VERSION ;
}

void ThrustEngine::setHistory(unsigned long long* devptr, unsigned int size)
{
    m_history = new ThrustHistogram<unsigned long long>(devptr, size) ;
}

void ThrustEngine::createIndices()
{
    if(!m_history) return ; 

    LOG(info) << "ThrustEngine::createIndices" ; 

    m_history->create();

    LOG(info) << "ThrustEngine::createIndices DONE " ; 

    m_history->dump();
}


