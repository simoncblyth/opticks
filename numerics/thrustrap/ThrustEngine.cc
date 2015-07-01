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

void ThrustEngine::setHistoryTarget(
             unsigned long long* history_devptr, 
                   unsigned int* target_devptr, 
                    unsigned int size
        )
{
    m_history = new ThrustHistogram<unsigned long long, unsigned int>(history_devptr, target_devptr, size) ;

    m_history->dumpHistory("ThrustEngine::setHistoryTarget", 100);
}

void ThrustEngine::createIndices()
{
    if(!m_history) return ; 

    LOG(info) << "ThrustEngine::createIndices" ; 

    m_history->createHistogram();

    LOG(info) << "ThrustEngine::createIndices DONE " ; 

    m_history->dumpHistogram();

    LOG(info) << "ThrustEngine::createIndices dump DONE " ; 
}


