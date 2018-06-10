#include <cstring>

#include "SLog.hh"
#include "PLOG.hh"

SLog::SLog(const char* label, const char* extra) 
   :
   m_label(strdup(label)),
   m_extra(strdup(extra))
{
    LOG(debug) 
        << m_label 
        << " " 
        << m_extra 
        ;  
}

void SLog::operator()(const char* msg)
{
    LOG(info) 
        << m_label 
        << " " 
        << m_extra 
        << " "
        << msg 
        ;  
}

void SLog::Nonce()
{
    LOG(trace) << "trace" ; 
    LOG(debug) << "debug" ; 
    LOG(info) << "info" ; 
    LOG(warning) << "warning" ; 
    LOG(error) << "error" ; 
    LOG(fatal) << "fatal" ; 
}



