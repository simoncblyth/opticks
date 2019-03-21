#include <cstring>

#include "SLog.hh"
#include "PLOG.hh"

SLog::SLog(const char* label, const char* extra, plog::Severity level) 
   :
   m_label(strdup(label)),
   m_extra(strdup(extra)),
   m_level(level)
{
    pLOG(m_level,0) 
        << " ( "
        << m_label 
        << " " 
        << m_extra 
        ;  
}


const char* SLog::exename() // static
{
    return PLOG::instance->args.exename() ; 
}

void SLog::operator()(const char* msg)
{
    pLOG(m_level,0) 
        << " ) "
        << m_label 
        << " " 
        << m_extra 
        << " "
        << msg 
        ;  
}

void SLog::Nonce()
{
    LOG(verbose) << "verbose" ; 
    LOG(debug) << "debug" ; 
    LOG(info) << "info" ; 
    LOG(warning) << "warning" ; 
    LOG(error) << "error" ; 
    LOG(fatal) << "fatal" ; 
}



