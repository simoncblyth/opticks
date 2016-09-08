#include <cstring>

#include "SLog.hh"
#include "PLOG.hh"

SLog::SLog(const char* label) 
   :
   m_label(strdup(label))
{
   (*this)();
}

void SLog::operator()(const char* msg)
{
    LOG(info) << m_label << " " << msg ;  
}

