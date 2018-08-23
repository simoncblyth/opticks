#pragma once

#include "SYSRAP_API_EXPORT.hh"
#include "plog/Severity.h"

// enable brief logging from ctor init lines 
// using a throwaway SLog instance 

class SYSRAP_API SLog 
{
    public:
        static const char* exename();
        static void Nonce(); 
    public:
        SLog(const char* label, const char* extra="", plog::Severity=plog::info );
        void operator()(const char* msg="");
    private:
        const char* m_label ; 
        const char* m_extra ; 
        plog::Severity m_level ; 
};


