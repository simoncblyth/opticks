#pragma once

/**
SLog
======

Trivial logger enabling bracketing 
of constructor initializer lists.

**/


#include "SYSRAP_API_EXPORT.hh"
#include "plog/Severity.h"
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


