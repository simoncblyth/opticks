#pragma once

#include "SYSRAP_API_EXPORT.hh"

// enable brief logging from ctor init lines 
// using a throwaway SLog instance 

class SYSRAP_API SLog 
{
    public:
        static void Nonce(); 
    public:
        SLog(const char* label, const char* extra="" );
        void operator()(const char* msg="");
    private:
        const char* m_label ; 
        const char* m_extra ; 
};


