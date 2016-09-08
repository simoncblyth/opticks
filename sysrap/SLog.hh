#pragma once

#include "SYSRAP_API_EXPORT.hh"

// enable brief logging from ctor init lines 
// using a throwaway SLog instance 

class SYSRAP_API SLog 
{
    public:
        SLog(const char* label);
        void operator()(const char* msg="");
    private:
        const char* m_label ; 
};


