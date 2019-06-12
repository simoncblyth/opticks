#pragma once

#include "BRAP_API_EXPORT.hh"
#include "BRAP_HEAD.hh"

struct SArgs ; 
class BOpticksKey ; 
class BOpticksResource ; 


/**
BOpticks
==========

Used as a lower level standin for Opticks
primarily in NPY tests that need to load resources, 
for example npy/tests/NCSG2Test.cc 


**/

class BRAP_API  BOpticks {
    public:
        BOpticks(int argc=0, char** argv=nullptr, const char* argforced=nullptr ); 
    public:
        const char* getPath(const char* rela=nullptr, const char* relb=nullptr, const char* relc=nullptr ) const ; 
        int         getError() const ; 
 
        const char* getFirstArg(const char* fallback=nullptr ) const ; 
        const char* getArg( int n=1, const char* fallback=nullptr) const ; 

    private:
        const char*          m_firstarg ; 
        SArgs*               m_sargs ; 
        int                  m_argc ; 
        char**               m_argv ; 
        bool                 m_envkey ; 
        bool                 m_testgeo ; 
        BOpticksResource*    m_resource ; 
        int                  m_error ; 
       
 
};

#include "BRAP_TAIL.hh"

