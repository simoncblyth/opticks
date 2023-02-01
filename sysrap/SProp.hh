#pragma once
/**
SProp.hh
==========

::

    epsilon:sysrap blyth$ opticks-f SProp.hh
    ./sysrap/SProp.cc:#include "SProp.hh"
    ./sysrap/CMakeLists.txt:    SProp.hh
    ./ggeo/GGeo.cc:#include "SProp.hh"
    ./qudarap/tests/QPropTest.cc:#include "SProp.hh"


**/

#include <vector>
#include "SYSRAP_API_EXPORT.hh"
#include "plog/Severity.h"
struct NP ; 

struct SYSRAP_API SProp
{
    static const plog::Severity LEVEL ; 
    static const char*  DEMO_PATH ;

    static const NP* MockupCombination(const char* path_ ); 
    static const NP* MockupCombination(const NP* a_ ); 

    // TODO: below functionality belongs in NP.hh not here 
    static const NP* NarrowCombine(const std::vector<const NP*>& aa ); 
    static const NP* Combine(const std::vector<const NP*>& aa ); 

}; 




