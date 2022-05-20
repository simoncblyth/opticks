#pragma once

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

    static const NP* NarrowCombine(const std::vector<const NP*>& aa ); 
    static const NP* Combine(const std::vector<const NP*>& aa ); 


}; 




