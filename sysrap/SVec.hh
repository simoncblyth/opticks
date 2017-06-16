#pragma once

#include <vector>

#include "SYSRAP_API_EXPORT.hh"
 
template <typename T>
struct SYSRAP_API SVec
{
    static T MaxDiff(const std::vector<T>& a, const std::vector<T>& b, bool dump);    
};


