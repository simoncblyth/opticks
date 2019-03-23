#pragma once

/**
SVec
=====

static vector<T> utilities 

**/

#include <vector>

#include "SYSRAP_API_EXPORT.hh"
 
template <typename T>
struct SYSRAP_API SVec
{
    static void Dump(const char* label, const std::vector<T>& a );    
    static void Dump2(const char* label, const std::vector<T>& a );    
    static T MaxDiff(const std::vector<T>& a, const std::vector<T>& b, bool dump);    
    static int FindIndexOfValue( const std::vector<T>& a, T value, T tolerance ); 


};


