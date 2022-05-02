#pragma once
/**
SGeoConfig
==============

**/

#include <string>
#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API SGeoConfig
{
    static std::string Desc(); 
    static std::string DescEMM(); 
    static unsigned long long EMM ; 
    static bool IsEnabledMergedMesh(unsigned mm); 

};

 
