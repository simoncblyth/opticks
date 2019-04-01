#pragma once

/**
SBase36
==========

Base36 encoder converting an unsigned into a base36 string.

Used in :doc:`/npy/NOpenMeshProp`


**/


#include <string>
#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API SBase36
{
    static const char* LABELS ;  
    std::string operator()(unsigned int val) const ; 
};





