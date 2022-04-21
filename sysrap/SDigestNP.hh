#pragma once
/**
SDigestNP.hh
==============

TODO: explore relocating this and SDigest functionality into NP.hh 

**/

struct NP ; 

#include <string>
#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API SDigestNP 
{
    static std::string Item( const NP* a, int i=-1, int j=-1, int k=-1, int l=-1, int m=-1, int o=-1 );  
}; 



