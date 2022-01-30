#pragma once

#include <vector>
#include <string>
struct quad4 ; 

/**
CSGRecord
===========

* CSGRecord_ENABLED envvar is the initial settin, this can be changed with SetEnabled. 
* operation also requires the special (non-default) compilation flag DEBUG_RECORD

**/

struct CSGRecord
{
    static bool ENABLED ;  
    static void SetEnabled(bool enabled); 

    static std::vector<quad4> record ;     
    static void Dump(const char* msg="CSGRecord::Dump"); 
    static std::string Desc( const quad4& rec, unsigned irec, const char* label  ); 
    static std::string Desc_q2( const quad4& rec ); 
    static std::string Desc_q3( const quad4& rec ); 
    static void Save(const char* dir); 
    static void Clear(); 

};


