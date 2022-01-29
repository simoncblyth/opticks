#pragma once

#include <vector>
#include <string>
struct quad4 ; 

struct CSGRecord
{
    static std::vector<quad4> record ;     
    static void Dump(const char* msg="CSGRecord::Dump"); 
    static std::string Desc( const quad4& rec, unsigned irec, const char* label  ); 
    static void Save(const char* dir); 

};


