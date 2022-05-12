#pragma once
/**
SDir.h : header only directory listing paths with supplied ext 
=================================================================

**/

#include <sstream>
#include <iostream>
#include <vector>
#include <string>

#include "dirent.h"

struct SDir 
{
    static void List(std::vector<std::string>& names, const char* path, const char* ext ); 
    static std::string Desc(const std::vector<std::string>& names); 
}; 

inline void SDir::List(std::vector<std::string>& names, const char* path, const char* ext )
{
    DIR* dir = opendir(path) ;
    if(!dir) std::cout << "SDir::List FAILED TO OPEN DIR " << ( path ? path : "-" ) << std::endl ; 
    if(!dir) return ; 
    struct dirent* entry ;
    while ((entry = readdir(dir)) != nullptr) 
    {   
        const char* name = entry->d_name ; 
        if(strlen(name) > strlen(ext) && strcmp(name + strlen(name) - strlen(ext), ext)==0)
        {   
            names.push_back(name); 
        }   
    }   
    closedir (dir);
    std::sort( names.begin(), names.end() );  

    if(names.size() == 0 ) std::cout 
        << "SDir::List" 
        << " path " << ( path ? path : "-" ) 
        << " ext " << ( ext ? ext : "-" ) 
        << " NO ENTRIES FOUND "
        << std::endl
        ;
}

inline std::string SDir::Desc(const std::vector<std::string>& names)
{
    std::stringstream ss ; 
    for(unsigned i=0 ; i < names.size() ; i++) ss << names[i] << std::endl ; 
    std::string s = ss.str(); 
    return s ; 
}


