#pragma once
/**
suniquename.h
==============

This allows storing large numbers of repetitive names 
without repeating them, by storing integer indices 
into the names vector instead of storing all the repeated 
names.  

**/

#include <string>
#include <vector>
#include <sstream>
#include <iomanip>

struct suniquename
{
    static int Add(const char* name, std::vector<std::string>& names ) ;    
    static std::string Desc( const std::vector<std::string>& names ) ; 
}; 

/**
suniquename::Add
------------------

Returns the index of the name in the vector, after first 
adding the name to the vector if not already present. 
A positive integer value is always returned. 

**/

inline int suniquename::Add( const char* name, std::vector<std::string>& names ) // static
{
    size_t size = names.size() ; 
    size_t idx = std::distance( names.begin(), std::find( names.begin(), names.end(), name ) ); 
    if(idx == size) names.push_back(name) ; 
    return idx  ; 
}

inline std::string suniquename::Desc( const std::vector<std::string>& names )
{
    std::stringstream ss ; 
    for(size_t i=0 ; i < names.size() ; i++ ) ss << std::setw(7) << i << " : " << names[i] << std::endl ; 
    std::string str = ss.str(); 
    return str ; 
}

