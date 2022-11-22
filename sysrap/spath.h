#pragma once

#include <cstring>
#include <string>
#include <sstream>
#include <vector>

struct spath
{
    template<typename ... Args>
    static std::string Join_( Args ... args_  ); 

    template<typename ... Args>
    static const char* Join( Args ... args ); 
};

template<typename ... Args>
inline std::string spath::Join_( Args ... args_  )  // static
{
    std::vector<std::string> args = {args_...};
    std::vector<std::string> elem ; 

    for(unsigned i=0 ; i < args.size() ; i++)
    {
        const std::string& arg = args[i] ; 
        if(!arg.empty()) elem.push_back(arg);  
    }

    unsigned num_elem = elem.size() ; 
    std::stringstream ss ; 
    for(unsigned i=0 ; i < num_elem ; i++)
    {
        const std::string& ele = elem[i] ; 
        ss << ele << ( i < num_elem - 1 ? "/" : "" ) ; 
    }
    std::string s = ss.str(); 
    return s ; 
}   

template std::string spath::Join_( const char*, const char* ); 
template std::string spath::Join_( const char*, const char*, const char* ); 
template std::string spath::Join_( const char*, const char*, const char*, const char* ); 

template<typename ... Args>
const char* spath::Join( Args ... args )  // static
{
    std::string s = Join_(args...)  ; 
    return strdup(s.c_str()) ; 
}   

template const char* spath::Join( const char*, const char* ); 
template const char* spath::Join( const char*, const char*, const char* ); 
template const char* spath::Join( const char*, const char*, const char*, const char* ); 


