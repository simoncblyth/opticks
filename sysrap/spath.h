#pragma once

#include <cstdlib>
#include <cstring>
#include <string>
#include <sstream>
#include <vector>

struct spath
{
    static const char* Resolve(const char* spec); 

    template<typename ... Args>
    static std::string Join_( Args ... args_  ); 

    template<typename ... Args>
    static const char* Join( Args ... args ); 
};

/**
spath::Resolve
----------------

::

    $TOK/remainder/path/name.npy   (tok_plus) 
    $TOK

**/

inline const char* spath::Resolve(const char* spec_)
{
    if(spec_ == nullptr) return nullptr ; 
    char* spec = strdup(spec_); 

    std::stringstream ss ; 
    if( spec[0] == '$' )
    {
        char* sep = strchr(spec, '/');       // point to first slash  
        char* end = strchr(spec, '\0' ); 
        bool tok_plus =  sep && end && sep != end ;  
        if(tok_plus) *sep = '\0' ;           // replace slash with null termination 
        char* pfx = getenv(spec+1) ; 
        if(tok_plus) *sep = '/' ;            // put back the slash 
        ss << ( pfx ? pfx : "/tmp" ) << ( sep ? sep : "" ) ; 
    }
    else
    {
        ss << spec ; 
    }
    std::string s = ss.str(); 
    const char* path = s.c_str(); 
    return strdup(path) ; 
}


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


