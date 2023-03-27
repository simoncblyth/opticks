#pragma once

#include <cstdlib>
#include <cstring>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>

struct spath
{
    static std::string _ResolvePath(const char* spec); 
    static const char* ResolvePath(const char* spec); 

    template<typename ... Args>
    static std::string _Resolve(Args ... args ); 

    template<typename ... Args>
    static const char* Resolve(Args ... args ); 


    template<typename ... Args>
    static std::string _Join( Args ... args_  ); 

    template<typename ... Args>
    static const char* Join( Args ... args ); 

    template<typename ... Args>
    static bool Exists( Args ... args ); 

};

/**
spath::_ResolvePath
---------------------
::

    $TOKEN/remainder/path/name.npy   (tok_plus) 
    $TOKEN

If the TOKEN envvar is not set then nullptr is returned.  

**/

inline std::string spath::_ResolvePath(const char* spec_)
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
        if(pfx == nullptr) return nullptr ;  
        if(tok_plus) *sep = '/' ;            // put back the slash 
        ss << pfx << ( sep ? sep : "" ) ; 
    }
    else
    {
        ss << spec ; 
    }
    std::string str = ss.str(); 
    return str ; 
}
inline const char* spath::ResolvePath(const char* spec_)
{
    std::string path = _ResolvePath(spec_) ;  
    return strdup(path.c_str()) ; 
}


template<typename ... Args>
inline std::string spath::_Resolve( Args ... args  )  // static
{
    std::string spec = _Join(std::forward<Args>(args)... ); 
    return _ResolvePath(spec.c_str()); 
}

template std::string spath::_Resolve( const char* ); 
template std::string spath::_Resolve( const char*, const char* ); 
template std::string spath::_Resolve( const char*, const char*, const char* ); 
template std::string spath::_Resolve( const char*, const char*, const char*, const char* ); 


template<typename ... Args>
inline const char* spath::Resolve( Args ... args  )  // static
{
    std::string spec = _Join(std::forward<Args>(args)... ); 
    std::string path = _ResolvePath(spec.c_str()); 
    return strdup(path.c_str()) ; 
}

template const char* spath::Resolve( const char* ); 
template const char* spath::Resolve( const char*, const char* ); 
template const char* spath::Resolve( const char*, const char*, const char* ); 
template const char* spath::Resolve( const char*, const char*, const char*, const char* ); 




template<typename ... Args>
inline std::string spath::_Join( Args ... args_  )  // static
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

template std::string spath::_Join( const char* ); 
template std::string spath::_Join( const char*, const char* ); 
template std::string spath::_Join( const char*, const char*, const char* ); 
template std::string spath::_Join( const char*, const char*, const char*, const char* ); 

template<typename ... Args>
inline const char* spath::Join( Args ... args )  // static
{
    std::string s = _Join(std::forward<Args>(args)...)  ; 
    return strdup(s.c_str()) ; 
}   

template const char* spath::Join( const char* ); 
template const char* spath::Join( const char*, const char* ); 
template const char* spath::Join( const char*, const char*, const char* ); 
template const char* spath::Join( const char*, const char*, const char*, const char* ); 



template<typename ... Args>
inline bool spath::Exists(Args ... args)
{
    std::string path = _Resolve(std::forward<Args>(args)...) ; 
    std::ifstream fp(path.c_str(), std::ios::in|std::ios::binary);
    return fp.fail() ? false : true ; 
}

template bool spath::Exists( const char* ); 
template bool spath::Exists( const char*, const char* ); 
template bool spath::Exists( const char*, const char*, const char* ); 
template bool spath::Exists( const char*, const char*, const char*, const char* ); 


