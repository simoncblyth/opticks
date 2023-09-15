#pragma once

#include <cstdlib>
#include <cstring>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>

struct spath
{
    static constexpr const bool VERBOSE = false ; 

    static std::string _ResolvePath0(const char* spec); 
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

    static bool LooksLikePath(const char* arg); 
    static const char* Basename(const char* path); 


};

/**
spath::_ResolvePath0 : old impl that only works for tokens at start of spec
-----------------------------------------------------------------------------
::

    $TOKEN/remainder/path/name.npy   (tok_plus) 
    $TOKEN

If the TOKEN envvar is not set then nullptr is returned.  

**/

inline std::string spath::_ResolvePath0(const char* spec_)
{
    if(spec_ == nullptr) return "" ; 
    char* spec = strdup(spec_); 

    std::stringstream ss ; 
    if( spec[0] == '$' )
    {
        char* sep = strchr(spec, '/');       // point to first slash  
        char* end = strchr(spec, '\0' ); 
        bool tok_plus =  sep && end && sep != end ;  
        if(tok_plus) *sep = '\0' ;           // replace slash with null termination 
        char* pfx = getenv(spec+1) ; 
        if(pfx == nullptr) return "" ;  
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

/**
spath::_ResolvePath
----------------------

This works with multiple tokens, eg::

    $HOME/.opticks/GEOM/$GEOM/CSGFoundry/meshname.txt

**/

inline std::string spath::_ResolvePath(const char* spec_)
{
    if(spec_ == nullptr) return "" ; 
    char* spec = strdup(spec_); 

    std::stringstream ss ; 
    int speclen = int(strlen(spec)) ;  
    char* end = strchr(spec, '\0' ); 
    int i = 0 ; 

    if(VERBOSE) std::cout << " spec " << spec << " speclen " << speclen << std::endl ; 

    while( i < speclen )
    {
        if(VERBOSE) std::cout << " i " << i << " spec[i] " << spec[i] << std::endl ;   
        if( spec[i] == '$' )
        {
            char* p = spec + i ; 
            char* sep = strchr( p, '/' ) ; // first slash after token   
            bool tok_plus =  sep && end && sep != end ;  
            if(tok_plus) *sep = '\0' ;           // replace slash with null termination 
            char* val = getenv(p+1) ;  // skip '$'
            int toklen = int(strlen(p)) ;  // strlen("TOKEN")  no need for +1 as already at '$'  
            if(VERBOSE) std::cout << " toklen " << toklen << std::endl ;  
            if(val == nullptr) 
            {
                std::cerr 
                    << "spath::_ResolvePath token [" 
                    << p+1 
                    << "] does not resolve " 
                    << std::endl 
                    ; 
                return "" ;    // all tokens must resolve 
            }
            if(tok_plus) *sep = '/' ;            // put back the slash 
            ss << val  ; 

            i += toklen ;   // skip over the token 
        }
        else
        {
           ss << spec[i] ; 
           i += 1 ; 
        }
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


inline bool spath::LooksLikePath(const char* arg)
{
    if(!arg) return false ;
    if(strlen(arg) < 2) return false ; 
    return arg[0] == '/' || arg[0] == '$' ; 
}

inline const char* spath::Basename(const char* path)
{
    std::string p = path ; 
    std::size_t pos = p.find_last_of("/");
    std::string base = pos == std::string::npos ? p : p.substr(pos+1) ; 
    return strdup( base.c_str() ) ; 
}





