#pragma once

#include <string>
#include <vector>
#include <cstring>
#include <sstream>

struct sstr
{
    static void Write(const char* path, const char* txt ); 
    static bool Match( const char* s, const char* q, bool starting=true ); 
    static bool StartsWith( const char* s, const char* q); 
    static const char* TrimTrailing(const char* s);
    static void PrefixSuffixParse(std::vector<std::string>& elem, const char* prefix, const char* suffix, const char* lines); 
    static void Split( const char* str, char delim,   std::vector<std::string>& elem ); 

    template<typename ... Args>
    static std::string Format_( const char* fmt, Args ... args ); 
};

inline void sstr::Write(const char* path, const char* txt )
{
    std::ofstream fp(path);
    fp << txt ;  
}

inline bool sstr::Match( const char* s, const char* q, bool starting )
{
    return starting ? StartsWith(s, q) : strcmp(s, q) == 0 ;
}

inline bool sstr::StartsWith( const char* s, const char* q)
{
    return s && q && strlen(q) <= strlen(s) && strncmp(s, q, strlen(q)) == 0 ;
}

inline const char* sstr::TrimTrailing(const char* s)
{
    char* p = strdup(s); 
    char* e = p + strlen(p) - 1 ;  
    while(e > p && ( *e == ' ' || *e == '\n' )) e-- ;
    e[1] = '\0' ;
    return p ;  
}

inline void sstr::PrefixSuffixParse(std::vector<std::string>& elem, const char* prefix, const char* suffix, const char* lines)
{
    std::stringstream ss;  
    ss.str(lines)  ;
    std::string s;
    while (std::getline(ss, s, '\n')) 
    {
        if(s.empty()) continue ;  
        const char* l = s.c_str(); 
        bool has_prefix = strlen(l) > strlen(prefix) && strncmp(l, prefix, strlen(prefix)) == 0 ; 
        bool has_suffix = strlen(l) > strlen(suffix) && strncmp(l+strlen(l)-strlen(suffix), suffix, strlen(suffix)) == 0 ; 
        //std::cout << "[" << l << "]"<< " has_prefix " << has_prefix << " has_suffix " << has_suffix << std::endl ; 
        if(has_prefix && has_suffix)
        {
              int count = strlen(l) - strlen(prefix) - strlen(suffix) ; 
              std::string sub = s.substr(strlen(prefix), count ); 
              //std::cout << " count " << count << " sub [" << sub << "]" << std::endl ; 
              elem.push_back(sub); 
        }
    }
}


inline void sstr::Split( const char* str, char delim,   std::vector<std::string>& elem )
{
    std::stringstream ss; 
    ss.str(str)  ;
    std::string s;
    while (std::getline(ss, s, delim)) elem.push_back(s) ; 
}


template<typename ... Args>
inline std::string sstr::Format_( const char* fmt, Args ... args )
{
    // see sysrap/tests/StringFormatTest.cc
    int sz = std::snprintf( nullptr, 0, fmt, args ... ) + 1 ; // +1 for null termination
    assert( sz > 0 );   
    std::vector<char> buf(sz) ;    
    std::snprintf( buf.data(), sz, fmt, args ... );
    return std::string( buf.begin(), buf.begin() + sz - 1 );  // exclude null termination 
}

template std::string sstr::Format_( const char*, const char*, int, int ); 





