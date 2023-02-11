#pragma once

#include <string>
#include <vector>
#include <cstring>
#include <sstream>
#include <fstream>
#include <iostream>
#include <cassert>

struct sstr
{
    static void Write(const char* path, const char* txt ); 
    static bool Match( const char* s, const char* q, bool starting=true ); 
    static bool StartsWith( const char* s, const char* q); 
    static const char* TrimTrailing(const char* s);
    static std::string StripTail(const std::string& name, const char* end="0x"); 
    static std::string StripTail(const char* name, const char* end="0x"); 

    static void PrefixSuffixParse(std::vector<std::string>& elem, const char* prefix, const char* suffix, const char* lines); 
    static void Split( const char* str, char delim,   std::vector<std::string>& elem ); 
    static void Chop( std::pair<std::string, std::string>& head__tail, const char* delim, const char* str ); 
    static void chop( char** head, char** tail, const char* delim, const char* str ); 

    template<typename T>
    static void split(std::vector<T>& elem, const char* str, char delim  ); 

    template<typename ... Args>
    static std::string Format_( const char* fmt, Args ... args ); 

    static bool Blank(const char* s ); 
    static bool All(const char* s, char q ); 
    static unsigned Count(const char* s, char q ); 
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

inline const char* sstr::TrimTrailing(const char* s) // reposition null terminator to skip trailing whitespace 
{
    char* p = strdup(s); 
    char* e = p + strlen(p) - 1 ;  
    while(e > p && ( *e == ' ' || *e == '\n' )) e-- ;
    e[1] = '\0' ;
    return p ;  
}



inline std::string sstr::StripTail(const std::string& name, const char* end)  // static 
{
    std::string sname = name.substr(0, name.find(end)) ;
    return sname ;
}

inline std::string sstr::StripTail(const char* name_, const char* end)  // static 
{
    std::string name(name_); 
    return StripTail(name, end) ; 
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

template<typename T>
inline void sstr::split( std::vector<T>& elem, const char* str, char delim )
{
    std::stringstream ss; 
    ss.str(str)  ;
    std::string s;
    while (std::getline(ss, s, delim)) 
    {
        std::istringstream iss(s);
        T v ;  
        iss >> v ;
        elem.push_back(v) ; 
    }
}





inline void sstr::Chop( std::pair<std::string, std::string>& head__tail, const char* delim, const char* str )
{
    char* head = strdup(str); 
    char* p = strstr(head, delim);  // pointer to first occurence of delim in str or null if not found
    if(p) p[0] = '\0' ; 
    const char* tail = p ? p + strlen(delim)  : nullptr ; 
    head__tail.first = head ; 
    head__tail.second = tail ? tail : ""  ; 
}  

inline void sstr::chop( char** head, char** tail, const char* delim, const char* str )
{
    *head = strdup(str); 
    char* p = strstr(*head, delim);  // pointer to first occurence of delim in str or null if not found
    if(p) p[0] = '\0' ; 
    *tail = p ? p + strlen(delim) : nullptr ; 
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


inline bool sstr::Blank( const char* s )
{
   unsigned n = strlen(s) ; 
   return n == 0 || All(s, ' ') ; 
}

inline bool sstr::All( const char* s , char q )
{
   unsigned n = strlen(s) ; 
   return n > 0 && Count(s, q) == n ; 

}
inline unsigned sstr::Count( const char* s , char q )
{
   unsigned n = strlen(s) ; 
   unsigned count = 0 ; 
   for(unsigned i=0 ; i < n ; i++) if( s[i] == q ) count += 1 ; 
   return count ;  
}



