#pragma once

struct sstr
{
    static const char* TrimTrailing(const char* s);
    static void PrefixSuffixParse(std::vector<std::string>& elem, const char* prefix, const char* suffix, const char* lines); 
};


#include <string>
#include <vector>
#include <cstring>
#include <sstream>

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

