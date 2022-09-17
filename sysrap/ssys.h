#pragma once
/**
ssys.h
========

**/

#include <cstdlib>
#include <cstring>
#include <string>
#include <sstream>

struct ssys
{
    static std::string popen(const char* cmd, bool chomp=true, int* rc=nullptr);      
    static const char* getenvvar(const char* ekey, const char* fallback); 
    static int getenvint(const char* ekey, int fallback);  
    static unsigned getenvunsigned(const char* ekey, unsigned fallback);  

    template<typename T>
    static T getenvv(const char* ekey, T fallback);  
}; 


inline std::string ssys::popen(const char* cmd, bool chomp, int* rc)
{
    std::stringstream ss ; 
    FILE *fp = ::popen(cmd, "r");
    char line[512];    
    while (fgets(line, sizeof(line), fp) != NULL) 
    {   
       if(chomp) line[strcspn(line, "\n")] = 0;
       ss << line ;   
    }   

    int retcode=0 ; 
    int st = pclose(fp);
    if(WIFEXITED(st)) retcode=WEXITSTATUS(st);

    if(rc) *rc = retcode ; 

    std::string s = ss.str(); 
    return s ; 
}

inline const char* ssys::getenvvar(const char* ekey, const char* fallback)
{
    char* val = getenv(ekey);
    return val ? val : fallback ; 
}
inline int ssys::getenvint(const char* ekey, int fallback)
{
    char* val = getenv(ekey);
    return val ? std::atoi(val) : fallback ; 
}

inline unsigned ssys::getenvunsigned(const char* ekey, unsigned fallback)
{
    int ival = getenvint(ekey, int(fallback)); 
    return ival > -1 ? ival : fallback ; 
}

template<typename T>
inline T ssys::getenvv(const char* ekey, T fallback)
{
    char* v = getenv(ekey);
    if(v == nullptr) return fallback ; 

    std::string s(v);
    std::istringstream iss(s);
    T t ; 
    iss >> t ; 
    return t ; 
}

template int      ssys::getenvv(const char*, int ); 
template unsigned ssys::getenvv(const char*, unsigned ); 
template float    ssys::getenvv(const char*, float ); 
template double   ssys::getenvv(const char*, double ); 

