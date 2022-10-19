#pragma once
/**
ssys.h
========

**/

#include <cstdlib>
#include <cassert>
#include <cstring>
#include <string>
#include <sstream>
#include <vector>

struct ssys
{
    static std::string popen(const char* cmd, bool chomp=true, int* rc=nullptr);      
    static const char* getenvvar(const char* ekey, const char* fallback); 
    static int getenvint(const char* ekey, int fallback);  
    static unsigned getenvunsigned(const char* ekey, unsigned fallback);  

    template<typename T>
    static T getenvv(const char* ekey, T fallback);  

    template<typename T>
    static std::vector<T>* getenvvec(const char* ekey, const char* fallback, char delim=','); 

    template<typename T>
    static std::string DescVec( const std::vector<T>* vec, unsigned edgeitems=5 ); 
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



template<typename T>
inline std::vector<T>* ssys::getenvvec(const char* ekey, const char* fallback, char delim)
{
    assert(fallback); 
    std::vector<T>* vec = new std::vector<T>() ; 
    char* line = getenv(ekey);
    std::stringstream ss; 
    ss.str(line ? line : fallback);
    std::string s;
    while (std::getline(ss, s, delim)) 
    {
        std::istringstream iss(s);
        T t ; 
        iss >> t ; 
        vec->push_back(t) ; 
    }
    return vec ; 
}



template std::vector<int>*      ssys::getenvvec(const char*, const char*, char);
template std::vector<unsigned>* ssys::getenvvec(const char*, const char*, char);
template std::vector<float>*    ssys::getenvvec(const char*, const char*, char);
template std::vector<double>*   ssys::getenvvec(const char*, const char*, char);
 
template<typename T>
inline std::string ssys::DescVec( const std::vector<T>* vec, unsigned edgeitems  )
{
    unsigned size = vec ? vec->size() : 0 ; 

    std::stringstream ss ; 
    ss << "(" ; 
    for(unsigned i=0 ; i < size ; i++) if(i < edgeitems || i > size - edgeitems ) ss << (*vec)[i] << " " ; 
    ss << ")" ; 

    std::string s = ss.str(); 
    return s; 
}


template std::string ssys::DescVec(const std::vector<int>* , unsigned ) ; 
template std::string ssys::DescVec(const std::vector<unsigned>* , unsigned ) ; 
template std::string ssys::DescVec(const std::vector<float>* , unsigned ) ; 
template std::string ssys::DescVec(const std::vector<double>* , unsigned ) ; 
