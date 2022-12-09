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
    static bool     getenvbool(const char* ekey);  

    template<typename T>
    static T getenv_(const char* ekey, T fallback);  

    template<typename T>
    static std::vector<T>* make_vec(const char* line, char delim=','); 

    template<typename T>
    static std::vector<T>* getenv_vec(const char* ekey, const char* fallback, char delim=','); 

    template<typename T>
    static std::string desc_vec( const std::vector<T>* vec, unsigned edgeitems=5 ); 
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
inline bool ssys::getenvbool( const char* ekey )
{
    char* val = getenv(ekey);
    bool ival = val ? true : false ;
    return ival ; 
}





template<typename T>
inline T ssys::getenv_(const char* ekey, T fallback)
{
    char* v = getenv(ekey);
    if(v == nullptr) return fallback ; 

    std::string s(v);
    std::istringstream iss(s);
    T t ; 
    iss >> t ; 
    return t ; 
}

template int      ssys::getenv_(const char*, int ); 
template unsigned ssys::getenv_(const char*, unsigned ); 
template float    ssys::getenv_(const char*, float ); 
template double   ssys::getenv_(const char*, double ); 
template std::string ssys::getenv_(const char*, std::string ); 



template<typename T>
inline std::vector<T>* ssys::make_vec(const char* line, char delim)
{
    std::vector<T>* vec = new std::vector<T>() ; 
    std::stringstream ss;  
    ss.str(line);
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

template<typename T>
inline std::vector<T>* ssys::getenv_vec(const char* ekey, const char* fallback, char delim)
{
    assert(fallback); 
    char* line = getenv(ekey);
    return make_vec<T>( line ? line : fallback, delim );  
}


template std::vector<int>*      ssys::getenv_vec(const char*, const char*, char);
template std::vector<unsigned>* ssys::getenv_vec(const char*, const char*, char);
template std::vector<float>*    ssys::getenv_vec(const char*, const char*, char);
template std::vector<double>*   ssys::getenv_vec(const char*, const char*, char);
template std::vector<std::string>*   ssys::getenv_vec(const char*, const char*, char);
 

template<typename T>
inline std::string ssys::desc_vec( const std::vector<T>* vec, unsigned edgeitems  )
{
    unsigned size = vec ? vec->size() : 0 ; 

    std::stringstream ss ; 
    ss << "(" ; 
    for(unsigned i=0 ; i < size ; i++) if(i < edgeitems || i > size - edgeitems ) ss << (*vec)[i] << " " ; 
    ss << ")" ; 

    std::string s = ss.str(); 
    return s; 
}


template std::string ssys::desc_vec(const std::vector<int>* , unsigned ) ; 
template std::string ssys::desc_vec(const std::vector<unsigned>* , unsigned ) ; 
template std::string ssys::desc_vec(const std::vector<float>* , unsigned ) ; 
template std::string ssys::desc_vec(const std::vector<double>* , unsigned ) ; 
template std::string ssys::desc_vec(const std::vector<std::string>* , unsigned ) ; 



