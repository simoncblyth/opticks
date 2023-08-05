#pragma once
/**
ssys.h
========

Note that strings like "1e-9" parse ok into float/double. 

**/

#include <cstdlib>
#include <cassert>
#include <cstring>
#include <string>
#include <regex>
#include <sstream>
#include <vector>
#include <iostream>
#include <map>
#include <limits>

#include "sstr.h"


struct ssys
{
    static constexpr const bool VERBOSE = false ; 

    static std::string popen(const char* cmd, bool chomp=true, int* rc=nullptr);      
    static std::string popen(const char* cmda, const char* cmdb, bool chomp=true, int* rc=nullptr); 

    static std::string uname(const char* args="-a"); 
    static std::string which(const char* script); 

    static const char* getenvvar(const char* ekey ); 
    static const char* getenvvar(const char* ekey, const char* fallback); 
    static const char* getenvvar(const char* ekey, const char* fallback, char q, char r ); 


    static int getenvint(const char* ekey, int fallback);  
    static unsigned getenvunsigned(const char* ekey, unsigned fallback);  
    static unsigned getenvunsigned_fallback_max(const char* ekey );  

    static double   getenvdouble(const char* ekey, double fallback);  
    static float    getenvfloat(const char* ekey, float fallback);  
    static bool     getenvbool(const char* ekey);  




    static bool hasenv_(const char* ekey);  
    static bool hastoken_(const char* ekey); 
    static char* _getenv(const char* ekey); 
    static char* replace_envvar_token(const char* ekey);  // checks for token 
    static char* _replace_envvar_token(const char* ekey); // MUST have token    
    static char* _tokenized_getenv(const char* ekey); 


    template<typename T>
    static T parse(const char* str);  


    template<typename T>
    static T getenv_(const char* ekey, T fallback);  

    template<typename T>
    static void getenv_(std::vector<std::pair<std::string, T>>& kv, const std::vector<std::string>& kk ); 

    template<typename T>
    static void getenv_(std::vector<std::pair<std::string, T>>& kv, const char* kk ); 

    template<typename T>
    static void fill_vec( std::vector<T>& vec, const char* line, char delim=',' ); 

    template<typename T>
    static void fill_evec( std::vector<T>& vec, const char* ekey, const char* fallback, char delim ); 


    template<typename T>
    static std::vector<T>* make_vec(const char* line, char delim=','); 

    template<typename T>
    static std::vector<T>* getenv_vec(const char* ekey, const char* fallback, char delim=','); 


    template<typename T>
    static std::string desc_vec( const std::vector<T>* vec, unsigned edgeitems=5 ); 

    static bool is_listed( const std::vector<std::string>* vec, const char* name ); 
    static const char* username(); 
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

inline std::string ssys::popen(const char* cmda, const char* cmdb, bool chomp, int* rc)
{
    std::stringstream ss ; 
    if(cmda) ss << cmda ; 
    ss << " " ; 
    if(cmdb) ss << cmdb ; 

    std::string s = ss.str(); 
    return popen(s.c_str(), chomp, rc ); 
}

inline std::string ssys::uname(const char* args)
{
    bool chomp = true ; 
    int rc(0);  
    std::string line = ssys::popen("uname", args, chomp, &rc ); 
    return rc == 0 ? line : "" ; 
}


inline std::string ssys::which(const char* script)
{
    bool chomp = true ; 
    int rc(0); 
    std::string path = ssys::popen("which 2>/dev/null", script, chomp, &rc );

    if(VERBOSE) std::cerr
         << " script " << script
         << " path " << path 
         << " rc " << rc
         << std::endl 
         ;

    std::string empty ; 
    return rc == 0 ? path : empty ; 
}

/**
ssys::getenvvar
----------------

For ekey with a comma such as "OPTICKS_ELV_SELECTION,ELV" the 
envvars are checked in order and the first to yield a value 
is returned.:: 

    OPTICKS_ELV_SELECTION=greetings ELV=hello ./ssys_test.sh 
    test_getenvvar ekey OPTICKS_ELV_SELECTION,ELV val greetings

    OPTICKS_ELV_SELECTION_=greetings ELV=hello ./ssys_test.sh 
    test_getenvvar ekey OPTICKS_ELV_SELECTION,ELV val hello

**/

inline const char* ssys::getenvvar(const char* ekey)
{
    std::vector<std::string> keys ; 
    sstr::Split(ekey, ',', keys) ; 
    char* val = getenv(ekey);
    for(unsigned i=0 ; i < keys.size() ; i++)
    {
        const char* key = keys[i].c_str(); 
        val = getenv(key) ; 
        if( val != nullptr ) break ; 
    }
    return val ; 
}
inline const char* ssys::getenvvar(const char* ekey, const char* fallback)
{
    const char* val = getenvvar(ekey);
    return val ? val : fallback ; 
}
inline const char* ssys::getenvvar(const char* ekey, const char* fallback, char q, char r)
{
     const char* v = getenvvar(ekey, fallback) ; 
     char* vv = v ? strdup(v) : nullptr  ; 
     for(int i=0 ; i < int(vv ? strlen(vv) : 0) ; i++) if(vv[i] == q ) vv[i] = r ;   
     return vv ; 
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
inline unsigned ssys::getenvunsigned_fallback_max(const char* ekey)
{
    return getenvunsigned(ekey, std::numeric_limits<unsigned>::max() ); 
}


inline bool ssys::getenvbool( const char* ekey )
{
    char* val = getenv(ekey);
    bool ival = val ? true : false ;
    return ival ; 
}

inline float  ssys::getenvfloat( const char* ekey, float  fallback){ return getenv_<float>(ekey,  fallback) ; }
inline double ssys::getenvdouble(const char* ekey, double fallback){ return getenv_<double>(ekey, fallback) ; }



inline bool ssys::hasenv_(const char* ekey)
{
    return ekey != nullptr && ( getenv(ekey) != nullptr ) ; 
}

inline bool ssys::hastoken_(const char* ekey)
{
    return ekey != nullptr && strstr(ekey, "${") != nullptr && strstr(ekey, "}") != nullptr ; 
}

/**
ssys::_getenv
---------------

This handles higher order ekey such as "${GEOM}_GEOMList" when the environmnent is::

    export GEOM=FewPMT
    export ${GEOM}_GEOMList=hamaLogicalPMT

**/

inline char* ssys::_getenv(const char* ekey)
{
    if(ekey == nullptr) return nullptr ; 
    return !hastoken_(ekey) ? getenv(ekey) : _tokenized_getenv(ekey ) ; 
}

/**
ssys::replace_envvar_token
----------------------------

1. extract VAR from head of string "${VAR}rest-of-string"  
2. construct string with the "${VAR}" replaced with its value obtained from envvar lookup, 
   when the envvar does not exist returns eg "VARrest-of-string" 

**/

inline char* ssys::replace_envvar_token(const char* ekey)
{
    return !hastoken_(ekey) ? strdup(ekey) : _replace_envvar_token(ekey ) ; 
}
inline char* ssys::_replace_envvar_token(const char* ekey)
{
    std::stringstream ss ; 

    char* ek = strdup(ekey) ; 
    char* o = strstr(ek, "${" ); 
    char* c = strstr(ek, "}" ); 
    char* t = c ? c+1 : nullptr ; 

    if( o != ek )  // chars before the token
    {
        *o = '\0' ; // temporily terminate at the '$'
        ss << ek ; 
    }

    o += 2 ;    // advance past "${"
    *c = '\0' ; // terminate at position of "}" 

    char* ov = getenv(o) ; 
    ss << ( ov ? ov : o ) << t ; 
    std::string str = ss.str(); 

    return strdup(str.c_str()) ; 
}


/**
ssys::_tokenized_getenv (hmm maybe second_order_getenv better name)
---------------------------------------------------------------------

1. replace the envvar token of form ${VAR} in the ekey argument
2. assuming the resulting string is in itself an envvar look that up, 
   otherwise just return the unexpanded string 

**/

inline char* ssys::_tokenized_getenv(const char* ekey)
{
    std::string str = _replace_envvar_token(ekey) ;
    char* k = strdup(str.c_str()) ; 
    char* kv = getenv(k) ; 
    return kv ? kv : k ; 
}




template<typename T>
inline T ssys::parse(const char* str_)
{
    std::string str(str_);
    std::istringstream iss(str);
    T tval ; 
    iss >> tval ; 
    return tval ; 
}

// template specialization for T=std::string 
// otherwise the parsed value is truncated to the first element prior to any 
// whitespace within the source value 
template<>
inline std::string ssys::parse(const char* str_)
{
    std::string str( str_ ? str_ : "" ) ; 
    return str ; 
}

template int      ssys::parse(const char*); 
template unsigned ssys::parse(const char*); 
template float    ssys::parse(const char*); 
template double   ssys::parse(const char*); 



template<typename T>
inline T ssys::getenv_(const char* ekey, T fallback)
{
    char* v = getenv(ekey);
    return v == nullptr ? fallback : parse<T>(v) ;  
}

template int      ssys::getenv_(const char*, int ); 
template unsigned ssys::getenv_(const char*, unsigned ); 
template float    ssys::getenv_(const char*, float ); 
template double   ssys::getenv_(const char*, double ); 
template std::string ssys::getenv_(const char*, std::string ); 







template<typename T>
void ssys::getenv_(std::vector<std::pair<std::string, T>>& kv, const std::vector<std::string>& kk )
{
    typedef typename std::pair<std::string,T> KV ; 
    for(int i=0 ; i < int(kk.size()) ; i++)
    {
        const char* k = kk[i].c_str() ; 
        const char* v_ = _getenv(k) ;   // supports higher level tokenized envvars like ${GEOM}_GEOMList 
        if(v_ == nullptr) continue ; 

        T v = parse<T>(v_) ; 
        kv.push_back(KV(k,v)) ; 
    }
}

template<typename T>
void ssys::getenv_(std::vector<std::pair<std::string, T>>& kv, const char* kk_ )
{
    std::vector<std::string> kk ; 
    std::stringstream ss(kk_) ;    
    std::string line ; 
    while (std::getline(ss, line))  // newlines are swallowed by getline
    {   
       if(line.empty()) continue ;   
       line = std::regex_replace(line, std::regex(R"(^\s+|\s+$)"), "");
       if(line.empty()) continue ;   
       kk.push_back(line); 
    }
    getenv_(kv, kk );
}



template<typename T>
inline void ssys::fill_vec( std::vector<T>& vec, const char* line, char delim )
{
    std::stringstream ss;  
    ss.str(line);
    std::string s;
    while (std::getline(ss, s, delim)) 
    {   
        if(delim == '\n' && sstr::IsWhitespace(s)) continue ; 
        std::istringstream iss(s);
        T t ;  
        iss >> t ;  
        vec.push_back(t) ; 
    }    
}

template void ssys::fill_vec( std::vector<int>&         , const char*, char );
template void ssys::fill_vec( std::vector<unsigned>&    , const char*, char );
template void ssys::fill_vec( std::vector<float>&       , const char*, char );
template void ssys::fill_vec( std::vector<double>&      , const char*, char );
template void ssys::fill_vec( std::vector<std::string>& , const char*, char );



template<typename T>
inline void ssys::fill_evec(std::vector<T>& vec, const char* ekey, const char* fallback, char delim )
{
    assert(fallback); 
    char* line_ = getenv(ekey);
    const char* line = line_ ? line_ : fallback ; 
    fill_vec<T>( vec, line, delim ); 
} 

template void ssys::fill_evec( std::vector<int>&         , const char*, const char*, char );
template void ssys::fill_evec( std::vector<unsigned>&    , const char*, const char*, char );
template void ssys::fill_evec( std::vector<float>&       , const char*, const char*, char );
template void ssys::fill_evec( std::vector<double>&      , const char*, const char*, char );
template void ssys::fill_evec( std::vector<std::string>& , const char*, const char*, char );


template<typename T>
inline std::vector<T>* ssys::make_vec(const char* line, char delim )
{
    std::vector<T>* vec = new std::vector<T>() ; 
    fill_vec<T>( *vec, line, delim ); 
    return vec ; 
}






template<typename T>
inline std::vector<T>* ssys::getenv_vec(const char* ekey, const char* fallback, char delim )
{
    assert(fallback); 
    char* line = getenv(ekey);
    return make_vec<T>( line ? line : fallback, delim );  
}


template std::vector<int>*      ssys::getenv_vec(const char*, const char*, char );
template std::vector<unsigned>* ssys::getenv_vec(const char*, const char*, char );
template std::vector<float>*    ssys::getenv_vec(const char*, const char*, char );
template std::vector<double>*   ssys::getenv_vec(const char*, const char*, char );
template std::vector<std::string>*   ssys::getenv_vec(const char*, const char*, char );











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


inline bool ssys::is_listed( const std::vector<std::string>* nn, const char* name ) // static 
{
    return nn && std::distance( nn->begin(), std::find( nn->begin(), nn->end(), name ) ) < int(nn->size()) ;  
}


inline const char* ssys::username()
{
#ifdef _MSC_VER
    const char* user = ssys::getenvvar("USERNAME", "no-USERNAME") ;
#else
    const char* user = ssys::getenvvar("USER", "no-USER" ) ;
#endif
    return user ? user : "ssys-username-undefined" ; 
}





