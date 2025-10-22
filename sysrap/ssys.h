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
#include <iomanip>
#include <map>
#include <limits>
#include <cstdint>

#include "sstr.h"
#include "spath.h"

extern char **environ;

struct ssys
{
    static constexpr const bool VERBOSE = false ;
    static constexpr const char* GETENVVAR_PATH_PREFIX = "filepath:" ;


    static std::string popen(const char* cmd, bool chomp=true, int* rc=nullptr);
    static std::string popen(const char* cmda, const char* cmdb, bool chomp=true, int* rc=nullptr);

    static std::string uname(const char* args="-a");
    static std::string which(const char* script);

    static bool value_is_path_prefixed(const char* val );
    static const char* get_replacement_path(const char* val );

    static std::string getenviron(const char* q=nullptr);
    static bool is_under_ctest();

    static const char* getenvvar(const char* ekey );
    static const char* getenvvar(const char* ekey, const char* fallback);
    static const char* getenvvar(const char* ekey, const char* fallback, char q, char r );


    static int getenv_ParseInt(const char* ekey, const char* fallback);
    static int64_t getenv_ParseInt64(const char* ekey, const char* fallback);
    static std::vector<int>* getenv_ParseIntSpecList(const char* ekey, const char* fallback);

    static unsigned long long getenvull(const char* ekey, unsigned long long fallback);
    static int getenvint(const char* ekey, int fallback);
    static int64_t getenvint64(const char* ekey, int64_t fallback);
    static int getenvintspec( const char* ekey, const char* fallback);
    static int64_t  getenvint64spec( const char* ekey, const char* fallback);
    static uint64_t getenvuint64spec(const char* ekey, const char* fallback );

    static int getenvintpick( const char* ekey, const std::vector<std::string>& strs, int fallback );

    static unsigned getenvunsigned(const char* ekey, unsigned fallback);
    static unsigned getenvunsigned_fallback_max(const char* ekey );



    static double   getenvdouble(const char* ekey, double fallback);
    static float    getenvfloat(const char* ekey, float fallback);
    static bool     getenvbool(const char* ekey);


    static int countenv(const char* ekey, char delim=',');

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

    // THESE METHODS ARE TO ASSIST MIGRATION FROM THE OLD SSys.hh
    static std::vector<int>*   getenvintvec( const char* envkey, char delim=',');
    static void getenvintvec( const char* ekey, std::vector<int>& vec, char delim, const char* fallback ) ;
    static std::vector<float>* getenvfloatvec(const char* envkey, const char* fallback, char delim=',');


    template<typename T>
    static std::string desc_vec( const std::vector<T>* vec, unsigned edgeitems=5 );

    static int  idx_listed( const std::vector<std::string>* nn, const char* n );
    static bool  is_listed( const std::vector<std::string>* nn, const char* n );
    static int              listed_count(       std::vector<int>* ncount, const std::vector<std::string>* nn, const char* n );
    static std::string desc_listed_count( const std::vector<int>* ncount, const std::vector<std::string>* nn );

    static bool is_remote_session();
    static const char* username();

    static void Dump(const char* msg);
    static int run(const char* cmd);

    static int setenvvar( const char* ekey, const char* value, bool overwrite=true, char special_empty_token='\0' );
    static int setenvmap( const std::map<std::string, std::string>& env, bool overwrite=true , char special_empty_token='\0' );

    template<typename ... Args>
    static int setenvctx( Args ... args  );

    static std::string Desc();
    static std::string PWD();

    static void getenv_with_prefix(std::vector<std::pair<std::string,std::string>>& kvs, const char* prefix) ;


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



inline bool ssys::value_is_path_prefixed(const char* val )
{
    return val && strlen(val) > strlen(GETENVVAR_PATH_PREFIX) && strncmp(val, GETENVVAR_PATH_PREFIX, strlen(GETENVVAR_PATH_PREFIX)) == 0 ;
}

inline const char* ssys::get_replacement_path(const char* val )
{
    assert(value_is_path_prefixed(val)) ;
    return val ? strdup(val + strlen(GETENVVAR_PATH_PREFIX)) : nullptr ;
}



inline std::string ssys::getenviron(const char* q)
{
    char** e = environ ;
    std::stringstream ss ;
    while(*e)
    {
        if( q == nullptr || strstr(*e, q)) ss << *e << "\n" ;
        e++ ;
    }
    std::string str = ss.str();
    return str;
}

inline bool ssys::is_under_ctest()
{
    return countenv("DASHBOARD_TEST_FROM_CTEST,DART_TEST_FROM_DART", ',') > 0 ;
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


If the string value of the envvar starts with GETENVVAR_PATH_PREFIX "filepath:"
then the remainder of the string is intepreted as a file path which is loaded
to replace the value or nullptr when no file is found.

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

    bool is_path_prefixed = value_is_path_prefixed(val) ;

    /*
    std::cout << "ssys::getenvvar "
              << " ekey " << ( ekey ? ekey : "-" )
              << " val  " << ( val ? val : "-" )
              << " is_path_prefixed " << is_path_prefixed
              << std::endl
              ;
    */

    if(is_path_prefixed)
    {
        const char* path = get_replacement_path(val) ;

        std::string txt ;
        bool path_exists = spath::Read( txt, path );
        val = path_exists ? strdup(txt.c_str()) : nullptr ;

        if(VERBOSE) std::cout
            << "ssys::getenvvar.is_path_prefixed "
            << " ekey " << ( ekey ? ekey : "-" )
            << " path " << ( path ? path : "-" )
            << " path_exists " << ( path_exists ? "YES" : "NO " )
            << " val " << ( val ? val : "-" )
            << std::endl
            ;

    }
    return val ;
}
inline const char* ssys::getenvvar(const char* ekey, const char* fallback)
{
    const char* val = getenvvar(ekey);
    return ( val && strlen(val)>0) ? val : fallback ;   // 2024/12 "" => fallback
}
inline const char* ssys::getenvvar(const char* ekey, const char* fallback, char q, char r)
{
     const char* v = getenvvar(ekey, fallback) ;
     char* vv = v ? strdup(v) : nullptr  ;
     for(int i=0 ; i < int(vv ? strlen(vv) : 0) ; i++) if(vv[i] == q ) vv[i] = r ;
     return vv ;
}


inline int ssys::getenv_ParseInt(const char* ekey, const char* fallback)
{
    const char* spec = getenvvar(ekey, fallback);
    bool valid = spec != nullptr && strlen(spec) > 0 ;
    if(!valid)
    {
        std::cerr
            << "ssys::getenv_ParseInt"
            << " ekey " << ( ekey ? ekey : "-" )
            << " fallback " << ( fallback ? fallback : "-" )
            << " spec [" << ( spec ? spec :  "-" ) << "]"
            << " valid " << ( valid ? "YES" : "NO " )
            << "\n"
            ;

        return -1 ;
    }
    return sstr::ParseInt<int>(spec) ;
}


inline int64_t ssys::getenv_ParseInt64(const char* ekey, const char* fallback)
{
    const char* spec = getenvvar(ekey, fallback);
    bool valid = spec != nullptr && strlen(spec) > 0 ;
    if(!valid)
    {
        std::cerr
            << "ssys::getenv_ParseInt64"
            << " ekey " << ( ekey ? ekey : "-" )
            << " fallback " << ( fallback ? fallback : "-" )
            << " spec [" << ( spec ? spec :  "-" ) << "]"
            << " valid " << ( valid ? "YES" : "NO " )
            << "\n"
            ;

        return -1 ;
    }
    return sstr::ParseInt<int64_t>(spec) ;
}




















inline std::vector<int>* ssys::getenv_ParseIntSpecList(const char* ekey, const char* fallback)
{
    const char* spec = getenvvar(ekey, fallback);
    bool valid = spec != nullptr && strlen(spec) > 0 ;
    if(!valid) return nullptr ;
    return sstr::ParseIntSpecList<int>( spec, ',' );
}


inline unsigned long long ssys::getenvull(const char* ekey, unsigned long long fallback)
{
    char* val = getenv(ekey);
    return val ? std::atoll(val) : fallback ;
}



inline int ssys::getenvint(const char* ekey, int fallback)
{
    char* val = getenv(ekey);
    return val ? std::atoi(val) : fallback ;
}

inline int64_t ssys::getenvint64(const char* ekey, int64_t fallback)
{
    char* val = getenv(ekey);
    return val ? std::atol(val) : fallback ;
}




/**
ssys::getenvintspec
--------------------

Uses sstr::ParseInt to convert spec like M1 M2 k10 to integers.

**/

inline int ssys::getenvintspec(const char* ekey, const char* fallback)
{
    char* val = getenv(ekey);
    const char* spec = val ? val : fallback ;
    int ival = sstr::ParseInt<int>( spec ? spec : "0" );
    return ival ;
}

inline int64_t ssys::getenvint64spec(const char* ekey, const char* fallback)
{
    char* val = getenv(ekey);
    const char* spec = val ? val : fallback ;
    int64_t ival = sstr::ParseInt<int64_t>( spec ? spec : "0" );
    return ival ;
}

inline uint64_t ssys::getenvuint64spec(const char* ekey, const char* fallback)
{
    char* val = getenv(ekey);
    const char* spec = val ? val : fallback ;
    uint64_t ival = sstr::ParseInt<uint64_t>( spec ? spec : "0" );
    return ival ;
}


inline int ssys::getenvintpick(const char* ekey, const std::vector<std::string>& strs, int fallback )
{
    char* v = getenv(ekey);
    if(v == nullptr) return fallback ;

    int pick = fallback ;
    int num_str = strs.size() ;
    for(int i=0 ; i < num_str ; i++)
    {
        const char* str = strs[i].c_str();
        if( str && v && strcmp(str, v) == 0 )
        {
            pick = i ;
            break ;
        }
    }
    return pick ;
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

    /*
    // special casing a value indicating NO ?
    if(val)
    {
        if(strcmp(val,"NO") == 0) ival = false ;
        if(strcmp(val,"no") == 0) ival = false ;
        if(strcmp(val,"False") == 0) ival = false ;
        if(strcmp(val,"false") == 0) ival = false ;
        if(strcmp(val,"0") == 0) ival = false ;
    }
    */

    return ival ;
}

inline float  ssys::getenvfloat( const char* ekey, float  fallback){ return getenv_<float>(ekey,  fallback) ; }
inline double ssys::getenvdouble(const char* ekey, double fallback){ return getenv_<double>(ekey, fallback) ; }


/**
ssys::countenv
---------------

**/

inline int ssys::countenv(const char* ekey, char delim)
{
    std::vector<std::string> keys ;
    sstr::Split(ekey, delim, keys) ;

    int count = 0 ;
    char* val = nullptr ;
    for(unsigned i=0 ; i < keys.size() ; i++)
    {
        const char* key = keys[i].c_str();
        val = getenv(key) ;
        if( val != nullptr ) count += 1 ;
    }
    return count ;
}


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

NB spath::Resolve provides much more flexible tokenization replacement

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
    int len = v ? strlen(v) : 0 ;
    return len == 0  ? fallback : parse<T>(v) ;
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
    if(line_ == nullptr && fallback == nullptr) return ;
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
    if(line == nullptr) return nullptr ;
    std::vector<T>* vec = new std::vector<T>() ;
    fill_vec<T>( *vec, line, delim );
    return vec ;
}






template<typename T>
inline std::vector<T>* ssys::getenv_vec(const char* ekey, const char* fallback, char delim )
{
    char* line = getenv(ekey);
    bool valid = line && strlen(line) > 0 ;  // blanks line are not valid
    return make_vec<T>( valid ? line : fallback, delim );
}


template std::vector<int>*      ssys::getenv_vec(const char*, const char*, char );
template std::vector<unsigned>* ssys::getenv_vec(const char*, const char*, char );
template std::vector<float>*    ssys::getenv_vec(const char*, const char*, char );
template std::vector<double>*   ssys::getenv_vec(const char*, const char*, char );
template std::vector<std::string>*   ssys::getenv_vec(const char*, const char*, char );


inline std::vector<int>* ssys::getenvintvec(const char* envkey, char delim)
{
    return getenv_vec<int>(envkey, nullptr, delim);
}

inline void ssys::getenvintvec( const char* ekey, std::vector<int>& vec, char delim, const char* fallback )
{
    fill_evec<int>( vec, ekey, fallback, delim );
}




inline std::vector<float>* ssys::getenvfloatvec(const char* envkey, const char* fallback, char delim)
{
    return getenv_vec<float>(envkey, fallback, delim);
}






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


/**
ssys::idx_listed
------------------

* if n is found within nn returns the index in range 0 to size-1 inclusive
* if n is not found returns size
* if nn is null return -1

**/

inline int ssys::idx_listed( const std::vector<std::string>* nn, const char* n ) // static
{
    return nn ? std::distance( nn->begin(), std::find( nn->begin(), nn->end(), n ) ) : -1 ;
}

inline bool ssys::is_listed( const std::vector<std::string>* nn, const char* n ) // static
{
    int sz = nn ? nn->size() : 0 ;
    int idx = idx_listed(nn, n) ;
    return idx > -1 && idx < sz ;
}

/**
ssys::listed_count
--------------------

1. ncount vector is resized to match the size of nn
2. index of the n within nn is found
3. count for that index is accessed from ncount vector
4. ncount entry for the index is incremented
5. count is returned providing a 0-based occurrence index

**/
inline int ssys::listed_count( std::vector<int>* ncount, const std::vector<std::string>* nn, const char* n )
{
    if(nn == nullptr || ncount == nullptr) return -1 ;
    int sz = nn->size() ;
    ncount->resize(sz);
    int idx = idx_listed(nn,n) ;
    if(idx >= sz) return -1 ;
    int count = ncount->at(idx) ;
    (*ncount)[idx] += 1 ;
    return count ;
}


inline std::string ssys::desc_listed_count( const std::vector<int>* ncount, const std::vector<std::string>* nn )
{
    int ncount_sz = ncount ? int(ncount->size()) : -1 ;
    int nn_sz = nn ? int(nn->size()) : -1 ;

    std::stringstream ss ;
    ss << "ssys::desc_listed_count"
       << " ncount_sz " << ncount_sz
       << " nn_sz " << nn_sz
       << std::endl
       ;

    if( ncount_sz == nn_sz && nn_sz > -1 )
    {
        for(int i=0 ; i < nn_sz ; i++ ) ss << std::setw(3) << i << " : " << (*ncount)[i] << " : " << (*nn)[i] << std::endl ;
    }
    std::string str = ss.str();
    return str ;
}


/**
ssys::is_remote_session
-------------------------

Returns true when the environment has one or more of the below envvars::

    SSH_CLIENT
    SSH_TTY

**/


inline bool ssys::is_remote_session()
{
    char* ssh_client = getenv("SSH_CLIENT");
    char* ssh_tty = getenv("SSH_TTY");
    bool is_remote = ssh_client != nullptr || ssh_tty != nullptr ;
    return is_remote ;
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




/**
ssys::Dump
------------

Dump the message using std::cout std::cerr printf and std::printf, used for testing stream redirection

**/

inline void ssys::Dump(const char* msg)
{
    static int COUNT = -1 ;
    COUNT++ ;
    std::cout << std::setw(3) << COUNT << "[" << std::setw(20) << "std::cout" << "] " << msg << std::endl;
    std::cerr << std::setw(3) << COUNT << "[" << std::setw(20) << "std::cerr" << "] " << msg << std::endl;
    printf("%3d[%20s] %s \n", COUNT, "printf", msg );
    std::printf("%3d[%20s] %s \n", COUNT, "std::printf", msg );
    std::cerr << std::endl  ;
}


inline int ssys::run(const char* cmd)
{
    int rc_raw = system(cmd);
    int rc =  WEXITSTATUS(rc_raw) ;

    std::cout
        << "ssys::run "
        <<  ( cmd ? cmd : "-" )
        << " rc_raw : " << rc_raw
        << " rc : " << rc
        << std::endl
        ;

    if(rc != 0) std::cout
        << "ssys::run"
        << " PATH ENVVAR MISCONFIGURED ? "
        << std::endl
        ;
    return rc ;
}




/**
ssys::setenvvar
-----------------

overwrite:false
    preexisting envvar is not overridden.

"value[0] == special_empty_token" and special_empty_token not default '\0' (eg use '-')
    indicates want value to be empty string, avoiding inconvenient shell
    handling of empty strings


**/


inline int ssys::setenvvar( const char* ekey, const char* value, bool overwrite, char special_empty_token)
{
    std::stringstream ss ;
    ss << ekey << "=" ;

    if(value)
    {
        if(special_empty_token != '\0' && strlen(value) == 1 && value[0] == special_empty_token)
        {
            ss << "" ;
        }
        else
        {
            ss << value ;
        }
    }

    std::string ekv = ss.str();
    const char* prior = getenv(ekey) ;

    char* ekv_ = const_cast<char*>(strdup(ekv.c_str()));

    int rc = ( overwrite || !prior ) ? putenv(ekv_) : 0  ;

    const char* after = getenv(ekey) ;

    if(VERBOSE) std::cerr
        << "ssys::setenvvar"
        << " ekey " << ekey
        << " ekv " << ekv
        << " overwrite " << overwrite
        << " prior " << ( prior ? prior : "NULL" )
        << " value " << ( value ? value : "NULL" )
        << " after " << ( after ? after : "NULL" )
        << " rc " << rc
        << std::endl
        ;

    //std::raise(SIGINT);
    return rc ;
}


inline int ssys::setenvmap( const std::map<std::string, std::string>& env, bool overwrite, char special_empty_token )
{
    typedef std::map<std::string, std::string>  MSS ;
    for(MSS::const_iterator it=env.begin() ; it != env.end() ; it++)
    {
        const std::string& key = it->first ;
        const std::string& val = it->second ;
        setenvvar(key.c_str(), val.c_str(), overwrite, special_empty_token );
    }
    return 0 ;
}


template<typename ... Args>
inline int ssys::setenvctx( Args ... args_  )
{
    std::vector<std::string> args = {args_...};
    std::vector<std::string> elem ;

    for(unsigned i=0 ; i < args.size() ; i++)
    {
        const std::string& arg = args[i] ;
        if(!arg.empty()) elem.push_back(arg);
    }

    unsigned num_elem = elem.size() ;
    assert( num_elem % 2 == 0 );

    bool overwrite = true ;
    char special_empty_token = '\0' ;

    for(unsigned i=0 ; i < num_elem/2 ; i++)
    {
        const std::string& key = elem[2*i+0] ;
        const std::string& val = elem[2*i+1] ;
        setenvvar(key.c_str(), val.c_str(), overwrite, special_empty_token );
    }
    return 0 ;
}


template int ssys::setenvctx(
          const char*, const char* );
template int ssys::setenvctx(
          const char*, const char*,
          const char*, const char* );
template int ssys::setenvctx(
          const char*, const char*,
          const char*, const char*,
          const char*, const char* );
template int ssys::setenvctx(
          const char*, const char*,
          const char*, const char*,
          const char*, const char*,
          const char*, const char* );



/**
ssys::Desc
-------------

Generated with::

   ~/opticks/sysrap/ssys__Desc.sh
   ~/opticks/sysrap/ssys__Desc.sh | pbcopy

Dump flags with::

    ssys_test
    QSimDescTest

**/
inline std::string ssys::Desc()  // static
{
    std::stringstream ss ;
    ss << "ssys::Desc"
       << std::endl
#ifdef CONFIG_Debug
       << "CONFIG_Debug"
#else
       << "NOT:CONFIG_Debug"
#endif
       << std::endl
#ifdef CONFIG_Release
       << "CONFIG_Release"
#else
       << "NOT:CONFIG_Release"
#endif
       << std::endl
#ifdef CONFIG_RelWithDebInfo
       << "CONFIG_RelWithDebInfo"
#else
       << "NOT:CONFIG_RelWithDebInfo"
#endif
       << std::endl
#ifdef CONFIG_MinSizeRel
       << "CONFIG_MinSizeRel"
#else
       << "NOT:CONFIG_MinSizeRel"
#endif
       << std::endl
#ifdef PRODUCTION
       << "PRODUCTION"
#else
       << "NOT:PRODUCTION"
#endif
       << std::endl
#ifdef WITH_CHILD
       << "WITH_CHILD"
#else
       << "NOT:WITH_CHILD"
#endif
       << std::endl
#ifdef WITH_CUSTOM4
       << "WITH_CUSTOM4"
#else
       << "NOT:WITH_CUSTOM4"
#endif
       << std::endl
#ifdef PLOG_LOCAL
       << "PLOG_LOCAL"
#else
       << "NOT:PLOG_LOCAL"
#endif
       << std::endl
#ifdef DEBUG_PIDX
       << "DEBUG_PIDX"
#else
       << "NOT:DEBUG_PIDX"
#endif
       << std::endl
#ifdef DEBUG_TAG
       << "DEBUG_TAG"
#else
       << "NOT:DEBUG_TAG"
#endif
       << std::endl
       ;
    std::string str = ss.str() ;
    return str ;
}

inline std::string ssys::PWD()  // static
{
    return getenvvar("PWD");    // note no newline
}


#ifdef _MSC_VER
#else
#include <unistd.h>
extern char **environ;
#endif


inline void ssys::getenv_with_prefix(std::vector<std::pair<std::string,std::string>>& kvs, const char* prefix)
{
#ifdef _MSC_VER
#else
    int i=0 ;
    char delim='=' ;
    while(environ[i])
    {
        std::string kv = environ[i++] ;
        size_t pos = kv.find(delim);
        if( pos == std::string::npos ) continue ;
        std::string k = kv.substr(0, pos);
        std::string v = kv.substr(pos+1);
        bool match_prefix = prefix ? sstr::StartsWith( k.c_str(), prefix ) : true ;
        //if(match_prefix) std::cout << "[" << k << "][" << v << "]" << ( match_prefix ? "YES" : "NO "  ) << std::endl ;
        if(!match_prefix) continue ;
        kvs.push_back( std::pair<std::string,std::string>( k, v ) );
    }
#endif
}


