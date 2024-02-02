#pragma once
/**
sprof.h
=========

Assuming sprof are persisted into an array rp of shape (n,3) with a two name layout::

    In [10]: fold.runprof_names[:4]
    Out[10]: array(['SEvt__setIndex_A000', 'SEvt__endIndex_A000', 'SEvt__setIndex_A001', 'SEvt__endIndex_A001'], dtype='<U19')

Then obtain times and VM, RS with::

    tp = (rp[:,0] - rp[0,0])/1e6         # seconds from first profile stamp
    vm = rp[:,1]/1e6                     # GB
    rs = rp[:,2]/1e6                     # GB
    drm = (rp[1::2,2] - rp[0::2,2])/1e3  # MB between each pair  

See for example ~/opticks/sysrap/tests/sleak.py 

**/

#include "sstamp.h"
#include "sproc.h"

struct sprof
{
    int64_t st ;  // microsecond timestamp
    int32_t vm ;  // KB
    int32_t rs ;  // KB 

    static bool Equal(const sprof& a, const sprof& b); 
    static void Stamp(sprof& prof); 
    static std::string Serialize(const sprof& prof);  // formerly Desc
    static std::string Desc_(const sprof& prof);  // dont call Desc for now as need to change all Desc to Serialize
    static int         Import(sprof& prof, const char* str); 

    static inline sprof Diff(const sprof& p0, const sprof& p1); 
    static std::string Desc(const sprof& p0, const sprof& p1); 
    static std::string Now(); 
    static bool LooksLikeProfileTriplet(const char* str); 

};

inline bool sprof::Equal(const sprof& a, const sprof& b)
{
    return a.st == b.st && a.vm == b.vm && a.rs == b.rs ; 
}

inline void sprof::Stamp(sprof& p)
{
    p.st = sstamp::Now(); 
    sproc::Query(p.vm, p.rs) ;  // sprof::Stamp
}

inline std::string sprof::Serialize(const sprof& prof)
{
    char delim = ',' ; 
    std::stringstream ss ; 
    ss << prof.st << delim << prof.vm << delim << prof.rs ; 
    std::string str = ss.str(); 
    return str ; 
}

inline std::string sprof::Desc_(const sprof& prof)  
{
    std::stringstream ss ; 
    ss << std::setw(20) << prof.st 
       << std::setw(10) << prof.vm 
       << std::setw(10) << prof.rs 
       << " " << sstamp::Format(prof.st) 
       ; 
    std::string str = ss.str(); 
    return str ; 
}

inline int sprof::Import(sprof& prof, const char* str)
{
    if(!LooksLikeProfileTriplet(str)) return 1 ; 
    char* end ; 
    prof.st = strtoll( str, &end, 10 ) ; 
    prof.vm = strtoll( end+1, &end, 10 ) ; 
    prof.rs = strtoll( end+1, &end, 10 ) ; 
    return 0 ; 
}

inline sprof sprof::Diff(const sprof& p0, const sprof& p1)
{
    sprof df ;
    df.st = p1.st - p0.st ; 
    df.vm = p1.vm - p0.vm ; 
    df.rs = p1.rs - p0.rs ; 
    return df ; 
}

inline std::string sprof::Desc(const sprof& p0, const sprof& p1)
{
    sprof df = Diff(p0, p1) ; 
    std::stringstream ss ; 
    ss << Desc_(p0) << std::endl ; 
    ss << Desc_(p1) << std::endl ; 
    ss << Desc_(df) << std::endl ; 
    std::string str = ss.str(); 
    return str ; 
}


inline std::string sprof::Now()
{
    sprof now ; 
    Stamp(now);  
    return Serialize(now);  
}



/**
sprof::LooksLikeProfileTriplet
--------------------------------

sprof strings have:

1. first field of 16 digits, see sstamp::LooksLikeStampInt to understand why 16 
2. two ',' delimiters
3. only digits and delimiters

For example::

    1700367820015746,4370198,884

**/

inline bool sprof::LooksLikeProfileTriplet(const char* str)
{
    int len = str ? int(strlen(str)) : 0 ; 
    int count_delim = 0 ; 
    int count_non_digit = 0 ; 
    int first_field_digits = 0 ; 

    for(int i=0 ; i < len ; i++ ) 
    {
        char c = str[i] ; 
        bool is_digit = c >= '0' && c <= '9' ;
        bool is_delim = c == ',' ; 
        if(!is_digit) count_non_digit += 1 ; 
        if(count_delim == 0 && is_digit ) first_field_digits += 1 ;  
        if(is_delim) count_delim += 1 ; 
    }
    bool heuristic = count_delim == 2 && count_non_digit == count_delim && first_field_digits == 16 ; 
    return heuristic ;    
}

