#pragma once
/**
sprof.h
=========


**/

#include "sstamp.h"
#include "sproc.h"

struct sprof
{
    int64_t st ;  // microsecond timestamp
    int32_t vm ;  // KB
    int32_t rs ;  // KB 

    static void Stamp(sprof& prof); 
    static std::string Desc(const sprof& prof); 

    static inline sprof Diff(const sprof& p0, const sprof& p1); 
    static std::string Desc(const sprof& p0, const sprof& p1); 
    static std::string Now(); 
    static bool LooksLikeProfileTriplet(const char* str); 
};

inline void sprof::Stamp(sprof& p)
{
    p.st = sstamp::Now(); 
    sproc::Query(p.vm, p.rs) ;  // sprof::Stamp
}

inline std::string sprof::Desc(const sprof& prof)
{
    char delim = ',' ; 
    std::stringstream ss ; 
    ss << prof.st << delim << prof.vm << delim << prof.rs ; 
    std::string str = ss.str(); 
    return str ; 
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
    ss << Desc(p0) << std::endl ; 
    ss << Desc(p1) << std::endl ; 
    ss << Desc(df) << std::endl ; 
    std::string str = ss.str(); 
    return str ; 
}





inline std::string sprof::Now()
{
    sprof now ; 
    Stamp(now);  
    return Desc(now);  
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

