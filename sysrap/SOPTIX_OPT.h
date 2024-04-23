#pragma once
/**
SOPTIX_OPT.h : enum strings
============================

**/

#include "sstr.h"

struct SOPTIX_OPT
{
    static constexpr const bool VERBOSE = false ; 

    static OptixCompileDebugLevel        DebugLevel(const char* option); 
    static const char *                  DebugLevel_( OptixCompileDebugLevel debugLevel ); 
#if OPTIX_VERSION == 70000
    static constexpr const char* OPTIX_COMPILE_DEBUG_LEVEL_NONE_     = "OPTIX_COMPILE_DEBUG_LEVEL_NONE" ; 
    static constexpr const char* OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO_ = "OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO" ; 
    static constexpr const char* OPTIX_COMPILE_DEBUG_LEVEL_FULL_     = "OPTIX_COMPILE_DEBUG_LEVEL_FULL" ; 
#elif OPTIX_VERSION >= 70500
    static constexpr const char* OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT_  = "OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT" ; 
    static constexpr const char* OPTIX_COMPILE_DEBUG_LEVEL_NONE_     = "OPTIX_COMPILE_DEBUG_LEVEL_NONE" ; 
    static constexpr const char* OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL_  = "OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL" ; 
    static constexpr const char* OPTIX_COMPILE_DEBUG_LEVEL_MODERATE_ = "OPTIX_COMPILE_DEBUG_LEVEL_MODERATE" ; 
    static constexpr const char* OPTIX_COMPILE_DEBUG_LEVEL_FULL_     = "OPTIX_COMPILE_DEBUG_LEVEL_FULL" ; 
#endif




    static OptixCompileOptimizationLevel OptimizationLevel(const char* option) ; 
    static const char* OptimizationLevel_( OptixCompileOptimizationLevel optLevel ); 
    static constexpr const char* OPTIX_COMPILE_OPTIMIZATION_DEFAULT_ = "OPTIX_COMPILE_OPTIMIZATION_DEFAULT" ;
    static constexpr const char* OPTIX_COMPILE_OPTIMIZATION_LEVEL_0_ = "OPTIX_COMPILE_OPTIMIZATION_LEVEL_0" ;
    static constexpr const char* OPTIX_COMPILE_OPTIMIZATION_LEVEL_1_ = "OPTIX_COMPILE_OPTIMIZATION_LEVEL_1" ;
    static constexpr const char* OPTIX_COMPILE_OPTIMIZATION_LEVEL_2_ = "OPTIX_COMPILE_OPTIMIZATION_LEVEL_2" ; 
    static constexpr const char* OPTIX_COMPILE_OPTIMIZATION_LEVEL_3_ = "OPTIX_COMPILE_OPTIMIZATION_LEVEL_3" ;




    static OptixExceptionFlags           ExceptionFlags_(const char* opt) ; 
    static const char*                   ExceptionFlags__(OptixExceptionFlags excFlag) ; 
    static unsigned                      ExceptionFlags(const char* options); 
    static std::string                   Desc_ExceptionFlags( unsigned flags ); 
    static constexpr const char* OPTIX_EXCEPTION_FLAG_NONE_           = "OPTIX_EXCEPTION_FLAG_NONE" ; 
    static constexpr const char* OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW_ = "OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW" ; 
    static constexpr const char* OPTIX_EXCEPTION_FLAG_TRACE_DEPTH_    = "OPTIX_EXCEPTION_FLAG_TRACE_DEPTH" ; 
    static constexpr const char* OPTIX_EXCEPTION_FLAG_USER_           = "OPTIX_EXCEPTION_FLAG_USER_" ; 
    static constexpr const char* OPTIX_EXCEPTION_FLAG_DEBUG_          = "OPTIX_EXCEPTION_FLAG_DEBUG" ; 

};




/**
SOPTIX_OPT::DebugLevel
------------------------

https://forums.developer.nvidia.com/t/gpu-program-optimization-questions/195238/2

droettger, Nov 2021::

    The new OPTIX_COMPILE_DEBUG_LEVEL_MODERATE is documented to have an impact on
    performance.  You should use OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL which keeps only
    the line information for profiling and OPTIX_COMPILE_DEBUG_LEVEL_NONE to remove
    even that.  Never profile compute kernels build as debug! That will completely
    change the code structure and does not represent the fully optimized code.


See optix7-;optix7-types

**/


inline OptixCompileDebugLevel SOPTIX_OPT::DebugLevel(const char* option)  // static
{
    OptixCompileDebugLevel level = OPTIX_COMPILE_DEBUG_LEVEL_NONE ; 
#if OPTIX_VERSION == 70000
    if(     strcmp(option, "NONE") == 0 )     level = OPTIX_COMPILE_DEBUG_LEVEL_NONE ; 
    else if(strcmp(option, "LINEINFO") == 0 ) level = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO ; 
    else if(strcmp(option, "FULL") == 0 )     level = OPTIX_COMPILE_DEBUG_LEVEL_FULL ; 
    else if(strcmp(option, "DEFAULT") == 0 )  level = OPTIX_COMPILE_DEBUG_LEVEL_NONE ; 
#elif OPTIX_VERSION >= 70500
    if(     strcmp(option, "DEFAULT") == 0 )  level = OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT ; 
    else if(strcmp(option, "NONE") == 0 )     level = OPTIX_COMPILE_DEBUG_LEVEL_NONE ; 
    else if(strcmp(option, "MINIMAL") == 0 )  level = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL ; 
    else if(strcmp(option, "MODERATE") == 0 ) level = OPTIX_COMPILE_DEBUG_LEVEL_MODERATE ; 
    else if(strcmp(option, "FULL") == 0 )     level = OPTIX_COMPILE_DEBUG_LEVEL_FULL ; 
#else
    std::cerr 
        << " NOT RECOGNIZED " << " option " << option  << " level " << level  
        << " OPTIX_VERSION " << OPTIX_VERSION 
        << std::endl 
        ; 
    assert(0);   
#endif
    if(VERBOSE) std::cout 
         << "SOPTIX_OPT::DebugLevel"
         << " option " << option 
         << " level " << level 
         << " OPTIX_VERSION " << OPTIX_VERSION  
         << std::endl
         ;  
    return level ; 
}

inline const char * SOPTIX_OPT::DebugLevel_( OptixCompileDebugLevel debugLevel )
{
    const char* s = nullptr ; 
    switch(debugLevel)
    {  
#if OPTIX_VERSION == 70000
        case OPTIX_COMPILE_DEBUG_LEVEL_NONE:     s = OPTIX_COMPILE_DEBUG_LEVEL_NONE_     ; break ; 
        case OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO: s = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO_ ; break ;
        case OPTIX_COMPILE_DEBUG_LEVEL_FULL:     s = OPTIX_COMPILE_DEBUG_LEVEL_FULL_     ; break ;
#elif OPTIX_VERSION >= 70500
        case OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT:  s = OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT_  ; break ; 
        case OPTIX_COMPILE_DEBUG_LEVEL_NONE:     s = OPTIX_COMPILE_DEBUG_LEVEL_NONE_     ; break ; 
        case OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL:  s = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL_  ; break ;
        case OPTIX_COMPILE_DEBUG_LEVEL_MODERATE: s = OPTIX_COMPILE_DEBUG_LEVEL_MODERATE_ ; break ;
        case OPTIX_COMPILE_DEBUG_LEVEL_FULL:     s = OPTIX_COMPILE_DEBUG_LEVEL_FULL_     ; break ;
#endif
    }

    if( s == nullptr ) std::cerr
        << "SOPTIX_OPT::DebugLevel_"
        << " debugLevel " << debugLevel 
        << " IS NOT RECOGNIZED  "
        << " OPTIX_VERSION " << OPTIX_VERSION 
        << std::endl 
        ; 
    return s ;    
}

inline OptixCompileOptimizationLevel SOPTIX_OPT::OptimizationLevel(const char* option) // static 
{
    OptixCompileOptimizationLevel level = OPTIX_COMPILE_OPTIMIZATION_DEFAULT ; 
    if(      strcmp(option, "LEVEL_0") == 0 )  level = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0  ; 
    else if( strcmp(option, "LEVEL_1") == 0 )  level = OPTIX_COMPILE_OPTIMIZATION_LEVEL_1  ; 
    else if( strcmp(option, "LEVEL_2") == 0 )  level = OPTIX_COMPILE_OPTIMIZATION_LEVEL_2  ; 
    else if( strcmp(option, "LEVEL_3") == 0 )  level = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3  ; 
    else if( strcmp(option, "DEFAULT") == 0 )  level = OPTIX_COMPILE_OPTIMIZATION_DEFAULT  ; 
    else 
    {
        std::cerr 
            << "SOPTIX_OPT::OptimizationLevel "
            << " option " << option 
            << " IS NOT RECOGNIZED  "
            << " level " << level 
            << " OPTIX_VERSION " << OPTIX_VERSION 
            << std::endl 
            ; 
        assert(0) ; 
    }
 
    if(VERBOSE) std::cout 
        << "SOPTIX_OPT::OptimizationLevel " 
        << " option " << option 
        << " level " << level 
        << std::endl 
        ;  
    return level ; 
}

inline const char* SOPTIX_OPT::OptimizationLevel_( OptixCompileOptimizationLevel optLevel )
{
    const char* s = nullptr ; 
    switch(optLevel)
    {
        case OPTIX_COMPILE_OPTIMIZATION_LEVEL_0: s = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0_ ; break ; 
        case OPTIX_COMPILE_OPTIMIZATION_LEVEL_1: s = OPTIX_COMPILE_OPTIMIZATION_LEVEL_1_ ; break ; 
        case OPTIX_COMPILE_OPTIMIZATION_LEVEL_2: s = OPTIX_COMPILE_OPTIMIZATION_LEVEL_2_ ; break ; 
        case OPTIX_COMPILE_OPTIMIZATION_LEVEL_3: s = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3_ ; break ; 
        case OPTIX_COMPILE_OPTIMIZATION_DEFAULT: s = OPTIX_COMPILE_OPTIMIZATION_DEFAULT_ ; break ; 
        default:                                                                         ; break ; 
    }

    if( s == nullptr ) 
    {
        std::cerr
            << "SOPTIX_OPT::OptimizationLevel_"
            << " optLevel " << optLevel 
            << " IS NOT RECOGNIZED  "
            << " OPTIX_VERSION " << OPTIX_VERSION 
            << std::endl 
            ; 
        assert(0) ; 
    }
    return s ; 
} 

inline OptixExceptionFlags SOPTIX_OPT::ExceptionFlags_(const char* opt)
{
    OptixExceptionFlags flag = OPTIX_EXCEPTION_FLAG_NONE ; 
    if(      strcmp(opt, "NONE") == 0 )          flag = OPTIX_EXCEPTION_FLAG_NONE ;  
    else if( strcmp(opt, "STACK_OVERFLOW") == 0) flag = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW ; 
    else if( strcmp(opt, "TRACE_DEPTH") == 0)    flag = OPTIX_EXCEPTION_FLAG_TRACE_DEPTH  ; 
    else if( strcmp(opt, "USER") == 0)           flag = OPTIX_EXCEPTION_FLAG_USER  ; 
    else if( strcmp(opt, "DEBUG") == 0)          flag = OPTIX_EXCEPTION_FLAG_DEBUG  ; 
    else 
    {
        std::cerr 
            << "SOPTIX_OPT::ExceptionFlags_"
            << " opt " << opt
            << " IS NOT RECOGNIZED  "
            << " flag " << flag
            << " OPTIX_VERSION " << OPTIX_VERSION 
            << std::endl 
            ; 
        assert(0) ; 
    }
    return flag ; 
}

inline const char* SOPTIX_OPT::ExceptionFlags__(OptixExceptionFlags excFlag)
{
    const char* s = nullptr ; 
    switch(excFlag)
    {
        case OPTIX_EXCEPTION_FLAG_NONE:           s = OPTIX_EXCEPTION_FLAG_NONE_            ; break ; 
        case OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW: s = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW_  ; break ;
        case OPTIX_EXCEPTION_FLAG_TRACE_DEPTH:    s = OPTIX_EXCEPTION_FLAG_TRACE_DEPTH_     ; break ; 
        case OPTIX_EXCEPTION_FLAG_USER:           s = OPTIX_EXCEPTION_FLAG_USER_            ; break ; 
        case OPTIX_EXCEPTION_FLAG_DEBUG:          s = OPTIX_EXCEPTION_FLAG_DEBUG_           ; break ;      
    }
    return s ; 
}

inline unsigned SOPTIX_OPT::ExceptionFlags(const char* options)
{
    std::vector<std::string> opts ; 
    sstr::Split( options, '|', opts );  

    unsigned exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE ; 
    for(unsigned i=0 ; i < opts.size() ; i++)
    {
        const std::string& opt = opts[i] ; 
        exceptionFlags |= ExceptionFlags_(opt.c_str()); 
    }
    if(VERBOSE) std::cerr 
        << "SOPTIX_OPT::ExceptionFlags"  
        << " options " << options 
        << " exceptionFlags " << exceptionFlags 
        << std::endl 
        ; 
    return exceptionFlags ;  
}

inline std::string SOPTIX_OPT::Desc_ExceptionFlags( unsigned flags )
{
    std::stringstream ss ; 
    if( flags & OPTIX_EXCEPTION_FLAG_NONE )           ss << OPTIX_EXCEPTION_FLAG_NONE_ << " " ; 
    if( flags & OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW ) ss << OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW_ << " " ; 
    if( flags & OPTIX_EXCEPTION_FLAG_TRACE_DEPTH )    ss << OPTIX_EXCEPTION_FLAG_TRACE_DEPTH_ << " " ; 
    if( flags & OPTIX_EXCEPTION_FLAG_USER )           ss << OPTIX_EXCEPTION_FLAG_USER_ << " " ; 
    if( flags & OPTIX_EXCEPTION_FLAG_DEBUG )          ss << OPTIX_EXCEPTION_FLAG_DEBUG_ << " " ; 
    std::string str = ss.str() ; 
    return str ; 
}


