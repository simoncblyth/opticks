#pragma once
/**
sbuild.h
=========

~/o/sysrap/tests/sbuild_test.sh 

Curious the simple script build has no problem
with variable names the same as the macros, but the 
CMake build gives error::

    <command-line>: error: expected unqualified-id before numeric constant
    /home/blyth/opticks/sysrap/tests/../sbuild.h:39:33: note: in expansion of macro ‘WITH_CHILD’
       39 |     static constexpr const bool WITH_CHILD = true ;
          |                                 ^~~~~~~~~~

The difference is not from c++11 vs c++17, must be some other option with the CMake build. 
Due to this prefixed the names with underscore. 

**/

#include <string>
#include <sstream>

#include "srng.h"

struct sbuild
{
    static constexpr const char* Debug   = "Debug" ; 
    static constexpr const char* Release = "Release" ; 


#if defined(CONFIG_Debug)
    static constexpr const char* BUILD_TYPE = "Debug" ; 
#elif defined(CONFIG_Release)
    static constexpr const char* BUILD_TYPE = "Release" ; 
#elif defined(CONFIG_RelWithDebInfo)
    static constexpr const char* BUILD_TYPE = "RelWithDebInfo" ; 
#elif defined(CONFIG_MinSizeRel)
    static constexpr const char* BUILD_TYPE = "MinSizeRel" ;
#else
    static constexpr const char* BUILD_TYPE = "sbuild-ERROR-NO-CONFIG-MACROS" ;
#endif

#if defined(PRODUCTION)
    static constexpr const bool _PRODUCTION = true ; 
#else
    static constexpr const bool _PRODUCTION = false ; 
#endif

#if defined(WITH_CHILD)
    static constexpr const bool _WITH_CHILD = true ; 
#else
    static constexpr const bool _WITH_CHILD = false ; 
#endif

#if defined(WITH_CUSTOM4)
    static constexpr const bool _WITH_CUSTOM4 = true ; 
#else
    static constexpr const bool _WITH_CUSTOM4 = false ; 
#endif

#if defined(PLOG_LOCAL)
    static constexpr const bool _PLOG_LOCAL = true ; 
#else
    static constexpr const bool _PLOG_LOCAL = false ; 
#endif

#if defined(DEBUG_PIDX)
    static constexpr const bool _DEBUG_PIDX = true ; 
#else
    static constexpr const bool _DEBUG_PIDX = false ; 
#endif

#if defined(DEBUG_TAG)
    static constexpr const bool _DEBUG_TAG = true ; 
#else
    static constexpr const bool _DEBUG_TAG = false ; 
#endif


#if defined(RNG_XORWOW)
    static constexpr const bool _RNG_XORWOW = true ; 
#else
    static constexpr const bool _RNG_XORWOW = false ; 
#endif

#if defined(RNG_PHILOX)
    static constexpr const bool _RNG_PHILOX = true ; 
#else
    static constexpr const bool _RNG_PHILOX = false ; 
#endif

#if defined(RNG_PHILITEOX)
    static constexpr const bool _RNG_PHILITEOX = true ; 
#else
    static constexpr const bool _RNG_PHILITEOX = false ; 
#endif


    static const char* RNGName(){ return srng<RNG>::NAME ; }
    static bool IsDebug(){   return strcmp(BUILD_TYPE, Debug) == 0 ; }
    static bool IsRelease(){ return strcmp(BUILD_TYPE, Release) == 0 ; }

    static std::string Desc(); 
    static std::string ContextString(); 

    static bool BuildTypeMatches(const char* arg); 
    static bool RNGMatches(const char* arg); 
    static bool Matches(const char* arg); 
}; 


inline std::string sbuild::Desc() // static
{
    std::stringstream ss ; 
    ss 
       << "[sbuild::Desc\n"
       << " sbuild::ContextString() : [" << ContextString() << "]\n" 
       << " sbuild::BUILD_TYPE      : [" << BUILD_TYPE << "]\n" 
       << " sbuild::IsDebug()       :  "  << ( IsDebug() ? "YES" : "NO " ) << "\n"  
       << " sbuild::IsRelease()     :  "  << ( IsRelease() ? "YES" : "NO " ) << "\n"  
       << " sbuild::BuildTypeMatches(\"Cheese\") : " << sbuild::BuildTypeMatches("Cheese") << "\n"
       << " sbuild::BuildTypeMatches(\"Cheese Debug \") : " << sbuild::BuildTypeMatches("Cheese Debug") << "\n"
       << " sbuild::BuildTypeMatches(\"Cheese Release \") : " << sbuild::BuildTypeMatches("Cheese Release ") << "\n"
       << " srng<RNG>::NAME      : [" << srng<RNG>::NAME << "]\n"        
       << " sbuild::RNGName()    : [" << RNGName() << "]\n"        
       << " sbuild::RNGMatches(\"Cheese\") : " << sbuild::RNGMatches("Cheese") << "\n" 
       << " sbuild::RNGMatches(\"Cheese XORWOW\") : " << sbuild::RNGMatches("Cheese XORWOW ") << "\n" 
       << " sbuild::RNGMatches(\"Cheese Philox\") : " << sbuild::RNGMatches("Cheese Philox ") << "\n" 
       << " _PRODUCTION          : " << ( _PRODUCTION ? "YES" :  "NO " ) << "\n"
       << " _WITH_CHILD          : " << ( _WITH_CHILD ? "YES" :  "NO " ) << "\n"
       << " _WITH_CUSTOM4        : " << ( _WITH_CUSTOM4 ? "YES" :  "NO " ) << "\n"
       << " _PLOG_LOCAL          : " << ( _PLOG_LOCAL ? "YES" :  "NO " ) << "\n"
       << " _DEBUG_PIDX          : " << ( _DEBUG_PIDX ? "YES" :  "NO " ) << "\n"
       << " _DEBUG_TAG           : " << ( _DEBUG_TAG ? "YES" :  "NO " ) << "\n"
       << " _RNG_XORWOW          : " << ( _RNG_XORWOW ? "YES" :  "NO " ) << "\n"
       << " _RNG_PHILOX          : " << ( _RNG_PHILOX ? "YES" :  "NO " ) << "\n"
       << " _RNG_PHILITEOX       : " << ( _RNG_PHILITEOX ? "YES" :  "NO " ) << "\n"
       << " sbuild::Matches(\"ALL99_Release_XORWOW\") : " << sbuild::Matches("ALL99_Release_XORWOW") << "\n"
       << " sbuild::Matches(\"ALL99_Release_Philox\") : " << sbuild::Matches("ALL99_Release_Philox") << "\n"
       << " sbuild::Matches(\"ALL99_Debug_XORWOW\") : " << sbuild::Matches("ALL99_Debug_XORWOW") << "\n"
       << " sbuild::Matches(\"ALL99_Debug_Philox\") : " << sbuild::Matches("ALL99_Debug_Philox") << "\n"
       << "]sbuild::Desc\n"
       ;
    std::string str = ss.str() ; 
    return str ; 
}

inline std::string sbuild::ContextString() // static
{
    std::stringstream ss ; 
    ss << BUILD_TYPE << "_" << RNGName() ;  
    std::string str = ss.str() ; 
    return str ; 
}


/**
sbuild::BuildTypeMatches
--------------------------

When the argument string contains "Debug" or "Release"
then then sbuild::BUILD_TYPE is required to match 
that to return true. 

**/

inline bool sbuild::BuildTypeMatches(const char* arg)
{
    if(arg == nullptr) return false ; 
    int match = 0 ; 
    if(strstr(arg, Debug)   && IsDebug())   match += 1 ; 
    if(strstr(arg, Release) && IsRelease()) match += 1 ; 
    return match == 1 ; 
}

inline bool sbuild::RNGMatches(const char* arg)
{
    return srng_Matches<RNG>(arg); 
}

/**
sbuild::Matches
-----------------

* Release/Debug 
* XORWOW/Philox

Returns true when the argument contains the choices of the above 
strings that match the build settings. 
To make them match it is necessary to rebuild with different
CMake level macro settings.

This is used to ensure that descriptive TEST strings
from envvars set in runner scripts correspond to the 
build currently being used. Those TEST strings are used
to control output directories of saved events. 

**/

inline bool sbuild::Matches(const char* arg)
{
    return RNGMatches(arg) && BuildTypeMatches(arg) ; 
}



