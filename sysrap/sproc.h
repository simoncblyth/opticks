#pragma once
/**
sproc.h
===========

macOS implementation of VirtualMemoryUsageMB of a process.

**/

#include <cstddef>
#include <cassert>
#include <iostream>
#include <cstring>
#include <cstdlib>

#include "sstr.h"


struct sproc 
{
    static int parseLine(char* line); 
    static float VirtualMemoryUsageMB();
    static float VirtualMemoryUsageKB();
    static float ResidentSetSizeMB();
    static float ResidentSetSizeKB();

    static char* ExecutablePath(bool basename=false); 
    static char* _ExecutableName(); 
    static bool StartsWith( const char* s, const char* q); 
    static char* ExecutableName(); 
};



/**
sproc::parseLine
-----------------

Expects a line of the below form with digits and ending in " Kb"::

   VmSize:	  108092 kB

**/

inline int sproc::parseLine(char* line){
    int i = strlen(line);
    const char* p = line;
    while (*p <'0' || *p > '9') p++; // advance until first digit 
    line[i-3] = '\0';  // chop off the " kB"
    i = atoi(p);
    return i;
}


#ifdef _MSC_VER
inline float sproc::VirtualMemoryUsageMB()
{
    return 0 ; 
}

#elif defined(__APPLE__)

#include<mach/mach.h>

/**

https://developer.apple.com/forums/thread/105088

**/


inline float sproc::VirtualMemoryUsageKB()
{
    struct mach_task_basic_info info;
    mach_msg_type_number_t size = MACH_TASK_BASIC_INFO_COUNT;
    kern_return_t kerr = task_info(mach_task_self(),
                                   MACH_TASK_BASIC_INFO,
                                   (task_info_t)&info,
                                   &size);

    if( kerr == KERN_SUCCESS ) 
    {
        vm_size_t vsize_ = info.virtual_size  ;  
        unsigned long long vsize(vsize_); 
        unsigned long long KB = 1000 ; 
        float usage = float(vsize/KB) ; 
        return usage  ;
    }

    std::cerr  << mach_error_string(kerr) << std::endl   ; 

    return 0 ;   
}


inline float sproc::ResidentSetSizeKB()
{
    struct mach_task_basic_info info;
    mach_msg_type_number_t size = MACH_TASK_BASIC_INFO_COUNT;
    kern_return_t kerr = task_info(mach_task_self(),
                                   MACH_TASK_BASIC_INFO,
                                   (task_info_t)&info,
                                   &size);

    if( kerr == KERN_SUCCESS ) 
    {
        vm_size_t rsize_ = info.resident_size  ;  
        unsigned long long rsize(rsize_); 
        unsigned long long KB = 1000 ; 
        float usage = float(rsize/KB) ; 
        return usage  ;
    }

    std::cerr << mach_error_string(kerr) << std::endl  ; 

    return 0 ;   
}

#else
    
// https://stackoverflow.com/questions/63166/how-to-determine-cpu-and-memory-consumption-from-inside-a-process

#include "stdlib.h"
#include "stdio.h"
#include "string.h"


inline float sproc::VirtualMemoryUsageKB()
{
    FILE* file = fopen("/proc/self/status", "r");
    float result = 0.f ;
    char line[128];

    while (fgets(line, 128, file) != NULL){
        if (strncmp(line, "VmSize:", 7) == 0){
            result = parseLine(line);   // value in Kb 
            break;
        }
    }
    fclose(file);
    return result;
}


inline float sproc::ResidentSetSizeKB()
{
    FILE* file = fopen("/proc/self/status", "r");
    float result = 0.f ;
    char line[128];

    while (fgets(line, 128, file) != NULL){
        if (strncmp(line, "VmRSS:", 6) == 0){
            result = parseLine(line);   // value in Kb 
            break;
        }
    }
    fclose(file);
    return result;
}


#endif

inline float sproc::VirtualMemoryUsageMB()
{
    float result = VirtualMemoryUsageKB() ; 
    return result/1000.f ;   
}

inline float sproc::ResidentSetSizeMB()
{
    float result = ResidentSetSizeKB() ; 
    return result/1000.f ;   
}






/**
sproc::ExecutablePath
-----------------------

* https://stackoverflow.com/questions/799679/programmatically-retrieving-the-absolute-path-of-an-os-x-command-line-app/1024933#1024933

**/


#ifdef _MSC_VER
inline char* sproc::ExecutablePath(bool basename)
{
    return NULL ; 
}
#elif defined(__APPLE__)

#include <mach-o/dyld.h>

inline char* sproc::ExecutablePath(bool basename)
{
    char buf[PATH_MAX];
    uint32_t size = sizeof(buf);
    bool ok = _NSGetExecutablePath(buf, &size) == 0 ; 

    if(!ok) std::cerr 
        << "_NSGetExecutablePath FAIL " 
        << " size " << size 
        << " buf " << buf 
        << std::endl
        ;

    assert(ok); 
    char* s = basename ? strrchr(buf, '/') : NULL ;  
    return s ? strdup(s+1) : strdup(buf) ; 
}
#else


#include <unistd.h>
#include <limits.h>

inline char* sproc::ExecutablePath(bool basename)
{
    char buf[PATH_MAX];
    ssize_t len = ::readlink("/proc/self/exe", buf, sizeof(buf)-1);
    if (len != -1) buf[len] = '\0';

    char* s = basename ? strrchr(buf, '/') : NULL ;  
    return s ? strdup(s+1) : strdup(buf) ; 
}

#endif


inline char* sproc::_ExecutableName()
{
    bool basename = true ; 
    return ExecutablePath(basename); 
}


inline bool sproc::StartsWith( const char* s, const char* q) 
{
    return s && q && strlen(q) <= strlen(s) && strncmp(s, q, strlen(q)) == 0 ; 
}

/**
sproc::ExecutableName
----------------------

In embedded running with python "main" the 
initial executable name is eg "python3.9".
That can be overridden with envvar OPTICKS_SCRIPT 

**/

inline char* sproc::ExecutableName()
{  
    char* exe0 = sproc::_ExecutableName() ; 
    bool is_python = sproc::StartsWith(exe0, "python") ;  
    char* script = getenv("OPTICKS_SCRIPT"); 
    char* exe = ( is_python && script ) ? script : exe0 ; 
    return exe ; 
}



