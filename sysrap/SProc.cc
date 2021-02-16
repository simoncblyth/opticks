/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

#include "SProc.hh"
#include "PLOG.hh"


#ifdef _MSC_VER
float SProc::VirtualMemoryUsageMB()
{
    return 0 ; 
}

#elif defined(__APPLE__)

#include<mach/mach.h>

/**

https://developer.apple.com/forums/thread/105088

**/


float SProc::VirtualMemoryUsageKB()
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

    LOG(error) << mach_error_string(kerr)   ; 

    return 0 ;   
}



float SProc::ResidentSetSizeKB()
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

    LOG(error) << mach_error_string(kerr)   ; 

    return 0 ;   
}






#else
    
// https://stackoverflow.com/questions/63166/how-to-determine-cpu-and-memory-consumption-from-inside-a-process

#include "stdlib.h"
#include "stdio.h"
#include "string.h"

/**
parseLine
-----------

Expects a line the below form with digits and ending in " Kb"::

   VmSize:	  108092 kB

**/

int parseLine(char* line){
    int i = strlen(line);
    const char* p = line;
    while (*p <'0' || *p > '9') p++;
    line[i-3] = '\0';
    i = atoi(p);
    return i;
}

float SProc::VirtualMemoryUsageKB()
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


float SProc::ResidentSetSizeKB()
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

float SProc::VirtualMemoryUsageMB()
{
    float result = VirtualMemoryUsageKB() ; 
    return result/1000.f ;   
}

float SProc::ResidentSetSizeMB()
{
    float result = ResidentSetSizeKB() ; 
    return result/1000.f ;   
}






/**
SProc::ExecutablePath
-----------------------

* https://stackoverflow.com/questions/799679/programmatically-retrieving-the-absolute-path-of-an-os-x-command-line-app/1024933#1024933

**/


#ifdef _MSC_VER
const char* SProc::ExecutablePath(bool basename)
{
    return NULL ; 
}
#elif defined(__APPLE__)

#include <mach-o/dyld.h>

const char* SProc::ExecutablePath(bool basename)
{
    char buf[PATH_MAX];
    uint32_t size = sizeof(buf);
    bool ok = _NSGetExecutablePath(buf, &size) == 0 ; 

    if(!ok)
       LOG(fatal) 
           << "_NSGetExecutablePath FAIL " 
           << " size " << size 
           << " buf " << buf 
           ;

    assert(ok); 
    const char* s = basename ? strrchr(buf, '/') : NULL ;  
    return s ? strdup(s+1) : strdup(buf) ; 
}
#else


#include <unistd.h>
#include <limits.h>

const char* SProc::ExecutablePath(bool basename)
{
    char buf[PATH_MAX];
    ssize_t len = ::readlink("/proc/self/exe", buf, sizeof(buf)-1);
    if (len != -1) buf[len] = '\0';

    const char* s = basename ? strrchr(buf, '/') : NULL ;  
    return s ? strdup(s+1) : strdup(buf) ; 
}

#endif


const char* SProc::ExecutableName()
{
    bool basename = true ; 
    return ExecutablePath(basename); 
}










