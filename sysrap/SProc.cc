#include "SProc.hh"
#include "PLOG.hh"


#ifdef _MSC_VER
float SProc::VirtualMemoryUsageMB()
{
    return 0 ; 
}

#elif defined(__APPLE__)

#include<mach/mach.h>

float SProc::VirtualMemoryUsageMB()
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
        unsigned long long MB = 1000000 ; 
        float usage = float(vsize/MB) ; 
        return usage  ;
    }

    LOG(error) << mach_error_string(kerr)   ; 

    return 0 ;   
}


#else
float SProc::VirtualMemoryUsageMB()
{
    return 0 ; 
}
#endif



