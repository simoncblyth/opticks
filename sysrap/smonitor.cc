/**
smonitor.cc : monitor GPU process memory usage saving into .npy array 
========================================================================

**/


#include <cstdio>
#include <chrono>
#include <thread>
#include <string>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <unistd.h>
#include <iostream>
#include <cstdlib>
#include <csignal>
#include <vector>

#include <nvml.h>
#include "NVML_CHECK.h"
#include "NPX.h"

struct smon
{
    uint64_t stamp ;   
    uint64_t device ; 
    uint64_t free ; 
    uint64_t total ; 

    uint64_t used ; 
    uint64_t pid ; 
    uint64_t usedGpuMemory ;
    uint64_t proc_count ; 
}; 

struct smonitor
{
    static constexpr const bool VERBOSE = false ; 
    static smonitor* INSTANCE ; 
    static uint64_t Stamp(); 
    static void signal_callback_handler(int signum); 

    unsigned  device_count ;
    std::vector<smon> mon ; 

    smonitor(); 

    void runloop(); 
    void check(); 
    void save(); 
};


smonitor* smonitor::INSTANCE = nullptr ; 

inline uint64_t smonitor::Stamp()
{
    using Clock = std::chrono::system_clock;
    using Unit  = std::chrono::microseconds ;
    std::chrono::time_point<Clock> t0 = Clock::now();
    return std::chrono::duration_cast<Unit>(t0.time_since_epoch()).count() ;   
}

inline smonitor::smonitor()
{
    INSTANCE = this ; 
    NVML_CHECK( nvmlInit() );
    NVML_CHECK( nvmlDeviceGetCount(&device_count) ); 
    printf("device_count: %u \n", device_count );

    signal(SIGINT, signal_callback_handler);
}

void smonitor::signal_callback_handler(int signum) 
{
    std::cout << "Caught signal " << signum << std::endl;
    smonitor::INSTANCE->save();  
    exit(signum);
}

inline void smonitor::save()
{
    std::cout << "smonitor::save mon.size " << mon.size() << std::endl ; 

    NP* a = NPX::ArrayFromVec<uint64_t, smon>(mon) ;  
    a->save("smonitor.npy"); 
}

inline void smonitor::runloop()
{
    while(true)
    {
        check(); 
        sleep(1);
    }
}

inline void smonitor::check()
{
    for(unsigned index=0 ; index < device_count ; index++) 
    {
        nvmlDevice_t device ; 
        NVML_CHECK( nvmlDeviceGetHandleByIndex_v2( index, &device )); 
         
        //const int maxchar = 32 ;           
        //char name[maxchar] ; 
        //NVML_CHECK( nvmlDeviceGetName(device, name, maxchar ) ); 
        //printf("device %d name %s \n" , index, name );  

        nvmlMemory_t memory ; 
        NVML_CHECK( nvmlDeviceGetMemoryInfo(device, &memory ) ); 

        if(VERBOSE) printf(" memory.free %llu memory.total %llu memory.used %llu \n", 
                             memory.free,     memory.total,     memory.used ); 

        unsigned proc_count(0) ; 
        nvmlReturn_t rc = nvmlDeviceGetComputeRunningProcesses_v3(device, &proc_count, nullptr );

        if( rc == NVML_ERROR_INSUFFICIENT_SIZE )  // documented that get this 
        {
            if(VERBOSE) printf("proc_count %d \n", proc_count ); 

            unsigned proc_alloc = proc_count + 3 ; 
            nvmlProcessInfo_t* procs = new nvmlProcessInfo_t[proc_alloc] ; 

            NVML_CHECK( nvmlDeviceGetComputeRunningProcesses_v3(device, &proc_alloc, procs ) );

            for(unsigned p=0 ; p <  proc_alloc ; p++)
            {
                int num_mon = mon.size(); 
                const nvmlProcessInfo_t& proc = procs[p] ; 

                //printf(" proc.computeInstanceId  %u  proc.gpuInstanceId %u  proc.pid %u  proc.usedGpuMemory %llu \n", 
                //         proc.computeInstanceId, proc.gpuInstanceId, proc.pid, proc.usedGpuMemory ); 

                printf(" num_mon %5d proc_count %2d proc.pid %u  proc.usedGpuMemory %llu [%10.3f GB] \n", 
                        num_mon, proc_count, proc.pid, proc.usedGpuMemory, float(proc.usedGpuMemory)/1e9 ); 

                smon m ; 

                m.stamp = Stamp(); 
                m.device = index ; 
                m.free = memory.free ; 
                m.total = memory.total ; 

                m.used = memory.used ;
                m.pid = proc.pid ; 
                m.usedGpuMemory = proc.usedGpuMemory ; 
                m.proc_count = proc_count ; 

                mon.push_back(m); 
            }
        }
    }
}

int main()
{
    smonitor sm ; 
    sm.runloop(); 
    return 0 ; 
}


