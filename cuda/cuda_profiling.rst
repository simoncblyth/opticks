CUDA Profiling
===============

* http://docs.nvidia.com/cuda/profiler-users-guide/

nvprof
---------

* http://devblogs.nvidia.com/parallelforall/cuda-pro-tip-nvprof-your-handy-universal-gpu-profiler/

    nvprof knows how to profile CUDA kernels running on NVIDIA GPUs, no matter what
    language they are written in (as long as they are launched using the CUDA
    runtime API or driver API).

* http://devblogs.nvidia.com/parallelforall/pro-tip-clean-up-after-yourself-ensure-correct-profiling/

    Therefore, you should clean up your application’s CUDA objects properly to make
    sure that the profiler is able to store all gathered data. This means not only
    freeing memory allocated on the GPU, but also resetting the device Context.

    If your application uses the CUDA Driver API, call cuProfilerStop() on each
    context to flush the profiling buffers before destroying the context with
    cuCtxDestroy().



cuda clock
----------

* http://stackoverflow.com/questions/11217117/equivalent-of-usleep-in-cuda-kernel

gpu burn
---------

* http://wili.cc/blog/gpu-burn.html

how many registers
-------------------

::

    Also, how many registers is your kernel using?? (pass --ptxas-options=-v
    argument to nvcc) If you can only launch 16 threads per block, the GPU will be
    idle most of the time.


cuda profile
-------------

From a headless simplecamera.py render run::

    1285 method=[ memcpyHtoD ] gputime=[ 1.312 ] cputime=[ 2.020 ]
    1286 method=[ memcpyHtoD ] gputime=[ 1.312 ] cputime=[ 11.104 ]
    1287 method=[ memcpyHtoD ] gputime=[ 1.280 ] cputime=[ 1.972 ]
    1288 method=[ memcpyHtoD ] gputime=[ 1.280 ] cputime=[ 2.006 ]
    1289 method=[ memcpyHtoD ] gputime=[ 1.280 ] cputime=[ 2.006 ]
    1290 method=[ memcpyHtoD ] gputime=[ 1.312 ] cputime=[ 1.996 ]
    1291 method=[ memcpyHtoD ] gputime=[ 1.312 ] cputime=[ 2.012 ]
    1292 method=[ memcpyHtoD ] gputime=[ 1.280 ] cputime=[ 2.022 ]
    1293 method=[ memcpyHtoD ] gputime=[ 1.280 ] cputime=[ 10.942 ]
    1294 method=[ memcpyHtoD ] gputime=[ 1.312 ] cputime=[ 4.039 ]
    1295 method=[ memcpyHtoD ] gputime=[ 1.312 ] cputime=[ 2.034 ]
    1296 method=[ memcpyHtoD ] gputime=[ 1.312 ] cputime=[ 1.891 ]
    1297 method=[ memcpyHtoD ] gputime=[ 1.312 ] cputime=[ 1.912 ]
    1298 method=[ memcpyHtoD ] gputime=[ 1794.976 ] cputime=[ 1993.471 ]
    1299 method=[ memcpyHtoD ] gputime=[ 1617.952 ] cputime=[ 1481.204 ]
    1300 method=[ memcpyHtoD ] gputime=[ 1601.280 ] cputime=[ 1472.250 ]
    1301 method=[ memcpyHtoD ] gputime=[ 7432.672 ] cputime=[ 7370.140 ]
    1302 method=[ memcpyHtoD ] gputime=[ 4602.432 ] cputime=[ 4620.065 ]
    1303 method=[ memcpyHtoD ] gputime=[ 2335.680 ] cputime=[ 2351.582 ]
    1304 method=[ memcpyHtoD ] gputime=[ 1.664 ] cputime=[ 5.372 ]
    1305 method=[ memcpyHtoD ] gputime=[ 1.312 ] cputime=[ 2.315 ]
    1306 method=[ memcpyHtoD ] gputime=[ 1.280 ] cputime=[ 2.037 ]
    1307 method=[ memcpyHtoD ] gputime=[ 1.280 ] cputime=[ 1.973 ]
    1308 method=[ memcpyHtoD ] gputime=[ 1.280 ] cputime=[ 2.185 ]
    1309 method=[ memcpyHtoD ] gputime=[ 1.280 ] cputime=[ 2.113 ]
    1310 method=[ memcpyHtoD ] gputime=[ 1.312 ] cputime=[ 2.008 ]
    1311 method=[ memcpyHtoD ] gputime=[ 1.312 ] cputime=[ 2.010 ]
    1312 method=[ memcpyHtoD ] gputime=[ 1.312 ] cputime=[ 2.372 ]
    1313 method=[ memcpyHtoD ] gputime=[ 1.312 ] cputime=[ 2.009 ]
    1314 method=[ memcpyHtoD ] gputime=[ 1.280 ] cputime=[ 1.959 ]
    1315 method=[ memcpyHtoD ] gputime=[ 612.832 ] cputime=[ 501.086 ]
    1316 method=[ memcpyHtoD ] gputime=[ 590.560 ] cputime=[ 449.675 ]
    1317 method=[ fill ] gputime=[ 24.544 ] cputime=[ 13.470 ] occupancy=[ 1.000 ]
    1318 method=[ fill ] gputime=[ 25.504 ] cputime=[ 7.263 ] occupancy=[ 1.000 ]
    1319 method=[ render ] gputime=[ 5259416.500 ] cputime=[ 234.175 ] occupancy=[ 0.500 ]
    1320 method=[ memcpyDtoH ] gputime=[ 194.016 ] cputime=[ 5260492.000 ]


cuda_profile_parse.py
----------------------

::

    (chroma_env)delta:chroma_camera blyth$ ./cuda_profile_parse.py cuda_profile_0.log
    WARNING:__main__:failed to parse : # CUDA_PROFILE_LOG_VERSION 2.0 
    WARNING:__main__:failed to parse : # CUDA_DEVICE 0 GeForce GT 750M 
    WARNING:__main__:failed to parse : # CUDA_CONTEXT 1 
    WARNING:__main__:failed to parse : method,gputime,cputime,occupancy 

    memcpyDtoH           : {'gputime': 201.504, 'cputime': 5260556.83} 
    write_size           : {'gputime': 6.208, 'cputime': 37.704, 'occupancy': 0.048} 
    fill                 : {'gputime': 50.048, 'cputime': 20.733, 'occupancy': 2.0} 
    render               : {'gputime': 5259416.5, 'cputime': 234.175, 'occupancy': 0.5} 
    memcpyHtoD           : {'gputime':   22289.11999999997, 'cputime': 23602.95499999999} 
    (chroma_env)delta:chroma_camera blyth$ 


#. memcpyDtoH consumes the same 'cputime' as render takes 'gputime' with the 
   vast majority of that at the last sample

::

    (chroma_env)delta:chroma_camera blyth$ tail -5 cuda_profile_0.log
    method=[ memcpyHtoD ] gputime=[ 590.560 ] cputime=[ 449.675 ] 
    method=[ fill ] gputime=[ 24.544 ] cputime=[ 13.470 ] occupancy=[ 1.000 ] 
    method=[ fill ] gputime=[ 25.504 ] cputime=[ 7.263 ] occupancy=[ 1.000 ] 
    method=[ render ] gputime=[ 5259416.500 ] cputime=[ 234.175 ] occupancy=[ 0.500 ] 
    method=[ memcpyDtoH ] gputime=[ 194.016 ] cputime=[ 5260492.000 ] 

method
-------

This is character string which gives the name of the GPU kernel or memory copy
method. In case of kernels the method name is the mangled name generated by the
compiler.


occupancy
---------

This column gives the multiprocessor occupancy which is the ratio of number of
active warps to the maximum number of warps supported on a multiprocessor of
the GPU. This is helpful in determining how effectively the GPU is kept busy.
This column is output only for GPU kernels and the column value is a single
precision floating point value in the range 0.0 to 1.0.

cputime
---------

For non-blocking methods the cputime is only the CPU or host side overhead to
launch the method. In this case:

walltime = cputime + gputime

For blocking methods cputime is the sum of gputime and CPU overhead. In this
case:

walltime = cputime

Note all kernel launches by default are non-blocking. But if any of the
profiler counters are enabled kernel launches are blocking. Also asynchronous
memory copy requests in different streams are non-blocking.

The column value is a single precision floating point value in microseconds.



gputime
--------


This column gives the execution time for the GPU kernel or memory copy method.
This value is calculated as (gpuendtimestamp - gpustarttimestamp)/1000.0. The
column value is a single precision floating point value in microseconds.



config
-------


The command line profiler is controlled using the following environment
variables:

COMPUTE_PROFILE: is set to either 1 or 0 (or unset) to enable or disable
profiling.

COMPUTE_PROFILE_LOG: is set to the desired file path for profiling output. In
case of multiple contexts you must add '%d' in the COMPUTE_PROFILE_LOG name.
This will generate separate profiler output files for each context - with '%d'
substituted by the context number. Contexts are numbered starting with zero. In
case of multiple processes you must add '%p' in the COMPUTE_PROFILE_LOG name.
This will generate separate profiler output files for each process - with '%p'
substituted by the process id. If there is no log path specified, the profiler
will log data to "cuda_profile_%d.log" in case of a CUDA context ('%d' is
substituted by the context number).

COMPUTE_PROFILE_CSV: is set to either 1 (set) or 0 (unset) to enable or disable
a comma separated version of the log output.

COMPUTE_PROFILE_CONFIG: is used to specify a config file for selecting
profiling options and performance counters.

Configuration details are covered in a subsequent section.

The following old environment variables used for the above functionalities are
still supported:

CUDA_PROFILE

CUDA_PROFILE_LOG

CUDA_PROFILE_CSV

CUDA_PROFILE_CONFIG



metrics
---------

::

    (chroma_env)delta:e blyth$ nvprof --query-metrics
    Available Metrics:
                                Name   Description
    Device 0 (GeForce GT 750M):
            l1_cache_global_hit_rate:  Hit rate in L1 cache for global loads
                   branch_efficiency:  Ratio of non-divergent branches to total branches
             l1_cache_local_hit_rate:  Hit rate in L1 cache for local loads and stores
                       sm_efficiency:  The percentage of time at least one warp is active on a multiprocessor
                                 ipc:  Instructions executed per cycle
                  achieved_occupancy:  Ratio of the average active warps per active cycle to the maximum number of warps supported on a multiprocessor
            gld_requested_throughput:  Requested global memory load throughput
            gst_requested_throughput:  Requested global memory store throughput
              sm_efficiency_instance:  The percentage of time at least one warp is active on a multiprocessor
                        ipc_instance:  Instructions executed per cycle
                inst_replay_overhead:  Average number of replays for each instruction executed
              shared_replay_overhead:  Average number of replays due to shared memory conflicts for each instruction executed
              global_replay_overhead:  Average number of replays due to local memory cache misses for each instruction executed
        global_cache_replay_overhead:  Average number of replays due to global memory cache misses for each instruction executed
                  tex_cache_hit_rate:  Texture cache hit rate
                tex_cache_throughput:  Texture cache throughput
                dram_read_throughput:  Device memory read throughput
               dram_write_throughput:  Device memory write throughput
                      gst_throughput:  Global memory store throughput
                      gld_throughput:  Global memory load throughput
               local_replay_overhead:  Average number of replays due to local memory accesses for each instruction executed
                   shared_efficiency:  Ratio of requested shared memory throughput to required shared memory throughput
                      gld_efficiency:  Ratio of requested global memory load throughput to required global memory load throughput
                      gst_efficiency:  Ratio of requested global memory store throughput to required global memory store throughput
                 l2_l1_read_hit_rate:  Hit rate at L2 cache for all read requests from L1 cache
            l2_texture_read_hit_rate:  Hit rate at L2 cache for all read requests from texture cache
               l2_l1_read_throughput:  Memory read throughput seen at L2 cache for read requests from L1 cache
          l2_texture_read_throughput:  Memory read throughput seen at L2 cache for read requests from the texture cache
               local_memory_overhead:  Ratio of local memory traffic to total memory traffic between the L1 and L2 caches
                          issued_ipc:  Instructions issued per cycle
                       inst_per_warp:  Average number of instructions executed by each warp
              issue_slot_utilization:  Percentage of issue slots that issued at least one instruction, averaged across all cycles
    local_load_transactions_per_request:  Average number of local memory load transactions performed for each local memory load
    local_store_transactions_per_request:  Average number of local memory store transactions performed for each local memory store
    shared_load_transactions_per_request:  Average number of shared memory load transactions performed for each shared memory load
    shared_store_transactions_per_request:  Average number of shared memory store transactions performed for each shared memory store
        gld_transactions_per_request:  Average number of global memory load transactions performed for each global memory load
        gst_transactions_per_request:  Average number of global memory store transactions performed for each global memory store
             local_load_transactions:  Number of local memory load transactions
            local_store_transactions:  Number of local memory store transactions
            shared_load_transactions:  Number of shared memory load transactions
           shared_store_transactions:  Number of shared memory store transactions
                    gld_transactions:  Number of global memory load transactions
                    gst_transactions:  Number of global memory store transactions
            sysmem_read_transactions:  Number of system memory read transactions
           sysmem_write_transactions:  Number of system memory write transactions
              tex_cache_transactions:  Texture cache read transactions
              dram_read_transactions:  Device memory read transactions
             dram_write_transactions:  Device memory write transactions
                l2_read_transactions:  Memory read transactions seen at L2 cache for all read requests
               l2_write_transactions:  Memory write transactions seen at L2 cache for all write requests
               local_load_throughput:  Local memory load throughput
              local_store_throughput:  Local memory store throughput
              shared_load_throughput:  Shared memory load throughput
             shared_store_throughput:  Shared memory store throughput
                  l2_read_throughput:  Memory read throughput seen at L2 cache for all read requests
                 l2_write_throughput:  Memory write throughput seen at L2 cache for all write requests
              sysmem_read_throughput:  System memory read throughput
             sysmem_write_throughput:  System memory write throughput
                           cf_issued:  Number of issued control-flow instructions
                         cf_executed:  Number of executed control-flow instructions
                         ldst_issued:  Number of issued load and store instructions
                       ldst_executed:  Number of executed load and store instructions
                            flops_sp:  Single-precision floating point operations executed
                        flops_sp_add:  Single-precision floating point add operations executed
                        flops_sp_mul:  Single-precision floating point multiply operations executed
                        flops_sp_fma:  Single-precision floating point multiply accumulate operations executed
                            flops_dp:  Double-precision floating point operations executed
                        flops_dp_add:  Double-precision floating point add operations executed
                        flops_dp_mul:  Double-precision floating point multiply operations executed
                        flops_dp_fma:  Double-precision floating point multiply accumulate operations executed
                    flops_sp_special:  Single-precision floating point special operations executed
               l1_shared_utilization:  The utilization level of the L1/shared memory relative to peak utilization
                      l2_utilization:  The utilization level of the L2 cache relative to the peak utilization
                     tex_utilization:  The utilization level of the texture cache relative to the peak utilization
                    dram_utilization:  The utilization level of the device memory relative to the peak utilization
                  sysmem_utilization:  The utilization level of the system memory relative to the peak utilization
                 ldst_fu_utilization:  The utilization level of the multiprocessor function units that execute load and store instructions
                  alu_fu_utilization:  The utilization level of the multiprocessor function units that execute integer and floating-point arithmetic instructions
                   cf_fu_utilization:  The utilization level of the multiprocessor function units that execute control-flow instructions
                  tex_fu_utilization:  The utilization level of the multiprocessor function units that execute texture instructions
                       inst_executed:  The number of instructions executed
                         inst_issued:  The number of instructions issued
                         issue_slots:  The number of issue slots used



events
------

::

    (chroma_env)delta:e blyth$ which nvprof
    /Developer/NVIDIA/CUDA-5.5/bin/nvprof
    (chroma_env)delta:e blyth$ 
    (chroma_env)delta:e blyth$ nvprof --query-events
    Available Events:
                                Name   Description
    Device 0 (GeForce GT 750M):
            Domain domain_a:
           tex0_cache_sector_queries:  Number of texture cache 0 requests. This increments by 1 for each 32-byte access.
           tex1_cache_sector_queries:  Number of texture cache 1 requests. This increments by 1 for each 32-byte access.
           tex2_cache_sector_queries:  Number of texture cache 2 requests. This increments by 1 for each 32-byte access. Value will be 0 for devices that contain only 2 texture units.
           tex3_cache_sector_queries:  Number of texture cache 3 requests. This increments by 1 for each 32-byte access. Value will be 0 for devices that contain only 2 texture units.
            tex0_cache_sector_misses:  Number of texture cache 0 misses. This increments by 1 for each 32-byte access.
            tex1_cache_sector_misses:  Number of texture cache 1 misses. This increments by 1 for each 32-byte access.
            tex2_cache_sector_misses:  Number of texture cache 2 misses. This increments by 1 for each 32-byte access. Value will be 0 for devices that contain only 2 texture units.
            tex3_cache_sector_misses:  Number of texture cache 3 misses. This increments by 1 for each 32-byte access. Value will be 0 for devices that contain only 2 texture units.
                   elapsed_cycles_sm:  Elapsed clocks

            Domain domain_b:
               fb_subp0_read_sectors:  Number of DRAM read requests to sub partition 0, increments by 1 for 32 byte access.
               fb_subp1_read_sectors:  Number of DRAM read requests to sub partition 1, increments by 1 for 32 byte access.
              fb_subp0_write_sectors:  Number of DRAM write requests to sub partition 0, increments by 1 for 32 byte access.
              fb_subp1_write_sectors:  Number of DRAM write requests to sub partition 1, increments by 1 for 32 byte access.
        l2_subp0_write_sector_misses:  Number of write misses in slice 0 of L2 cache. This increments by 1 for each 32-byte access.
        l2_subp1_write_sector_misses:  Number of write misses in slice 1 of L2 cache. This increments by 1 for each 32-byte access.
        l2_subp2_write_sector_misses:  Number of write misses in slice 2 of L2 cache. This increments by 1 for each 32-byte access.
        l2_subp3_write_sector_misses:  Number of write misses in slice 3 of L2 cache. This increments by 1 for each 32-byte access.
         l2_subp0_read_sector_misses:  Number of read misses in slice 0 of L2 cache. This increments by 1 for each 32-byte access.
         l2_subp1_read_sector_misses:  Number of read misses in slice 1 of L2 cache. This increments by 1 for each 32-byte access.
         l2_subp2_read_sector_misses:  Number of read misses in slice 2 of L2 cache. This increments by 1 for each 32-byte access.
         l2_subp3_read_sector_misses:  Number of read misses in slice 3 of L2 cache. This increments by 1 for each 32-byte access.
    l2_subp0_write_l1_sector_queries:  Number of write requests from L1 to slice 0 of L2 cache. This increments by 1 for each 32-byte access.
    l2_subp1_write_l1_sector_queries:  Number of write requests from L1 to slice 1 of L2 cache. This increments by 1 for each 32-byte access.
    l2_subp2_write_l1_sector_queries:  Number of write requests from L1 to slice 2 of L2 cache. This increments by 1 for each 32-byte access.
    l2_subp3_write_l1_sector_queries:  Number of write requests from L1 to slice 3 of L2 cache. This increments by 1 for each 32-byte access.
     l2_subp0_read_l1_sector_queries:  Number of read requests from L1 to slice 0 of L2 cache. This increments by 1 for each 32-byte access.
     l2_subp1_read_l1_sector_queries:  Number of read requests from L1 to slice 1 of L2 cache. This increments by 1 for each 32-byte access.
     l2_subp2_read_l1_sector_queries:  Number of read requests from L1 to slice 2 of L2 cache. This increments by 1 for each 32-byte access.
     l2_subp3_read_l1_sector_queries:  Number of read requests from L1 to slice 3 of L2 cache. This increments by 1 for each 32-byte access.
        l2_subp0_read_l1_hit_sectors:  Number of read requests from L1 that hit in slice 0 of L2 cache. This increments by 1 for each 32-byte access.
        l2_subp1_read_l1_hit_sectors:  Number of read requests from L1 that hit in slice 1 of L2 cache. This increments by 1 for each 32-byte access.
        l2_subp2_read_l1_hit_sectors:  Number of read requests from L1 that hit in slice 2 of L2 cache. This increments by 1 for each 32-byte access.
        l2_subp3_read_l1_hit_sectors:  Number of read requests from L1 that hit in slice 3 of L2 cache. This increments by 1 for each 32-byte access.
    l2_subp0_read_tex_sector_queries:  Number of read requests from Texture cache to slice 0 of L2 cache. This increments by 1 for each 32-byte access.
    l2_subp1_read_tex_sector_queries:  Number of read requests from Texture cache to slice 1 of L2 cache. This increments by 1 for each 32-byte access.
    l2_subp2_read_tex_sector_queries:  Number of read requests from Texture cache to slice 2 of L2 cache. This increments by 1 for each 32-byte access.
    l2_subp3_read_tex_sector_queries:  Number of read requests from Texture cache to slice 3 of L2 cache. This increments by 1 for each 32-byte access.
       l2_subp0_read_tex_hit_sectors:  Number of read requests from Texture cache that hit in slice 0 of L2 cache. This increments by 1 for each 32-byte access.
       l2_subp1_read_tex_hit_sectors:  Number of read requests from Texture cache that hit in slice 1 of L2 cache. This increments by 1 for each 32-byte access.
       l2_subp2_read_tex_hit_sectors:  Number of read requests from Texture cache that hit in slice 2 of L2 cache. This increments by 1 for each 32-byte access.
       l2_subp3_read_tex_hit_sectors:  Number of read requests from Texture cache that hit in slice 3 of L2 cache. This increments by 1 for each 32-byte access.
    l2_subp0_read_sysmem_sector_queries:  Number of system memory read requests to slice 0 of L2 cache. This increments by 1 for each 32-byte access.
    l2_subp1_read_sysmem_sector_queries:  Number of system memory read requests to slice 1 of L2 cache. This increments by 1 for each 32-byte access.
    l2_subp2_read_sysmem_sector_queries:  Number of system memory read requests to slice 2 of L2 cache. This increments by 1 for each 32-byte access.
    l2_subp3_read_sysmem_sector_queries:  Number of system memory read requests to slice 3 of L2 cache. This increments by 1 for each 32-byte access.
    l2_subp0_write_sysmem_sector_queries:  Number of system memory write requests to slice 0 of L2 cache. This increments by 1 for each 32-byte access.
    l2_subp1_write_sysmem_sector_queries:  Number of system memory write requests to slice 1 of L2 cache. This increments by 1 for each 32-byte access.
    l2_subp2_write_sysmem_sector_queries:  Number of system memory write requests to slice 2 of L2 cache. This increments by 1 for each 32-byte access.
    l2_subp3_write_sysmem_sector_queries:  Number of system memory write requests to slice 3 of L2 cache. This increments by 1 for each 32-byte access.
    l2_subp0_total_read_sector_queries:  Total read requests to slice 0 of L2 cache. This includes requests from  L1, Texture cache, system memory. This increments by 1 for each 32-byte access.
    l2_subp1_total_read_sector_queries:  Total read requests to slice 1 of L2 cache. This includes requests from  L1, Texture cache, system memory. This increments by 1 for each 32-byte access.
    l2_subp2_total_read_sector_queries:  Total read requests to slice 2 of L2 cache. This includes requests from  L1, Texture cache, system memory. This increments by 1 for each 32-byte access.
    l2_subp3_total_read_sector_queries:  Total read requests to slice 3 of L2 cache. This includes requests from  L1, Texture cache, system memory. This increments by 1 for each 32-byte access.
    l2_subp0_total_write_sector_queries:  Total write requests to slice 0 of L2 cache. This includes requests from  L1, Texture cache, system memory. This increments by 1 for each 32-byte access.
    l2_subp1_total_write_sector_queries:  Total write requests to slice 1 of L2 cache. This includes requests from  L1, Texture cache, system memory. This increments by 1 for each 32-byte access.
    l2_subp2_total_write_sector_queries:  Total write requests to slice 2 of L2 cache. This includes requests from  L1, Texture cache, system memory. This increments by 1 for each 32-byte access.
    l2_subp3_total_write_sector_queries:  Total write requests to slice 3 of L2 cache. This includes requests from  L1, Texture cache, system memory. This increments by 1 for each 32-byte access.

            Domain domain_c:
                       gld_inst_8bit:  Total number of 8-bit global load instructions that are executed by all the threads across all thread blocks.
                      gld_inst_16bit:  Total number of 16-bit global load instructions that are executed by all the threads across all thread blocks.
                      gld_inst_32bit:  Total number of 32-bit global load instructions that are executed by all the threads across all thread blocks.
                      gld_inst_64bit:  Total number of 64-bit global load instructions that are executed by all the threads across all thread blocks.
                     gld_inst_128bit:  Total number of 128-bit global load instructions that are executed by all the threads across all thread blocks.
                       gst_inst_8bit:  Total number of 8-bit global store instructions that are executed by all the threads across all thread blocks.
                      gst_inst_16bit:  Total number of 16-bit global store instructions that are executed by all the threads across all thread blocks.
                      gst_inst_32bit:  Total number of 32-bit global store instructions that are executed by all the threads across all thread blocks.
                      gst_inst_64bit:  Total number of 64-bit global store instructions that are executed by all the threads across all thread blocks.
                     gst_inst_128bit:  Total number of 128-bit global store instructions that are executed by all the threads across all thread blocks.

            Domain domain_d:
                     prof_trigger_00:  User profiled generic trigger that can be inserted in any place of the code to collect the related information. Increments per warp.
                     prof_trigger_01:  User profiled generic trigger that can be inserted in any place of the code to collect the related information. Increments per warp.
                     prof_trigger_02:  User profiled generic trigger that can be inserted in any place of the code to collect the related information. Increments per warp.
                     prof_trigger_03:  User profiled generic trigger that can be inserted in any place of the code to collect the related information. Increments per warp.
                     prof_trigger_04:  User profiled generic trigger that can be inserted in any place of the code to collect the related information. Increments per warp.
                     prof_trigger_05:  User profiled generic trigger that can be inserted in any place of the code to collect the related information. Increments per warp.
                     prof_trigger_06:  User profiled generic trigger that can be inserted in any place of the code to collect the related information. Increments per warp.
                     prof_trigger_07:  User profiled generic trigger that can be inserted in any place of the code to collect the related information. Increments per warp.
                      warps_launched:  Number of warps launched on a multiprocessor.
                    threads_launched:  Number of threads launched on a multiprocessor.
                        inst_issued1:  Number of single instruction issued per cycle
                        inst_issued2:  Number of dual instructions issued per cycle
                       inst_executed:  Number of instructions executed, do not include replays.
                         shared_load:  Number of executed load instructions where state space is specified as shared, increments per warp on a multiprocessor.
                        shared_store:  Number of executed store instructions where state space is specified as shared, increments per warp on a multiprocessor.
                          local_load:  Number of executed load instructions where state space is specified as local, increments per warp on a multiprocessor.
                         local_store:  Number of executed store instructions where state space is specified as local, increments per warp on a multiprocessor.
                         gld_request:  Number of executed load instructions where the state space is not specified and hence generic addressing is used, increments per warp on a multiprocessor. It can include the load operations from global,local and shared state space.
                         gst_request:  Number of executed store instructions where the state space is not specified and hence generic addressing is used, increments per warp on a multiprocessor. It can include the store operations to global,local and shared state space.
                          atom_count:  Number of warps executing atomic reduction operations. Increments by one if at least one thread in a warp executes the instruction.
                          gred_count:  Number of warps executing reduction operations on global and shared memory. Increments by one if at least one thread in a warp executes the instruction
                              branch:  Number of branch instructions executed per warp on a multiprocessor.
                    divergent_branch:  Number of divergent branches within a warp. This counter will be incremented by one if at least one thread in a warp diverges (that is, follows a different execution path) via a conditional branch.
                       active_cycles:  Number of cycles a multiprocessor has at least one active warp. This event can increment by 0 - 1 on each cycle.
                        active_warps:  Accumulated number of active warps per cycle. For every cycle it increments by the number of active warps in the cycle which can be in the range 0 to 64.
                     sm_cta_launched:  Number of thread blocks launched on a multiprocessor.
             local_load_transactions:  Number of local load transactions from L1 cache. Increments by 1 per transaction. Transaction can be 32/64/96/128B.
            local_store_transactions:  Number of local store transactions to L1 cache. Increments by 1 per transaction. Transaction can be 32/64/96/128B.
         l1_shared_load_transactions:  Number of shared load transactions. Increments by 1 per transaction. Transaction can be 32/64/96/128B.
        l1_shared_store_transactions:  Number of shared store transactions. Increments by 1 per transaction. Transaction can be 32/64/96/128B.
       __l1_global_load_transactions:  Number of global load transactions from L1 cache. Increments by 1 per transaction. Transaction can be 32/64/96/128B.
      __l1_global_store_transactions:  Number of global store transactions from L1 cache. Increments by 1 per transaction. Transaction can be 32/64/96/128B.
                   l1_local_load_hit:  Number of cache lines that hit in L1 cache for local memory load accesses. In case of perfect coalescing this increments by 1,2, and 4 for 32, 64 and 128 bit accesses by a warp respectively.
                  l1_local_load_miss:  Number of cache lines that miss in L1 cache for local memory load accesses. In case of perfect coalescing this increments by 1,2, and 4 for 32, 64 and 128 bit accesses by a warp respectively.
                  l1_local_store_hit:  Number of cache lines that hit in L1 cache for local memory store accesses. In case of perfect coalescing this increments by 1,2, and 4 for 32, 64 and 128 bit accesses by a warp respectively.
                 l1_local_store_miss:  Number of cache lines that miss in L1 cache for local memory store accesses. In case of perfect coalescing this increments by 1,2, and 4 for 32,64 and 128 bit accesses by a warp respectively.
                  l1_global_load_hit:  Number of cache lines that hit in L1 cache for global memory load accesses. In case of perfect coalescing this increments by 1,2, and 4 for 32, 64 and 128 bit accesses by a warp respectively.
                 l1_global_load_miss:  Number of cache lines that miss in L1 cache for global memory load accesses. In case of perfect coalescing this increments by 1,2, and 4 for 32, 64 and 128 bit accesses by a warp respectively.
    uncached_global_load_transaction:  Number of uncached global load transactions. Increments by 1 per transaction. Transaction can be 32/64/96/128B.
            global_store_transaction:  Number of global store transactions. Increments by 1 per transaction. Transaction can be 32/64/96/128B.
                  shared_load_replay:  Replays caused due to shared load bank conflict (when the addresses for two or more shared memory load requests fall in the same memory bank) or when there is no conflict but the total number of words accessed by all threads in the warp executing that instruction exceed the number of words that can be loaded in one cycle (256 bytes).
                 shared_store_replay:  Replays caused due to shared store bank conflict (when the addresses for two or more shared memory store requests fall in the same memory bank) or when there is no conflict but the total number of words accessed by all threads in the warp executing that instruction exceed the number of words that can be stored in one cycle.
    global_ld_mem_divergence_replays:  global ld is replayed due to divergence
    global_st_mem_divergence_replays:  global st is replayed due to divergence

