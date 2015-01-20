CUDA Context Cleanup
====================

Observe that GPU memory usage is high and stays high for minutes
even when no applications are actively using the GPU.  

Is this due to bad actors not cleaning up their CUDA Contexts ?

* does host thread death automatically cleanup any CUDA contexts that were openend ?

Running with OSX in normal GUI mode
-------------------------------------

Immediately after restart with Finder, Safari and Terminal::

    delta:~ blyth$ cuda_info.sh
    timestamp                Mon Dec  1 13:25:26 2014
    tag                      default
    name                     GeForce GT 750M
    compute capability       (3, 0)
    memory total             2.1G
    memory used              447.8M
    memory free              1.7G
    delta:~ blyth$ 

After a ~20min in Safari, Terminal, Sys Prefs a whole gig of GPU memory is gone::

    delta:~ blyth$ cuda_info.sh
    timestamp                Mon Dec  1 13:42:33 2014
    tag                      default
    name                     GeForce GT 750M
    compute capability       (3, 0)
    memory total             2.1G
    memory used              1.4G
    memory free              738.7M
    delta:~ blyth$ 

After sleeping during lunch, 1G of VRAM frees up::

    delta:env blyth$ cuda_info.sh
    timestamp                Mon Dec  1 14:55:09 2014
    tag                      default
    name                     GeForce GT 750M
    compute capability       (3, 0)
    memory total             2.1G
    memory used              440.8M
    memory free              1.7G
    delta:env blyth$ 



Pragmatic Solution
------------------

* run in ">console" mode when you need maximum GPU memory 
* for GUI usage, restart the machine often to ensure will have enough GPU memory 

  * TODO: add a GPU memory configurable minimum to g4daeview.py, it will try to 
    run with mapped/pinned host memory but the performance is factor 3-4 lower that 
    when sufficient memory to go GPU resident 


OSX VRAM 
-----------

* https://forums.adobe.com/thread/1326404

  * some Adobe raycaster, running into VRAM pressure

* http://www.anandtech.com/show/2804
* http://arstechnica.com/apple/2005/04/macosx-10-4/13/

  * abouts OSX VRAM usage, number of open windows matters

* retina screen support is presumably eating lots of VRAM 


Running without GUI using ">console" login
---------------------------------------------

GPU memory is almost entirely free::

    delta:~ blyth$ cuda_info.sh
    timestamp                Mon Nov 24 12:57:17 2014
    tag                      default
    name                     GeForce GT 750M
    compute capability       (3, 0)
    memory total             2.1G
    memory used              96.0M
    memory free              2.1G
    delta:~ blyth$ 

While running g4daechroma.sh::

    delta:~ blyth$ cuda_info.sh
    timestamp                Mon Nov 24 13:01:46 2014
    tag                      default
    name                     GeForce GT 750M
    compute capability       (3, 0)
    memory total             2.1G
    memory used              111.2M
    memory free              2.0G

Huh memory usage seems variable, sometimes get 220M used.

After one mocknuwa run::

    elta:~ blyth$ cuda_info.sh
    timestamp                Mon Nov 24 13:04:28 2014
    tag                      default
    name                     GeForce GT 750M
    compute capability       (3, 0)
    memory total             2.1G
    memory used              277.2M
    memory free              1.9G
    delta:~ blyth$ 

After 2nd mocknuwa, not increasing::

    delta:~ blyth$ cuda_info.sh 
    timestamp                Mon Nov 24 13:06:13 2014
    tag                      default
    name                     GeForce GT 750M
    compute capability       (3, 0)
    memory total             2.1G
    memory used              277.2M
    memory free              1.9G
    delta:~ blyth$ 

ctrl-c interrupt g4daechroma.py cleans up ok::

    delta:~ blyth$ cuda_info.sh 
    timestamp                Mon Nov 24 13:07:50 2014
    tag                      default
    name                     GeForce GT 750M
    compute capability       (3, 0)
    memory total             2.1G
    memory used              96.0M
    memory free              2.1G
    delta:~ blyth$ 

Repeating::

    delta:~ blyth$ cuda_info.sh 
    timestamp                Mon Nov 24 13:09:06 2014
    tag                      default
    name                     GeForce GT 750M
    compute capability       (3, 0)
    memory total             2.1G
    memory used              111.2M
    memory free              2.0G


Mem reporting from inside the process doesnt match the above::

    chroma_env)delta:MockNuWa blyth$ python mocknuwa.py 
                 a_min_free_gpu_mem :     300.00M  300000000  
                 b_node_array_usage :      54.91M  54909600  
                    b_node_itemsize :      16.00M  16  
                      b_split_index :       3.43M  3431850  
                          b_n_extra :       1.00M  1  
                          b_n_nodes :       3.43M  3431850  
                        b_splitting :       0.00M  0  
                  c_triangle_nbytes :      28.83M  28829184  
                     c_triangle_gpu :       1.00M  1  
                  d_vertices_nbytes :      14.60M  14597424  
                     d_triangle_gpu :       1.00M  1  
                         a_gpu_used :      99.57M  99573760  
                         b_gpu_used :     129.72M  129720320  
                         c_gpu_used :     184.64M  184639488  
                         d_gpu_used :     213.48M  213475328  
                         e_gpu_used :     228.16M  228155392  
    (chroma_env)delta:MockNuWa blyth$ 



Huh GPUGeometry init only happening when the first evt arrives::

    2014-11-24 13:22:58,720 INFO    env.geant4.geometry.collada.g4daeview.daedirectpropagator:53  DAEDirectPropagator ctrl {u'reset_rng_states': 1, u'nthreads_per_block': 64, u'seed': 0, u'max_blocks': 1024, u'max_steps': 30, u'COLUMNS': u'max_blocks:i,max_steps:i,nthreads_per_block:i,reset_rng_states:i,seed:i'} 
    2014-11-24 13:22:58,720 WARNING env.geant4.geometry.collada.g4daeview.daedirectpropagator:63  reset_rng_states
    2014-11-24 13:22:58,720 INFO    env.geant4.geometry.collada.g4daeview.daechromacontext:182 _set_rng_states
    2014-11-24 13:22:58,851 INFO    chroma.gpu.geometry :19  GPUGeometry.__init__ min_free_gpu_mem 300000000.0 
    2014-11-24 13:22:59,073 INFO    chroma.gpu.geometry :206 Optimization: Sufficient memory to move triangles onto GPU
    2014-11-24 13:22:59,085 INFO    chroma.gpu.geometry :220 Optimization: Sufficient memory to move vertices onto GPU
    2014-11-24 13:22:59,085 INFO    chroma.gpu.geometry :248 device usage:
    ----------
    nodes             3.4M  54.9M
    total                   54.9M
    ----------
    device total             2.1G
    device used            228.2M
    device free              1.9G

    2014-11-24 13:22:59,089 INFO    env.geant4.geometry.collada.g4daeview.daechromacontext:177 _get_rng_states
    2014-11-24 13:22:59,090 INFO    env.geant4.geometry.collada.g4daeview.daechromacontext:132 setup_rng_states using seed 0 
    2014-11-24 13:22:59,512 INFO    chroma.gpu.photon_hit:204 nwork 4165 step 0 max_steps 30 nsteps 30 
    2014-11-24 13:23:00,157 INFO    chroma.gpu.photon_hit:242 step 0 propagate_hit_kernel times  [0.6453909912109375] 
    2014-11-24 13:23:00,319 INFO    env.geant4.geometry.collada.g4daeview.daedirectpropagator:86  daedirectpropagator:propagate returning photons_end.as_npl()



Timings are not stable, even when running in console mode with no memory or other GPU user
contention.






Stuck Python Process
----------------------

Killing an old stuck process succeeds to free some ~200M of GPU memory, 
but still how is 1.7 G being used. 
When running with visible apps only Finder and Terminal.

::

    (chroma_env)delta:MockNuWa blyth$ cuda_info.py 
    timestamp                Mon Nov 24 12:40:09 2014
    tag                      default
    name                     GeForce GT 750M
    compute capability       (3, 0)
    memory total             2.1G
    memory used              1.9G
    memory free              232.1M
    (chroma_env)delta:MockNuWa blyth$ ps aux | grep python
    blyth           69938   1.2  0.2 35266100  31340 s000  S+    3Nov14 126:41.78 python /Users/blyth/env/bin/daedirectpropagator.py mock001
    blyth            2313   0.0  0.0  2423368    284 s007  R+   12:40PM   0:00.00 grep python
    (chroma_env)delta:MockNuWa blyth$ kill -9 69938 
    (chroma_env)delta:MockNuWa blyth$ ps aux | grep python
    blyth            2315   0.0  0.0  2423368    240 s007  R+   12:40PM   0:00.00 grep python
    (chroma_env)delta:MockNuWa blyth$ cuda_info.py 
    timestamp                Mon Nov 24 12:40:47 2014
    tag                      default
    name                     GeForce GT 750M
    compute capability       (3, 0)
    memory total             2.1G
    memory used              1.7G
    memory free              400.0M





Search
-------

* :google:`CUDA Context Cleanup`

https://devblogs.nvidia.com/parallelforall/pro-tip-clean-up-after-yourself-ensure-correct-profiling/ 

    If your application uses the CUDA Runtime API, call cudaDeviceReset() just
    before exiting, or when the application finishes making CUDA calls and using
    device data. If your application uses the CUDA Driver API, call
    cuProfilerStop() on each context to flush the profiling buffers before
    destroying the context with cuCtxDestroy().


