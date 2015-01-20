CUDA GDB
=========

Docs
----

* http://docs.nvidia.com/cuda/cuda-gdb/
* http://developer.download.nvidia.com/compute/cuda/2_1/cudagdb/CUDA_GDB_User_Manual.pdf  NEWER PDF? 
* http://on-demand.gputechconf.com/gtc/2012/presentations/S0027B-GTC2012-Debugging-MEMCHECK.pdf


Dr Dobbs
----------

Debugging CUDA and using CUDA-GDB

* http://www.drdobbs.com/parallel/cuda-supercomputing-for-the-masses-part/220601124



Courses
-------

* http://istar.cse.cuhk.edu.hk/icuda/

a hands-on seminar series on pragmatic CUDA programming. It emphasizes samples, libraries and tools



(Manual) Configurations for GPU Debugging
-------------------------------------------

Debugging a CUDA GPU involves pausing that GPU. When the graphics desktop 
manager is running on the same GPU, then debugging that GPU freezes the GUI and 
makes the desktop unusable. To avoid this, use CUDA-GDB in the following system 
configurations:

Single GPU
~~~~~~~~~~~~

In a single GPU system, CUDA-GDB can be used to debug CUDA applications only if 
no X11 server (on Linux) or no Aqua desktop manager (on Mac OS X) is running on that 
system. On Linux you can stop the X11 server by stopping the gdm service. On Mac OS 
X you can log in with >console as the user name in the desktop UI login screen. This 
allows CUDA applications to be executed and debugged in a single GPU configuration. 

Multi-GPU Debugging with the Desktop Manager Running
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 
This can be achieved by running the desktop GUI on one GPU and CUDA on the other 
GPU to avoid hanging the desktop GUI.

On Linux 
^^^^^^^^^^

The CUDA driver automatically excludes the GPU used by X11 from being visible to 
the application being debugged. This might alter the behavior of the application since, if 
there are n GPUs in the system, then only n-1 GPUs will be visible to the application. 

On Mac OS X 
^^^^^^^^^^^^^

The CUDA driver exposes every CUDA-capable GPU in the system, including the one 
used by Aqua desktop manager. To determine which GPU should be used for CUDA, 
run the deviceQuery app from the CUDA SDK sample. The output of deviceQuery 
as shown in Figure 1  deviceQuery Output indicates all the GPUs in the system. 
For example, if you have two GPUs you will see Device0: "GeForce xxxx" and 
Device1: "GeForce xxxx". Choose the Device<index> that is not rendering the 
desktop on your connected monitor. If Device0 is rendering the desktop, then choose 
Device1 for running and debugging the CUDA application. This exclusion of the 
desktop can be achieved by setting the CUDA_VISIBLE_DEVICES environment variable 
to 1:: 

   `export CUDA_VISIBLE_DEVICES=1`


.. sidebar:: BUT need to force Desktop/OpenGL and pygame/SDL/OpenGL to use Integrated Graphics only ? 

   See `gfxcardstatus-` which uses IOKit kSetMux calls



Extras
-------

* `info cuda threads`  long list of threads showing file and line numbers in the kernels

Debug Setup
--------------------------

Its more convenient to debug by

#. put rMBP into console mode, by logging in as username ">console"
#. ssh into rMPB from G4PB and run debug scripts like 3199.sh from there 
#. its real slow debugging, consider using smaller image sizes



g4daechroma debug
-------------------

::

    Launch of CUDA Kernel 0 (write_size<<<(1,1,1),(1,1,1)>>>) on Device 0]
    *** compiler output in /var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/tmpFOMSIi
    [Launch of CUDA Kernel 1 (write_size<<<(1,1,1),(1,1,1)>>>) on Device 0]
    *** compiler output in /var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/tmpAN87U6
    [Launch of CUDA Kernel 2 (write_size<<<(1,1,1),(1,1,1)>>>) on Device 0]
    2014-11-24 13:44:22,093 INFO    chroma.gpu.geometry :206 Optimization: Sufficient memory to move triangles onto GPU
    2014-11-24 13:44:22,105 INFO    chroma.gpu.geometry :220 Optimization: Sufficient memory to move vertices onto GPU
    2014-11-24 13:44:22,105 INFO    chroma.gpu.geometry :248 device usage:
    ----------
    nodes             3.4M  54.9M
    total                   54.9M
    ----------
    device total             2.1G
    device used            244.9M
    device free              1.9G

    *** compiler output in /var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/tmpdgM3nQ
    [Launch of CUDA Kernel 3 (write_size<<<(1,1,1),(1,1,1)>>>) on Device 0]
    2014-11-24 13:44:23,206 INFO    daechromacontext    :177 _get_rng_states
    2014-11-24 13:44:23,206 INFO    daechromacontext    :132 setup_rng_states using seed 0 
    *** compiler output in /var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/tmpVvN6DN
    [Launch of CUDA Kernel 4 (write_size<<<(1,1,1),(1,1,1)>>>) on Device 0]
    *** compiler output in /var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/tmpIdukSq
    [Launch of CUDA Kernel 5 (init_rng<<<(1025,1,1),(64,1,1)>>>) on Device 0]
    2014-11-24 13:44:28,228 INFO    chroma.gpu.photon_hit:204 nwork 4165 step 0 max_steps 30 nsteps 30 
    [Launch of CUDA Kernel 6 (propagate_hit<<<(66,1,1),(64,1,1)>>>) on Device 0]
    2014-11-24 13:45:08,737 WARNING chroma.gpu.photon_hit:238 kernel launch time 40.5083046875 > max_time 4.0 : ABORTING 
    2014-11-24 13:45:08,737 INFO    chroma.gpu.photon_hit:242 step 0 propagate_hit_kernel times  [40.5083046875] 
    *** compiler output in /var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/tmp2zPlV7
    *** compiler output in /var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/tmpA5dzDs
    [Launch of CUDA Kernel 7 (reduce_kernel_stage1<<<(3,1,1),(512,1,1)>>>) on Device 0]
    [Launch of CUDA Kernel 8 (reduce_kernel_stage2<<<(1,1,1),(512,1,1)>>>) on Device 0]
    2014-11-24 13:45:11,809 INFO    daedirectpropagator :86  daedirectpropagator:propagate returning photons_end.as_npl()
    2014-11-24 13:45:12,039 INFO    daedirectresponder  :52  DAEDirectResponder response (95, 4, 4) 



#. setup rng states doing compilation, maybe just because cache disabled in debug mode



::

    (chroma_env)delta:g4daeview blyth$ g4daechroma.sh --cuda-gdb
    /Users/blyth/env/bin/g4daechroma.sh Mon Nov 24 15:41:14 CST 2014
    /Users/blyth/env/bin/g4daechroma.sh Mon Nov 24 15:41:15 CST 2014
    NVIDIA (R) CUDA Debugger
    5.5 release
    Portions Copyright (C) 2007-2013 NVIDIA Corporation
    GNU gdb (GDB) 7.2
    ...
    2014-11-24 15:43:20,404 INFO    __main__            :54  polling 90 
    2014-11-24 15:43:31,435 INFO    __main__            :54  polling 100 
    2014-11-24 15:43:42,442 INFO    __main__            :54  polling 110 
    2014-11-24 15:43:43,548 INFO    env.zeromq.pyzmq.npyresponder:179 NPYResponder converting request.meta to dict [{u'args': {u'htag': u'zzmock007'},
      u'ctrl': {u'COLUMNS': u'max_blocks:i,max_steps:i,nthreads_per_block:i,reset_rng_states:i,seed:i',
                u'max_blocks': 1024,
                u'max_steps': 30,
                u'nthreads_per_block': 64,
                u'reset_rng_states': 1,
                u'seed': 0}}] 
    2014-11-24 15:43:43,548 INFO    daedirectresponder  :50  DAEDirectResponder request (684, 4, 4) 
    2014-11-24 15:43:43,548 INFO    __main__            :43  handler got obj (cpl or npl)
    2014-11-24 15:43:43,548 INFO    daedirectpropagator :53  DAEDirectPropagator ctrl {u'reset_rng_states': 1, u'nthreads_per_block': 64, u'seed': 0, u'max_blocks': 1024, u'max_steps': 30, u'COLUMNS': u'max_blocks:i,max_steps:i,nthreads_per_block:i,reset_rng_states:i,seed:i'} 
    2014-11-24 15:43:43,549 WARNING daedirectpropagator :63  reset_rng_states
    2014-11-24 15:43:43,549 INFO    daechromacontext    :187 _set_rng_states
    *** compiler output in /var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/tmpRWcjoc
    2014-11-24 15:43:45,794 INFO    daechromacontext    :182 _get_rng_states
    2014-11-24 15:43:45,794 INFO    daechromacontext    :137 setup_rng_states using seed 0 
    *** compiler output in /var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/tmpW1b0ii
    [Launch of CUDA Kernel 4 (write_size<<<(1,1,1),(1,1,1)>>>) on Device 0]
    *** compiler output in /var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/tmpA9FyVR
    [Launch of CUDA Kernel 5 (init_rng<<<(1025,1,1),(64,1,1)>>>) on Device 0]
    2014-11-24 15:43:50,093 INFO    chroma.gpu.photon_hit:204 nwork 684 step 0 max_steps 30 nsteps 30 
    [Launch of CUDA Kernel 6 (propagate_hit<<<(11,1,1),(64,1,1)>>>) on Device 0]
    ^C
    Program received signal SIGINT, Interrupt.
    [Switching focus to CUDA kernel 6, grid 7, block (0,0,0), thread (32,0,0), device 0, sm 1, warp 0, lane 0]
    0x000000011996e7e8 in intersect_mesh(const float3 * @generic, const float3 * @generic, Geometry * @generic, float * @generic, int) (origin=<value optimized out>, direction=<value optimized out>, 
        g=<value optimized out>, min_distance=<value optimized out>, last_hit_triangle=<value optimized out>) at mesh.h:108
    108     } // loop over children, starting with first_child
    (cuda-gdb) info cuda threads 
      BlockIdx ThreadIdx To BlockIdx ThreadIdx Count         Virtual PC           Filename  Line 
    Kernel 6
    *  (0,0,0)  (32,0,0)     (0,0,0)  (44,0,0)    13 0x000000011996e7e8             mesh.h   108 
       (0,0,0)  (45,0,0)     (0,0,0)  (47,0,0)     3 0x000000011995d8e0        intersect.h    21 
       (0,0,0)  (48,0,0)     (0,0,0)  (55,0,0)     8 0x000000011996e6c0             mesh.h   104 
       (0,0,0)  (56,0,0)     (0,0,0)  (57,0,0)     2 0x000000011996e7e8             mesh.h   108 
       (0,0,0)  (58,0,0)     (0,0,0)  (59,0,0)     2 0x000000011996e7d8             mesh.h   108 
       (0,0,0)  (60,0,0)     (0,0,0)  (63,0,0)     4 0x000000011996e7e8             mesh.h   108 
       (2,0,0)  (32,0,0)     (2,0,0)  (41,0,0)    10 0x000000011996e7e8             mesh.h   108 
       (2,0,0)  (42,0,0)     (2,0,0)  (43,0,0)     2 0x00000001199716c8        intersect.h   110 
       (2,0,0)  (44,0,0)     (2,0,0)  (52,0,0)     9 0x000000011996e7e8             mesh.h   108 
       ...
       (9,0,0)  (31,0,0)     (9,0,0)  (31,0,0)     1 0x00000001199715c0        intersect.h   102 
       (9,0,0)  (32,0,0)     (9,0,0)  (32,0,0)     1 0x000000011996e7e8             mesh.h   108 
       (9,0,0)  (33,0,0)     (9,0,0)  (33,0,0)     1 0x000000011999cf70         geometry.h    57 
       (9,0,0)  (34,0,0)     (9,0,0)  (63,0,0)    30 0x000000011996e7e8             mesh.h   108 
    (cuda-gdb) list
    103         
    104         if (curr >= STACK_SIZE) {
    105             printf("warning: intersect_mesh() aborted; node > tail\n");
    106             break;
    107         }
    108     } // loop over children, starting with first_child
    109 
    110     } // while nodes on stack
    111 
    112     //if (blockIdx.x == 0 && threadIdx.x == 0) {
    (cuda-gdb) p curr
    $1 = <value optimized out>


    (cuda-gdb) bt 
    #0  0x000000011996e7e8 in intersect_mesh(const float3 * @generic, const float3 * @generic, Geometry * @generic, float * @generic, int) (origin=<value optimized out>, 
        direction=<value optimized out>, g=<value optimized out>, min_distance=<value optimized out>, last_hit_triangle=<value optimized out>) at mesh.h:108
    #1  0x000000011997c618 in fill_state(State * @generic, Photon * @generic, Geometry * @generic) (s=<value optimized out>, p=<value optimized out>, g=<value optimized out>) at photon.h:168
    #2  0x0000000119993c28 in propagate_hit(int, int, unsigned int * @generic, unsigned int * @generic, curandState * @generic, float3 * @generic, float3 * @generic, float * @generic, float3 * @generic, float * @generic, unsigned int * @generic, int * @generic, float * @generic, int, int, int, Geometry * @generic, int * @generic, int * @generic)<<<(11,1,1),(64,1,1)>>> (first_photon=0, 
        nthreads=684, input_queue=0x707fe3c04, output_queue=0x707fe4800, rng_states=0x7080e0000, positions=0x707ee0000, directions=0x707ee2200, wavelengths=0x707fe0000, polarizations=0x707ee4400, 
        times=0x707fe0c00, histories=0x707fe2400, last_hit_triangles=0x707fe1800, weights=0x707fe3000, max_steps=30, use_weights=0, __val_paramscatter_first=0, __val_paramg=0x700177c00, 
        solid_map=0x7014c0000, solid_id_to_channel_id=0x707be8e00) at kernel.cu:189
    (cuda-gdb) f 


    neg_origin_inv_dir = {x = -76474.7656, y = -824920.688, z = -1.23878184e+20}


    cuda-gdb) info locals
    __T227 = 0
    root = {lower = {x = -23327.6914, y = -809820.562, z = -12110}, upper = {x = -9712.1377, y = -794399.188, z = -2139.94434}, child = 1, nchild = 2}
    inv_dir = {x = -4.775702, y = -1.0226711, z = -1.63317796e+16}
    child_ptr_stack = {130, 540, 7549, 27652, 7580, 7857, 2454275289, 825045562, 1900092336, 3697343089, 760470915, 2146882409, 2000637694, 343303227, 3891626781, 2757016765, 3997473261, 4015377800, 
      745490961, 2855539022, 36229159, 2375394427, 2208656873, 3920625546, 2614469314, 213338525, 2055366274, 2729976209, 827771411, 2430496777, 1198164420, 1789631187, 1922667440, 2747674190, 
      1461695896, 1770331341, 4284442852, 357535311, 919857376, 3053795110, 3533531372, 2124270060, 1004072597, 201138876, 3221275220, 1589659138, 1418386120, 1148950143, 4211579707, 1799726917, 
      2838977761, 3540708069, 892664404, 416103844, 2248389027, 1790015671, 3936883, 2612357890, 3481887924, 1619955951, 4188275966, 2963623483, 2005534713, 564854400, 747724021, 4037561738, 
      3431155922, 2620990454, 604900912, 610898209, 1473244091, 3880001339, 2879060316, 3300897679, 3960972039, 3201086624, 3814462934, 3426650044, 1930881632, 1981178788, 2956279691, 4272406256, 
      372705521, 1359389771, 1590302979, 3940206208, 3817999127, 2527835456, 2739078164, 716997849, 3235607043, 2550297745, 3688700200, 354502605, 2285793656, 2339138034, 3912354142, 2262255668, 
      469322622, 1319943359, 1916101235, 200441823, 509436982, 2160284593, 1687919695, 4153615582, 495735041, 3694469424, 2086893117, 4223008799, 105344742, 1698033424, 1149223145, 4183918790, 
      4176151950, 415739351, 817762972, 3768072560, 1931430949, 2698979439, 3481477932, 1994322914, 4078299950, 1268233995, 3254069145, 91029129, 498234704, 1636613942, 3710087092, 3876816560, 
      3510446387, 3870169008, 1370156410, 2442498047, 2324396523, 1258730334, 621954739, 1053015373, 491820717, 3386515432, 2203703266, 120167176, 2383669740, 1038666440, 2927342870, 3583197824, 
      1236241846, 2474675929, 679052891, 2451259584, 2177706146, 606842882, 3546980104, 2289281509, 353873434, 2041926837, 1238346748, 2729109726, 2843938395, 2938124210, 2554443866, 1494477920, 
      693378319, 2020963566, 2000385949, 3744098787, 650307220, 2631327075, 1529128757, 595871428, 3206666562, 458062987, 875238192, 3729317374, 1368843921, 3478430230, 3234384578, 3232435428, 
      321359326, 994274524, 361184397, 4285497594, 915263578, 1486882838, 9988613, 829077170, 677216046, 4141828204, 165804609, 1086678519, 2933434608, 1351662802, 2640085040, 2611502932, 
      2033698714, 2008873254, 3995557835, 1020873906, 67873555, 2230337823...}
    __T228 = {x = 16013.3027, y = 806633.438, z = 7585.1001}
    neg_origin_inv_dir = {x = -76474.7656, y = -824920.688, z = -1.23878184e+20}


switching cuda threads::

    (cuda-gdb) cuda thread
    thread (32,0,0)
    (cuda-gdb) cuda thread all
    CUDA focus unchanged.
    (cuda-gdb) cuda thread (54)
    [Switching focus to CUDA kernel 6, grid 7, block (0,0,0), thread (54,0,0), device 0, sm 1, warp 0, lane 22]
    0x000000011996df80  77          if (intersect_node(neg_origin_inv_dir, inv_dir, g, node, min_distance)) {
    (cuda-gdb) 


Hmm unhealthy infinities ? Nope the code handles it, just need one of three coordinates to be finite::

    (cuda-gdb) info locals
    node = {lower = {x = -11941.7461, y = -807158.438, z = -9682.72656}, upper = {x = -11920.0967, y = -807139.125, z = -9674.96191}, child = 1561362, nchild = 0}
    i = 1548710
    first_child = 1548708
    nchild = 6
    __T227 = 0
    root = {lower = {x = -23327.6914, y = -809820.562, z = -12110}, upper = {x = -9712.1377, y = -794399.188, z = -2139.94434}, child = 1, nchild = 2}
    inv_dir = {x = -inf, y = inf, z = -1}


    (cuda-gdb) p origin
    $13 = (const @generic float3 * @register) 0x3fffbb0
    (cuda-gdb) p *origin
    $14 = {x = -11603.4365, y = -805333.562, z = -9297.40039}
    (cuda-gdb) 

    (cuda-gdb) p direction
    $15 = (const @generic float3 * @register) 0x3fffbbc
    (cuda-gdb) p *direction
    $16 = {x = -0, y = 0, z = -1}



kernel pileup ?
-----------------


::

    (cuda-gdb) kill
    Kill the program being debugged? (y or n) y
    [Termination of CUDA Kernel 29 (init_rng<<<(1025,1,1),(64,1,1)>>>) on Device 0]
    [Termination of CUDA Kernel 26 (init_rng<<<(1025,1,1),(64,1,1)>>>) on Device 0]
    [Termination of CUDA Kernel 23 (init_rng<<<(1025,1,1),(64,1,1)>>>) on Device 0]
    [Termination of CUDA Kernel 20 (init_rng<<<(1025,1,1),(64,1,1)>>>) on Device 0]
    [Termination of CUDA Kernel 17 (init_rng<<<(1025,1,1),(64,1,1)>>>) on Device 0]
    [Termination of CUDA Kernel 14 (init_rng<<<(1025,1,1),(64,1,1)>>>) on Device 0]
    [Termination of CUDA Kernel 11 (init_rng<<<(1025,1,1),(64,1,1)>>>) on Device 0]
    [Termination of CUDA Kernel 8 (init_rng<<<(1025,1,1),(64,1,1)>>>) on Device 0]
    [Termination of CUDA Kernel 28 (reduce_kernel_stage1<<<(1,1,1),(512,1,1)>>>) on Device 0]
    [Termination of CUDA Kernel 25 (reduce_kernel_stage1<<<(1,1,1),(512,1,1)>>>) on Device 0]
    [Termination of CUDA Kernel 22 (reduce_kernel_stage1<<<(1,1,1),(512,1,1)>>>) on Device 0]
    [Termination of CUDA Kernel 19 (reduce_kernel_stage1<<<(1,1,1),(512,1,1)>>>) on Device 0]
    [Termination of CUDA Kernel 16 (reduce_kernel_stage1<<<(1,1,1),(512,1,1)>>>) on Device 0]
    [Termination of CUDA Kernel 13 (reduce_kernel_stage1<<<(1,1,1),(512,1,1)>>>) on Device 0]
    [Termination of CUDA Kernel 10 (reduce_kernel_stage1<<<(1,1,1),(512,1,1)>>>) on Device 0]
    [Termination of CUDA Kernel 7 (reduce_kernel_stage1<<<(1,1,1),(512,1,1)>>>) on Device 0]
    [Termination of CUDA Kernel 5 (init_rng<<<(1025,1,1),(64,1,1)>>>) on Device 0]
    [Termination of CUDA Kernel 4 (write_size<<<(1,1,1),(1,1,1)>>>) on Device 0]
    [Termination of CUDA Kernel 30 (propagate_hit<<<(11,1,1),(64,1,1)>>>) on Device 0]
    [Termination of CUDA Kernel 27 (propagate_hit<<<(1,1,1),(64,1,1)>>>) on Device 0]
    [Termination of CUDA Kernel 24 (propagate_hit<<<(1,1,1),(64,1,1)>>>) on Device 0]
    [Termination of CUDA Kernel 21 (propagate_hit<<<(1,1,1),(64,1,1)>>>) on Device 0]
    [Termination of CUDA Kernel 18 (propagate_hit<<<(1,1,1),(64,1,1)>>>) on Device 0]
    [Termination of CUDA Kernel 15 (propagate_hit<<<(1,1,1),(64,1,1)>>>) on Device 0]
    [Termination of CUDA Kernel 12 (propagate_hit<<<(1,1,1),(64,1,1)>>>) on Device 0]
    [Termination of CUDA Kernel 9 (propagate_hit<<<(1,1,1),(64,1,1)>>>) on Device 0]
    [Termination of CUDA Kernel 6 (propagate_hit<<<(1,1,1),(64,1,1)>>>) on Device 0]
    [Termination of CUDA Kernel 3 (write_size<<<(1,1,1),(1,1,1)>>>) on Device 0]
    [Termination of CUDA Kernel 2 (write_size<<<(1,1,1),(1,1,1)>>>) on Device 0]
    [Termination of CUDA Kernel 1 (write_size<<<(1,1,1),(1,1,1)>>>) on Device 0]
    [Termination of CUDA Kernel 0 (write_size<<<(1,1,1),(1,1,1)>>>) on Device 0]
    (cuda-gdb) 


Termination before prop gives::

    uit anyway? (y or n) y
    [Termination of CUDA Kernel 3 (write_size<<<(1,1,1),(1,1,1)>>>) on Device 0]
    [Termination of CUDA Kernel 2 (write_size<<<(1,1,1),(1,1,1)>>>) on Device 0]
    [Termination of CUDA Kernel 1 (write_size<<<(1,1,1),(1,1,1)>>>) on Device 0]
    [Termination of CUDA Kernel 0 (write_size<<<(1,1,1),(1,1,1)>>>) on Device 0]
    (chroma_env)delta:g4daeview blyth$ 
    i







Something about the bullseye photons causes lock up with only 10 photons

::

    (chroma_env)delta:MockNuWa blyth$ RANGE=0:10 MockNuWa mock007 zz


    cuda-gdb) info cuda threads
      BlockIdx ThreadIdx To BlockIdx ThreadIdx Count         Virtual PC           Filename  Line 
    Kernel 9
    *  (0,0,0)   (0,0,0)     (0,0,0)   (0,0,0)     1 0x0000000100f5a1e8             mesh.h   112 
       (0,0,0)   (1,0,0)     (0,0,0)   (1,0,0)     1 0x0000000134f8ae50 vector_functions.h   239 
       (0,0,0)   (2,0,0)     (0,0,0)   (2,0,0)     1 0x0000000100f5a1e8             mesh.h   112 
       (0,0,0)   (3,0,0)     (0,0,0)   (3,0,0)     1 0x0000000134f8ae50 vector_functions.h   239 
       (0,0,0)   (4,0,0)     (0,0,0)   (4,0,0)     1 0x0000000100f5a1e8             mesh.h   112 
       (0,0,0)   (5,0,0)     (0,0,0)   (5,0,0)     1 0x0000000134f8ae50 vector_functions.h   239 
       (0,0,0)   (6,0,0)     (0,0,0)   (9,0,0)     4 0x0000000100f5a1e8             mesh.h   112 
       (0,0,0)  (10,0,0)     (0,0,0)  (31,0,0)    22 0x0000000100f7e490          kernel.cu   151 
    (cuda-gdb) c
    Continuing.
    ^C
    Program received signal SIGINT, Interrupt.
    intersect_mesh(const float3 * @generic, const float3 * @generic, Geometry * @generic, float * @generic, int) (origin=<value optimized out>, direction=<value optimized out>, g=<value optimized out>, min_distance=<value optimized out>, 
        last_hit_triangle=<value optimized out>) at mesh.h:112
    112     if (blockIdx.x == 0 && threadIdx.x == 0) {
    (cuda-gdb) info cuda threads
      BlockIdx ThreadIdx To BlockIdx ThreadIdx Count         Virtual PC  Filename  Line 
    Kernel 9
    *  (0,0,0)   (0,0,0)     (0,0,0)   (0,0,0)     1 0x0000000100f5a1e8    mesh.h   112 
       (0,0,0)   (1,0,0)     (0,0,0)   (1,0,0)     1 0x0000000134f404a8  linalg.h    55 
       (0,0,0)   (2,0,0)     (0,0,0)   (2,0,0)     1 0x0000000100f5a1e8    mesh.h   112 
       (0,0,0)   (3,0,0)     (0,0,0)   (3,0,0)     1 0x0000000134f404a8  linalg.h    55 
       (0,0,0)   (4,0,0)     (0,0,0)   (4,0,0)     1 0x0000000100f5a1e8    mesh.h   112 
       (0,0,0)   (5,0,0)     (0,0,0)   (5,0,0)     1 0x0000000134f404a8  linalg.h    55 
       (0,0,0)   (6,0,0)     (0,0,0)   (9,0,0)     4 0x0000000100f5a1e8    mesh.h   112 
       (0,0,0)  (10,0,0)     (0,0,0)  (31,0,0)    22 0x0000000100f7e490 kernel.cu   151 
    (cuda-gdb) quit
    A debugging session is active.

        Inferior 1 [process 35265] will be killed.

    Quit anyway? (y or n) y
    [Termination of CUDA Kernel 8 (init_rng<<<(1025,1,1),(64,1,1)>>>) on Device 0]
    [Termination of CUDA Kernel 7 (reduce_kernel_stage1<<<(1,1,1),(512,1,1)>>>) on Device 0]
    [Termination of CUDA Kernel 5 (init_rng<<<(1025,1,1),(64,1,1)>>>) on Device 0]
    [Termination of CUDA Kernel 4 (write_size<<<(1,1,1),(1,1,1)>>>) on Device 0]
    [Termination of CUDA Kernel 9 (propagate_hit<<<(1,1,1),(64,1,1)>>>) on Device 0]
    [Termination of CUDA Kernel 6 (propagate_hit<<<(1,1,1),(64,1,1)>>>) on Device 0]
    [Termination of CUDA Kernel 3 (write_size<<<(1,1,1),(1,1,1)>>>) on Device 0]
    [Termination of CUDA Kernel 2 (write_size<<<(1,1,1),(1,1,1)>>>) on Device 0]
    [Termination of CUDA Kernel 1 (write_size<<<(1,1,1),(1,1,1)>>>) on Device 0]
    [Termination of CUDA Kernel 0 (write_size<<<(1,1,1),(1,1,1)>>>) on Device 0]
    (chroma_env)delta:g4daeview blyth$ 



even 5 is enough
-------------------

::

    (chroma_env)delta:MockNuWa blyth$ RANGE=0:5 MockNuWa mock007 zz


::

    014-11-24 18:24:10,110 INFO    daechromacontext    :137 setup_rng_states using seed 0 
    *** compiler output in /var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/tmpb2cKE5
    [Launch of CUDA Kernel 4 (write_size<<<(1,1,1),(1,1,1)>>>) on Device 0]
    *** compiler output in /var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/tmpO2rITA
    [Launch of CUDA Kernel 5 (init_rng<<<(1025,1,1),(64,1,1)>>>) on Device 0]
    2014-11-24 18:24:15,149 INFO    chroma.gpu.photon_hit:204 nwork 5 step 0 max_steps 30 nsteps 30 
    [Launch of CUDA Kernel 6 (propagate_hit<<<(1,1,1),(64,1,1)>>>) on Device 0]
    ^Cwarning: (Internal error: pc 0x104b257e8 in read in psymtab, but not in symtab.)


    Program received signal SIGINT, Interrupt.
    [Switching focus to CUDA kernel 6, grid 7, block (0,0,0), thread (0,0,0), device 0, sm 1, warp 3, lane 0]
    intersect_mesh(const float3 * @generic, const float3 * @generic, Geometry * @generic, float * @generic, int) (origin=<value optimized out>, direction=<value optimized out>, g=<value optimized out>, min_distance=<value optimized out>, 
        last_hit_triangle=<value optimized out>) at mesh.h:112
    112     if (blockIdx.x == 0 && threadIdx.x == 0) {
    (cuda-gdb) info cuda threads
      BlockIdx ThreadIdx To BlockIdx ThreadIdx Count         Virtual PC  Filename  Line 
    Kernel 6
    *  (0,0,0)   (0,0,0)     (0,0,0)   (0,0,0)     1 0x0000000104b257e8    mesh.h   112 
       (0,0,0)   (1,0,0)     (0,0,0)   (1,0,0)     1 0x0000000122d2dcd0  linalg.h   109 
       (0,0,0)   (2,0,0)     (0,0,0)   (2,0,0)     1 0x0000000104b257e8    mesh.h   112 
       (0,0,0)   (3,0,0)     (0,0,0)   (3,0,0)     1 0x0000000122d2dcd0  linalg.h   109 
       (0,0,0)   (4,0,0)     (0,0,0)   (4,0,0)     1 0x0000000104b257e8    mesh.h   112 
       (0,0,0)   (5,0,0)     (0,0,0)  (31,0,0)    27 0x0000000104b49a90 kernel.cu   151 
    (cuda-gdb) cuda thread <<<0,5>>>
    Unrecognized argument(s).
    (cuda-gdb) cuda thread <<<(0,5)>>>
    Unrecognized argument(s).
    (cuda-gdb) cuda thread (0,5)
    Request cannot be satisfied. CUDA focus unchanged.
    (cuda-gdb) cuda thread (5)
    [Switching focus to CUDA kernel 6, grid 7, block (0,0,0), thread (5,0,0), device 0, sm 1, warp 3, lane 5]
    0x0000000104b49a90  151     if (id >= nthreads)
    (cuda-gdb) cuda thread (6)
    [Switching focus to CUDA kernel 6, grid 7, block (0,0,0), thread (6,0,0), device 0, sm 1, warp 3, lane 6]
    0x0000000104b49a90  151     if (id >= nthreads)
    (cuda-gdb) cuda thread (7)
    [Switching focus to CUDA kernel 6, grid 7, block (0,0,0), thread (7,0,0), device 0, sm 1, warp 3, lane 7]
    0x0000000104b49a90  151     if (id >= nthreads)
    (cuda-gdb) 




actually 2 will do it 
-----------------------

Processing gets bogged down inside fill_state::

    (cuda-gdb) bt
    #0  intersect_mesh(const float3 * @generic, const float3 * @generic, Geometry * @generic, float * @generic, int) (origin=0x3fffbb0, direction=0x3fffbbc, g=0x1000000, min_distance=0x3fffc1c, last_hit_triangle=1263) at mesh.h:74
    #1  0x0000000103976a18 in fill_state(State * @generic, Photon * @generic, Geometry * @generic) (s=<value optimized out>, p=<value optimized out>, g=<value optimized out>) at photon.h:168
    #2  0x000000010398e028 in propagate_hit(int, int, unsigned int * @generic, unsigned int * @generic, curandState * @generic, float3 * @generic, float3 * @generic, float * @generic, float3 * @generic, float * @generic, unsigned int * @generic, int * @generic, float * @generic, int, int, int, Geometry * @generic, int * @generic, int * @generic)<<<(1,1,1),(64,1,1)>>> (first_photon=0, nthreads=2, input_queue=0x700179804, output_queue=0x700179a00, rng_states=0x707ee0000, 
        positions=0x700178800, directions=0x700178a00, wavelengths=0x700178e00, polarizations=0x700178c00, times=0x700179000, histories=0x700179400, last_hit_triangles=0x700179200, weights=0x700179600, max_steps=30, use_weights=0, 
        __val_paramscatter_first=0, __val_paramg=0x700177c00, solid_map=0x7014c0000, solid_id_to_channel_id=0x707be8e00) at kernel.cu:189
    (cuda-gdb) 


Looks like some repetition in the bvh node intersection sequence, on hitting breakpoint::

    cuda-gdb) p i
    $7 = 821365
    (cuda-gdb) p node
    $8 = {lower = {x = -18878.123, y = -801907.875, z = -5291.75146}, upper = {x = -18850.5898, y = -801877.75, z = -5267.04346}, child = 2729356, nchild = 5}
    (cuda-gdb) c
    Continuing.

    Breakpoint 3, intersect_mesh(const float3 * @generic, const float3 * @generic, Geometry * @generic, float * @generic, int) (origin=0x3fffbb0, direction=0x3fffbbc, g=0x1000000, min_distance=0x3fffc1c, last_hit_triangle=1263) at mesh.h:74
    74          Node node = get_node(g, i);
    (cuda-gdb) p node
    $9 = {lower = {x = -18864.0039, y = -801907.625, z = -5290.81006}, upper = {x = -18839.0605, y = -801884.125, z = -5267.04346}, child = 2729361, nchild = 2}
    (cuda-gdb) c
    Continuing.

    Breakpoint 3, intersect_mesh(const float3 * @generic, const float3 * @generic, Geometry * @generic, float * @generic, int) (origin=0x3fffbb0, direction=0x3fffbbc, g=0x1000000, min_distance=0x3fffc1c, last_hit_triangle=1263) at mesh.h:74
    74          Node node = get_node(g, i);
    (cuda-gdb) 
    Continuing.

    Breakpoint 3, intersect_mesh(const float3 * @generic, const float3 * @generic, Geometry * @generic, float * @generic, int) (origin=0x3fffbb0, direction=0x3fffbbc, g=0x1000000, min_distance=0x3fffc1c, last_hit_triangle=1263) at mesh.h:74
    74          Node node = get_node(g, i);
    (cuda-gdb) p node
    $10 = {lower = {x = -18857.8848, y = -801907.625, z = -5290.81006}, upper = {x = -18839.0605, y = -801884.812, z = -5267.04346}, child = 593861, nchild = 0}
    (cuda-gdb) c
    Continuing.

    Breakpoint 3, intersect_mesh(const float3 * @generic, const float3 * @generic, Geometry * @generic, float * @generic, int) (origin=0x3fffbb0, direction=0x3fffbbc, g=0x1000000, min_distance=0x3fffc1c, last_hit_triangle=1263) at mesh.h:74
    74          Node node = get_node(g, i);
    (cuda-gdb) p i
    $11 = 2729357
    (cuda-gdb) p node
    $12 = {lower = {x = -18876.9453, y = -801902, z = -5291.28076}, upper = {x = -18857.4141, y = -801884.125, z = -5267.98486}, child = 594123, nchild = 0}
    (cuda-gdb) c
    Continuing.

    Breakpoint 3, intersect_mesh(const float3 * @generic, const float3 * @generic, Geometry * @generic, float * @generic, int) (origin=0x3fffbb0, direction=0x3fffbbc, g=0x1000000, min_distance=0x3fffc1c, last_hit_triangle=1263) at mesh.h:74
    74          Node node = get_node(g, i);
    (cuda-gdb) p node
    $13 = {lower = {x = -18864.2383, y = -801907.625, z = -5290.81006}, upper = {x = -18850.5898, y = -801884.812, z = -5267.04346}, child = 593860, nchild = 0}
    (cuda-gdb) c
    Continuing.

    Breakpoint 3, intersect_mesh(const float3 * @generic, const float3 * @generic, Geometry * @generic, float * @generic, int) (origin=0x3fffbb0, direction=0x3fffbbc, g=0x1000000, min_distance=0x3fffc1c, last_hit_triangle=1263) at mesh.h:74
    74          Node node = get_node(g, i);
    (cuda-gdb) p node
    $14 = {lower = {x = -18878.123, y = -801907.875, z = -5291.75146}, upper = {x = -18860.2383, y = -801885.75, z = -5270.33789}, child = 591819, nchild = 0}
    (cuda-gdb) c
    Continuing.

    Breakpoint 3, intersect_mesh(const float3 * @generic, const float3 * @generic, Geometry * @generic, float * @generic, int) (origin=0x3fffbb0, direction=0x3fffbbc, g=0x1000000, min_distance=0x3fffc1c, last_hit_triangle=1263) at mesh.h:74
    74          Node node = get_node(g, i);
    (cuda-gdb) p node
    $15 = {lower = {x = -18876.9453, y = -801897.5, z = -5291.28076}, upper = {x = -18857.4141, y = -801877.75, z = -5267.98486}, child = 594122, nchild = 0}
    (cuda-gdb) 


even single photon (the 2nd photon in mock007) will do it
------------------------------------------------------------

::

    chroma_env)delta:MockNuWa blyth$ RANGE=1:2 MockNuWa mock007 zz


    (cuda-gdb) bt
    #0  0x000000012374a0b8 in make_float3 (x=<value optimized out>, y=<value optimized out>, z=<value optimized out>) at vector_functions.h:239
    #1  0x0000000123703e28 in operator-(const float3 * @generic, const float3 * @generic) (a=<value optimized out>, b=<value optimized out>) at linalg.h:55
    #2  0x0000000115519988 in intersect_triangle(const float3 * @generic, const float3 * @generic, const Triangle * @generic, float * @generic) (origin=<value optimized out>, direction=<value optimized out>, triangle=<value optimized out>, 
        distance=<value optimized out>) at intersect.h:31
    #3  0x0000000115529f88 in intersect_mesh(const float3 * @generic, const float3 * @generic, Geometry * @generic, float * @generic, int) (origin=<value optimized out>, direction=<value optimized out>, g=<value optimized out>, 
        min_distance=<value optimized out>, last_hit_triangle=<value optimized out>) at mesh.h:88
    #4  0x0000000115538218 in fill_state(State * @generic, Photon * @generic, Geometry * @generic) (s=<value optimized out>, p=<value optimized out>, g=<value optimized out>) at photon.h:168
    #5  0x000000011554f828 in propagate_hit(int, int, unsigned int * @generic, unsigned int * @generic, curandState * @generic, float3 * @generic, float3 * @generic, float * @generic, float3 * @generic, float * @generic, unsigned int * @generic, int * @generic, float * @generic, int, int, int, Geometry * @generic, int * @generic, int * @generic)<<<(1,1,1),(64,1,1)>>> (first_photon=0, nthreads=1, input_queue=0x700179804, output_queue=0x700179a00, rng_states=0x707ee0000, 
        positions=0x700178800, directions=0x700178a00, wavelengths=0x700178e00, polarizations=0x700178c00, times=0x700179000, histories=0x700179400, last_hit_triangles=0x700179200, weights=0x700179600, max_steps=30, use_weights=0, 
        __val_paramscatter_first=0, __val_paramg=0x700177c00, solid_map=0x7014c0000, solid_id_to_channel_id=0x707be8e00) at kernel.cu:189
    (cuda-gdb) info cu threads
      BlockIdx ThreadIdx To BlockIdx ThreadIdx Count         Virtual PC           Filename  Line 
    Kernel 6
    *  (0,0,0)   (0,0,0)     (0,0,0)   (0,0,0)     1 0x000000012374a0b8 vector_functions.h   239 
       (0,0,0)   (1,0,0)     (0,0,0)  (31,0,0)    31 0x000000011554e290          kernel.cu   151 
    (cuda-gdb) 


    (cuda-gdb) p p
    $1 = {position = {x = -18289.8398, y = -800004.438, z = -7723.5}, direction = {x = 0, y = -0, z = -1}, polarization = {x = 0, y = 0, z = 1}, wavelength = 550, time = 1, weight = 1, history = 0, last_hit_triangle = -1}
    (cuda-gdb) 


skipping the photons with axis aligned directions succeeds to complete
------------------------------------------------------------------------

* :google:`cuda axis aligned rays`

Long kernel launch times as in debugger. Dont cause a problem as running in console mode::

    Launch of CUDA Kernel 5 (init_rng<<<(1025,1,1),(64,1,1)>>>) on Device 0]
    2014-11-24 20:33:36,756 INFO    chroma.gpu.photon_hit:204 nwork 63 step 0 max_steps 30 nsteps 30 
    [Launch of CUDA Kernel 6 (propagate_hit<<<(1,1,1),(64,1,1)>>>) on Device 0]
    2014-11-24 20:34:05,232 WARNING chroma.gpu.photon_hit:238 kernel launch time 28.4752539062 > max_time 4.0 : ABORTING 
    2014-11-24 20:34:05,232 INFO    chroma.gpu.photon_hit:242 step 0 propagate_hit_kernel times  [28.47525390625] 

Running CUDAGDB costs a factor of 25, here the same work without it::

    2014-11-24 20:44:59,730 INFO    chroma.gpu.photon_hit:204 nwork 63 step 0 max_steps 30 nsteps 30 
    2014-11-24 20:45:01,007 INFO    chroma.gpu.photon_hit:242 step 0 propagate_hit_kernel times  [1.275962890625] 


push the work up to 628 hardly touches the time
--------------------------------------------------

::

    "results":  {
        "nwork":    628,
        "name": "propagate_hit",
        "mintime":  1.550168,
        "npass":    1,
        "nphotons": 628,
        "nabort":   0,
        "nsmall":   8192,
        "nlaunch":  1,
        "maxtime":  1.550168,
        "COLUMNS":  "name:s,nphotons:i,nwork:i,nsmall:i,npass:i,nabort:i,nlaunch:i,tottime:f,maxtime:f,mintime:f",
        "tottime":  1.550168
    },





gdb compilation fail
----------------------

::

     File "daechromacontext.py", line 158, in setup_gpu_detector
        return GPUDetector( self.chroma_geometry )
      File "/usr/local/env/chroma_env/src/chroma/chroma/gpu/detector.py", line 18, in __init__
        GPUGeometry.__init__(self, detector, wavelengths=wavelengths, print_usage=False)
      File "/usr/local/env/chroma_env/src/chroma/chroma/gpu/geometry.py", line 37, in __init__
        surface_struct_size = characterize.sizeof('Surface', geometry_source)
      File "<string>", line 2, in sizeof
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/tools.py", line 404, in context_dependent_memoize
        result = func(*args)
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/characterize.py", line 40, in sizeof
        """ % (preamble, type_name), no_extern_c=True)
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/compiler.py", line 262, in __init__
        arch, code, cache_dir, include_dirs)
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/compiler.py", line 252, in compile
        return compile_plain(source, options, keep, nvcc, cache_dir)
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/compiler.py", line 134, in compile_plain
        cmdline, stdout=stdout.decode("utf-8"), stderr=stderr.decode("utf-8"))
    pycuda.driver.CompileError: nvcc compilation of /var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/tmpdYkTX5/kernel.cu failed
    [command: nvcc --cubin -g -G -arch sm_30 -m64 -I/usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/cuda --keep kernel.cu]
    [stderr:
    kernel.cu(45): warning: variable "NCHILD_MASK" was declared but never referenced


    Compilation terminated.
    ]
    [Context Pop of context 0x115700400 on Device 0]





Debug Example
---------------

::

    cd ~/e/chroma/chroma_camera
    ./3199.sh

    (chroma_env)delta:chroma_camera blyth$ cat 3199.sh
    #!/bin/bash -l
    chroma-
    cd $ENV_HOME/chroma/chroma_camera
    cuda-gdb --args python -m pycuda.debug simplecamera.py -s3199 -d3 -f10 --eye=0,1,0 --lookat=0,0,0 -G -o 3199_000.png



::


    [Launch of CUDA Kernel 0 (write_size<<<(1,1,1),(1,1,1)>>>) on Device 0]
    *** compiler output in /var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/tmpsMRB8a
    [Launch of CUDA Kernel 1 (write_size<<<(1,1,1),(1,1,1)>>>) on Device 0]
    *** compiler output in /var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/tmpPoL2DR
    [Launch of CUDA Kernel 2 (write_size<<<(1,1,1),(1,1,1)>>>) on Device 0]
    INFO:chroma:Optimization: Sufficient memory to move triangles onto GPU
    INFO:chroma:Optimization: Sufficient memory to move vertices onto GPU
    INFO:chroma:device usage:
    ----------
    nodes             2.8M  44.7M
    total                   44.7M
    ----------
    device total             2.1G
    device used            233.4M
    device free              1.9G

    *** compiler output in /var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/tmpAmcMTC
    *** compiler output in /var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/tmp6ewtAR
    *** compiler output in /var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/tmpLWLa8A
    *** compiler output in /var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/tmpjbg9nG
    [Launch of CUDA Kernel 3 (fill<<<(64,1,1),(128,1,1)>>>) on Device 0]
    [Launch of CUDA Kernel 4 (fill<<<(64,1,1),(128,1,1)>>>) on Device 0]
    [Launch of CUDA Kernel 5 (render<<<(4801,1,1),(64,1,1)>>>) on Device 0]
    ^Cwarning: (Internal error: pc 0x104199240 in read in psymtab, but not in symtab.)


    Program received signal SIGINT, Interrupt.
    [Switching focus to CUDA kernel 5, grid 6, block (48,0,0), thread (32,0,0), device 0, sm 1, warp 8, lane 0]
    render(int, float3 * @generic, float3 * @generic, Geometry * @generic, unsigned int, unsigned int * @generic, float * @generic, unsigned int * @generic, float4 * @generic)<<<(4801,1,1),(64,1,1)>>> (nthreads=307200, _origin=0x707680000, _direction=0x707a20000, 
        __val_paramg=0x700181800, alpha_depth=3, pixels=0x7090c0000, _dx=0x707dc0000, dxlen=0x708f80000, _color=0x708160000) at kernel.cu:144
    144         if (n < 1) {
    (cuda-gdb) info cuda threads
       BlockIdx ThreadIdx To BlockIdx ThreadIdx Count         Virtual PC           Filename  Line 
    Kernel 5
    *  (48,0,0)  (32,0,0)    (48,0,0)  (47,0,0)    16 0x0000000104199240          kernel.cu   144 
       (48,0,0)  (48,0,0)    (48,0,0)  (48,0,0)     1 0x0000000131e17980           linalg.h    39 
       (48,0,0)  (49,0,0)    (48,0,0)  (63,0,0)    15 0x0000000104199240          kernel.cu   144 
       (56,0,0)   (0,0,0)    (56,0,0)  (15,0,0)    16 0x0000000104199240          kernel.cu   144 
       (56,0,0)  (16,0,0)    (56,0,0)  (16,0,0)     1 0x00000001041868b8         geometry.h    37 
       (56,0,0)  (17,0,0)    (56,0,0)  (31,0,0)    15 0x0000000104199240          kernel.cu   144 
       (63,0,0)  (32,0,0)    (63,0,0)  (47,0,0)    16 0x0000000104199240          kernel.cu   144 
       (63,0,0)  (48,0,0)    (63,0,0)  (48,0,0)     1 0x0000000131e2a4d8           matrix.h   222 
       (63,0,0)  (49,0,0)    (63,0,0)  (63,0,0)    15 0x0000000104199240          kernel.cu   144 
       (71,0,0)   (0,0,0)    (71,0,0)  (15,0,0)    16 0x0000000104199240          kernel.cu   144 
       (71,0,0)  (16,0,0)    (71,0,0)  (16,0,0)     1 0x0000000131e45828   math_functions.h  8215 
       (71,0,0)  (17,0,0)    (71,0,0)  (31,0,0)    15 0x0000000104199240          kernel.cu   144 
       (78,0,0)  (32,0,0)    (78,0,0)  (47,0,0)    16 0x0000000104199240          kernel.cu   144 
       (78,0,0)  (48,0,0)    (78,0,0)  (48,0,0)     1 0x0000000104156310        intersect.h    56 
      ...
      (199,0,0)   (3,0,0)   (199,0,0)   (3,0,0)     1 0x0000000104155518        intersect.h    33 
      (199,0,0)   (4,0,0)   (199,0,0)   (5,0,0)     2 0x00000001041991c0          kernel.cu    90 
      (199,0,0)   (6,0,0)   (199,0,0)   (6,0,0)     1 0x0000000104155518        intersect.h    33 
      (199,0,0)   (7,0,0)   (199,0,0)   (8,0,0)     2 0x0000000104199230          kernel.cu    90 
      (199,0,0)   (9,0,0)   (199,0,0)   (9,0,0)     1 0x00000001041991c0          kernel.cu    90 
    ---Type <return> to continue, or q <return> to quit---q
    Quit
    (cuda-gdb) list
    139             } // loop over children, starting with first_child
    140
    141         } // while nodes on stack
    142         
    143
    144         if (n < 1) {
    145             pixels[id] = 0;
    146             return;
    147         }
    148
    (cuda-gdb) p n
    $1 = 3
    (cuda-gdb) p id
    $2 = 3104
    (cuda-gdb) 





info cuda threads
-------------------

From the manual::

    CUDA-GDB provides an additional command (info cuda threads) which displays 
    a summary of all CUDA threads that are currently resident on the GPU.  CUDA 
    threads are specified using the same syntax as described in Section 4.6 and are 
    summarized by grouping all contiguous threads that are stopped at the same 
    program location.  A sample display can be seen below: 
     
    <<<(0,0),(0,0,0)>>> ... <<<(0,0),(31,0,0)>>>  
    GPUBlackScholesCallPut () at blackscholes.cu:73 
    <<<(0,0),(32,0,0)>>> ... <<<(119,0),(0,0,0)>>> 
     GPUBlackScholesCallPut () at blackscholes.cu:72 
     
    The above example shows 32 threads (a warp) that have been advanced to line 73 of 
    blackscholes.cu, and the remainder of the resident threads stopped at line 72. 
    Since this summary only shows thread coordinates for the start and end range, it 
    may be unclear how many threads or blocks are actually within the displayed range.  
    This can be checked by printing the values of gridDim and/or blockDim. 
    CUDA-GDB also has the ability to display a full list of each individual thread that is 
    currently resident on the GPU by using the info cuda threads all command. 



kernel debug
-------------


::

    simon:cuda blyth$ grep STACK_SIZE *.*
    mesh.h:#define STACK_SIZE 1000
    mesh.h:    unsigned int child_ptr_stack[STACK_SIZE];
    mesh.h:    unsigned int nchild_ptr_stack[STACK_SIZE];
    mesh.h:     if (curr >= STACK_SIZE) {
    render.cu:    unsigned int child_ptr_stack[STACK_SIZE];
    render.cu:    unsigned int nchild_ptr_stack[STACK_SIZE];
    render.cu:          //if (curr >= STACK_SIZE) {


::

    (998,0,0)  (18,0,0)   (998,0,0)  (18,0,0)     1 0x0000000104199230          kernel.cu    90 
    (998,0,0)  (19,0,0)   (998,0,0)  (28,0,0)    10 0x0000000104151750        intersect.h    71 
    (998,0,0)  (29,0,0)   (998,0,0)  (29,0,0)     1 0x0000000104199230          kernel.cu    90 
    ---Type <return> to continue, or q <return> to quit---q
    Quit
    (cuda-gdb) list
    139             } // loop over children, starting with first_child
    140
    141         } // while nodes on stack
    142         
    143
    144         if (n < 1) {
    145             pixels[id] = 0;
    146             return;
    147         }
    148
    (cuda-gdb) p origin
    $4 = {x = -16566.293, y = -801040.938, z = -8842.5}
    (cuda-gdb) p direction
    $5 = {x = 0.740086973, y = -0.669951439, z = 0.0586207509}
    (cuda-gdb) p n
    $6 = 3
    (cuda-gdb) p distance
    $7 = 9562.18848
    (cuda-gdb) p STACK_SIZE
    No symbol "STACK_SIZE" in current context.
    (cuda-gdb) p child_ptr_stack
    $8 = {139, 1644, 509, 6978, 1898, 30875, 622018, 622033, 622063, 622078, 622178, 622193, 622208, 622343, 622493, 622523, 622538, 622553, 622568, 622583, 1692132, 1692147, 1692162, 1692177, 1692192, 1692207, 1692222, 1692514, 1692529, 1418963, 1419182, 1419197, 
      2109819, 2111777, 2111779, 2111794, 2111809, 2111824, 2095481, 1734707, 2653458, 2653473, 2653741, 2653756, 2653771, 2653996, 2656458, 2656473, 2656488, 2656713, 4281938739, 4281873459, 4281938995, 4281873203, 4281873458, 4282004788, 4282004788, 4281939251, 
      4281938996, 4282004532, 4281938996, 4282004532, 4281939253, 4281873202, 4281873202, 4281873203, 4281938995, 4281873458, 4281938740, 4281873202, 4281938739, 4282004532, 4281938995, 4282004533, 4282004787, 4282004788, 4282004788, 4282004531, 4281938995, 
      4281873458, 4281873459, 4281939251, 4281938996, 4282004532, 4281938996, 4281938739, 4281873460, 4281939253, 4282004532, 4282004789, 4282070325, 4282004789, 4282005044, 4282004788, 4282004787, 4281873203, 4281938739, 4281938995, 4281938995, 4281938995, 
      4282004532, 4281873458, 4281938739, 4282070325, 4282004788, 4282004532, 4281939251, 4282004788, 4281938996, 4282070324, 4282004788, 4281939251, 4281938996, 4281939251, 4281939252, 4282004532, 4281939253, 4281938996, 4281873460, 4282005046, 4282070325, 
      4282004789, 4282070325, 4282004789, 4282005044, 4282070581, 4282070580, 4282004532, 4281938995, 4282004532, 4281939251, 4282004532, 4281938996, 4281938995, 4281938995, 4282004788, 4282070324, 4282070325, 4282070581, 4282070580, 4282070325, 4282004788, 
      4282070325, 4281938996, 4282004532, 4281939253, 4282004788, 4282004789, 4282005044, 4281938996, 4281939251, 4282070580, 4282005045, 4282070837, 4282070582, 4282136118, 4282070582, 4282070325, 4282005046, 4281873204, 4281938995, 4281807668, 4281938994, 
      4281872947, 4281938739, 4281873203, 4281938739, 4282070068, 4281938996, 4282004532, 4281938740, 4282004787, 4281938997, 4282070324, 4281938997, 4281938997, 4282004787, 4281938740, 4281938994, 4281873458, 4281873204, 4281939251, 4281938996, 4282004789, 
      4282005044, 4281939253, 4281939251, 4282004787, 4282004789, 4282070580, 4282004790, 4281938739, 4281873203, 4282004532, 4281938739, 4282004787, 4281873461, 4281938995, 4281873204, 4282004790...}
    (cuda-gdb) p nchild_ptr_stack
    $9 = {2, 2, 7, 2, 6, 3, 15, 15, 15, 15, 15, 15, 15, 8, 15 <repeats 17 times>, 2, 15, 2, 15 <repeats 16 times>, 4282267703, 4282333495, 4282333241, 4282333752, 4282399033, 4282070581, 4282070581, 4282136118, 4282070581, 4282136118, 4282070581, 4282136118, 
      4282070580, 4282202166, 4282267447, 4282136374, 4282201911, 4282136374, 4282201910, 4282201911, 4282202167, 4282070582, 4282070837, 4282136374, 4282136374, 4282136375, 4282136374, 4282070582, 4282136118, 4282267703, 4282202168, 4282201911, 4282136375, 
      4282202166, 4282202167, 4282202423, 4282202167, 4282070581, 4282136118, 4282136373, 4282201911, 4282136374, 4282136374, 4282070581, 4282136117, 4282201911, 4282136374, 4282267703, 4282202167, 4282267704, 4282202166, 4282201911, 4282136374, 4282136118, 
      4282070839, 4282201911, 4282136375, 4282136630, 4282136375, 4282136373, 4282136374, 4282202167, 4282202166, 4282267960, 4282267959, 4282202168, 4282267704, 4282202168, 4282267703, 4282070837, 4282136118, 4282136374, 4282201911, 4282136374, 4282201910, 
      4282136117, 4282136374, 4282267703, 4282267703, 4282202167, 4282202167, 4282267703, 4282202166, 4282267704, 4282202167, 4282201910, 4282136375, 4282201911, 4282136631, 4282202166, 4282202167, 4282136630, 4282136374, 4282267960, 4282267959, 4282202168, 
      4282202423, 4282202168, 4282267704, 4282267961, 4282333496, 4282201910, 4282136374, 4282202167, 4282201910, 4282201911, 4282202166, 4282201911, 4282136373, 4282202167, 4282267704, 4282267959, 4282333240, 4282267960, 4282333496, 4282202167, 4282267703, 
      4282136375, 4282136630, 4282202167, 4282202423, 4282202168, 4282267703, 4282136375, 4282201911, 4282267704, 4282202424, 4282333496, 4282267961, 4282268216, 4282267961, 4282267959, 4282267960, 4282136118, 4282201910, 4282070325, 4282136118, 4282070583, 
      4282136373, 4282070583, 4282201910, 4282267703, 4282202168, 4282267702, 4282136376, 4282201911, 4282136375, 4282267703, 4282201911, 4282136630, 4282136375, 4282136373, 4282070838, 4282136375, 4282136373, 4282201912, 4282202166, 4282267959, 4282267705, 
      4282267958, 4282202168, 4282202168, 4282202166...}
    (cuda-gdb) 



    (cuda-gdb) p sg
    $11 = {vertices = 0x706600000, triangles = 0x704980000, material_codes = 0x700240000, colors = 0x700bc0000, primary_nodes = 0x701ec0000, extra_nodes = 0x202b00000, materials = 0x70015b000, surfaces = 0x700181600, world_origin = {x = -2400000, y = -2400000, 
        z = -2400000}, world_scale = 73.2444229, nprimary_nodes = 2794974}
    (cuda-gdb) p g
    $12 = <value optimized out>
    (cuda-gdb) p id
    $13 = 56864
    (cuda-gdb) p root
    $14 = {lower = {x = -2400000, y = -2400000, z = -2400000}, upper = {x = 2400073.5, y = 2400073.5, z = 2400073.5}, child = 1, nchild = 2}
    (cuda-gdb) p neg_origin_inv_dir
    $15 = {x = 22384.252, y = -1195670.12, z = 150842.484}
    (cuda-gdb) p inv_dir
    $16 = {x = 1.35119259, y = -1.4926455, z = 17.0588055}
    (cuda-gdb) p count
    $17 = <value optimized out>
    (cuda-gdb) p tri_count
    $18 = <value optimized out>
    (cuda-gdb) p alpha_depth
    $19 = 3
    (cuda-gdb) p _dx
    $20 = (@generic float * @parameter) 0x707dc0000
    (cuda-gdb) p dx
    $21 = <value optimized out>
    (cuda-gdb) p _color
    $22 = (@generic float4 * @parameter) 0x708160000
    (cuda-gdb) p color_a
    $23 = (@generic float4 * @register) 0x7083fa600




::

    kernel.cu    90 
       (998,0,0)   (9,0,0)   (998,0,0)  (13,0,0)     5 0x0000000104151750        intersect.h    71 
       (998,0,0)  (14,0,0)   (998,0,0)  (14,0,0)     1 0x0000000104199230          kernel.cu    90 
       (998,0,0)  (15,0,0)   (998,0,0)  (15,0,0)     1 0x0000000104151750        intersect.h    71 
       (998,0,0)  (16,0,0)   (998,0,0)  (17,0,0)     2 0x0000000104199240          kernel.cu   144 
       (998,0,0)  (18,0,0)   (998,0,0)  (18,0,0)     1 0x0000000104199230          kernel.cu    90 
       (998,0,0)  (19,0,0)   (998,0,0)  (28,0,0)    10 0x0000000104151750        intersect.h    71 
       (998,0,0)  (29,0,0)   (998,0,0)  (29,0,0)     1 0x0000000104199230          kernel.cu    90 
    ---Type <return> to continue, or q <return> to quit--- q
    Quit
    (cuda-gdb) info cuda state
    Unrecognized option: 'state'.
    (cuda-gdb) bt
    #0  render(int, float3 * @generic, float3 * @generic, Geometry * @generic, unsigned int, unsigned int * @generic, float * @generic, unsigned int * @generic, float4 * @generic)<<<(4801,1,1),(64,1,1)>>> (nthreads=307200, _origin=0x707680000, _direction=0x707a20000, 
        __val_paramg=0x700181800, alpha_depth=3, pixels=0x7090c0000, _dx=0x707dc0000, dxlen=0x708f80000, _color=0x708160000) at kernel.cu:144
    (cuda-gdb) list
    149         dxlen[id] = n;
    150
    151         float scale = 1.0f;
    152         float fr = 0.0f;
    153         float fg = 0.0f;
    154         float fb = 0.0f;
    155         for (int i=0; i < n; i++) {
    156             float alpha = color_a[i].w;
    157
    158             fr += scale*color_a[i].x*alpha;
    (cuda-gdb) c
    Continuing.
    ^C
    Program received signal SIGINT, Interrupt.
    [Switching focus to CUDA kernel 5, grid 6, block (1623,0,0), thread (32,0,0), device 0, sm 1, warp 4, lane 0]
    render(int, float3 * @generic, float3 * @generic, Geometry * @generic, unsigned int, unsigned int * @generic, float * @generic, unsigned int * @generic, float4 * @generic)<<<(4801,1,1),(64,1,1)>>> (nthreads=307200, _origin=0x707680000, _direction=0x707a20000, 
        __val_paramg=0x700181800, alpha_depth=3, pixels=0x7090c0000, _dx=0x707dc0000, dxlen=0x708f80000, _color=0x708160000) at kernel.cu:144
    144         if (n < 1) {
    (cuda-gdb) bt
    #0  render(int, float3 * @generic, float3 * @generic, Geometry * @generic, unsigned int, unsigned int * @generic, float * @generic, unsigned int * @generic, float4 * @generic)<<<(4801,1,1),(64,1,1)>>> (nthreads=307200, _origin=0x707680000, _direction=0x707a20000, 
        __val_paramg=0x700181800, alpha_depth=3, pixels=0x7090c0000, _dx=0x707dc0000, dxlen=0x708f80000, _color=0x708160000) at kernel.cu:144
    (cuda-gdb) p id
    $26 = 103904
    (cuda-gdb) thread
    Focus not set on any host thread.
    (cuda-gdb) print blockIdx
    $27 = {x = 1623, y = 0, z = 0}
    (cuda-gdb) print threadIdx
    $28 = {x = 32, y = 0, z = 0}
    (cuda-gdb) print blockDim
    $29 = {x = 64, y = 1, z = 1}
    (cuda-gdb) print gridDim
    $30 = {x = 4801, y = 1, z = 1}
    (cuda-gdb) p nthreads
    $31 = 307200
    (cuda-gdb) thread <<<0>>>
    A syntax error in expression, near `<<<0>>>'.
    (cuda-gdb) c
    Continuing.
    ^C[New Thread 0x297b of process 6669]
    warning: (Internal error: pc 0x10412b390 in read in psymtab, but not in symtab.)


    Program received signal SIGINT, Interrupt.
    [Switching focus to CUDA kernel 5, grid 6, block (2400,0,0), thread (0,0,0), device 0, sm 0, warp 12, lane 0]
    0x000000010412b390 in intersect_node(Geometry * @generic, const float3 * @generic, const float3 * @generic, const Node * @generic, const float) (g=0x1000000, neg_origin_inv_dir=<value optimized out>, inv_dir=<value optimized out>, node=<value optimized out>, 
        min_distance=<value optimized out>) at mesh.h:32
    32              return false;
    (cuda-gdb) list
    27                  return false;
    28
    29              return true;
    30          }
    31          else {
    32              return false;
    33          }
    34      }
    35
    36      /* Finds the intersection between a ray and `geometry`. If the ray does
    (cuda-gdb) p id
    No symbol "id" in current context.
    (cuda-gdb) bt
    #0  0x000000010412b390 in intersect_node(Geometry * @generic, const float3 * @generic, const float3 * @generic, const Node * @generic, const float) (g=0x1000000, neg_origin_inv_dir=<value optimized out>, inv_dir=<value optimized out>, node=<value optimized out>, 
        min_distance=<value optimized out>) at mesh.h:32
    #1  0x0000000104198158 in render(int, float3 * @generic, float3 * @generic, Geometry * @generic, unsigned int, unsigned int * @generic, float * @generic, unsigned int * @generic, float4 * @generic)<<<(4801,1,1),(64,1,1)>>> (nthreads=307200, _origin=0x707680000, 
        _direction=0x707a20000, __val_paramg=0x700181800, alpha_depth=3, pixels=0x7090c0000, _dx=0x707dc0000, dxlen=0x708f80000, _color=0x708160000) at kernel.cu:94
    (cuda-gdb) u
    warning: (Internal error: pc 0x104198158 in read in psymtab, but not in symtab.)

    render(int, float3 * @generic, float3 * @generic, Geometry * @generic, unsigned int, unsigned int * @generic, float * @generic, unsigned int * @generic, float4 * @generic)<<<(4801,1,1),(64,1,1)>>> (nthreads=307200, _origin=0x707680000, _direction=0x707a20000, 
        __val_paramg=0x700181800, alpha_depth=3, pixels=0x7090c0000, _dx=0x707dc0000, dxlen=0x708f80000, _color=0x708160000) at kernel.cu:90
    90              for (unsigned int i=first_child; i < first_child + nchild; i++) {
    (cuda-gdb) p id
    $32 = 153600
    (cuda-gdb) p first_child
    $33 = 1671291
    (cuda-gdb) p nchild
    $34 = 15
    (cuda-gdb) p curr
    $35 = 19
    (cuda-gdb) p g
    $36 = (Geometry * @generic) 0x1000000
    (cuda-gdb) p node
    $37 = {lower = {x = -17725.25, y = -802099.625, z = -7910.5}, upper = {x = -17578.75, y = -801953.125, z = -7690.75}, child = 268267, nchild = 0}
    (cuda-gdb) 




::

    (cuda-gdb) info threads
      7 Thread 0x1553 of process 6669  0x00007fff8a183a1a in mach_msg_trap () from /usr/lib/system/libsystem_kernel.dylib
      6 Thread 0x2703 of process 6669  0x00007fff8a183a1a in mach_msg_trap () from /usr/lib/system/libsystem_kernel.dylib
      3 Thread 0x1623 of process 6669  0x00007fff8a188662 in kevent64 () from /usr/lib/system/libsystem_kernel.dylib
      2 Thread 0x1807 of process 6669  0x00007fff8a187a3a in __semwait_signal () from /usr/lib/system/libsystem_kernel.dylib
    * 1 Thread 0x2303 of process 6669  0x0000000103bce666 in cudbgMain () from /Library/Frameworks/CUDA.framework/Versions/A/Libraries/libcuda_310.40.25_mercury.dylib
    (cuda-gdb) bt
    #0  render(int, float3 * @generic, float3 * @generic, Geometry * @generic, unsigned int, unsigned int * @generic, float * @generic, unsigned int * @generic, float4 * @generic)<<<(4801,1,1),(64,1,1)>>> (nthreads=307200, _origin=0x707680000, _direction=0x707a20000, 
        __val_paramg=0x700181800, alpha_depth=3, pixels=0x7090c0000, _dx=0x707dc0000, dxlen=0x708f80000, _color=0x708160000) at kernel.cu:144
    (cuda-gdb) thread 1
    [Switching to thread 1 (Thread 0x2303 of process 6669)]#0  0x0000000103bce666 in cudbgMain () from /Library/Frameworks/CUDA.framework/Versions/A/Libraries/libcuda_310.40.25_mercury.dylib
    (cuda-gdb) bt
    #0  0x0000000103bce666 in cudbgMain () from /Library/Frameworks/CUDA.framework/Versions/A/Libraries/libcuda_310.40.25_mercury.dylib
    #1  0x0000000103b730c9 in cuGraphicsGLRegisterImage () from /Library/Frameworks/CUDA.framework/Versions/A/Libraries/libcuda_310.40.25_mercury.dylib
    #2  0x0000000103a8a1f3 in cuGraphicsGLRegisterImage () from /Library/Frameworks/CUDA.framework/Versions/A/Libraries/libcuda_310.40.25_mercury.dylib
    #3  0x0000000103b75e66 in cuGraphicsGLRegisterImage () from /Library/Frameworks/CUDA.framework/Versions/A/Libraries/libcuda_310.40.25_mercury.dylib
    #4  0x0000000103b75fd1 in cuGraphicsGLRegisterImage () from /Library/Frameworks/CUDA.framework/Versions/A/Libraries/libcuda_310.40.25_mercury.dylib
    #5  0x0000000103b61c1e in cuGraphicsGLRegisterImage () from /Library/Frameworks/CUDA.framework/Versions/A/Libraries/libcuda_310.40.25_mercury.dylib
    #6  0x0000000103b61f0d in cuGraphicsGLRegisterImage () from /Library/Frameworks/CUDA.framework/Versions/A/Libraries/libcuda_310.40.25_mercury.dylib
    #7  0x0000000103b571e5 in cuGraphicsGLRegisterImage () from /Library/Frameworks/CUDA.framework/Versions/A/Libraries/libcuda_310.40.25_mercury.dylib
    #8  0x0000000103a7eb51 in cuGraphicsGLRegisterImage () from /Library/Frameworks/CUDA.framework/Versions/A/Libraries/libcuda_310.40.25_mercury.dylib
    #9  0x0000000103a8224f in cuGraphicsGLRegisterImage () from /Library/Frameworks/CUDA.framework/Versions/A/Libraries/libcuda_310.40.25_mercury.dylib
    #10 0x0000000103a7105d in cuMemcpyDtoH_v2 () from /Library/Frameworks/CUDA.framework/Versions/A/Libraries/libcuda_310.40.25_mercury.dylib
    #11 0x0000000101816ba4 in (anonymous namespace)::py_memcpy_dtoh(pycudaboost::python::api::object, unsigned long long) () from /usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/_driver.so
    #12 0x0000000101839e1d in pycudaboost::python::detail::caller_arity<2u>::impl<void (*)(pycudaboost::python::api::object, unsigned long long), pycudaboost::python::default_call_policies, pycudaboost::mpl::vector3<void, pycudaboost::python::api::object, unsigned long long> >::operator()(_object*, _object*) () from /usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/_driver.so
    #13 0x0000000101869d4e in pycudaboost::python::objects::function::call(_object*, _object*) const () from /usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/_driver.so
    #14 0x000000010186bf7a in pycudaboost::detail::function::void_function_ref_invoker0<pycudaboost::python::objects::(anonymous namespace)::bind_return, void>::invoke(pycudaboost::detail::function::function_buffer&) ()
       from /usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/_driver.so
    #15 0x00000001018799f3 in pycudaboost::python::detail::exception_handler::operator()(pycudaboost::function0<void> const&) const () from /usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/_driver.so
    #16 0x0000000101851f76 in pycudaboost::detail::function::function_obj_invoker2<pycudaboost::_bi::bind_t<bool, pycudaboost::python::detail::translate_exception<pycuda::error, void (*)(pycuda::error const&)>, pycudaboost::_bi::list3<pycudaboost::arg<1>, pycudaboost::arg<2>, pycudaboost::_bi::value<void (*)(pycuda::error const&)> > >, bool, pycudaboost::python::detail::exception_handler const&, pycudaboost::function0<void> const&>::invoke(pycudaboost::detail::function::function_buffer&, pycudaboost::python::detail::exception_handler const&, pycudaboost::function0<void> const&) () from /usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/_driver.so
    #17 0x0000000101879783 in pycudaboost::python::handle_exception_impl(pycudaboost::function0<void>) () from /usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/_driver.so
    #18 0x000000010186b963 in function_call () from /usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/_driver.so
    #19 0x0000000100011665 in PyObject_Call () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #20 0x00000001000a60b4 in PyEval_EvalFrameEx () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #21 0x00000001000a2076 in PyEval_EvalCodeEx () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #22 0x00000001000a8f36 in fast_function () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #23 0x00000001000a528b in PyEval_EvalFrameEx () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #24 0x00000001000a8ed2 in fast_function () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #25 0x00000001000a528b in PyEval_EvalFrameEx () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #26 0x00000001000a2076 in PyEval_EvalCodeEx () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #27 0x00000001000a8f36 in fast_function () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #28 0x00000001000a528b in PyEval_EvalFrameEx () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #29 0x00000001000a8ed2 in fast_function () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #30 0x00000001000a528b in PyEval_EvalFrameEx () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #31 0x00000001000a2076 in PyEval_EvalCodeEx () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #32 0x00000001000a19a6 in PyEval_EvalCode () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #33 0x00000001000c9611 in PyRun_FileExFlags () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #34 0x000000010009dfe6 in builtin_execfile () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #35 0x00000001000a4010 in PyEval_EvalFrameEx () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #36 0x00000001000a2076 in PyEval_EvalCodeEx () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #37 0x00000001000a6752 in PyEval_EvalFrameEx () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #38 0x00000001000a2076 in PyEval_EvalCodeEx () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #39 0x00000001000a8f36 in fast_function () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #40 0x00000001000a528b in PyEval_EvalFrameEx () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #41 0x00000001000a2076 in PyEval_EvalCodeEx () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #42 0x00000001000350c6 in function_call () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #43 0x0000000100011665 in PyObject_Call () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #44 0x00000001000dd131 in RunModule () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #45 0x00000001000dcc12 in Py_Main () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #46 0x00007fff904935fd in start () from /usr/lib/system/libdyld.dylib
    #47 0x00007fff904935fd in start () from /usr/lib/system/libdyld.dylib
    #48 0x0000000000000000 in ?? ()
    (cuda-gdb) 


::

    (cuda-gdb) c
    Continuing.
    ^Cwarning: (Internal error: pc 0x104199240 in read in psymtab, but not in symtab.)


    Program received signal SIGINT, Interrupt.
    [Switching focus to CUDA kernel 5, grid 6, block (2403,0,0), thread (32,0,0), device 0, sm 1, warp 2, lane 0]
    render(int, float3 * @generic, float3 * @generic, Geometry * @generic, unsigned int, unsigned int * @generic, float * @generic, unsigned int * @generic, float4 * @generic)<<<(4801,1,1),(64,1,1)>>> (nthreads=307200, _origin=0x707680000, _direction=0x707a20000, 
        __val_paramg=0x700181800, alpha_depth=3, pixels=0x7090c0000, _dx=0x707dc0000, dxlen=0x708f80000, _color=0x708160000) at kernel.cu:144
    144         if (n < 1) {
    (cuda-gdb) cuda device sm warp lane block thread
    block (2403,0,0), thread (32,0,0), device 0, sm 1, warp 2, lane 0
    (cuda-gdb) cuda kernel block thread
    kernel 5, block (2403,0,0), thread (32,0,0)
    (cuda-gdb) cuda kernel
    kernel 5
    (cuda-gdb) cuda device 0 sm 1 warp 2 lane 3
    [Switching focus to CUDA kernel 5, grid 6, block (2403,0,0), thread (35,0,0), device 0, sm 1, warp 2, lane 3]
    144         if (n < 1) {
    (cuda-gdb) list
    139             } // loop over children, starting with first_child
    140
    141         } // while nodes on stack
    142         
    143
    144         if (n < 1) {
    145             pixels[id] = 0;
    146             return;
    147         }
    148
    (cuda-gdb) p id
    $40 = 153827
    (cuda-gdb) p n
    $41 = 3
    (cuda-gdb) c
    Continuing.



Stopping when the fans spin up, is pointing at device to host memcopy::

    (cuda-gdb) info threads
      7 Thread 0x1553 of process 6669  0x00007fff8a183a1a in mach_msg_trap () from /usr/lib/system/libsystem_kernel.dylib
      6 Thread 0x2703 of process 6669  0x00007fff8a183a1a in mach_msg_trap () from /usr/lib/system/libsystem_kernel.dylib
      3 Thread 0x1623 of process 6669  0x00007fff8a188662 in kevent64 () from /usr/lib/system/libsystem_kernel.dylib
      2 Thread 0x1807 of process 6669  0x00007fff8a187a3a in __semwait_signal () from /usr/lib/system/libsystem_kernel.dylib
    * 1 Thread 0x2303 of process 6669  0x0000000103b757a4 in cuGraphicsGLRegisterImage () from /Library/Frameworks/CUDA.framework/Versions/A/Libraries/libcuda_310.40.25_mercury.dylib
    (cuda-gdb) thread 1
    [Switching to thread 1 (Thread 0x2303 of process 6669)]#0  0x0000000103b757a4 in cuGraphicsGLRegisterImage () from /Library/Frameworks/CUDA.framework/Versions/A/Libraries/libcuda_310.40.25_mercury.dylib
    (cuda-gdb) bt
    #0  0x0000000103b757a4 in cuGraphicsGLRegisterImage () from /Library/Frameworks/CUDA.framework/Versions/A/Libraries/libcuda_310.40.25_mercury.dylib
    #1  0x0000000103b75ce0 in cuGraphicsGLRegisterImage () from /Library/Frameworks/CUDA.framework/Versions/A/Libraries/libcuda_310.40.25_mercury.dylib
    #2  0x0000000103b75fd1 in cuGraphicsGLRegisterImage () from /Library/Frameworks/CUDA.framework/Versions/A/Libraries/libcuda_310.40.25_mercury.dylib
    #3  0x0000000103b61c1e in cuGraphicsGLRegisterImage () from /Library/Frameworks/CUDA.framework/Versions/A/Libraries/libcuda_310.40.25_mercury.dylib
    #4  0x0000000103b61f0d in cuGraphicsGLRegisterImage () from /Library/Frameworks/CUDA.framework/Versions/A/Libraries/libcuda_310.40.25_mercury.dylib
    #5  0x0000000103b571e5 in cuGraphicsGLRegisterImage () from /Library/Frameworks/CUDA.framework/Versions/A/Libraries/libcuda_310.40.25_mercury.dylib
    #6  0x0000000103a7eb51 in cuGraphicsGLRegisterImage () from /Library/Frameworks/CUDA.framework/Versions/A/Libraries/libcuda_310.40.25_mercury.dylib
    #7  0x0000000103a8224f in cuGraphicsGLRegisterImage () from /Library/Frameworks/CUDA.framework/Versions/A/Libraries/libcuda_310.40.25_mercury.dylib
    #8  0x0000000103a7105d in cuMemcpyDtoH_v2 () from /Library/Frameworks/CUDA.framework/Versions/A/Libraries/libcuda_310.40.25_mercury.dylib
    #9  0x0000000101816ba4 in (anonymous namespace)::py_memcpy_dtoh(pycudaboost::python::api::object, unsigned long long) () from /usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/_driver.so
    #10 0x0000000101839e1d in pycudaboost::python::detail::caller_arity<2u>::impl<void (*)(pycudaboost::python::api::object, unsigned long long), pycudaboost::python::default_call_policies, pycudaboost::mpl::vector3<void, pycudaboost::python::api::object, unsigned long long> >::operator()(_object*, _object*) () from /usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/_driver.so
    #11 0x0000000101869d4e in pycudaboost::python::objects::function::call(_object*, _object*) const () from /usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/_driver.so
    #12 0x000000010186bf7a in pycudaboost::detail::function::void_function_ref_invoker0<pycudaboost::python::objects::(anonymous namespace)::bind_return, void>::invoke(pycudaboost::detail::function::function_buffer&) ()
       from /usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/_driver.so
    #13 0x00000001018799f3 in pycudaboost::python::detail::exception_handler::operator()(pycudaboost::function0<void> const&) const () from /usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/_driver.so
    #14 0x0000000101851f76 in pycudaboost::detail::function::function_obj_invoker2<pycudaboost::_bi::bind_t<bool, pycudaboost::python::detail::translate_exception<pycuda::error, void (*)(pycuda::error const&)>, pycudaboost::_bi::list3<pycudaboost::arg<1>, pycudaboost::arg<2>, pycudaboost::_bi::value<void (*)(pycuda::error const&)> > >, bool, pycudaboost::python::detail::exception_handler const&, pycudaboost::function0<void> const&>::invoke(pycudaboost::detail::function::function_buffer&, pycudaboost::python::detail::exception_handler const&, pycudaboost::function0<void> const&) () from /usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/_driver.so
    #15 0x0000000101879783 in pycudaboost::python::handle_exception_impl(pycudaboost::function0<void>) () from /usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/_driver.so
    #16 0x000000010186b963 in function_call () from /usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/_driver.so
    #17 0x0000000100011665 in PyObject_Call () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #18 0x00000001000a60b4 in PyEval_EvalFrameEx () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #19 0x00000001000a2076 in PyEval_EvalCodeEx () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #20 0x00000001000a8f36 in fast_function () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #21 0x00000001000a528b in PyEval_EvalFrameEx () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #22 0x00000001000a8ed2 in fast_function () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #23 0x00000001000a528b in PyEval_EvalFrameEx () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #24 0x00000001000a2076 in PyEval_EvalCodeEx () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #25 0x00000001000a8f36 in fast_function () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #26 0x00000001000a528b in PyEval_EvalFrameEx () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #27 0x00000001000a8ed2 in fast_function () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #28 0x00000001000a528b in PyEval_EvalFrameEx () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #29 0x00000001000a2076 in PyEval_EvalCodeEx () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #30 0x00000001000a19a6 in PyEval_EvalCode () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #31 0x00000001000c9611 in PyRun_FileExFlags () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #32 0x000000010009dfe6 in builtin_execfile () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #33 0x00000001000a4010 in PyEval_EvalFrameEx () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #34 0x00000001000a2076 in PyEval_EvalCodeEx () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #35 0x00000001000a6752 in PyEval_EvalFrameEx () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #36 0x00000001000a2076 in PyEval_EvalCodeEx () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #37 0x00000001000a8f36 in fast_function () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #38 0x00000001000a528b in PyEval_EvalFrameEx () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #39 0x00000001000a2076 in PyEval_EvalCodeEx () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #40 0x00000001000350c6 in function_call () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #41 0x0000000100011665 in PyObject_Call () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #42 0x00000001000dd131 in RunModule () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #43 0x00000001000dcc12 in Py_Main () from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python
    #44 0x00007fff904935fd in start () from /usr/lib/system/libdyld.dylib
    #45 0x00007fff904935fd in start () from /usr/lib/system/libdyld.dylib
    #46 0x0000000000000000 in ?? ()
    (cuda-gdb) 


::

    (cuda-gdb) c
    Continuing.
    ^Cwarning: (Internal error: pc 0x104199240 in read in psymtab, but not in symtab.)


    Program received signal SIGINT, Interrupt.
    [Switching focus to CUDA kernel 5, grid 6, block (2403,0,0), thread (35,0,0), device 0, sm 1, warp 2, lane 3]
    render(int, float3 * @generic, float3 * @generic, Geometry * @generic, unsigned int, unsigned int * @generic, float * @generic, unsigned int * @generic, float4 * @generic)<<<(4801,1,1),(64,1,1)>>> (nthreads=307200, _origin=0x707680000, _direction=0x707a20000, 
        __val_paramg=0x700181800, alpha_depth=3, pixels=0x7090c0000, _dx=0x707dc0000, dxlen=0x708f80000, _color=0x708160000) at kernel.cu:144
    144         if (n < 1) {
    (cuda-gdb) info contexts
    Undefined info command: "contexts".  Try "help info".
    (cuda-gdb) info cuda contexts
         Context Dev    State 
    * 0x10097d200   0   active 
    (cuda-gdb) info cuda blocks
        BlockIdx To BlockIdx Count   State 
    Kernel 5
    * (2403,0,0)  (2403,0,0)     1 running 
    (cuda-gdb) info cuda threads
        BlockIdx ThreadIdx To BlockIdx ThreadIdx Count         Virtual PC   Filename  Line 
    Kernel 5
    * (2403,0,0)  (32,0,0)  (2403,0,0)  (47,0,0)    16 0x0000000104199240  kernel.cu   144 
      (2403,0,0)  (48,0,0)  (2403,0,0)  (48,0,0)     1 0x0000000131e41a30 geometry.h    10 
      (2403,0,0)  (49,0,0)  (2403,0,0)  (63,0,0)    15 0x0000000104199240  kernel.cu   144 
    (cuda-gdb) info cuda kernels
      Kernel Parent Dev Grid Status   SMs Mask    GridDim BlockDim Invocation 
    *      5      -   0    6 Active 0x00000002 (4801,1,1) (64,1,1) render(nthreads=307200, _origin=0x707680000, _direction=0x707a20000, __val_paramg=0x700181800, alpha_depth=3, pixels=0x7090c0000, _dx=0x707dc0000, dxlen=0x708f80000, _color=0x708160000) 
    (cuda-gdb) c
    Continuing.
    INFO:chroma:saving screen to 3199_000.png 
    [New Thread 0x391b of process 6669]
    [Context Pop of context 0x10097d200 on Device 0]
    [Termination of CUDA Kernel 5 (render<<<(4801,1,1),(64,1,1)>>>) on Device 0]
    [Context Push of context 0x10097d200 on Device 0]
    [Context Pop of context 0x10097d200 on Device 0]
    [Context Push of context 0x10097d200 on Device 0]
    [Context Pop of context 0x10097d200 on Device 0]
    [Context Push of context 0x10097d200 on Device 0]
    [Context Pop of context 0x10097d200 on Device 0]

    Before eliding...

        simon:cuda blyth$ grep Context debug.rst | wc -l   
            1078

    [Context Push of context 0x10097d200 on Device 0]
    [Context Pop of context 0x10097d200 on Device 0]
    [Context Push of context 0x10097d200 on Device 0]
    [Context Pop of context 0x10097d200 on Device 0]
    [Context Push of context 0x10097d200 on Device 0]
    [Context Pop of context 0x10097d200 on Device 0]
    [Context Push of context 0x10097d200 on Device 0]
    [Context Pop of context 0x10097d200 on Device 0]
    [Context Push of context 0x10097d200 on Device 0]
    [Context Pop of context 0x10097d200 on Device 0]

    Program exited normally.
    [Termination of CUDA Kernel 4 (fill<<<(64,1,1),(128,1,1)>>>) on Device 0]
    [Termination of CUDA Kernel 3 (fill<<<(64,1,1),(128,1,1)>>>) on Device 0]
    [Termination of CUDA Kernel 2 (write_size<<<(1,1,1),(1,1,1)>>>) on Device 0]
    [Termination of CUDA Kernel 1 (write_size<<<(1,1,1),(1,1,1)>>>) on Device 0]
    [Termination of CUDA Kernel 0 (write_size<<<(1,1,1),(1,1,1)>>>) on Device 0]
    (cuda-gdb) 




PyCUDA Version
----------------

::

    (chroma_env)delta:chroma_camera blyth$ python -c "import pycuda ; print pycuda.VERSION "
    (2013, 1, 1)
    (chroma_env)delta:chroma_camera blyth$ python -c "import pycuda ; print pycuda.VERSION_STATUS "

    (chroma_env)delta:chroma_camera blyth$ python -c "import pycuda ; print pycuda.VERSION_TEXT "
    2013.1.1


pudb : Console based python debugger
-------------------------------------

Referenced from PyCUDA FAQ

* https://pypi.python.org/pypi/pudb




