Opticks Performance
======================

Levels of performance
-----------------------

Performance needs to be considered at three levels:

1. **optical**: optical photon simulation, ie the time and memory required to generate and propagate optical photons
2. **node**: overall simulation performance using a single node with one or more GPUs 
3. **cluster**: production performance of a GPU cluster creating large monte carlo samples
   with many hundreds or thousands of production jobs orchestrated with a batch system

For each of these three levels the below sections seek to answer the 
below questions.

1. what are the principal determinants of performance ?
2. how to measure performance ?
3. how to optimize ?


Opticks measurements/optimizations so far have all been at the **optical** level. 
Because of this the next steps in optimization, should be at other levels. 
As the **node** speedup depends on the non-optical part of the simulation 
it is somewhat out-of-scope for Opticks, thus I think the next steps for Opticks optimization 
should be at **cluster** level by reviving the Opticks server functionality.

**optical** performance
-------------------------

Principal determinant is the BVH (bounding volume hierarchy) acceleration 
structure which is automatically constructed by NVIDIA OptiX based on the input 
geometry.  As the actual form of the BVH and its construction are proprietry to NVIDIA 
it is necessary to treat this as a "black box" and adopt an experimental approach 
to tuning.

1. establish several benchmark geometries including both analytic and triangulated forms 
   and a range of optical simulations that are similar to the intended target simulation 
   but more controllable in the sense of being able to scan the numbers of photons 

2. make measurements scanning the numbers of photons up to the VRAM 
   limits of the GPUs being used

3. vary geometry modelling (ie how the NVIDIA OptiX geometry is constructed) 
   and even change geometry in an attempt to develop some intuition, proceed to 
   iterate steps 2 and 3.

An important secondary determinant is the VRAM available, as for large 
simulations insufficient VRAM will force splittling of GPU launches.


Task **A5** and **A6** from the tasks list :doc:`../tasks/tasks.rst` 
covers the transition to OptiX 7+ and associated geometry model tuning.


**node** performance
-------------------------

Amdahls "Law" explains that performance of a partially parallized process 
will assuming a reasonable degree of parallelism be limited by the serial 
portion because the parallel part will go to zero. For example if 99% of the 
overall simulation time is taken up by the optical photon simulation then 
the overal simulation performance can only achieve a factor of 100x speedup. 
Note that time is not the only criteria, CPU memory also limits simulations.  

Assuming **optical** performance is sufficient the principal determinant of **node** performance 
will be the non-optical part of the simulation which is out of scope of Opticks. 
This assumption needs to be confirmed for each geometry. Measurements with JUNO analytic geometry 
suggest it will be the case, which is why task **B3** "Multi-GPU scaling in OptiX 7+" from the tasks list :doc:`../tasks/tasks.rst` 
is not a top priority **A** task. 

 
**cluster** performance
----------------------------

The aim at this level is to maximise the throughput of a cluster running 
simulation production runs that use resources of hundreds or thousands of nodes 
rather than the speed of a single node.  

OptiX 6 **optical** performance has been measured to almost perfectly linearly scale 
out to 4 connected GPUs. Although single node **optical** performance is best like this 
it would almost certainly be a waste of GPU resources with GPUs starved of work.

OptiX 7 removed the transparent multi-GPU linear performance scaling 
provided by OptiX 6, so once the transition to OptiX 7 is underway there is the 
open question of whether it is worthwhile to support multiple 
GPUs and if so how to implement it.  

While multi-GPU support clearly improves **optical** performance 
I remain doubtful that it will improve **cluster** performance as 
it will mean more GPUs will be in a state of starvation 
that will limit the number of concurrent batch jobs that can run. 

Open questions:

1. can multiple jobs sharing the same node and GPUs work ?
2. how many GPUs should be used for each CPU node ? 
3. can GPU cluster architects to pursuaded to spread GPUs across more CPU nodes ?
   some clusters have 8 GPUs attached to each node : this seems like a waste of resources 
4. too little VRAM will force split launches  
5. too much VRAM wastes resources unless VRAM usage be effectively limited and hence shared by multiple jobs ?

One possible solution to the GPU starvation problem is to revive Opticks server functionality. 
This is implemented using ZeroMQ + Boost-Asio and uses the network transport of NumPy 
arrays of gensteps and hits. Possibly using a networked Opticks would allow 
simulation batch jobs on larger clusters of pure CPU clusters to more effectively 
share the limited resources of smaller GPU clusters running Opticks servers.  

Because this enables more resources to be brought into play to create 
large monte carlo samples it will clearly be beneficial.

The practicality/reliability of such networked approaches with existing batch systems 
such as SLURM is uncertain. Actually an Opticks server doesnt fit into the 
typical batch job approach, it needs to be more like a high performance web site 
or database with a load balancer frontend that distributes requests to a pool
of workers.

Tasks **A3** and **A4** from the tasks list :doc:`../tasks/tasks.rst` reflect the 
high priority of **cluster** optimization efforts. 


profiling and run metadata machinery
--------------------------------------

Opticks executables profile memory and time at various points and write these 
into OpticksProfile.npy files together with saved event files. 
To understand profiling start with optickscore/OpticksProfile.cc.
Also so called Accumulators are used to measure launch times.
Also metadata about running conditions are written to json and ini files. 
There is machinery
for scanning of command parameters such as numbers of photons and 
varying options such as number of visible GPUs or RTX options.  


bin/scan.bash      
    top level control of command scanning/profiling 

ana/profile.py 
    Profile instances are instanciated by reading OpticksProfile.npy files 

ana/profilesmry.py 
    ProfileSmry is an ordered dict of Profile keyed by run categories such as cvd_0_rtx_1_100M
    which indicate the number of GPUs in use, RTX mode and the number of photons

ana/profilesmrytab.py
    Loads and displays ProfileSmry 
 
ana/profilesmryplot.py 
    performance comparison plots 



