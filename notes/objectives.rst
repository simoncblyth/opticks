Objectives
===========

::

    the main research direction and ideas, 
    the key problems to be solved, the expected goals. 
    the expected goals are listed for each year.


Main research direction and ideas
-----------------------------------

Make the best use of Opticks GPU optical photon
simulation within the JUNO simulation. 

Automate validation and performance measurement to enable 
these to be monitored continuously as the JUNOSW and Opticks 
simulations evolve together in the face of comparison with real data.

Key problems to be solved
----------------------------

The huge numbers of optical photons in high energy events
within JUNO such as the critical cosmogenic muon background 
presents a major problem for JUNO simulation both in 
processing time and memory requirements. Opticks is able to 
solve these two problems by offloading the simulation of 
optical photons to the GPU. However Opticks is cutting edge 
software that needs optimization and associated development
in order to fulfil its promise. 
  
Expected Goals
---------------





Geometry : complete missing features 
---------------------------------------

* guide tube torus via triangulated geometry option (2024 Apr)
* deep CSG tree solids for support stick solids (2024 Apr)
* investigate Chimney geometry issue (2024 Apr) 

New development for optimization option comparison 
----------------------------------------------------

* triangulation for more than just Guide Tube torus ? 
* alt GAS (1 multi-Prim solid -> 1 GAS) 
* sibling-instancing for sticks : inst joining or rationalize source geom 

New development for deployment
--------------------------------

* event splitting/joining implementation
* multi-GPU support : how much effort ? how much benefit ? 

  * unlike OptiX < 7, does not come for free  

New CSG development for robustness, performance
-----------------------------------------------

* more volume-centric CSG, SDF based ?  
* CSG intersect alg, perhaps changing alg can avoid:

  * poor performance with deep trees 
  * not compatible with tree balancing

* eg: instead of bit twiddle navigating complete binary tree => try postorder next node offsets
* alt to ray tracing : SDF based "sphere-tracing" 

Optimization (workstation level)
-------------------------------------

* tests with variety of GPUs of different VRAM and with/without RT cores
* develop simulation benchmarks 

  * standard event samples of different energies and positions etc.. 
  * relate them to ray trace benchmarks

* vary geometry ray trace implemention while running benchmarks   

  * perf impl of many choices esp: geometry modelling,  eg instancing criteria


Production Optimization/Planning (cluster level) 
--------------------------------------------------

* maximize throughput when submitted multiple production jobs   

* use experience to form realistic resource estimates as function
  of production sample size


Validation + Testing
---------------------

* further automation to provide continuous validation 
* continuous integration system (late 2024)  

Evolution
----------

As real data arrives the models and geometry of the simulation 
will evolve to match it, for example accomodating shifts/deformations.
Opticks will need to evolve in tandem with JUNOSW. 

In particular increasing the complexity of the geometry may have 
performance implications. However certain techniques such as 
limited use of triangulated geometry for some solids may mitigate 
and even speed up the GPU simulation.  


Engage with Opticks users from other experiments
-------------------------------------------------

Expanding the community of Opticks users 
and developers is essential for its long term viability. 
Targetted assistance to high profile Opticks users 
is important to demonstrate Opticks beyond JUNO. 


