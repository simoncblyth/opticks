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
  

Geometry : complete missing features 
---------------------------------------

* guide tube torus via triangulated geometry option (2024 Apr)
* deep CSG tree solids for support stick solids (2024 Apr)
* investigate Chimney geometry issue (2024 Apr) 

Geometry Issues
------------------

* Updated WaterPool geometry uses complex CSG breaking Opticks geometry translation


New development for optimization option comparison 
----------------------------------------------------

* triangulation for more than just JUNO Guide Tube torus ? 
* sibling-instancing for JUNO stick geometry

  * cleverer sibling-instancing or make Geant4 stick geometry more heirachical 


New development for deployment
--------------------------------

* event splitting/joining implementation
* multi-GPU support : how much effort ? how much benefit ? 

  * unlike OptiX < 7, does not come for free  


Visualization : revival of old opticks features into new opticks
------------------------------------------------------------------------

* compositing of raytrace geometry and OpenGL event representations 
* animated OpenGL photon propagation visualization 
* animated interpolation between bookmarked viewpoints 
* ImGUI OpenGL interface, GPU side menu system eg to control view params, photon selection, viewpoint "bookmarks"
* improved interactive navigation around geometry (eg WASDQE+mouse) 
* networked control over UDP, for commandline control of OpenGL application

Server/Client Opticks  
-----------------------

* requires use of boost asio (asynchronous IO) or similar + message queue (eg ZeroMQ) 
* server picks up gensteps and responds with hits
* needs consistent geometry check, eg with geometry digest in genstep metadata


New CSG development : test alternative CSG intersect alg
----------------------------------------------------------

Perhaps changing CSG alg can avoid:

* poor performance with deep trees (complex solids)
* non-compatibility with tree balancing

Perhaps changing CSG alg can enable:

* n-ary trees, not just binary 

Poor performance with deep trees is 
due to the reliance on complete binary tree serialization
which enables simple postorder tree navigation by 
bit twiddling. 

Test alternative ways to represent CSG trees, eg:

1. index offsets, num_child  



New CSG development : novel ray marching(sphere tracing) SDF based intersection
--------------------------------------------------------------------------------

* investigate ray-marching techniques such as sphere tracing, which are an iterative alternative to ray tracing
  expected to provide high performnce for highly complex solids as shape complexity
  can be distilled into generated CUDA code of mathematical signed distance functions (SDF)
  rather than traversing a tree of constituent nodes while ray tracing.




New CSG development : change alg for robustness
---------------------------------------------------

* more volume-centric CSG, SDF based ?  
* alt to ray tracing : SDF based "sphere-tracing" 


New CSG development : general spherical shell, theta phi segment
------------------------------------------------------------------

G4Sphere impl of ray DistanceToIn and DistanceToOut are 900 lines each
covering the many cases of intersecting rays with a "funny" shape. 

* Is there a simpler way ? 

* See the below headers for an old attempt to 
  decompose into infinite phi + theta cuts, with idea that CSG algorithm 
  can do the heavy lifting of handling all the cases. This worked to 
  some extent, needs careful testing, maybe some cases not working:: 

  CSG/csg_intersect_leaf_phicut.h
  CSG/csg_intersect_leaf_thetacut.h
  CSG/tests/intersect_leaf_thetacut_test.cc


* https://core.ac.uk/download/pdf/158318251.pdf
* ~/opticks_refs/spherical_ray_tracing_david_a_hannasch_158318251.pdf
* 76 page MSc Thesis looks relevant, might provide insights



New development : find/develop CUDA torus intersect alg
---------------------------------------------------------

Adapt algs from any sources or develop your own algs, 
numerical or analytic. Get them to work with CUDA. 

* compare accuracy/performance/resource-use with triangulated approach 
* tweak triangulated approach (changing number and disposition of the triangles)

Possible sources:

* Geant4 U-solids, vecgeom, elsewhere, ... 
* open source ray trace frameworks
* quartic polynomial solvers analytic OR numerical (eg from computer science OR numerical math papers)
* search github/bitbucket repos, web searches  

+-------------------------------------------------------------------------------------+
| Double heavy analytic torus intersect alg always problematic                        |
+=================+===================================================================+
| OptiX 5.0, 5.5  | worked but was performance problem, also imprecise in CSG         |
+-----------------+-------------------------------------------------------------------+
| OptiX 6.0.0     | caused crash : so excluded torus                                  |
+-----------------+-------------------------------------------------------------------+
| OptiX 7, 8:     | TODO: check again, revisit alg (may depend on GPU resources)      |  
+-----------------+-------------------------------------------------------------------+



Compare perf between PTX and OptiX-IR (different ways to compile OptiX kernels)
--------------------------------------------------------------------------------

* Look for low hanging fruit using NVIDIA kernel profiling tools 
* experience with Nsight likely needed for multi-GPU impl 


Optimization (workstation level)
-------------------------------------

* tests with variety of GPUs of different VRAM and with/without RT cores

* evaluate SER : Shader Execution Reordering (available with Ada and OptiX 8.0)

  * ~/opticks_refs/nvidia-ser-whitepaper.pdf 
  * https://d29g4g2dyqv443.cloudfront.net/sites/default/files/akamai/gameworks%2Fser-whitepaper.pdf

* develop simulation benchmarks 

  * standard event samples of different energies and positions etc.. 
  * relate them to ray trace benchmarks

* vary geometry ray trace implemention while running benchmarks   

  * perf impl of many choices esp: geometry modelling,  eg instancing criteria


Multi-GPU is it worthwhile ?
------------------------------

* process/job level could be trivial : scripts exploiting slurm capabilities
* https://medium.com/gpgpu/multi-gpu-programming-6768eeb42e2c


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


