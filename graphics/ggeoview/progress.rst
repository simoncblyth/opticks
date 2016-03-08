Progress
=========

Use *hg-month n* to review commits for the numbered month.

July
-----

* trying to get JUNO geometry to work

Aug
----

* OptiX instancing 
* intro BBox standins
* Thrust interop

Sept
-----

* use interop Thrust/CUDA/OptiX to make photons fully GPU resident, eliminating overheads
* add Torch for testing
* investigate bad material for upwards going photons, find cause is bad geometry,
  integrate OpenMesh to enable mesh fixing 

October
-------- 

* vertex deduping as standard  
* IAV and OAV mesh surgery
* sensor handling
* identity with instancing
* analytic geometry description of DYB PMT via detdesc parsing and geometrical partitioning
* flexible boundary creation

November
---------

* overhaul material/surface/boundary handling to allow dynamic boundary creation post geocache
* implement dynamic test geometry creation controlled by commandline argument, using "--test" option 
* npy analysis for Fresnel reflection testing
* adopt more rational PMT partitioning surfaces (not a direct translation)









