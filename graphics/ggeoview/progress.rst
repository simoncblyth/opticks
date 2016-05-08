Progress
=========

Use *hg-month n* to review commits for the numbered month, 
negative n eg -12 for December of last year.

May
----

* start bringing materials to GPU via textures
* introduce the geocache
* material code translation in Lookup
* adopt ImGui

June
-----

* develop highly compressed photon records
* ViewNPY machinery for OpenGL uploading 
* get animation working 
* add GOpticalSurface, for transporting surface props thru Assimp/AssimpWrap into GGeo
* learning Thrust
* OptiX 3.8 , CUDA 7.0 update 

July
-----

* photon indexing with Thrust
* verifying ThrustIndex by comparison against the much slower SequenceNPY
* auto-finding repeated geometry assemblies by progeny transform/mesh-index digests in GTreeCheck
* interim Linux compatibility working with Tao
* 4-GPU machine testing with Tao
* OpenGL instancing 
* trying to get JUNO geometry to work
* computeTest timings for Juno Scintillation as vary CUDA core counts

Aug
----

* OptiX instancing 
* intro BBox standins
* Thrust interop

Sept
-----

* use interop Thrust/CUDA/OptiX to make photons fully GPU resident, eliminating overheads
* add Torch for testing
* investigate bad material for upwards going photons, find cause is bad geometry
* integrate OpenMesh to enable mesh fixing

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

December
---------

* prism test with Plankian light source using GPU texture
* rainbow comparisons against expectation
* cfg4 : new package for comparison against standalone geant4
* cfg4 G4StepPoint recording : creating opticks format photon/step/history records with cfg4-
* Opticks/Geant4 rainbow scatter matching achieved
* enable loading of photons/records into ggv, in pricipal enables visualizing both Opticks and G4 cfg4- generated/propagated events on non-CUDA machines
* begin revival of compute mode

January
---------

* rework Bookmarks, split off state handling into NState
* add InterpolatedView for viewpoint animation 
* JUNO meeting presentation 
* PSROC meeting presentation 

February
---------

* create analytic geometry description of Dayabay PMT 
* PMTInBox debugging
* compositing OptiX raytrace with OpenGL rasterized

March
-------

* resolved PMT skimmer BR BR vs BR BT issue : turned out to be Opticks TIR bug
* PmtInBox step-by-step record distribution chi2 comparison 
* rejig material/surface/boundary buffer layout to match OptiX tex2d float4 textures, with wavelength samples and float4 at the tip of the array serialization
* Dayabay presentation
* screen capture movie making 
* GTC presentation

April
------

* GTC
* factoring usage of OptiX to provide functionality on non-CUDA/OptiX capable nodes
* CMake superbuild with CTests 
* external get/build/install scripts
* prep for spawning Opticks repository 

May
----

* shifts
* getting more CTests to pass 


